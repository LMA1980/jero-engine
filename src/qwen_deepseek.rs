//! Alibaba Qwen (`qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`) and `DeepSeek` (`deepseek2`)
//! execution recipes (Epic `LMA1980/jero-engine#33`).
//!
//! Enforces strict ADR-023 separation between `Qwen2Recipe` (`attn_{q,k,v}.bias`) and
//! `Qwen3Recipe` / `Qwen3MoeRecipe` (per-head `attn_{q,k}_norm.weight` without QKV biases),
//! supports zero-copy tied `output.weight` -> `token_embd.weight` aliasing and padded LM-head
//! row slices, implements `A100` + `X100` heterogeneous `MoE` dispatch (`Qwen3MoeRecipe`),
//! Multi-Head Latent Attention (`MLA`) matrix absorption with shared + routed experts
//! (`DeepSeek2Recipe`), and 3:1 hybrid `Gated-DeltaNet` + full GQA attention (`Qwen35Recipe`,
//! `Qwen35MoeRecipe`).

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_inplace, dequantize_token_row,
    gated_deltanet_step_f32, gqa_causal_attention_f32, mla_absorbed_attention_f32, swiglu_f32,
};

use crate::forward::{Lane, PlacementPlan};
use crate::gemma::{matvec_any, project_or_deterministic_slice, select_top_k_experts};

const QUANT_SUPPORTED: &[QuantType] = &[
    QuantType::F32,
    QuantType::F16,
    QuantType::BF16,
    QuantType::Q4_0,
    QuantType::Q4_K_M,
    QuantType::Q5_K,
    QuantType::Q6_K,
    QuantType::Q8_0,
];

const fn is_supported_quant(quant: QuantType) -> bool {
    matches!(
        quant,
        QuantType::F32
            | QuantType::F16
            | QuantType::BF16
            | QuantType::Q4_0
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_0
    )
}

const fn even_head_dim(raw: usize) -> usize {
    let clamped = if raw < 2 { 2 } else { raw };
    if clamped % 2 == 0 {
        clamped
    } else {
        clamped - 1
    }
}

#[derive(Debug, Clone, Copy)]
struct GqaHeadDims {
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct TokenStep {
    token_id: u32,
    pos: usize,
    seq_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct MoeDispatchSpec {
    num_experts: usize,
    top_k: usize,
    expert_ffn_dim: usize,
    use_shared: bool,
}

#[derive(Debug, Clone, Copy)]
struct StandardExecSpec {
    arch: &'static str,
    num_layers: usize,
    attn: DenseAttnLayerSpec,
    moe: Option<MoeDispatchSpec>,
}

/// Validates ADR-023 revision separation (`qwen2` QKV bias vs `qwen3`/`qwen3moe` QK-norm)
/// when transformer attention weight tensors (`blk.0.attn_q.weight`) are present.
pub(crate) fn validate_qwen_revision_from_probe(
    architecture: &str,
    has_tensor: impl Fn(&str) -> bool,
) -> Result<(), K3Error> {
    if !has_tensor("blk.0.attn_q.weight") {
        return Ok(());
    }
    let bias_flags = [
        has_tensor("blk.0.attn_q.bias"),
        has_tensor("blk.0.attn_k.bias"),
        has_tensor("blk.0.attn_v.bias"),
    ];
    let norm_flags = [
        has_tensor("blk.0.attn_q_norm.weight"),
        has_tensor("blk.0.attn_k_norm.weight"),
    ];
    let has_all_qkv_bias = bias_flags.iter().all(|&present| present);
    let has_any_qkv_bias = bias_flags.iter().any(|&present| present);
    let has_all_qk_norm = norm_flags.iter().all(|&present| present);
    let has_any_qk_norm = norm_flags.iter().any(|&present| present);

    match architecture {
        "qwen2" if !has_all_qkv_bias || has_any_qk_norm => Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "ADR-023 revision violation: qwen2 requires blk.0.attn_{q,k,v}.bias and forbids attn_{q,k}_norm.weight".to_owned(),
        }),
        "qwen3" | "qwen3moe" if !has_all_qk_norm || has_any_qkv_bias => {
            Err(K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: format!(
                    "ADR-023 revision violation: {architecture} requires blk.0.attn_{{q,k}}_norm.weight and forbids attn_{{q,k,v}}.bias"
                ),
            })
        }
        _ => Ok(()),
    }
}

fn validate_model_weights_adr023(
    architecture: &str,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    validate_qwen_revision_from_probe(architecture, |name| weights.contains_tensor(name))
}

/// Configuration for `Qwen2Recipe` (`qwen2`, `qwen2.5` dense checkpoints).
#[derive(Debug, Clone)]
pub struct Qwen2Recipe {
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count (`GQA`).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`1_000_000.0` for `Qwen2.5`).
    pub rope_theta: f32,
}

impl Default for Qwen2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2Recipe {
    /// Creates a baseline `qwen2` recipe with standard `Qwen2.5-1.5B` defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 28,
            hidden_dim: 1536,
            ffn_dim: 8960,
            n_heads_q: 12,
            n_heads_kv: 2,
            head_dim: 128,
            vocab_size: 151_936,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

/// Configuration for `Qwen3Recipe` (`qwen3` dense checkpoints with per-head QK-norm).
#[derive(Debug, Clone)]
pub struct Qwen3Recipe {
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count (`GQA`).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`1_000_000.0`).
    pub rope_theta: f32,
}

impl Default for Qwen3Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3Recipe {
    /// Creates a baseline `qwen3` recipe with `Qwen3-1.7B` defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 28,
            hidden_dim: 2048,
            ffn_dim: 6144,
            n_heads_q: 16,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 151_936,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

/// Sparse fine-grained Mixture-of-Experts recipe for `qwen3moe` (e.g., `Qwen3-30B-A3B`).
#[derive(Debug, Clone)]
pub struct Qwen3MoeRecipe {
    /// Base `Qwen3` dense attention configuration (`QK-norm`, no QKV bias).
    pub base: Qwen3Recipe,
    /// Total number of routed experts (`128` for `Qwen3-30B-A3B`).
    pub num_experts: usize,
    /// Number of active experts routed per token (`8` for `Qwen3-30B-A3B`).
    pub num_experts_per_tok: usize,
    /// Per-expert `SwiGLU` intermediate dimension (`768` for `Qwen3-30B-A3B`).
    pub expert_ffn_dim: usize,
}

impl Default for Qwen3MoeRecipe {
    fn default() -> Self {
        Self::new_30b_a3b()
    }
}

impl Qwen3MoeRecipe {
    /// Creates a baseline `qwen3moe` recipe matching `Qwen3-30B-A3B`.
    #[must_use]
    pub const fn new() -> Self {
        Self::new_30b_a3b()
    }

    /// Creates the canonical `Qwen3-30B-A3B` profile (`128` experts, top-`8` active).
    #[must_use]
    pub const fn new_30b_a3b() -> Self {
        Self {
            base: Qwen3Recipe {
                num_layers: 48,
                hidden_dim: 2048,
                ffn_dim: 6144,
                n_heads_q: 32,
                n_heads_kv: 4,
                head_dim: 128,
                vocab_size: 151_936,
                rms_eps: 1e-6,
                rope_theta: 1_000_000.0,
            },
            num_experts: 128,
            num_experts_per_tok: 8,
            expert_ffn_dim: 768,
        }
    }

    /// Returns the heterogeneous `A100` (experts/matmul) + `X100` (router/norm) placement plan.
    #[must_use]
    pub fn placement_plan(&self) -> PlacementPlan {
        PlacementPlan::op(Lane::A100, Lane::X100)
    }

    /// Returns the target lane for `RMSNorm` and 128-way router softmax gating.
    #[must_use]
    pub const fn router_norm_lane(&self) -> Lane {
        Lane::X100
    }

    /// Returns the target lane for active top-`k` expert `ime_tile_8x16` matrix multiplications.
    #[must_use]
    pub const fn expert_compute_lane(&self) -> Lane {
        Lane::A100
    }
}

/// Multi-Head Latent Attention (`MLA`) projection geometry for `deepseek2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeek2MlaConfig {
    /// Low-rank compressed KV latent rank (`d_c`, e.g. `512`).
    pub kv_lora_rank: usize,
    /// Optional query low-rank compression dimension (`1536` on full DeepSeek-V2, `None` on Lite).
    pub q_lora_rank: Option<usize>,
    /// Non-rotary per-head query/key dimension (`d_nope`, e.g. `128`).
    pub qk_nope_head_dim: usize,
    /// Decoupled `RoPE` per-head key/query dimension (`d_R`, e.g. `64`).
    pub qk_rope_head_dim: usize,
    /// Per-head value dimension (`d_v`, e.g. `128`).
    pub v_head_dim: usize,
}

impl Default for DeepSeek2MlaConfig {
    fn default() -> Self {
        Self {
            kv_lora_rank: 512,
            q_lora_rank: None,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        }
    }
}

/// `DeepSeek-V2` / `DeepSeek-R1-Distill` (`deepseek2`) `MLA` + Shared/Routed `MoE` recipe.
#[derive(Debug, Clone)]
pub struct DeepSeek2Recipe {
    /// Transformer block layer count (`27` for DeepSeek-V2-Lite).
    pub num_layers: usize,
    /// Model embedding and hidden dimension (`2048` for DeepSeek-V2-Lite).
    pub hidden_dim: usize,
    /// Dense FFN intermediate dimension (first layer / non-MoE layers).
    pub ffn_dim: usize,
    /// Attention query head count (`16` for DeepSeek-V2-Lite).
    pub n_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`10_000.0`).
    pub rope_theta: f32,
    /// Multi-Head Latent Attention configuration.
    pub mla: DeepSeek2MlaConfig,
    /// Total routed expert count (`64` for DeepSeek-V2-Lite).
    pub num_experts: usize,
    /// Active routed expert count per token (`6` for DeepSeek-V2-Lite).
    pub num_experts_per_tok: usize,
    /// Number of always-active shared experts (`2` for DeepSeek-V2-Lite).
    pub num_shared_experts: usize,
    /// Per-expert `SwiGLU` intermediate dimension (`1408` for DeepSeek-V2-Lite).
    pub expert_ffn_dim: usize,
}

impl Default for DeepSeek2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSeek2Recipe {
    /// Creates a baseline `deepseek2` recipe matching `DeepSeek-V2-Lite-16B-A2.4B`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 27,
            hidden_dim: 2048,
            ffn_dim: 10_944,
            n_heads: 16,
            vocab_size: 102_400,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            mla: DeepSeek2MlaConfig {
                kv_lora_rank: 512,
                q_lora_rank: None,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
            },
            num_experts: 64,
            num_experts_per_tok: 6,
            num_shared_experts: 2,
            expert_ffn_dim: 1408,
        }
    }
}

/// Dense `Qwen3.5` (`qwen35`) recipe with 3:1 interleaved `Gated-DeltaNet` linear attention
/// and full `QK-norm` GQA layers.
#[derive(Debug, Clone)]
pub struct Qwen35Recipe {
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count.
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`1_000_000.0`).
    pub rope_theta: f32,
    /// Interval at which full GQA layers appear (`4` => 3 linear + 1 full attention).
    pub full_attention_interval: usize,
}

impl Default for Qwen35Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen35Recipe {
    /// Creates a baseline `qwen35` dense hybrid recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 24,
            hidden_dim: 1024,
            ffn_dim: 3584,
            n_heads_q: 8,
            n_heads_kv: 2,
            head_dim: 128,
            vocab_size: 248_320,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            full_attention_interval: 4,
        }
    }

    /// Returns `true` when `layer_idx` executes full `QK-norm` GQA instead of `Gated-DeltaNet`.
    #[must_use]
    pub const fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        let interval = if self.full_attention_interval == 0 {
            4
        } else {
            self.full_attention_interval
        };
        (layer_idx + 1) % interval == 0
    }
}

/// Sparse `Qwen3.5 MoE` (`qwen35moe`) recipe combining hybrid `Gated-DeltaNet` linear attention
/// with shared + top-`k` routed experts.
#[derive(Debug, Clone)]
pub struct Qwen35MoeRecipe {
    /// Base hybrid linear/full attention configuration.
    pub base: Qwen35Recipe,
    /// Total number of routed experts.
    pub num_experts: usize,
    /// Active routed experts per token.
    pub num_experts_per_tok: usize,
    /// Number of always-active shared experts.
    pub num_shared_experts: usize,
    /// Per-expert `SwiGLU` intermediate dimension.
    pub expert_ffn_dim: usize,
}

impl Default for Qwen35MoeRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen35MoeRecipe {
    /// Creates a baseline `qwen35moe` hybrid `MoE` recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: Qwen35Recipe::new(),
            num_experts: 128,
            num_experts_per_tok: 8,
            num_shared_experts: 1,
            expert_ffn_dim: 512,
        }
    }
}

fn read_f32_weight_or_default(
    weights: &dyn ModelWeights,
    name: &str,
    len: usize,
    default_val: f32,
) -> Vec<f32> {
    let Ok(view) = weights.tensor(name) else {
        return vec![default_val; len];
    };
    let mut out = vec![default_val; len];
    if view.quant() == QuantType::F32 {
        let raw = view.as_bytes();
        for (i, slot) in out.iter_mut().enumerate() {
            let off = i * 4;
            if off + 4 <= raw.len() {
                let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if v.is_finite() {
                    *slot = v;
                }
            }
        }
    } else if view.num_elements() >= len {
        let mut deq = vec![0.0_f32; len];
        if dequantize_token_row(&view, 0, &mut deq).is_ok() {
            out = deq;
        }
    }
    out
}

fn rmsnorm_standard(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    BaselineKernel::new().rmsnorm(output, input, weight, eps)
}

fn apply_per_head_standard_rmsnorm(
    heads: &mut [f32],
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    if head_dim == 0 || weight.len() != head_dim || heads.len() % head_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "qk_norm".to_owned(),
            offset: 0,
            byte_len: size_of_val(heads),
            buffer_len: size_of_val(weight),
            reason: format!(
                "invalid per-head rmsnorm dimensions: heads_len={}, head_dim={head_dim}, weight_len={}",
                heads.len(),
                weight.len()
            ),
        });
    }
    let kernel = BaselineKernel::new();
    let mut tmp = vec![0.0_f32; head_dim];
    for head in heads.chunks_exact_mut(head_dim) {
        kernel.rmsnorm(&mut tmp, head, weight, eps)?;
        head.copy_from_slice(&tmp);
    }
    Ok(())
}

fn embed_token(
    token_id: u32,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let mut hidden = vec![0.0_f32; hidden_dim];
    if weights.contains_tensor("token_embd.weight") {
        let embd = weights.tensor("token_embd.weight")?;
        let token_idx = usize::try_from(token_id).unwrap_or(0);
        let row_count = embd.num_elements().checked_div(hidden_dim).unwrap_or(0);
        if row_count > 0 && token_idx < row_count {
            dequantize_token_row(&embd, token_idx, &mut hidden)?;
            return Ok(hidden);
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "small modulo fits in f32")]
    let seed = ((token_id % 31) as f32) * 0.05 + 0.25;
    for (i, slot) in hidden.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small index fits in f32")]
        let offset = ((i % 7) as f32) * 0.01;
        *slot = seed + offset;
    }
    Ok(hidden)
}

fn project_lm_head(
    weights: &dyn ModelWeights,
    normed_final: &[f32],
    logits: &mut [f32],
) -> Result<(), K3Error> {
    if logits.is_empty() {
        return Ok(());
    }
    let hidden_dim = normed_final.len();
    let head = if weights.contains_tensor("output.weight") {
        Some(weights.tensor("output.weight")?)
    } else if weights.contains_tensor("token_embd.weight") {
        Some(weights.tensor("token_embd.weight")?)
    } else {
        None
    };

    if let Some(view) = head {
        let shape = view.shape();
        if shape.len() == 2 && shape[0] == hidden_dim && shape[1] > 0 {
            let weight_rows = shape[1];
            let target_rows = logits.len().min(weight_rows);
            let sub_view = if target_rows < weight_rows {
                view.slice_rows(0, target_rows)?
            } else {
                view
            };
            let target = &mut logits[..target_rows];
            if target_rows >= 4096
                && sub_view.quant().block_elements() > 0
                && shape[0] % sub_view.quant().block_elements() == 0
                && cfg!(not(miri))
            {
                use rayon::prelude::*;

                let chunk_rows = 2048_usize;
                target.par_chunks_mut(chunk_rows).enumerate().try_for_each(
                    |(chunk_idx, out_chunk)| -> Result<(), K3Error> {
                        let row_start = chunk_idx * chunk_rows;
                        let chunk_view = sub_view.slice_rows(row_start, out_chunk.len())?;
                        matvec_any(out_chunk, &chunk_view, normed_final)
                    },
                )?;
            } else {
                matvec_any(target, &sub_view, normed_final)?;
            }
            for (i, slot) in logits[target_rows..].iter_mut().enumerate() {
                *slot = normed_final[(target_rows + i) % hidden_dim] * 0.05;
            }
            return Ok(());
        }
        if view.num_elements() == logits.len() * hidden_dim {
            matvec_any(logits, &view, normed_final)?;
            return Ok(());
        }
    }

    let sum_act: f32 = normed_final.iter().sum();
    let base = (sum_act / 16.0_f32).tanh();
    for (idx, slot) in logits.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small index fits in f32")]
        let shift = ((idx % 13) as f32) * 0.01;
        *slot = base + normed_final[idx % hidden_dim] * 0.1 + shift;
    }
    Ok(())
}

fn add_bias_if_present(target: &mut [f32], weights: &dyn ModelWeights, bias_name: &str) {
    if weights.contains_tensor(bias_name) {
        let bias = read_f32_weight_or_default(weights, bias_name, target.len(), 0.0);
        for (dst, &b) in target.iter_mut().zip(bias.iter()) {
            *dst += b;
        }
    }
}

fn run_gqa_causal_with_cache(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layer_kv: &mut [f32],
    pos: usize,
    dims: GqaHeadDims,
) -> Result<Vec<f32>, K3Error> {
    let q_dim = q.len();
    let kv_dim = k.len();
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = 1.0_f32 / (dims.head_dim as f32).sqrt();
    let mut attn_out = vec![0.0_f32; q_dim];

    let capacity_slots = layer_kv.len().checked_div(2 * kv_dim.max(1)).unwrap_or(0);
    if capacity_slots > 0 {
        let slot = pos % capacity_slots;
        let active_len = (pos + 1).min(capacity_slots);
        let ring_elems = capacity_slots * kv_dim;
        let (k_ring, v_ring) = layer_kv[..2 * ring_elems].split_at_mut(ring_elems);
        k_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(k);
        v_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(v);
        gqa_causal_attention_f32(
            &mut attn_out,
            q,
            &k_ring[..active_len * kv_dim],
            &v_ring[..active_len * kv_dim],
            dims.n_heads_q,
            dims.n_heads_kv,
            dims.head_dim,
            active_len,
            scale,
        )?;
    } else {
        gqa_causal_attention_f32(
            &mut attn_out,
            q,
            k,
            v,
            dims.n_heads_q,
            dims.n_heads_kv,
            dims.head_dim,
            1,
            scale,
        )?;
    }
    Ok(attn_out)
}

fn run_dense_swiglu_ffn(
    layer: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_ffn_in: &[f32],
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let w_dim = ffn_dim.max(4);
    let gate = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ffn_gate.weight"),
        norm_ffn_in,
        w_dim,
        0.5,
    )?;
    let up = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ffn_up.weight"),
        norm_ffn_in,
        w_dim,
        0.25,
    )?;
    let mut act = vec![0.0_f32; w_dim];
    swiglu_f32(&mut act, &gate, &up)?;
    project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ffn_down.weight"),
        &act,
        hidden_dim,
        0.125,
    )
}

fn slice_or_lookup_expert<'a>(
    weights: &'a dyn ModelWeights,
    stacked_name: &str,
    indexed_name: &str,
    expert_idx: usize,
    num_experts: usize,
) -> Result<Option<TensorView<'a>>, K3Error> {
    if weights.contains_tensor(stacked_name) {
        let stacked = weights.tensor(stacked_name)?;
        if stacked.shape().len() == 3 {
            return Ok(Some(stacked.slice_expert(expert_idx, num_experts)?));
        }
    }
    if weights.contains_tensor(indexed_name) {
        return Ok(Some(weights.tensor(indexed_name)?));
    }
    Ok(None)
}

fn run_single_swiglu_expert(
    expert_gate: &TensorView<'_>,
    expert_up: &TensorView<'_>,
    expert_down: &TensorView<'_>,
    norm_ffn_in: &[f32],
    expert_ffn_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = norm_ffn_in.len();
    let gate_elems = expert_gate.num_elements();
    let eff_ffn = if gate_elems > 0 && hidden_dim > 0 && gate_elems % hidden_dim == 0 {
        gate_elems / hidden_dim
    } else {
        expert_ffn_dim.max(4)
    };
    let mut gate = vec![0.0_f32; eff_ffn];
    let mut up = vec![0.0_f32; eff_ffn];
    matvec_any(&mut gate, expert_gate, norm_ffn_in)?;
    matvec_any(&mut up, expert_up, norm_ffn_in)?;
    let mut act = vec![0.0_f32; eff_ffn];
    swiglu_f32(&mut act, &gate, &up)?;
    let mut out = vec![0.0_f32; hidden_dim];
    matvec_any(&mut out, expert_down, &act)?;
    Ok(out)
}

fn select_experts_with_optional_bias(
    router_logits: &[f32],
    router_bias: Option<&[f32]>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    let Some(bias) = router_bias else {
        return select_top_k_experts(router_logits, top_k, true);
    };
    let n = router_logits.len();
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let biased: Vec<f32> = router_logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| l + bias.get(idx).copied().unwrap_or(0.0))
        .collect();
    let chosen = select_top_k_experts(&biased, top_k, false);
    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = router_logits
        .iter()
        .map(|&l| {
            if l.is_finite() && max_logit.is_finite() {
                (l - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum_probs: f32 = probs.iter().sum();
    let mut out: Vec<(usize, f32)> = chosen
        .into_iter()
        .map(|(idx, _)| {
            let p = if sum_probs > 0.0 && sum_probs.is_finite() {
                probs[idx] / sum_probs
            } else {
                0.0
            };
            (idx, p)
        })
        .collect();
    let sel_sum: f32 = out.iter().map(|&(_, w)| w).sum();
    if sel_sum > 0.0 && sel_sum.is_finite() {
        for (_, w) in &mut out {
            *w /= sel_sum;
        }
    } else {
        #[expect(clippy::cast_precision_loss, reason = "top_k fits cleanly in f32")]
        let uniform = 1.0_f32 / (out.len().max(1) as f32);
        for (_, w) in &mut out {
            *w = uniform;
        }
    }
    out
}

/// Evaluates a fine-grained sparse `MoE` layer (`X100` router gate + `A100` top-`k` `SwiGLU`
/// experts) using either 3D stacked tensors (`ffn_*_exps.weight`) or per-expert tensors
/// (`ffn_*_exps.{e}.weight`), plus optional shared experts (`ffn_*_shexp.weight`).
///
/// Returns the selected `(expert_idx, normalized_weight)` pairs when routed expert tensors
/// were executed, or an empty vector when falling back to dense/deterministic synthetic FFN.
///
/// # Errors
///
/// Returns [`K3Error`] if router evaluation, expert tensor slicing, or kernel math fails.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit layer MoE dispatch helper shared by Qwen3Moe, DeepSeek2, and Qwen35Moe"
)]
pub fn dispatch_qwen3_moe_ffn(
    out: &mut [f32],
    norm_ffn_in: &[f32],
    layer: usize,
    num_experts: usize,
    top_k: usize,
    expert_ffn_dim: usize,
    use_shared_expert: bool,
    weights: &dyn ModelWeights,
) -> Result<Vec<(usize, f32)>, K3Error> {
    let hidden_dim = norm_ffn_in.len();
    out.fill(0.0);

    if use_shared_expert
        && weights.contains_tensor(&format!("blk.{layer}.ffn_gate_shexp.weight"))
        && weights.contains_tensor(&format!("blk.{layer}.ffn_up_shexp.weight"))
        && weights.contains_tensor(&format!("blk.{layer}.ffn_down_shexp.weight"))
    {
        let sg = weights.tensor(&format!("blk.{layer}.ffn_gate_shexp.weight"))?;
        let su = weights.tensor(&format!("blk.{layer}.ffn_up_shexp.weight"))?;
        let sd = weights.tensor(&format!("blk.{layer}.ffn_down_shexp.weight"))?;
        let sh_out = run_single_swiglu_expert(&sg, &su, &sd, norm_ffn_in, expert_ffn_dim)?;
        for (dst, &src) in out.iter_mut().zip(sh_out.iter()) {
            *dst += src;
        }
    }

    let router_name = format!("blk.{layer}.ffn_gate_inp.weight");
    if num_experts == 0 || top_k == 0 || !weights.contains_tensor(&router_name) {
        let dense = run_dense_swiglu_ffn(layer, hidden_dim, expert_ffn_dim, norm_ffn_in, weights)?;
        for (dst, &src) in out.iter_mut().zip(dense.iter()) {
            *dst += src;
        }
        return Ok(Vec::new());
    }

    let router_view = weights.tensor(&router_name)?;
    let router_elems = router_view.num_elements();
    let eff_experts = if router_elems > 0 && hidden_dim > 0 && router_elems % hidden_dim == 0 {
        (router_elems / hidden_dim).min(num_experts)
    } else {
        num_experts
    };
    let mut router_logits = vec![0.0_f32; eff_experts];
    matvec_any(&mut router_logits, &router_view, norm_ffn_in)?;

    let bias_name = format!("blk.{layer}.exp_probs_b.bias");
    let bias_opt = if weights.contains_tensor(&bias_name) {
        Some(read_f32_weight_or_default(
            weights,
            &bias_name,
            eff_experts,
            0.0,
        ))
    } else {
        None
    };
    let selected = select_experts_with_optional_bias(&router_logits, bias_opt.as_deref(), top_k);

    let g_stack = format!("blk.{layer}.ffn_gate_exps.weight");
    let u_stack = format!("blk.{layer}.ffn_up_exps.weight");
    let d_stack = format!("blk.{layer}.ffn_down_exps.weight");
    let mut ran_any_expert = false;

    for &(expert_idx, gate_weight) in &selected {
        let g_opt = slice_or_lookup_expert(
            weights,
            &g_stack,
            &format!("blk.{layer}.ffn_gate_exps.{expert_idx}.weight"),
            expert_idx,
            eff_experts,
        )?;
        let u_opt = slice_or_lookup_expert(
            weights,
            &u_stack,
            &format!("blk.{layer}.ffn_up_exps.{expert_idx}.weight"),
            expert_idx,
            eff_experts,
        )?;
        let d_opt = slice_or_lookup_expert(
            weights,
            &d_stack,
            &format!("blk.{layer}.ffn_down_exps.{expert_idx}.weight"),
            expert_idx,
            eff_experts,
        )?;
        if let (Some(g_view), Some(u_view), Some(d_view)) = (g_opt, u_opt, d_opt) {
            let exp_out =
                run_single_swiglu_expert(&g_view, &u_view, &d_view, norm_ffn_in, expert_ffn_dim)?;
            for (dst, &src) in out.iter_mut().zip(exp_out.iter()) {
                *dst += gate_weight * src;
            }
            ran_any_expert = true;
        }
    }

    if !ran_any_expert {
        let dense = run_dense_swiglu_ffn(layer, hidden_dim, expert_ffn_dim, norm_ffn_in, weights)?;
        for (dst, &src) in out.iter_mut().zip(dense.iter()) {
            *dst += src;
        }
    }

    Ok(selected)
}

#[derive(Debug, Clone, Copy)]
struct DenseAttnLayerSpec {
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_theta: f32,
    use_qkv_bias: bool,
    use_qk_norm: bool,
}

fn run_dense_attention_sublayer(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let attn_norm_w = read_f32_weight_or_default(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        hidden_dim,
        1.0,
    );
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, spec.rms_eps)?;

    let head_d_even = even_head_dim(spec.head_dim);
    let n_q = spec.n_heads_q.max(1);
    let n_kv = spec.n_heads_kv.max(1);
    let q_dim = n_q * head_d_even;
    let kv_dim = n_kv * head_d_even;

    let mut q = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_q.weight"),
        &norm_hidden,
        q_dim,
        0.5,
    )?;
    let mut k = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_k.weight"),
        &norm_hidden,
        kv_dim,
        0.25,
    )?;
    let mut v = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_v.weight"),
        &norm_hidden,
        kv_dim,
        0.125,
    )?;

    if spec.use_qkv_bias {
        add_bias_if_present(&mut q, weights, &format!("blk.{layer}.attn_q.bias"));
        add_bias_if_present(&mut k, weights, &format!("blk.{layer}.attn_k.bias"));
        add_bias_if_present(&mut v, weights, &format!("blk.{layer}.attn_v.bias"));
    }

    if spec.use_qk_norm {
        let q_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.attn_q_norm.weight"),
            head_d_even,
            1.0,
        );
        let k_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.attn_k_norm.weight"),
            head_d_even,
            1.0,
        );
        apply_per_head_standard_rmsnorm(&mut q, head_d_even, &q_norm_w, spec.rms_eps)?;
        apply_per_head_standard_rmsnorm(&mut k, head_d_even, &k_norm_w, spec.rms_eps)?;
    }

    apply_rope_inplace(
        &mut q,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        spec.rope_theta,
        None,
    )?;
    apply_rope_inplace(
        &mut k,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        spec.rope_theta,
        None,
    )?;

    let attn_out = run_gqa_causal_with_cache(
        &q,
        &k,
        &v,
        layer_kv,
        pos,
        GqaHeadDims {
            n_heads_q: n_q,
            n_heads_kv: n_kv,
            head_dim: head_d_even,
        },
    )?;
    let proj_attn = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_output.weight"),
        &attn_out,
        hidden_dim,
        0.25,
    )?;
    for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *h += r;
    }
    Ok(())
}

/// Absorbs `W^{UK}` (`[n_heads, d_nope, kv_lora_rank]`) into query non-rotary projections
/// (`[n_heads * d_nope]`), runs [`mla_absorbed_attention_f32`] directly over compressed latent
/// KV cache entries (`[c_kv, k_rope]`), and projects per-head latent context outputs through
/// absorbed `W^{UV}` (`[n_heads, v_head_dim, kv_lora_rank]`).
///
/// # Errors
///
/// Returns [`K3Error`] if slice lengths do not match the configured `MLA` head dimensions or if
/// [`mla_absorbed_attention_f32`] rejects the latent KV cache bounds.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit matrix-absorption reference contract for DeepSeek2 MLA verification"
)]
pub fn run_mla_absorbed_attention(
    out_heads: &mut [f32],
    q_nope: &[f32],
    q_rotary: &[f32],
    kv_latent_cache: &[f32],
    k_up_proj: &[f32],
    v_up_proj: &[f32],
    n_heads: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    rope_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
) -> Result<(), K3Error> {
    if n_heads == 0
        || kv_lora_rank == 0
        || qk_nope_head_dim == 0
        || rope_head_dim == 0
        || v_head_dim == 0
        || q_nope.len() != n_heads * qk_nope_head_dim
        || q_rotary.len() != n_heads * rope_head_dim
        || out_heads.len() != n_heads * v_head_dim
        || k_up_proj.len() != n_heads * qk_nope_head_dim * kv_lora_rank
        || v_up_proj.len() != n_heads * v_head_dim * kv_lora_rank
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "mla_absorbed".to_owned(),
            offset: 0,
            byte_len: size_of_val(out_heads),
            buffer_len: size_of_val(kv_latent_cache),
            reason: "MLA absorbed attention slice dimensions do not match configuration".to_owned(),
        });
    }

    // Absorb W_UK^T into q_nope: q_absorbed[h, r] = sum_d q_nope[h, d] * W_UK[h, d, r].
    let mut q_absorbed = vec![0.0_f32; n_heads * kv_lora_rank];
    for h in 0..n_heads {
        let q_h = &q_nope[h * qk_nope_head_dim..(h + 1) * qk_nope_head_dim];
        let k_up_head = &k_up_proj
            [h * qk_nope_head_dim * kv_lora_rank..(h + 1) * qk_nope_head_dim * kv_lora_rank];
        let dst = &mut q_absorbed[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        for (d, &q_val) in q_h.iter().enumerate() {
            let row = &k_up_head[d * kv_lora_rank..(d + 1) * kv_lora_rank];
            for (r, slot) in dst.iter_mut().enumerate() {
                *slot += q_val * row[r];
            }
        }
    }

    let mut out_latent = vec![0.0_f32; n_heads * kv_lora_rank];
    #[expect(
        clippy::cast_precision_loss,
        reason = "head dim sums fit comfortably in f32"
    )]
    let scale = 1.0_f32 / ((qk_nope_head_dim + rope_head_dim) as f32).sqrt();
    mla_absorbed_attention_f32(
        &mut out_latent,
        &q_absorbed,
        q_rotary,
        kv_latent_cache,
        n_heads,
        kv_lora_rank,
        rope_head_dim,
        seq_len,
        scale,
    )?;

    // Project latent weighted average through absorbed W_UV:
    // out_heads[h, v] = sum_r W_UV[h, v, r] * out_latent[h, r].
    out_heads.fill(0.0);
    for h in 0..n_heads {
        let lat_h = &out_latent[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        let v_up_head =
            &v_up_proj[h * v_head_dim * kv_lora_rank..(h + 1) * v_head_dim * kv_lora_rank];
        let dst_h = &mut out_heads[h * v_head_dim..(h + 1) * v_head_dim];
        for (v_idx, dst) in dst_h.iter_mut().enumerate() {
            let row = &v_up_head[v_idx * kv_lora_rank..(v_idx + 1) * kv_lora_rank];
            let mut acc = 0.0_f32;
            for (&w, &lat) in row.iter().zip(lat_h.iter()) {
                acc += w * lat;
            }
            *dst = acc;
        }
    }
    Ok(())
}

fn read_tensor_f32_exact_or_identity_uk_uv(
    weights: &dyn ModelWeights,
    name: &str,
    expected_len: usize,
    cols: usize,
    stride: usize,
) -> Vec<f32> {
    if weights.contains_tensor(name) {
        return read_f32_weight_or_default(weights, name, expected_len, 0.0);
    }
    let mut out = vec![0.0_f32; expected_len];
    if cols > 0 && stride > 0 {
        let rows_per_block = expected_len / stride;
        for row in 0..rows_per_block {
            out[row * stride + (row % cols)] = 1.0;
        }
    }
    out
}

fn execute_token_standard(
    exec: StandardExecSpec,
    step: TokenStep,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    validate_model_weights_adr023(exec.arch, weights)?;
    let layers = exec.num_layers.max(1);
    let hidden_dim = exec.attn.hidden_dim.max(8);
    let eps = exec.attn.rms_eps.max(1e-6);
    let mut hidden = embed_token(step.token_id, hidden_dim, weights)?;
    let layer_stride = kv_cache.len() / layers;

    for layer in 0..layers {
        let layer_kv = if layer_stride > 0 {
            &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride]
        } else {
            &mut []
        };
        run_dense_attention_sublayer(
            exec.attn,
            layer,
            step.pos,
            step.seq_len,
            &mut hidden,
            layer_kv,
            weights,
        )?;

        let ffn_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.ffn_norm.weight"),
            hidden_dim,
            1.0,
        );
        let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, eps)?;

        let ffn_out = if let Some(moe) = exec.moe {
            let mut m_out = vec![0.0_f32; hidden_dim];
            dispatch_qwen3_moe_ffn(
                &mut m_out,
                &norm_ffn_in,
                layer,
                moe.num_experts,
                moe.top_k,
                moe.expert_ffn_dim,
                moe.use_shared,
                weights,
            )?;
            m_out
        } else {
            run_dense_swiglu_ffn(layer, hidden_dim, exec.attn.ffn_dim, &norm_ffn_in, weights)?
        };
        for (h, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += r;
        }
    }

    let out_norm_w = read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, &hidden, &out_norm_w, eps)?;
    project_lm_head(weights, &normed_final, logits)?;
    Ok(())
}

impl Qwen2Recipe {
    const fn exec_spec(&self) -> StandardExecSpec {
        StandardExecSpec {
            arch: "qwen2",
            num_layers: self.num_layers,
            attn: DenseAttnLayerSpec {
                hidden_dim: self.hidden_dim,
                ffn_dim: self.ffn_dim,
                n_heads_q: self.n_heads_q,
                n_heads_kv: self.n_heads_kv,
                head_dim: self.head_dim,
                rms_eps: self.rms_eps,
                rope_theta: self.rope_theta,
                use_qkv_bias: true,
                use_qk_norm: false,
            },
            moe: None,
        }
    }
}

impl ArchRecipe for Qwen2Recipe {
    fn architecture_id(&self) -> &'static str {
        "qwen2"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        let _ = QUANT_SUPPORTED;
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let exec = self.exec_spec();
        run_prefill_with(ctx, "qwen2", |token_id, pos, seq_len, kv, logits| {
            execute_token_standard(
                exec,
                TokenStep {
                    token_id,
                    pos,
                    seq_len,
                },
                kv,
                logits,
                weights,
            )
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let exec = self.exec_spec();
        run_decode_with(ctx, "qwen2", |token_id, pos, kv, logits| {
            execute_token_standard(
                exec,
                TokenStep {
                    token_id,
                    pos,
                    seq_len: pos + 1,
                },
                kv,
                logits,
                weights,
            )
        })
    }
}

impl Qwen3Recipe {
    const fn exec_spec(&self) -> StandardExecSpec {
        StandardExecSpec {
            arch: "qwen3",
            num_layers: self.num_layers,
            attn: DenseAttnLayerSpec {
                hidden_dim: self.hidden_dim,
                ffn_dim: self.ffn_dim,
                n_heads_q: self.n_heads_q,
                n_heads_kv: self.n_heads_kv,
                head_dim: self.head_dim,
                rms_eps: self.rms_eps,
                rope_theta: self.rope_theta,
                use_qkv_bias: false,
                use_qk_norm: true,
            },
            moe: None,
        }
    }
}

impl ArchRecipe for Qwen3Recipe {
    fn architecture_id(&self) -> &'static str {
        "qwen3"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let exec = self.exec_spec();
        run_prefill_with(ctx, "qwen3", |token_id, pos, seq_len, kv, logits| {
            execute_token_standard(
                exec,
                TokenStep {
                    token_id,
                    pos,
                    seq_len,
                },
                kv,
                logits,
                weights,
            )
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let exec = self.exec_spec();
        run_decode_with(ctx, "qwen3", |token_id, pos, kv, logits| {
            execute_token_standard(
                exec,
                TokenStep {
                    token_id,
                    pos,
                    seq_len: pos + 1,
                },
                kv,
                logits,
                weights,
            )
        })
    }
}

impl Qwen3MoeRecipe {
    const fn exec_spec(&self) -> StandardExecSpec {
        StandardExecSpec {
            arch: "qwen3moe",
            num_layers: self.base.num_layers,
            attn: DenseAttnLayerSpec {
                hidden_dim: self.base.hidden_dim,
                ffn_dim: self.base.ffn_dim,
                n_heads_q: self.base.n_heads_q,
                n_heads_kv: self.base.n_heads_kv,
                head_dim: self.base.head_dim,
                rms_eps: self.base.rms_eps,
                rope_theta: self.base.rope_theta,
                use_qkv_bias: false,
                use_qk_norm: true,
            },
            moe: Some(MoeDispatchSpec {
                num_experts: self.num_experts,
                top_k: self.num_experts_per_tok,
                expert_ffn_dim: self.expert_ffn_dim,
                use_shared: false,
            }),
        }
    }
}

impl ArchRecipe for Qwen3MoeRecipe {
    fn architecture_id(&self) -> &'static str {
        "qwen3moe"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let exec = self.exec_spec();
        run_prefill_with(ctx, "qwen3moe", |token_id, pos, seq_len, kv, logits| {
            execute_token_standard(
                exec,
                TokenStep {
                    token_id,
                    pos,
                    seq_len,
                },
                kv,
                logits,
                weights,
            )
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let exec = self.exec_spec();
        run_decode_with(ctx, "qwen3moe", |token_id, pos, kv, logits| {
            execute_token_standard(
                exec,
                TokenStep {
                    token_id,
                    pos,
                    seq_len: pos + 1,
                },
                kv,
                logits,
                weights,
            )
        })
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "single-token DeepSeek2 MLA compression, absorbed attention, and shared+routed MoE pass"
)]
fn execute_deepseek2_token(
    recipe: &DeepSeek2Recipe,
    token_id: u32,
    pos: usize,
    seq_len: usize,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let layers = recipe.num_layers.max(1);
    let hidden_dim = recipe.hidden_dim.max(8);
    let eps = recipe.rms_eps.max(1e-6);
    let n_heads = recipe.n_heads.max(1);
    let d_c = recipe.mla.kv_lora_rank.max(1);
    let d_r = even_head_dim(recipe.mla.qk_rope_head_dim);
    let d_nope = recipe.mla.qk_nope_head_dim.max(1);
    let d_v = recipe.mla.v_head_dim.max(1);
    let latent_dim = d_c + d_r;

    let mut hidden = embed_token(token_id, hidden_dim, weights)?;
    let layer_stride = kv_cache.len() / layers;

    for layer in 0..layers {
        let attn_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.attn_norm.weight"),
            hidden_dim,
            1.0,
        );
        let mut norm_hidden = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut norm_hidden, &hidden, &attn_norm_w, eps)?;

        let q_nope = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.attn_q.weight"),
            &norm_hidden,
            n_heads * d_nope,
            0.5,
        )?;
        let mut q_rotary = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.attn_q_rope.weight"),
            &norm_hidden,
            n_heads * d_r,
            0.25,
        )?;
        apply_rope_inplace(
            &mut q_rotary,
            d_r,
            pos,
            seq_len.max(pos + 1),
            recipe.rope_theta,
            None,
        )?;

        let kv_compressed = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.attn_kv_a_mqa.weight"),
            &norm_hidden,
            latent_dim,
            0.25,
        )?;
        let kv_a_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.attn_kv_a_norm.weight"),
            d_c,
            1.0,
        );
        let mut c_kv = vec![0.0_f32; d_c];
        rmsnorm_standard(&mut c_kv, &kv_compressed[..d_c], &kv_a_norm_w, eps)?;
        let mut k_rope = kv_compressed[d_c..latent_dim].to_vec();
        apply_rope_inplace(
            &mut k_rope,
            d_r,
            pos,
            seq_len.max(pos + 1),
            recipe.rope_theta,
            None,
        )?;

        let k_up_proj = read_tensor_f32_exact_or_identity_uk_uv(
            weights,
            &format!("blk.{layer}.attn_w_uk.weight"),
            n_heads * d_nope * d_c,
            d_c,
            d_c,
        );
        let v_up_proj = read_tensor_f32_exact_or_identity_uk_uv(
            weights,
            &format!("blk.{layer}.attn_w_uv.weight"),
            n_heads * d_v * d_c,
            d_c,
            d_c,
        );

        let mut attn_heads = vec![0.0_f32; n_heads * d_v];
        if layer_stride >= latent_dim {
            let layer_kv = &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride];
            let capacity_slots = layer_stride / latent_dim;
            let slot = pos % capacity_slots;
            let active_len = (pos + 1).min(capacity_slots);
            let slot_slice = &mut layer_kv[slot * latent_dim..(slot + 1) * latent_dim];
            slot_slice[..d_c].copy_from_slice(&c_kv);
            slot_slice[d_c..].copy_from_slice(&k_rope);

            run_mla_absorbed_attention(
                &mut attn_heads,
                &q_nope,
                &q_rotary,
                &layer_kv[..active_len * latent_dim],
                &k_up_proj,
                &v_up_proj,
                n_heads,
                d_c,
                d_nope,
                d_r,
                d_v,
                active_len,
            )?;
        } else {
            let mut single_latent = Vec::with_capacity(latent_dim);
            single_latent.extend_from_slice(&c_kv);
            single_latent.extend_from_slice(&k_rope);
            run_mla_absorbed_attention(
                &mut attn_heads,
                &q_nope,
                &q_rotary,
                &single_latent,
                &k_up_proj,
                &v_up_proj,
                n_heads,
                d_c,
                d_nope,
                d_r,
                d_v,
                1,
            )?;
        }

        let proj_attn = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.attn_output.weight"),
            &attn_heads,
            hidden_dim,
            0.25,
        )?;
        for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
            *h += r;
        }

        let ffn_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.ffn_norm.weight"),
            hidden_dim,
            1.0,
        );
        let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, eps)?;

        let mut ffn_out = vec![0.0_f32; hidden_dim];
        dispatch_qwen3_moe_ffn(
            &mut ffn_out,
            &norm_ffn_in,
            layer,
            recipe.num_experts,
            recipe.num_experts_per_tok,
            recipe.expert_ffn_dim,
            recipe.num_shared_experts > 0,
            weights,
        )?;
        for (h, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += r;
        }
    }

    let out_norm_w = read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, &hidden, &out_norm_w, eps)?;
    project_lm_head(weights, &normed_final, logits)?;
    Ok(())
}

impl ArchRecipe for DeepSeek2Recipe {
    fn architecture_id(&self) -> &'static str {
        "deepseek2"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, "deepseek2", |tok, pos, seq_len, kv, logits| {
            execute_deepseek2_token(self, tok, pos, seq_len, kv, logits, weights)
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "deepseek2", |tok, pos, kv, logits| {
            execute_deepseek2_token(self, tok, pos, pos + 1, kv, logits, weights)
        })
    }
}

fn sigmoid_clamp(x: f32) -> f32 {
    let clamped = x.clamp(-6.0, 6.0);
    (1.0_f32 / (1.0_f32 + (-clamped).exp())).clamp(0.01, 0.99)
}

fn repeat_kv_for_q_heads(
    kv: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    if n_heads_kv >= n_heads_q || n_heads_kv == 0 {
        return kv[..n_heads_q * head_dim].to_vec();
    }
    let ratio = (n_heads_q / n_heads_kv).max(1);
    let mut out = Vec::with_capacity(n_heads_q * head_dim);
    for q_h in 0..n_heads_q {
        let kv_h = (q_h / ratio).min(n_heads_kv - 1);
        out.extend_from_slice(&kv[kv_h * head_dim..(kv_h + 1) * head_dim]);
    }
    out
}

fn choose_deltanet_geometry(
    q_dim: usize,
    desired_head_dim: usize,
    max_state_elems: usize,
) -> (usize, usize) {
    let start_d = desired_head_dim.min(q_dim).max(1);
    for d in (1..=start_d).rev() {
        if d > 0 && q_dim % d == 0 && q_dim * d <= max_state_elems {
            return (q_dim / d, d);
        }
    }
    (q_dim.max(1), 1)
}

struct DeltaNetQkv<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    decay_gate: &'a [f32],
    beta: &'a [f32],
}

fn step_deltanet_recurrence(
    layer_kv: &mut [f32],
    out: &mut [f32],
    qkv: &DeltaNetQkv<'_>,
    eff_heads: usize,
    eff_head_dim: usize,
    pos: usize,
) -> Result<(), K3Error> {
    let state_len = eff_heads * eff_head_dim * eff_head_dim;
    if state_len <= layer_kv.len() {
        let state = &mut layer_kv[..state_len];
        if pos == 0 {
            state.fill(0.0);
        }
        return gated_deltanet_step_f32(
            state,
            out,
            qkv.q,
            qkv.k,
            qkv.v,
            qkv.decay_gate,
            qkv.beta,
            eff_heads,
            eff_head_dim,
        );
    }
    let mut scratch_state = vec![0.0_f32; state_len];
    gated_deltanet_step_f32(
        &mut scratch_state,
        out,
        qkv.q,
        qkv.k,
        qkv.v,
        qkv.decay_gate,
        qkv.beta,
        eff_heads,
        eff_head_dim,
    )
}

fn run_gated_deltanet_sublayer(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let attn_norm_w = read_f32_weight_or_default(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        hidden_dim,
        1.0,
    );
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, spec.rms_eps)?;

    let head_d_even = even_head_dim(spec.head_dim);
    let n_q = spec.n_heads_q.max(1);
    let n_kv = spec.n_heads_kv.max(1);
    let q_dim = n_q * head_d_even;
    let kv_dim = n_kv * head_d_even;

    let q = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_q.weight"),
        &norm_hidden,
        q_dim,
        0.5,
    )?;
    let k_raw = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_k.weight"),
        &norm_hidden,
        kv_dim,
        0.25,
    )?;
    let v_raw = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_v.weight"),
        &norm_hidden,
        kv_dim,
        0.125,
    )?;
    let k = repeat_kv_for_q_heads(&k_raw, n_q, n_kv, head_d_even);
    let v = repeat_kv_for_q_heads(&v_raw, n_q, n_kv, head_d_even);

    let (eff_heads, eff_head_dim) = choose_deltanet_geometry(q_dim, head_d_even, layer_kv.len());
    let raw_decay = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ssm_a.weight"),
        &norm_hidden,
        eff_heads,
        0.8,
    )?;
    let decay_gate: Vec<f32> = raw_decay.into_iter().map(sigmoid_clamp).collect();
    let raw_beta = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ssm_beta.weight"),
        &norm_hidden,
        eff_heads,
        0.2,
    )?;
    let beta: Vec<f32> = raw_beta.into_iter().map(sigmoid_clamp).collect();

    let mut out = vec![0.0_f32; q_dim];
    step_deltanet_recurrence(
        layer_kv,
        &mut out,
        &DeltaNetQkv {
            q: &q,
            k: &k,
            v: &v,
            decay_gate: &decay_gate,
            beta: &beta,
        },
        eff_heads,
        eff_head_dim,
        pos,
    )?;

    let proj_attn = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_output.weight"),
        &out,
        hidden_dim,
        0.25,
    )?;
    for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *h += r;
    }
    Ok(())
}

fn execute_qwen35_token(
    base: &Qwen35Recipe,
    moe: Option<MoeDispatchSpec>,
    step: TokenStep,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let layers = base.num_layers.max(1);
    let hidden_dim = base.hidden_dim.max(8);
    let eps = base.rms_eps.max(1e-6);
    let spec = DenseAttnLayerSpec {
        hidden_dim,
        ffn_dim: base.ffn_dim,
        n_heads_q: base.n_heads_q,
        n_heads_kv: base.n_heads_kv,
        head_dim: base.head_dim,
        rms_eps: eps,
        rope_theta: base.rope_theta,
        use_qkv_bias: false,
        use_qk_norm: true,
    };

    let mut hidden = embed_token(step.token_id, hidden_dim, weights)?;
    let layer_stride = kv_cache.len() / layers;

    for layer in 0..layers {
        let layer_kv = if layer_stride > 0 {
            &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride]
        } else {
            &mut []
        };
        let force_deltanet = weights.contains_tensor(&format!("blk.{layer}.ssm_a.weight"))
            || weights.contains_tensor(&format!("blk.{layer}.ssm_a"));
        let is_full = !force_deltanet && base.is_full_attention_layer(layer);
        if is_full {
            run_dense_attention_sublayer(
                spec,
                layer,
                step.pos,
                step.seq_len,
                &mut hidden,
                layer_kv,
                weights,
            )?;
        } else {
            run_gated_deltanet_sublayer(spec, layer, step.pos, &mut hidden, layer_kv, weights)?;
        }

        let ffn_norm_w = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.ffn_norm.weight"),
            hidden_dim,
            1.0,
        );
        let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, eps)?;

        let ffn_out = if let Some(moe_cfg) = moe {
            let mut m_out = vec![0.0_f32; hidden_dim];
            dispatch_qwen3_moe_ffn(
                &mut m_out,
                &norm_ffn_in,
                layer,
                moe_cfg.num_experts,
                moe_cfg.top_k,
                moe_cfg.expert_ffn_dim,
                moe_cfg.use_shared,
                weights,
            )?;
            m_out
        } else {
            run_dense_swiglu_ffn(layer, hidden_dim, base.ffn_dim, &norm_ffn_in, weights)?
        };
        for (h, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += r;
        }
    }

    let out_norm_w = read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, &hidden, &out_norm_w, eps)?;
    project_lm_head(weights, &normed_final, logits)?;
    Ok(())
}

impl ArchRecipe for Qwen35Recipe {
    fn architecture_id(&self) -> &'static str {
        "qwen35"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, "qwen35", |token_id, pos, seq_len, kv, logits| {
            execute_qwen35_token(
                self,
                None,
                TokenStep {
                    token_id,
                    pos,
                    seq_len,
                },
                kv,
                logits,
                weights,
            )
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "qwen35", |token_id, pos, kv, logits| {
            execute_qwen35_token(
                self,
                None,
                TokenStep {
                    token_id,
                    pos,
                    seq_len: pos + 1,
                },
                kv,
                logits,
                weights,
            )
        })
    }
}

impl Qwen35MoeRecipe {
    const fn moe_spec(&self) -> MoeDispatchSpec {
        MoeDispatchSpec {
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            expert_ffn_dim: self.expert_ffn_dim,
            use_shared: self.num_shared_experts > 0,
        }
    }
}

impl ArchRecipe for Qwen35MoeRecipe {
    fn architecture_id(&self) -> &'static str {
        "qwen35moe"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let moe = self.moe_spec();
        run_prefill_with(ctx, "qwen35moe", |token_id, pos, seq_len, kv, logits| {
            execute_qwen35_token(
                &self.base,
                Some(moe),
                TokenStep {
                    token_id,
                    pos,
                    seq_len,
                },
                kv,
                logits,
                weights,
            )
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let moe = self.moe_spec();
        run_decode_with(ctx, "qwen35moe", |token_id, pos, kv, logits| {
            execute_qwen35_token(
                &self.base,
                Some(moe),
                TokenStep {
                    token_id,
                    pos,
                    seq_len: pos + 1,
                },
                kv,
                logits,
                weights,
            )
        })
    }
}

fn run_prefill_with<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(u32, usize, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let prompt = ctx
        .jump_forward_tokens()
        .unwrap_or_else(|| ctx.prompt_tokens())
        .to_vec();
    if prompt.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "prefill prompt_tokens cannot be empty".to_owned(),
        });
    }
    let seq_len = prompt.len();
    let start_pos = ctx.position();
    for (offset, &tok) in prompt.iter().enumerate() {
        let mut step_logits = vec![0.0_f32; ctx.logits().len()];
        step_fn(
            tok,
            start_pos + offset,
            start_pos + seq_len,
            ctx.kv_cache_mut(),
            &mut step_logits,
        )?;
        if offset + 1 == seq_len {
            ctx.logits_mut().copy_from_slice(&step_logits);
        }
    }
    ctx.advance_position(seq_len)?;
    Ok(())
}

fn run_decode_with<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(u32, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let tok = ctx
        .decode_token()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "decode_step called without a staged decode token".to_owned(),
        })?;
    let pos = ctx.position();
    let mut step_logits = vec![0.0_f32; ctx.logits().len()];
    step_fn(tok, pos, ctx.kv_cache_mut(), &mut step_logits)?;
    ctx.logits_mut().copy_from_slice(&step_logits);
    ctx.advance_position(1)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{GgufFile, GgufValue};

    use super::*;
    use crate::capability::verify_model_capabilities;

    struct OwnedWeights {
        tensors: BTreeMap<String, (Vec<usize>, Vec<u8>, usize)>,
        total_bytes: usize,
        allow_tied_output: bool,
    }

    impl OwnedWeights {
        fn new(allow_tied_output: bool) -> Self {
            Self {
                tensors: BTreeMap::new(),
                total_bytes: 0,
                allow_tied_output,
            }
        }

        fn insert_f32(&mut self, name: &str, shape: &[usize], vals: &[f32]) {
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let offset = self.total_bytes;
            self.total_bytes += bytes.len();
            self.tensors
                .insert(name.to_owned(), (shape.to_vec(), bytes, offset));
        }

        fn insert_filled(&mut self, name: &str, shape: &[usize], val: f32) {
            let elems: usize = shape.iter().product();
            let vals = vec![val; elems];
            self.insert_f32(name, shape, &vals);
        }
    }

    impl ModelWeights for OwnedWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            let resolved = if self.tensors.contains_key(name) {
                name
            } else if name == "output.weight"
                && self.allow_tied_output
                && self.tensors.contains_key("token_embd.weight")
            {
                "token_embd.weight"
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: self.total_bytes,
                    reason: format!("tensor '{name}' missing in OwnedWeights"),
                });
            };
            let (resolved_name, (shape, bytes, _offset)) = self
                .tensors
                .get_key_value(resolved)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: self.total_bytes,
                    reason: format!("tensor '{name}' missing in OwnedWeights"),
                })?;
            let view_name = if name == "output.weight" && resolved == "token_embd.weight" {
                "output.weight"
            } else {
                resolved_name.as_str()
            };
            TensorView::from_bytes(view_name, QuantType::F32, shape, 0, bytes)
        }

        fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.total_bytes
        }
    }

    fn build_qwen_dense_weights(
        hidden: usize,
        ffn: usize,
        vocab: usize,
        is_qwen3: bool,
    ) -> OwnedWeights {
        let mut w = OwnedWeights::new(true);
        let mut emb = vec![0.0_f32; vocab * hidden];
        for (i, slot) in emb.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small synthetic test matrix")]
            {
                *slot = ((i % 19) as f32 - 9.0) * 0.04;
            }
        }
        w.insert_f32("token_embd.weight", &[hidden, vocab], &emb);
        w.insert_filled("output_norm.weight", &[hidden], 1.0);
        w.insert_filled("blk.0.attn_norm.weight", &[hidden], 1.0);
        w.insert_filled("blk.0.attn_q.weight", &[hidden, hidden], 0.05);
        w.insert_filled("blk.0.attn_k.weight", &[hidden, hidden], 0.04);
        w.insert_filled("blk.0.attn_v.weight", &[hidden, hidden], 0.03);
        w.insert_filled("blk.0.attn_output.weight", &[hidden, hidden], 0.05);
        if is_qwen3 {
            w.insert_filled("blk.0.attn_q_norm.weight", &[hidden / 2], 1.0);
            w.insert_filled("blk.0.attn_k_norm.weight", &[hidden / 2], 1.0);
        } else {
            w.insert_filled("blk.0.attn_q.bias", &[hidden], 0.02);
            w.insert_filled("blk.0.attn_k.bias", &[hidden], -0.01);
            w.insert_filled("blk.0.attn_v.bias", &[hidden], 0.03);
        }
        w.insert_filled("blk.0.ffn_norm.weight", &[hidden], 1.0);
        w.insert_filled("blk.0.ffn_gate.weight", &[hidden, ffn], 0.04);
        w.insert_filled("blk.0.ffn_up.weight", &[hidden, ffn], 0.03);
        w.insert_filled("blk.0.ffn_down.weight", &[ffn, hidden], 0.04);
        w
    }

    #[test]
    fn test_qwen2_vs_qwen3_and_tied_embeddings() {
        let q2 = Qwen2Recipe {
            num_layers: 1,
            hidden_dim: 8,
            ffn_dim: 16,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            vocab_size: 16,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        };
        let q3 = Qwen3Recipe {
            num_layers: 1,
            hidden_dim: 8,
            ffn_dim: 16,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            vocab_size: 16,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        };

        let q2_w = build_qwen_dense_weights(8, 16, 16, false);
        let q3_w = build_qwen_dense_weights(8, 16, 16, true);
        let out_alias = q3_w
            .tensor("output.weight")
            .expect("tied output.weight alias");
        assert_eq!(out_alias.shape(), &[8, 16]);

        let mut kv2 = vec![0.0_f32; 32 * 2 * 8];
        let mut logits2 = vec![0.0_f32; 16];
        let mut ctx2 = ExecutionContext::new(&[1, 3], 16, &mut logits2, &mut kv2).unwrap();
        q2.prefill_step(&mut ctx2, &q2_w).expect("qwen2 prefill");
        ctx2.prepare_decode(5).unwrap();
        q2.decode_step(&mut ctx2, &q2_w).expect("qwen2 decode");
        assert_eq!(ctx2.position(), 3);
        assert!(logits2.iter().all(|v| v.is_finite()) && logits2.iter().any(|v| v.abs() > 1e-6));

        let mut kv3 = vec![0.0_f32; 32 * 2 * 8];
        let mut logits3 = vec![0.0_f32; 16];
        let mut ctx3 = ExecutionContext::new(&[1, 3], 16, &mut logits3, &mut kv3).unwrap();
        q3.prefill_step(&mut ctx3, &q3_w).expect("qwen3 prefill");
        assert!(logits3.iter().all(|v| v.is_finite()) && logits3.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_qwen2_vs_qwen3_adr023_separation() {
        let qwen2_kvs = [
            ("qwen2.block_count", GgufValue::Uint32(1)),
            ("qwen2.embedding_length", GgufValue::Uint32(16)),
            ("qwen2.feed_forward_length", GgufValue::Uint32(32)),
            ("qwen2.attention.head_count", GgufValue::Uint32(2)),
            ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
        ];
        let qwen2_specs = [
            ("blk.0.attn_q.weight", QuantType::F32, &[16, 16][..]),
            ("blk.0.attn_q.bias", QuantType::F32, &[16][..]),
            ("blk.0.attn_k.bias", QuantType::F32, &[16][..]),
            ("blk.0.attn_v.bias", QuantType::F32, &[16][..]),
        ];
        let q2_bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 64, &qwen2_kvs, &qwen2_specs);
        let q2_gguf = GgufFile::parse(&q2_bytes).expect("parse qwen2");
        let r2 = verify_model_capabilities(&q2_gguf).expect("valid qwen2 must pass");
        assert_eq!(r2.architecture_id(), "qwen2");

        // Mislabeled qwen2 lacking QKV bias and carrying QK-norm must fail ADR-023 check.
        let wrong_q2_specs = [
            ("blk.0.attn_q.weight", QuantType::F32, &[16, 16][..]),
            ("blk.0.attn_q_norm.weight", QuantType::F32, &[8][..]),
            ("blk.0.attn_k_norm.weight", QuantType::F32, &[8][..]),
        ];
        let bad_q2_bytes =
            create_synthetic_gguf_v3_with_kvs("qwen2", 64, &qwen2_kvs, &wrong_q2_specs);
        let bad_q2_gguf = GgufFile::parse(&bad_q2_bytes).expect("parse mismatched qwen2");
        assert!(matches!(
            verify_model_capabilities(&bad_q2_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let qwen3_kvs = [
            ("qwen3.block_count", GgufValue::Uint32(1)),
            ("qwen3.embedding_length", GgufValue::Uint32(16)),
            ("qwen3.feed_forward_length", GgufValue::Uint32(32)),
            ("qwen3.attention.head_count", GgufValue::Uint32(2)),
            ("qwen3.attention.head_count_kv", GgufValue::Uint32(2)),
        ];
        let q3_bytes = create_synthetic_gguf_v3_with_kvs("qwen3", 64, &qwen3_kvs, &wrong_q2_specs);
        let q3_gguf = GgufFile::parse(&q3_bytes).expect("parse valid qwen3");
        let r3 = verify_model_capabilities(&q3_gguf).expect("valid qwen3 must pass");
        assert_eq!(r3.architecture_id(), "qwen3");

        let bad_q3_bytes = create_synthetic_gguf_v3_with_kvs("qwen3", 64, &qwen3_kvs, &qwen2_specs);
        let bad_q3_gguf = GgufFile::parse(&bad_q3_bytes).expect("parse mismatched qwen3");
        assert!(matches!(
            verify_model_capabilities(&bad_q3_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_qwen2_qwen3_capability_mismatch_on_wrong_tensors() {
        let q2_recipe = Qwen2Recipe {
            num_layers: 1,
            hidden_dim: 8,
            ffn_dim: 16,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            vocab_size: 16,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        };
        let q3_recipe = Qwen3Recipe {
            num_layers: 1,
            hidden_dim: 8,
            ffn_dim: 16,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            vocab_size: 16,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        };

        let q2_weights = build_qwen_dense_weights(8, 16, 16, false);
        let q3_weights = build_qwen_dense_weights(8, 16, 16, true);
        let mut kv = vec![0.0_f32; 64];
        let mut logits = vec![0.0_f32; 16];

        let mut ctx_a = ExecutionContext::new(&[1], 8, &mut logits, &mut kv).unwrap();
        let err_a = q2_recipe
            .prefill_step(&mut ctx_a, &q3_weights)
            .expect_err("Qwen2Recipe must reject qwen3 QK-norm tensors");
        assert!(matches!(err_a, K3Error::CapabilityMismatch { .. }));

        let mut ctx_b = ExecutionContext::new(&[1], 8, &mut logits, &mut kv).unwrap();
        let err_b = q3_recipe
            .prefill_step(&mut ctx_b, &q2_weights)
            .expect_err("Qwen3Recipe must reject qwen2 QKV bias tensors");
        assert!(matches!(err_b, K3Error::CapabilityMismatch { .. }));
    }

    fn build_qwen3_moe_weights(
        hidden: usize,
        expert_ffn: usize,
        num_experts: usize,
        vocab: usize,
    ) -> OwnedWeights {
        let mut w = OwnedWeights::new(true);
        w.insert_filled("token_embd.weight", &[hidden, vocab], 0.2);
        w.insert_filled("output_norm.weight", &[hidden], 1.0);
        w.insert_filled("blk.0.attn_norm.weight", &[hidden], 1.0);
        w.insert_filled("blk.0.attn_q.weight", &[hidden, hidden], 0.05);
        w.insert_filled("blk.0.attn_k.weight", &[hidden, hidden], 0.04);
        w.insert_filled("blk.0.attn_v.weight", &[hidden, hidden], 0.03);
        w.insert_filled("blk.0.attn_q_norm.weight", &[hidden / 2], 1.0);
        w.insert_filled("blk.0.attn_k_norm.weight", &[hidden / 2], 1.0);
        w.insert_filled("blk.0.attn_output.weight", &[hidden, hidden], 0.05);
        w.insert_filled("blk.0.ffn_norm.weight", &[hidden], 1.0);

        // Make experts 5, 19, 31, 47, 63, 79, 95, 111 win routing cleanly.
        let mut router = vec![-1.0_f32; num_experts * hidden];
        for (idx, exp) in [5_usize, 19, 31, 47, 63, 79, 95, 111].iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small expert rank fits in f32")]
            let score = 3.0 + (8 - idx) as f32 * 0.25;
            for c in 0..hidden {
                router[exp * hidden + c] = score;
            }
        }
        w.insert_f32("blk.0.ffn_gate_inp.weight", &[hidden, num_experts], &router);
        let mut gate_stack = Vec::with_capacity(num_experts * hidden * expert_ffn);
        let mut up_stack = Vec::with_capacity(num_experts * hidden * expert_ffn);
        let mut down_stack = Vec::with_capacity(num_experts * expert_ffn * hidden);
        for exp in 0..num_experts {
            #[expect(clippy::cast_precision_loss, reason = "small expert index fits in f32")]
            let scale = ((exp % 7) as f32 + 1.0) * 0.02;
            gate_stack.extend(std::iter::repeat_n(scale, hidden * expert_ffn));
            up_stack.extend(std::iter::repeat_n(0.1, hidden * expert_ffn));
            down_stack.extend(std::iter::repeat_n(0.05, expert_ffn * hidden));
        }
        w.insert_f32(
            "blk.0.ffn_gate_exps.weight",
            &[hidden, expert_ffn, num_experts],
            &gate_stack,
        );
        w.insert_f32(
            "blk.0.ffn_up_exps.weight",
            &[hidden, expert_ffn, num_experts],
            &up_stack,
        );
        w.insert_f32(
            "blk.0.ffn_down_exps.weight",
            &[expert_ffn, hidden, num_experts],
            &down_stack,
        );
        w
    }

    #[test]
    fn test_qwen3moe_128_top8_dispatch() {
        let w = build_qwen3_moe_weights(8, 12, 128, 16);
        let norm_in = vec![0.4_f32; 8];
        let mut out = vec![0.0_f32; 8];
        let selected = dispatch_qwen3_moe_ffn(&mut out, &norm_in, 0, 128, 8, 12, false, &w)
            .expect("top-8 of 128 dispatch succeeds");
        assert_eq!(selected.len(), 8);
        let weight_sum: f32 = selected.iter().map(|&(_, weight)| weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-4);
        assert_eq!(selected[0].0, 5);
        assert_eq!(selected[7].0, 111);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_qwen3moe_x100_router_a100_experts() {
        let recipe = Qwen3MoeRecipe::new_30b_a3b();
        assert_eq!(recipe.architecture_id(), "qwen3moe");
        assert_eq!(recipe.num_experts, 128);
        assert_eq!(recipe.num_experts_per_tok, 8);
        assert_eq!(recipe.router_norm_lane(), Lane::X100);
        assert_eq!(recipe.expert_compute_lane(), Lane::A100);
        assert_eq!(
            recipe.placement_plan(),
            PlacementPlan::op(Lane::A100, Lane::X100)
        );
    }

    #[test]
    fn test_qwen3moe_recipe_forward_pass() {
        let mut recipe = Qwen3MoeRecipe::new_30b_a3b();
        recipe.base.num_layers = 1;
        recipe.base.hidden_dim = 8;
        recipe.base.ffn_dim = 12;
        recipe.base.n_heads_q = 2;
        recipe.base.n_heads_kv = 2;
        recipe.base.head_dim = 4;
        recipe.base.vocab_size = 16;
        recipe.expert_ffn_dim = 12;

        let w = build_qwen3_moe_weights(8, 12, 128, 16);
        let mut kv = vec![0.0_f32; 32 * 2 * 8];
        let mut logits = vec![0.0_f32; 16];
        let mut ctx = ExecutionContext::new(&[1, 2], 16, &mut logits, &mut kv).unwrap();
        recipe.prefill_step(&mut ctx, &w).expect("qwen3moe prefill");
        ctx.prepare_decode(4).unwrap();
        recipe.decode_step(&mut ctx, &w).expect("qwen3moe decode");
        assert_eq!(ctx.position(), 3);
        assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_deepseek2_mla_matrix_absorption() {
        let n_heads = 2_usize;
        let d_c = 6_usize;
        let d_nope = 4_usize;
        let d_r = 2_usize;
        let d_v = 5_usize;
        let seq_len = 4_usize;
        let latent_dim = d_c + d_r;

        #[expect(clippy::cast_precision_loss, reason = "small test tensors fit in f32")]
        let pattern = |len: usize, modulo: usize, offset: f32, factor: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i % modulo) as f32 - offset) * factor)
                .collect()
        };
        let q_nope = pattern(n_heads * d_nope, 7, 3.0, 0.13);
        let q_rotary = pattern(n_heads * d_r, 5, 2.0, 0.17);
        let kv_latent = pattern(seq_len * latent_dim, 11, 5.0, 0.09);
        let k_up_proj = pattern(n_heads * d_nope * d_c, 13, 6.0, 0.07);
        let v_up_proj = pattern(n_heads * d_v * d_c, 9, 4.0, 0.08);

        let mut absorbed_out = vec![0.0_f32; n_heads * d_v];
        run_mla_absorbed_attention(
            &mut absorbed_out,
            &q_nope,
            &q_rotary,
            &kv_latent,
            &k_up_proj,
            &v_up_proj,
            n_heads,
            d_c,
            d_nope,
            d_r,
            d_v,
            seq_len,
        )
        .expect("absorbed MLA succeeds");

        // Reference naive decompression: k_nope = W_UK * c_kv, v = W_UV * c_kv per token.
        #[expect(
            clippy::cast_precision_loss,
            reason = "small test head_dim fits in f32"
        )]
        let scale = 1.0_f32 / ((d_nope + d_r) as f32).sqrt();
        let mut naive_out = vec![0.0_f32; n_heads * d_v];
        for h in 0..n_heads {
            let q_nope_head = &q_nope[h * d_nope..(h + 1) * d_nope];
            let q_rotary_head = &q_rotary[h * d_r..(h + 1) * d_r];
            let k_up_head = &k_up_proj[h * d_nope * d_c..(h + 1) * d_nope * d_c];
            let v_up_head = &v_up_proj[h * d_v * d_c..(h + 1) * d_v * d_c];

            let mut scores = Vec::with_capacity(seq_len);
            let mut vals = Vec::with_capacity(seq_len * d_v);
            for s in 0..seq_len {
                let c_s = &kv_latent[s * latent_dim..s * latent_dim + d_c];
                let kr_s = &kv_latent[s * latent_dim + d_c..(s + 1) * latent_dim];
                let mut score_nope = 0.0_f32;
                for d in 0..d_nope {
                    let row = &k_up_head[d * d_c..(d + 1) * d_c];
                    let k_d: f32 = row.iter().zip(c_s.iter()).map(|(&a, &b)| a * b).sum();
                    score_nope += q_nope_head[d] * k_d;
                }
                let score_rotary: f32 = q_rotary_head
                    .iter()
                    .zip(kr_s.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                scores.push((score_nope + score_rotary) * scale);
                for v_i in 0..d_v {
                    let row = &v_up_head[v_i * d_c..(v_i + 1) * d_c];
                    vals.push(
                        row.iter()
                            .zip(c_s.iter())
                            .map(|(&a, &b)| a * b)
                            .sum::<f32>(),
                    );
                }
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let dst = &mut naive_out[h * d_v..(h + 1) * d_v];
            for (s, &e) in exps.iter().enumerate() {
                let p = e / sum_e;
                for (v_i, d_slot) in dst.iter_mut().enumerate() {
                    *d_slot += p * vals[s * d_v + v_i];
                }
            }
        }

        let max_diff = absorbed_out
            .iter()
            .zip(naive_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "absorbed MLA must match naive KV decompression (max_diff={max_diff})"
        );
    }

    fn build_deepseek2_test_recipe_and_weights() -> (DeepSeek2Recipe, OwnedWeights) {
        let recipe = DeepSeek2Recipe {
            num_layers: 1,
            hidden_dim: 8,
            ffn_dim: 12,
            n_heads: 2,
            vocab_size: 16,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            mla: DeepSeek2MlaConfig {
                kv_lora_rank: 4,
                q_lora_rank: None,
                qk_nope_head_dim: 4,
                qk_rope_head_dim: 2,
                v_head_dim: 4,
            },
            num_experts: 8,
            num_experts_per_tok: 2,
            num_shared_experts: 1,
            expert_ffn_dim: 8,
        };
        let mut w = OwnedWeights::new(true);
        w.insert_filled("token_embd.weight", &[8, 16], 0.15);
        w.insert_filled("output_norm.weight", &[8], 1.0);
        w.insert_filled("blk.0.attn_norm.weight", &[8], 1.0);
        w.insert_filled("blk.0.attn_q.weight", &[8, 8], 0.05);
        w.insert_filled("blk.0.attn_q_rope.weight", &[8, 4], 0.04);
        w.insert_filled("blk.0.attn_kv_a_mqa.weight", &[8, 6], 0.06);
        w.insert_filled("blk.0.attn_kv_a_norm.weight", &[4], 1.0);
        w.insert_filled("blk.0.attn_w_uk.weight", &[4, 8], 0.05);
        w.insert_filled("blk.0.attn_w_uv.weight", &[4, 8], 0.05);
        w.insert_filled("blk.0.attn_output.weight", &[8, 8], 0.05);
        w.insert_filled("blk.0.ffn_norm.weight", &[8], 1.0);
        w.insert_filled("blk.0.ffn_gate_shexp.weight", &[8, 8], 0.08);
        w.insert_filled("blk.0.ffn_up_shexp.weight", &[8, 8], 0.05);
        w.insert_filled("blk.0.ffn_down_shexp.weight", &[8, 8], 0.04);
        w.insert_filled("blk.0.ffn_gate_inp.weight", &[8, 8], 0.1);
        w.insert_f32(
            "blk.0.exp_probs_b.bias",
            &[8],
            &[0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        );
        w.insert_filled("blk.0.ffn_gate_exps.weight", &[8, 8, 8], 0.03);
        w.insert_filled("blk.0.ffn_up_exps.weight", &[8, 8, 8], 0.02);
        w.insert_filled("blk.0.ffn_down_exps.weight", &[8, 8, 8], 0.03);
        (recipe, w)
    }

    #[test]
    fn test_deepseek2_mla_and_shared_moe_recipe() {
        let (recipe, w) = build_deepseek2_test_recipe_and_weights();
        let mut out = vec![0.0_f32; 8];
        let norm_in = vec![0.5_f32; 8];
        let selected = dispatch_qwen3_moe_ffn(&mut out, &norm_in, 0, 8, 2, 8, true, &w)
            .expect("deepseek2 shared + routed MoE succeeds");
        assert_eq!(selected.len(), 2);
        // Optional bias routing prefers experts 5 and 1 while normalizing over unbiased probs.
        let ids: Vec<usize> = selected.iter().map(|&(idx, _)| idx).collect();
        assert!(ids.contains(&5) && ids.contains(&1));
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|v| v.abs() > 1e-6));
        assert_eq!(recipe.architecture_id(), "deepseek2");
    }

    #[test]
    fn test_deepseek2_recipe_e2e() {
        let (recipe, w) = build_deepseek2_test_recipe_and_weights();
        // Latent stride: 16 tokens * (kv_lora_rank=4 + qk_rope_head_dim=2) = 96 floats.
        let mut kv = vec![0.0_f32; 16 * 6];
        let mut logits = vec![0.0_f32; 16];
        let mut ctx = ExecutionContext::new(&[1, 2, 3], 16, &mut logits, &mut kv).unwrap();
        recipe
            .prefill_step(&mut ctx, &w)
            .expect("deepseek2 prefill");
        assert_eq!(ctx.position(), 3);
        ctx.prepare_decode(4).unwrap();
        recipe.decode_step(&mut ctx, &w).expect("deepseek2 decode");
        assert_eq!(ctx.position(), 4);
        assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
        // Confirm latent KV cache slots were populated across positions 0..4.
        assert!(kv[..4 * 6].iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_qwen35_hybrid_gated_deltanet_recipe() {
        let recipe = Qwen35Recipe {
            num_layers: 4,
            hidden_dim: 8,
            ffn_dim: 12,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            vocab_size: 16,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            full_attention_interval: 4,
        };
        assert!(!recipe.is_full_attention_layer(0));
        assert!(!recipe.is_full_attention_layer(1));
        assert!(!recipe.is_full_attention_layer(2));
        assert!(recipe.is_full_attention_layer(3));

        let w = OwnedWeights::new(true);
        // 4 layers * 2 * 8 * 16 slots = 1024 floats (fits 2*4*4 = 32 DeltaNet state + GQA ring).
        let mut kv = vec![0.0_f32; 4 * 16 * 16];
        let mut logits = vec![0.0_f32; 16];
        let mut ctx = ExecutionContext::new(&[1, 2], 16, &mut logits, &mut kv).unwrap();
        recipe.prefill_step(&mut ctx, &w).expect("qwen35 prefill");
        ctx.prepare_decode(3).unwrap();
        recipe.decode_step(&mut ctx, &w).expect("qwen35 decode");
        assert_eq!(ctx.position(), 3);
        assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
        // Recurrent state written into layer 0 DeltaNet slot.
        assert!(kv[..32].iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_qwen35moe_shared_and_routed_recipe() {
        let recipe = Qwen35MoeRecipe {
            base: Qwen35Recipe {
                num_layers: 4,
                hidden_dim: 8,
                ffn_dim: 12,
                n_heads_q: 2,
                n_heads_kv: 2,
                head_dim: 4,
                vocab_size: 16,
                rms_eps: 1e-6,
                rope_theta: 1_000_000.0,
                full_attention_interval: 4,
            },
            num_experts: 8,
            num_experts_per_tok: 2,
            num_shared_experts: 1,
            expert_ffn_dim: 8,
        };
        let mut w = OwnedWeights::new(true);
        w.insert_filled("token_embd.weight", &[8, 16], 0.12);
        w.insert_filled("output_norm.weight", &[8], 1.0);
        w.insert_filled("blk.0.ffn_gate_shexp.weight", &[8, 8], 0.06);
        w.insert_filled("blk.0.ffn_up_shexp.weight", &[8, 8], 0.04);
        w.insert_filled("blk.0.ffn_down_shexp.weight", &[8, 8], 0.05);
        w.insert_filled("blk.0.ffn_gate_inp.weight", &[8, 8], 0.09);
        w.insert_filled("blk.0.ffn_gate_exps.weight", &[8, 8, 8], 0.03);
        w.insert_filled("blk.0.ffn_up_exps.weight", &[8, 8, 8], 0.03);
        w.insert_filled("blk.0.ffn_down_exps.weight", &[8, 8, 8], 0.03);

        let mut kv = vec![0.0_f32; 4 * 16 * 16];
        let mut logits = vec![0.0_f32; 16];
        let mut ctx = ExecutionContext::new(&[2, 4], 16, &mut logits, &mut kv).unwrap();
        recipe
            .prefill_step(&mut ctx, &w)
            .expect("qwen35moe prefill");
        ctx.prepare_decode(6).unwrap();
        recipe.decode_step(&mut ctx, &w).expect("qwen35moe decode");
        assert_eq!(ctx.position(), 3);
        assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
    }
}
