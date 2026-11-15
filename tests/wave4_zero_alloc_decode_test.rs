//! Zero-allocation decode loop verification (`#171`, `#172`).
//!
//! Uses a dedicated process-wide `GlobalAlloc` wrapper gated strictly around
//! `EnginePhase::Decode` after engine warm-up so caller-thread and worker-thread heap
//! allocations inside the decode hot path are both measured cleanly.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_core::{
    ArchRecipe, CoordinationProfile, ExecutionContext, GgufFile, K3Error, ModelWeights, QuantType,
};
use jero_engine::affinity::{ClusterPhasePolicy, WorkerPool};
use jero_engine::engine::{SinglePromptEngine, StreamAction};
use jero_engine::kv::SingleSlotKvArena;

struct CountingAllocator;

static ALLOC_TEST_MUTEX: Mutex<()> = Mutex::new(());
static ALLOC_TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_COUNTING_ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: Delegates allocation/deallocation directly to `std::alloc::System` while maintaining
// lock-free atomic counters that neither allocate nor alias caller buffers.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ALLOC_TRACKING.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Delegates directly to `std::alloc::System` with caller-validated layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Pointer and layout originate from `std::alloc::System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ALLOC_TRACKING.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Delegates directly to `std::alloc::System` with caller-validated layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ALLOC_TRACKING.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: Pointer and layout originate from `std::alloc::System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Deterministic recipe exercising `ScratchWorkspace`, KV updates, logits writes, and
/// `WorkerPool::with_current` worker-thread fan-out without any heap allocation during decode.
#[derive(Debug)]
struct ScratchAndPoolDecodeRecipe;

impl ArchRecipe for ScratchAndPoolDecodeRecipe {
    fn architecture_id(&self) -> &'static str {
        "scratch-pool-zero-alloc"
    }

    fn supports_quant(&self, _quant: QuantType) -> bool {
        true
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_len = ctx.prompt_tokens().len();
        ctx.advance_position(prompt_len)?;
        let logits = ctx.logits_mut();
        logits.fill(-1.0);
        if logits.len() > 2 {
            logits[2] = 4.0;
        }
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode token missing".to_owned(),
            })?;
        let pos = ctx.position();
        {
            let (logits, kv_cache, scratch_opt) = ctx.buffers_and_scratch_mut();
            let scratch = scratch_opt.ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "ScratchWorkspace must be attached by SinglePromptEngine".to_owned(),
            })?;
            assert_eq!(scratch.used_bytes(), 0, "scratch must reset each step");

            // Perform bump allocations inside attached `ScratchWorkspace` + fan-out across workers.
            let scope = scratch.scope();
            let act_buf = scope.alloc_f32_zeroed(logits.len())?;
            for (idx, slot) in act_buf.iter_mut().enumerate() {
                let kv_bias = kv_cache
                    .get(pos % kv_cache.len().max(1))
                    .copied()
                    .unwrap_or(0.0);
                #[expect(clippy::cast_precision_loss, reason = "small bounded vocab index")]
                let base = -((idx + 1) as f32) * 0.01 + kv_bias * 0.0001;
                *slot = base;
            }
            let next_idx = ((token as usize) + 3) % act_buf.len();
            act_buf[next_idx] = 8.5;

            if WorkerPool::with_current(|pool| {
                pool.par_chunks_mut(logits, 8, |chunk_idx, dst| {
                    let offset = chunk_idx * 8;
                    dst.copy_from_slice(&act_buf[offset..offset + dst.len()]);
                });
            })
            .is_none()
            {
                logits.copy_from_slice(act_buf);
            }
            if !kv_cache.is_empty() {
                kv_cache[pos % kv_cache.len()] = 1.0;
            }
        }
        ctx.advance_position(1)?;
        Ok(())
    }
}

fn measure_decode_allocations(
    recipe: &dyn ArchRecipe,
    gguf: &GgufFile<'_>,
    pool: Option<&WorkerPool>,
) -> (usize, usize, Vec<u32>) {
    let mut engine = SinglePromptEngine::new();
    let mut kv_warm = SingleSlotKvArena::new(gguf.metadata(), 64).expect("kv_warm");
    // Warm up reusable logits buffer and ScratchWorkspace once before measuring decode steps.
    let _ = engine
        .evaluate_prompt_with_pool(recipe, gguf, &mut kv_warm, pool, &[1, 2], 32, 2, |_| {
            Ok(StreamAction::Continue)
        })
        .expect("warmup pass");

    let mut kv = SingleSlotKvArena::new(gguf.metadata(), 64).expect("kv");
    let mut step_seen = 0_usize;
    let summary = engine
        .evaluate_prompt_with_pool(recipe, gguf, &mut kv, pool, &[1, 2], 32, 17, |_| {
            if step_seen == 0 {
                // Enable right before `EnginePhase::Decode` begins across all 16 steps + teardown.
                ALLOC_COUNT.store(0, Ordering::Release);
                ALLOC_BYTES.store(0, Ordering::Release);
                ALLOC_TRACKING.store(true, Ordering::Release);
            }
            step_seen += 1;
            Ok(StreamAction::Continue)
        })
        .expect("measured decode pass");
    ALLOC_TRACKING.store(false, Ordering::Release);

    assert_eq!(summary.timing.decode_steps, 16);
    assert_eq!(summary.generated_tokens.len(), 17);
    (
        ALLOC_COUNT.load(Ordering::Acquire),
        ALLOC_BYTES.load(Ordering::Acquire),
        summary.generated_tokens,
    )
}

#[test]
fn test_phase_decode_zero_heap_allocations_inline_and_dedicated_pools() {
    let _guard = ALLOC_TEST_MUTEX.lock().expect("alloc test mutex");
    let bytes = create_synthetic_gguf_v3(
        "scratch-pool-zero-alloc",
        64,
        &[("output_norm.weight", QuantType::F32, &[16][..])],
    );
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");
    let recipe = ScratchAndPoolDecodeRecipe;

    let spin_pool = WorkerPool::for_phase_policy(
        ClusterPhasePolicy::X100Only,
        gguf.metadata(),
        8,
        None,
        Some(2),
        CoordinationProfile::Spin,
    )
    .expect("spin pool");
    let futex_pool = WorkerPool::for_phase_policy(
        ClusterPhasePolicy::X100Only,
        gguf.metadata(),
        8,
        None,
        Some(2),
        CoordinationProfile::Futex,
    )
    .expect("futex pool");

    // Pre-warm each DedicatedThreads worker once outside measurement so TLS/thread-init settles.
    for pool in [&spin_pool, &futex_pool] {
        pool.execute_scoped(&|_, _| {});
    }

    let (inline_cnt, inline_bytes, inline_tokens) =
        measure_decode_allocations(&recipe, &gguf, None);
    assert_eq!(
        (inline_cnt, inline_bytes),
        (0, 0),
        "inline EnginePhase::Decode allocated {inline_cnt} times ({inline_bytes} bytes)"
    );

    let (spin_cnt, spin_bytes, spin_tokens) =
        measure_decode_allocations(&recipe, &gguf, Some(&spin_pool));
    assert_eq!(
        (spin_cnt, spin_bytes),
        (0, 0),
        "Spin DedicatedThreads EnginePhase::Decode allocated {spin_cnt} times ({spin_bytes} bytes)"
    );
    assert_eq!(spin_tokens, inline_tokens);

    let (futex_cnt, futex_bytes, futex_tokens) =
        measure_decode_allocations(&recipe, &gguf, Some(&futex_pool));
    assert_eq!(
        (futex_cnt, futex_bytes),
        (0, 0),
        "Futex DedicatedThreads EnginePhase::Decode allocated {futex_cnt} times ({futex_bytes} bytes)"
    );
    assert_eq!(futex_tokens, inline_tokens);

    #[cfg(feature = "jero-arch-synthetic")]
    {
        use jero_arch_synthetic::SyntheticRecipe;

        let synth_bytes = create_synthetic_gguf_v3(
            "synthetic",
            64,
            &[
                ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
                ("output_norm.weight", QuantType::F32, &[16][..]),
            ],
        );
        let synth_gguf = GgufFile::parse(&synth_bytes).expect("parse synthetic gguf");
        let synth_recipe = SyntheticRecipe::new();
        let (synth_cnt_inline, synth_bytes_inline, synth_tok_inline) =
            measure_decode_allocations(&synth_recipe, &synth_gguf, None);
        assert_eq!((synth_cnt_inline, synth_bytes_inline), (0, 0));
        let (synth_cnt_spin, synth_bytes_spin, synth_tok_spin) =
            measure_decode_allocations(&synth_recipe, &synth_gguf, Some(&spin_pool));
        assert_eq!((synth_cnt_spin, synth_bytes_spin), (0, 0));
        assert_eq!(synth_tok_spin, synth_tok_inline);
    }
}
