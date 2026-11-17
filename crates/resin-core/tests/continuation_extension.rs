//! Decode continuations that cross cache blocks (`RunnerConfig::continuation_extension`): the engine leases the
//! next step's pages inside the running batch's lease instead of falling back to a synchronous step.

use resin_core::Phase;
use resin_core::engine::{
    EngineCore, Error, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig,
    Submission, TokenRows,
};
use resin_core::resources::kv::{KvGroupFamily, KvGroupLayout, KvLayout, KvRetention};
use std::num::NonZeroU32;

#[derive(Default)]
struct Model;

impl Runner<&'static str> for Model {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
        let rows = plan.token_indptr.len() - 1;
        OwnedTokenRows::new(vec![7; rows], (0..=rows as u32).collect())
    }
}

fn input(tokens: Vec<u32>, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(tokens, "opaque model metadata");
    input.max_output_tokens = Some(limit);
    input
}

fn engine(
    layout: &KvLayout,
    extension: Option<u32>,
    prefix_cache: bool,
) -> EngineCore<&'static str> {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(layout.clone()),
            cache_write_lookahead_tokens: 3,
            continuation_prefill: true,
            continuation_extension: extension.map(|tokens| NonZeroU32::new(tokens).unwrap()),
            ..Default::default()
        },
        4,
        4096,
        SchedulerConfig::default(),
        prefix_cache,
    )
    .unwrap();
    engine.enable_metrics().unwrap();
    engine
}

/// The exact active range of a decode row whose step leases `leased` tokens (what `PagedCachePlan.from_plan`
/// requires of every table).
fn active(group: &KvGroupLayout, leased: u32) -> (u32, u32) {
    let block = group.block_tokens().get();
    let total = leased.div_ceil(block);
    let count = if group.family() == KvGroupFamily::State {
        total.min(2)
    } else {
        match group.retention() {
            KvRetention::SlidingWindow { tokens } => {
                total.min(1 + (tokens.get() - 1).div_ceil(block))
            }
            KvRetention::FullContext => total,
        }
    };
    (total - count, count)
}

/// Every decode table is the row's exact active range and every append lands in its own row's block.
fn check(layout: &KvLayout, plan: &ExecutionPlan<&'static str>, lookahead: u32) {
    for (group, table) in layout.groups().iter().zip(&plan.cache_groups) {
        let block = group.block_tokens().get();
        for (row, &context) in plan.context_lengths.iter().enumerate() {
            let (first, count) = active(group, context + 1 + lookahead);
            let start = table.block_indptr[row];
            assert_eq!(
                (
                    table.block_table_start_indices[row],
                    table.block_indptr[row + 1] - start
                ),
                (first, count),
                "row {row} at {context}, group {:?}",
                group.retention()
            );
            let column = start + context / block - first;
            assert_eq!(
                table.append_slab_ids[row],
                table.block_slab_ids[column as usize]
            );
            assert_eq!(
                table.append_slotnums[row],
                table.block_slotnums[column as usize]
            );
            assert_eq!(table.append_block_offsets[row], context % block);
        }
    }
}

#[derive(Debug, Default)]
struct Run {
    decode_forwards: usize,
    continuations: usize,
}

/// Drive the engine as the overlapped runtime does: enqueue a decode step's successor whenever the engine retains a
/// continuation, harvest the previous step after that, and run everything else synchronously.
fn run(engine: &mut EngineCore<&'static str>, layout: &KvLayout) -> Run {
    let mut model = Model;
    let mut out = Run::default();
    let mut pending: Option<usize> = None; // rows of the enqueued step
    for _ in 0..100_000 {
        if let Some(rows) = pending.take() {
            let ids = vec![7; rows];
            let offsets: Vec<u32> = (0..=rows as u32).collect();
            let output = TokenRows::new(&ids, &offsets);
            if let Some(next) = engine.continuation_plan().unwrap() {
                check(layout, &next, 3);
                out.continuations += 1;
                pending = Some(next.sequence_ids.len());
                engine
                    .complete_continuing_step_with(&mut model, output, Ok)
                    .unwrap();
            } else {
                engine.complete_step_with(&mut model, output, Ok).unwrap();
            }
            continue;
        }
        if !engine.has_work() {
            return out;
        }
        let Some(plan) = engine.prepare_step(&mut model, None).unwrap() else {
            return out;
        };
        if plan.phase == Phase::Decode {
            check(layout, &plan, 3);
            if engine.retain_continuation(&plan) {
                pending = Some(plan.sequence_ids.len());
                continue;
            }
            out.decode_forwards += 1;
        }
        let output = model.forward(plan);
        let output = output
            .as_ref()
            .map(OwnedTokenRows::view)
            .map_err(Clone::clone);
        engine.complete_step_with(&mut model, output, Ok).unwrap();
    }
    panic!("the engine did not drain");
}

fn full(block: u32) -> KvGroupLayout {
    KvGroupLayout::new(block, 1, KvRetention::FullContext).unwrap()
}

fn window(block: u32, tokens: u32, read: u32) -> KvGroupLayout {
    KvGroupLayout::new(
        block,
        1,
        KvRetention::SlidingWindow {
            tokens: NonZeroU32::new(tokens).unwrap(),
        },
    )
    .unwrap()
    .with_read_window_tokens(read)
    .unwrap()
}

#[test]
fn continuations_cross_full_context_blocks_in_place() {
    // Positions 120..420 cross the blocks at 128, 256 and 384.
    let layout = KvLayout::new(16, vec![full(128)]).unwrap();
    let mut baseline = engine(&layout, None, false);
    let free = baseline.cache_free_parent_count().unwrap();
    baseline.submit(input(vec![1; 120], 300)).unwrap();
    let before = run(&mut baseline, &layout);
    let counts = baseline.metrics_snapshot().unwrap().continuations;
    assert_eq!(counts.extended, 0);
    assert!(counts.block >= 3, "{counts:?}");

    let mut extended = engine(&layout, Some(4096), false);
    extended.submit(input(vec![1; 120], 300)).unwrap();
    let after = run(&mut extended, &layout);
    let counts = extended.metrics_snapshot().unwrap().continuations;
    assert!(counts.extended >= 3, "{counts:?}");
    assert_eq!(counts.block, 0, "{counts:?}");
    assert!(
        after.decode_forwards + 3 <= before.decode_forwards,
        "{before:?} {after:?}"
    );
    assert!(after.continuations > before.continuations);
    assert_eq!(extended.cache_free_parent_count().unwrap(), free); // every page returned
    extended.close(&mut Model).unwrap();
}

#[test]
fn continuations_rotate_a_saturated_window_ring() {
    // A 256-token full-context group and a 128-token ring of 4 blocks (300-token window, 128 read): from 600 the
    // ring is full, so crossing 640 and 896 rotates it while 768 also grows the full-context group.
    let layout = KvLayout::new(32, vec![full(256), window(128, 300, 128)]).unwrap();
    for rows in [1, 3] {
        let mut engine = engine(&layout, Some(4096), false);
        let free = engine.cache_free_parent_count().unwrap();
        for row in 0..rows {
            engine
                .submit(input(vec![row + 1; 600 + 37 * row as usize], 320))
                .unwrap();
        }
        let out = run(&mut engine, &layout);
        let counts = engine.metrics_snapshot().unwrap().continuations;
        assert!(counts.extended >= 3, "{rows}: {counts:?} {out:?}");
        assert_eq!(counts.block, 0, "{rows}: {counts:?}");
        assert_eq!(engine.cache_free_parent_count().unwrap(), free);
        engine.close(&mut Model).unwrap();
    }
}

#[test]
fn shared_window_pages_are_copied_on_write_inside_the_lease() {
    // The second request restores the first one's published prefix, so its ring starts on shared pages; rotating
    // onto one replaces it inside the lease and leaves the cached prefix intact.
    let layout = KvLayout::new(64, vec![full(256), window(128, 300, 128)]).unwrap();
    let mut engine = engine(&layout, Some(4096), true);
    engine.submit(input((0..600).collect(), 40)).unwrap();
    run(&mut engine, &layout);
    let cached = engine.cache_metrics().unwrap();
    engine.submit(input((0..600).collect(), 320)).unwrap();
    let out = run(&mut engine, &layout);
    let snapshot = engine.metrics_snapshot().unwrap();
    assert!(snapshot.prefill_cache_tokens > 0, "no prefix hit");
    let counts = snapshot.continuations;
    assert!(counts.extended >= 3, "{counts:?} {out:?}");
    assert_eq!(counts.block, 0, "{counts:?}");
    let entries =
        |metrics: resin_core::resources::paged_kv::CacheMetrics| metrics.prefix.unwrap().entries;
    assert!(entries(engine.cache_metrics().unwrap()) >= entries(cached)); // the shared prefix survived
    engine.close(&mut Model).unwrap();
}

#[test]
fn chains_end_at_the_publication_interval_when_the_prefix_cache_is_on() {
    let layout = KvLayout::new(16, vec![full(128)]).unwrap();
    for (prefix_cache, publishes) in [(false, false), (true, true)] {
        let mut engine = engine(&layout, Some(256), prefix_cache);
        engine.submit(input(vec![1; 120], 300)).unwrap();
        run(&mut engine, &layout);
        let counts = engine.metrics_snapshot().unwrap().continuations;
        assert_eq!(counts.publish > 0, publishes, "{counts:?}");
        engine.close(&mut Model).unwrap();
    }
}

#[test]
fn a_recurrent_group_still_falls_back_at_its_blocks() {
    // 16-token KV blocks extend in place; a 64-token state block moves its two-page ring, which falls back.
    let layout = KvLayout::new(
        64,
        vec![
            full(16),
            KvGroupLayout::new_with_family(64, 1, KvRetention::FullContext, KvGroupFamily::State)
                .unwrap(),
        ],
    )
    .unwrap();
    let mut engine = engine(&layout, Some(4096), false);
    engine.submit(input(vec![1; 10], 200)).unwrap();
    run(&mut engine, &layout);
    let counts = engine.metrics_snapshot().unwrap().continuations;
    assert!(counts.extended > 0, "{counts:?}");
    assert!(counts.block > 0, "{counts:?}");
    engine.close(&mut Model).unwrap();
}
