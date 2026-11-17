//! A runner that turns its drafts off and on between steps (`EngineCore::set_max_speculative_rows`): each decode
//! step leases the write lookahead of the limit it runs under, and a request keeps the pages that an earlier step
//! leased past its position.

use resin_core::Phase;
use resin_core::engine::{
    EngineCore, Error, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig,
    Submission, TokenRows,
};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};

const LOOKAHEAD: u32 = 3;
const BLOCK: u32 = 16;

#[derive(Default)]
struct Model;

impl Runner<&'static str> for Model {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
        let rows = plan.token_indptr.len() - 1;
        OwnedTokenRows::new(vec![7; rows], (0..=rows as u32).collect())
    }
}

fn engine(layout: &KvLayout, rows: Option<u32>) -> EngineCore<&'static str> {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(layout.clone()),
            cache_write_lookahead_tokens: LOOKAHEAD,
            continuation_prefill: true,
            max_speculative_rows: rows,
            draft_tokens: LOOKAHEAD,
            ..Default::default()
        },
        4,
        4096,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    engine.set_step_reports(true);
    engine
}

fn input(tokens: usize, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(vec![1; tokens], "opaque model metadata");
    input.max_output_tokens = Some(limit);
    input
}

/// The decode row's blocks are exactly those its token and `lookahead` positions after it need.
fn check(plan: &ExecutionPlan<&'static str>, lookahead: u32) {
    let table = &plan.cache_groups[0];
    for (row, &context) in plan.context_lengths.iter().enumerate() {
        let blocks = (context + 1 + lookahead).div_ceil(BLOCK);
        assert_eq!(table.block_table_start_indices[row], 0);
        assert_eq!(
            table.block_indptr[row + 1] - table.block_indptr[row],
            blocks,
            "row {row} at {context} with lookahead {lookahead}"
        );
    }
}

/// One synchronous step; returns its phase.
fn step(engine: &mut EngineCore<&'static str>, lookahead: u32) -> Phase {
    let mut model = Model;
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    let phase = plan.phase;
    if phase == Phase::Decode {
        check(&plan, lookahead);
    }
    let output = model.forward(plan);
    let output = output
        .as_ref()
        .map(OwnedTokenRows::view)
        .map_err(Clone::clone);
    engine.complete_step_with(&mut model, output, Ok).unwrap();
    phase
}

#[test]
fn decode_steps_lease_by_the_limit_they_run_under() {
    let layout = KvLayout::new(
        16,
        vec![KvGroupLayout::new(BLOCK, 1, KvRetention::FullContext).unwrap()],
    )
    .unwrap();
    let mut engine = engine(&layout, Some(2));
    let free = engine.cache_free_parent_count().unwrap();
    assert_eq!(engine.max_speculative_rows(), Some(2));
    // a 30-token prompt: its first decode step leases 30 + 1 + 3 = 34 positions (three blocks)
    engine.submit(input(30, 60)).unwrap();
    while step(&mut engine, LOOKAHEAD) == Phase::Prefill {}
    let mut reports = Vec::new();
    for (limit, lookahead, steps) in [(Some(0), 0, 10), (Some(2), LOOKAHEAD, 10), (Some(0), 0, 10)]
    {
        reports.extend(engine.take_step_reports());
        engine.set_max_speculative_rows(limit).unwrap();
        assert_eq!(engine.max_speculative_rows(), limit);
        for _ in 0..steps {
            assert_eq!(step(&mut engine, lookahead), Phase::Decode);
        }
        let drafts: Vec<u32> = engine
            .take_step_reports()
            .iter()
            .map(|report| report.draft_tokens)
            .collect();
        assert_eq!(drafts, vec![lookahead; steps], "{limit:?}");
    }
    assert!(!reports.is_empty());
    while engine.has_work() {
        step(&mut engine, 0);
    }
    assert_eq!(engine.cache_free_parent_count().unwrap(), free); // every page returned
    engine.close(&mut Model).unwrap();
}

#[test]
fn the_limit_changes_only_between_steps() {
    let layout = KvLayout::new(
        16,
        vec![KvGroupLayout::new(BLOCK, 1, KvRetention::FullContext).unwrap()],
    )
    .unwrap();
    let mut engine = engine(&layout, Some(2));
    let mut model = Model;
    // a 2-token prompt: the next decode steps (4 tokens and 3 of lookahead each) stay in its first block
    engine.submit(input(2, 60)).unwrap();
    while step(&mut engine, LOOKAHEAD) == Phase::Prefill {}
    // a step in flight, then also its retained continuation, hold the old lease
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert!(engine.set_max_speculative_rows(Some(0)).is_err());
    assert!(engine.retain_continuation(&plan));
    assert!(engine.set_max_speculative_rows(Some(0)).is_err());
    assert_eq!(engine.max_speculative_rows(), Some(2));
    let ids = [7];
    let offsets = [0, 1];
    engine
        .complete_step_with(&mut model, TokenRows::new(&ids, &offsets), Ok)
        .unwrap();
    engine.set_max_speculative_rows(Some(0)).unwrap();
    assert_eq!(step(&mut engine, 0), Phase::Decode);
    engine.close(&mut Model).unwrap();
    assert!(engine.set_max_speculative_rows(Some(2)).is_err()); // closed
}
