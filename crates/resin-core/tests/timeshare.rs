//! Prefill/decode time sharing through the Engine, with a manual clock.

use resin_core::engine::{
    EngineCore, Error, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig,
    Submission, TokenRows,
};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use resin_core::{DecodeShare, Phase};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
struct ManualClock(Arc<Mutex<Instant>>);

impl ManualClock {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Instant::now())))
    }

    fn advance(&self, seconds: f64) {
        *self.0.lock().unwrap() += Duration::from_secs_f64(seconds);
    }

    fn install(&self, engine: &mut EngineCore<&'static str>) {
        let time = Arc::clone(&self.0);
        engine.set_clock(Arc::new(move || *time.lock().unwrap()));
    }
}

/// A runner whose prefill and decode steps take fixed simulated times.
struct TimedModel {
    clock: ManualClock,
    prefill_seconds: f64,
    decode_seconds: f64,
    phases: Vec<Phase>,
}

impl Runner<&'static str> for TimedModel {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
        self.clock.advance(match plan.phase {
            Phase::Prefill => self.prefill_seconds,
            Phase::Decode => self.decode_seconds,
        });
        self.phases.push(plan.phase);
        let mut ids = Vec::new();
        let mut offsets = vec![0];
        for range in plan.token_indptr.windows(2) {
            ids.push(plan.token_ids[range[1] as usize - 1] + 1);
            offsets.push(u32::try_from(ids.len()).unwrap());
        }
        OwnedTokenRows::new(ids, offsets)
    }
}

fn engine(share: Option<DecodeShare>, block_tokens: u32) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    8192 / block_tokens,
                    vec![KvGroupLayout::new(block_tokens, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            ..Default::default()
        },
        4,
        2048,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(16).unwrap(),
            decode_share: share,
            ..Default::default()
        },
        false,
    )
    .unwrap()
}

fn input(tokens: Vec<u32>, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(tokens, "payload");
    input.max_output_tokens = Some(limit);
    input
}

fn compact(phases: &[Phase]) -> String {
    phases
        .iter()
        .map(|phase| match phase {
            Phase::Prefill => 'P',
            Phase::Decode => 'D',
        })
        .collect()
}

/// One decoding stream, then a 64-token prompt in 16-token chunks.
fn interleaving(share: Option<DecodeShare>) -> String {
    let clock = ManualClock::new();
    let mut engine = engine(share, 16);
    clock.install(&mut engine);
    let mut model = TimedModel {
        clock: clock.clone(),
        prefill_seconds: 1.0,
        decode_seconds: 0.25,
        phases: Vec::new(),
    };
    engine.submit(input(vec![1, 2, 3, 4], 1000)).unwrap();
    engine.step(&mut model).unwrap();
    engine.step(&mut model).unwrap();
    model.phases.clear();
    engine.submit(input((0..64).collect(), 1000)).unwrap();
    for _ in 0..20 {
        engine.step(&mut model).unwrap();
    }
    compact(&model.phases)
}

#[test]
fn prefill_first_without_a_decode_share() {
    assert_eq!(interleaving(None), "PPPPDDDDDDDDDDDDDDDD");
}

#[test]
fn decode_receives_its_share_between_prefill_chunks() {
    // At 50%, each 1 s prefill chunk buys four 0.25 s decode steps.
    assert_eq!(
        interleaving(DecodeShare::new(0.5, 0.5).ok()),
        "PDDDDPDDDDPDDDDPDDDD"
    );
}

#[test]
fn zero_share_keeps_prefill_first() {
    assert_eq!(
        interleaving(DecodeShare::new(0.0, 0.0).ok()),
        "PPPPDDDDDDDDDDDDDDDD"
    );
}

#[test]
fn target_share_follows_the_decoding_rows() {
    // One of four slots decodes: 0.2 + (0.6 - 0.2) / 4 = 0.3, so a 1 s chunk
    // buys 0.43 s of decode, which is two 0.25 s steps.
    assert_eq!(
        interleaving(DecodeShare::new(0.2, 0.6).ok()),
        "PDDPDDPDDPDDDDDDDDDD"
    );
}

#[test]
fn decode_continues_overlapped_while_owed_and_drains_when_prefill_is_due() {
    let clock = ManualClock::new();
    let mut engine = engine(DecodeShare::new(0.5, 0.5).ok(), 128);
    clock.install(&mut engine);
    let mut model = TimedModel {
        clock: clock.clone(),
        prefill_seconds: 1.0,
        decode_seconds: 0.3,
        phases: Vec::new(),
    };
    engine.submit(input(vec![1, 2, 3, 4], 1000)).unwrap();
    engine.step(&mut model).unwrap();
    engine.submit(input((0..64).collect(), 1000)).unwrap();
    // The new prompt goes first. Its chunk owes decode 1 s.
    engine.step(&mut model).unwrap();
    assert_eq!(compact(&model.phases), "PP");

    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.phase, Phase::Decode);
    assert!(
        engine.retain_continuation(&plan),
        "a waiting prompt keeps overlap"
    );
    let mut continued = 0;
    let mut token = 100;
    while let Some(next) = engine.continuation_plan().unwrap() {
        assert_eq!(next.phase, Phase::Decode);
        assert!(engine.has_continuation());
        clock.advance(0.3);
        engine
            .complete_continuing_step_with(&mut model, TokenRows::new(&[token], &[0, 1]), Ok)
            .unwrap();
        token += 1;
        continued += 1;
        assert!(continued < 10, "decode must stop for the prompt");
    }
    // 1 s owed: four 0.3 s steps are collected before prefill is due again.
    assert_eq!(continued, 4);
    assert!(!engine.has_continuation());
    clock.advance(0.3);
    engine
        .complete_step_with(&mut model, TokenRows::new(&[token], &[0, 1]), Ok)
        .unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.phase, Phase::Prefill);
}

#[test]
fn a_waiting_prompt_stops_overlap_without_a_decode_share() {
    let clock = ManualClock::new();
    let mut engine = engine(None, 128);
    clock.install(&mut engine);
    let mut model = TimedModel {
        clock: clock.clone(),
        prefill_seconds: 1.0,
        decode_seconds: 0.3,
        phases: Vec::new(),
    };
    engine.submit(input(vec![1, 2, 3, 4], 1000)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert!(engine.retain_continuation(&plan));
    engine.submit(input((0..64).collect(), 1000)).unwrap();
    assert!(!engine.has_continuation());
    assert!(engine.continuation_plan().unwrap().is_none());
}

#[test]
fn busy_seconds_and_target_are_reported() {
    let clock = ManualClock::new();
    let mut engine = engine(DecodeShare::new(0.5, 0.9).ok(), 16);
    engine.enable_metrics().unwrap();
    clock.install(&mut engine);
    let mut model = TimedModel {
        clock: clock.clone(),
        prefill_seconds: 1.0,
        decode_seconds: 0.25,
        phases: Vec::new(),
    };
    engine.submit(input(vec![1, 2, 3, 4], 1000)).unwrap();
    engine.step(&mut model).unwrap();
    engine.step(&mut model).unwrap();
    let metrics = engine.metrics_snapshot().unwrap();
    assert!((metrics.prefill_busy_seconds - 1.0).abs() < 1e-9);
    assert!((metrics.decode_busy_seconds - 0.25).abs() < 1e-9);
    // One of four slots decodes.
    assert!((metrics.decode_share_target.unwrap() - 0.6).abs() < 1e-9);
}
