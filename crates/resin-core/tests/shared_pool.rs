//! Several Engines (one per loaded model) that allocate cache slabs from one
//! shared pool: global LRU eviction of cached prefixes, live requests that no
//! other Engine evicts, floors, and waiting without a deadlock.

use resin_core::Phase;
use resin_core::engine::{
    EngineCore, Error, ErrorKind, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig,
    SchedulerConfig, Submission,
};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use resin_core::resources::paged_kv::SharedCachePool;
use std::num::NonZeroU32;

/// Each step returns the last input token + 1 for every row.
#[derive(Default)]
struct Model {
    steps: usize,
    prefills: usize,
}

impl Runner<&'static str> for Model {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
        self.steps += 1;
        self.prefills += usize::from(plan.phase == Phase::Prefill);
        let mut ids = Vec::new();
        let mut offsets = vec![0];
        for range in plan.token_indptr.windows(2) {
            ids.push(plan.token_ids[range[1] as usize - 1] + 1);
            offsets.push(ids.len() as u32);
        }
        OwnedTokenRows::new(ids, offsets)
    }
}

/// Slabs of one 2-token block each.
fn layout(capacity: u32) -> KvLayout {
    KvLayout::new(
        capacity,
        vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
    )
    .unwrap()
}

fn engine(pool: &SharedCachePool, floor: u32, prefix_cache: bool) -> EngineCore<&'static str> {
    EngineCore::in_pool(
        RunnerConfig {
            cache_layout: Some(layout(pool.parent_capacity())),
            continuation_prefill: true,
            ..Default::default()
        },
        4,
        64,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(64).unwrap(),
            ..Default::default()
        },
        prefix_cache,
        Some((pool, floor)),
    )
    .unwrap()
}

fn input(tokens: Vec<u32>, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(tokens, "model metadata");
    input.max_output_tokens = Some(limit);
    input
}

fn run(engine: &mut EngineCore<&'static str>, model: &mut Model) {
    while engine.has_work() {
        engine.step(model).unwrap();
    }
}

fn prefix_entries(engine: &EngineCore<&'static str>) -> usize {
    engine.prefix_cache_stats().unwrap().0
}

#[test]
fn the_oldest_prefix_of_any_model_goes_first() {
    let pool = SharedCachePool::new(6).unwrap();
    let (mut a, mut b) = (engine(&pool, 0, true), engine(&pool, 0, true));
    let mut model = Model::default();
    let first = a.submit(input(vec![1, 2, 3, 4], 1)).unwrap();
    run(&mut a, &mut model);
    first.take_result().unwrap();
    let second = b.submit(input(vec![9, 8, 7, 6], 1)).unwrap();
    run(&mut b, &mut model);
    second.take_result().unwrap();
    assert!(prefix_entries(&a) > 0 && prefix_entries(&b) > 0);
    assert_eq!(pool.free_parent_count(), 2, "both prefixes stay cached");

    // B needs 4 slabs: the 2 free ones and A's prefix, the oldest of the pool.
    let large = b.submit(input((20..28).collect(), 1)).unwrap();
    run(&mut b, &mut model);
    large.take_result().unwrap();
    assert_eq!(prefix_entries(&a), 0, "A's older prefix was evicted");
    assert!(prefix_entries(&b) > 0, "B's own newer prefix stays");
}

#[test]
fn one_model_never_evicts_the_live_requests_of_another() {
    let pool = SharedCachePool::new(4).unwrap();
    let (mut a, mut b) = (engine(&pool, 0, false), engine(&pool, 0, false));
    let (mut model_a, mut model_b) = (Model::default(), Model::default());
    // A's request grows to 4 slabs (8 tokens) while it decodes.
    let long = a.submit(input(vec![1, 2, 3, 4, 5], 3)).unwrap();
    a.step(&mut model_a).unwrap();
    let waiting = b.submit(input(vec![9, 8, 7], 1)).unwrap();
    // B's prompt needs 2 slabs; A holds 3 live ones and grows to 4: B waits.
    while a.has_work() {
        assert!(b.step(&mut model_b).unwrap().is_empty());
        a.step(&mut model_a).unwrap();
    }
    assert_eq!(model_b.steps, 0, "B ran nothing while A held the slabs");
    assert_eq!(long.take_result().unwrap().len(), 3);
    run(&mut b, &mut model_b);
    assert_eq!(waiting.take_result().unwrap(), vec![8]);
}

#[test]
fn a_floor_keeps_free_slabs_for_its_model() {
    let mut model = Model::default();
    // A may plan for 6 - 2 = 4 slabs: a 10-token prompt (5 slabs) cannot fit.
    let pool = SharedCachePool::new(6).unwrap();
    let (mut a, _b) = (engine(&pool, 0, false), engine(&pool, 2, false));
    let _too_large = a.submit(input((1..11).collect(), 1)).unwrap();
    let error = loop {
        match a.step(&mut model) {
            Ok(_) if a.has_work() => continue,
            Ok(_) => panic!("the request ran past B's floor"),
            Err(error) => break error,
        }
    };
    assert_eq!(error.kind, ErrorKind::Memory);

    // B still gets its 2 slabs while A's request holds the other 4.
    let pool = SharedCachePool::new(6).unwrap();
    let (mut a, mut b) = (engine(&pool, 0, false), engine(&pool, 2, false));
    // 7 prompt tokens and one decode step stay in 4 slabs
    let held = a.submit(input((1..8).collect(), 2)).unwrap();
    a.step(&mut model).unwrap();
    assert!(a.has_work(), "A's request still decodes");
    assert_eq!(pool.bound_slabs(), vec![4, 0]);
    let small = b.submit(input(vec![5, 6, 7], 1)).unwrap();
    run(&mut b, &mut model);
    assert_eq!(small.take_result().unwrap(), vec![8]);
    assert!(a.has_work(), "B ran while A's request held its slabs");
    run(&mut a, &mut model);
    assert_eq!(held.take_result().unwrap().len(), 2);
}

#[test]
fn a_model_waits_with_its_request_while_another_model_runs() {
    let pool = SharedCachePool::new(5).unwrap();
    let (mut a, mut b) = (engine(&pool, 0, false), engine(&pool, 0, false));
    let (mut model_a, mut model_b) = (Model::default(), Model::default());
    let first = a.submit(input(vec![1, 2, 3, 4], 3)).unwrap();
    let second = b.submit(input(vec![5, 6, 7, 8], 2)).unwrap();
    a.step(&mut model_a).unwrap(); // A: 2 slabs
    b.step(&mut model_b).unwrap(); // B: 2 slabs
    a.step(&mut model_a).unwrap(); // A's decode takes the last slab
    // B's decode needs a third slab; A still runs, so B keeps its request and waits.
    assert!(b.step(&mut model_b).unwrap().is_empty());
    run(&mut a, &mut model_a);
    run(&mut b, &mut model_b);
    assert_eq!(first.take_result().unwrap().len(), 3);
    assert_eq!(second.take_result().unwrap().len(), 2);
    assert_eq!(model_b.prefills, 1, "B's request was not given back");
}

#[test]
fn two_models_that_wait_for_each_other_do_not_deadlock() {
    let pool = SharedCachePool::new(4).unwrap();
    let (mut a, mut b) = (engine(&pool, 0, false), engine(&pool, 0, false));
    let (mut model_a, mut model_b) = (Model::default(), Model::default());
    // Each prompt fills 2 slabs; each decode step then needs a third.
    let first = a.submit(input(vec![1, 2, 3, 4], 4)).unwrap();
    let second = b.submit(input(vec![5, 6, 7, 8], 4)).unwrap();
    a.step(&mut model_a).unwrap();
    b.step(&mut model_b).unwrap();
    assert_eq!(pool.free_parent_count(), 0);
    // A waits for B; B sees A waiting and gives its slabs back.
    for _ in 0..64 {
        if !a.has_work() && !b.has_work() {
            break;
        }
        if a.has_work() {
            a.step(&mut model_a).unwrap();
        }
        if b.has_work() {
            b.step(&mut model_b).unwrap();
        }
    }
    assert!(!a.has_work() && !b.has_work(), "both requests finished");
    assert_eq!(first.take_result().unwrap().len(), 4);
    assert_eq!(second.take_result().unwrap().len(), 4);
}

#[test]
fn an_engine_must_declare_the_pool_capacity() {
    let pool = SharedCachePool::new(6).unwrap();
    let error = EngineCore::<&'static str>::in_pool(
        RunnerConfig {
            cache_layout: Some(layout(5)),
            ..Default::default()
        },
        4,
        64,
        SchedulerConfig::default(),
        false,
        Some((&pool, 0)),
    )
    .err()
    .expect("a layout of 5 slabs cannot join a pool of 6");
    assert!(error.to_string().contains("declares 5 slabs"), "{error}");
}

#[test]
fn every_engine_reports_its_own_slabs() {
    let pool = SharedCachePool::new(8).unwrap();
    let (mut a, mut b) = (engine(&pool, 0, true), engine(&pool, 0, true));
    let mut model = Model::default();
    let one = a.submit(input(vec![1, 2, 3, 4], 1)).unwrap();
    run(&mut a, &mut model);
    one.take_result().unwrap();
    let two = b.submit(input(vec![1, 2, 3, 4, 5, 6], 1)).unwrap();
    run(&mut b, &mut model);
    two.take_result().unwrap();
    assert_eq!(pool.bound_slabs(), vec![2, 3], "each model's cached prefix");
    assert_eq!(
        a.cache_free_parent_count(),
        Some(3),
        "free slabs are the pool's"
    );
}
