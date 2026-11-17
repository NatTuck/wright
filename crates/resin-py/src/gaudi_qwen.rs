//! The native step executor of the Qwen3.5/3.8 models on Gaudi (cargo feature `gaudi`).
//!
//! Python builds the model at startup: it loads the weights, compiles the recipes and builds one step program
//! (`resin._resin.GaudiProgram`) per step mode, whose table bindings point into one device table image, and
//! with a drafter one block-step program per (rows, verify width) over a block-step image. Each step then runs
//! here without Python: the step's tables go into pinned host copies of the images (`resin_gaudi::qwen`,
//! `resin_gaudi::qwen_draft`), one copy per image uploads them, the programs make the launches, and the
//! outputs come back through pinned staging buffers behind events. The runner object hands the Engine this
//! executor's `forward` and `enqueue`, so the Engine calls into Rust directly.
//!
//! Steps: prefill chunks (with their prompt logprobs, one score launch at a time) and plain decode steps (with a
//! drafter, each also writes the drafter's cache, and a prompt's last chunk drafts after its token; with the MTP
//! layer, each writes the MTP layer's keys, and a prompt's last chunk takes its first draft), and verify steps.
//! A DFlash 2 or DSpine drafter accepts on the device (`spec_fix`): the verify pass, the commit and the next
//! block step are queued together, and the host checks the device's acceptance when the next step needs the
//! positions. With the MTP layer the host accepts when the next step needs the positions (`QwenVerification::
//! finish_of`), and then queues the commit and the draft chain from an image of their own.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use numpy::{PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use resin_gaudi::qwen_draft::{self, BlockLayout, DraftRow, FixLayout, InjectLayout, StateRow};
use resin_gaudi::{qwen, qwen_mtp, sampling, speculative, step};

use crate::gaudi::{GaudiGlue, GaudiProgram, QwenFill, item};
use crate::gaudi_exec::{
    Images, Pinned, ScoreRequest, Scored, Timings, post_accept, prompt_columns, score_columns,
    score_requests, since,
};
use crate::gaudi_ranks::{
    GaudiRanks, OP_CLOSE, OP_ENQUEUE, OP_FORWARD, OP_RELEASE, Ranked, Replay,
};
use crate::output::ForwardOutput;
use crate::{BatchPhase, BatchPlan};

/// Registers of the step programs, in the order the Python builder uses them.
pub(crate) const R_IMAGE: usize = 0; // which host copy of the step image the upload reads (0 or 1)
pub(crate) const R_BYTES: usize = 1; // bytes of the step image to upload
pub(crate) const R_LAUNCHES: usize = 2; // slot launches of each full-attention layer
pub(crate) const R_TOK: usize = 3; // the device address of the columns' tokens
pub(crate) const R_STAGE: usize = 4; // which staging set the outputs go to (0 or 1)
pub(crate) const R_ROWS4: usize = 5; // 4 bytes a column: tokens and logprobs to copy out
pub(crate) const R_TOP: usize = 6; // TOP_MAX x 4 bytes a column: the top lists to copy out
pub(crate) const R_BIMAGE: usize = 7; // which host copy of the block-step image the upload reads
pub(crate) const R_BBYTES: usize = 8; // bytes of the block-step image to upload
pub(crate) const R_BLAUNCHES: usize = 9; // slot launches of each drafter layer
pub(crate) const R_FIXTIN: usize = 10; // the device address of spec_fix's input tokens
pub(crate) const R_NEXT: usize = 11; // bytes of the block step's tokens to copy out
pub(crate) const R_ACC: usize = 12; // bytes of spec_fix's accepted counts to copy out
pub(crate) const R_MPREV: usize = 13; // the device address of the MTP layer's hidden before a prefill chunk
pub(crate) const R_MLAUNCHES: usize = 14; // slot launches of the step's MTP pass with attention
pub(crate) const R_CIMAGE: usize = 15; // which host copy of the chain image the upload reads
pub(crate) const R_CBYTES: usize = 16; // bytes of the chain image to upload
pub(crate) const R_ROWS: usize = 17; // the chain's rows (its drafts into the next step's tokens)
pub(crate) const R_LINK: usize = 18; // slot launches of the chain's first link (link j: R_LINK + j - 1)
/// links of an MTP draft chain (K - 1): up to 8 drafts a step
pub(crate) const MAX_LINKS: usize = 7;
// wide slot launches of each full-attention layer (before the R_LAUNCHES others)
pub(crate) const R_WLAUNCHES: usize = R_LINK + MAX_LINKS;
pub(crate) const REGISTERS: usize = R_WLAUNCHES + 1;
const SECTIONS: [&str; 16] = [
    "target", "inject", "fix", "out", "scores", "record", "commit", "next", "record1", "mkeys",
    "mhidden", "mdraft", "mfull", "mout", "mcommit", "mchain",
];
const S_TARGET: usize = 0;
const S_INJECT: usize = 1;
const S_FIX: usize = 2;
const S_OUT: usize = 3;
const S_SCORES: usize = 4;
const S_RECORD: usize = 5;
const S_COMMIT: usize = 6;
const S_NEXT: usize = 7;
const S_RECORD1: usize = 8;
const S_MKEYS: usize = 9;
const S_MHIDDEN: usize = 10;
const S_MDRAFT: usize = 11;
const S_MFULL: usize = 12;
const S_MOUT: usize = 13;
const S_MCOMMIT: usize = 14;
const S_MCHAIN: usize = 15;
const BLOCK_SECTIONS: [&str; 2] = ["upload", "run"];
const TOP: usize = sampling::TOP_MAX;
const PARAMS: usize = sampling::PARAMS;
/// the widest decode bucket of a speculative step: wider steps decode plainly
const SPEC_ROWS: usize = 32;
/// prompt positions a score launch scores, and the (position, token) logprobs it gathers (`qwen3_5.layers`)
const SCORE_ROWS: usize = 128;
const SCORE_PAIRS: usize = 1024;
/// the top-list lengths of the score launches (`qwen3_5.model.SCORE_TOPS`)
const SCORE_TOPS: [usize; 3] = [0, 64, 1024];
/// the decode buckets of the draft policy (`qwen3_5.model.DECODE_BUCKETS`)
const POLICY_BUCKETS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// The executor's table images: the step modes', a drafter's block steps', the MTP layer's chains'.
#[derive(Clone, Copy)]
enum Image {
    Step,
    Block,
    Chain,
}

struct NativeMode {
    mode: qwen::Mode,
    layout: qwen::Layout,
    program: Arc<step::Program>,
    sections: [Option<usize>; SECTIONS.len()],
    inject: Option<InjectLayout>,
    fix: Option<FixLayout>,
    mtp: MtpTables,
    /// the shape of the MTP layer's passes of this mode: the mode with the MTP layer's room for slot launches (a
    /// replicated MTP layer of a tensor-parallel model has more KV groups than the rank's target layers)
    mtp_mode: qwen::Mode,
}

/// The MTP layer's tables in a mode's image: a prefill chunk's or a plain decode step's keys pass, a prefill
/// chunk's hidden store and the prompt's first draft (a pass of the decode mode of one row), a verify step's full
/// pass.
#[derive(Clone, Copy, Default)]
struct MtpTables {
    keys: Option<qwen_mtp::PassLayout>,
    hidden: Option<usize>,
    draft: Option<qwen_mtp::PassLayout>,
    full: Option<qwen_mtp::PassLayout>,
}

impl NativeMode {
    fn section(&self, k: usize) -> PyResult<(Arc<step::Program>, usize)> {
        let section = self.sections[k].ok_or_else(|| {
            PyValueError::new_err(format!(
                "mode {}'s program has no section {}",
                self.mode.name, SECTIONS[k]
            ))
        })?;
        Ok((Arc::clone(&self.program), section))
    }
}

/// The rows of a step: what `qwen::Rows`, `StateRow`, `DraftRow` and `qwen_mtp::Pass` borrow.
#[derive(Clone, Default)]
struct RowTables {
    pos0: Vec<i64>,
    state_src: Vec<i64>,
    state_dst: Vec<i64>,
    indptr: Vec<i64>,
    blocks: Vec<i64>,
    states: Vec<Vec<i64>>,
    state_first: Vec<i64>,
    drafts: Vec<Vec<i64>>,
    draft_first: Vec<i64>,
    /// with the MTP layer: each row's MTP KV block rows, `mtp_indptr` delimiting them
    mtp_indptr: Vec<i64>,
    mtp_blocks: Vec<i64>,
}

impl RowTables {
    fn rows(&self) -> qwen::Rows<'_> {
        qwen::Rows {
            pos0: &self.pos0,
            state_src: &self.state_src,
            state_dst: &self.state_dst,
            blocks_indptr: &self.indptr,
            blocks: &self.blocks,
        }
    }

    /// row `r` alone, at first position `pos0`, reading its state from `src`
    fn row(&self, r: usize, pos0: i64, src: i64) -> RowTables {
        let (a, b) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
        let mtp_blocks = match self.mtp_indptr.get(r + 1) {
            Some(&end) => self.mtp_blocks[self.mtp_indptr[r] as usize..end as usize].to_vec(),
            None => Vec::new(),
        };
        RowTables {
            mtp_indptr: if self.mtp_indptr.is_empty() {
                Vec::new()
            } else {
                vec![0, mtp_blocks.len() as i64]
            },
            mtp_blocks,
            pos0: vec![pos0],
            state_src: vec![src],
            state_dst: vec![self.state_dst[r]],
            indptr: vec![0, (b - a) as i64],
            blocks: self.blocks[a..b].to_vec(),
            states: vec![self.states[r].clone()],
            state_first: vec![self.state_first[r]],
            drafts: self.drafts.get(r).cloned().into_iter().collect(),
            draft_first: self.draft_first.get(r).copied().into_iter().collect(),
        }
    }

    fn state_row(&self, r: usize) -> StateRow<'_> {
        StateRow {
            pos0: self.pos0[r],
            src: self.state_src[r],
            blocks: &self.states[r],
            first: self.state_first[r],
        }
    }

    fn draft_row(&self, r: usize, position: i64) -> DraftRow<'_> {
        DraftRow {
            position,
            blocks: &self.drafts[r],
            first: self.draft_first[r],
        }
    }
}

/// (real tokens, bucket) of each prefill chunk (`qwen3_5.model.prefill_chunks`).
fn prefill_chunks(mut n: usize, buckets: &[usize]) -> Vec<(usize, usize)> {
    let largest = *buckets.last().expect("prefill buckets");
    let mut chunks = Vec::new();
    while n > 0 {
        let mut real = n.min(largest);
        if real < largest {
            let bucket = buckets
                .iter()
                .copied()
                .filter(|&b| b >= real)
                .min()
                .expect("a bucket");
            if bucket as f64 > 1.25 * real as f64 {
                real = buckets
                    .iter()
                    .copied()
                    .filter(|&b| b <= real)
                    .max()
                    .unwrap_or(real);
            }
        }
        let bucket = buckets
            .iter()
            .copied()
            .filter(|&b| b >= real)
            .min()
            .expect("a bucket");
        chunks.push((real, bucket));
        n -= real;
    }
    chunks
}

// ------------------------------------------------------------------ the draft policy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolicyMode {
    Auto,
    Always,
    Never,
}

/// Whether a decode step verifies drafts (`qwen3_5.runner.DraftPolicy`): the measured tokens per ms of the
/// bucket's verify and plain steps, with the step-time fits until measured, a margin to stop drafting, and
/// probes after runs of plain steps.
struct Policy {
    k: usize,
    /// the MTP layer's verify steps (their step-time fit), else a drafter's
    mtp: bool,
    mode: PolicyMode,
    /// (verify, bucket) -> [steps, average tokens, average ms]
    stats: HashMap<(bool, usize), [f64; 3]>,
    drafting: HashMap<usize, bool>,
    plain_run: HashMap<usize, usize>,
    probe_after: HashMap<usize, usize>,
    probe: usize,
}

impl Policy {
    const BETA: f64 = 0.05;
    const SPIKE: f64 = 3.0;
    const MARGIN: f64 = 1.1;
    const PROBE_FIRST: usize = 32;
    const PROBE_MAX: usize = 512;
    const PROBE_STEPS: usize = 8;

    fn new(k: usize, mode: &str, mtp: bool) -> PyResult<Self> {
        let mode = match mode {
            "auto" => PolicyMode::Auto,
            "always" => PolicyMode::Always,
            "never" => PolicyMode::Never,
            other => {
                return Err(PyValueError::new_err(format!(
                    "draft policy {other:?}: one of auto, always, never"
                )));
            }
        };
        Ok(Self {
            k,
            mtp,
            mode,
            stats: HashMap::new(),
            drafting: HashMap::new(),
            plain_run: HashMap::new(),
            probe_after: HashMap::new(),
            probe: 0,
        })
    }

    fn fit(&self, bucket: usize, width: usize) -> f64 {
        let ms = if self.mtp {
            qwen_draft::mtp_step_ms(width, bucket)
        } else {
            qwen_draft::step_ms(width, bucket)
        };
        bucket as f64 * (1.0 + 0.6 * self.k as f64).min(3.0).min(width as f64) / ms
    }

    fn rate(&self, verify: bool, bucket: usize, widths: &[usize]) -> f64 {
        if let Some(stats) = self.stats.get(&(verify, bucket)) {
            return stats[1] / stats[2];
        }
        if !verify {
            return bucket as f64 / qwen_draft::plain_ms(bucket);
        }
        widths
            .iter()
            .map(|&w| self.fit(bucket, w))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn speculate(&mut self, rows: usize, widths: &[usize]) -> bool {
        if self.k == 0 || rows > SPEC_ROWS || self.mode == PolicyMode::Never {
            return false;
        }
        if self.mode == PolicyMode::Always {
            return true;
        }
        if self.probe > 0 {
            self.probe -= 1;
            return true;
        }
        let b = POLICY_BUCKETS
            .iter()
            .copied()
            .find(|&x| x >= rows)
            .unwrap_or(64);
        let spec = self.rate(true, b, widths);
        let plain = self.rate(false, b, widths);
        let drafting = *self.drafting.get(&b).unwrap_or(&true);
        let decide = if drafting {
            spec * Self::MARGIN >= plain
        } else {
            spec > plain
        };
        self.drafting.insert(b, decide);
        if decide {
            if !drafting {
                self.probe_after.insert(b, Self::PROBE_FIRST);
            }
            self.plain_run.insert(b, 0);
            return true;
        }
        let run = self.plain_run.get(&b).copied().unwrap_or(0) + 1;
        let after = self
            .probe_after
            .get(&b)
            .copied()
            .unwrap_or(Self::PROBE_FIRST);
        if run >= after {
            self.plain_run.insert(b, 0);
            self.probe_after.insert(b, (2 * after).min(Self::PROBE_MAX));
            self.probe = Self::PROBE_STEPS - 1;
            self.stats.remove(&(true, b));
            return true;
        }
        self.plain_run.insert(b, run);
        false
    }

    fn record(&mut self, verify: bool, bucket: usize, tokens: f64, ms: f64) {
        if ms <= 0.0 {
            return;
        }
        let stats = self.stats.entry((verify, bucket)).or_insert([0.0; 3]);
        if stats[0] > 0.0 && ms > Self::SPIKE * stats[2] {
            return;
        }
        stats[0] += 1.0;
        let w = Self::BETA.max(1.0 / stats[0]);
        stats[1] += w * (tokens - stats[1]);
        stats[2] += w * (ms - stats[2]);
    }
}

/// The native executor's draft policy (`Policy`) on its own, for the CPU checks of its decisions: `k` drafts a
/// step, a drafter's (`drafter`) or the MTP layer's step-time fits, and its `mode`. A step is keyed ("plain",
/// bucket) or ("verify", bucket).
#[pyclass(module = "resin._resin")]
pub(crate) struct QwenDraftPolicy {
    policy: Policy,
}

fn policy_key(key: &(String, usize)) -> PyResult<(bool, usize)> {
    match key.0.as_str() {
        "plain" => Ok((false, key.1)),
        "verify" => Ok((true, key.1)),
        other => Err(PyValueError::new_err(format!(
            "a step kind {other:?}: plain or verify"
        ))),
    }
}

#[pymethods]
impl QwenDraftPolicy {
    #[classattr]
    const BETA: f64 = Policy::BETA;
    #[classattr]
    const SPIKE: f64 = Policy::SPIKE;
    #[classattr]
    const MARGIN: f64 = Policy::MARGIN;
    #[classattr]
    const PROBE_FIRST: usize = Policy::PROBE_FIRST;
    #[classattr]
    const PROBE_MAX: usize = Policy::PROBE_MAX;
    #[classattr]
    const PROBE_STEPS: usize = Policy::PROBE_STEPS;

    #[new]
    #[pyo3(signature = (k, drafter, mode="auto"))]
    fn new(k: usize, drafter: bool, mode: &str) -> PyResult<Self> {
        Ok(Self {
            policy: Policy::new(k, mode, !drafter)?,
        })
    }

    /// Whether a decode step of `rows` rows verifies drafts (a drafter's at the best of `widths`).
    fn speculate(&mut self, rows: usize, widths: Vec<usize>) -> bool {
        self.policy.speculate(rows, &widths)
    }

    /// A measured step of `key`: the tokens it committed and its time.
    fn record(&mut self, key: (String, usize), tokens: f64, ms: f64) -> PyResult<()> {
        let (verify, bucket) = policy_key(&key)?;
        self.policy.record(verify, bucket, tokens, ms);
        Ok(())
    }

    /// The tokens per ms of `key`: measured, else the fit (a verify step's at the best of `widths`).
    #[pyo3(signature = (key, widths=Vec::new()))]
    fn rate(&self, key: (String, usize), widths: Vec<usize>) -> PyResult<f64> {
        let (verify, bucket) = policy_key(&key)?;
        Ok(self.policy.rate(verify, bucket, &widths))
    }

    /// A verify step's tokens per ms by the fits, at about 60% accepted drafts.
    fn fit(&self, bucket: usize, width: usize) -> f64 {
        self.policy.fit(bucket, width)
    }
}

/// The last enqueued decode step, for the policy's timing: its kind and bucket, when it was chosen, its rows,
/// its verification (whose committed tokens count) and whether it switched kind.
struct Timed {
    verify: bool,
    bucket: usize,
    at: Instant,
    rows: usize,
    verification: Option<Py<QwenVerification>>,
    switched: bool,
}

/// The speculative side of the executor: what verify steps of a drafter and of the MTP layer share (the draft
/// policy, carried drafts, the open verification, the counters), and each one's own (`side`).
struct Draft {
    k: usize,
    widths: Vec<usize>,
    policy: Policy,
    next_width: usize,
    /// sequence -> (the next anchor's position, the anchor, its drafts)
    carried: HashMap<u64, (i64, i64, Vec<i64>)>,
    open: Option<Py<QwenVerification>>,
    timed: Option<Timed>,
    /// blocks, drafts, accepted, emitted
    stats: [u64; 4],
    /// a test aid: every verify step of a drafter at this width (`QwenExecutor.force_width`)
    force_width: Option<usize>,
    side: Side,
}

/// A drafter's block steps, or the MTP layer.
enum Side {
    Blocks(Box<Blocks>),
    Mtp(Box<Mtp>),
}

/// A DFlash 2 or DSpine drafter: its block steps after each verify step (device acceptance).
struct Blocks {
    drafter: qwen_draft::Drafter,
    layouts: HashMap<usize, BlockLayout>,
    /// (bucket, verify width) -> the block step's program and its upload and run sections
    programs: HashMap<(usize, usize), (Arc<step::Program>, [usize; 2])>,
    images: Images,
    next_buf: u64,
    group: usize,
    draft_block_rows: i64,
    geometry: qwen_draft::StateGeometry,
    last_accepted: Vec<i64>,
    last_width: usize,
}

/// The MTP layer: its attention and cache, the chain image of the commit and draft chain after each verify step
/// (the host accepts), and the staging of its drafts.
struct Mtp {
    attention: qwen_mtp::MtpAttention,
    /// its cache group, and the rows of one MTP KV block
    group: usize,
    block_rows: i64,
    /// the zero hidden (no hidden before a sequence's first position), the arena, its row bytes
    zero: u64,
    arena: u64,
    row_bytes: u64,
    geometry: qwen_draft::StateGeometry,
    /// a chained verify step's tokens [anchor, drafts] a row
    next_tok: u64,
    images: Images,
    /// bucket -> the chain image after a verify step of that many rows
    chains: HashMap<usize, qwen_mtp::ChainLayout>,
    /// the MTP layer's drafts at every column of a verify step (and a prompt's first draft): two staging sets
    tokens: Pinned,
    /// the chain's drafts 2 .. K of each row: two staging sets of (K, rows)
    drafts: Pinned,
}

impl Draft {
    fn blocks(&self) -> &Blocks {
        match &self.side {
            Side::Blocks(blocks) => blocks,
            Side::Mtp(_) => panic!("the MTP layer has no block steps"),
        }
    }

    fn blocks_mut(&mut self) -> &mut Blocks {
        match &mut self.side {
            Side::Blocks(blocks) => blocks,
            Side::Mtp(_) => panic!("the MTP layer has no block steps"),
        }
    }

    fn mtp(&self) -> Option<&Mtp> {
        match &self.side {
            Side::Mtp(mtp) => Some(mtp),
            Side::Blocks(_) => None,
        }
    }
}

/// Prompt scoring (`Qwen35Gaudi._score`): one program with a section per top-list length (`SCORE_TOPS`), that
/// uploads its tables (a prefill chunk's positions `rows` and (row, token) `pairs`) from `host`, makes the score
/// launch and copies its outputs (the pairs' logprobs, each row's statistics, its top values and ids) to the
/// pinned buffers behind `event`. A score launch runs alone: the host waits for its outputs before the next.
struct Scoring {
    program: Arc<step::Program>,
    sections: [usize; 3],
    host: Pinned,
    pairs_at: usize,
    lp: Pinned,
    stats: Pinned,
    top_v: Pinned,
    top_i: Pinned,
    event: i32,
}

/// Pinned staging sets of the outputs: two of each, `stride` bytes apart.
struct Staging {
    tokens: Pinned,
    lp: Pinned,
    top_idx: Pinned,
    top_lp: Pinned,
    accepted: Option<Pinned>,
    next: Option<Pinned>,
    columns: usize,
    rows: usize,
    block: usize,
    /// stage k's event e is slot events + k + 2 e
    events: i32,
}

impl Staging {
    fn event(&self, stage: usize, e: usize) -> i32 {
        self.events + (stage + 2 * e) as i32
    }
}

#[pyclass(module = "resin._resin")]
pub(crate) struct QwenExecutor {
    glue: Py<GaudiGlue>,
    model: qwen::Model,
    modes: HashMap<String, NativeMode>,
    prefill_buckets: Vec<usize>,
    decode_buckets: Vec<usize>,
    images: Images,
    staging: Staging,
    stage_next: usize,
    tok_out: u64,
    spin_ns: i64,
    vocab: u64,
    slab_rows: i64,
    kv_block_rows: i64,
    state_block_rows: i64,
    state_tokens: i64,
    kv_group: usize,
    state_group: usize,
    max_batch_size: usize,
    max_seq_len: i64,
    lookahead: i64,
    draft: Option<Draft>,
    native_steps: u64,
    timings: Timings,
    /// a tensor-parallel model's rank group (`attach_ranks`)
    ranks: Option<Ranked>,
    /// a follower: the driver's draft decision for the step it replays
    forced_speculating: Option<bool>,
    scoring: Option<Scoring>,
}

/// The MTP layer's part of a plain pass (`QwenExecutor::mtp_plain`).
enum MtpPlain<'a> {
    Decode,
    /// a prefill chunk of tokens `ids` (the chunk's own), and after the prompt the first draft's sampler inputs
    Prefill {
        ids: &'a [i64],
        draft: Option<&'a [f32]>,
    },
}

/// What a target pass reads its tokens from.
enum Tokens<'a> {
    Host(&'a [i64]),
    Device(u64),
}

/// One run of programs' sections, with the step's registers.
struct Run {
    steps: Vec<(Arc<step::Program>, usize)>,
    registers: [u64; REGISTERS],
}

impl QwenExecutor {
    fn sample(&self, plan: &BatchPlan, r: usize) -> PyResult<sampling::Sample> {
        let p = &plan.sampling_params[r];
        let seed = p
            .seed
            .ok_or_else(|| PyValueError::new_err("sampling params must contain effective seeds"))?;
        Ok(sampling::Sample {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k.map(u64::from),
            seed: seed as u64,
            sample: plan.sampling_mask[r],
            logprobs: p.logprobs,
            top_logprobs: p.top_logprobs as u64,
        })
    }

    /// Each row's first position, STATE and KV block rows, and with a drafter its DRAFT block rows
    /// (`Qwen3_5GaudiRunner._rows`); `positions` (a continuation's exact first positions) replace the plan's
    /// context lengths.
    fn rows(&self, plan: &BatchPlan, positions: Option<&[i64]>) -> PyResult<RowTables> {
        let group = |index: usize| {
            plan.cache_groups
                .get(index)
                .map(|group| group.inner_native())
                .ok_or_else(|| {
                    PyValueError::new_err(format!("the plan has no cache group {index}"))
                })
        };
        let (kv, state) = (group(self.kv_group)?, group(self.state_group)?);
        let draft = match self.draft.as_ref().map(|d| &d.side) {
            Some(Side::Blocks(b)) => Some((group(b.group)?, b.draft_block_rows)),
            _ => None,
        };
        let mtp = match self.draft.as_ref().and_then(Draft::mtp) {
            Some(m) => Some((group(m.group)?, m.block_rows)),
            None => None,
        };
        let n = plan.sequence_ids.len();
        let mut out = RowTables {
            indptr: vec![0],
            mtp_indptr: if mtp.is_some() { vec![0] } else { Vec::new() },
            ..RowTables::default()
        };
        for r in 0..n {
            let context = match positions {
                Some(positions) => positions[r],
                None => i64::from(plan.context_lengths[r]),
            };
            let query = i64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
            let last = context + query - 1;
            for i in kv.block_indptr[r] as usize..kv.block_indptr[r + 1] as usize {
                out.blocks.push(
                    i64::from(kv.block_slab_ids[i]) * self.slab_rows
                        + i64::from(kv.block_slotnums[i]) * self.kv_block_rows,
                );
            }
            out.indptr.push(out.blocks.len() as i64);
            let states: Vec<i64> = (state.block_indptr[r] as usize
                ..state.block_indptr[r + 1] as usize)
                .map(|i| {
                    i64::from(state.block_slab_ids[i]) * self.slab_rows
                        + i64::from(state.block_slotnums[i]) * self.state_block_rows
                })
                .collect();
            let first = i64::from(state.block_table_start_indices[r]);
            let row = StateRow {
                pos0: context,
                src: -1,
                blocks: &states,
                first,
            };
            let state_row = |position: i64| -> PyResult<i64> {
                row.state_at(position, self.state_tokens)
                    .map_err(|error| PyValueError::new_err(format!("row {r}: {error}")))
            };
            out.pos0.push(context);
            out.state_src.push(if context == 0 {
                -1
            } else {
                state_row(context - 1)?
            });
            out.state_dst.push(state_row(last)?);
            out.states.push(states);
            out.state_first.push(first);
            if let Some((draft, rows)) = draft {
                out.drafts.push(
                    (draft.block_indptr[r] as usize..draft.block_indptr[r + 1] as usize)
                        .map(|i| {
                            i64::from(draft.block_slab_ids[i]) * self.slab_rows
                                + i64::from(draft.block_slotnums[i]) * rows
                        })
                        .collect(),
                );
                out.draft_first
                    .push(i64::from(draft.block_table_start_indices[r]));
            }
            if let Some((group, rows)) = mtp {
                // like the KV blocks: from the row's first block
                for i in group.block_indptr[r] as usize..group.block_indptr[r + 1] as usize {
                    out.mtp_blocks.push(
                        i64::from(group.block_slab_ids[i]) * self.slab_rows
                            + i64::from(group.block_slotnums[i]) * rows,
                    );
                }
                out.mtp_indptr.push(out.mtp_blocks.len() as i64);
            }
        }
        Ok(out)
    }

    /// Whether the model drafts with its MTP layer.
    fn has_mtp(&self) -> bool {
        self.draft.as_ref().and_then(Draft::mtp).is_some()
    }

    fn mode(&self, name: &str) -> PyResult<&NativeMode> {
        self.modes
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("no native step mode {name}")))
    }

    fn err(issued: bool, py: Python<'_>, error: String) -> PyErr {
        if issued {
            post_accept(py, error)
        } else {
            PyValueError::new_err(error)
        }
    }

    /// The host copy of the step image to fill next (after its last upload is done): its index and bytes.
    fn take_image<'a>(&mut self, py: Python<'_>, block: bool) -> PyResult<(usize, &'a mut [u8])> {
        self.take_image_of(py, if block { Image::Block } else { Image::Step })
    }

    fn image_mut(&mut self, image: Image) -> &mut Images {
        match image {
            Image::Step => &mut self.images,
            Image::Block => &mut self.draft.as_mut().expect("a drafter").blocks_mut().images,
            Image::Chain => match &mut self.draft.as_mut().expect("the MTP layer").side {
                Side::Mtp(mtp) => &mut mtp.images,
                Side::Blocks(_) => panic!("a drafter has no chain image"),
            },
        }
    }

    /// `take_image` of any of the images.
    fn take_image_of<'a>(
        &mut self,
        py: Python<'_>,
        image: Image,
    ) -> PyResult<(usize, &'a mut [u8])> {
        let (k, used, event) = {
            let images = self.image_mut(image);
            let k = images.next;
            (k, images.used[k], images.events + k as i32)
        };
        if used {
            let glue = self.glue.get();
            let spin = self.spin_ns;
            let start = Instant::now();
            py.detach(|| glue.glue.wait(event, spin))
                .map_err(|error| post_accept(py, error))?;
            self.timings.image_wait += since(start);
        }
        let images = self.image_mut(image);
        images.used[k] = true;
        images.next ^= 1;
        // SAFETY: host copy k's last upload is done (its event, just waited for), and only this step writes it
        // until its own upload.
        let host = unsafe { images.host.bytes_mut(k * images.stride, images.stride) };
        Ok((k, host))
    }

    /// With ranks: tell the group whether this rank is ready for the run (its tables filled) and go ahead only when
    /// every rank is (so no rank enters a collective alone). Without: the prepared run.
    fn gated(&mut self, py: Python<'_>, prepared: PyResult<Run>) -> PyResult<Run> {
        match &self.ranks {
            Some(ranks) => ranks.gate(py, prepared, &mut self.timings.gate),
            None => prepared,
        }
    }

    fn submit(&mut self, py: Python<'_>, run: Run, name: &str) -> PyResult<()> {
        let glue = self.glue.get();
        let start = Instant::now();
        py.detach(|| {
            let steps: Vec<(&step::Program, usize)> = run
                .steps
                .iter()
                .map(|(program, section)| (program.as_ref(), *section))
                .collect();
            // SAFETY: the Python builder bound device allocations and the pinned host images and staging buffers
            // that this executor keeps, and recipes and event slots of the loaded runtime.
            let mut stream = unsafe { step::CardStream::new(&glue.glue) };
            step::run_sequence(&steps, &run.registers, &mut stream)
        })
        .map_err(|error| post_accept(py, format!("Qwen step {name}: {}", error.message)))?;
        self.timings.submit += since(start);
        self.native_steps += 1;
        Ok(())
    }

    /// Wait for staging set `stage`'s event `e` (0: the step's outputs, 1: the block step's tokens).
    fn wait_stage(&mut self, py: Python<'_>, stage: usize, e: usize) -> PyResult<()> {
        let glue = self.glue.get();
        let (event, spin) = (self.staging.event(stage, e), self.spin_ns);
        let start = Instant::now();
        py.detach(|| glue.glue.wait(event, spin))
            .map_err(|error| post_accept(py, error))?;
        self.timings.output_wait += since(start);
        Ok(())
    }

    fn take_stage(&mut self) -> usize {
        let stage = self.stage_next;
        self.stage_next ^= 1;
        stage
    }

    fn decode_bucket(&self, rows: usize) -> PyResult<usize> {
        self.decode_buckets
            .iter()
            .copied()
            .filter(|&b| b >= rows)
            .min()
            .ok_or_else(|| {
                PyValueError::new_err(format!("{rows} rows exceed {}", self.max_batch_size))
            })
    }

    fn check_plan(&self, plan: &BatchPlan) -> PyResult<()> {
        if plan.media.iter().any(Option::is_some) {
            return Err(PyNotImplementedError::new_err(
                "this runner has no vision tower; text prompts only",
            ));
        }
        if plan.sequence_ids.len() > self.max_batch_size {
            return Err(PyValueError::new_err(format!(
                "{} rows exceed {}",
                plan.sequence_ids.len(),
                self.max_batch_size
            )));
        }
        for (r, p) in plan.sampling_params.iter().enumerate() {
            if plan.sampling_mask[r] && p.logprobs && p.top_logprobs > TOP {
                return Err(PyNotImplementedError::new_err(format!(
                    "top_logprobs above {TOP}"
                )));
            }
        }
        Ok(())
    }

    /// The steps' outputs of staging set `stage`: `rows` x `per_row` columns' tokens, and with `scores` their
    /// logprobs and top lists, as copied out.
    fn staged(
        &self,
        stage: usize,
        columns: usize,
        scores: bool,
    ) -> (Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>) {
        let s = &self.staging;
        let at = stage * s.columns * 4;
        let tokens = s.tokens.words(at, columns);
        if !scores {
            return (tokens, Vec::new(), Vec::new(), Vec::new());
        }
        let top_at = stage * s.columns * TOP * 4;
        (
            tokens,
            s.lp.words(at, columns),
            s.top_idx.words(top_at, columns * TOP),
            s.top_lp.words(top_at, columns * TOP),
        )
    }

    /// One target pass of plain mode `name` (a prefill chunk or a plain decode step): its tables, with a
    /// drafter its injection of `inject` (column, position, count) runs, with the MTP layer its pass (`mtp`), and
    /// with `stage` its outputs copied out behind the stage's event 0.
    #[allow(clippy::too_many_arguments)]
    fn plain_prepare(
        &mut self,
        py: Python<'_>,
        name: &str,
        rows: &RowTables,
        real: usize,
        tokens: Tokens<'_>,
        params: &[f32],
        inject: &[(usize, i64, usize)],
        mtp: Option<MtpPlain<'_>>,
        stage: Option<(usize, usize, bool)>,
        issued: bool,
    ) -> PyResult<Run> {
        let (k, host) = self.take_image(py, false)?;
        let start = Instant::now();
        let native = self.mode(name)?;
        let image = self.images.device;
        let (host_tokens, tok) = match tokens {
            Tokens::Host(t) => (Some(t), image + native.layout.tok as u64),
            Tokens::Device(address) => (None, address),
        };
        let step = qwen::Step {
            rows: rows.rows(),
            real,
            tokens: host_tokens,
            params,
        };
        let filled = qwen::fill_target(&self.model, &native.mode, &native.layout, host, &step)
            .map_err(|e| Self::err(issued, py, e))?;
        let mut run = Run {
            steps: vec![native.section(S_TARGET)?],
            registers: [0; REGISTERS],
        };
        if let (Some(draft), Some(layout)) = (&self.draft, native.inject) {
            let runs: Vec<qwen_draft::Injected<'_>> = inject
                .iter()
                .enumerate()
                .map(|(i, &(column, position, count))| qwen_draft::Injected {
                    column,
                    position,
                    count,
                    row: rows.draft_row(i, position),
                })
                .collect();
            qwen_draft::fill_inject(
                &draft.blocks().drafter,
                &layout,
                host,
                native.layout.nbytes,
                &runs,
            )
            .map_err(|e| Self::err(issued, py, e))?;
            run.steps.push(native.section(S_INJECT)?);
        }
        let drafted = match mtp {
            Some(pass) => self
                .mtp_plain(native, rows, real, host, &pass, &mut run)
                .map_err(|e| Self::err(issued, py, e))?,
            None => false,
        };
        let r = &mut run.registers;
        r[R_IMAGE] = k as u64;
        r[R_BYTES] = filled.nbytes as u64;
        r[R_LAUNCHES] = filled.launches as u64;
        r[R_WLAUNCHES] = filled.wide as u64;
        r[R_TOK] = tok;
        if let Some((stage, columns, scores)) = stage {
            r[R_STAGE] = stage as u64;
            r[R_ROWS4] = 4 * columns as u64;
            r[R_TOP] = (4 * TOP * columns) as u64;
            run.steps.push(native.section(S_OUT)?);
            if scores {
                run.steps.push(native.section(S_SCORES)?);
            }
            if drafted {
                run.steps.push(native.section(S_MOUT)?);
            }
            run.steps.push(native.section(S_RECORD)?);
        }
        self.timings.fill += since(start);
        Ok(run)
    }

    /// The MTP layer's part of a plain pass, in the pass's image (`host`) and run: a decode step's keys at each
    /// row's position (its sampled token, the target's hidden), or a prefill chunk's keys (`Qwen35Gaudi.
    /// _prefill_mtp`: column c at position pos0 - 1 + c, its token ids[c], hidden c of [prev | x]; a sequence's
    /// first chunk starts one later), the chunk's last hidden into its STATE block, and after the prompt the
    /// first draft after its sampled token. Returns whether the run drafts (its draft goes to the staging).
    fn mtp_plain(
        &self,
        native: &NativeMode,
        rows: &RowTables,
        real: usize,
        host: &mut [u8],
        pass: &MtpPlain<'_>,
        run: &mut Run,
    ) -> Result<bool, String> {
        let mtp = self
            .draft
            .as_ref()
            .and_then(Draft::mtp)
            .ok_or("no MTP layer")?;
        let tables = native.mtp;
        let nbytes = native.layout.nbytes;
        let keys = tables
            .keys
            .ok_or_else(|| format!("mode {} has no MTP keys", native.mode.name))?;
        let mode = &native.mtp_mode;
        match pass {
            MtpPlain::Decode => {
                let cols: Vec<i64> = (1..=mode.columns as i64).collect();
                let pass = qwen_mtp::Pass {
                    starts: &rows.pos0,
                    real: 1,
                    blocks_indptr: &rows.mtp_indptr,
                    blocks: &rows.mtp_blocks,
                    cols: &cols,
                    tokens: None,
                    params: None,
                };
                qwen_mtp::fill_pass(&mtp.attention, mode, &keys, host, nbytes, &pass)?;
                run.steps
                    .push(native.section(S_MKEYS).map_err(|e| e.to_string())?);
                Ok(false)
            }
            MtpPlain::Prefill { ids, draft } => {
                let pos0 = rows.pos0[0];
                let shift = usize::from(pos0 == 0);
                if real > shift {
                    let n = real - shift;
                    let cols: Vec<i64> = (shift as i64..real as i64).collect();
                    let pass = qwen_mtp::Pass {
                        starts: &[pos0 - 1 + shift as i64],
                        real: n,
                        blocks_indptr: &rows.mtp_indptr,
                        blocks: &rows.mtp_blocks,
                        cols: &cols,
                        tokens: Some(&ids[shift..real]),
                        params: None,
                    };
                    qwen_mtp::fill_pass(&mtp.attention, mode, &keys, host, nbytes, &pass)?;
                    // the hidden before the chunk: its source STATE block's (none before position 0)
                    let src = rows.state_src[0];
                    run.registers[R_MPREV] = if src < 0 {
                        mtp.zero
                    } else {
                        mtp.arena + (src + mtp.geometry.hidden_offset) as u64 * mtp.row_bytes
                    };
                    run.steps
                        .push(native.section(S_MKEYS).map_err(|e| e.to_string())?);
                }
                let hidden = tables
                    .hidden
                    .ok_or_else(|| format!("mode {} has no hidden store", mode.name))?;
                qwen_mtp::fill_hidden(
                    &mtp.geometry,
                    host,
                    nbytes,
                    hidden,
                    real as i64 - 1,
                    rows.state_dst[0],
                )?;
                run.steps
                    .push(native.section(S_MHIDDEN).map_err(|e| e.to_string())?);
                let Some(params) = draft else {
                    return Ok(false);
                };
                // the sampled token at pos0 + real: the MTP layer at pos0 + real - 1 drafts pos0 + real + 1
                let layout = tables
                    .draft
                    .ok_or_else(|| format!("mode {} has no prompt draft", mode.name))?;
                let one = &self.mode_of_decode(1)?.mtp_mode;
                let pass = qwen_mtp::Pass {
                    starts: &[pos0 + real as i64 - 1],
                    real: 1,
                    blocks_indptr: &rows.mtp_indptr,
                    blocks: &rows.mtp_blocks,
                    cols: &[real as i64],
                    tokens: None,
                    params: Some(params),
                };
                let launches =
                    qwen_mtp::fill_pass(&mtp.attention, one, &layout, host, nbytes, &pass)?;
                run.registers[R_MLAUNCHES] = launches as u64;
                run.steps
                    .push(native.section(S_MDRAFT).map_err(|e| e.to_string())?);
                Ok(true)
            }
        }
    }

    /// The decode mode of `bucket` rows.
    fn mode_of_decode(&self, bucket: usize) -> Result<&NativeMode, String> {
        self.modes
            .get(&format!("d{bucket}"))
            .ok_or_else(|| format!("no native step mode d{bucket}"))
    }

    /// `plain_prepare`, then the run once every rank is ready.
    #[allow(clippy::too_many_arguments)]
    fn plain_pass(
        &mut self,
        py: Python<'_>,
        name: &str,
        rows: &RowTables,
        real: usize,
        tokens: Tokens<'_>,
        params: &[f32],
        inject: &[(usize, i64, usize)],
        mtp: Option<MtpPlain<'_>>,
        stage: Option<(usize, usize, bool)>,
        issued: &mut bool,
    ) -> PyResult<()> {
        let prepared = self.plain_prepare(
            py, name, rows, real, tokens, params, inject, mtp, stage, *issued,
        );
        let run = self.gated(py, prepared)?;
        *issued = true;
        self.submit(py, run, name)
    }

    /// A block step of `bucket` rows after host anchors (a prompt's drafts), its tokens copied out behind
    /// stage `stage`'s event 1 through mode `name`'s program. Returns its width.
    #[allow(clippy::too_many_arguments)]
    fn host_block_prepare(
        &mut self,
        py: Python<'_>,
        name: &str,
        rows: &[DraftRow<'_>],
        anchors: &[i64],
        bucket: usize,
        stage: usize,
        issued: bool,
    ) -> PyResult<(Run, usize)> {
        let (kb, host) = self.take_image(py, true)?;
        let draft = self.draft.as_ref().expect("a drafter");
        let width = draft.k + 1;
        let blocks = draft.blocks();
        let layout = blocks
            .layouts
            .get(&bucket)
            .ok_or_else(|| PyValueError::new_err(format!("no block step of {bucket} rows")))?;
        let filled = qwen_draft::fill_block(
            &blocks.drafter,
            layout,
            host,
            rows,
            Some(anchors),
            None,
            width,
            self.model.max_positions,
        )
        .map_err(|e| Self::err(issued, py, e))?;
        let (program, sections) = blocks.programs.get(&(bucket, width)).ok_or_else(|| {
            PyValueError::new_err(format!(
                "no block-step program for {bucket} rows at width {width}"
            ))
        })?;
        let native = self.mode(name)?;
        let mut run = Run {
            steps: vec![
                (Arc::clone(program), sections[0]),
                (Arc::clone(program), sections[1]),
                native.section(S_NEXT)?,
                native.section(S_RECORD1)?,
            ],
            registers: [0; REGISTERS],
        };
        let r = &mut run.registers;
        r[R_BIMAGE] = kb as u64;
        r[R_BBYTES] = filled.nbytes as u64;
        r[R_BLAUNCHES] = filled.launches as u64;
        r[R_STAGE] = stage as u64;
        r[R_NEXT] = (4 * bucket * blocks.drafter.block_of(width)) as u64;
        Ok((run, width))
    }

    /// `host_block_prepare`, then the run once every rank is ready. Returns the block step's width.
    #[allow(clippy::too_many_arguments)]
    fn host_block(
        &mut self,
        py: Python<'_>,
        name: &str,
        rows: &[DraftRow<'_>],
        anchors: &[i64],
        bucket: usize,
        stage: usize,
        issued: &mut bool,
    ) -> PyResult<usize> {
        let prepared = self.host_block_prepare(py, name, rows, anchors, bucket, stage, *issued);
        let (run, width) = match prepared {
            Ok((run, width)) => (Ok(run), width),
            Err(error) => (Err(error), 0),
        };
        let run = self.gated(py, run)?;
        *issued = true;
        self.submit(py, run, name)?;
        Ok(width)
    }

    /// The MTP layer's drafts at the first `columns` columns of staging set `stage` (after its event 0).
    fn mtp_tokens(&self, stage: usize, columns: usize) -> Vec<i32> {
        let mtp = self
            .draft
            .as_ref()
            .and_then(Draft::mtp)
            .expect("the MTP layer");
        mtp.tokens.words(stage * self.staging.columns * 4, columns)
    }

    /// The draft chain's drafts of staging set `stage` (after its event 1): (k, staging rows), link j's in row j.
    fn chain_drafts(&self, stage: usize, k: usize) -> Vec<i32> {
        let mtp = self
            .draft
            .as_ref()
            .and_then(Draft::mtp)
            .expect("the MTP layer");
        let words = k * self.staging.rows;
        mtp.drafts.words(stage * words * 4, words)
    }

    /// The block step's tokens in stage `stage` after its event 1: per row [anchor, drafts] of `width`.
    fn next_tokens(
        &mut self,
        py: Python<'_>,
        stage: usize,
        rows: usize,
        width: usize,
    ) -> PyResult<Vec<i32>> {
        self.wait_stage(py, stage, 1)?;
        let s = &self.staging;
        let next = s.next.as_ref().expect("a drafter's staging");
        Ok(next.words(stage * s.rows * s.block * 4, rows * width))
    }

    /// The drafts after a block step's tokens: (rows, K), zeros past its width (`Staging.next_drafts`).
    fn drafts_of(next: &[i32], rows: usize, width: usize, k: usize) -> Vec<Vec<i64>> {
        (0..rows)
            .map(|b| {
                let mut out = vec![0i64; k];
                for (j, slot) in out.iter_mut().enumerate().take(k.min(width - 1)) {
                    *slot = i64::from(next[b * width + 1 + j]);
                }
                out
            })
            .collect()
    }

    /// Finish the open verification that `feedback` does not continue, so its bookkeeping precedes any other
    /// step (`Qwen3_5GaudiRunner._settle`).
    fn settle(&mut self, py: Python<'_>, feedback: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let Some(draft) = self.draft.as_mut() else {
            return Ok(());
        };
        let Some(open) = draft.open.take() else {
            return Ok(());
        };
        if let Some(feedback) = feedback
            && feedback.is(open.bind(py))
        {
            return Ok(());
        }
        QwenVerification::result_of(open.bind(py), py, self)?;
        Ok(())
    }

    /// A synchronous step: every prefill row in order, or one decode step (with a drafter, a verify step when
    /// the policy speculates).
    #[allow(clippy::needless_range_loop)] // rows index several parallel tables
    fn forward_native(&mut self, py: Python<'_>, plan: &BatchPlan) -> PyResult<Py<ForwardOutput>> {
        self.check_plan(plan)?;
        if plan.needs_prompt_logprobs() && plan.phase != BatchPhase::Prefill {
            return Err(PyValueError::new_err(
                "prompt logprobs are scored by prefill plans",
            ));
        }
        let n = plan.sequence_ids.len();
        let rows = self.rows(plan, None)?;
        let scores = plan.needs_logprobs();
        let mut issued = false;
        let mut params = Vec::with_capacity(n);
        for r in 0..n {
            let query = u64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
            let offset = (u64::from(plan.context_lengths[r]) + query) * sampling::OFFSET_STRIDE;
            params.push(sampling::row_params(
                &self.sample(plan, r)?,
                self.vocab,
                offset,
            ));
        }
        for r in 0..n {
            let end = rows.pos0[r] + i64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
            if end > self.max_seq_len {
                return Err(PyValueError::new_err(format!(
                    "a row exceeds {} positions",
                    self.max_seq_len
                )));
            }
        }
        if plan.phase == BatchPhase::Prefill {
            let output = self.prefill(py, plan, &rows, &params, scores, &mut issued)?;
            return Py::new(py, output);
        }
        let bucket = self.decode_bucket(n)?;
        let speculating = self.decide(n);
        let anchors: Vec<i64> = plan
            .token_ids
            .iter()
            .take(n)
            .map(|&t| i64::from(t))
            .collect();
        if speculating {
            let drafts = self.carried_drafts(plan, &rows, &anchors);
            let positions: Vec<i64> = rows.pos0.clone();
            let verification = self.verify(
                py,
                plan,
                rows,
                positions,
                Anchors::Host(anchors, drafts),
                scores,
            )?;
            let verification = Bound::new(py, verification)?;
            return QwenVerification::result_of(&verification, py, self);
        }
        let mut columns = vec![0i64; bucket];
        columns[..n].copy_from_slice(&anchors);
        let flat: Vec<f32> = params.iter().flatten().copied().collect();
        let stage = self.take_stage();
        let inject: Vec<(usize, i64, usize)> = (0..n).map(|b| (b, rows.pos0[b], 1)).collect();
        let mtp = self.has_mtp().then_some(MtpPlain::Decode);
        self.plain_pass(
            py,
            &format!("d{bucket}"),
            &rows,
            n,
            Tokens::Host(&columns),
            &flat,
            &inject,
            mtp,
            Some((stage, n, scores)),
            &mut issued,
        )?;
        self.wait_stage(py, stage, 0)?;
        Py::new(py, self.plain_output(py, stage, &params, scores)?)
    }

    fn plain_output(
        &self,
        py: Python<'_>,
        stage: usize,
        params: &[[f32; PARAMS]],
        scores: bool,
    ) -> PyResult<ForwardOutput> {
        let rows = params.len();
        let (tokens, lp, ids, values) = self.staged(stage, rows, scores);
        let token_ids = tokens.iter().map(|&t| t as u32).collect();
        let indptr = (0..=rows as u32).collect();
        let (scores, top) = if scores {
            let (s, t) = score_columns(params, &vec![1; rows], &lp, &ids, &values);
            (Some(s), t)
        } else {
            (None, None)
        };
        ForwardOutput::native(py, token_ids, indptr, scores, top)
    }

    #[allow(clippy::needless_range_loop)]
    fn prefill(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        rows: &RowTables,
        params: &[[f32; PARAMS]],
        scores: bool,
        issued: &mut bool,
    ) -> PyResult<ForwardOutput> {
        let n = plan.sequence_ids.len();
        let stage = self.take_stage();
        let mut tokens = Vec::with_capacity(n);
        let (mut lp, mut top_ids, mut top_lp) = (Vec::new(), Vec::new(), Vec::new());
        // prompt logprobs (a follower's outputs are not read: it scores nothing)
        let prompt = plan.needs_prompt_logprobs() && !self.follower();
        let requests = if prompt {
            score_requests(plan, self.vocab)
        } else {
            Vec::new()
        };
        let mut scored: Vec<Vec<Scored>> = requests
            .iter()
            .map(|r| vec![Scored::default(); r.as_ref().map_or(0, |r| r.positions.len())])
            .collect();
        let mut samples = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);
        for r in 0..n {
            let query = u64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
            samples.push(self.sample(plan, r)?);
            offsets.push((u64::from(plan.context_lengths[r]) + query) * sampling::OFFSET_STRIDE);
        }
        for r in 0..n {
            let ids: Vec<i64> = plan.token_ids
                [plan.token_indptr[r] as usize..plan.token_indptr[r + 1] as usize]
                .iter()
                .map(|&t| i64::from(t))
                .collect();
            let drafting = plan.sampling_mask[r];
            let chunks = prefill_chunks(ids.len(), &self.prefill_buckets);
            let mut i = 0usize;
            let mut last_mode = String::new();
            for (c, &(real, bucket)) in chunks.iter().enumerate() {
                let last = c + 1 == chunks.len();
                let src = if i == 0 {
                    rows.state_src[r]
                } else {
                    rows.state_dst[r]
                };
                let chunk = rows.row(r, rows.pos0[r] + i as i64, src);
                let mut columns = vec![0i64; bucket];
                columns[..real].copy_from_slice(&ids[i..i + real]);
                let chunk_params: &[f32] = if last { &params[r] } else { &[0.0; PARAMS] };
                last_mode = format!("p{bucket}");
                // the MTP layer after the prompt: its first draft, sampled one position later
                let draft_params = drafting.then(|| {
                    sampling::row_params(
                        &samples[r],
                        self.vocab,
                        offsets[r] + sampling::OFFSET_STRIDE,
                    )
                });
                let mtp = self.has_mtp().then_some(MtpPlain::Prefill {
                    ids: &ids[i..i + real],
                    draft: draft_params.as_ref().filter(|_| last).map(|p| p.as_slice()),
                });
                self.plain_pass(
                    py,
                    &last_mode,
                    &chunk,
                    real,
                    Tokens::Host(&columns),
                    chunk_params,
                    &[(0, chunk.pos0[0], real)],
                    mtp,
                    last.then_some((stage, 1, scores)),
                    issued,
                )?;
                if let Some(Some(request)) = requests.get(r) {
                    let which: Vec<usize> = (0..request.positions.len())
                        .filter(|&k| (i..i + real).contains(&request.positions[k]))
                        .collect();
                    self.score(py, request, &which, i, &mut scored[r])?;
                }
                i += real;
            }
            self.wait_stage(py, stage, 0)?;
            let (token, l, ti, tl) = self.staged(stage, 1, scores);
            tokens.push(token[0] as u32);
            if scores {
                lp.extend(l);
                top_ids.extend(ti);
                top_lp.extend(tl);
            }
            // the MTP layer's first draft after the prompt's sampled token (the chain would write MTP KV past the
            // prompt, where a prefill leases no blocks, so drafts 2 .. K start as zeros)
            if self.has_mtp() && drafting {
                let position = rows.pos0[r] + ids.len() as i64;
                let first = i64::from(self.mtp_tokens(stage, 1)[0]);
                let draft = self.draft.as_mut().expect("the MTP layer");
                let mut drafts = vec![0i64; draft.k];
                drafts[0] = first;
                draft.carried.insert(
                    plan.sequence_ids[r],
                    (position, i64::from(token[0]), drafts),
                );
            }
            // the drafter: the block after the prompt's sampled token, carried to the row's first verify step
            if self.draft.is_some() && !self.has_mtp() && plan.sampling_mask[r] {
                let position = rows.pos0[r] + ids.len() as i64;
                let one = rows.row(r, position, -1);
                let row = one.draft_row(0, position);
                let anchor = i64::from(token[0]);
                let width = self.host_block(py, &last_mode, &[row], &[anchor], 1, stage, issued)?;
                let next = self.next_tokens(py, stage, 1, width)?;
                let draft = self.draft.as_mut().expect("a drafter");
                let drafts = Self::drafts_of(&next, 1, width, draft.k).remove(0);
                draft.next_width = draft.k + 1; // the prompt's drafts: all of them
                draft
                    .carried
                    .insert(plan.sequence_ids[r], (position, anchor, drafts));
            }
        }
        let (scores, top) = if scores {
            let (s, t) = score_columns(params, &vec![1; n], &lp, &top_ids, &top_lp);
            (Some(s), t)
        } else {
            (None, None)
        };
        let indptr = (0..=n as u32).collect();
        let prompt = prompt.then(|| prompt_columns(plan, &scored));
        ForwardOutput::native_with_prompt(py, tokens, indptr, scores, top, prompt)
    }

    /// Score request positions `which` (indices into `request`) of a prefill chunk from token `start` of its row
    /// (`Qwen35Gaudi._score`), in launches of up to SCORE_ROWS positions and SCORE_PAIRS target logprobs, into
    /// `out` (by request index). The chunk's target pass is queued: each launch reads its final hidden states.
    fn score(
        &mut self,
        py: Python<'_>,
        request: &ScoreRequest,
        which: &[usize],
        start: usize,
        out: &mut [Scored],
    ) -> PyResult<()> {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let (mut group, mut pairs) = (Vec::new(), 0usize);
        for &k in which {
            let n = request.targets[k].len();
            if !group.is_empty() && (group.len() == SCORE_ROWS || pairs + n > SCORE_PAIRS) {
                groups.push(std::mem::take(&mut group));
                pairs = 0;
            }
            if n > SCORE_PAIRS {
                return Err(post_accept(
                    py,
                    format!("more than {SCORE_PAIRS} prompt targets at one position"),
                ));
            }
            group.push(k);
            pairs += n;
        }
        if !group.is_empty() {
            groups.push(group);
        }
        for group in groups {
            let scoring = self
                .scoring
                .as_ref()
                .ok_or_else(|| post_accept(py, "no prompt scoring".into()))?;
            // SAFETY: the last score launch's outputs were waited for, so its upload of these tables is done.
            let host = unsafe { scoring.host.bytes_mut(0, scoring.host.bytes) };
            let words = unsafe {
                std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<i32>(), host.len() / 4)
            };
            words[..SCORE_ROWS].fill(0);
            let pairs_at = scoring.pairs_at / 4;
            words[pairs_at..pairs_at + 2 * SCORE_PAIRS].fill(-1);
            let mut spans = Vec::with_capacity(group.len());
            let mut p = 0usize;
            for (r, &k) in group.iter().enumerate() {
                words[r] = (request.positions[k] - start) as i32;
                let targets = &request.targets[k];
                for (j, &target) in targets.iter().enumerate() {
                    words[pairs_at + 2 * (p + j)] = r as i32;
                    words[pairs_at + 2 * (p + j) + 1] = target as i32;
                }
                spans.push((p, p + targets.len()));
                p += targets.len();
            }
            let top = group.iter().map(|&k| request.tops[k]).max().unwrap_or(0);
            let index = SCORE_TOPS
                .iter()
                .position(|&t| t >= top)
                .ok_or_else(|| post_accept(py, format!("a top list of {top} entries")))?;
            let tops = SCORE_TOPS[index];
            let (program, section, event) = (
                Arc::clone(&scoring.program),
                scoring.sections[index],
                scoring.event,
            );
            let run = Run {
                steps: vec![(program, section)],
                registers: [0; REGISTERS],
            };
            self.submit(py, run, "score")?;
            let glue = self.glue.get();
            let spin = self.spin_ns;
            let begin = Instant::now();
            py.detach(|| glue.glue.wait(event, spin))
                .map_err(|error| post_accept(py, error))?;
            self.timings.output_wait += since(begin);
            let scoring = self.scoring.as_ref().expect("prompt scoring");
            let lp: Vec<f32> = scoring.lp.words(0, SCORE_PAIRS);
            let (lse, values, ids): (Vec<f32>, Vec<f32>, Vec<i32>) = if tops > 0 {
                let stats: Vec<f32> = scoring.stats.words(0, 16 * SCORE_ROWS);
                (
                    (0..SCORE_ROWS).map(|r| stats[16 * r + 1]).collect(),
                    scoring.top_v.words(0, tops * SCORE_ROWS),
                    scoring.top_i.words(0, tops * SCORE_ROWS),
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            for (r, &k) in group.iter().enumerate() {
                let (a, b) = spans[r];
                let n = request.tops[k];
                out[k] = Scored {
                    targets: lp[a..b].to_vec(),
                    top_ids: (0..n).map(|j| ids[r * tops + j] as u32).collect(),
                    top_lp: (0..n).map(|j| values[r * tops + j] - lse[r]).collect(),
                };
            }
        }
        Ok(())
    }

    /// Whether this executor replays a driver's runner calls.
    fn follower(&self) -> bool {
        self.ranks.as_ref().is_some_and(|ranks| !ranks.driver)
    }

    /// Each row's drafts carried from its last step, zeros when they do not continue from this anchor
    /// (`_carried`).
    fn carried_drafts(&self, plan: &BatchPlan, rows: &RowTables, anchors: &[i64]) -> Vec<Vec<i64>> {
        let draft = self.draft.as_ref().expect("a drafter");
        (0..plan.sequence_ids.len())
            .map(|r| match draft.carried.get(&plan.sequence_ids[r]) {
                Some((position, anchor, drafts))
                    if *position == rows.pos0[r] && *anchor == anchors[r] =>
                {
                    drafts.clone()
                }
                _ => vec![0; draft.k],
            })
            .collect()
    }

    /// Launch a verify step of the rows (anchors at their positions): the target over each row's anchor and
    /// drafts, the injection, `spec_fix`'s acceptance on the device, the outputs, the commit, and the next block
    /// step behind it (`Qwen35Gaudi.verify_launch` with device acceptance).
    fn verify(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        rows: RowTables,
        positions: Vec<i64>,
        anchors: Anchors,
        scores: bool,
    ) -> PyResult<QwenVerification> {
        let mtp = self.draft.as_ref().and_then(Draft::mtp).is_some();
        let prepared = if mtp {
            self.verify_prepare_mtp(py, plan, rows, positions, anchors, scores)
        } else {
            self.verify_prepare(py, plan, rows, positions, anchors, scores)
        };
        let (run, rest) = match prepared {
            Ok((run, name, width, verification)) => (Ok(run), Some((name, width, verification))),
            Err(error) => (Err(error), None),
        };
        let run = self.gated(py, run)?;
        let (name, width, verification) = rest.expect("prepared");
        self.submit(py, run, &name)?;
        self.draft.as_mut().expect("a drafter").next_width = width;
        Ok(verification)
    }

    /// Each row's sampler inputs at its verify columns: column t of row b samples position positions[b] + t + 1
    /// (`Sampling.grid`), from `shift` columns on (the MTP layer's drafts: one later). Returns the (bucket x width)
    /// PARAMS table and each row's first column's inputs.
    #[allow(clippy::type_complexity)]
    fn verify_params(
        &self,
        plan: &BatchPlan,
        positions: &[i64],
        bucket: usize,
        width: usize,
        shift: u64,
    ) -> PyResult<(Vec<f32>, Vec<[f32; PARAMS]>)> {
        let mut params = vec![0f32; bucket * width * PARAMS];
        let mut first = Vec::with_capacity(positions.len());
        for (b, &position) in positions.iter().enumerate() {
            let sample = self.sample(plan, b)?;
            let base = (position as u64 + 1 + shift) * sampling::OFFSET_STRIDE;
            for t in 0..width {
                let p = sampling::row_params(
                    &sample,
                    self.vocab,
                    base + t as u64 * sampling::OFFSET_STRIDE,
                );
                params[(b * width + t) * PARAMS..(b * width + t + 1) * PARAMS].copy_from_slice(&p);
                if t == 0 {
                    first.push(p);
                }
            }
        }
        Ok((params, first))
    }

    /// Launch an MTP verify step of the rows (`Qwen35Gaudi.verify_launch` with the MTP layer): the target over
    /// each row's anchor and K drafts, the MTP layer at every column (its next drafts), and the outputs. The host
    /// accepts at the finish, which queues the commit and the draft chain.
    #[allow(clippy::type_complexity)]
    fn verify_prepare_mtp(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        rows: RowTables,
        positions: Vec<i64>,
        anchors: Anchors,
        scores: bool,
    ) -> PyResult<(Run, String, usize, QwenVerification)> {
        let n = rows.pos0.len();
        let bucket = self.decode_bucket(n)?;
        let draft = self.draft.as_ref().expect("the MTP layer");
        let tq = draft.k + 1;
        let chained = matches!(anchors, Anchors::Device(_));
        let name = format!("v{bucket}");
        let (params, row_params) = self.verify_params(plan, &positions, bucket, tq, 0)?;
        // the MTP layer's column t drafts position pos0 + t + 2
        let (mtp_params, _) = self.verify_params(plan, &positions, bucket, tq, 1)?;
        let mut samples = Vec::with_capacity(n);
        for b in 0..n {
            samples.push(self.sample(plan, b)?);
        }
        let mut columns = vec![0i64; bucket * tq];
        let host_drafts = match &anchors {
            Anchors::Host(a, d) => {
                for b in 0..n {
                    columns[b * tq] = a[b];
                    columns[b * tq + 1..(b + 1) * tq].copy_from_slice(&d[b][..tq - 1]);
                }
                Drafts::Host(d.iter().map(|d| d[..tq - 1].to_vec()).collect())
            }
            Anchors::Device(previous) => Drafts::Previous(previous.clone_ref(py)),
        };
        let stage = self.take_stage();
        let (k_img, host) = self.take_image(py, false)?;
        let start = Instant::now();
        let draft = self.draft.as_ref().expect("the MTP layer");
        let mtp = draft.mtp().expect("the MTP layer");
        let native = self.mode(&name)?;
        let image = self.images.device;
        let tok = if chained {
            mtp.next_tok
        } else {
            image + native.layout.tok as u64
        };
        let step = qwen::Step {
            rows: rows.rows(),
            real: n,
            tokens: (!chained).then_some(columns.as_slice()),
            params: &params,
        };
        let filled = qwen::fill_target(&self.model, &native.mode, &native.layout, host, &step)
            .map_err(PyValueError::new_err)?;
        let full = native
            .mtp
            .full
            .ok_or_else(|| PyValueError::new_err(format!("mode {name} has no MTP pass")))?;
        let cols: Vec<i64> = (1..=(bucket * tq) as i64).collect();
        let pass = qwen_mtp::Pass {
            starts: &rows.pos0,
            real: tq,
            blocks_indptr: &rows.mtp_indptr,
            blocks: &rows.mtp_blocks,
            cols: &cols,
            tokens: None,
            params: Some(&mtp_params[..n * tq * PARAMS]),
        };
        let launches = qwen_mtp::fill_pass(
            &mtp.attention,
            &native.mtp_mode,
            &full,
            host,
            native.layout.nbytes,
            &pass,
        )
        .map_err(PyValueError::new_err)?;
        let mut run = Run {
            steps: vec![
                native.section(S_TARGET)?,
                native.section(S_MFULL)?,
                native.section(S_OUT)?,
                native.section(S_MOUT)?,
            ],
            registers: [0; REGISTERS],
        };
        if scores {
            run.steps.push(native.section(S_SCORES)?);
        }
        run.steps.push(native.section(S_RECORD)?);
        let r = &mut run.registers;
        r[R_IMAGE] = k_img as u64;
        r[R_BYTES] = filled.nbytes as u64;
        r[R_LAUNCHES] = filled.launches as u64;
        r[R_WLAUNCHES] = filled.wide as u64;
        r[R_TOK] = tok;
        r[R_MLAUNCHES] = launches as u64;
        r[R_STAGE] = stage as u64;
        r[R_ROWS4] = (4 * bucket * tq) as u64;
        r[R_TOP] = (4 * TOP * bucket * tq) as u64;
        self.timings.fill += since(start);
        let verification = QwenVerification {
            executor: None,
            stage,
            rows: n,
            bucket,
            width: tq,
            next_width: tq,
            start: positions,
            params: row_params,
            scores,
            drafts: Some(host_drafts),
            sequences: plan.sequence_ids.clone(),
            done: None,
            continued: false,
            next_read: None,
            mtp: Some(MtpVerification {
                rows,
                samples,
                first: Vec::new(),
            }),
        };
        Ok((run, name, tq, verification))
    }

    /// After an MTP verify step's acceptance: queue its commit and the draft chain (`Qwen35Gaudi._commit`,
    /// `_chain`) from the chain image, the chain's drafts 2 .. K into staging set `stage` behind its event 1, and
    /// the next step's tokens [anchor, drafts] into the device buffer a chained step reads. No collectives: each
    /// rank runs its own.
    #[allow(clippy::too_many_arguments)]
    fn chain(
        &mut self,
        py: Python<'_>,
        verify: &MtpVerification,
        stage: usize,
        start: &[i64],
        accepted: &[i64],
        anchors: &[i64],
        first: &[i64],
    ) -> PyResult<()> {
        let n = accepted.len();
        let bucket = self.decode_bucket(n)?;
        let name = format!("v{bucket}");
        let draft = self.draft.as_ref().expect("the MTP layer");
        let k = draft.k;
        // link j samples each row's draft for position anchor + j + 1: step accepted + j + 1 past its first
        let mut params = Vec::with_capacity(k.saturating_sub(1));
        for j in 1..k {
            let mut link = Vec::with_capacity(n * PARAMS);
            for b in 0..n {
                let step = accepted[b] as u64 + j as u64 + 1;
                let offset = (start[b] as u64 + 1 + step) * sampling::OFFSET_STRIDE;
                link.extend(sampling::row_params(&verify.samples[b], self.vocab, offset));
            }
            params.push(link);
        }
        let (kc, host) = self.take_image_of(py, Image::Chain)?;
        let begin = Instant::now();
        let draft = self.draft.as_ref().expect("the MTP layer");
        let mtp = draft.mtp().expect("the MTP layer");
        let layout = mtp
            .chains
            .get(&bucket)
            .ok_or_else(|| post_accept(py, format!("no chain of {bucket} rows")))?;
        let decode = &self
            .mode_of_decode(bucket)
            .map_err(|e| post_accept(py, e))?
            .mtp_mode;
        let state_rows: Vec<StateRow<'_>> = (0..n).map(|b| verify.rows.state_row(b)).collect();
        let chain = qwen_mtp::Chain {
            rows: &state_rows,
            accepted,
            anchors,
            first,
            blocks_indptr: &verify.rows.mtp_indptr,
            blocks: &verify.rows.mtp_blocks,
            params: &params,
        };
        let filled =
            qwen_mtp::fill_chain(&mtp.attention, decode, layout, &mtp.geometry, host, &chain)
                .map_err(|e| post_accept(py, e))?;
        let native = self.mode(&name)?;
        let mut run = Run {
            steps: vec![
                native.section(S_MCOMMIT)?,
                native.section(S_MCHAIN)?,
                native.section(S_RECORD1)?,
            ],
            registers: [0; REGISTERS],
        };
        let r = &mut run.registers;
        r[R_CIMAGE] = kc as u64;
        r[R_CBYTES] = filled.nbytes as u64;
        r[R_STAGE] = stage as u64;
        r[R_ROWS] = n as u64;
        for (j, &launches) in filled.launches.iter().enumerate() {
            r[R_LINK + j] = launches as u64;
        }
        self.timings.fill += since(begin);
        self.submit(py, run, &name)
    }

    #[allow(clippy::type_complexity)]
    fn verify_prepare(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        rows: RowTables,
        positions: Vec<i64>,
        anchors: Anchors,
        scores: bool,
    ) -> PyResult<(Run, String, usize, QwenVerification)> {
        let n = rows.pos0.len();
        let bucket = self.decode_bucket(n)?;
        let draft = self.draft.as_ref().expect("a drafter");
        let full = draft.k + 1;
        let chained = matches!(anchors, Anchors::Device(_));
        let tq = draft.force_width.unwrap_or(draft.next_width);
        if !draft.widths.contains(&tq) {
            return Err(PyValueError::new_err(format!(
                "no verify step of {tq} columns"
            )));
        }
        let name = if tq == full {
            format!("v{bucket}")
        } else {
            format!("v{bucket}w{tq}")
        };
        // the next block step's width, from the rows' last acceptance
        let width = if let Some(width) = draft.force_width {
            width
        } else if draft.blocks().last_accepted.len() == n {
            qwen_draft::choose_width(
                &draft.widths,
                &draft.blocks().last_accepted,
                draft.blocks().last_width,
                bucket,
            )
        } else {
            qwen_draft::choose_width(&draft.widths, &vec![tq as i64 - 1; n], tq, bucket)
        };
        let mut params = vec![0f32; bucket * tq * PARAMS];
        let mut row_params = Vec::with_capacity(n);
        for b in 0..n {
            let sample = self.sample(plan, b)?;
            let base = (positions[b] as u64 + 1) * sampling::OFFSET_STRIDE;
            for t in 0..tq {
                let p = sampling::row_params(
                    &sample,
                    self.vocab,
                    base + t as u64 * sampling::OFFSET_STRIDE,
                );
                params[(b * tq + t) * PARAMS..(b * tq + t + 1) * PARAMS].copy_from_slice(&p);
                if t == 0 {
                    row_params.push(p);
                }
            }
        }
        let mut columns = vec![0i64; bucket * tq];
        let host_drafts = match &anchors {
            Anchors::Host(a, d) => {
                for b in 0..n {
                    columns[b * tq] = a[b];
                    columns[b * tq + 1..(b + 1) * tq].copy_from_slice(&d[b][..tq - 1]);
                }
                Drafts::Host(d.iter().map(|d| d[..tq - 1].to_vec()).collect())
            }
            Anchors::Device(previous) => Drafts::Previous(previous.clone_ref(py)),
        };
        let stage = self.take_stage();
        let (k_img, host) = self.take_image(py, false)?;
        let (kb, block_host) = self.take_image(py, true)?;
        let start = Instant::now();
        let draft = self.draft.as_ref().expect("a drafter");
        let blocks = draft.blocks();
        let native = self.mode(&name)?;
        let image = self.images.device;
        let next_buf = blocks.next_buf;
        let tok = if chained {
            next_buf
        } else {
            image + native.layout.tok as u64
        };
        let step = qwen::Step {
            rows: rows.rows(),
            real: n,
            tokens: (!chained).then_some(columns.as_slice()),
            params: &params,
        };
        let filled = qwen::fill_target(&self.model, &native.mode, &native.layout, host, &step)
            .map_err(PyValueError::new_err)?;
        let inject_layout = native
            .inject
            .ok_or_else(|| PyValueError::new_err(format!("mode {name} has no injection")))?;
        let runs: Vec<qwen_draft::Injected<'_>> = (0..n)
            .map(|b| qwen_draft::Injected {
                column: b * tq,
                position: rows.pos0[b],
                count: tq,
                row: rows.draft_row(b, rows.pos0[b]),
            })
            .collect();
        qwen_draft::fill_inject(
            &blocks.drafter,
            &inject_layout,
            host,
            native.layout.nbytes,
            &runs,
        )
        .map_err(PyValueError::new_err)?;
        // the next block step, its anchors decided on the device
        let block_layout = blocks
            .layouts
            .get(&bucket)
            .ok_or_else(|| PyValueError::new_err(format!("no block step of {bucket} rows")))?;
        let block_rows: Vec<DraftRow<'_>> = (0..n)
            .map(|b| rows.draft_row(b, rows.pos0[b] + tq as i64))
            .collect();
        let low: Vec<i64> = rows.pos0.iter().map(|p| p + 1).collect();
        let block = qwen_draft::fill_block(
            &blocks.drafter,
            block_layout,
            block_host,
            &block_rows,
            None,
            Some(&low),
            width,
            self.model.max_positions,
        )
        .map_err(PyValueError::new_err)?;
        let patch = qwen_draft::patch_entries(block_layout, &block.records);
        let fix_layout = native
            .fix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("mode {name} has no spec_fix tables")))?;
        let state_rows: Vec<StateRow<'_>> = (0..n).map(|b| rows.state_row(b)).collect();
        qwen_draft::fill_fix(
            fix_layout,
            &blocks.geometry,
            host,
            native.layout.nbytes,
            &state_rows,
            tq,
            &patch,
            blocks.drafter.block_of(width),
            blocks.drafter.window,
        )
        .map_err(PyValueError::new_err)?;
        let (program, sections) = blocks.programs.get(&(bucket, width)).ok_or_else(|| {
            PyValueError::new_err(format!(
                "no block-step program for {bucket} rows at width {width}"
            ))
        })?;
        let mut run = Run {
            steps: vec![
                native.section(S_TARGET)?,
                native.section(S_INJECT)?,
                (Arc::clone(program), sections[0]),
                native.section(S_FIX)?,
                native.section(S_OUT)?,
            ],
            registers: [0; REGISTERS],
        };
        if scores {
            run.steps.push(native.section(S_SCORES)?);
        }
        run.steps.extend([
            native.section(S_RECORD)?,
            native.section(S_COMMIT)?,
            (Arc::clone(program), sections[1]),
            native.section(S_NEXT)?,
            native.section(S_RECORD1)?,
        ]);
        let r = &mut run.registers;
        r[R_IMAGE] = k_img as u64;
        r[R_BYTES] = filled.nbytes as u64;
        r[R_LAUNCHES] = filled.launches as u64;
        r[R_WLAUNCHES] = filled.wide as u64;
        r[R_TOK] = tok;
        r[R_STAGE] = stage as u64;
        r[R_ROWS4] = (4 * bucket * tq) as u64;
        r[R_TOP] = (4 * TOP * bucket * tq) as u64;
        r[R_BIMAGE] = kb as u64;
        r[R_BBYTES] = block.nbytes as u64;
        r[R_BLAUNCHES] = block.launches as u64;
        r[R_FIXTIN] = tok;
        r[R_NEXT] = (4 * bucket * blocks.drafter.block_of(width)) as u64;
        r[R_ACC] = (4 * bucket) as u64;
        self.timings.fill += since(start);
        let verification = QwenVerification {
            executor: None,
            stage,
            rows: n,
            bucket,
            width: tq,
            next_width: width,
            start: positions,
            params: row_params,
            scores,
            drafts: Some(host_drafts),
            sequences: plan.sequence_ids.clone(),
            done: None,
            continued: false,
            next_read: None,
            mtp: None,
        };
        Ok((run, name, width, verification))
    }
}

impl QwenExecutor {
    /// The driver: send this runner call to the followers, which replay it (`serve`).
    fn announce(
        &mut self,
        py: Python<'_>,
        op: u8,
        plan: Option<&BatchPlan>,
        continues: bool,
        speculating: bool,
        ids: &[u64],
    ) -> PyResult<()> {
        if let Some(ranks) = self.ranks.as_ref() {
            self.timings.announce += ranks.announce(py, op, plan, continues, speculating, ids)?;
        }
        Ok(())
    }

    /// This step's draft decision: a follower's from its driver, else the policy's.
    fn decide(&mut self, rows: usize) -> bool {
        if let Some(decision) = self.forced_speculating.take() {
            return decision;
        }
        match self.draft.as_mut() {
            Some(draft) => {
                let widths = draft.widths.clone();
                draft.policy.speculate(rows, &widths)
            }
            None => false,
        }
    }
}

/// The anchors of a verify step: on the host with each row's drafts, or on the device after the previous
/// verify step (whose staged block step holds the drafts).
enum Anchors {
    Host(Vec<i64>, Vec<Vec<i64>>),
    Device(Py<QwenVerification>),
}

/// The drafts a verify step verified: known on the host, or the previous step's block step's (read at the
/// finish).
enum Drafts {
    Host(Vec<Vec<i64>>),
    Previous(Py<QwenVerification>),
}

/// A finished verification: each row's accepted drafts and emitted tokens, the rows' next positions and
/// anchors, and the step's output.
struct Done {
    accepted: Vec<i64>,
    positions: Vec<i64>,
    anchors: Vec<i64>,
    committed: u64,
    output: Py<ForwardOutput>,
}

#[pymethods]
impl QwenExecutor {
    #[classattr]
    const REGISTERS: usize = REGISTERS;
    #[classattr]
    const R_IMAGE: usize = R_IMAGE;
    #[classattr]
    const R_BYTES: usize = R_BYTES;
    #[classattr]
    const R_LAUNCHES: usize = R_LAUNCHES;
    #[classattr]
    const R_TOK: usize = R_TOK;
    #[classattr]
    const R_STAGE: usize = R_STAGE;
    #[classattr]
    const R_ROWS4: usize = R_ROWS4;
    #[classattr]
    const R_TOP: usize = R_TOP;
    #[classattr]
    const R_BIMAGE: usize = R_BIMAGE;
    #[classattr]
    const R_BBYTES: usize = R_BBYTES;
    #[classattr]
    const R_BLAUNCHES: usize = R_BLAUNCHES;
    #[classattr]
    const R_FIXTIN: usize = R_FIXTIN;
    #[classattr]
    const R_NEXT: usize = R_NEXT;
    #[classattr]
    const R_ACC: usize = R_ACC;
    #[classattr]
    const R_MPREV: usize = R_MPREV;
    #[classattr]
    const R_MLAUNCHES: usize = R_MLAUNCHES;
    #[classattr]
    const R_CIMAGE: usize = R_CIMAGE;
    #[classattr]
    const R_CBYTES: usize = R_CBYTES;
    #[classattr]
    const R_ROWS: usize = R_ROWS;
    #[classattr]
    const R_LINK: usize = R_LINK;
    #[classattr]
    const MAX_LINKS: usize = MAX_LINKS;
    #[classattr]
    const R_WLAUNCHES: usize = R_WLAUNCHES;
    #[classattr]
    const SECTIONS: [&'static str; SECTIONS.len()] = SECTIONS;
    #[classattr]
    const BLOCK_SECTIONS: [&'static str; 2] = BLOCK_SECTIONS;

    /// `fill`'s modes, each with its program (`programs`, with sections of SECTIONS) and with a drafter its
    /// injection and verify tables (`extras`: per mode a dict of `inject` (pos, kv_rows, kv_stride) and
    /// `fix` (patches and the FixLayout tables' offsets)). `image` / `host` / `host_stride` / `image_events`: the
    /// step image, its two pinned host copies and their events. `staging`: a dict of the pinned staging sets
    /// (tokens, lp, top_idx, top_lp, and with a drafter accepted and next) with `columns`, `rows`, `block` and
    /// the first of its four event slots (`events`). `draft`: the drafter's config dict, or None.
    #[new]
    #[pyo3(signature = (*, glue, fill, programs, extras, prefill_buckets, decode_buckets, image, host, host_stride, image_events, staging, tok_out, spin_ns, vocab, cache, max_batch_size, max_seq_len, lookahead, draft, mtp=None, scoring=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        glue: Py<GaudiGlue>,
        fill: PyRef<'_, QwenFill>,
        programs: HashMap<String, PyRef<'_, GaudiProgram>>,
        extras: HashMap<String, Bound<'_, PyDict>>,
        prefill_buckets: Vec<usize>,
        decode_buckets: Vec<usize>,
        image: u64,
        host: PyReadwriteArray1<'_, u8>,
        host_stride: usize,
        image_events: i32,
        staging: Bound<'_, PyDict>,
        tok_out: u64,
        spin_ns: i64,
        vocab: u64,
        cache: (i64, i64, i64, i64, usize, usize),
        max_batch_size: usize,
        max_seq_len: i64,
        lookahead: i64,
        draft: Option<Bound<'_, PyDict>>,
        mtp: Option<Bound<'_, PyDict>>,
        scoring: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (model, fill_modes) = fill.parts();
        let mut modes = HashMap::new();
        for (name, (mode, layout)) in fill_modes {
            let program = programs
                .get(name)
                .ok_or_else(|| PyValueError::new_err(format!("no program for mode {name}")))?
                .program()?;
            if program.registers() != REGISTERS {
                return Err(PyValueError::new_err(format!(
                    "mode {name}'s program has {} registers, not {REGISTERS}",
                    program.registers()
                )));
            }
            if layout.nbytes > host_stride {
                return Err(PyValueError::new_err(format!(
                    "mode {name}'s image of {} bytes exceeds the host images",
                    layout.nbytes
                )));
            }
            let mut sections = [None; SECTIONS.len()];
            for (k, section) in SECTIONS.iter().enumerate() {
                sections[k] = program.sections().iter().position(|s| s.name == *section);
            }
            if sections[S_TARGET].is_none()
                || sections[S_OUT].is_none()
                || sections[S_RECORD].is_none()
            {
                return Err(PyValueError::new_err(format!(
                    "mode {name}'s program lacks a section"
                )));
            }
            let (mut inject, mut fix) = (None, None);
            let mut mtp_tables = MtpTables::default();
            let mut mtp_mode = mode.clone();
            if let Some(extra) = extras.get(name) {
                if let Some(launches) = extra.get_item("mlaunches")? {
                    mtp_mode.max_launches = launches.extract()?;
                }
                if let Some(i) = extra.get_item("inject")? {
                    let (pos, kv_rows, kv_stride): (usize, usize, usize) = i.extract()?;
                    inject = Some(InjectLayout {
                        tokens: mode.columns,
                        pos,
                        kv_rows,
                        kv_stride,
                    });
                }
                if let Some(f) = extra.get_item("fix")? {
                    let (patches, offsets): (usize, Vec<usize>) = f.extract()?;
                    fix = Some(FixLayout::new(
                        mode.rows(),
                        model.gdn_layers,
                        patches,
                        &offsets,
                    ));
                }
                let pass =
                    |key: &str, pass_mode: &qwen::Mode| -> PyResult<Option<qwen_mtp::PassLayout>> {
                        Ok(match extra.get_item(key)? {
                            Some(offsets) => {
                                let offsets: Vec<usize> = offsets.extract()?;
                                if offsets.len() != 6 {
                                    return Err(PyValueError::new_err(format!(
                                        "mode {name}'s {key} takes 6 tables"
                                    )));
                                }
                                Some(qwen_mtp::PassLayout::new(pass_mode, &offsets))
                            }
                            None => None,
                        })
                    };
                // a prompt's first draft is a pass of one row's decode mode
                let one = &fill_modes
                    .get("d1")
                    .ok_or_else(|| PyValueError::new_err("no mode d1"))?
                    .0;
                mtp_tables = MtpTables {
                    keys: pass("mkeys", mode)?,
                    hidden: extra
                        .get_item("mhidden")?
                        .map(|offset| offset.extract())
                        .transpose()?,
                    draft: pass("mdraft", one)?,
                    full: pass("mfull", mode)?,
                };
            }
            modes.insert(
                name.clone(),
                NativeMode {
                    mode: mode.clone(),
                    layout: layout.clone(),
                    program,
                    sections,
                    inject,
                    fix,
                    mtp: mtp_tables,
                    mtp_mode,
                },
            );
        }
        for (b, kind) in prefill_buckets
            .iter()
            .map(|b| (b, 'p'))
            .chain(decode_buckets.iter().map(|b| (b, 'd')))
        {
            if !modes.contains_key(&format!("{kind}{b}")) {
                return Err(PyValueError::new_err(format!("no native mode {kind}{b}")));
            }
        }
        if prefill_buckets.is_empty() || decode_buckets.is_empty() {
            return Err(PyValueError::new_err(
                "prefill and decode buckets are needed",
            ));
        }
        let images = Images::new(image, Pinned::new("host", host)?, host_stride, image_events)?;
        let columns: usize = item(&staging, "columns")?;
        let rows: usize = item(&staging, "rows")?;
        let block: usize = item(&staging, "block")?;
        let pinned = |name: &str, words: usize| -> PyResult<Pinned> {
            let value = staging
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
            let array: PyReadwriteArray1<'_, u8> = value.extract()?;
            Pinned::at_least(name, array, 2 * 4 * words)
        };
        let optional = |name: &str, words: usize| -> PyResult<Option<Pinned>> {
            match staging.get_item(name)? {
                Some(value) if !value.is_none() => {
                    let array: PyReadwriteArray1<'_, u8> = value.extract()?;
                    Ok(Some(Pinned::at_least(name, array, 2 * 4 * words)?))
                }
                _ => Ok(None),
            }
        };
        let staging = Staging {
            tokens: pinned("tokens", columns)?,
            lp: pinned("lp", columns)?,
            top_idx: pinned("top_idx", columns * TOP)?,
            top_lp: pinned("top_lp", columns * TOP)?,
            accepted: optional("accepted", rows)?,
            next: optional("next", rows * block)?,
            columns,
            rows,
            block,
            events: item(&staging, "events")?,
        };
        let draft = match draft {
            None => None,
            Some(d) => {
                let drafter = qwen_draft::Drafter {
                    layers: item(&d, "layers")?,
                    pairs: item(&d, "pairs")?,
                    layer_rows: item(&d, "layer_rows")?,
                    block: item(&d, "block")?,
                    window: item(&d, "window")?,
                    causal: item(&d, "causal")?,
                    mask_token: item(&d, "mask_token")?,
                    scratch_rows: item(&d, "scratch_rows")?,
                    max_rows: item(&d, "max_rows")?,
                };
                let k: usize = item(&d, "k")?;
                let widths: Vec<usize> = item(&d, "widths")?;
                let mut layouts = HashMap::new();
                let mut block_programs = HashMap::new();
                let block_stride: usize = item(&d, "host_stride")?;
                let programs_value = d
                    .get_item("programs")?
                    .ok_or_else(|| PyValueError::new_err("missing programs"))?;
                let programs: HashMap<(usize, usize), PyRef<'_, GaudiProgram>> =
                    programs_value.extract()?;
                for ((bucket, width), program) in &programs {
                    let layout = BlockLayout::new(&drafter, *bucket);
                    if layout.nbytes > block_stride {
                        return Err(PyValueError::new_err(
                            "a block-step image exceeds its host copies",
                        ));
                    }
                    let program = program.program()?;
                    let mut sections = [0usize; 2];
                    for (i, section) in BLOCK_SECTIONS.iter().enumerate() {
                        sections[i] = program
                            .sections()
                            .iter()
                            .position(|s| s.name == *section)
                            .ok_or_else(|| {
                                PyValueError::new_err(format!(
                                    "a block-step program lacks {section}"
                                ))
                            })?;
                    }
                    block_programs.insert((*bucket, *width), (program, sections));
                    layouts.insert(*bucket, layout);
                }
                let block_value = d
                    .get_item("host")?
                    .ok_or_else(|| PyValueError::new_err("missing host"))?;
                let block_host: PyReadwriteArray1<'_, u8> = block_value.extract()?;
                let images = Images::new(
                    item(&d, "image")?,
                    Pinned::new("block host", block_host)?,
                    block_stride,
                    item(&d, "image_events")?,
                )?;
                let policy: String = item(&d, "policy")?;
                let (state_layer_rows, ssm_rows, hidden_offset): (i64, i64, i64) =
                    item(&d, "state")?;
                let full = k + 1;
                if staging.accepted.is_none() || staging.next.is_none() {
                    return Err(PyValueError::new_err(
                        "a drafter needs the accepted and next staging",
                    ));
                }
                Some(Draft {
                    k,
                    widths,
                    policy: Policy::new(k, &policy, false)?,
                    next_width: full,
                    carried: HashMap::new(),
                    open: None,
                    timed: None,
                    stats: [0; 4],
                    force_width: None,
                    side: Side::Blocks(Box::new(Blocks {
                        drafter,
                        layouts,
                        programs: block_programs,
                        images,
                        next_buf: item(&d, "next")?,
                        group: item(&d, "group")?,
                        draft_block_rows: item(&d, "draft_block_rows")?,
                        geometry: qwen_draft::StateGeometry {
                            state_layer_rows,
                            ssm_rows,
                            state_tokens: cache.3,
                            hidden_offset,
                        },
                        last_accepted: Vec::new(),
                        last_width: full,
                    })),
                })
            }
        };
        let draft = match (draft, mtp) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "the MTP layer and a drafter are alternatives",
                ));
            }
            (None, Some(m)) => Some(Self::mtp_side(
                &m,
                &modes,
                &staging,
                model.gdn_layers,
                cache.3,
            )?),
            (draft, None) => draft,
        };
        let scoring = scoring.map(|d| Self::scoring(&d)).transpose()?;
        let (slab_rows, kv_block_rows, state_block_rows, state_tokens, kv_group, state_group) =
            cache;
        Ok(Self {
            glue,
            model,
            modes,
            prefill_buckets,
            decode_buckets,
            images,
            staging,
            stage_next: 0,
            tok_out,
            spin_ns,
            vocab,
            slab_rows,
            kv_block_rows,
            state_block_rows,
            state_tokens,
            kv_group,
            state_group,
            max_batch_size,
            max_seq_len,
            lookahead,
            draft,
            native_steps: 0,
            timings: Timings::default(),
            ranks: None,
            forced_speculating: None,
            scoring,
        })
    }

    /// Program runs this executor made.
    #[getter]
    fn native_steps(&self) -> u64 {
        self.native_steps
    }

    /// Cumulative host nanoseconds of the native steps: fill, submit, image_wait, output_wait, and with ranks
    /// gate (waiting until every rank is ready) and announce (the driver sending the calls).
    #[getter]
    fn timings(&self) -> std::collections::BTreeMap<&'static str, u64> {
        let t = self.timings;
        [
            ("fill", t.fill),
            ("submit", t.submit),
            ("image_wait", t.image_wait),
            ("output_wait", t.output_wait),
            ("gate", t.gate),
            ("announce", t.announce),
        ]
        .into_iter()
        .collect()
    }

    /// The verify steps' acceptance counters: (blocks, drafts, accepted, emitted).
    #[getter]
    fn speculation(&self) -> (u64, u64, u64, u64) {
        let s = self.draft.as_ref().map_or([0; 4], |d| d.stats);
        (s[0], s[1], s[2], s[3])
    }

    /// The runner's `metrics_snapshot` (the speculation counters), with no Python: the runner binds it.
    fn metrics_snapshot(&self) -> crate::metrics::RunnerMetrics {
        let (blocks, drafts, accepted, _) = self.speculation();
        crate::metrics::RunnerMetrics::speculation(drafts, accepted, blocks)
    }

    /// The draft policy's mode: auto, always or never (None without a drafter).
    #[getter]
    fn policy_mode(&self) -> Option<&'static str> {
        self.draft.as_ref().map(|d| match d.policy.mode {
            PolicyMode::Auto => "auto",
            PolicyMode::Always => "always",
            PolicyMode::Never => "never",
        })
    }

    #[setter]
    fn set_policy_mode(&mut self, mode: &str) -> PyResult<()> {
        let draft = self
            .draft
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("no drafter: no draft policy"))?;
        draft.policy.mode = Policy::new(draft.k, mode, false)?.mode;
        Ok(())
    }

    /// A test aid: every verify step of the drafter at `width` columns (one of its verify widths), or None (the
    /// width that the last step's acceptance gives).
    #[getter]
    fn force_width(&self) -> Option<usize> {
        self.draft.as_ref().and_then(|d| d.force_width)
    }

    #[setter]
    fn set_force_width(&mut self, width: Option<usize>) -> PyResult<()> {
        let draft = self
            .draft
            .as_mut()
            .filter(|d| matches!(d.side, Side::Blocks(_)))
            .ok_or_else(|| PyValueError::new_err("no drafter: no verify widths to force"))?;
        if let Some(w) = width
            && !draft.widths.contains(&w)
        {
            return Err(PyValueError::new_err(format!(
                "no verify step of {w} columns ({:?})",
                draft.widths
            )));
        }
        draft.force_width = width;
        Ok(())
    }

    /// A test aid: the drafts carried for `sequence`'s next verify step: (the next anchor's position, the
    /// anchor, its drafts), or None.
    fn carried(&self, sequence: u64) -> Option<(i64, i64, Vec<i64>)> {
        self.draft.as_ref()?.carried.get(&sequence).cloned()
    }

    /// Drop the sequences' carried drafts.
    fn release(&mut self, py: Python<'_>, sequence_ids: Vec<u64>) -> PyResult<()> {
        self.announce(py, OP_RELEASE, None, false, false, &sequence_ids)?;
        self.release_ids(&sequence_ids);
        Ok(())
    }

    /// The Engine's synchronous step: a prefill (with its prompt logprobs) or a decode step.
    fn forward(&mut self, py: Python<'_>, plan: Bound<'_, BatchPlan>) -> PyResult<Py<PyAny>> {
        self.lead(py, OP_FORWARD, plan.get(), None)?;
        let out = self.forward_step(py, plan);
        self.forced_speculating = None;
        self.led(py, &out);
        out
    }

    /// Submit a decode step and return without waiting for it (`Qwen3_5GaudiRunner.enqueue`'s contract): the
    /// device continues from `feedback` (the previous step's exact positions; its tokens stay on the device).
    /// Other plans run synchronously.
    #[pyo3(signature = (plan, *, feedback=None))]
    fn enqueue(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow_mut()
            .lead(py, OP_ENQUEUE, plan.get(), feedback.as_ref())?;
        let out = Self::enqueue_step(&slf, py, plan, feedback);
        let mut this = slf.borrow_mut();
        this.forced_speculating = None;
        this.led(py, &out);
        out
    }

    /// Join a tensor-parallel model's rank group: rank 0 drives (it sends each runner call to the others), and
    /// every other rank replays the calls (`serve`).
    fn attach_ranks(&mut self, py: Python<'_>, ranks: Py<GaudiRanks>) {
        self.ranks = Ranked::of(py, ranks);
    }

    /// The driver: tell the followers to stop serving.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        self.announce(py, OP_CLOSE, None, false, false, &[])?;
        self.ranks = None;
        Ok(())
    }

    /// A follower: replay the driver's runner calls until it closes the group. Returns the calls replayed.
    fn serve(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<u64> {
        crate::gaudi_ranks::serve(&slf, py)
    }
}

impl QwenExecutor {
    /// The MTP layer's side (`mtp`: the config dict of `qwen3_5.native`): `k` drafts a step, its `attention`
    /// (KV groups, heads per group, rope positions), cache `group` and `block_rows`, the `zero` hidden, the
    /// `arena` and its `row_bytes`, the `state` geometry, the chained step's tokens `next_tok`, the chain image
    /// (`image`, `host`, `host_stride`, `image_events`) of each speculative bucket (`buckets`), the staging of its
    /// drafts (`tokens`, `drafts`) and the draft `policy`.
    fn mtp_side(
        m: &Bound<'_, PyDict>,
        modes: &HashMap<String, NativeMode>,
        staging: &Staging,
        gdn_layers: usize,
        state_tokens: i64,
    ) -> PyResult<Draft> {
        let k: usize = item(m, "k")?;
        if k == 0 || k > MAX_LINKS + 1 {
            return Err(PyValueError::new_err(format!(
                "the MTP layer drafts 1 .. {} tokens a step, not {k}",
                MAX_LINKS + 1
            )));
        }
        let (kv_groups, hpg, max_positions): (i64, i64, i64) = item(m, "attention")?;
        let kv8: u8 = item(m, "kv8")?;
        let (state_layer_rows, ssm_rows, hidden_offset): (i64, i64, i64) = item(m, "state")?;
        let host_stride: usize = item(m, "host_stride")?;
        let pinned = |name: &str| -> PyResult<Pinned> {
            let value = m
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
            let array: PyReadwriteArray1<'_, u8> = value.extract()?;
            Pinned::new(name, array)
        };
        let images = Images::new(
            item(m, "image")?,
            pinned("host")?,
            host_stride,
            item(m, "image_events")?,
        )?;
        let mut chains = HashMap::new();
        let buckets: Vec<usize> = item(m, "buckets")?;
        for bucket in buckets {
            let decode = &modes
                .get(&format!("d{bucket}"))
                .ok_or_else(|| PyValueError::new_err(format!("no mode d{bucket}")))?
                .mtp_mode;
            let layout = qwen_mtp::ChainLayout::new(decode, k + 1, gdn_layers);
            if layout.nbytes > host_stride {
                return Err(PyValueError::new_err(
                    "a chain image exceeds its host copies",
                ));
            }
            for name in [format!("v{bucket}")] {
                let native = modes
                    .get(&name)
                    .ok_or_else(|| PyValueError::new_err(format!("no mode {name}")))?;
                for section in [S_MFULL, S_MOUT, S_MCOMMIT, S_MCHAIN, S_RECORD1] {
                    native.section(section)?;
                }
            }
            chains.insert(bucket, layout);
        }
        let (tokens, drafts) = (pinned("tokens")?, pinned("drafts")?);
        for (name, array, bytes) in [
            ("tokens", &tokens, 2 * 4 * staging.columns),
            ("drafts", &drafts, 2 * 4 * k * staging.rows),
        ] {
            if array.bytes < bytes {
                return Err(PyValueError::new_err(format!(
                    "the MTP {name} staging holds {} bytes, not {bytes}",
                    array.bytes
                )));
            }
        }
        let policy: String = item(m, "policy")?;
        Ok(Draft {
            k,
            widths: vec![k + 1],
            policy: Policy::new(k, &policy, true)?,
            next_width: k + 1,
            carried: HashMap::new(),
            open: None,
            timed: None,
            stats: [0; 4],
            force_width: None,
            side: Side::Mtp(Box::new(Mtp {
                attention: qwen_mtp::MtpAttention {
                    kv_groups,
                    pages: qwen::Pages {
                        k8: kv8 & 1 != 0,
                        v8: kv8 & 2 != 0,
                    },
                    hpg,
                    max_positions,
                },
                group: item(m, "group")?,
                block_rows: item(m, "block_rows")?,
                zero: item(m, "zero")?,
                arena: item(m, "arena")?,
                row_bytes: item(m, "row_bytes")?,
                geometry: qwen_draft::StateGeometry {
                    state_layer_rows,
                    ssm_rows,
                    state_tokens,
                    hidden_offset,
                },
                next_tok: item(m, "next_tok")?,
                images,
                chains,
                tokens,
                drafts,
            })),
        })
    }

    /// Prompt scoring (`scoring`: the config dict of `qwen3_5.native`): the score `program` with a section `k{K}`
    /// per top-list length, its tables' pinned `host` copy (the positions, then the pairs at `pairs_at`), the
    /// pinned outputs (`lp`, `stats`, `top_v`, `top_i`) and their `event`.
    fn scoring(d: &Bound<'_, PyDict>) -> PyResult<Scoring> {
        let program = d
            .get_item("program")?
            .ok_or_else(|| PyValueError::new_err("missing program"))?;
        let program = program.cast::<GaudiProgram>()?.borrow().program()?;
        if program.registers() != REGISTERS {
            return Err(PyValueError::new_err(format!(
                "the score program has {} registers, not {REGISTERS}",
                program.registers()
            )));
        }
        let mut sections = [0usize; 3];
        for (i, top) in SCORE_TOPS.iter().enumerate() {
            let name = format!("k{top}");
            sections[i] = program
                .sections()
                .iter()
                .position(|s| s.name == name)
                .ok_or_else(|| PyValueError::new_err(format!("the score program lacks {name}")))?;
        }
        let pinned = |name: &str, bytes: usize| -> PyResult<Pinned> {
            let value = d
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
            let array: PyReadwriteArray1<'_, u8> = value.extract()?;
            Pinned::at_least(name, array, bytes)
        };
        let pairs_at: usize = item(d, "pairs_at")?;
        if pairs_at < 4 * SCORE_ROWS || !pairs_at.is_multiple_of(4) {
            return Err(PyValueError::new_err(
                "the score pairs overlap the positions",
            ));
        }
        let top = SCORE_TOPS[2] * SCORE_ROWS * 4;
        Ok(Scoring {
            program,
            sections,
            host: pinned("host", pairs_at + 8 * SCORE_PAIRS)?,
            pairs_at,
            lp: pinned("lp", 4 * SCORE_PAIRS)?,
            stats: pinned("stats", 4 * 16 * SCORE_ROWS)?,
            top_v: pinned("top_v", top)?,
            top_i: pinned("top_i", top)?,
            event: item(d, "event")?,
        })
    }

    fn release_ids(&mut self, sequence_ids: &[u64]) {
        if let Some(draft) = self.draft.as_mut() {
            for sequence in sequence_ids {
                draft.carried.remove(sequence);
            }
        }
    }

    /// The driver: decide the call's draft step (the followers take the same) and send the call to them.
    fn lead(
        &mut self,
        py: Python<'_>,
        op: u8,
        plan: &BatchPlan,
        feedback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let Some(ranks) = self.ranks.as_ref().filter(|r| r.driver) else {
            return Ok(());
        };
        // the plans that no rank can run fail here, before the followers see them
        self.check_plan(plan)?;
        if plan.needs_prompt_logprobs() && plan.phase != BatchPhase::Prefill {
            return Err(PyValueError::new_err(
                "prompt logprobs are scored by prefill plans",
            ));
        }
        if let Some(feedback) = feedback {
            let continues_last = match feedback.cast::<QwenFeedback>() {
                Ok(plain) => ranks.continues_last(py, plain.get().completion.bind(py).as_any()),
                Err(_) => ranks.continues_last(py, feedback),
            };
            if !continues_last {
                return Err(PyValueError::new_err(
                    "tensor-parallel ranks continue only the last enqueued step",
                ));
            }
        }
        let decides = plan.phase == BatchPhase::Decode && self.draft.is_some();
        let speculating = decides && self.decide(plan.sequence_ids.len());
        self.announce(py, op, Some(plan), feedback.is_some(), speculating, &[])?;
        if decides {
            self.forced_speculating = Some(speculating); // the step takes the decision that the followers got
        }
        Ok(())
    }

    /// The driver: the call's result is the step that the next call can continue.
    fn led(&mut self, py: Python<'_>, out: &PyResult<Py<PyAny>>) {
        if let Some(ranks) = self.ranks.as_mut() {
            ranks.led(py, out);
        }
    }

    fn forward_step(&mut self, py: Python<'_>, plan: Bound<'_, BatchPlan>) -> PyResult<Py<PyAny>> {
        self.settle(py, None)?;
        Ok(self.forward_native(py, plan.get())?.into_any())
    }

    fn enqueue_step(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let p = plan.get();
        if p.phase != BatchPhase::Decode || p.needs_prompt_logprobs() {
            return slf.borrow_mut().forward_step(py, plan);
        }
        let mut this = slf.borrow_mut();
        this.settle(py, feedback.as_ref())?;
        this.check_plan(p)?;
        let n = p.sequence_ids.len();
        // the previous step: its positions, and what the next tokens come from
        enum Previous {
            None,
            Plain(Py<QwenCompletion>),
            Verify(Py<QwenVerification>),
        }
        let (positions, previous) = match &feedback {
            None => (
                p.context_lengths
                    .iter()
                    .map(|&c| i64::from(c))
                    .collect::<Vec<_>>(),
                Previous::None,
            ),
            Some(feedback) => {
                if let Ok(plain) = feedback.cast::<QwenFeedback>() {
                    (
                        plain.get().positions.clone(),
                        Previous::Plain(plain.get().completion.clone_ref(py)),
                    )
                } else if let Ok(verification) = feedback.cast::<QwenVerification>() {
                    drop(this);
                    QwenVerification::finish_of(verification, py, &mut slf.borrow_mut())?;
                    this = slf.borrow_mut();
                    verification.borrow_mut().continued = true;
                    let positions = verification
                        .borrow()
                        .done
                        .as_ref()
                        .expect("finished")
                        .positions
                        .clone();
                    (positions, Previous::Verify(verification.clone().unbind()))
                } else {
                    return Err(PyValueError::new_err(
                        "the native executor continues only its own steps",
                    ));
                }
            }
        };
        if positions.len() != n {
            return Err(PyValueError::new_err(format!(
                "feedback has {} rows, the plan {n}",
                positions.len()
            )));
        }
        let lookahead = if n <= SPEC_ROWS { this.lookahead } else { 0 };
        if positions.iter().copied().max().unwrap_or(0) + 1 + lookahead > this.max_seq_len {
            return Err(PyValueError::new_err(format!(
                "a row exceeds {} positions",
                this.max_seq_len
            )));
        }
        let rows = this.rows(p, Some(&positions))?;
        let bucket = this.decode_bucket(n)?;
        let scores = p.needs_logprobs();
        // the policy: whether this step verifies drafts, and the last step's time
        let now = Instant::now();
        let speculating = this.decide(n);
        if let Some(draft) = this.draft.as_mut()
            && let Some(last) = &draft.timed
            && feedback.is_some()
            && last.verify == speculating
            && last.bucket == bucket
            && !last.switched
        {
            let tokens = match &last.verification {
                None => last.rows as f64,
                Some(v) => v.borrow(py).done.as_ref().map_or(0, |d| d.committed) as f64,
            };
            let ms = now.duration_since(last.at).as_secs_f64() * 1e3;
            draft.policy.record(speculating, bucket, tokens, ms);
        }
        let switched = match this.draft.as_ref().and_then(|d| d.timed.as_ref()) {
            Some(last) => feedback.is_none() || last.verify != speculating,
            None => true,
        };
        if !speculating {
            let mut params = Vec::with_capacity(n);
            for (r, &position) in positions.iter().enumerate() {
                let offset = (position as u64 + 1) * sampling::OFFSET_STRIDE;
                params.push(sampling::row_params(
                    &this.sample(p, r)?,
                    this.vocab,
                    offset,
                ));
            }
            let flat: Vec<f32> = params.iter().flatten().copied().collect();
            let stage = this.take_stage();
            let mut issued = false;
            let columns: Vec<i64>;
            let tokens = match &previous {
                Previous::None => {
                    columns = (0..bucket)
                        .map(|c| {
                            p.token_ids
                                .get(c)
                                .filter(|_| c < n)
                                .map_or(0, |&t| i64::from(t))
                        })
                        .collect();
                    Tokens::Host(&columns)
                }
                Previous::Plain(_) => Tokens::Device(this.tok_out), // its samples, row for row
                Previous::Verify(v) => {
                    // its samples are not one a row: the anchors on the host
                    let anchors = v
                        .borrow(py)
                        .done
                        .as_ref()
                        .expect("finished")
                        .anchors
                        .clone();
                    let mut c = vec![0i64; bucket];
                    c[..n].copy_from_slice(&anchors);
                    columns = c;
                    Tokens::Host(&columns)
                }
            };
            let inject: Vec<(usize, i64, usize)> = (0..n).map(|b| (b, rows.pos0[b], 1)).collect();
            let mtp = this.has_mtp().then_some(MtpPlain::Decode);
            this.plain_pass(
                py,
                &format!("d{bucket}"),
                &rows,
                n,
                tokens,
                &flat,
                &inject,
                mtp,
                Some((stage, n, scores)),
                &mut issued,
            )?;
            if let Some(draft) = this.draft.as_mut() {
                draft.timed = Some(Timed {
                    verify: false,
                    bucket,
                    at: now,
                    rows: n,
                    verification: None,
                    switched,
                });
            }
            drop(this);
            let completion = QwenCompletion {
                executor: slf.clone().unbind(),
                stage,
                params,
                scores,
                positions: positions.iter().map(|p| p + 1).collect(),
                output: None,
            };
            return Ok(Py::new(py, completion)?.into_any());
        }
        let anchors = match previous {
            Previous::None => {
                let anchors: Vec<i64> = p.token_ids.iter().take(n).map(|&t| i64::from(t)).collect();
                let drafts = this.carried_drafts(p, &rows, &anchors);
                Anchors::Host(anchors, drafts)
            }
            Previous::Plain(completion) => {
                // after a plain step: its samples (the host waits for them), no drafts
                drop(this);
                let output = completion.borrow_mut(py).result(py)?;
                this = slf.borrow_mut();
                let ids = output.bind(py).getattr("token_ids")?;
                let ids = ids.cast::<PyArray1<u32>>()?.readonly();
                let anchors: Vec<i64> = ids.as_slice()?.iter().map(|&t| i64::from(t)).collect();
                let k = this.draft.as_ref().expect("a drafter").k;
                Anchors::Host(anchors, vec![vec![0; k]; n])
            }
            Previous::Verify(verification) => Anchors::Device(verification),
        };
        let mut verification = this.verify(py, p, rows, positions, anchors, scores)?;
        verification.executor = Some(slf.clone().unbind());
        for sequence in &p.sequence_ids {
            this.draft
                .as_mut()
                .expect("a drafter")
                .carried
                .remove(sequence);
        }
        drop(this);
        let verification = Py::new(py, verification)?;
        let mut this = slf.borrow_mut();
        let draft = this.draft.as_mut().expect("a drafter");
        draft.open = Some(verification.clone_ref(py));
        draft.timed = Some(Timed {
            verify: true,
            bucket,
            at: now,
            rows: n,
            verification: Some(verification.clone_ref(py)),
            switched,
        });
        Ok(verification.into_any())
    }
}

/// An enqueued plain step: `result()` waits for it and gives its ForwardOutput; `feedback` continues it.
#[pyclass(module = "resin._resin")]
pub(crate) struct QwenCompletion {
    executor: Py<QwenExecutor>,
    stage: usize,
    params: Vec<[f32; PARAMS]>,
    scores: bool,
    positions: Vec<i64>,
    output: Option<Py<ForwardOutput>>,
}

#[pymethods]
impl QwenCompletion {
    fn result(&mut self, py: Python<'_>) -> PyResult<Py<ForwardOutput>> {
        if let Some(output) = &self.output {
            return Ok(output.clone_ref(py));
        }
        let output = {
            let mut executor = self.executor.borrow_mut(py);
            executor.wait_stage(py, self.stage, 0)?;
            executor.plain_output(py, self.stage, &self.params, self.scores)?
        };
        let output = Py::new(py, output)?;
        self.output = Some(output.clone_ref(py));
        Ok(output)
    }

    #[getter]
    fn feedback(slf: Bound<'_, Self>) -> PyResult<QwenFeedback> {
        let positions = slf.borrow().positions.clone();
        Ok(QwenFeedback {
            positions,
            completion: slf.unbind(),
        })
    }
}

/// A plain step's continuation: the rows' next positions; `tokens()` waits for the step's samples.
#[pyclass(module = "resin._resin", frozen)]
pub(crate) struct QwenFeedback {
    positions: Vec<i64>,
    completion: Py<QwenCompletion>,
}

#[pymethods]
impl QwenFeedback {
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.positions)
    }

    fn tokens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let output = self.completion.borrow_mut(py).result(py)?;
        let ids = output.bind(py).getattr("token_ids")?;
        let ids = ids.cast::<PyArray1<u32>>()?.readonly();
        let ids: Vec<i64> = ids.as_slice()?.iter().map(|&t| i64::from(t)).collect();
        Ok(PyArray1::from_vec(py, ids))
    }
}

/// An enqueued verify step, and its continuation's feedback (`qwen3_5.runner._Verification`): `finish` (from the
/// continuation's enqueue, or from `result`) waits for the verify pass, accepts the drafts on the host and
/// checks the device's acceptance; then `positions` and `anchors` are each row's next anchor position and
/// token, and the block step's drafts are read at the next finish.
#[pyclass(module = "resin._resin")]
pub(crate) struct QwenVerification {
    /// the executor that launched it (None: a synchronous step's, finished at once)
    executor: Option<Py<QwenExecutor>>,
    stage: usize,
    rows: usize,
    bucket: usize,
    width: usize,
    next_width: usize,
    start: Vec<i64>,
    params: Vec<[f32; PARAMS]>,
    scores: bool,
    drafts: Option<Drafts>,
    sequences: Vec<u64>,
    done: Option<Done>,
    continued: bool,
    /// the block step's drafts (rows x K) once read
    next_read: Option<Vec<Vec<i64>>>,
    /// an MTP verify step's: what its commit and chain take
    mtp: Option<MtpVerification>,
}

/// What an MTP verify step's finish needs: its rows (their state and MTP KV blocks), each row's sampling request
/// (the chain's sampler inputs), and after the finish each row's first next draft.
struct MtpVerification {
    rows: RowTables,
    samples: Vec<sampling::Sample>,
    first: Vec<i64>,
}

impl QwenVerification {
    fn executor(this: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<QwenExecutor>> {
        this.borrow()
            .executor
            .as_ref()
            .map(|e| e.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err("a synchronous step's verification"))
    }

    /// Finish `this` with the executor (borrowed by the caller).
    fn finish_of(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut QwenExecutor,
    ) -> PyResult<()> {
        if this.borrow().done.is_some() {
            return Ok(());
        }
        // the drafts it verified: on the host, or the previous step's block step's
        let drafts = this
            .borrow_mut()
            .drafts
            .take()
            .expect("drafts before the finish");
        let (rows, width) = {
            let v = this.borrow();
            (v.rows, v.width)
        };
        let drafts: Vec<Vec<i64>> = match drafts {
            Drafts::Host(d) => d,
            Drafts::Previous(previous) => {
                let k = executor.draft.as_ref().expect("a drafter").k;
                Self::drafts_of(previous.bind(py), py, executor, k)?
                    .into_iter()
                    .map(|d| d[..width - 1].to_vec())
                    .collect()
            }
        };
        let v = this.borrow();
        executor.wait_stage(py, v.stage, 0)?;
        let (samples, lp, ids, values) = executor.staged(v.stage, v.bucket * width, v.scores);
        let samples: Vec<i64> = samples[..rows * width]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let flat: Vec<i64> = drafts.iter().flatten().copied().collect();
        let (accepted, emitted) = speculative::accept(&flat, &samples, rows, width, None);
        let anchors: Vec<i64> = emitted
            .iter()
            .map(|row| *row.last().expect("a token"))
            .collect();
        let mut first = Vec::new();
        if let Some(verify) = &v.mtp {
            // the host accepted: the commit and the draft chain from each row's accepted column (the MTP layer's
            // draft there is the first of the next step's)
            let staged = executor.mtp_tokens(v.stage, v.bucket * width);
            first = (0..rows)
                .map(|b| i64::from(staged[b * width + accepted[b] as usize]))
                .collect();
            executor.chain(py, verify, v.stage, &v.start, &accepted, &anchors, &first)?;
        } else {
            let staging = &executor.staging;
            let on_device: Vec<i32> = staging
                .accepted
                .as_ref()
                .expect("a drafter's staging")
                .words(v.stage * staging.rows * 4, rows);
            if on_device
                .iter()
                .zip(&accepted)
                .any(|(&d, &h)| i64::from(d) != h)
            {
                return Err(post_accept(
                    py,
                    format!("device acceptance {on_device:?} != host {accepted:?}"),
                ));
            }
        }
        let counts: Vec<usize> = emitted.iter().map(Vec::len).collect();
        let token_ids: Vec<u32> = emitted.iter().flatten().map(|&t| t as u32).collect();
        let mut indptr = vec![0u32];
        for &c in &counts {
            indptr.push(indptr.last().unwrap() + c as u32);
        }
        let (scores, top) = if v.scores {
            // the scored columns of each row: its accepted drafts' and the bonus token's
            let (mut l, mut ti, mut tl) = (Vec::new(), Vec::new(), Vec::new());
            for (b, &count) in counts.iter().enumerate() {
                for j in 0..count {
                    let column = b * width + j;
                    l.push(lp[column]);
                    ti.extend_from_slice(&ids[column * TOP..(column + 1) * TOP]);
                    tl.extend_from_slice(&values[column * TOP..(column + 1) * TOP]);
                }
            }
            let (s, t) = score_columns(&v.params, &counts, &l, &ti, &tl);
            (Some(s), t)
        } else {
            (None, None)
        };
        let output = Py::new(
            py,
            ForwardOutput::native(py, token_ids, indptr, scores, top)?,
        )?;
        let positions: Vec<i64> = v
            .start
            .iter()
            .zip(&accepted)
            .map(|(p, a)| p + a + 1)
            .collect();
        let committed = accepted.iter().map(|a| (a + 1) as u64).sum();
        let draft = executor.draft.as_mut().expect("a drafter");
        if let Side::Blocks(blocks) = &mut draft.side {
            blocks.last_accepted = accepted.clone();
            blocks.last_width = width;
        }
        draft.stats[0] += rows as u64;
        draft.stats[1] += (rows * (width - 1)) as u64;
        draft.stats[2] += accepted.iter().sum::<i64>() as u64;
        draft.stats[3] += counts.iter().sum::<usize>() as u64;
        drop(v);
        if let Some(verify) = &mut this.borrow_mut().mtp {
            verify.first = first;
        }
        this.borrow_mut().done = Some(Done {
            accepted,
            positions,
            anchors,
            committed,
            output,
        });
        Ok(())
    }

    /// The block step's drafts after this step (rows x K; waits for its event 1).
    fn drafts_of(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut QwenExecutor,
        k: usize,
    ) -> PyResult<Vec<Vec<i64>>> {
        if let Some(drafts) = &this.borrow().next_read {
            return Ok(drafts.clone());
        }
        let (stage, rows, width) = {
            let v = this.borrow();
            (v.stage, v.rows, v.next_width)
        };
        let first = this.borrow().mtp.as_ref().map(|m| m.first.clone());
        let drafts = match first {
            // the MTP layer's: its first draft from the host, the chain's after its event 1
            Some(first) => {
                if first.len() != rows {
                    return Err(PyValueError::new_err("the verification is not finished"));
                }
                let mut drafts: Vec<Vec<i64>> = first
                    .iter()
                    .map(|&f| {
                        let mut d = vec![0i64; k];
                        d[0] = f;
                        d
                    })
                    .collect();
                if k > 1 {
                    executor.wait_stage(py, stage, 1)?;
                    let chain = executor.chain_drafts(stage, k);
                    let srows = executor.staging.rows;
                    for (b, d) in drafts.iter_mut().enumerate() {
                        for (j, slot) in d.iter_mut().enumerate().skip(1) {
                            *slot = i64::from(chain[j * srows + b]);
                        }
                    }
                }
                drafts
            }
            None => {
                let next = executor.next_tokens(py, stage, rows, width)?;
                QwenExecutor::drafts_of(&next, rows, width, k)
            }
        };
        this.borrow_mut().next_read = Some(drafts.clone());
        Ok(drafts)
    }

    /// `result` with the executor borrowed by the caller: finish, and when no step continues this one, carry
    /// its drafts by sequence.
    fn result_of(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut QwenExecutor,
    ) -> PyResult<Py<ForwardOutput>> {
        Self::finish_of(this, py, executor)?;
        if !this.borrow().continued {
            this.borrow_mut().continued = true;
            let k = executor.draft.as_ref().expect("a drafter").k;
            let drafts = Self::drafts_of(this, py, executor, k)?;
            let v = this.borrow();
            let done = v.done.as_ref().expect("finished");
            let draft = executor.draft.as_mut().expect("a drafter");
            for (r, &sequence) in v.sequences.iter().enumerate() {
                draft.carried.insert(
                    sequence,
                    (done.positions[r], done.anchors[r], drafts[r].clone()),
                );
            }
        }
        let v = this.borrow();
        Ok(v.done.as_ref().expect("finished").output.clone_ref(py))
    }
}

#[pymethods]
impl QwenVerification {
    fn result(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<ForwardOutput>> {
        let executor = Self::executor(&slf, py)?;
        let mut executor = executor.borrow_mut(py);
        Self::result_of(&slf, py, &mut executor)
    }

    fn finish(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let executor = Self::executor(&slf, py)?;
        let mut executor = executor.borrow_mut(py);
        Self::finish_of(&slf, py, &mut executor)
    }

    #[getter]
    fn feedback(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[getter]
    fn committed(&self) -> u64 {
        self.done.as_ref().map_or(0, |d| d.committed)
    }

    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let done = self
            .done
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("not finished"))?;
        Ok(PyArray1::from_slice(py, &done.positions))
    }

    #[getter]
    fn accepted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let done = self
            .done
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("not finished"))?;
        Ok(PyArray1::from_slice(py, &done.accepted))
    }
}

impl Replay for QwenExecutor {
    fn follower_group(&self, py: Python<'_>) -> Option<Py<GaudiRanks>> {
        self.ranks
            .as_ref()
            .filter(|r| !r.driver)
            .map(|r| r.group.clone_ref(py))
    }

    fn set_speculating(&mut self, decision: Option<bool>) {
        self.forced_speculating = decision;
    }

    fn replay_forward(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow_mut().forward_step(py, plan)
    }

    fn replay_enqueue(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Self::enqueue_step(slf, py, plan, feedback)
    }

    fn replay_release(slf: &Bound<'_, Self>, py: Python<'_>, ids: &[u64]) -> PyResult<()> {
        let mut this = slf.borrow_mut();
        this.settle(py, None).map(|()| this.release_ids(ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_chunks_match_the_reference() {
        let buckets = [128, 256, 512, 1024, 2048, 4096];
        // qwen3_5.model.prefill_chunks
        assert_eq!(prefill_chunks(300, &buckets), vec![(256, 256), (44, 128)]);
        assert_eq!(prefill_chunks(4096, &buckets), vec![(4096, 4096)]);
        assert_eq!(
            prefill_chunks(5000, &buckets),
            vec![(4096, 4096), (904, 1024)]
        );
        assert_eq!(prefill_chunks(37, &buckets), vec![(37, 128)]);
    }

    #[test]
    fn the_policy_fits_the_mtp_layer_s_steps() {
        // DraftPolicy(2, False).rate(("verify", 1), (3,)): 2.2 tokens a row at mtp_step_ms(3, 1)
        let mtp = Policy::new(2, "auto", true).unwrap();
        assert!((mtp.rate(true, 1, &[3]) - 2.2 / (18.0 + 0.92 * 3.0)).abs() < 1e-12);
        // DraftPolicy(2, True): the drafter's step_ms(3, 1)
        let drafter = Policy::new(2, "auto", false).unwrap();
        assert!((drafter.rate(true, 1, &[3]) - 2.2 / (22.3 + 0.71 * 3.0)).abs() < 1e-12);
        assert!((mtp.rate(false, 8, &[3]) - 8.0 / (16.0 + 1.07 * 8.0)).abs() < 1e-12);
    }

    #[test]
    fn the_policy_follows_its_modes_and_rates() {
        let widths = [4, 8, 16];
        let mut always = Policy::new(15, "always", false).unwrap();
        assert!(always.speculate(8, &widths) && !always.speculate(33, &widths));
        let mut never = Policy::new(15, "never", false).unwrap();
        assert!(!never.speculate(1, &widths));
        assert!(Policy::new(15, "sometimes", false).is_err());
        let mut auto = Policy::new(15, "auto", false).unwrap();
        // measured: plain steps far faster than verify steps -> plain, then a probe after PROBE_FIRST
        for _ in 0..30 {
            auto.record(false, 8, 8.0, 20.0);
            auto.record(true, 8, 8.0, 100.0);
        }
        assert!(!auto.speculate(8, &widths));
        // DraftPolicy's decisions after the same records (qwen3_5.runner): 30 plain, then a probe of 8 steps
        // (and the plain rate still wins, so the probe's steps stay verify steps until it ends)
        let decisions: String = (0..40)
            .map(|_| if auto.speculate(8, &widths) { '1' } else { '0' })
            .collect();
        assert_eq!(decisions, "0000000000000000000000000000001111111111");
        // a step far longer than its kind's average is not counted
        let before = auto.stats[&(false, 8)];
        auto.record(false, 8, 8.0, 1000.0);
        assert_eq!(auto.stats[&(false, 8)], before);
    }
}
