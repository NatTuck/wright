//! The native step executor of DeepSeek-V4.1 on Gaudi (cargo feature `gaudi`).
//!
//! Python builds the model at startup: it loads the weights, compiles the recipes and builds, per step shape,
//! the step program (`DeepSeekV41Gaudi._step_program`: embedding, layers, injection, head, owner store) and an
//! I/O program (`deepseek_v41.native`: the table image's upload, the device tokens' copy, the sampler, the
//! outputs' download, bounded replay's compaction). Each step then runs here without Python: the tables go into
//! a pinned host copy of the table image (`resin_gaudi::ds41`), one copy uploads them, the programs make the
//! launches, and the samples come back through pinned staging buffers behind events. The runner hands the Engine
//! this executor's `forward`, `enqueue` and `release`.
//!
//! Steps: prefill chunks (each row's in order, bounded replay's two parts, prompt logprobs scored on the host),
//! plain decode steps (synchronous, or enqueued: a continuation reads its tokens from the previous step's samples
//! on the device) and DSpark steps (drafts, then a verify step; acceptance on the host). Each step records its
//! committed tokens in the Engram context (`Ds41Context`), which the next steps' tables read. With a rank group,
//! rank 0 sends each runner call to the other ranks (`gaudi_ranks::serve`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use numpy::{PyArray2, PyArrayMethods, PyReadwriteArray1};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use resin_gaudi::sampling::{self, PARAMS, TOP_MAX as TOP};
use resin_gaudi::{ds41, speculative, step};

use crate::cache::CacheLayout;
use crate::gaudi::{Ds41Context, Ds41Fill, GaudiGlue, GaudiProgram, item};
use crate::gaudi_exec::{
    Images, Pinned, ScoreRequest, Scored, Timings, check_paged_plan, post_accept, prompt_columns,
    score_columns, score_position, score_requests, since,
};
use crate::gaudi_gate::CardGate;
use crate::gaudi_ranks::{
    GaudiRanks, OP_CLOSE, OP_ENQUEUE, OP_FORWARD, OP_RELEASE, Ranked, Replay,
};
use crate::output::ForwardOutput;
use crate::{BatchPhase, BatchPlan};

// the registers of a step's programs: the step program's (`model._R_*`), then the I/O program's
const R_SPARSE: usize = 0;
const R_CAND: usize = 1;
const R_IXC: usize = 2;
const R_RADIX: [usize; 2] = [3, 4];
const R_IX: [usize; 2] = [5, 6];
const R_COLUMN: usize = 7;
const R_DENSE: [usize; 2] = [8, 9];
const R_PF: [usize; 2] = [10, 11];
/// which host copy of the table image the upload reads (0 or 1)
pub(crate) const R_IMAGE: usize = 12;
/// bytes of the table image to upload
pub(crate) const R_BYTES: usize = 13;
/// which staging set the samples go to (0 or 1)
pub(crate) const R_STAGE: usize = 14;
/// bytes of the previous step's sampled tokens that a continuation copies into its tokens
pub(crate) const R_TOKBYTES: usize = 15;
/// bounded replay: the columns before the kept ones, and the kept columns
pub(crate) const R_SKIP: usize = 16;
pub(crate) const R_KEPT: usize = 17;
/// prompt logprobs: the first scored column of a download, and its bytes per rank
pub(crate) const R_SCORE_COL: usize = 18;
pub(crate) const R_SCORE_BYTES: usize = 19;
pub(crate) const REGISTERS: usize = 20;

/// The I/O program's sections.
const IO_SECTIONS: [&str; 8] = [
    "upload",
    "tokens",
    "sample",
    "sample_last",
    "out",
    "draft_out",
    "sample_at",
    "scores",
];
const IO_UPLOAD: usize = 0;
const IO_TOKENS: usize = 1;
const IO_SAMPLE: usize = 2;
const IO_SAMPLE_LAST: usize = 3;
const IO_OUT: usize = 4;
/// a draft shape's drafts (`R_TOKBYTES` bytes of them) to the pinned drafts buffer, then the draft event
const IO_DRAFT_OUT: usize = 5;
/// a prefill chunk's sampler over its column `last` of every column's logits (a head of every column)
const IO_SAMPLE_AT: usize = 6;
/// prompt logprobs: `R_SCORE_BYTES` of each rank's logits from column `R_SCORE_COL` to the pinned score buffer,
/// then the score event
const IO_SCORES: usize = 7;

/// (pinned staging sets, stride, first event slot, (tok, lp, top_idx, top_lp, bytes) offsets)
type StagingArgs<'py> = (
    PyReadwriteArray1<'py, u8>,
    usize,
    i32,
    (usize, usize, usize, usize, usize),
);

/// One step shape's programs and their sections.
struct Shaped {
    step: Arc<step::Program>,
    io: Arc<step::Program>,
    io_sections: [Option<usize>; IO_SECTIONS.len()],
    /// bounded replay: the compaction section of each kept bucket
    compact: HashMap<usize, usize>,
}

/// Prompt logprobs: the pinned buffer the scored columns' logits come back to (each rank's part `stride` bytes
/// apart, `columns` columns of `vocab` logits), and its event.
struct ScoreSide {
    host: Pinned,
    stride: usize,
    columns: usize,
    vocab: usize,
    ranks: usize,
    event: i32,
}

/// DSpark: the draft programs (by draft step bucket), the pinned buffer the drafts come back to, its event, the
/// block's settings and the acceptance counters.
struct DraftSide {
    programs: HashMap<usize, Arc<step::Program>>,
    host: Pinned,
    event: i32,
    settings: ds41::Draft,
    stats: speculative::Stats,
}

/// The cache geometry of the rows' block addresses (`CacheGeometry`).
#[derive(Clone, Copy)]
struct Geometry {
    slab_rows: i64,
    window_block_rows: i64,
    history_block_rows: i64,
    window_per_slab: i64,
}

/// The sampler's outputs in a staging set: byte offsets of the tokens, logprobs, top ids and top logprobs.
#[derive(Clone, Copy)]
struct Sampled {
    tok: usize,
    lp: usize,
    top_idx: usize,
    top_lp: usize,
}

/// A step's rows, flat (`ds41::Rows` borrows them).
#[derive(Default)]
struct StepRows {
    column: Vec<i64>,
    pos0: Vec<i64>,
    real: Vec<i64>,
    window_first: Vec<i64>,
    window_indptr: Vec<i64>,
    window: Vec<i64>,
    window_ids: Vec<i64>,
    history_indptr: Vec<i64>,
    history: Vec<i64>,
    before_indptr: Vec<i64>,
    before: Vec<i64>,
}

impl StepRows {
    fn new() -> Self {
        Self {
            window_indptr: vec![0],
            history_indptr: vec![0],
            before_indptr: vec![0],
            ..Self::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        column: i64,
        pos0: i64,
        real: i64,
        first: i64,
        window: &[i64],
        ids: &[i64],
        history: &[i64],
        before: &[i64],
    ) {
        self.column.push(column);
        self.pos0.push(pos0);
        self.real.push(real);
        self.window_first.push(first);
        self.window.extend_from_slice(window);
        self.window_ids.extend_from_slice(ids);
        self.window_indptr.push(self.window.len() as i64);
        self.history.extend_from_slice(history);
        self.history_indptr.push(self.history.len() as i64);
        self.before.extend_from_slice(before);
        self.before_indptr.push(self.before.len() as i64);
    }

    fn view(&self) -> ds41::Rows<'_> {
        ds41::Rows {
            column: &self.column,
            pos0: &self.pos0,
            real: &self.real,
            window_first: &self.window_first,
            window_indptr: &self.window_indptr,
            window: &self.window,
            window_ids: &self.window_ids,
            history_indptr: &self.history_indptr,
            history: &self.history,
            before_indptr: &self.before_indptr,
            before: &self.before,
        }
    }
}

/// One row's cache blocks from the plan: WINDOW rows and ids (from logical block `first`), HISTORY rows.
struct Blocks {
    first: i64,
    window: Vec<i64>,
    ids: Vec<i64>,
    history: Vec<i64>,
}

/// A step's outputs read from a staging set: per column its token, logprob and top list.
struct Outputs {
    tokens: Vec<i32>,
    lp: Vec<f32>,
    top_ids: Vec<i32>,
    top_lp: Vec<f32>,
}

impl Outputs {
    fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            lp: Vec::new(),
            top_ids: Vec::new(),
            top_lp: Vec::new(),
        }
    }

    fn extend(&mut self, other: Outputs) {
        self.tokens.extend(other.tokens);
        self.lp.extend(other.lp);
        self.top_ids.extend(other.top_ids);
        self.top_lp.extend(other.top_lp);
    }

    /// `_unsampled`: a row whose chunk only advances the cache
    fn unsampled() -> Self {
        Self {
            tokens: vec![0],
            lp: vec![0.0],
            top_ids: vec![0; TOP],
            top_lp: vec![0.0; TOP],
        }
    }
}

#[pyclass(module = "resin._resin")]
pub(crate) struct Ds41Executor {
    glue: Py<GaudiGlue>,
    model: ds41::Model,
    tables: HashMap<(usize, bool), ds41::Tables>,
    cs: [Py<PyArray2<f32>>; 2],
    context: Py<Ds41Context>,
    shapes: HashMap<(usize, bool), Shaped>,
    layout: Py<CacheLayout>,
    geometry: Geometry,
    image: Images,
    staging: Pinned,
    staging_stride: usize,
    stage_events: i32,
    stage_next: usize,
    /// the completion whose samples each staging set holds, until they are read
    stage_owner: [Option<u64>; 2],
    sample_host: Pinned,
    /// the sampler table's `last` field (byte offset in it)
    sample_last: usize,
    sampled: Sampled,
    vocab: u64,
    n_layers: usize,
    /// bounded replay: the cut layer and the kept span (None: off)
    cut: Option<usize>,
    replay: i64,
    prefill_buckets: Vec<usize>,
    decode_buckets: Vec<usize>,
    /// a prefill chunk's most tokens
    tmax: usize,
    /// tensor parallel: a head of one column all-gathers that column alone
    one_column: bool,
    max_batch_size: usize,
    max_seq_len: i64,
    /// DSpark: the drafts a speculative step verifies, and the most rows it takes
    draft_tokens: i64,
    dspark_rows: usize,
    /// which decode steps draft: the runner's switch (`drafting`), and on a follower the driver's decision
    drafting: ds41::Drafting,
    /// the Engram hash's tokens before a position (max n-gram - 1)
    engram_n: usize,
    spin_ns: i64,
    registers: [u64; REGISTERS],
    /// enqueued steps whose samples are not yet recorded in the context, oldest first
    inflight: Vec<Py<Ds41Completion>>,
    next_id: u64,
    timings: Timings,
    native_steps: u64,
    draft: Option<DraftSide>,
    /// the last speculative step, until its samples are read
    verification: Option<Py<Ds41Verification>>,
    /// enqueued steps that continued the previous one (`DeepSeekV41GaudiRunner.continuations`)
    continuations: u64,
    /// a tensor-parallel model's rank group (`attach_ranks`)
    ranks: Option<Ranked>,
    scoring: Option<ScoreSide>,
    /// a layout's card gate, which a replica on this rank's card shares (`attach_gate`)
    card_gate: Option<Py<CardGate>>,
}

/// What a decode or prefill pass of a shape runs after its upload.
struct Pass {
    shape: (usize, bool),
    /// the step program's sections (`DeepSeekV41Gaudi._sections`)
    sections: Vec<usize>,
}

impl Ds41Executor {
    fn bucket(&self, n: usize, prefill: bool) -> PyResult<usize> {
        let buckets = if prefill {
            &self.prefill_buckets
        } else {
            &self.decode_buckets
        };
        buckets.iter().copied().find(|&b| n <= b).ok_or_else(|| {
            let kind = if prefill { "prefill" } else { "decode" };
            PyValueError::new_err(format!("{n} columns exceed the {kind} buckets"))
        })
    }

    fn shape(&self, t: usize, prefill: bool) -> PyResult<(&Shaped, &ds41::Tables)> {
        match (
            self.shapes.get(&(t, prefill)),
            self.tables.get(&(t, prefill)),
        ) {
            (Some(shaped), Some(tables)) => Ok((shaped, tables)),
            _ => Err(PyValueError::new_err(format!(
                "no step shape of {t} columns (prefill {prefill})"
            ))),
        }
    }

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

    /// `plan_params`: each row's sampler inputs, its uniform at offset (context + query length) x 32.
    fn params(&self, plan: &BatchPlan) -> PyResult<Vec<[f32; PARAMS]>> {
        (0..plan.sequence_ids.len())
            .map(|r| {
                let query = u64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
                let offset = (u64::from(plan.context_lengths[r]) + query) * sampling::OFFSET_STRIDE;
                Ok(sampling::row_params(
                    &self.sample(plan, r)?,
                    self.vocab,
                    offset,
                ))
            })
            .collect()
    }

    /// The sampler inputs of the `width` columns of each row, row r's first column at position `starts[r]`
    /// (`sampling::column_params`): from the executor's positions, not the plan's context lengths, which a
    /// continuation's plan only bounds.
    fn params_at(
        &self,
        plan: &BatchPlan,
        starts: &[i64],
        width: usize,
    ) -> PyResult<Vec<[f32; PARAMS]>> {
        let mut out = Vec::with_capacity(starts.len() * width);
        for (r, &start) in starts.iter().enumerate() {
            out.extend(sampling::column_params(
                &self.sample(plan, r)?,
                self.vocab,
                start,
                width,
            ));
        }
        Ok(out)
    }

    /// Row `r`'s cache blocks (`DeepSeekV41GaudiRunner._row`).
    fn blocks(&self, plan: &BatchPlan, r: usize) -> PyResult<Blocks> {
        let group = |index: usize| {
            plan.cache_groups
                .get(index)
                .map(|group| group.inner_native())
                .ok_or_else(|| {
                    PyValueError::new_err(format!("the plan has no cache group {index}"))
                })
        };
        let (win, hist) = (group(0)?, group(1)?);
        let g = self.geometry;
        let (lo, hi) = (
            win.block_indptr[r] as usize,
            win.block_indptr[r + 1] as usize,
        );
        let mut window = Vec::with_capacity(hi - lo);
        let mut ids = Vec::with_capacity(hi - lo);
        for i in lo..hi {
            let (slab, slot) = (
                i64::from(win.block_slab_ids[i]),
                i64::from(win.block_slotnums[i]),
            );
            window.push(slab * g.slab_rows + slot * g.window_block_rows);
            ids.push(slab * g.window_per_slab + slot);
        }
        let (lo, hi) = (
            hist.block_indptr[r] as usize,
            hist.block_indptr[r + 1] as usize,
        );
        let history = (lo..hi)
            .map(|i| {
                i64::from(hist.block_slab_ids[i]) * g.slab_rows
                    + i64::from(hist.block_slotnums[i]) * g.history_block_rows
            })
            .collect();
        Ok(Blocks {
            first: i64::from(win.block_table_start_indices[r]),
            window,
            ids,
            history,
        })
    }

    fn before(
        &self,
        py: Python<'_>,
        sequence: u64,
        pos0: i64,
        blocks: &Blocks,
    ) -> PyResult<Vec<i64>> {
        self.context
            .borrow_mut(py)
            .context
            .before(sequence, pos0, blocks.first, &blocks.ids)
            .map_err(PyRuntimeError::new_err)
    }

    fn seen(&self, py: Python<'_>, sequence: u64, pos0: i64, tokens: &[i64]) -> PyResult<()> {
        self.context
            .borrow_mut(py)
            .context
            .seen(sequence, pos0, tokens)
            .map_err(PyRuntimeError::new_err)
    }

    /// The next host copy of the table image, once its last upload is done.
    fn take_image(&mut self, py: Python<'_>) -> PyResult<usize> {
        let k = self.image.next;
        self.image.next ^= 1;
        if self.image.used[k] {
            let glue = self.glue.get();
            let (event, spin) = (self.image.events + k as i32, self.spin_ns);
            let start = Instant::now();
            py.detach(|| glue.glue.wait(event, spin))
                .map_err(|error| post_accept(py, error))?;
            self.timings.image_wait += since(start);
        }
        Ok(k)
    }

    /// Fill host copy `k` with a step's tables (`with_tokens`: its tokens too, None: the device's); returns the
    /// fill's counts.
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &mut self,
        py: Python<'_>,
        k: usize,
        shape: (usize, bool),
        rows: &StepRows,
        positions: &[i64],
        tokens: Option<&[i64]>,
        floor: Option<i64>,
    ) -> PyResult<ds41::Counts> {
        self.fill_with(py, k, shape, rows, positions, tokens, true, floor)
    }

    /// `fill` (`with_tokens`) or the positions' tables only (`prepare`).
    #[allow(clippy::too_many_arguments)]
    fn fill_with(
        &mut self,
        py: Python<'_>,
        k: usize,
        shape: (usize, bool),
        rows: &StepRows,
        positions: &[i64],
        tokens: Option<&[i64]>,
        with_tokens: bool,
        floor: Option<i64>,
    ) -> PyResult<ds41::Counts> {
        let tables = self
            .tables
            .get(&shape)
            .ok_or_else(|| PyValueError::new_err(format!("no tables of shape {shape:?}")))?;
        let start = Instant::now();
        let cs0 = self.cs[0].bind(py).readonly();
        let cs1 = self.cs[1].bind(py).readonly();
        let cs_all = [cs0.as_slice()?, cs1.as_slice()?];
        // SAFETY: host copy k's last upload is done (take_image), and no device copy reads it until this step's.
        let host = unsafe {
            self.image
                .host
                .bytes_mut(k * self.image.stride, tables.nbytes)
        };
        let rows = rows.view();
        let model = &self.model;
        let counts = py
            .detach(|| {
                ds41::fill(
                    model,
                    tables,
                    host,
                    &rows,
                    positions,
                    tokens,
                    with_tokens,
                    floor,
                    cs_all,
                )
            })
            .map_err(PyValueError::new_err)?;
        self.timings.fill += since(start);
        Ok(counts)
    }

    /// The step program's registers of a fill's counts (`_step_registers`), and the I/O registers.
    fn set_counts(&mut self, counts: &ds41::Counts) {
        let r = &mut self.registers;
        r[R_SPARSE] = u64::from(counts.sparse);
        r[R_CAND] = u64::from(counts.cand);
        r[R_IXC] = if counts.sparse { counts.ixc as u64 } else { 0 };
        for ratio in 0..2 {
            r[R_RADIX[ratio]] = u64::from(counts.radix[ratio]);
            r[R_IX[ratio]] = counts.ix[ratio] as u64;
            r[R_PF[ratio]] = u64::from(counts.pf[ratio]);
            r[R_DENSE[ratio]] = u64::from(counts.dense[ratio]);
        }
    }

    /// The sampler's table: the rows' parameters, then `last` (`_sample_launch`).
    fn write_sample_table(&mut self, params: &[[f32; PARAMS]], last: i64) {
        // SAFETY: the pinned table is only read by a submit's upload_ring, which copies it at once; no submit runs
        // while the executor writes.
        let table = unsafe { self.sample_host.bytes_mut(0, self.sample_host.bytes) };
        table.fill(0);
        for (r, row) in params.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let at = 4 * (r * PARAMS + j);
                table[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        let at = self.sample_last;
        table[at..at + 4].copy_from_slice(&(last as i32).to_le_bytes());
    }

    fn submit(
        &mut self,
        py: Python<'_>,
        steps: &[(Arc<step::Program>, usize)],
        name: &str,
    ) -> PyResult<()> {
        let glue = self.glue.get();
        let registers = self.registers;
        let start = Instant::now();
        py.detach(|| {
            let steps: Vec<(&step::Program, usize)> = steps
                .iter()
                .map(|(program, section)| (program.as_ref(), *section))
                .collect();
            // SAFETY: the Python builder bound device allocations, the pinned host images, staging buffers and
            // sampler table that this executor keeps, and recipes and event slots of the loaded runtime.
            let mut stream = unsafe { step::CardStream::new(&glue.glue) };
            step::run_sequence(&steps, &registers, &mut stream)
        })
        .map_err(|error| post_accept(py, format!("DeepSeek step {name}: {}", error.message)))?;
        self.timings.submit += since(start);
        self.native_steps += 1;
        Ok(())
    }

    fn io(&self, shape: (usize, bool), section: usize) -> PyResult<(Arc<step::Program>, usize)> {
        let shaped = &self.shapes[&shape];
        let at = shaped.io_sections[section].ok_or_else(|| {
            PyValueError::new_err(format!(
                "shape {shape:?} has no {} section",
                IO_SECTIONS[section]
            ))
        })?;
        Ok((Arc::clone(&shaped.io), at))
    }

    /// The step program's sections of layers `first` .. `last` - 1, with the embedding when `first` is 0 and the
    /// injection and head after them when `last` is the end (`_sections`); `logits`: None (no head), Some(None)
    /// (every column), Some(Some(c)) (column c, all-gathered alone with ranks).
    fn sections(&self, first: usize, last: usize, logits: Option<Option<usize>>) -> Vec<usize> {
        let n = self.n_layers;
        let mut out = Vec::new();
        if first == 0 {
            out.push(0);
        }
        out.extend(1 + first..1 + last);
        if last == n {
            out.push(n + 1);
            match logits {
                None => {}
                Some(Some(_)) if self.one_column => out.push(n + 3),
                Some(_) => out.push(n + 2),
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    /// Queue a pass: the image upload (host copy `k`, `bytes`), the device tokens' copy (`device_tokens` of
    /// them), the step program's `sections`, and after them `then` (sampler, outputs).
    fn pass(
        &mut self,
        py: Python<'_>,
        k: usize,
        pass: &Pass,
        device_tokens: usize,
        before: &[(Arc<step::Program>, usize)],
        then: &[(Arc<step::Program>, usize)],
        name: &str,
    ) -> PyResult<()> {
        let (shaped, tables) = self.shape(pass.shape.0, pass.shape.1)?;
        let (step_program, nbytes) = (Arc::clone(&shaped.step), tables.nbytes);
        self.registers[R_IMAGE] = k as u64;
        self.registers[R_BYTES] = nbytes as u64;
        self.registers[R_TOKBYTES] = 4 * device_tokens as u64;
        let mut steps = vec![self.io(pass.shape, IO_UPLOAD)?];
        if device_tokens > 0 {
            steps.push(self.io(pass.shape, IO_TOKENS)?);
        }
        steps.extend(before.iter().map(|(p, s)| (Arc::clone(p), *s)));
        steps.extend(
            pass.sections
                .iter()
                .map(|&s| (Arc::clone(&step_program), s)),
        );
        steps.extend(then.iter().map(|(p, s)| (Arc::clone(p), *s)));
        self.image.used[k] = true;
        self.submit(py, &steps, name)
    }

    /// The next staging set, once the completion that holds its samples has read them.
    fn take_stage(&mut self, py: Python<'_>) -> PyResult<usize> {
        let k = self.stage_next;
        self.stage_next ^= 1;
        if let Some(owner) = self.stage_owner[k].take() {
            let open: Vec<Py<Ds41Completion>> = self
                .inflight
                .iter()
                .filter(|c| c.borrow(py).id == owner)
                .map(|c| c.clone_ref(py))
                .collect();
            for completion in open {
                Ds41Completion::finish(&completion.bind(py).clone(), py, self)?;
            }
            if let Some(verification) = self.verification.as_ref().map(|v| v.clone_ref(py))
                && verification.borrow(py).id == owner
            {
                Ds41Verification::settle(verification.bind(py), py, self)?;
            }
        }
        Ok(k)
    }

    fn wait_stage(&mut self, py: Python<'_>, k: usize) -> PyResult<()> {
        let glue = self.glue.get();
        let (event, spin) = (self.stage_events + k as i32, self.spin_ns);
        let start = Instant::now();
        py.detach(|| glue.glue.wait(event, spin))
            .map_err(|error| post_accept(py, error))?;
        self.timings.output_wait += since(start);
        Ok(())
    }

    /// The first `n` columns' samples in staging set `k` (once its event is done).
    fn staged(&self, k: usize, n: usize) -> Outputs {
        let base = k * self.staging_stride;
        let s = self.sampled;
        Outputs {
            tokens: self.staging.words(base + s.tok, n),
            lp: self.staging.words(base + s.lp, n),
            top_ids: self.staging.words(base + s.top_idx, n * TOP),
            top_lp: self.staging.words(base + s.top_lp, n * TOP),
        }
    }

    /// The sampler over a pass just queued (`sample_launched`): every column (`last` None) or column `last`, into
    /// a staging set; waits for it and reads the first `n` columns.
    fn sample_now(
        &mut self,
        py: Python<'_>,
        shape: (usize, bool),
        params: &[[f32; PARAMS]],
        last: Option<usize>,
        one_column: bool,
    ) -> PyResult<Outputs> {
        let k = self.take_stage(py)?;
        // the step gathered only column `last`: its sampler reads a one-column step's logits
        let table_last = if one_column {
            0
        } else {
            last.map_or(0, |c| c as i64)
        };
        self.write_sample_table(params, table_last);
        self.registers[R_STAGE] = k as u64;
        let section = match last {
            None => IO_SAMPLE,
            Some(_) if one_column => IO_SAMPLE_LAST,
            Some(_) => IO_SAMPLE_AT,
        };
        let steps = [self.io(shape, section)?, self.io(shape, IO_OUT)?];
        self.submit(py, &steps, "sample")?;
        self.wait_stage(py, k)?;
        Ok(self.staged(k, params.len()))
    }

    /// Record the samples of all but the newest `keep` enqueued steps in the context (`_harvest`); with `keep`
    /// 0, also the open speculative step's.
    fn harvest(&mut self, py: Python<'_>, keep: usize) -> PyResult<()> {
        while self.inflight.len() > keep {
            let completion = self.inflight[0].clone_ref(py);
            Ds41Completion::finish(&completion.bind(py).clone(), py, self)?;
        }
        if keep == 0
            && let Some(open) = self.verification.as_ref().map(|v| v.clone_ref(py))
        {
            Ds41Verification::settle(open.bind(py), py, self)?;
        }
        Ok(())
    }

    /// Settle every open step (the driver's Engine took their results before it released rows), then forget
    /// the sequences' Engram records.
    fn release_ids(&mut self, py: Python<'_>, ids: &[u64]) -> PyResult<()> {
        self.harvest(py, 0)?;
        self.context.borrow_mut(py).context.release(ids);
        Ok(())
    }

    /// `PagedCachePlan.from_plan`'s checks, and the top lists' length (`enqueue`'s checks).
    fn check_cache(&self, plan: &BatchPlan) -> PyResult<()> {
        let lookahead = self.decode_lookahead(plan.sequence_ids.len());
        check_paged_plan(plan, self.layout.get().inner_native(), lookahead as u64)
            .map_err(PyValueError::new_err)?;
        if plan
            .sampling_params
            .iter()
            .any(|p| p.logprobs && p.top_logprobs > TOP)
        {
            return Err(PyNotImplementedError::new_err(format!(
                "top_logprobs above {TOP}"
            )));
        }
        Ok(())
    }

    /// `forward`'s checks.
    fn check_plan(&self, plan: &BatchPlan) -> PyResult<()> {
        if plan.media.iter().any(Option::is_some) {
            return Err(PyNotImplementedError::new_err(
                "this runner has no vision tower; text prompts only",
            ));
        }
        if plan.needs_prompt_logprobs() && plan.phase != BatchPhase::Prefill {
            return Err(PyValueError::new_err(
                "prompt logprobs are scored by prefill plans",
            ));
        }
        self.check_cache(plan)?;
        let rows = plan.sequence_ids.len();
        if rows > self.max_batch_size {
            return Err(PyValueError::new_err(format!(
                "{rows} rows exceed {}",
                self.max_batch_size
            )));
        }
        let longest = (0..rows)
            .map(|r| {
                i64::from(plan.context_lengths[r] + plan.token_indptr[r + 1] - plan.token_indptr[r])
            })
            .max()
            .unwrap_or(0);
        if longest + self.draft_tokens > self.max_seq_len {
            return Err(PyValueError::new_err(format!(
                "a row exceeds {} positions",
                self.max_seq_len
            )));
        }
        Ok(())
    }

    /// `_decode_lookahead`: the positions the Engine leases past a decode row's token (none for steps of more
    /// than DSPARK_ROWS rows, and none while drafting is off).
    fn decode_lookahead(&self, rows: usize) -> i64 {
        self.drafting.lookahead(rows)
    }

    /// `_speculating`: a decode step of up to DSPARK_ROWS rows that verifies drafts, while drafting is on.
    fn speculating(&self, plan: &BatchPlan) -> bool {
        plan.phase == BatchPhase::Decode
            && !plan.needs_prompt_logprobs()
            && self.drafting.drafts(plan.sequence_ids.len())
    }

    #[allow(clippy::needless_range_loop)] // rows index several parallel tables
    /// `_speculate`: one speculative decode step. DSpark drafts from each row's anchor (at position `starts`, not
    /// yet processed); the target step's position tables are written while the device drafts; then one target
    /// step over [anchor, drafts] at starts .. starts + K, every column sampled.
    fn speculate(
        &mut self,
        py: Python<'_>,
        slf: &Py<Self>,
        plan: &BatchPlan,
        starts: Vec<i64>,
        anchors: Vec<i64>,
    ) -> PyResult<Py<Ds41Verification>> {
        let b = plan.sequence_ids.len();
        let k_drafts = self.draft_tokens as usize;
        let width = k_drafts + 1;
        let block = self
            .draft
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("speculative steps need DSpark"))?
            .settings
            .block;
        if b == 0 || b > self.dspark_rows {
            return Err(PyValueError::new_err(format!(
                "drafts need 1 to {} rows",
                self.dspark_rows
            )));
        }
        let (mut drafting, mut verify) = (StepRows::new(), StepRows::new());
        for r in 0..b {
            let blocks = self.blocks(plan, r)?;
            let before = self.before(py, plan.sequence_ids[r], starts[r], &blocks)?;
            for (rows, column, real) in [
                (&mut drafting, 0, 1),
                (&mut verify, (r * width) as i64, width as i64),
            ] {
                rows.push(
                    column,
                    starts[r],
                    real,
                    blocks.first,
                    &blocks.window,
                    &blocks.ids,
                    &blocks.history,
                    &before,
                );
            }
        }
        let positions: Vec<i64> = starts
            .iter()
            .flat_map(|&p| (0..width as i64).map(move |j| p + j))
            .collect();
        // the drafts
        let td = self.bucket(b * block, false)?;
        let k = self.take_image(py)?;
        {
            let draft = self.draft.as_ref().expect("checked above");
            let tables = &self.tables[&(td, false)];
            let cs0 = self.cs[0].bind(py).readonly();
            let cs1 = self.cs[1].bind(py).readonly();
            let cs_all = [cs0.as_slice()?, cs1.as_slice()?];
            // SAFETY: host copy k's last upload is done (take_image), and no device copy reads it until this step's.
            let host = unsafe {
                self.image
                    .host
                    .bytes_mut(k * self.image.stride, tables.nbytes)
            };
            let start = Instant::now();
            let (model, rows, settings) = (&self.model, drafting.view(), draft.settings);
            py.detach(|| ds41::fill_draft(model, tables, host, &rows, &anchors, settings, cs_all))
                .map_err(PyValueError::new_err)?;
            self.timings.fill += since(start);
        }
        let program = Arc::clone(&self.draft.as_ref().expect("checked above").programs[&td]);
        self.registers[R_IMAGE] = k as u64;
        self.registers[R_BYTES] = self.tables[&(td, false)].nbytes as u64;
        self.registers[0] = b as u64; // the draft program's register: its rows
        self.registers[R_TOKBYTES] = (4 * block * b) as u64;
        let steps = [
            self.io((td, false), IO_UPLOAD)?,
            (program, 0),
            self.io((td, false), IO_DRAFT_OUT)?,
        ];
        self.image.used[k] = true;
        self.submit(py, &steps, "draft")?;
        // the target step's position tables while the device drafts: only its tokens wait for the drafts
        let tv = self.bucket(b * width, false)?;
        let k2 = self.take_image(py)?;
        let counts = self.fill_with(py, k2, (tv, false), &verify, &positions, None, false, None)?;
        let drafts = {
            let draft = self.draft.as_ref().expect("checked above");
            let glue = self.glue.get();
            let (event, spin) = (draft.event, self.spin_ns);
            let start = Instant::now();
            py.detach(|| glue.glue.wait(event, spin))
                .map_err(|error| post_accept(py, error))?;
            self.timings.output_wait += since(start);
            // (block, rows): the rows' drafts interleaved
            let out: Vec<i32> = draft.host.words(0, block * b);
            (0..b)
                .map(|r| {
                    (0..k_drafts)
                        .map(|i| i64::from(out[i * b + r]))
                        .collect::<Vec<i64>>()
                })
                .collect::<Vec<_>>()
        };
        let tokens: Vec<i64> = (0..b)
            .flat_map(|r| std::iter::once(anchors[r]).chain(drafts[r].iter().copied()))
            .collect();
        {
            let tables = &self.tables[&(tv, false)];
            // SAFETY: as for the positions' fill: host copy k2 is this step's, not uploaded yet.
            let host = unsafe {
                self.image
                    .host
                    .bytes_mut(k2 * self.image.stride, tables.nbytes)
            };
            let start = Instant::now();
            let (model, rows) = (&self.model, verify.view());
            py.detach(|| ds41::fill_tokens(model, tables, host, &rows, b * width, Some(&tokens)))
                .map_err(|error| post_accept(py, error))?;
            self.timings.fill += since(start);
        }
        self.set_counts(&counts);
        self.registers[R_COLUMN] = 0;
        let pass = Pass {
            shape: (tv, false),
            sections: self.sections(0, self.n_layers, Some(None)),
        };
        self.pass(py, k2, &pass, 0, &[], &[], "verify")?;
        // each column samples with its own position's uniform; `params` (column 0 of each row) scores the rows
        let columns = self.params_at(plan, &starts, width)?;
        let params: Vec<[f32; PARAMS]> = columns.iter().step_by(width).copied().collect();
        let stage = self.take_stage(py)?;
        self.write_sample_table(&columns, 0);
        self.registers[R_STAGE] = stage as u64;
        let steps = [
            self.io((tv, false), IO_SAMPLE)?,
            self.io((tv, false), IO_OUT)?,
        ];
        self.submit(py, &steps, "verify sample")?;
        let id = self.next_id;
        self.next_id += 1;
        self.stage_owner[stage] = Some(id);
        let verification = Py::new(
            py,
            Ds41Verification {
                executor: slf.clone_ref(py),
                id,
                stage,
                params,
                scores: plan.needs_logprobs(),
                sequences: plan.sequence_ids.clone(),
                starts,
                anchors,
                drafts,
                done: None,
            },
        )?;
        self.verification = Some(verification.clone_ref(py));
        Ok(verification)
    }

    fn output(
        &self,
        py: Python<'_>,
        plan: &BatchPlan,
        params: &[[f32; PARAMS]],
        out: &Outputs,
        prompt: Option<crate::output::PromptColumns>,
    ) -> PyResult<Py<ForwardOutput>> {
        let n = out.tokens.len();
        let token_ids = out.tokens.iter().map(|&t| t as u32).collect();
        let indptr = (0..=n as u32).collect();
        let (scores, top) = if plan.needs_logprobs() {
            let (s, t) = score_columns(params, &vec![1; n], &out.lp, &out.top_ids, &out.top_lp);
            (Some(s), t)
        } else {
            (None, None)
        };
        Py::new(
            py,
            ForwardOutput::native_with_prompt(py, token_ids, indptr, scores, top, prompt)?,
        )
    }

    /// Score positions `which` of `request` (columns `which - start` of the chunk just queued, whose head made
    /// every column's logits) into `out` (`_Scorer.add`): their logits come back in batches of up to the score
    /// buffer's columns, each a run of consecutive columns, and are scored on the host.
    #[allow(clippy::too_many_arguments)]
    fn score_chunk(
        &mut self,
        py: Python<'_>,
        shape: (usize, bool),
        request: &ScoreRequest,
        which: &[usize],
        start: usize,
        out: &mut [Scored],
    ) -> PyResult<()> {
        let side = self.scoring.as_ref().ok_or_else(|| {
            PyNotImplementedError::new_err("prompt logprobs need the score buffer")
        })?;
        let (cap, vocab, ranks, stride, event) = (
            side.columns,
            side.vocab,
            side.ranks,
            side.stride,
            side.event,
        );
        // the scored columns of the chunk, in order, each with the request positions it scores
        let mut columns: Vec<(usize, Vec<usize>)> = Vec::new();
        for &k in which {
            let c = request.positions[k] - start;
            match columns.iter_mut().find(|(col, _)| *col == c) {
                Some((_, ks)) => ks.push(k),
                None => columns.push((c, vec![k])),
            }
        }
        columns.sort_by_key(|(c, _)| *c);
        let mut at = 0;
        while at < columns.len() {
            // a run of consecutive columns, at most `cap` of them
            let first = columns[at].0;
            let mut end = at + 1;
            while end < columns.len() && end - at < cap && columns[end].0 == columns[end - 1].0 + 1
            {
                end += 1;
            }
            let n = end - at;
            self.registers[R_SCORE_COL] = first as u64;
            self.registers[R_SCORE_BYTES] = (4 * n * vocab) as u64;
            let steps = [self.io(shape, IO_SCORES)?];
            self.submit(py, &steps, "scores")?;
            let glue = self.glue.get();
            let spin = self.spin_ns;
            let start_wait = Instant::now();
            py.detach(|| glue.glue.wait(event, spin))
                .map_err(|error| post_accept(py, error))?;
            self.timings.output_wait += since(start_wait);
            let side = self.scoring.as_ref().expect("checked above");
            for (i, (_, ks)) in columns[at..end].iter().enumerate() {
                // the column's logits over the whole vocabulary: each rank's part in rank order
                let mut z = Vec::with_capacity(ranks * vocab);
                for r in 0..ranks {
                    z.extend(side.host.words::<f32>(r * stride + 4 * i * vocab, vocab));
                }
                for &k in ks {
                    out[k] = score_position(&z, &request.targets[k], request.tops[k]);
                }
            }
            at = end;
        }
        Ok(())
    }

    /// A synchronous step: every prefill row in order, or one decode step.
    fn forward_native(&mut self, py: Python<'_>, plan: &BatchPlan) -> PyResult<Py<ForwardOutput>> {
        let checked = self.check_plan(plan);
        self.gated(py, checked)?;
        self.harvest(py, 0)?;
        let params = self.params(plan)?;
        let mut out = Outputs::empty();
        let mut prompt = None;
        if plan.phase == BatchPhase::Prefill {
            let requests = if plan.needs_prompt_logprobs() {
                score_requests(plan, self.vocab)
            } else {
                (0..plan.sequence_ids.len()).map(|_| None).collect()
            };
            let mut scored = Vec::new();
            for r in 0..plan.sequence_ids.len() {
                let mut row = requests[r]
                    .as_ref()
                    .map(|request| vec![Scored::default(); request.positions.len()]);
                out.extend(self.prefill_row(
                    py,
                    plan,
                    r,
                    &params[r..r + 1],
                    requests[r].as_ref(),
                    row.as_deref_mut(),
                )?);
                scored.extend(row);
            }
            prompt = plan
                .needs_prompt_logprobs()
                .then(|| prompt_columns(plan, &scored));
        } else {
            let n = plan.sequence_ids.len();
            let positions: Vec<i64> = plan.context_lengths.iter().map(|&c| i64::from(c)).collect();
            let tokens: Vec<i64> = plan.token_ids.iter().map(|&t| i64::from(t)).collect();
            let (_, t) = self.decode_pass(py, plan, &positions, Some(&tokens))?;
            out = self.sample_now(py, (t, false), &params, None, false)?;
            debug_assert_eq!(out.tokens.len(), n);
        }
        self.output(py, plan, &params, &out, prompt)
    }

    #[allow(clippy::needless_range_loop)] // rows index several parallel tables
    /// Fill and queue a decode pass of every row of `plan` at `positions` (`tokens` None: the device's, the
    /// previous step's samples); records host tokens in the context. Returns (host copy, bucket).
    fn decode_pass(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        positions: &[i64],
        tokens: Option<&[i64]>,
    ) -> PyResult<(usize, usize)> {
        let n = plan.sequence_ids.len();
        if positions.iter().any(|&p| p + 1 > self.max_seq_len) {
            return Err(PyValueError::new_err(format!(
                "a row exceeds {} positions",
                self.max_seq_len
            )));
        }
        let t = self.bucket(n, false)?;
        let mut rows = StepRows::new();
        for r in 0..n {
            let blocks = self.blocks(plan, r)?;
            let before = self.before(py, plan.sequence_ids[r], positions[r], &blocks)?;
            rows.push(
                r as i64,
                positions[r],
                1,
                blocks.first,
                &blocks.window,
                &blocks.ids,
                &blocks.history,
                &before,
            );
        }
        let k = self.take_image(py)?;
        let counts = self.fill(py, k, (t, false), &rows, positions, tokens, None)?;
        self.set_counts(&counts);
        self.registers[R_COLUMN] = 0;
        let pass = Pass {
            shape: (t, false),
            sections: self.sections(0, self.n_layers, Some(None)),
        };
        let device = if tokens.is_none() { n } else { 0 };
        self.pass(py, k, &pass, device, &[], &[], "decode")?;
        if let Some(tokens) = tokens {
            for r in 0..n {
                self.seen(py, plan.sequence_ids[r], positions[r], &tokens[r..r + 1])?;
            }
        }
        Ok((k, t))
    }

    /// One prefill row in chunks of up to `tmax` tokens (bounded replay: the layers after the cut only on the
    /// prompt's last `replay` positions); returns its last chunk's sample.
    fn prefill_row(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        r: usize,
        params: &[[f32; PARAMS]],
        request: Option<&ScoreRequest>,
        mut scored: Option<&mut [Scored]>,
    ) -> PyResult<Outputs> {
        let sequence = plan.sequence_ids[r];
        let ids: Vec<i64> = plan.token_ids
            [plan.token_indptr[r] as usize..plan.token_indptr[r + 1] as usize]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let pos0 = i64::from(plan.context_lengths[r]);
        let floor = i64::from(plan.prefill_lengths[r]) - self.replay;
        let replay = ds41::replays(self.cut.is_some(), self.replay, floor, request.is_some());
        let blocks = self.blocks(plan, r)?;
        let mut sampled = None;
        let mut s = 0;
        while s < ids.len() {
            let chunk = &ids[s..(s + self.tmax).min(ids.len())];
            let p = pos0 + s as i64;
            let last = s + chunk.len() >= ids.len();
            let n = chunk.len();
            let before = self.before(py, sequence, p, &blocks)?;
            let mut rows = StepRows::new();
            rows.push(
                0,
                p,
                n as i64,
                blocks.first,
                &blocks.window,
                &blocks.ids,
                &blocks.history,
                &before,
            );
            let positions: Vec<i64> = (0..n as i64).map(|j| p + j).collect();
            if replay {
                let kept = (n as i64).min((p + n as i64 - p.max(floor)).max(0)) as usize;
                let logits = (last && kept > 0).then(|| kept - 1);
                let t =
                    self.replay_chunk(py, chunk, &positions, &rows, &before, kept, floor, logits)?;
                self.seen(py, sequence, p, chunk)?;
                if last {
                    sampled = Some(match t {
                        Some(t) if kept > 0 => {
                            self.sample_now(py, (t, true), params, Some(kept - 1), self.one_column)?
                        }
                        _ => {
                            if plan.sampling_mask[r] {
                                return Err(PyValueError::new_err(
                                    "a sampled prefill ends before its kept span",
                                ));
                            }
                            Outputs::unsampled()
                        }
                    });
                }
            } else {
                let t = self.bucket(n, true)?;
                let k = self.take_image(py)?;
                let counts = self.fill(py, k, (t, true), &rows, &positions, Some(chunk), None)?;
                self.set_counts(&counts);
                // the chunk's scored positions (`_Scorer.columns`): their columns need every column's logits
                let which: Vec<usize> = request.map_or(Vec::new(), |req| {
                    (0..req.positions.len())
                        .filter(|&k| req.positions[k] >= s && req.positions[k] < s + n)
                        .collect()
                });
                let all = !which.is_empty();
                let logits = if all {
                    Some(None)
                } else {
                    last.then_some(Some(n - 1))
                };
                let column = last && !all && self.one_column;
                self.registers[R_COLUMN] = if column { (n - 1) as u64 } else { 0 };
                let pass = Pass {
                    shape: (t, true),
                    sections: self.sections(0, self.n_layers, logits),
                };
                self.pass(py, k, &pass, 0, &[], &[], "prefill")?;
                self.seen(py, sequence, p, chunk)?;
                if let (Some(req), Some(out)) = (request, scored.as_deref_mut())
                    && all
                {
                    self.score_chunk(py, (t, true), req, &which, s, out)?;
                }
                if last {
                    sampled = Some(self.sample_now(py, (t, true), params, Some(n - 1), column)?);
                }
            }
            s += n;
        }
        sampled.ok_or_else(|| PyValueError::new_err("a prefill row without tokens"))
    }

    /// `launch_replay`: every column through the layers before the cut and the cut owner's store; then the last
    /// `kept` columns, compacted, through the cut layer and every later one, the injection and the head
    /// (`logits` relative to the kept columns) at their own bucket, no window reaching before `floor`. Returns
    /// the kept columns' bucket (None: nothing kept).
    #[allow(clippy::too_many_arguments)]
    fn replay_chunk(
        &mut self,
        py: Python<'_>,
        tokens: &[i64],
        positions: &[i64],
        rows: &StepRows,
        before: &[i64],
        kept: usize,
        floor: i64,
        logits: Option<usize>,
    ) -> PyResult<Option<usize>> {
        let cut = self
            .cut
            .ok_or_else(|| PyValueError::new_err("bounded replay is off"))?;
        let n = positions.len();
        let t = self.bucket(n, true)?;
        let k = self.take_image(py)?;
        let counts = self.fill(py, k, (t, true), rows, positions, Some(tokens), None)?;
        self.set_counts(&counts);
        self.registers[R_COLUMN] = 0;
        let mut sections = self.sections(0, cut, None);
        sections.push(self.n_layers + 4); // the owner store
        let pass = Pass {
            shape: (t, true),
            sections,
        };
        self.pass(py, k, &pass, 0, &[], &[], "replay")?;
        if kept == 0 {
            return Ok(None);
        }
        let skip = n - kept;
        let nb = before.len();
        let mut seen = before.to_vec();
        seen.extend_from_slice(tokens);
        let lo = (nb + skip).saturating_sub(self.engram_n);
        let keep_before = seen[lo..nb + skip].to_vec();
        let mut keep_rows = StepRows::new();
        keep_rows.push(
            0,
            rows.pos0[0] + skip as i64,
            kept as i64,
            rows.window_first[0],
            &rows.window,
            &rows.window_ids,
            &rows.history,
            &keep_before,
        );
        let tk = self.bucket(kept, true)?;
        let k2 = self.take_image(py)?;
        let counts = self.fill(
            py,
            k2,
            (tk, true),
            &keep_rows,
            &positions[skip..],
            Some(&tokens[skip..]),
            Some(floor),
        )?;
        self.set_counts(&counts);
        self.registers[R_SKIP] = skip as u64;
        self.registers[R_KEPT] = kept as u64;
        let compact = {
            let shaped = &self.shapes[&(t, true)];
            let section = *shaped.compact.get(&tk).ok_or_else(|| {
                PyValueError::new_err(format!("no compaction from {t} to {tk} columns"))
            })?;
            (Arc::clone(&shaped.io), section)
        };
        self.registers[R_COLUMN] = match logits {
            Some(c) if self.one_column => c as u64,
            _ => 0,
        };
        let pass = Pass {
            shape: (tk, true),
            sections: self.sections(cut, self.n_layers, logits.map(Some)),
        };
        self.pass(py, k2, &pass, 0, &[compact], &[], "replay kept")?;
        Ok(Some(tk))
    }
}

#[pymethods]
impl Ds41Executor {
    #[classattr]
    const REGISTERS: usize = REGISTERS;
    #[classattr]
    const R_IMAGE: usize = R_IMAGE;
    #[classattr]
    const R_BYTES: usize = R_BYTES;
    #[classattr]
    const R_STAGE: usize = R_STAGE;
    #[classattr]
    const R_TOKBYTES: usize = R_TOKBYTES;
    #[classattr]
    const R_SKIP: usize = R_SKIP;
    #[classattr]
    const R_KEPT: usize = R_KEPT;
    #[classattr]
    const R_SCORE_COL: usize = R_SCORE_COL;
    #[classattr]
    const R_SCORE_BYTES: usize = R_SCORE_BYTES;

    /// `shapes`: per (columns, prefill) a dict of `step` and `io` (`GaudiProgram`s) and `compact` (kept bucket ->
    /// the io section). `image`: (device table, pinned host copies, stride, first event slot); `staging`: (pinned
    /// sets, stride, first event slot, (tok, lp, top_idx, top_lp, bytes) offsets); `sample`: (pinned sampler
    /// table, offset of its `last`); `geometry`: (slab rows, WINDOW block rows, HISTORY block rows, WINDOW blocks
    /// per slab); `model`: a dict of the model's numbers.
    #[new]
    #[pyo3(signature = (*, glue, fill, context, layout, shapes, image, staging, sample, geometry, model, draft=None, scoring=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        glue: Py<GaudiGlue>,
        fill: PyRef<'_, Ds41Fill>,
        context: Py<Ds41Context>,
        layout: Py<CacheLayout>,
        shapes: HashMap<(usize, bool), Bound<'_, PyDict>>,
        image: (u64, PyReadwriteArray1<'_, u8>, usize, i32),
        staging: StagingArgs<'_>,
        sample: (PyReadwriteArray1<'_, u8>, usize),
        geometry: (i64, i64, i64, i64),
        model: Bound<'_, PyDict>,
        draft: Option<Bound<'_, PyDict>>,
        scoring: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let scoring = match scoring {
            Some(d) => {
                let host = d
                    .get_item("host")?
                    .ok_or_else(|| PyValueError::new_err("scoring has no host"))?;
                let host = Pinned::new("scores", host.extract::<PyReadwriteArray1<'_, u8>>()?)?;
                let side = ScoreSide {
                    host,
                    stride: item(&d, "stride")?,
                    columns: item(&d, "columns")?,
                    vocab: item(&d, "vocab")?,
                    ranks: item(&d, "ranks")?,
                    event: item(&d, "event")?,
                };
                if side.ranks * side.stride > side.host.bytes
                    || 4 * side.columns * side.vocab > side.stride
                {
                    return Err(PyValueError::new_err("the score buffer is too small"));
                }
                Some(side)
            }
            None => None,
        };
        let draft = match draft {
            Some(d) => {
                let listed = d
                    .get_item("programs")?
                    .ok_or_else(|| PyValueError::new_err("draft has no programs"))?;
                let mut programs = HashMap::new();
                for (t, program) in listed.cast::<PyDict>()?.iter() {
                    let program = program.extract::<PyRef<'_, GaudiProgram>>()?.program()?;
                    programs.insert(t.extract::<usize>()?, program);
                }
                let host = d
                    .get_item("host")?
                    .ok_or_else(|| PyValueError::new_err("draft has no host"))?;
                let host = Pinned::new("drafts", host.extract::<PyReadwriteArray1<'_, u8>>()?)?;
                Some(DraftSide {
                    programs,
                    host,
                    event: item(&d, "event")?,
                    settings: ds41::Draft {
                        block: item(&d, "block")?,
                        noise: item(&d, "noise")?,
                        stages: item(&d, "stages")?,
                    },
                    stats: speculative::Stats::default(),
                })
            }
            None => None,
        };
        let (fill_model, fill_tables, cs) = fill.parts(py);
        let mut built = HashMap::new();
        for (key, d) in &shapes {
            let program = |name: &str| -> PyResult<Arc<step::Program>> {
                let p: PyRef<'_, GaudiProgram> = d
                    .get_item(name)?
                    .ok_or_else(|| PyValueError::new_err(format!("shape {key:?} has no {name}")))?
                    .extract()?;
                p.program()
            };
            let (step_program, io) = (program("step")?, program("io")?);
            if io.registers() != REGISTERS || step_program.registers() > REGISTERS {
                return Err(PyValueError::new_err(format!(
                    "shape {key:?}'s programs take more than {REGISTERS} registers"
                )));
            }
            let tables = fill_tables
                .get(key)
                .ok_or_else(|| PyValueError::new_err(format!("the fill has no shape {key:?}")))?;
            if tables.nbytes > image.2 {
                return Err(PyValueError::new_err(format!(
                    "shape {key:?}'s tables of {} bytes exceed the host images",
                    tables.nbytes
                )));
            }
            let mut io_sections = [None; IO_SECTIONS.len()];
            for (k, name) in IO_SECTIONS.iter().enumerate() {
                io_sections[k] = io.sections().iter().position(|s| s.name == *name);
            }
            if io_sections[IO_UPLOAD].is_none() || io_sections[IO_OUT].is_none() {
                return Err(PyValueError::new_err(format!(
                    "shape {key:?}'s io program lacks upload or out"
                )));
            }
            let compact: HashMap<usize, String> = item(d, "compact")?;
            let compact = compact
                .into_iter()
                .map(|(kept, name)| {
                    io.sections()
                        .iter()
                        .position(|s| s.name == name)
                        .map(|at| (kept, at))
                        .ok_or_else(|| PyValueError::new_err(format!("no io section {name}")))
                })
                .collect::<PyResult<_>>()?;
            built.insert(
                *key,
                Shaped {
                    step: step_program,
                    io,
                    io_sections,
                    compact,
                },
            );
        }
        let (device, host, stride, events) = image;
        let host = Pinned::at_least("image host copies", host, 2 * stride)?;
        let (staging_host, staging_stride, stage_events, (tok, lp, top_idx, top_lp, bytes)) =
            staging;
        let staging_host = Pinned::at_least("staging", staging_host, 2 * staging_stride)?;
        if bytes > staging_stride {
            return Err(PyValueError::new_err(
                "the sampler's outputs exceed a staging set",
            ));
        }
        let (sample_host, sample_last) = sample;
        let sample_host = Pinned::new("sampler table", sample_host)?;
        if sample_last + 4 > sample_host.bytes
            || 4 * PARAMS * item::<usize>(&model, "max_batch_size")? > sample_last
        {
            return Err(PyValueError::new_err("the sampler table is too small"));
        }
        let (slab_rows, window_block_rows, history_block_rows, window_per_slab) = geometry;
        let (draft_tokens, dspark_rows) =
            (item(&model, "draft_tokens")?, item(&model, "dspark_rows")?);
        Ok(Self {
            glue,
            model: fill_model,
            tables: fill_tables,
            cs,
            context,
            shapes: built,
            layout,
            geometry: Geometry {
                slab_rows,
                window_block_rows,
                history_block_rows,
                window_per_slab,
            },
            image: Images::new(device, host, stride, events)?,
            staging: staging_host,
            staging_stride,
            stage_events,
            stage_next: 0,
            stage_owner: [None, None],
            sample_host,
            sample_last,
            sampled: Sampled {
                tok,
                lp,
                top_idx,
                top_lp,
            },
            vocab: item(&model, "vocab")?,
            n_layers: item(&model, "layers")?,
            cut: item(&model, "cut")?,
            replay: item(&model, "replay")?,
            prefill_buckets: item(&model, "prefill_buckets")?,
            decode_buckets: item(&model, "decode_buckets")?,
            tmax: item(&model, "tmax")?,
            one_column: item(&model, "one_column")?,
            max_batch_size: item(&model, "max_batch_size")?,
            max_seq_len: item(&model, "max_seq_len")?,
            draft_tokens,
            dspark_rows,
            drafting: ds41::Drafting {
                tokens: draft_tokens,
                rows: dspark_rows,
                on: true,
                decided: None,
            },
            engram_n: item(&model, "engram_n")?,
            spin_ns: item(&model, "spin_ns")?,
            registers: [0; REGISTERS],
            inflight: Vec::new(),
            next_id: 0,
            timings: Timings::default(),
            native_steps: 0,
            draft,
            verification: None,
            continuations: 0,
            ranks: None,
            scoring,
            card_gate: None,
        })
    }

    /// Runs of the native step programs so far.
    #[getter]
    fn native_steps(&self) -> u64 {
        self.native_steps
    }

    /// Host nanoseconds so far: fill, submit, image_wait, output_wait.
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
        self.draft
            .as_ref()
            .map_or((0, 0, 0, 0), |d| d.stats.counts())
    }

    /// The runner's `metrics_snapshot` (the speculation counters), with no Python: the runner binds it.
    fn metrics_snapshot(&self) -> crate::metrics::RunnerMetrics {
        let (blocks, drafts, accepted, _) = self.speculation();
        crate::metrics::RunnerMetrics::speculation(drafts, accepted, blocks)
    }

    /// The benchmark's fixed acceptance (`SpeculativeStats.benchmark_accepted_drafts`): every row accepts this
    /// many drafts (None: the samples decide).
    #[getter]
    fn fixed_acceptance(&self) -> Option<usize> {
        self.draft.as_ref().and_then(|d| d.stats.fixed)
    }

    #[setter]
    fn set_fixed_acceptance(&mut self, fixed: Option<usize>) -> PyResult<()> {
        if fixed.is_some_and(|f| f as i64 > self.draft_tokens) {
            return Err(PyValueError::new_err(
                "fixed acceptance must fit the drafts",
            ));
        }
        if let Some(draft) = self.draft.as_mut() {
            draft.stats.fixed = fixed;
        }
        if let Some(ranks) = self.ranks.as_mut() {
            ranks.fixed = fixed; // the followers take it from each call's frame
        }
        Ok(())
    }

    /// DSpark's switch: while it is on, a decode step of up to DSPARK_ROWS rows drafts and verifies; while it is
    /// off, every decode step decodes one token a row and expects no lookahead lease (the Engine's
    /// `max_speculative_rows` must change with it). Plain steps keep the drafter's state current, so drafts are
    /// good again right after the switch goes back on. Switch it only between steps.
    #[getter]
    fn drafting(&self) -> bool {
        self.drafting.on && self.drafting.tokens > 0
    }

    #[setter]
    fn set_drafting(&mut self, on: bool) -> PyResult<()> {
        if on && self.draft.is_none() {
            return Err(PyValueError::new_err(
                "drafting needs DSpark (a runner with draft_tokens)",
            ));
        }
        if !self.inflight.is_empty() || self.verification.is_some() {
            return Err(PyValueError::new_err(
                "switch drafting between steps: a step is still open",
            ));
        }
        self.drafting.on = on;
        Ok(())
    }

    /// Enqueued steps that continued the previous one.
    #[getter]
    fn continuations(&self) -> u64 {
        self.continuations
    }

    /// Forget released sequences (their Engram records; their blocks' records stay).
    fn release(&mut self, py: Python<'_>, sequence_ids: Vec<u64>) -> PyResult<()> {
        if let Some(ranks) = self.ranks.as_ref() {
            self.timings.announce +=
                ranks.announce(py, OP_RELEASE, None, false, false, &sequence_ids)?;
        }
        self.release_ids(py, &sequence_ids)
    }

    /// The Engine's synchronous step: prefill chunks, a decode step or a speculative step. With ranks, the driver
    /// sends the call to the followers first.
    fn forward(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow_mut().lead(py, OP_FORWARD, plan.get(), None)?;
        let out = Self::forward_step(&slf, py, plan);
        if let Some(ranks) = slf.borrow_mut().ranks.as_mut() {
            ranks.led(py, &out);
        }
        out
    }

    /// Submit a decode step and return without waiting for it: a continuation (`feedback`) reads its tokens from
    /// the previous step's samples on the device, or continues its verification. Other plans run synchronously.
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
        if let Some(ranks) = slf.borrow_mut().ranks.as_mut() {
            ranks.led(py, &out);
        }
        out
    }

    /// Join a tensor-parallel model's rank group: rank 0 drives (it sends each runner call to the others), and
    /// every other rank replays the calls (`serve`).
    fn attach_ranks(&mut self, py: Python<'_>, ranks: Py<GaudiRanks>) {
        self.ranks = Ranked::of(py, ranks);
        let fixed = self.draft.as_ref().and_then(|d| d.stats.fixed);
        if let Some(ranks) = self.ranks.as_mut() {
            ranks.fixed = fixed;
        }
    }

    /// A layout's card gate: every runner call is a TP step on this card, which a replica on it waits for.
    fn attach_gate(&mut self, gate: Py<CardGate>) {
        self.card_gate = Some(gate);
    }

    /// The driver: tell the followers to stop serving.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(ranks) = self.ranks.take() {
            self.timings.announce += ranks.announce(py, OP_CLOSE, None, false, false, &[])?;
        }
        Ok(())
    }

    /// A follower: replay the driver's runner calls until it closes the group. Returns the calls replayed.
    fn serve(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<u64> {
        crate::gaudi_ranks::serve(&slf, py)
    }
}

/// One enqueued decode step: `result` waits for its staged samples once and records each row's sampled token
/// (the token at its position + 1) as the sequence's newest in the context.
#[pyclass(module = "resin._resin")]
pub(crate) struct Ds41Completion {
    executor: Py<Ds41Executor>,
    id: u64,
    stage: usize,
    params: Vec<[f32; PARAMS]>,
    scores: bool,
    sequences: Vec<u64>,
    positions: Vec<i64>,
    output: Option<Py<ForwardOutput>>,
}

impl Ds41Completion {
    /// Read the step's samples (once), record them in the context, and leave the executor's open steps.
    fn finish(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut Ds41Executor,
    ) -> PyResult<Py<ForwardOutput>> {
        if let Some(output) = &this.borrow().output {
            return Ok(output.clone_ref(py));
        }
        let (id, stage, rows) = {
            let c = this.borrow();
            (c.id, c.stage, c.sequences.len())
        };
        executor.wait_stage(py, stage)?;
        let out = executor.staged(stage, rows);
        let output = {
            let c = this.borrow();
            let n = out.tokens.len();
            let token_ids = out.tokens.iter().map(|&t| t as u32).collect();
            let indptr = (0..=n as u32).collect();
            let (scores, top) = if c.scores {
                let (s, t) =
                    score_columns(&c.params, &vec![1; n], &out.lp, &out.top_ids, &out.top_lp);
                (Some(s), t)
            } else {
                (None, None)
            };
            Py::new(
                py,
                ForwardOutput::native(py, token_ids, indptr, scores, top)?,
            )?
        };
        {
            let c = this.borrow();
            let mut context = executor.context.borrow_mut(py);
            for (r, &sequence) in c.sequences.iter().enumerate() {
                if context.context.has(sequence) {
                    // not released meanwhile
                    context
                        .context
                        .seen(sequence, c.positions[r] + 1, &[i64::from(out.tokens[r])])
                        .map_err(PyRuntimeError::new_err)?;
                }
            }
        }
        executor.inflight.retain(|c| c.borrow(py).id != id);
        if executor.stage_owner[stage] == Some(id) {
            executor.stage_owner[stage] = None;
        }
        this.borrow_mut().output = Some(output.clone_ref(py));
        Ok(output)
    }
}

#[pymethods]
impl Ds41Completion {
    fn result(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<ForwardOutput>> {
        if let Some(output) = &slf.borrow().output {
            return Ok(output.clone_ref(py));
        }
        let executor = slf.borrow().executor.clone_ref(py);
        let mut executor = executor.borrow_mut(py);
        Self::finish(&slf, py, &mut executor)
    }

    #[getter]
    fn feedback(slf: Bound<'_, Self>) -> PyResult<Ds41Feedback> {
        let positions = slf.borrow().positions.iter().map(|&p| p + 1).collect();
        Ok(Ds41Feedback {
            positions,
            completion: slf.unbind(),
        })
    }
}

/// A plain decode step's continuation: each row's next position; its tokens stay on the device.
#[pyclass(module = "resin._resin", frozen)]
pub(crate) struct Ds41Feedback {
    positions: Vec<i64>,
    #[allow(dead_code)] // keeps the step it continues alive
    completion: Py<Ds41Completion>,
}

#[pymethods]
impl Ds41Feedback {
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        numpy::PyArray1::from_slice(py, &self.positions)
    }
}

/// A finished verification: each row's next anchor position and anchor.
struct Verified {
    positions: Vec<i64>,
    anchors: Vec<i64>,
    output: Py<ForwardOutput>,
}

/// An enqueued speculative step and its continuation's feedback (`deepseek_v41.runner._Verification`): `finish`
/// waits for its samples, accepts each row's longest draft prefix that the target's samples confirm, records the
/// processed tokens it commits (anchor and accepted drafts), and leaves each row's next anchor.
#[pyclass(module = "resin._resin")]
pub(crate) struct Ds41Verification {
    executor: Py<Ds41Executor>,
    id: u64,
    stage: usize,
    params: Vec<[f32; PARAMS]>,
    scores: bool,
    sequences: Vec<u64>,
    starts: Vec<i64>,
    anchors: Vec<i64>,
    drafts: Vec<Vec<i64>>,
    done: Option<Verified>,
}

impl Ds41Verification {
    fn settle(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut Ds41Executor,
    ) -> PyResult<Verified> {
        if let Some(done) = &this.borrow().done {
            return Ok(Verified {
                positions: done.positions.clone(),
                anchors: done.anchors.clone(),
                output: done.output.clone_ref(py),
            });
        }
        let (id, stage, b, width) = {
            let v = this.borrow();
            (
                v.id,
                v.stage,
                v.sequences.len(),
                v.drafts.first().map_or(1, |d| d.len() + 1),
            )
        };
        executor.wait_stage(py, stage)?;
        let out = executor.staged(stage, b * width);
        let samples: Vec<i64> = out.tokens.iter().map(|&t| i64::from(t)).collect();
        let verified = {
            let v = this.borrow();
            let flat: Vec<i64> = v.drafts.iter().flatten().copied().collect();
            let stats = &mut executor
                .draft
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("speculative steps need DSpark"))?
                .stats;
            let (accepted, rows) = stats.accept(&flat, &samples, b, width);
            let counts: Vec<usize> = accepted.iter().map(|&a| a as usize + 1).collect();
            {
                let mut context = executor.context.borrow_mut(py);
                for (r, &sequence) in v.sequences.iter().enumerate() {
                    let mut committed = vec![v.anchors[r]];
                    committed.extend_from_slice(&rows[r][..rows[r].len() - 1]);
                    context
                        .context
                        .seen(sequence, v.starts[r], &committed)
                        .map_err(PyRuntimeError::new_err)?;
                }
            }
            let positions: Vec<i64> = (0..b).map(|r| v.starts[r] + counts[r] as i64).collect();
            let anchors: Vec<i64> = rows
                .iter()
                .map(|row| *row.last().expect("a sample"))
                .collect();
            let token_ids: Vec<u32> = rows.iter().flatten().map(|&t| t as u32).collect();
            let mut indptr = vec![0u32];
            for &count in &counts {
                indptr.push(indptr.last().copied().unwrap_or(0) + count as u32);
            }
            let (scores, top) = if v.scores {
                let picked: Vec<usize> = (0..b)
                    .flat_map(|r| (0..counts[r]).map(move |j| r * width + j))
                    .collect();
                let lp: Vec<f32> = picked.iter().map(|&c| out.lp[c]).collect();
                let ids: Vec<i32> = picked
                    .iter()
                    .flat_map(|&c| out.top_ids[c * TOP..(c + 1) * TOP].iter().copied())
                    .collect();
                let values: Vec<f32> = picked
                    .iter()
                    .flat_map(|&c| out.top_lp[c * TOP..(c + 1) * TOP].iter().copied())
                    .collect();
                let (s, t) = score_columns(&v.params, &counts, &lp, &ids, &values);
                (Some(s), t)
            } else {
                (None, None)
            };
            let output = Py::new(
                py,
                ForwardOutput::native(py, token_ids, indptr, scores, top)?,
            )?;
            Verified {
                positions,
                anchors,
                output,
            }
        };
        if executor.stage_owner[stage] == Some(id) {
            executor.stage_owner[stage] = None;
        }
        if executor
            .verification
            .as_ref()
            .is_some_and(|open| open.borrow(py).id == id)
        {
            executor.verification = None;
        }
        let result = Verified {
            positions: verified.positions.clone(),
            anchors: verified.anchors.clone(),
            output: verified.output.clone_ref(py),
        };
        this.borrow_mut().done = Some(verified);
        Ok(result)
    }
}

#[pymethods]
impl Ds41Verification {
    fn result(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<ForwardOutput>> {
        if let Some(done) = &slf.borrow().done {
            return Ok(done.output.clone_ref(py));
        }
        let executor = slf.borrow().executor.clone_ref(py);
        let mut executor = executor.borrow_mut(py);
        Ok(Self::settle(&slf, py, &mut executor)?.output)
    }

    fn finish(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        Self::result(slf, py).map(drop)
    }

    #[getter]
    fn feedback(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Each row's next anchor position (after `finish`).
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
        let done = self
            .done
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("not finished"))?;
        Ok(numpy::PyArray1::from_slice(py, &done.positions))
    }

    /// Each row's next anchor (after `finish`).
    #[getter]
    fn anchors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
        let done = self
            .done
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("not finished"))?;
        Ok(numpy::PyArray1::from_slice(py, &done.anchors))
    }
}

impl Ds41Executor {
    /// `step` as one TP step on a layout's card gate (`GatedTransport`'s begin and end around a forward command).
    fn on_card<T>(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        step: impl FnOnce() -> PyResult<T>,
    ) -> PyResult<T> {
        let gate = slf.borrow().card_gate.as_ref().map(|g| g.clone_ref(py));
        let Some(gate) = gate else {
            return step();
        };
        gate.get().begin(py)?;
        let out = step();
        let ended = gate.get().end(py);
        let out = out?;
        ended?;
        Ok(out)
    }

    fn forward_step(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        Self::on_card(slf, py, || Self::forward_call(slf, py, plan))
    }

    fn enqueue_step(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Self::on_card(slf, py, || Self::enqueue_call(slf, py, plan, feedback))
    }

    fn forward_call(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        let speculating = slf.borrow().speculating(plan.get());
        if speculating {
            let p = plan.get();
            let mut this = slf.borrow_mut();
            let checked = this.check_plan(p);
            this.gated(py, checked)?;
            this.harvest(py, 0)?;
            let starts = p.context_lengths.iter().map(|&c| i64::from(c)).collect();
            let anchors = p.token_ids.iter().map(|&t| i64::from(t)).collect();
            let verification = this.speculate(py, &slf.clone().unbind(), p, starts, anchors)?;
            drop(this);
            return Ok(Ds41Verification::result(verification.bind(py).clone(), py)?.into_any());
        }
        let out = slf.borrow_mut().forward_native(py, plan.get())?;
        Ok(out.into_any())
    }

    fn enqueue_call(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let p = plan.get();
        let native_feedback = feedback.as_ref().and_then(|f| {
            f.cast::<Ds41Feedback>()
                .ok()
                .map(|f| f.get().positions.clone())
        });
        let verified = feedback.as_ref().and_then(|f| {
            f.cast::<Ds41Verification>()
                .ok()
                .map(|v| v.clone().unbind())
        });
        let python_feedback = feedback.is_some() && native_feedback.is_none() && verified.is_none();
        if python_feedback {
            return Err(PyValueError::new_err(
                "a continuation of a step this executor did not make",
            ));
        }
        let speculating = slf.borrow().speculating(p);
        if speculating && native_feedback.is_none() {
            // a speculative step from the plan, or continuing the previous one's verification
            let mut this = slf.borrow_mut();
            let checked = this.check_cache(p);
            this.gated(py, checked)?;
            this.harvest(py, 0)?;
            let (starts, anchors) = match &verified {
                Some(previous) => {
                    let done = Ds41Verification::settle(previous.bind(py), py, &mut this)?;
                    this.continuations += 1;
                    (done.positions, done.anchors)
                }
                None => (
                    p.context_lengths.iter().map(|&c| i64::from(c)).collect(),
                    p.token_ids.iter().map(|&t| i64::from(t)).collect(),
                ),
            };
            let verification = this.speculate(py, &slf.clone().unbind(), p, starts, anchors)?;
            return Ok(verification.into_any());
        }
        if verified.is_some() {
            return Err(PyValueError::new_err(
                "a verification continues only into a speculative step",
            ));
        }
        let n = p.sequence_ids.len();
        let sync = {
            let this = slf.borrow();
            p.phase != BatchPhase::Decode
                || p.needs_prompt_logprobs()
                || n > *this.decode_buckets.last().unwrap_or(&0)
                || p.media.iter().any(Option::is_some)
        };
        if sync {
            let out = slf.borrow_mut().forward_native(py, p)?;
            return Ok(out.into_any());
        }
        let mut this = slf.borrow_mut();
        let checked = this.check_cache(p);
        this.gated(py, checked)?;
        // the tokens before each row's position are known once every step but the one feeding this is recorded
        this.harvest(py, usize::from(native_feedback.is_some()))?;
        let (positions, tokens) = match &native_feedback {
            Some(positions) => (positions.clone(), None),
            None => (
                p.context_lengths
                    .iter()
                    .map(|&c| i64::from(c))
                    .collect::<Vec<_>>(),
                Some(
                    p.token_ids
                        .iter()
                        .map(|&t| i64::from(t))
                        .collect::<Vec<_>>(),
                ),
            ),
        };
        let params = this.params_at(p, &positions, 1)?;
        let (_, t) = this.decode_pass(py, p, &positions, tokens.as_deref())?;
        let k = this.take_stage(py)?;
        this.write_sample_table(&params, 0);
        this.registers[R_STAGE] = k as u64;
        let steps = [
            this.io((t, false), IO_SAMPLE)?,
            this.io((t, false), IO_OUT)?,
        ];
        this.submit(py, &steps, "sample")?;
        let id = this.next_id;
        this.next_id += 1;
        this.stage_owner[k] = Some(id);
        let completion = Py::new(
            py,
            Ds41Completion {
                executor: slf.clone().unbind(),
                id,
                stage: k,
                params,
                scores: p.needs_logprobs(),
                sequences: p.sequence_ids.clone(),
                positions,
                output: None,
            },
        )?;
        this.inflight.push(completion.clone_ref(py));
        Ok(completion.into_any())
    }

    /// The driver: check the call's plan (a plan that no rank can run fails here, before the followers see it),
    /// that a continuation continues the last enqueued step, and send the call to the followers with this rank's
    /// draft decision (a follower does not know the driver's `drafting` switch).
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
        if op == OP_FORWARD {
            self.check_plan(plan)?;
        } else {
            self.check_cache(plan)?;
        }
        if let Some(feedback) = feedback {
            let step = match feedback.cast::<Ds41Feedback>() {
                Ok(plain) => plain.get().completion.bind(py).clone().into_any(),
                Err(_) => feedback.clone(),
            };
            if !ranks.continues_last(py, &step) {
                return Err(PyValueError::new_err(
                    "tensor-parallel ranks continue only the last enqueued step",
                ));
            }
        }
        let speculating = self.speculating(plan);
        self.timings.announce +=
            ranks.announce(py, op, Some(plan), feedback.is_some(), speculating, &[])?;
        Ok(())
    }

    /// With ranks: go ahead with this call's device work only when every rank is ready (one gate a call, after
    /// the plan's checks and before the wait for the previous step). Without: `prepared`.
    fn gated<T>(&mut self, py: Python<'_>, prepared: PyResult<T>) -> PyResult<T> {
        match &self.ranks {
            Some(ranks) => ranks.gate(py, prepared, &mut self.timings.gate),
            None => prepared,
        }
    }
}

impl Replay for Ds41Executor {
    const HARVESTS: bool = true;

    fn set_speculating(&mut self, decision: Option<bool>) {
        self.drafting.decided = decision;
    }

    fn set_fixed(&mut self, fixed: Option<usize>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.stats.fixed = fixed;
        }
    }

    fn follower_group(&self, py: Python<'_>) -> Option<Py<GaudiRanks>> {
        self.ranks
            .as_ref()
            .filter(|r| !r.driver)
            .map(|r| r.group.clone_ref(py))
    }

    fn replay_forward(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        Self::forward_step(slf, py, plan)
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
        slf.borrow_mut().release_ids(py, ids)
    }
}
