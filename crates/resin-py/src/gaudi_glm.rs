//! The native step executor of GLM-5.3-Flash on Gaudi (cargo feature `gaudi`).
//!
//! Python builds the model at startup: it loads the weights, compiles the recipes and builds every step
//! program (`Glm53Gaudi._native_program`: the embedding, each layer's launches and collectives, the head and the
//! logits all-gather, for every (bucket, indexer, sparse, verify) shape), the MTP layer's pass programs
//! (`_mtp_native`) and per bucket I/O programs (`glm5_3_flash.native`: the table images' ranged uploads, the
//! samplers, the outputs' downloads, a prefill chunk's head at scored columns, the MTP side's hidden columns, tokens,
//! final hiddens, commit and chain copies). Each step then runs here without Python: the tables go into pinned
//! host copies of the table images (`resin_gaudi::glm`), the uploads copy the ranges the fills used, the programs
//! make the launches, and the samples come back through pinned staging buffers behind events. The runner hands
//! the Engine this executor's `forward`, `enqueue` and `release`.
//!
//! Steps: prefill chunks (each row's in order, prompt logprobs scored on the host), plain decode steps
//! (synchronous, or enqueued: a continuation's embedding reads the previous step's samples on the device), and
//! with MTP drafts verify steps (the target over [anchor, drafts] of each row and the MTP layer at every column;
//! acceptance and the KDA commit after the samples; the draft chain, one MTP pass a draft, where the next step
//! continues the row). Whether a decode step verifies drafts is the draft policy's (`resin_gaudi::draft_choice`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use numpy::PyReadwriteArray1;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use resin_gaudi::draft_choice::{DraftChoice, Kind, Mode, StepCosts};
use resin_gaudi::qwen_draft::{
    self, BlockLayout, DraftRow, InjectLayout, Injected, StateGeometry, StateRow,
};
use resin_gaudi::qwen_mtp::{CommitLayout, fill_commit};
use resin_gaudi::sampling::{self, PARAMS, TOP_MAX as TOP};
use resin_gaudi::{glm, speculative, step};

use crate::cache::CacheLayout;
use crate::gaudi::{GaudiGlue, GaudiProgram, GlmFill, item};
use crate::gaudi_exec::{
    Images, Outputs, Pinned, Sampled, ScoreRequest, Scored, Timings, bucket_of, check_paged_plan,
    group_blocks, plan_params, post_accept, prompt_columns, row_sample, score_columns,
    score_position, score_requests, since, write_sample_table,
};
use crate::gaudi_gate::CardGate;
use crate::gaudi_ranks::{
    GaudiRanks, OP_CLOSE, OP_ENQUEUE, OP_FORWARD, OP_RELEASE, Ranked, Replay,
};
use crate::output::ForwardOutput;
use crate::step_trace;
use crate::{BatchPhase, BatchPlan};

// the registers of a step's programs: the step program's (`model._R_*`), the I/O programs', the MTP pass
// program's (`model._M_*`) and the chain's copies
const R_DEVICE_TOKENS: usize = 0;
const R_INDEX: usize = 1;
const R_SLOTS: usize = 2;
/// which host copy of the table image the uploads read (0 or 1)
pub(crate) const R_IMAGE: usize = 3;
/// which staging set the samples go to (0 or 1)
pub(crate) const R_STAGE: usize = 4;
/// bytes of the slot records and of the index records to upload
pub(crate) const R_SLOT_BYTES: usize = 5;
pub(crate) const R_INDEX_BYTES: usize = 6;
/// the MTP side's: its image's host copy, slot and index record bytes, staging set
pub(crate) const R_MIMAGE: usize = 7;
pub(crate) const R_MSLOT_BYTES: usize = 8;
pub(crate) const R_MINDEX_BYTES: usize = 9;
pub(crate) const R_MSTAGE: usize = 10;
/// the chain's rows whose accepted MTP hidden it copies, and (from R_ACC) each row's accepted drafts
pub(crate) const R_CHAIN_ROWS: usize = 11;
/// the drafter's: its block image's host copy, the bytes of it, its slot launches a layer, and the injection
/// image's host copy
pub(crate) const R_BIMAGE: usize = 12;
pub(crate) const R_BBYTES: usize = 13;
pub(crate) const R_BLAUNCHES: usize = 14;
pub(crate) const R_DIMAGE: usize = 15;
const M_PREV: usize = 16;
const M_HF: usize = 17;
const M_TOK: usize = 18;
const M_HF_OUT: usize = 19;
const M_INDEX: usize = 20;
const M_SLOTS: usize = 21;
pub(crate) const R_ACC: usize = 22;
const ACC_ROWS: usize = 8;
pub(crate) const REGISTERS: usize = R_ACC + ACC_ROWS;

/// The target I/O program's sections.
const IO_SECTIONS: [&str; 12] = [
    "upload",
    "upload_blocks",
    "upload_slots",
    "upload_index",
    "uploaded",
    "sample",
    "sample_last",
    "out",
    "head_scored",
    "scores",
    "inject",
    "commit",
];
/// the image's header, row and KDA tables (bytes fixed by the bucket)
const IO_UPLOAD: usize = 0;
/// a sparse step's block table
const IO_UPLOAD_BLOCKS: usize = 1;
/// `R_SLOT_BYTES` of slot records, `R_INDEX_BYTES` of index records
const IO_UPLOAD_SLOTS: usize = 2;
const IO_UPLOAD_INDEX: usize = 3;
/// the host copy `R_IMAGE`'s event, after its uploads
const IO_UPLOADED: usize = 4;
/// a decode step's sampler over every column
const IO_SAMPLE: usize = 5;
/// a prefill chunk's sampler over its last column (its head's first column)
const IO_SAMPLE_LAST: usize = 6;
/// the sampler's outputs to staging set `R_STAGE`, then its event
const IO_OUT: usize = 7;
/// a prefill chunk's head at the head columns' table (prompt logprobs), then the logits all-gather
const IO_HEAD_SCORED: usize = 8;
/// every rank's logits of the head columns to the pinned score buffer, then the score event
const IO_SCORES: usize = 9;
/// the drafter's injection: host copy `R_DIMAGE` of its image to the drafter's buffers, its event, the launch
const IO_INJECT: usize = 10;
/// a verify step's commit: its tables (ring uploads from the commit's pinned tables) and the commit launch
const IO_COMMIT: usize = 11;

/// The MTP side's I/O program's sections (`native._mtp_io`).
const MIO_SECTIONS: [&str; 13] = [
    "upload",
    "upload_blocks",
    "upload_slots",
    "upload_index",
    "uploaded",
    "cols",
    "tok",
    "sample",
    "out",
    "final",
    "hidden",
    "commit",
    "chain_hf",
];
const MIO_COLS: usize = 5;
const MIO_TOK: usize = 6;
const MIO_SAMPLE: usize = 7;
const MIO_OUT: usize = 8;
const MIO_FINAL: usize = 9;
const MIO_HIDDEN: usize = 10;
const MIO_CHAIN_HF: usize = 12;

/// (pinned staging sets, stride, first event slot, (tok, lp, top_idx, top_lp, bytes) offsets)
type StagingArgs<'py> = (
    PyReadwriteArray1<'py, u8>,
    usize,
    i32,
    (usize, usize, usize, usize, usize),
);

/// One bucket's I/O program and its sections (by the program's list of names).
struct Shaped {
    io: Arc<step::Program>,
    sections: Vec<Option<usize>>,
}

impl Shaped {
    fn of(program: Arc<step::Program>, names: &[&str]) -> Self {
        let sections = names
            .iter()
            .map(|name| program.sections().iter().position(|s| s.name == *name))
            .collect();
        Self {
            io: program,
            sections,
        }
    }

    fn section(&self, t: usize, k: usize, names: &[&str]) -> PyResult<(Arc<step::Program>, usize)> {
        let at = self.sections[k].ok_or_else(|| {
            PyValueError::new_err(format!("bucket {t} has no {} section", names[k]))
        })?;
        Ok((Arc::clone(&self.io), at))
    }
}

/// Prompt logprobs: the pinned head columns' table (HEAD_COLS i32), the pinned buffer each rank's logits of the
/// head columns come back to (`stride` bytes a rank, HEAD_COLS columns of `vocab`), and its event.
struct ScoreSide {
    columns: Pinned,
    host: Pinned,
    stride: usize,
    vocab: usize,
    ranks: usize,
    event: i32,
}

/// Two pinned staging sets behind events, each held by the step whose samples it holds until they are read.
struct Staging {
    host: Pinned,
    stride: usize,
    events: i32,
    next: usize,
    owner: [Option<u64>; 2],
}

/// The cache geometry of the rows' block addresses (`CacheGeometry`).
#[derive(Clone, Copy)]
struct Geometry {
    slab_rows: i64,
    kv_block_rows: i64,
    state_block_rows: i64,
    state_tokens: i64,
    mtp_block_rows: i64,
    draft_block_rows: i64,
}

/// A step's rows, flat (`glm::Rows` borrows them).
#[derive(Default)]
struct StepRows {
    column: Vec<i64>,
    pos0: Vec<i64>,
    real: Vec<i64>,
    state_src: Vec<i64>,
    state_dst: Vec<i64>,
    blocks_indptr: Vec<i64>,
    blocks: Vec<i64>,
}

impl StepRows {
    fn new() -> Self {
        Self {
            blocks_indptr: vec![0],
            ..Self::default()
        }
    }

    fn push(&mut self, column: i64, pos0: i64, real: i64, state: (i64, i64), blocks: &[i64]) {
        self.column.push(column);
        self.pos0.push(pos0);
        self.real.push(real);
        self.state_src.push(state.0);
        self.state_dst.push(state.1);
        self.blocks.extend_from_slice(blocks);
        self.blocks_indptr.push(self.blocks.len() as i64);
    }

    fn total(&self) -> usize {
        self.real.iter().sum::<i64>() as usize
    }

    fn view(&self) -> glm::Rows<'_> {
        glm::Rows {
            column: &self.column,
            pos0: &self.pos0,
            real: &self.real,
            state_src: &self.state_src,
            state_dst: &self.state_dst,
            blocks_indptr: &self.blocks_indptr,
            blocks: &self.blocks,
        }
    }
}

/// One row's cache blocks from the plan: KV block rows, STATE block rows from logical block `state_first`, and
/// with MTP the MTP layer's KV block rows.
#[derive(Clone)]
struct Blocks {
    kv: Vec<i64>,
    states: Vec<i64>,
    state_first: i64,
    mtp: Vec<i64>,
    /// with the drafter: its DRAFT blocks (a window), the first one's logical block
    draft: Vec<i64>,
    draft_first: i64,
}

impl Blocks {
    /// The STATE block holding `position` (`_state`).
    fn state(&self, r: usize, position: i64, tokens: i64) -> PyResult<i64> {
        let index = position / tokens - self.state_first;
        usize::try_from(index)
            .ok()
            .and_then(|i| self.states.get(i).copied())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "row {r} has no STATE block for position {position}"
                ))
            })
    }

    /// The drafter's view of the row with its block at `position` (`Row.draft_row`).
    fn draft_row(&self, position: i64) -> DraftRow<'_> {
        DraftRow {
            position,
            blocks: &self.draft,
            first: self.draft_first,
        }
    }

    /// (state_src, state_dst) of `real` tokens from `pos0` (`_row`).
    fn states(&self, r: usize, pos0: i64, real: i64, tokens: i64) -> PyResult<(i64, i64)> {
        let src = if pos0 == 0 {
            -1
        } else {
            self.state(r, pos0 - 1, tokens)?
        };
        Ok((src, self.state(r, pos0 + real - 1, tokens)?))
    }
}

/// What an MTP pass reads its tokens from.
enum Tokens<'a> {
    Host(&'a [i64]),
    Device(u64),
}

/// The MTP layer's side: its table image, I/O programs, pass programs and pinned buffers, and the device buffers
/// its passes bind through registers.
struct Mtp {
    image: Images,
    tables: HashMap<usize, glm::Tables>,
    io: HashMap<usize, Shaped>,
    /// by (columns, hidden columns, indexer, sparse)
    programs: HashMap<(usize, usize, bool, bool), Arc<step::Program>>,
    cols: Pinned,
    tok: Pinned,
    sample: Pinned,
    hidden_table: Pinned,
    staging: Staging,
    hf: [u64; 2],
    io_hf: u64,
    zero: u64,
    tok_out: u64,
    mtp_tok: u64,
    arena: u64,
    state: StateGeometry,
    row_bytes: i64,
}

/// A verify step's commit (either drafter's): the pinned tables `qwen_mtp::fill_commit` writes (the bases at 0, the
/// counts and the hidden table at `at`), the STATE geometry and the KDA layers.
struct Commit {
    host: Pinned,
    at: (usize, usize),
    state: StateGeometry,
    layers: usize,
}

impl Mtp {
    fn io(&self, t: usize, k: usize) -> PyResult<(Arc<step::Program>, usize)> {
        self.io
            .get(&t)
            .ok_or_else(|| PyValueError::new_err(format!("no MTP I/O program of {t} columns")))?
            .section(t, k, &MIO_SECTIONS)
    }

    /// A STATE block's hidden slot as a device address (`hidden_row` x ROW_BYTES), the zero hidden for none.
    fn hidden_address(&self, block: i64) -> u64 {
        if block < 0 {
            self.zero
        } else {
            self.arena + ((block + self.state.hidden_offset) * self.row_bytes) as u64
        }
    }
}

/// The DFlash 2 drafter's side: its tables' settings, the injection image, the block image and the block step
/// programs by row bucket, and the pinned buffer the next drafts come back to (its program and event).
struct Dflash {
    drafter: qwen_draft::Drafter,
    row_buckets: Vec<usize>,
    layouts: HashMap<usize, BlockLayout>,
    blocks: Images,
    programs: HashMap<usize, Arc<step::Program>>,
    inject: Images,
    inject_kv_rows: usize,
    inject_kv_stride: usize,
    next: Pinned,
    next_program: Arc<step::Program>,
    next_event: i32,
    max_positions: i64,
}

/// A row's drafts carried to its next step (`_drafts`): the anchor's position and token, the drafts, and whether
/// they are all the row's own (a chain made them) rather than the first and zeros.
#[derive(Clone)]
struct Carried {
    position: i64,
    anchor: i64,
    drafts: Vec<i64>,
    full: bool,
}

/// One row of a verify step: what its acceptance, commit and chain read.
#[derive(Clone)]
struct VerifyRow {
    sequence: u64,
    start: i64,
    blocks: Blocks,
    state_src: i64,
    sample: sampling::Sample,
}

#[pyclass(module = "resin._resin")]
pub(crate) struct GlmExecutor {
    glue: Py<GaudiGlue>,
    model: glm::Model,
    tables: HashMap<usize, glm::Tables>,
    /// the step programs by (bucket, indexer on, sparse, verify columns a row)
    steps: HashMap<(usize, bool, bool, usize), Arc<step::Program>>,
    shapes: HashMap<usize, Shaped>,
    layout: Py<CacheLayout>,
    geometry: Geometry,
    image: Images,
    staging: Staging,
    sample_host: Pinned,
    /// the sampler table's `last` field (byte offset in it)
    sample_last: usize,
    sampled: Sampled,
    vocab: u64,
    prefill_buckets: Vec<usize>,
    decode_buckets: Vec<usize>,
    /// a decode step's most columns
    max_t: usize,
    /// a prefill chunk's head columns
    head_cols: usize,
    max_batch_size: usize,
    max_seq_len: i64,
    spin_ns: i64,
    registers: [u64; REGISTERS],
    /// enqueued steps whose samples are not read yet, oldest first
    inflight: Vec<Py<GlmCompletion>>,
    next_id: u64,
    timings: Timings,
    native_steps: u64,
    /// enqueued steps that continued the previous one (`Glm53FlashGaudiRunner.continuations`)
    continuations: u64,
    scoring: Option<ScoreSide>,
    /// drafts a verify step verifies (0: none), its columns a row, and the rows of one verify launch
    k: usize,
    tq: usize,
    speculative_rows: usize,
    /// decode steps of up to this many rows may verify drafts and lease `lookahead` positions past each row
    spec_rows: usize,
    lookahead: i64,
    mtp: Option<Mtp>,
    dflash: Option<Dflash>,
    commit: Option<Commit>,
    stats: speculative::Stats,
    choice: DraftChoice,
    /// without a native rank group: the first rank's step times to every rank (`ranks.from_first_rank`)
    share: Option<Py<PyAny>>,
    carried: HashMap<u64, Carried>,
    /// the last enqueued verify step, until a step that does not continue it settles it
    open: Option<Py<GlmVerification>>,
    /// the last enqueued decode step: (kind, rows, time of its choice, whether it switched kind or started a run)
    timed: Option<(Kind, usize, Instant, bool)>,
    /// this executor as a Python object, for the verifications it makes (set by its first call)
    this: Option<Py<GlmExecutor>>,
    /// a tensor-parallel model's rank group (`attach_ranks`)
    ranks: Option<Ranked>,
    /// a follower: the driver's draft decision for the call it replays
    forced: Option<bool>,
    /// a layout's card gate, which a replica on this rank's card shares (`attach_gate`)
    card_gate: Option<Py<CardGate>>,
}

/// Fill host copy `k` of `image` with a step's tables (`tables`); returns the fill's counts.
fn fill_image(
    py: Python<'_>,
    model: &glm::Model,
    tables: &glm::Tables,
    image: &Images,
    k: usize,
    rows: &StepRows,
    step: &glm::Step<'_>,
) -> PyResult<glm::Counts> {
    // SAFETY: host copy k's last upload is done (take_image), and no device copy reads it until this step's.
    let host = unsafe { image.host.bytes_mut(k * image.stride, tables.nbytes) };
    let rows = rows.view();
    py.detach(|| glm::fill(model, tables, host, &rows, step))
        .map_err(PyValueError::new_err)
}

/// i32 words into a pinned buffer that only a later ring upload reads (from its start).
fn write_words(pinned: &Pinned, values: &[i64]) {
    // SAFETY: the buffer is read by a submit's ring upload, which copies it at once; no submit runs meanwhile.
    let bytes = unsafe { pinned.bytes_mut(0, 4 * values.len()) };
    for (j, &value) in values.iter().enumerate() {
        bytes[4 * j..4 * j + 4].copy_from_slice(&(value as i32).to_le_bytes());
    }
}

impl GlmExecutor {
    fn decode_bucket(&self, n: usize) -> PyResult<usize> {
        bucket_of(&self.decode_buckets, n, false)
    }

    fn mtp(&self) -> PyResult<&Mtp> {
        self.mtp
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the model has no MTP layer"))
    }

    /// Row `r`'s KV, STATE and MTP blocks (`_blocks`, `_state`, `_row`).
    fn blocks(&self, plan: &BatchPlan, r: usize) -> PyResult<Blocks> {
        let g = self.geometry;
        let (kv, _) = group_blocks(plan, 0, r, g.slab_rows, g.kv_block_rows)?;
        let (states, state_first) = group_blocks(plan, 1, r, g.slab_rows, g.state_block_rows)?;
        let mtp = if self.mtp.is_some() {
            group_blocks(plan, 2, r, g.slab_rows, g.mtp_block_rows)?.0
        } else {
            Vec::new()
        };
        let (draft, draft_first) = match &self.dflash {
            Some(_) => group_blocks(plan, 2, r, g.slab_rows, g.draft_block_rows)?,
            None => (Vec::new(), 0),
        };
        Ok(Blocks {
            kv,
            states,
            state_first,
            mtp,
            draft,
            draft_first,
        })
    }

    fn wait_event(&mut self, py: Python<'_>, event: i32) -> PyResult<()> {
        let glue = self.glue.get();
        let spin = self.spin_ns;
        let start = Instant::now();
        py.detach(|| glue.glue.wait(event, spin))
            .map_err(|error| post_accept(py, error))?;
        self.timings.output_wait += since(start);
        Ok(())
    }

    /// The next host copy of an image (`mtp`: the MTP side's), once its last upload is done.
    fn take_image(&mut self, py: Python<'_>, mtp: bool) -> PyResult<usize> {
        let image = if mtp {
            &mut self.mtp.as_mut().expect("an MTP layer").image
        } else {
            &mut self.image
        };
        Self::take_copy(py, &self.glue, self.spin_ns, &mut self.timings, image)
    }

    /// The next host copy of `image`, once its last upload is done.
    fn take_copy(
        py: Python<'_>,
        glue: &Py<GaudiGlue>,
        spin: i64,
        timings: &mut Timings,
        image: &mut Images,
    ) -> PyResult<usize> {
        let k = image.next;
        image.next ^= 1;
        if image.used[k] {
            let event = image.events + k as i32;
            let glue = glue.get();
            let start = Instant::now();
            py.detach(|| glue.glue.wait(event, spin))
                .map_err(|error| post_accept(py, error))?;
            timings.image_wait += since(start);
        }
        Ok(k)
    }

    // ------------------------------------------------------------------ the DFlash drafter
    fn dflash(&self) -> PyResult<&Dflash> {
        self.dflash
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the model has no DFlash drafter"))
    }

    /// Fill a host copy of the injection image with a target pass's columns (`Drafter.inject`): `runs` of
    /// (column, position, count, blocks); returns the section to run after the pass (bucket `t`'s `inject`).
    fn inject(
        &mut self,
        py: Python<'_>,
        t: usize,
        runs: &[(usize, i64, usize, &Blocks)],
    ) -> PyResult<(Arc<step::Program>, usize)> {
        let k = {
            let d = self.dflash.as_mut().expect("a drafter");
            Self::take_copy(
                py,
                &self.glue,
                self.spin_ns,
                &mut self.timings,
                &mut d.inject,
            )?
        };
        let start = Instant::now();
        {
            let d = self.dflash()?;
            let layout = InjectLayout {
                tokens: t,
                pos: 0,
                kv_rows: d.inject_kv_rows,
                kv_stride: d.inject_kv_stride,
            };
            let injected: Vec<Injected<'_>> = runs
                .iter()
                .map(|&(column, position, count, blocks)| Injected {
                    column,
                    position,
                    count,
                    row: blocks.draft_row(position),
                })
                .collect();
            let nbytes = layout.kv_rows + d.drafter.layers * layout.kv_stride;
            // SAFETY: host copy k's last upload is done (take_copy), and no device copy reads it until this pass's.
            let host = unsafe { d.inject.host.bytes_mut(k * d.inject.stride, nbytes) };
            qwen_draft::fill_inject(&d.drafter, &layout, host, nbytes, &injected)
                .map_err(PyValueError::new_err)?;
        }
        self.timings.fill += since(start);
        self.registers[R_DIMAGE] = k as u64;
        self.dflash.as_mut().expect("a drafter").inject.used[k] = true;
        self.io(t, IO_INJECT)
    }

    /// `_draft_blocks`: one block step of the drafter after each row's anchor (`anchors` [b] at `rows` [b]'s
    /// position) -> each row's next K drafts (waits).
    fn draft_blocks(
        &mut self,
        py: Python<'_>,
        rows: &[(i64, &Blocks)],
        anchors: &[i64],
    ) -> PyResult<Vec<Vec<i64>>> {
        let b = rows.len();
        let width = self.tq;
        let bucket = self
            .dflash()?
            .row_buckets
            .iter()
            .copied()
            .find(|&r| r >= b)
            .ok_or_else(|| PyValueError::new_err(format!("{b} rows exceed the block steps")))?;
        let k = {
            let d = self.dflash.as_mut().expect("a drafter");
            Self::take_copy(
                py,
                &self.glue,
                self.spin_ns,
                &mut self.timings,
                &mut d.blocks,
            )?
        };
        let start = Instant::now();
        let filled = {
            let d = self.dflash()?;
            let layout = &d.layouts[&bucket];
            let draft: Vec<DraftRow<'_>> = rows
                .iter()
                .map(|(p, blocks)| blocks.draft_row(*p))
                .collect();
            // SAFETY: host copy k's last upload is done (take_copy), and no device copy reads it until this step's.
            let host = unsafe { d.blocks.host.bytes_mut(k * d.blocks.stride, layout.nbytes) };
            qwen_draft::fill_block(
                &d.drafter,
                layout,
                host,
                &draft,
                Some(anchors),
                None,
                width,
                d.max_positions,
            )
            .map_err(PyValueError::new_err)?
        };
        self.timings.fill += since(start);
        let r = &mut self.registers;
        r[R_BIMAGE] = k as u64;
        r[R_BBYTES] = filled.nbytes as u64;
        r[R_BLAUNCHES] = filled.launches as u64;
        let (steps, event) = {
            let d = self.dflash()?;
            let program = Arc::clone(&d.programs[&bucket]);
            (
                vec![
                    (Arc::clone(&program), 0),
                    (program, 1),
                    (Arc::clone(&d.next_program), 0),
                ],
                d.next_event,
            )
        };
        self.dflash.as_mut().expect("a drafter").blocks.used[k] = true;
        self.submit(py, &steps, "block step")?;
        self.wait_event(py, event)?;
        let next: Vec<i32> = self.dflash()?.next.words(0, bucket * width);
        Ok((0..b)
            .map(|i| {
                next[i * width + 1..(i + 1) * width]
                    .iter()
                    .map(|&t| i64::from(t))
                    .collect()
            })
            .collect())
    }

    /// Fill host copy `k` of the target's image with a step's tables (bucket `t`).
    fn fill(
        &mut self,
        py: Python<'_>,
        k: usize,
        t: usize,
        rows: &StepRows,
        step: &glm::Step<'_>,
    ) -> PyResult<glm::Counts> {
        let tables = self
            .tables
            .get(&t)
            .ok_or_else(|| PyValueError::new_err(format!("no tables of {t} columns")))?;
        let start = Instant::now();
        let counts = fill_image(py, &self.model, tables, &self.image, k, rows, step)?;
        self.timings.fill += since(start);
        Ok(counts)
    }

    fn io(&self, t: usize, section: usize) -> PyResult<(Arc<step::Program>, usize)> {
        self.shapes
            .get(&t)
            .ok_or_else(|| PyValueError::new_err(format!("no I/O program of {t} columns")))?
            .section(t, section, &IO_SECTIONS)
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
            // sampler tables that this executor keeps, and recipes and event slots of the loaded runtime.
            let mut stream = unsafe { step::CardStream::new(&glue.glue) };
            step::run_sequence(&steps, &registers, &mut stream)
        })
        .map_err(|error| post_accept(py, format!("GLM step {name}: {}", error.message)))?;
        self.timings.submit += since(start);
        Ok(())
    }

    /// Queue a step of bucket `t` whose tables are in host copy `k`: the uploads of the ranges the fill used, the
    /// copy's event, the step program of the fill's shape (`verify` columns a row), and `then`.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &mut self,
        py: Python<'_>,
        k: usize,
        t: usize,
        counts: &glm::Counts,
        verify: usize,
        device_tokens: bool,
        then: &[(Arc<step::Program>, usize)],
        name: &str,
    ) -> PyResult<()> {
        let key = (t, counts.index, counts.sparse, verify);
        let program = self.steps.get(&key).map(Arc::clone).ok_or_else(|| {
            PyValueError::new_err(format!(
                "no step program of {t} columns (indexer {}, sparse {}, verify {verify})",
                counts.index, counts.sparse
            ))
        })?;
        let r = &mut self.registers;
        r[R_DEVICE_TOKENS] = u64::from(device_tokens);
        r[R_INDEX] = counts.index_launches as u64;
        r[R_SLOTS] = counts.slots as u64;
        r[R_IMAGE] = k as u64;
        r[R_SLOT_BYTES] = counts.uploads[2].1 as u64;
        r[R_INDEX_BYTES] = counts.uploads[3].1 as u64;
        let mut steps = vec![self.io(t, IO_UPLOAD)?];
        for (range, section) in [
            (1, IO_UPLOAD_BLOCKS),
            (2, IO_UPLOAD_SLOTS),
            (3, IO_UPLOAD_INDEX),
        ] {
            if counts.uploads[range].1 > 0 {
                steps.push(self.io(t, section)?);
            }
        }
        steps.push(self.io(t, IO_UPLOADED)?);
        steps.push((program, 0));
        steps.extend(then.iter().map(|(p, s)| (Arc::clone(p), *s)));
        self.image.used[k] = true;
        self.submit(py, &steps, name)?;
        self.native_steps += 1;
        Ok(())
    }

    /// The next staging set (`mtp`: the MTP side's), once the step that holds its samples has read them.
    fn take_stage(&mut self, py: Python<'_>, mtp: bool) -> PyResult<usize> {
        let staging = if mtp {
            &mut self.mtp.as_mut().expect("an MTP layer").staging
        } else {
            &mut self.staging
        };
        let k = staging.next;
        staging.next ^= 1;
        if let Some(owner) = staging.owner[k].take() {
            let open: Vec<Py<GlmCompletion>> = self
                .inflight
                .iter()
                .filter(|c| c.borrow(py).id == owner)
                .map(|c| c.clone_ref(py))
                .collect();
            for completion in open {
                GlmCompletion::finish(&completion.bind(py).clone(), py, self)?;
            }
            if let Some(v) = self.open.as_ref().map(|v| v.clone_ref(py))
                && v.borrow(py).id == owner
            {
                GlmVerification::settle(v.bind(py), py, self)?;
            }
        }
        Ok(k)
    }

    fn wait_stage(&mut self, py: Python<'_>, k: usize, mtp: bool) -> PyResult<()> {
        let events = if mtp {
            self.mtp()?.staging.events
        } else {
            self.staging.events
        };
        self.wait_event(py, events + k as i32)
    }

    /// The first `n` columns' samples in staging set `k` (`mtp`: the MTP side's), once its event is done.
    fn staged(&self, k: usize, n: usize, mtp: bool) -> Outputs {
        let staging = if mtp {
            &self.mtp.as_ref().expect("an MTP layer").staging
        } else {
            &self.staging
        };
        Outputs::read(&staging.host, k * staging.stride, self.sampled, n)
    }

    /// The target's sampler table (the rows' parameters, then `last`).
    fn sample_table(&mut self, params: &[[f32; PARAMS]], last: i64) {
        // SAFETY: the pinned table is only read by a submit's upload_ring, which copies it at once; no submit runs
        // while the executor writes.
        let table = unsafe { self.sample_host.bytes_mut(0, self.sample_host.bytes) };
        write_sample_table(table, params, self.sample_last, last);
    }

    /// The sampler over the step just queued (`sample_launched`): every column of decode bucket `t`, or a prefill
    /// chunk's last column (`last`), into a staging set; waits for it and reads the first `params.len()` columns.
    fn sample_now(
        &mut self,
        py: Python<'_>,
        t: usize,
        params: &[[f32; PARAMS]],
        last: bool,
    ) -> PyResult<Outputs> {
        let k = self.take_stage(py, false)?;
        self.sample_table(params, 0);
        self.registers[R_STAGE] = k as u64;
        let section = if last { IO_SAMPLE_LAST } else { IO_SAMPLE };
        let steps = [self.io(t, section)?, self.io(t, IO_OUT)?];
        self.submit(py, &steps, "sample")?;
        self.wait_stage(py, k, false)?;
        Ok(self.staged(k, params.len(), false))
    }

    /// Read every enqueued step's samples and settle the open verify step (a synchronous step follows them).
    fn harvest(&mut self, py: Python<'_>) -> PyResult<()> {
        while let Some(completion) = self.inflight.first().map(|c| c.clone_ref(py)) {
            GlmCompletion::finish(&completion.bind(py).clone(), py, self)?;
        }
        Ok(())
    }

    /// `_settle`: finish the open verify step unless `continued` continues it, so its commit lands first.
    fn settle_open(
        &mut self,
        py: Python<'_>,
        continued: Option<&Py<GlmVerification>>,
    ) -> PyResult<()> {
        if let Some(open) = self.open.take()
            && continued.is_none_or(|c| !c.is(&open))
        {
            GlmVerification::settle(open.bind(py), py, self)?;
        }
        Ok(())
    }

    /// `_decode_lookahead`: the positions the Engine leases past each decode row's token.
    fn decode_lookahead(&self, rows: usize) -> i64 {
        if rows <= self.spec_rows {
            self.lookahead
        } else {
            0
        }
    }

    /// `PagedCachePlan.from_plan`'s checks, and the top lists' length (`enqueue`'s checks).
    fn check_cache(&self, plan: &BatchPlan) -> PyResult<()> {
        let lookahead = self.decode_lookahead(plan.sequence_ids.len()) as u64;
        check_paged_plan(plan, self.layout.get().inner_native(), lookahead)
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
        if longest + self.lookahead > self.max_seq_len {
            return Err(PyValueError::new_err(format!(
                "a row exceeds {} positions",
                self.max_seq_len
            )));
        }
        if plan.needs_prompt_logprobs() && plan.phase != BatchPhase::Prefill {
            return Err(PyValueError::new_err(
                "prompt logprobs are scored by prefill plans",
            ));
        }
        Ok(())
    }

    // ------------------------------------------------------------------ the draft policy
    /// `_speculating`: whether this decode step verifies drafts, once a step (the policy's `speculate`); without a
    /// native rank group the ranks first share the first rank's times at the policy's sync steps.
    fn speculating(&mut self, py: Python<'_>, plan: &BatchPlan) -> PyResult<bool> {
        if let Some(decision) = self.forced {
            return Ok(decision); // a follower takes the driver's
        }
        if self.choice.sync_due()
            && let Some(share) = self.share.as_ref().map(|s| s.clone_ref(py))
        {
            let table = PyDict::new(py);
            for ((kind, bucket), ms) in self.choice.times.table() {
                table.set_item((kind.name(), bucket), ms)?;
            }
            let shared: HashMap<(String, usize), f64> = share.call1(py, (table,))?.extract(py)?;
            let shared = shared
                .into_iter()
                .filter_map(|((kind, bucket), ms)| Kind::of(&kind).map(|k| ((k, bucket), ms)))
                .collect();
            self.choice.set_shared(shared);
        }
        Ok(self.choice.speculate(&plan.sequence_ids))
    }

    /// `_time_step`: at this decode step's choice, the time since the last step's choice is the last step's time
    /// when this step continues it with a step of the same kind and bucket.
    fn time_step(&mut self, kind: Kind, rows: usize, continues: bool) {
        let now = Instant::now();
        if let Some((last_kind, last_rows, at, switched)) = self.timed
            && continues
            && last_kind == kind
            && self.choice.bucket(last_rows) == self.choice.bucket(rows)
            && !switched
        {
            let ms = now.duration_since(at).as_secs_f64() * 1e3;
            self.choice.measured(kind, last_rows, ms);
        }
        let switched = !continues || self.timed.is_none_or(|t| t.0 != kind);
        self.timed = Some((kind, rows, now, switched));
    }

    /// `_carried`: each row's drafts carried from its last step (zeros when they do not continue from this
    /// anchor), and which rows carried drafts of their own.
    fn carried_drafts(
        &self,
        sequences: &[u64],
        positions: &[i64],
        anchors: &[i64],
    ) -> (Vec<Vec<i64>>, Vec<bool>) {
        let mut drafts = Vec::with_capacity(sequences.len());
        let mut own = Vec::with_capacity(sequences.len());
        for (r, sequence) in sequences.iter().enumerate() {
            match self.carried.get(sequence) {
                Some(c) if c.position == positions[r] && c.anchor == anchors[r] => {
                    drafts.push(c.drafts.clone());
                    own.push(c.full);
                }
                _ => {
                    drafts.push(vec![0; self.k]);
                    own.push(false);
                }
            }
        }
        (drafts, own)
    }

    // ------------------------------------------------------------------ the MTP layer
    /// One MTP pass (`_mtp_pass`): bucket `t` reading `columns` hidden columns, column c at its row's position
    /// with token `tokens` [c] and hidden column `cols` [c] of [`prev` | `hf`]. With `params` the whole layer and
    /// the head (its hidden into `hf_out`) and the sampler into an MTP staging set (returned), else its keys only.
    #[allow(clippy::too_many_arguments)]
    fn mtp_pass(
        &mut self,
        py: Python<'_>,
        t: usize,
        columns: usize,
        rows: &StepRows,
        cols: &[i64],
        tokens: Tokens<'_>,
        prev: u64,
        hf: u64,
        hf_out: u64,
        params: Option<&[[f32; PARAMS]]>,
        before: &[(Arc<step::Program>, usize)],
    ) -> PyResult<Option<usize>> {
        let k = self.take_image(py, true)?;
        let start = Instant::now();
        let counts = {
            let mtp = self.mtp()?;
            let tables = mtp
                .tables
                .get(&t)
                .ok_or_else(|| PyValueError::new_err(format!("no MTP tables of {t} columns")))?;
            let step = glm::Step {
                tokens: None,
                columns: t,
                first_position: rows.pos0[0],
                head: false,
                verify: 0,
                keep_tails: true,
                meta_columns: Some(rows.total()),
            };
            fill_image(py, &self.model, tables, &mtp.image, k, rows, &step)?
        };
        self.timings.fill += since(start);
        let layer = params.is_some();
        let key = (t, columns, counts.index && layer, counts.sparse && layer);
        let (program, mut steps) = {
            let mtp = self.mtp()?;
            let program = mtp.programs.get(&key).map(Arc::clone).ok_or_else(|| {
                PyValueError::new_err(format!("no MTP pass program of shape {key:?}"))
            })?;
            write_words(&mtp.cols, cols);
            let mut steps: Vec<(Arc<step::Program>, usize)> =
                before.iter().map(|(p, s)| (Arc::clone(p), *s)).collect();
            steps.push(mtp.io(t, 0)?);
            for (range, section) in [(1, 1), (2, 2), (3, 3)] {
                if counts.uploads[range].1 > 0 {
                    steps.push(mtp.io(t, section)?);
                }
            }
            steps.push(mtp.io(t, 4)?);
            steps.push(mtp.io(t, MIO_COLS)?);
            (program, steps)
        };
        let tok = match tokens {
            Tokens::Host(tokens) => {
                let mtp = self.mtp()?;
                let mut padded = tokens.to_vec();
                padded.resize(t, 0);
                write_words(&mtp.tok, &padded);
                steps.push(mtp.io(t, MIO_TOK)?);
                mtp.mtp_tok
            }
            Tokens::Device(address) => address,
        };
        let r = &mut self.registers;
        r[R_MIMAGE] = k as u64;
        r[R_MSLOT_BYTES] = counts.uploads[2].1 as u64;
        r[R_MINDEX_BYTES] = counts.uploads[3].1 as u64;
        r[M_PREV] = prev;
        r[M_HF] = hf;
        r[M_TOK] = tok;
        r[M_HF_OUT] = hf_out;
        r[M_INDEX] = counts.index_launches as u64;
        r[M_SLOTS] = counts.slots as u64;
        let section = |name: &str| -> PyResult<usize> {
            program
                .sections()
                .iter()
                .position(|s| s.name == name)
                .ok_or_else(|| PyValueError::new_err(format!("the MTP pass has no {name} section")))
        };
        steps.push((Arc::clone(&program), section("keys")?));
        let mut stage = None;
        if let Some(params) = params {
            steps.push((Arc::clone(&program), section("layer")?));
            let m = self.take_stage(py, true)?;
            {
                let mtp = self.mtp()?;
                // SAFETY: read by this submit's ring upload only, which copies it at once.
                let table = unsafe { mtp.sample.bytes_mut(0, mtp.sample.bytes) };
                write_sample_table(table, params, self.sample_last, 0);
                steps.push(mtp.io(t, MIO_SAMPLE)?);
                steps.push(mtp.io(t, MIO_OUT)?);
            }
            self.registers[R_MSTAGE] = m as u64;
            stage = Some(m);
        }
        self.mtp.as_mut().expect("an MTP layer").image.used[k] = true;
        self.submit(py, &steps, "MTP pass")?;
        Ok(stage)
    }

    /// After a plain decode step: the MTP layer's keys (`mtp_keys`), or the drafter's cache entries of each row's
    /// column (`draft_keys`), so that the row's later drafts see every position.
    fn draft_keys(&mut self, py: Python<'_>, t: usize, rows: &[(i64, Blocks)]) -> PyResult<()> {
        if self.mtp.is_some() {
            self.mtp_keys(py, t, rows)?;
        } else if self.dflash.is_some() {
            let runs: Vec<(usize, i64, usize, &Blocks)> = rows
                .iter()
                .enumerate()
                .map(|(c, (pos0, blocks))| (c, *pos0, 1, blocks))
                .collect();
            let inject = self.inject(py, t, &runs)?;
            self.submit(py, &[inject], "inject")?;
        }
        Ok(())
    }

    /// `mtp_keys`: the MTP layer's keys of a plain decode step just sampled: column c at its row's position, its
    /// sampled next token and the target's hidden at the column.
    fn mtp_keys(&mut self, py: Python<'_>, t: usize, rows: &[(i64, Blocks)]) -> PyResult<()> {
        let mut m_rows = StepRows::new();
        for (column, (pos0, blocks)) in rows.iter().enumerate() {
            m_rows.push(column as i64, *pos0, 1, (-1, -1), &blocks.mtp);
        }
        let cols: Vec<i64> = (1..=t as i64).collect();
        let (tok_out, zero, io_hf, hf0) = {
            let m = self.mtp()?;
            (m.tok_out, m.zero, m.io_hf, m.hf[0])
        };
        self.mtp_pass(
            py,
            t,
            t,
            &m_rows,
            &cols,
            Tokens::Device(tok_out),
            zero,
            io_hf,
            hf0,
            None,
            &[],
        )?;
        Ok(())
    }

    /// `prefill_mtp`: after a prefill chunk (bucket `t`, tokens `ids` from `pos0`), the MTP keys of the chunk, its
    /// last hidden into its STATE block, and after the prompt's last chunk (`draft`: its sampler inputs) the first
    /// draft following its sampled token (drafts 2 .. K start as zeros).
    #[allow(clippy::too_many_arguments)]
    fn prefill_mtp(
        &mut self,
        py: Python<'_>,
        t: usize,
        pos0: i64,
        ids: &[i64],
        blocks: &Blocks,
        states: (i64, i64),
        draft: Option<[f32; PARAMS]>,
    ) -> PyResult<Option<Vec<i64>>> {
        let real = ids.len();
        let (final_step, prev, io_hf, zero, hf0, tok_out) = {
            let m = self.mtp()?;
            (
                m.io(t, MIO_FINAL)?,
                m.hidden_address(states.0),
                m.io_hf,
                m.zero,
                m.hf[0],
                m.tok_out,
            )
        };
        // column c: position pos0 - 1 + c, token ids[c], hidden c of [prev | hf]; a sequence's first chunk starts
        // one later (no position -1)
        let shift = usize::from(pos0 == 0);
        let n = real - shift;
        if n > 0 {
            let mut tok = vec![0i64; t];
            tok[..n].copy_from_slice(&ids[shift..real]);
            let mut cols = vec![0i64; t];
            for (c, col) in cols.iter_mut().enumerate().take(n) {
                *col = (c + shift) as i64;
            }
            let mut key_rows = StepRows::new();
            key_rows.push(0, pos0 - 1 + shift as i64, n as i64, (-1, -1), &blocks.mtp);
            self.mtp_pass(
                py,
                t,
                t,
                &key_rows,
                &cols,
                Tokens::Host(&tok),
                prev,
                io_hf,
                hf0,
                None,
                &[final_step],
            )?;
        } else {
            self.submit(py, &[final_step], "final hidden")?;
        }
        let hidden = {
            let m = self.mtp()?;
            let row = if states.1 < 0 {
                -1
            } else {
                states.1 + m.state.hidden_offset
            };
            write_words(&m.hidden_table, &[real as i64 - 1, row]);
            m.io(t, MIO_HIDDEN)?
        };
        self.submit(py, &[hidden], "hidden store")?;
        let Some(params) = draft else {
            return Ok(None);
        };
        // the sampled token at pos0 + real: MTP at pos0 + real - 1 drafts pos0 + real + 1
        let mut row = StepRows::new();
        row.push(0, pos0 + real as i64 - 1, 1, (-1, -1), &blocks.mtp);
        let m = self
            .mtp_pass(
                py,
                1,
                t,
                &row,
                &[real as i64],
                Tokens::Device(tok_out),
                zero,
                io_hf,
                hf0,
                Some(&[params]),
                &[],
            )?
            .expect("a full pass samples");
        self.wait_stage(py, m, true)?;
        let first = self.staged(m, 1, true).tokens[0];
        let mut drafts = vec![0i64; self.k];
        drafts[0] = i64::from(first);
        Ok(Some(drafts))
    }

    // ------------------------------------------------------------------ steps
    /// Score positions `which` of `request` (columns `which - start` of the prefill chunk just queued, `t` its
    /// bucket) into `out` (`head_logits`, `_Scorer.add`): the head reruns at HEAD_COLS columns at a time and
    /// their logits come back to the host. Call it after the chunk's sampler: the head overwrites the logits.
    #[allow(clippy::too_many_arguments)]
    fn score_chunk(
        &mut self,
        py: Python<'_>,
        t: usize,
        request: &ScoreRequest,
        which: &[usize],
        start: usize,
        out: &mut [Scored],
    ) -> PyResult<()> {
        // the chunk's scored columns, in order (`_Scorer.columns`: unique), each with the positions it scores
        let mut columns: Vec<(usize, Vec<usize>)> = Vec::new();
        for &k in which {
            let c = request.positions[k] - start;
            match columns.iter_mut().find(|(col, _)| *col == c) {
                Some((_, ks)) => ks.push(k),
                None => columns.push((c, vec![k])),
            }
        }
        columns.sort_by_key(|(c, _)| *c);
        let head_cols = self.head_cols;
        for group in columns.chunks(head_cols) {
            let side = self.scoring.as_ref().ok_or_else(|| {
                PyNotImplementedError::new_err("prompt logprobs need the score buffer")
            })?;
            // np.resize: the group's columns repeated over HEAD_COLS
            let cols: Vec<i64> = (0..head_cols)
                .map(|j| group[j % group.len()].0 as i64)
                .collect();
            write_words(&side.columns, &cols);
            let (event, vocab, ranks, stride) = (side.event, side.vocab, side.ranks, side.stride);
            let steps = [self.io(t, IO_HEAD_SCORED)?, self.io(t, IO_SCORES)?];
            self.submit(py, &steps, "scores")?;
            self.wait_event(py, event)?;
            let side = self.scoring.as_ref().expect("checked above");
            for (i, (_, ks)) in group.iter().enumerate() {
                // the column's logits over the whole vocabulary: each rank's part in rank order
                let mut z = Vec::with_capacity(ranks * vocab);
                for r in 0..ranks {
                    z.extend(side.host.words::<f32>(r * stride + 4 * i * vocab, vocab));
                }
                for &k in ks {
                    out[k] = score_position(&z, &request.targets[k], request.tops[k]);
                }
            }
        }
        Ok(())
    }

    /// One prefill row in chunks (`prefill_chunks`: 1024 tokens up to three such chunks, then the widest);
    /// returns its last chunk's sample. With MTP, each chunk's MTP keys and hidden store, and after the prompt
    /// (a sampled row) the first draft, carried to the row's first decode step.
    #[allow(clippy::too_many_arguments)]
    fn prefill_row(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        r: usize,
        params: &[[f32; PARAMS]],
        request: Option<&ScoreRequest>,
        mut scored: Option<&mut [Scored]>,
    ) -> PyResult<Outputs> {
        let ids: Vec<i64> = plan.token_ids
            [plan.token_indptr[r] as usize..plan.token_indptr[r + 1] as usize]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let pos0 = i64::from(plan.context_lengths[r]);
        let blocks = self.blocks(plan, r)?;
        let widest = *self.prefill_buckets.last().unwrap_or(&1);
        let step = if ids.len() <= 3 * 1024 { 1024 } else { widest };
        let mut sampled: Option<Outputs> = None;
        let mut s = 0;
        while s < ids.len() {
            let chunk = &ids[s..(s + step).min(ids.len())];
            let n = chunk.len();
            let p = pos0 + s as i64;
            let last = s + n >= ids.len();
            let mut rows = StepRows::new();
            let states = blocks.states(r, p, n as i64, self.geometry.state_tokens)?;
            rows.push(0, p, n as i64, states, &blocks.kv);
            let t = bucket_of(&self.prefill_buckets, n, true)?;
            let k = self.take_image(py, false)?;
            let counts = self.fill(
                py,
                k,
                t,
                &rows,
                &glm::Step {
                    tokens: Some(chunk),
                    columns: n,
                    first_position: p,
                    head: true,
                    verify: 0,
                    keep_tails: false,
                    meta_columns: None,
                },
            )?;
            self.pass(py, k, t, &counts, 0, false, &[], "prefill")?;
            if last {
                sampled = Some(self.sample_now(py, t, params, true)?);
            }
            // after sampling: the chunk's head reruns at the scored columns
            if let (Some(req), Some(out)) = (request, scored.as_deref_mut()) {
                let which: Vec<usize> = (0..req.positions.len())
                    .filter(|&k| req.positions[k] >= s && req.positions[k] < s + n)
                    .collect();
                if !which.is_empty() {
                    self.score_chunk(py, t, req, &which, s, out)?;
                }
            }
            if self.dflash.is_some() {
                // the chunk into the drafter's cache, and after the prompt the block after its sampled token
                let inject = self.inject(py, t, &[(0, p, n, &blocks)])?;
                self.submit(py, &[inject], "inject")?;
                if last
                    && plan.sampling_mask[r]
                    && let Some(out) = sampled.as_ref()
                {
                    let token = i64::from(out.tokens[0]);
                    let drafts = self.draft_blocks(py, &[(p + n as i64, &blocks)], &[token])?;
                    self.carried.insert(
                        plan.sequence_ids[r],
                        Carried {
                            position: p + n as i64,
                            anchor: token,
                            drafts: drafts.into_iter().next().expect("a row"),
                            full: true,
                        },
                    );
                }
            }
            if self.mtp.is_some() {
                // the first draft after the prompt, at its sampled token's position + 1
                let draft = (last && plan.sampling_mask[r])
                    .then(|| -> PyResult<[f32; PARAMS]> {
                        let query = u64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
                        let offset = (u64::from(plan.context_lengths[r]) + query + 1)
                            * sampling::OFFSET_STRIDE;
                        Ok(sampling::row_params(
                            &row_sample(plan, r)?,
                            self.vocab,
                            offset,
                        ))
                    })
                    .transpose()?;
                let drafts = self.prefill_mtp(py, t, p, chunk, &blocks, states, draft)?;
                if let (Some(drafts), Some(out)) = (drafts, sampled.as_ref()) {
                    self.carried.insert(
                        plan.sequence_ids[r],
                        Carried {
                            position: p + n as i64,
                            anchor: i64::from(out.tokens[0]),
                            drafts,
                            full: self.k == 1,
                        },
                    );
                }
            }
            s += n;
        }
        sampled.ok_or_else(|| PyValueError::new_err("a prefill row without tokens"))
    }

    /// Fill and queue a decode pass of `plan`'s rows `range` at `positions` (`tokens` None: the device's, the
    /// previous step's samples), then `then`. Returns the bucket and the rows' positions and blocks.
    fn decode_pass(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        range: std::ops::Range<usize>,
        positions: &[i64],
        tokens: Option<&[i64]>,
        then: &[usize],
    ) -> PyResult<(usize, Vec<(i64, Blocks)>)> {
        let n = range.len();
        let t = self.decode_bucket(n)?;
        let mut rows = StepRows::new();
        let mut kept = Vec::with_capacity(n);
        for (column, r) in range.clone().enumerate() {
            let blocks = self.blocks(plan, r)?;
            let states = blocks.states(r, positions[column], 1, self.geometry.state_tokens)?;
            rows.push(column as i64, positions[column], 1, states, &blocks.kv);
            kept.push((positions[column], blocks));
        }
        let k = self.take_image(py, false)?;
        let counts = self.fill(
            py,
            k,
            t,
            &rows,
            &glm::Step {
                tokens,
                columns: n,
                first_position: positions[0],
                head: false,
                verify: 0,
                keep_tails: false,
                meta_columns: None,
            },
        )?;
        let then = then
            .iter()
            .map(|&section| self.io(t, section))
            .collect::<PyResult<Vec<_>>>()?;
        self.pass(py, k, t, &counts, 0, tokens.is_none(), &then, "decode")?;
        Ok((t, kept))
    }

    /// A synchronous step (`forward`): every prefill row in order, one decode step in steps of up to `max_t` rows,
    /// or a speculative step (`speculating`: the policy's decision; None: decide here). `exact`: each row's
    /// position and token, for a decode continuation (its plan only bounds them).
    fn forward_native(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        speculating: Option<bool>,
        exact: Option<(Vec<i64>, Vec<i64>)>,
    ) -> PyResult<Py<ForwardOutput>> {
        self.check_plan(plan)?;
        self.harvest(py)?;
        let decode = plan.phase != BatchPhase::Prefill;
        let speculating = match speculating {
            Some(s) => s,
            None if decode => {
                self.settle_open(py, None)?; // the last step's acceptance comes before this step's choice
                self.speculating(py, plan)?
            }
            None => false,
        };
        if decode {
            self.timed = None; // a synchronous step: the next step does not continue it
        }
        let params = plan_params(plan, self.vocab)?;
        if decode && speculating {
            let started = Instant::now();
            let output = self.speculate_sync(py, plan, &params, exact)?;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            self.choice
                .measured(Kind::Verify, plan.sequence_ids.len(), ms);
            return Ok(output);
        }
        let mut out = Outputs::empty();
        let mut prompt = None;
        let n = plan.sequence_ids.len();
        if plan.phase == BatchPhase::Prefill {
            let requests = if plan.needs_prompt_logprobs() {
                score_requests(plan, self.vocab)
            } else {
                (0..n).map(|_| None).collect()
            };
            let mut scored = Vec::new();
            for r in 0..n {
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
            let (positions, tokens) = exact.unwrap_or_else(|| {
                (
                    plan.context_lengths.iter().map(|&c| i64::from(c)).collect(),
                    plan.token_ids.iter().map(|&t| i64::from(t)).collect(),
                )
            });
            out = self.decode_steps(py, plan, &positions, &tokens, &params)?;
        }
        out.output(py, plan.needs_logprobs(), &params, prompt)
    }

    /// Synchronous decode steps of up to `max_t` rows each at `positions` with host `tokens` (with MTP, each
    /// step's MTP keys after its sampler).
    fn decode_steps(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        positions: &[i64],
        tokens: &[i64],
        params: &[[f32; PARAMS]],
    ) -> PyResult<Outputs> {
        let n = plan.sequence_ids.len();
        let mut out = Outputs::empty();
        let mut g = 0;
        while g < n {
            let end = (g + self.max_t).min(n);
            let (t, rows) = self.decode_pass(
                py,
                plan,
                g..end,
                &positions[g..end],
                Some(&tokens[g..end]),
                &[],
            )?;
            out.extend(self.sample_now(py, t, &params[g..end], false)?);
            self.draft_keys(py, t, &rows)?;
            g = end;
        }
        Ok(out)
    }

    /// The sampler inputs of `row` at `step` positions past its first sampled one (`Sampling.at`): its uniform at
    /// offset (start + 1 + step) x 32.
    fn row_params_at(&self, row: &VerifyRow, step: i64) -> [f32; PARAMS] {
        let offset = (row.start + 1 + step) as u64 * sampling::OFFSET_STRIDE;
        sampling::row_params(&row.sample, self.vocab, offset)
    }

    /// `verify_launch`: one speculative step of `rows` left running: the target over [anchor, d_1 .. d_K] (column
    /// b Tq + t), sampled at every column, and the MTP layer at every column (token y_t, the target's hidden at the
    /// column), sampled too. Returns the verification.
    #[allow(clippy::too_many_arguments)]
    fn verify_launch(
        &mut self,
        py: Python<'_>,
        slf: &Py<Self>,
        rows: Vec<VerifyRow>,
        anchors: &[i64],
        drafts: Vec<Vec<i64>>,
        own: Option<Vec<bool>>,
        params: Vec<[f32; PARAMS]>,
        scores: bool,
    ) -> PyResult<Py<GlmVerification>> {
        let (b, tq) = (rows.len(), self.tq);
        let t = self.decode_bucket(b * tq)?;
        let mut target = StepRows::new();
        let mut tokens = Vec::with_capacity(b * tq);
        for (i, row) in rows.iter().enumerate() {
            target.push(
                (i * tq) as i64,
                row.start,
                tq as i64,
                (row.state_src, -1),
                &row.blocks.kv,
            );
            tokens.push(anchors[i]);
            tokens.extend_from_slice(&drafts[i]);
        }
        // column t samples position start + t + 1
        let mut grid = vec![[0f32; PARAMS]; t];
        for (i, row) in rows.iter().enumerate() {
            for j in 0..tq {
                grid[i * tq + j] = self.row_params_at(row, j as i64);
            }
        }
        let stage = self.take_stage(py, false)?;
        self.sample_table(&grid, 0);
        self.registers[R_STAGE] = stage as u64;
        let k = self.take_image(py, false)?;
        let counts = self.fill(
            py,
            k,
            t,
            &target,
            &glm::Step {
                tokens: Some(&tokens),
                columns: b * tq,
                first_position: rows[0].start,
                head: false,
                verify: tq,
                keep_tails: true,
                meta_columns: None,
            },
        )?;
        let then = [self.io(t, IO_SAMPLE)?, self.io(t, IO_OUT)?];
        self.pass(py, k, t, &counts, tq, false, &then, "verify")?;
        let mstage = if self.dflash.is_some() {
            // every column into the drafter's cache
            let runs: Vec<(usize, i64, usize, &Blocks)> = rows
                .iter()
                .enumerate()
                .map(|(i, row)| (i * tq, row.start, tq, &row.blocks))
                .collect();
            let inject = self.inject(py, t, &runs)?;
            self.submit(py, &[inject], "inject")?;
            None
        } else {
            Some(self.verify_mtp(py, t, &rows)?)
        };
        let id = self.next_id;
        self.next_id += 1;
        self.staging.owner[stage] = Some(id);
        if let (Some(m), Some(mstage)) = (self.mtp.as_mut(), mstage) {
            m.staging.owner[mstage] = Some(id);
        }
        let verification = Py::new(
            py,
            GlmVerification {
                executor: slf.clone_ref(py),
                id,
                t,
                stage,
                mstage,
                params,
                scores,
                rows,
                drafts,
                own,
                done: None,
            },
        )?;
        Ok(verification)
    }

    /// A verify step's MTP pass: the MTP layer at every column (token y_t, the target's hidden at the column),
    /// sampled; returns its MTP staging set.
    fn verify_mtp(&mut self, py: Python<'_>, t: usize, rows: &[VerifyRow]) -> PyResult<usize> {
        let tq = self.tq;
        // the MTP layer at every column: column t drafts position start + t + 2
        let mut m_rows = StepRows::new();
        let mut mgrid = vec![[0f32; PARAMS]; t];
        for (i, row) in rows.iter().enumerate() {
            m_rows.push(
                (i * tq) as i64,
                row.start,
                tq as i64,
                (-1, -1),
                &row.blocks.mtp,
            );
            for j in 0..tq {
                mgrid[i * tq + j] = self.row_params_at(row, j as i64 + 1);
            }
        }
        let cols: Vec<i64> = (1..=t as i64).collect();
        let (tok_out, zero, io_hf, hf0) = {
            let m = self.mtp()?;
            (m.tok_out, m.zero, m.io_hf, m.hf[0])
        };
        Ok(self
            .mtp_pass(
                py,
                t,
                t,
                &m_rows,
                &cols,
                Tokens::Device(tok_out),
                zero,
                io_hf,
                hf0,
                Some(&mgrid),
                &[],
            )?
            .expect("a full pass samples"))
    }

    /// `chain`: each row's drafts d_1 .. d_K after a settled verification: link j (MTP at position anchor + j - 1,
    /// token d_j, hidden: link j - 1's output, the first link's the accepted column of the verify pass's MTP
    /// hiddens) samples d_(j+1), one MTP pass a link (the host reads each link's drafts). Runs collectives: call
    /// it only where every rank does, before the next launch overwrites the verify pass's MTP hiddens.
    #[allow(clippy::needless_range_loop)] // links and rows index the drafts
    fn chain(&mut self, py: Python<'_>, v: &Bound<'_, GlmVerification>) -> PyResult<Vec<Vec<i64>>> {
        let (rows, done, t_verify) = {
            let v = v.borrow();
            let done = v
                .done
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("the verification is not settled"))?;
            (v.rows.clone(), done.0.clone(), v.t)
        };
        let mut drafts = done.next.clone();
        let (k, b) = (self.k, rows.len());
        if k == 1 || self.dflash.is_some() {
            return Ok(drafts); // with the drafter, its block step made every draft
        }
        let bt = self.decode_bucket(b)?;
        let (hf, copy, zero) = {
            let m = self.mtp()?;
            (m.hf, m.io(t_verify, MIO_CHAIN_HF)?, m.zero)
        };
        // the verify pass's MTP hiddens, by accepted column: the chain's first hiddens
        if b > ACC_ROWS {
            return Err(PyValueError::new_err(format!("{b} rows exceed a chain")));
        }
        self.registers[R_CHAIN_ROWS] = b as u64;
        for (i, &a) in done.accepted.iter().enumerate() {
            self.registers[R_ACC + i] = a as u64;
        }
        let mut before = vec![copy];
        let mut tokens: Vec<i64> = drafts.iter().map(|d| d[0]).collect();
        let cols: Vec<i64> = (1..=bt as i64).collect();
        let mut current = hf[1];
        for j in 1..k {
            let mut link = StepRows::new();
            let mut params = vec![[0f32; PARAMS]; bt];
            for (i, row) in rows.iter().enumerate() {
                let a = done.accepted[i];
                link.push(
                    i as i64,
                    done.positions[i] + j as i64 - 1,
                    1,
                    (-1, -1),
                    &row.blocks.mtp,
                );
                params[i] = self.row_params_at(row, a + j as i64 + 1);
            }
            let out_hf = if current == hf[1] { hf[0] } else { hf[1] };
            let stage = self
                .mtp_pass(
                    py,
                    bt,
                    bt,
                    &link,
                    &cols,
                    Tokens::Host(&tokens),
                    zero,
                    current,
                    out_hf,
                    Some(&params),
                    &before,
                )?
                .expect("a full pass samples");
            before.clear();
            step_trace::mark("lw0", j as i64);
            self.wait_stage(py, stage, true)?;
            step_trace::mark("lw1", j as i64);
            let sampled = self.staged(stage, b, true).tokens;
            for i in 0..b {
                drafts[i][j] = i64::from(sampled[i]);
                tokens[i] = i64::from(sampled[i]);
            }
            current = out_hf;
        }
        Ok(drafts)
    }

    /// The rows of a speculative step: each row's start, blocks, state before it and sampler settings.
    fn verify_rows(
        &self,
        plan: &BatchPlan,
        range: std::ops::Range<usize>,
        starts: &[i64],
    ) -> PyResult<Vec<VerifyRow>> {
        range
            .enumerate()
            .map(|(i, r)| {
                let blocks = self.blocks(plan, r)?;
                let start = starts[i];
                let src = if start == 0 {
                    -1
                } else {
                    blocks.state(r, start - 1, self.geometry.state_tokens)?
                };
                Ok(VerifyRow {
                    sequence: plan.sequence_ids[r],
                    start,
                    blocks,
                    state_src: src,
                    sample: row_sample(plan, r)?,
                })
            })
            .collect()
    }

    /// `_speculate_sync`: one speculative step of a decode plan, `speculative_rows` rows at a time (`exact`: each
    /// row's position and anchor, for a continuation).
    fn speculate_sync(
        &mut self,
        py: Python<'_>,
        plan: &BatchPlan,
        params: &[[f32; PARAMS]],
        exact: Option<(Vec<i64>, Vec<i64>)>,
    ) -> PyResult<Py<ForwardOutput>> {
        let slf: Py<Self> = self.handle(py)?;
        let n = plan.sequence_ids.len();
        let (positions, anchors) = exact.unwrap_or_else(|| {
            (
                plan.context_lengths.iter().map(|&c| i64::from(c)).collect(),
                plan.token_ids.iter().map(|&t| i64::from(t)).collect(),
            )
        });
        let mut token_ids = Vec::new();
        let mut counts = Vec::new();
        let (mut lp, mut top_ids, mut top_lp) = (Vec::new(), Vec::new(), Vec::new());
        let mut g = 0;
        while g < n {
            let end = (g + self.speculative_rows).min(n);
            let sequences = &plan.sequence_ids[g..end];
            let rows = self.verify_rows(plan, g..end, &positions[g..end])?;
            let (drafts, own) =
                self.carried_drafts(sequences, &positions[g..end], &anchors[g..end]);
            let v = self.verify_launch(
                py,
                &slf,
                rows,
                &anchors[g..end],
                drafts,
                Some(own),
                params[g..end].to_vec(),
                plan.needs_logprobs(),
            )?;
            let done = GlmVerification::settle(v.bind(py), py, self)?;
            let next = self.chain(py, v.bind(py))?;
            for (i, &sequence) in sequences.iter().enumerate() {
                self.carried.insert(
                    sequence,
                    Carried {
                        position: done.positions[i],
                        anchor: done.anchors[i],
                        drafts: next[i].clone(),
                        full: true,
                    },
                );
            }
            for row in &done.tokens {
                token_ids.extend(row.iter().map(|&t| t as u32));
                counts.push(row.len());
            }
            if let Some((l, ids, values)) = &done.scored {
                lp.extend_from_slice(l);
                top_ids.extend_from_slice(ids);
                top_lp.extend_from_slice(values);
            }
            g = end;
        }
        let mut indptr = vec![0u32];
        for &c in &counts {
            indptr.push(indptr.last().copied().unwrap_or(0) + c as u32);
        }
        let (scores, top) = if plan.needs_logprobs() {
            let (s, t) = score_columns(params, &counts, &lp, &top_ids, &top_lp);
            (Some(s), t)
        } else {
            (None, None)
        };
        Py::new(
            py,
            ForwardOutput::native(py, token_ids, indptr, scores, top)?,
        )
    }

    /// This executor as a Python object (the verifications it makes keep it).
    fn handle(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        // the executor is borrowed mutably by the caller: its own handle comes from the open steps' references
        // or, the first time, from the one the runner stored
        self.this
            .as_ref()
            .map(|h| h.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err("the executor has no handle (bind it first)"))
    }
}

/// Parse the MTP side's dict (`native._mtp_side`).
fn mtp_side(d: &Bound<'_, PyDict>, fill: &GlmFill) -> PyResult<Mtp> {
    let get = |name: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("mtp has no {name}")))
    };
    let pinned = |name: &str| -> PyResult<Pinned> {
        Pinned::new(name, get(name)?.extract::<PyReadwriteArray1<'_, u8>>()?)
    };
    let (device, host, stride, events): (u64, PyReadwriteArray1<'_, u8>, usize, i32) =
        get("image")?.extract()?;
    let host = Pinned::at_least("MTP image host copies", host, 2 * stride)?;
    let tables: HashMap<usize, glm::Tables> = fill
        .shapes
        .iter()
        .filter(|((_, mtp), _)| *mtp)
        .map(|((t, _), tables)| (*t, tables.clone()))
        .collect();
    if tables.values().any(|tb| tb.nbytes > stride) {
        return Err(PyValueError::new_err(
            "the MTP tables exceed the MTP host images",
        ));
    }
    let mut io = HashMap::new();
    for (t, program) in get("io")?.cast::<PyDict>()?.iter() {
        let program = program.extract::<PyRef<'_, GaudiProgram>>()?.program()?;
        if program.registers() != REGISTERS {
            return Err(PyValueError::new_err(
                "an MTP I/O program takes other registers",
            ));
        }
        io.insert(t.extract::<usize>()?, Shaped::of(program, &MIO_SECTIONS));
    }
    let mut programs = HashMap::new();
    for (key, program) in get("programs")?.cast::<PyDict>()?.iter() {
        let program = program.extract::<PyRef<'_, GaudiProgram>>()?.program()?;
        if program.registers() > REGISTERS {
            return Err(PyValueError::new_err(
                "an MTP pass program takes too many registers",
            ));
        }
        programs.insert(key.extract::<(usize, usize, bool, bool)>()?, program);
    }
    let staging = Pinned::new(
        "MTP staging",
        get("staging")?.extract::<PyReadwriteArray1<'_, u8>>()?,
    )?;
    let staging_stride: usize = item(d, "staging_stride")?;
    if staging.bytes < 2 * staging_stride {
        return Err(PyValueError::new_err("the MTP staging sets are too small"));
    }
    let addresses = get("addresses")?;
    let addresses = addresses.cast::<PyDict>()?;
    let hf: Vec<u64> = item(addresses, "hf")?;
    let geometry = get("geometry")?;
    let geometry = geometry.cast::<PyDict>()?;
    Ok(Mtp {
        image: Images::new(device, host, stride, events)?,
        tables,
        io,
        programs,
        cols: pinned("cols")?,
        tok: pinned("tok")?,
        sample: pinned("sample")?,
        hidden_table: pinned("hidden")?,
        staging: Staging {
            host: staging,
            stride: staging_stride,
            events: item(d, "stage_events")?,
            next: 0,
            owner: [None, None],
        },
        hf: [hf[0], hf[1]],
        io_hf: item(addresses, "io_hf")?,
        zero: item(addresses, "zero")?,
        tok_out: item(addresses, "tok_out")?,
        mtp_tok: item(addresses, "mtp_tok")?,
        arena: item(addresses, "arena")?,
        state: StateGeometry {
            state_layer_rows: item(geometry, "state_layer_rows")?,
            ssm_rows: item(geometry, "ssm_rows")?,
            state_tokens: 0, // set by the executor (its geometry's)
            hidden_offset: item(geometry, "hidden_offset")?,
        },
        row_bytes: item(geometry, "row_bytes")?,
    })
}

#[pymethods]
impl GlmExecutor {
    #[classattr]
    const REGISTERS: usize = REGISTERS;
    #[classattr]
    const R_IMAGE: usize = R_IMAGE;
    #[classattr]
    const R_STAGE: usize = R_STAGE;
    #[classattr]
    const R_SLOT_BYTES: usize = R_SLOT_BYTES;
    #[classattr]
    const R_INDEX_BYTES: usize = R_INDEX_BYTES;
    #[classattr]
    const R_MIMAGE: usize = R_MIMAGE;
    #[classattr]
    const R_MSLOT_BYTES: usize = R_MSLOT_BYTES;
    #[classattr]
    const R_MINDEX_BYTES: usize = R_MINDEX_BYTES;
    #[classattr]
    const R_MSTAGE: usize = R_MSTAGE;
    #[classattr]
    const R_CHAIN_ROWS: usize = R_CHAIN_ROWS;
    #[classattr]
    const R_ACC: usize = R_ACC;
    #[classattr]
    const R_BIMAGE: usize = R_BIMAGE;
    #[classattr]
    const R_BBYTES: usize = R_BBYTES;
    #[classattr]
    const R_BLAUNCHES: usize = R_BLAUNCHES;
    #[classattr]
    const R_DIMAGE: usize = R_DIMAGE;

    /// `steps`: per (bucket, indexer, sparse, verify) its step program; `io`: per bucket its I/O program.
    /// `image`: (device table, pinned host copies, stride, first event slot); `staging`: (pinned sets, stride,
    /// first event slot, (tok, lp, top_idx, top_lp, bytes) offsets); `sample`: (pinned sampler table, offset of
    /// its `last`); `geometry`: (slab rows, KV block rows, STATE block rows, STATE tokens); `model`: a dict of the
    /// model's numbers; `scoring`: the prompt logprobs' pinned buffers and event; `mtp`: the MTP layer's side
    /// (`native._mtp_side`); `share`: the ranks' step times from the first rank (None: one card or a native rank
    /// group).
    #[new]
    #[pyo3(signature = (*, glue, fill, layout, steps, io, image, staging, sample, geometry, model, scoring=None, mtp=None, dflash=None, commit=None, share=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        glue: Py<GaudiGlue>,
        fill: PyRef<'_, GlmFill>,
        layout: Py<CacheLayout>,
        steps: HashMap<(usize, bool, bool, usize), PyRef<'_, GaudiProgram>>,
        io: HashMap<usize, PyRef<'_, GaudiProgram>>,
        image: (u64, PyReadwriteArray1<'_, u8>, usize, i32),
        staging: StagingArgs<'_>,
        sample: (PyReadwriteArray1<'_, u8>, usize),
        geometry: (i64, i64, i64, i64),
        model: Bound<'_, PyDict>,
        scoring: Option<Bound<'_, PyDict>>,
        mtp: Option<Bound<'_, PyDict>>,
        dflash: Option<Bound<'_, PyDict>>,
        commit: Option<Bound<'_, PyDict>>,
        share: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let pinned = |d: &Bound<'_, PyDict>, name: &str| -> PyResult<Pinned> {
            let array = d
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("scoring has no {name}")))?;
            Pinned::new(name, array.extract::<PyReadwriteArray1<'_, u8>>()?)
        };
        let head_cols: usize = item(&model, "head_cols")?;
        let scoring = match scoring {
            Some(d) => {
                let side = ScoreSide {
                    columns: pinned(&d, "columns")?,
                    host: pinned(&d, "host")?,
                    stride: item(&d, "stride")?,
                    vocab: item(&d, "vocab")?,
                    ranks: item(&d, "ranks")?,
                    event: item(&d, "event")?,
                };
                if side.ranks * side.stride > side.host.bytes
                    || 4 * head_cols * side.vocab > side.stride
                    || side.columns.bytes < 4 * head_cols
                {
                    return Err(PyValueError::new_err("the score buffers are too small"));
                }
                Some(side)
            }
            None => None,
        };
        let tables: HashMap<usize, glm::Tables> = fill
            .shapes
            .iter()
            .filter(|((_, mtp), _)| !mtp)
            .map(|((t, _), tables)| (*t, tables.clone()))
            .collect();
        let mut built_steps = HashMap::new();
        for (key, program) in &steps {
            let program = program.program()?;
            if program.registers() > REGISTERS {
                return Err(PyValueError::new_err(format!(
                    "step {key:?}'s program takes more than {REGISTERS} registers"
                )));
            }
            let t = tables.get(&key.0).ok_or_else(|| {
                PyValueError::new_err(format!("the fill has no tables of {} columns", key.0))
            })?;
            if t.nbytes > image.2 {
                return Err(PyValueError::new_err(format!(
                    "bucket {}'s tables of {} bytes exceed the host images",
                    key.0, t.nbytes
                )));
            }
            built_steps.insert(*key, program);
        }
        let mut shapes = HashMap::new();
        for (t, program) in &io {
            let program = program.program()?;
            if program.registers() != REGISTERS {
                return Err(PyValueError::new_err(format!(
                    "bucket {t}'s I/O program takes {} registers, not {REGISTERS}",
                    program.registers()
                )));
            }
            let shaped = Shaped::of(program, &IO_SECTIONS);
            if shaped.sections[IO_UPLOAD].is_none()
                || shaped.sections[IO_UPLOADED].is_none()
                || shaped.sections[IO_OUT].is_none()
            {
                return Err(PyValueError::new_err(format!(
                    "bucket {t}'s I/O program lacks upload, uploaded or out"
                )));
            }
            shapes.insert(*t, shaped);
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
        let max_t: usize = item(&model, "max_t")?;
        if sample_last + 4 > sample_host.bytes || 4 * PARAMS * max_t > sample_last {
            return Err(PyValueError::new_err("the sampler table is too small"));
        }
        let (slab_rows, kv_block_rows, state_block_rows, state_tokens) = geometry;
        let k: usize = item(&model, "draft_tokens")?;
        let mut mtp = match mtp {
            Some(d) => Some(mtp_side(&d, &fill)?),
            None => None,
        };
        let mtp_block_rows = match &mut mtp {
            Some(m) => {
                m.state.state_tokens = state_tokens;
                if m.sample.bytes < sample_last + 4 {
                    return Err(PyValueError::new_err("the MTP sampler table is too small"));
                }
                let d = mtp_geometry(&model)?;
                d.unwrap_or(0)
            }
            None => 0,
        };
        let costs = model
            .get_item("costs")?
            .ok_or_else(|| PyValueError::new_err("model has no costs"))?;
        let costs = costs.cast::<PyDict>()?;
        let costs = StepCosts {
            plain_0: item(costs, "plain_0")?,
            plain_row: item(costs, "plain_row")?,
            launch_0: item(costs, "launch_0")?,
            launch_row: item(costs, "launch_row")?,
            scale: item(costs, "scale")?,
            group: item::<usize>(costs, "group")?.max(1),
        };
        let decode_buckets: Vec<usize> = item(&model, "decode_buckets")?;
        let mode = Mode::of(&item::<String>(&model, "policy")?).map_err(PyValueError::new_err)?;
        let spec_rows: usize = item(&model, "spec_rows")?;
        let choice = DraftChoice::new(
            k,
            costs,
            spec_rows,
            item(&model, "prior")?,
            &decode_buckets,
            share.is_some(),
            mode,
        );
        let dflash = match dflash {
            Some(d) => Some(dflash_side(&d)?),
            None => None,
        };
        let draft_block_rows: i64 = match &dflash {
            Some(_) => item(&model, "draft_block_rows")?,
            None => 0,
        };
        let commit = match commit {
            Some(d) => {
                let host = d
                    .get_item("host")?
                    .ok_or_else(|| PyValueError::new_err("commit has no host"))?;
                let geometry = d
                    .get_item("geometry")?
                    .ok_or_else(|| PyValueError::new_err("commit has no geometry"))?;
                let geometry = geometry.cast::<PyDict>()?;
                Some(Commit {
                    host: Pinned::new(
                        "commit tables",
                        host.extract::<PyReadwriteArray1<'_, u8>>()?,
                    )?,
                    at: item(&d, "at")?,
                    state: StateGeometry {
                        state_layer_rows: item(geometry, "state_layer_rows")?,
                        ssm_rows: item(geometry, "ssm_rows")?,
                        state_tokens,
                        hidden_offset: item(geometry, "hidden_offset")?,
                    },
                    layers: item(geometry, "kda_layers")?,
                })
            }
            None => None,
        };
        if k > 0 && commit.is_none() {
            return Err(PyValueError::new_err("drafts need the commit"));
        }
        if k > 0 && mtp.is_none() && dflash.is_none() {
            return Err(PyValueError::new_err(
                "drafts need the MTP layer or the drafter",
            ));
        }
        Ok(Self {
            glue,
            model: fill.model.clone(),
            tables,
            steps: built_steps,
            shapes,
            layout,
            geometry: Geometry {
                slab_rows,
                kv_block_rows,
                state_block_rows,
                state_tokens,
                mtp_block_rows,
                draft_block_rows,
            },
            image: Images::new(device, host, stride, events)?,
            staging: Staging {
                host: staging_host,
                stride: staging_stride,
                events: stage_events,
                next: 0,
                owner: [None, None],
            },
            sample_host,
            sample_last,
            sampled: Sampled {
                tok,
                lp,
                top_idx,
                top_lp,
            },
            vocab: item(&model, "vocab")?,
            prefill_buckets: item(&model, "prefill_buckets")?,
            decode_buckets,
            max_t,
            head_cols,
            max_batch_size: item(&model, "max_batch_size")?,
            max_seq_len: item(&model, "max_seq_len")?,
            spin_ns: item(&model, "spin_ns")?,
            registers: [0; REGISTERS],
            inflight: Vec::new(),
            next_id: 0,
            timings: Timings::default(),
            native_steps: 0,
            continuations: 0,
            scoring,
            k,
            tq: k + 1,
            speculative_rows: item(&model, "speculative_rows")?,
            spec_rows,
            lookahead: item(&model, "lookahead")?,
            mtp,
            dflash,
            commit,
            stats: speculative::Stats::default(),
            choice,
            share,
            carried: HashMap::new(),
            open: None,
            timed: None,
            this: None,
            ranks: None,
            forced: None,
            card_gate: None,
        })
    }

    /// `RESIN_STEP_TRACE`: mark this process's rank in its trace.
    #[staticmethod]
    fn trace_rank(rank: i64) {
        step_trace::mark("rank", rank);
        step_trace::flush();
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

    /// Enqueued steps that continued the previous one.
    #[getter]
    fn continuations(&self) -> u64 {
        self.continuations
    }

    /// The verify steps' acceptance counters: (blocks, drafts, accepted, emitted).
    #[getter]
    fn speculation(&self) -> (u64, u64, u64, u64) {
        self.stats.counts()
    }

    /// The draft policy's mode (auto, always or never).
    #[getter]
    fn policy(&self) -> &'static str {
        self.choice.mode.name()
    }

    /// Change the draft policy's mode (tests and benchmarks; with ranks, on the driver, which decides).
    #[setter]
    fn set_policy(&mut self, mode: &str) -> PyResult<()> {
        self.choice.mode = Mode::of(mode).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Every sequence's drafts carried to its next step.
    fn carried_all(&self) -> HashMap<u64, (i64, i64, Vec<i64>)> {
        self.carried
            .iter()
            .map(|(s, c)| (*s, (c.position, c.anchor, c.drafts.clone())))
            .collect()
    }

    /// A sequence's drafts carried to its next step: (the anchor's position, the anchor, the drafts).
    fn carried(&self, sequence: u64) -> Option<(i64, i64, Vec<i64>)> {
        self.carried
            .get(&sequence)
            .map(|c| (c.position, c.anchor, c.drafts.clone()))
    }

    /// The benchmark's fixed acceptance (`SpeculativeStats.benchmark_accepted_drafts`).
    #[getter]
    fn fixed_acceptance(&self) -> Option<usize> {
        self.stats.fixed
    }

    #[setter]
    fn set_fixed_acceptance(&mut self, fixed: Option<usize>) -> PyResult<()> {
        if fixed.is_some_and(|f| f > self.k) {
            return Err(PyValueError::new_err(
                "fixed acceptance must fit the drafts",
            ));
        }
        self.stats.fixed = fixed;
        if let Some(ranks) = self.ranks.as_mut() {
            ranks.fixed = fixed; // the followers take it from each call's frame
        }
        Ok(())
    }

    /// The runner's `metrics_snapshot` (the speculation counters), with no Python: the runner binds it.
    fn metrics_snapshot(&self) -> crate::metrics::RunnerMetrics {
        let (blocks, drafts, accepted, _) = self.stats.counts();
        crate::metrics::RunnerMetrics::speculation(drafts, accepted, blocks)
    }

    /// Forget released sequences: their acceptance and carried drafts (the Engine took the open steps' results
    /// first). With ranks, the driver sends the call to the followers first.
    fn release(&mut self, py: Python<'_>, sequence_ids: Vec<u64>) -> PyResult<()> {
        if let Some(ranks) = self.ranks.as_ref() {
            self.timings.announce +=
                ranks.announce(py, OP_RELEASE, None, false, false, &sequence_ids)?;
        }
        self.release_ids(py, &sequence_ids)
    }

    /// Join a tensor-parallel model's rank group: rank 0 drives (it sends each runner call to the others, with its
    /// draft decision), and every other rank replays the calls (`serve`).
    fn attach_ranks(&mut self, py: Python<'_>, ranks: Py<GaudiRanks>) {
        self.ranks = Ranked::of(py, ranks);
        if let Some(ranks) = self.ranks.as_mut() {
            // the driver decides for every rank, with the step times it measures
            self.share = None;
            self.choice.decide_alone();
            ranks.fixed = self.stats.fixed;
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
        if slf.borrow().this.is_none() {
            slf.borrow_mut().this = Some(slf.clone().unbind());
        }
        crate::gaudi_ranks::serve(&slf, py)
    }

    /// The Engine's synchronous step: prefill chunks, a decode step or a speculative step.
    fn forward(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        let out = Self::on_card(&slf, py, || Self::forward_call(&slf, py, &plan));
        if let Some(ranks) = slf.borrow_mut().ranks.as_mut() {
            ranks.led(py, &out);
        }
        out
    }

    /// Submit a decode step and return without waiting for it: a continuation (`feedback`) reads its tokens from
    /// the previous step's samples on the device, at the previous positions + 1, or continues its verification.
    /// Other plans run synchronously.
    #[pyo3(signature = (plan, *, feedback=None))]
    fn enqueue(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let out = Self::on_card(&slf, py, || Self::enqueue_call(&slf, py, &plan, feedback));
        if let Some(ranks) = slf.borrow_mut().ranks.as_mut() {
            ranks.led(py, &out);
        }
        out
    }
}

impl GlmExecutor {
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

    /// Settle every open step, then forget the sequences (`release`).
    fn release_ids(&mut self, py: Python<'_>, ids: &[u64]) -> PyResult<()> {
        self.harvest(py)?;
        self.settle_open(py, None)?;
        self.choice.forget(ids);
        for sequence in ids {
            self.carried.remove(sequence);
        }
        Ok(())
    }

    /// The driver: send the call to the followers, once the plan's checks passed on this rank (a plan that no
    /// rank can run fails here, before the followers see it). The draft decision goes with the call's GO.
    fn lead(&mut self, py: Python<'_>, op: u8, plan: &BatchPlan, continues: bool) -> PyResult<()> {
        if let Some(ranks) = self.ranks.as_ref().filter(|r| r.driver) {
            self.timings.announce += ranks.announce(py, op, Some(plan), continues, false, &[])?;
        }
        Ok(())
    }

    /// With ranks: go ahead with this call's collectives only when every rank is ready (one gate a call, after
    /// the plan's checks and the settle of the last step), with the draft decision: the driver's `speculating`
    /// goes to the followers with GO (the driver announced the call before it settled the last step and decided),
    /// and a follower returns the driver's. Without: `prepared` and `speculating`.
    fn gated_decision<T>(
        &mut self,
        py: Python<'_>,
        prepared: PyResult<T>,
        speculating: bool,
    ) -> PyResult<(T, bool)> {
        match &self.ranks {
            Some(ranks) => ranks
                .gate_word(py, prepared, u32::from(speculating), &mut self.timings.gate)
                .map(|(value, word)| (value, word & 1 == 1)),
            None => prepared.map(|value| (value, speculating)),
        }
    }

    fn driver(&self) -> bool {
        self.ranks.as_ref().is_some_and(|r| r.driver)
    }

    /// The Engine's synchronous step on this rank: the plan's checks (the driver fails a bad plan before
    /// announcing it), the open verification settled and the draft decision (a decode plan), the announcement,
    /// the gate, then the step.
    fn forward_call(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: &Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        let p = plan.get();
        let mut this = slf.borrow_mut();
        if this.this.is_none() {
            this.this = Some(slf.clone().unbind());
        }
        let checked = this.check_plan(p);
        if this.driver()
            && let Err(error) = &checked
        {
            return Err(error.clone_ref(py));
        }
        let decode = p.phase != BatchPhase::Prefill;
        // the driver announces first, so the followers come to the gate while it settles the last step; the
        // draft decision goes with GO
        this.lead(py, OP_FORWARD, p, false)?;
        let prepared = checked.and_then(|()| {
            if !decode {
                return Ok(None);
            }
            this.settle_open(py, None)?; // the last step's acceptance comes before this step's choice
            Ok(Some(this.speculating(py, p)?))
        });
        let mine = matches!(prepared, Ok(Some(true)));
        let (decoded, decision) = this.gated_decision(py, prepared, mine)?;
        let speculating = decoded.map(|_| decision);
        Ok(this.forward_native(py, p, speculating, None)?.into_any())
    }

    fn enqueue_call(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: &Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let p = plan.get();
        let mut plain: Option<Py<GlmCompletion>> = None;
        let mut verified: Option<Py<GlmVerification>> = None;
        if let Some(f) = &feedback {
            if let Ok(fb) = f.cast::<GlmFeedback>() {
                plain = Some(fb.get().completion.clone_ref(py));
            } else if let Ok(v) = f.cast::<GlmVerification>() {
                verified = Some(v.clone().unbind());
            } else {
                return Err(PyValueError::new_err(
                    "a continuation of a step this executor did not make",
                ));
            }
        }
        let mut this = slf.borrow_mut();
        if this.this.is_none() {
            this.this = Some(slf.clone().unbind());
        }
        step_trace::mark("enq", p.sequence_ids.len() as i64);
        let n = p.sequence_ids.len();
        let decode = p.phase == BatchPhase::Decode;
        let checked = this.check_cache(p);
        if this.driver() {
            if let Err(error) = &checked {
                return Err(error.clone_ref(py));
            }
            if let Some(f) = &feedback {
                let step = match f.cast::<GlmFeedback>() {
                    Ok(plain) => plain.get().completion.bind(py).clone().into_any(),
                    Err(_) => f.clone(),
                };
                let ranks = this.ranks.as_ref().expect("the driver");
                if !ranks.continues_last(py, &step) {
                    return Err(PyValueError::new_err(
                        "tensor-parallel ranks continue only the last enqueued step",
                    ));
                }
            }
        }
        // the driver announces first, so the followers come to the gate while it waits for the last step's
        // samples; the draft decision goes with GO
        this.lead(py, OP_ENQUEUE, p, feedback.is_some())?;
        step_trace::mark("led", 0);
        let prepared = checked.and_then(|()| {
            this.settle_open(py, verified.as_ref())?;
            // the last step's acceptance comes before this step's choice
            let done = match &verified {
                Some(v) => Some(GlmVerification::settle(v.bind(py), py, &mut this)?),
                None => None,
            };
            step_trace::mark("settled", 0);
            let speculating = decode && this.speculating(py, p)?;
            step_trace::mark("decided", i64::from(speculating));
            Ok((done, speculating))
        });
        let mine = prepared.as_ref().is_ok_and(|(_, s)| *s);
        let ((done, _), speculating) = this.gated_decision(py, prepared, mine)?;
        step_trace::mark("gated", 0);
        if decode {
            let kind = if speculating {
                Kind::Verify
            } else {
                Kind::Plain
            };
            this.time_step(kind, n, feedback.is_some());
        }
        let widest = if speculating {
            this.speculative_rows
        } else {
            this.max_t
        };
        if !decode || p.needs_prompt_logprobs() || n > widest || p.media.iter().any(Option::is_some)
        {
            // a continuation: its plan only bounds positions and tokens
            let exact = match (decode, &plain, &done) {
                (true, Some(previous), _) => {
                    GlmCompletion::finish(previous.bind(py), py, &mut this)?;
                    let c = previous.borrow(py);
                    Some((
                        c.positions.iter().map(|&q| q + 1).collect(),
                        c.tokens.clone(),
                    ))
                }
                (true, None, Some(done)) => Some((done.positions.clone(), done.anchors.clone())),
                _ => None,
            };
            let out = this.forward_native(py, p, decode.then_some(speculating), exact)?;
            return Ok(out.into_any());
        }
        this.check_cache(p)?;
        let (positions, tokens): (Vec<i64>, Option<Vec<i64>>) = match (&plain, &done) {
            (Some(previous), _) => (
                previous
                    .borrow(py)
                    .positions
                    .iter()
                    .map(|&q| q + 1)
                    .collect(),
                None,
            ),
            (None, Some(done)) => (done.positions.clone(), Some(done.anchors.clone())),
            (None, None) => (
                p.context_lengths.iter().map(|&c| i64::from(c)).collect(),
                Some(p.token_ids.iter().map(|&t| i64::from(t)).collect()),
            ),
        };
        if feedback.is_some() {
            this.continuations += 1;
        }
        let lookahead = this.decode_lookahead(n);
        if positions
            .iter()
            .any(|&q| q + 1 + lookahead > this.max_seq_len)
        {
            return Err(PyValueError::new_err(format!(
                "a row exceeds {} positions",
                this.max_seq_len
            )));
        }
        let params = crate::gaudi_exec::params_at(p, this.vocab, &positions)?;
        if speculating {
            let (anchors, drafts, own) = match (&plain, &verified) {
                (_, Some(v)) => {
                    // every rank is here, in the same order: the chain's collectives line up
                    let drafts = this.chain(py, v.bind(py))?;
                    step_trace::mark("chained", 0);
                    (
                        tokens.clone().expect("a verification's anchors"),
                        drafts,
                        None,
                    )
                }
                (Some(previous), None) => {
                    // after a plain step: its samples are the anchors
                    GlmCompletion::finish(previous.bind(py), py, &mut this)?;
                    let anchors = previous.borrow(py).tokens.clone();
                    let (drafts, own) = this.carried_drafts(&p.sequence_ids, &positions, &anchors);
                    (anchors, drafts, Some(own))
                }
                (None, None) => {
                    let anchors = tokens.clone().expect("the plan's tokens");
                    let (drafts, own) = this.carried_drafts(&p.sequence_ids, &positions, &anchors);
                    (anchors, drafts, Some(own))
                }
            };
            let rows = this.verify_rows(p, 0..n, &positions)?;
            let v = this.verify_launch(
                py,
                &slf.clone().unbind(),
                rows,
                &anchors,
                drafts,
                own,
                params,
                p.needs_logprobs(),
            )?;
            for sequence in &p.sequence_ids {
                this.carried.remove(sequence); // this step carries the drafts now
            }
            this.open = Some(v.clone_ref(py));
            step_trace::mark("launched", 0);
            return Ok(v.into_any());
        }
        let k = this.take_stage(py, false)?;
        this.sample_table(&params, 0);
        this.registers[R_STAGE] = k as u64;
        let (t, rows) = this.decode_pass(
            py,
            p,
            0..n,
            &positions,
            tokens.as_deref(),
            &[IO_SAMPLE, IO_OUT],
        )?;
        this.draft_keys(py, t, &rows)?;
        let id = this.next_id;
        this.next_id += 1;
        this.staging.owner[k] = Some(id);
        let completion = Py::new(
            py,
            GlmCompletion {
                executor: slf.clone().unbind(),
                id,
                stage: k,
                params,
                scores: p.needs_logprobs(),
                positions,
                tokens: Vec::new(),
                output: None,
            },
        )?;
        this.inflight.push(completion.clone_ref(py));
        Ok(completion.into_any())
    }
}

impl Replay for GlmExecutor {
    const HARVESTS: bool = true;

    fn follower_group(&self, py: Python<'_>) -> Option<Py<GaudiRanks>> {
        self.ranks
            .as_ref()
            .filter(|r| !r.driver)
            .map(|r| r.group.clone_ref(py))
    }

    /// The frame's decision is a placeholder (GLM's comes with GO); it stops the follower's own policy.
    fn set_speculating(&mut self, decision: Option<bool>) {
        self.forced = decision;
    }

    fn set_fixed(&mut self, fixed: Option<usize>) {
        self.stats.fixed = fixed;
    }

    fn replay_forward(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>> {
        Self::on_card(slf, py, || Self::forward_call(slf, py, &plan))
    }

    fn replay_enqueue(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // a verify step's samples are awaited off a layout's card gate (the call settles it on the gate), so that
        // a replica on this card can use the wait
        if let Some(v) = feedback
            .as_ref()
            .and_then(|f| f.cast::<GlmVerification>().ok())
            && v.borrow().done.is_none()
        {
            let stage = v.borrow().stage;
            slf.borrow_mut().wait_stage(py, stage, false)?;
        }
        Self::on_card(slf, py, || Self::enqueue_call(slf, py, &plan, feedback))
    }

    fn replay_release(slf: &Bound<'_, Self>, py: Python<'_>, ids: &[u64]) -> PyResult<()> {
        slf.borrow_mut().release_ids(py, ids)
    }
}

/// Parse the drafter's dict (`native._dflash_side`).
fn dflash_side(d: &Bound<'_, PyDict>) -> PyResult<Dflash> {
    let get = |name: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("dflash has no {name}")))
    };
    let cfg = get("drafter")?;
    let cfg = cfg.cast::<PyDict>()?;
    let drafter = qwen_draft::Drafter {
        layers: item(cfg, "layers")?,
        pairs: item(cfg, "pairs")?,
        layer_rows: item(cfg, "layer_rows")?,
        block: item(cfg, "block")?,
        window: item(cfg, "window")?,
        causal: item(cfg, "causal")?,
        mask_token: item(cfg, "mask_token")?,
        scratch_rows: item(cfg, "scratch_rows")?,
        max_rows: item(cfg, "max_rows")?,
    };
    let row_buckets: Vec<usize> = item(d, "row_buckets")?;
    let layouts: HashMap<usize, BlockLayout> = row_buckets
        .iter()
        .map(|&b| (b, BlockLayout::new(&drafter, b)))
        .collect();
    let (device, host, stride, events): (u64, PyReadwriteArray1<'_, u8>, usize, i32) =
        get("blocks")?.extract()?;
    if layouts.values().any(|l| l.nbytes > stride) {
        return Err(PyValueError::new_err(
            "a block step's tables exceed the block images",
        ));
    }
    let blocks = Images::new(
        device,
        Pinned::at_least("block image host copies", host, 2 * stride)?,
        stride,
        events,
    )?;
    let mut programs = HashMap::new();
    for (b, program) in get("programs")?.cast::<PyDict>()?.iter() {
        let program = program.extract::<PyRef<'_, GaudiProgram>>()?.program()?;
        if program.registers() != REGISTERS || program.sections().len() != 2 {
            return Err(PyValueError::new_err(
                "a block step program needs upload and run",
            ));
        }
        programs.insert(b.extract::<usize>()?, program);
    }
    let (ihost, istride, ievents, kv_rows, kv_stride): (
        PyReadwriteArray1<'_, u8>,
        usize,
        i32,
        usize,
        usize,
    ) = get("inject_image")?.extract()?;
    if kv_rows + drafter.layers * kv_stride > istride {
        return Err(PyValueError::new_err(
            "the injection tables exceed their image",
        ));
    }
    let inject = Images::new(
        0,
        Pinned::at_least("injection host copies", ihost, 2 * istride)?,
        istride,
        ievents,
    )?;
    let (next, next_program, next_event): (
        PyReadwriteArray1<'_, u8>,
        PyRef<'_, GaudiProgram>,
        i32,
    ) = get("next")?.extract()?;
    let next = Pinned::new("next drafts", next)?;
    if next.bytes < 4 * drafter.block * drafter.max_rows {
        return Err(PyValueError::new_err(
            "the next drafts' buffer is too small",
        ));
    }
    Ok(Dflash {
        drafter,
        row_buckets,
        layouts,
        blocks,
        programs,
        inject,
        inject_kv_rows: kv_rows,
        inject_kv_stride: kv_stride,
        next,
        next_program: next_program.program()?,
        next_event,
        max_positions: item(d, "max_positions")?,
    })
}

/// The MTP KV block rows from the model dict (`mtp_block_rows`, None without MTP).
fn mtp_geometry(model: &Bound<'_, PyDict>) -> PyResult<Option<i64>> {
    match model.get_item("mtp_block_rows")? {
        Some(value) => Ok(Some(value.extract()?)),
        None => Ok(None),
    }
}

/// One enqueued decode step: `result` waits for its staged samples once.
#[pyclass(module = "resin._resin")]
pub(crate) struct GlmCompletion {
    executor: Py<GlmExecutor>,
    id: u64,
    stage: usize,
    params: Vec<[f32; PARAMS]>,
    scores: bool,
    positions: Vec<i64>,
    /// the sampled tokens, once read
    tokens: Vec<i64>,
    output: Option<Py<ForwardOutput>>,
}

impl GlmCompletion {
    /// Read the step's samples (once), and leave the executor's open steps.
    fn finish(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut GlmExecutor,
    ) -> PyResult<Py<ForwardOutput>> {
        if let Some(output) = &this.borrow().output {
            return Ok(output.clone_ref(py));
        }
        let (id, stage, rows) = {
            let c = this.borrow();
            (c.id, c.stage, c.positions.len())
        };
        executor.wait_stage(py, stage, false)?;
        let out = executor.staged(stage, rows, false);
        let output = {
            let c = this.borrow();
            out.output(py, c.scores, &c.params, None)?
        };
        executor.inflight.retain(|c| c.borrow(py).id != id);
        if executor.staging.owner[stage] == Some(id) {
            executor.staging.owner[stage] = None;
        }
        let mut c = this.borrow_mut();
        c.tokens = out.tokens.iter().map(|&t| i64::from(t)).collect();
        c.output = Some(output.clone_ref(py));
        Ok(output)
    }
}

#[pymethods]
impl GlmCompletion {
    fn result(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<ForwardOutput>> {
        if let Some(output) = &slf.borrow().output {
            return Ok(output.clone_ref(py));
        }
        let executor = slf.borrow().executor.clone_ref(py);
        let mut executor = executor.borrow_mut(py);
        Self::finish(&slf, py, &mut executor)
    }

    #[getter]
    fn feedback(slf: Bound<'_, Self>) -> PyResult<GlmFeedback> {
        let positions = slf.borrow().positions.iter().map(|&p| p + 1).collect();
        Ok(GlmFeedback {
            positions,
            completion: slf.unbind(),
        })
    }
}

/// A plain decode step's continuation: each row's next position; its tokens stay on the device.
#[pyclass(module = "resin._resin", frozen)]
pub(crate) struct GlmFeedback {
    positions: Vec<i64>,
    completion: Py<GlmCompletion>,
}

#[pymethods]
impl GlmFeedback {
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        numpy::PyArray1::from_slice(py, &self.positions)
    }
}

/// A settled verification: each row's accepted drafts, emitted tokens, next anchor position and anchor, its next
/// drafts (the first from the MTP pass, the rest zeros until a chain makes them), and the emitted columns'
/// logprobs.
#[derive(Clone)]
struct Done {
    accepted: Vec<i64>,
    tokens: Vec<Vec<i64>>,
    positions: Vec<i64>,
    anchors: Vec<i64>,
    next: Vec<Vec<i64>>,
    scored: Option<(Vec<f32>, Vec<i32>, Vec<f32>)>,
}

/// An enqueued speculative step and its continuation's feedback (`runner._Verification`): `settle` waits for its
/// samples, accepts each row's longest draft prefix that the target's samples confirm, commits the accepted
/// tokens' KDA state, reads each row's next first draft, and leaves each row's next anchor.
#[pyclass(module = "resin._resin")]
pub(crate) struct GlmVerification {
    executor: Py<GlmExecutor>,
    id: u64,
    /// the verify step's bucket
    t: usize,
    stage: usize,
    /// the MTP pass's staging set (None with the drafter)
    mstage: Option<usize>,
    /// each row's sampler inputs at its first position (for the logprob columns)
    params: Vec<[f32; PARAMS]>,
    scores: bool,
    rows: Vec<VerifyRow>,
    drafts: Vec<Vec<i64>>,
    /// the rows that verified drafts of their own (None: all)
    own: Option<Vec<bool>>,
    done: Option<(Done, Py<ForwardOutput>)>,
}

impl GlmVerification {
    fn settle(
        this: &Bound<'_, Self>,
        py: Python<'_>,
        executor: &mut GlmExecutor,
    ) -> PyResult<Done> {
        if let Some((done, _)) = &this.borrow().done {
            return Ok(done.clone());
        }
        let (id, t, stage, mstage, b) = {
            let v = this.borrow();
            (v.id, v.t, v.stage, v.mstage, v.rows.len())
        };
        let tq = executor.tq;
        step_trace::mark("vs0", id as i64);
        executor.wait_stage(py, stage, false)?;
        step_trace::mark("vs1", id as i64);
        let out = executor.staged(stage, b * tq, false);
        let samples: Vec<i64> = out.tokens.iter().map(|&t| i64::from(t)).collect();
        let (done, output) = {
            let v = this.borrow();
            let flat: Vec<i64> = v.drafts.iter().flatten().copied().collect();
            let (accepted, emitted) = executor.stats.accept(&flat, &samples, b, tq);
            // the commit: each row's first accepted + 1 tokens' KDA updates, its last one's hidden
            let commit = {
                let c = executor
                    .commit
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("verify steps need the commit"))?;
                let layout = CommitLayout {
                    bucket: t / tq,
                    layers: c.layers,
                    commit_bases: 0,
                    commit_counts: c.at.0,
                    hidden_table: c.at.1,
                };
                let rows: Vec<StateRow<'_>> = v
                    .rows
                    .iter()
                    .map(|row| StateRow {
                        pos0: row.start,
                        src: row.state_src,
                        blocks: &row.blocks.states,
                        first: row.blocks.state_first,
                    })
                    .collect();
                // SAFETY: the commit tables are read by the next submit's ring uploads only, which copy them.
                let host = unsafe { c.host.bytes_mut(0, c.host.bytes) };
                fill_commit(&layout, &c.state, host, c.host.bytes, &rows, &accepted, tq)
                    .map_err(PyValueError::new_err)?;
                executor.io(t, IO_COMMIT)?
            };
            executor.submit(py, &[commit], "commit")?;
            step_trace::mark("vcommit", 0);
            let sequences: Vec<u64> = v.rows.iter().map(|r| r.sequence).collect();
            executor
                .choice
                .verified(&sequences, &accepted, v.own.as_deref());
            let k = executor.k;
            let positions: Vec<i64> = (0..b).map(|i| v.rows[i].start + accepted[i] + 1).collect();
            let anchors: Vec<i64> = emitted
                .iter()
                .map(|row| *row.last().expect("a sample"))
                .collect();
            let next: Vec<Vec<i64>> = match mstage {
                Some(mstage) => {
                    step_trace::mark("vm0", 0);
                    executor.wait_stage(py, mstage, true)?;
                    step_trace::mark("vm1", 0);
                    let q = executor.staged(mstage, b * tq, true).tokens;
                    (0..b)
                        .map(|i| {
                            let mut d = vec![0i64; k];
                            d[0] = i64::from(q[i * tq + accepted[i] as usize]);
                            d
                        })
                        .collect()
                }
                None => {
                    // the block step after each row's new anchor makes all of the next step's drafts
                    let rows: Vec<(i64, &Blocks)> =
                        (0..b).map(|i| (positions[i], &v.rows[i].blocks)).collect();
                    executor.draft_blocks(py, &rows, &anchors)?
                }
            };
            let full = k == 1 || mstage.is_none();
            for i in 0..b {
                executor.carried.insert(
                    sequences[i],
                    Carried {
                        position: positions[i],
                        anchor: anchors[i],
                        drafts: next[i].clone(),
                        full,
                    },
                );
            }
            let counts: Vec<usize> = emitted.iter().map(Vec::len).collect();
            let token_ids: Vec<u32> = emitted.iter().flatten().map(|&t| t as u32).collect();
            let mut indptr = vec![0u32];
            for &count in &counts {
                indptr.push(indptr.last().copied().unwrap_or(0) + count as u32);
            }
            let scored = v.scores.then(|| {
                let picked: Vec<usize> = (0..b)
                    .flat_map(|i| (0..counts[i]).map(move |j| i * tq + j))
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
                (lp, ids, values)
            });
            let (scores, top) = match &scored {
                Some((lp, ids, values)) => {
                    let (s, t) = score_columns(&v.params, &counts, lp, ids, values);
                    (Some(s), t)
                }
                None => (None, None),
            };
            let output = Py::new(
                py,
                ForwardOutput::native(py, token_ids, indptr, scores, top)?,
            )?;
            (
                Done {
                    accepted,
                    tokens: emitted,
                    positions,
                    anchors,
                    next,
                    scored,
                },
                output,
            )
        };
        if executor.staging.owner[stage] == Some(id) {
            executor.staging.owner[stage] = None;
        }
        if let (Some(m), Some(mstage)) = (executor.mtp.as_mut(), mstage)
            && m.staging.owner[mstage] == Some(id)
        {
            m.staging.owner[mstage] = None;
        }
        if executor
            .open
            .as_ref()
            .is_some_and(|open| open.borrow(py).id == id)
        {
            executor.open = None;
        }
        this.borrow_mut().done = Some((done.clone(), output));
        Ok(done)
    }
}

#[pymethods]
impl GlmVerification {
    fn result(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<ForwardOutput>> {
        if let Some((_, output)) = &slf.borrow().done {
            return Ok(output.clone_ref(py));
        }
        let executor = slf.borrow().executor.clone_ref(py);
        // the commit and the drafter's block step are device work on the card: one TP step of a layout
        GlmExecutor::on_card(executor.bind(py), py, || {
            let mut executor = executor.borrow_mut(py);
            Self::settle(&slf, py, &mut executor).map(drop)
        })?;
        let v = slf.borrow();
        Ok(v.done.as_ref().expect("settled").1.clone_ref(py))
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
        let (done, _) = self
            .done
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("not finished"))?;
        Ok(numpy::PyArray1::from_slice(py, &done.positions))
    }

    /// Each row's next anchor (after `finish`).
    #[getter]
    fn anchors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
        let (done, _) = self
            .done
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("not finished"))?;
        Ok(numpy::PyArray1::from_slice(py, &done.anchors))
    }
}
