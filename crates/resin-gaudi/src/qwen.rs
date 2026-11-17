//! Qwen3.5/3.8 step tables, written into one host copy of the device table image of a step mode. This is the
//! port of the per-step table builders of `resin_models_gaudi/qwen3_5/model.py` (`_target`,
//! `_write_state_tables`, `_state_fields`, `_slot_tables`) and `resin_models_gaudi/common/attention.py`
//! (`kv_rows`, `plan_slots`), and of the host side of `common/sampling.py` (`philox_uniform`, `row_params`).
//! The numpy code was the reference: the tables are the same values. (The Python step code of `qwen3_5/model.py`
//! is deleted; the CPU tests keep reference copies of its tables.)
//!
//! One image holds every table of a target pass, at offsets that [`Layout::new`] chooses for the mode. The
//! device copy is a single region, so a step uploads its tables in one copy: the fixed tables first, then the
//! slot tables of the launches the step makes.

// Index loops over rows and layers keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

use crate::sampling::PARAMS;
use crate::tables::{SMALL, Words, round};

pub const PAGE_TOKENS: i64 = 128;
/// Rows of an FP8 page: 128 keys of 256 bytes, two a row, then a row of their f32 scales (`kv_store8.c`).
pub const FP8_PAGE_ROWS: i64 = 65;

/// The formats of a full-attention layer's K and V pages (`qwen3_5/cache.py`): bf16 pages are 128 rows, one
/// key a row; FP8 pages are `FP8_PAGE_ROWS` rows, whose keys `kv_rows` gives as 256-byte units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pages {
    pub k8: bool,
    pub v8: bool,
}

impl Pages {
    pub const BF16: Pages = Pages {
        k8: false,
        v8: false,
    };

    pub fn rows(fp8: bool) -> i64 {
        if fp8 { FP8_PAGE_ROWS } else { PAGE_TOKENS }
    }

    /// The first rows of KV group `grp`'s K and V pages in the KV block at `block` (`kv_groups` groups).
    pub fn first(self, block: i64, kv_groups: i64, grp: i64) -> (i64, i64) {
        let (k, v) = (Self::rows(self.k8), Self::rows(self.v8));
        (block + grp * k, block + kv_groups * k + grp * v)
    }

    /// The `kv_rows` entry of the key at `slot` of the page at `first`: its row, or for an FP8 page its
    /// 256-byte unit.
    pub fn key(fp8: bool, first: i64, slot: i64) -> i64 {
        if fp8 { 2 * first + slot } else { first + slot }
    }
}
pub const SLOT_KEYS: i64 = 2048;
const SLOT_PAGES: usize = (SLOT_KEYS / PAGE_TOKENS) as usize;

/// What every step's tables depend on besides the step itself (one rank's dimensions).
#[derive(Clone, Debug)]
pub struct Model {
    /// GDN (linear attention) layers
    pub gdn_layers: usize,
    /// KV groups of a full-attention layer on this rank
    pub kv_groups: i64,
    /// the K and V page formats
    pub pages: Pages,
    /// query heads per KV group
    pub hpg: i64,
    pub state_layer_rows: i64,
    pub ssm_rows: i64,
    pub max_positions: i64,
}

/// A step shape (`_Mode`): `columns` tokens, `per_row` of them a row (1: decode; the width: verify), prefill
/// when `prefill` (one row of `columns` tokens), with `slots` slots a slot launch and room for `max_launches`
/// launches. With `wide_slots` (a multiple of `slots`, 0: none) a step's first slots go to launches of that many
/// slots, room for `max_wide`, and only the rest to the narrow launches (then `max_launches` is at most
/// `wide_slots / slots - 1`). With `wide_full`, the wide launches take instead the slots whose every key each
/// query of the block sees (a fused prefill's launches without a mask), and the narrow ones the others.
#[derive(Clone, Debug)]
pub struct Mode {
    pub name: String,
    pub columns: usize,
    pub per_row: usize,
    pub query_block: i64,
    pub slots: usize,
    pub prefill: bool,
    pub max_launches: usize,
    pub wide_slots: usize,
    pub max_wide: usize,
    pub wide_full: bool,
}

impl Mode {
    /// rows of the mode's tables (a prefill has one)
    pub fn rows(&self) -> usize {
        if self.prefill {
            1
        } else {
            self.columns / self.per_row
        }
    }

    /// columns that the head samples (a prefill samples its last token only)
    pub fn sampled(&self) -> usize {
        if self.prefill { 1 } else { self.columns }
    }
}

/// Byte offsets of a mode's tables in its image.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub tok: usize,
    pub pos: usize,
    pub meta: usize,
    pub last: usize,
    pub samp_params: usize,
    pub kv_rows: usize,
    /// decode / verify: per GDN layer `gdn_bases` (4, rows) then `conv_dst` (rows,), `state_stride` apart;
    /// prefill: per GDN layer [conv_src, ssm_src, conv_dst, ssm_dst] one SMALL table each
    pub state: usize,
    pub state_stride: usize,
    /// `conv_dst` within a layer's decode state tables
    pub conv_dst: usize,
    /// the first slot launch's tables, `launch_stride` apart: slots (S, 8), then K and V pages (S, 16)
    pub slots: usize,
    pub launch_stride: usize,
    pub k_pages: usize,
    pub v_pages: usize,
    /// the first wide launch's tables (after the narrow ones), `wide_stride` apart, with their K and V pages at
    /// `wide_k_pages` and `wide_v_pages` (all 0 without wide launches)
    pub wide: usize,
    pub wide_stride: usize,
    pub wide_k_pages: usize,
    pub wide_v_pages: usize,
    /// the image's size with every launch
    pub nbytes: usize,
}

impl Layout {
    pub fn new(model: &Model, mode: &Mode) -> Self {
        Self::with_extras(model, mode, &[]).0
    }

    /// The layout with more tables (of `extras` bytes each) after the fixed ones and before the slot tables:
    /// the layout and each extra table's offset.
    pub fn with_extras(model: &Model, mode: &Mode, extras: &[usize]) -> (Self, Vec<usize>) {
        let (t, rows) = (mode.columns, mode.rows());
        let mut at = 0usize;
        let mut take = |nbytes: usize| {
            let offset = at;
            at += round(nbytes);
            offset
        };
        let tok = take(4 * t);
        let pos = take(4 * t);
        let meta = take(16);
        let last = take(4);
        let samp_params = take(4 * PARAMS * mode.sampled());
        let kv_rows = take(4 * 8 * t);
        let (state_stride, conv_dst) = if mode.prefill {
            (4 * SMALL, 0)
        } else {
            (round(16 * rows) + round(4 * rows), round(16 * rows))
        };
        let state = take(model.gdn_layers * state_stride);
        let extra = extras.iter().map(|&nbytes| take(nbytes)).collect();
        let s = mode.slots;
        let (k_pages, v_pages) = (
            round(4 * 8 * s),
            round(4 * 8 * s) + round(4 * SLOT_PAGES * s),
        );
        let launch_stride = v_pages + round(4 * SLOT_PAGES * s);
        let slots = at;
        let wide = slots + mode.max_launches * launch_stride;
        let (wide_k_pages, wide_v_pages, wide_stride) = if mode.wide_slots > 0 {
            Attention::launch_tables(mode.wide_slots)
        } else {
            (0, 0, 0)
        };
        let layout = Layout {
            tok,
            pos,
            meta,
            last,
            samp_params,
            kv_rows,
            state,
            state_stride,
            conv_dst,
            slots,
            launch_stride,
            k_pages,
            v_pages,
            wide,
            wide_stride,
            wide_k_pages,
            wide_v_pages,
            nbytes: wide + mode.max_wide * wide_stride,
        };
        (layout, extra)
    }

    /// the bytes to upload for a step of `launches` narrow and `wide` wide slot launches
    pub fn upload_bytes(&self, launches: usize, wide: usize) -> usize {
        if wide > 0 {
            self.wide + wide * self.wide_stride
        } else {
            self.slots + launches * self.launch_stride
        }
    }
}

/// One step's rows (sequences), flat: per row its first position, its STATE block rows (-1: a fresh source),
/// and its KV block rows (`blocks_indptr` delimits each row's part).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub pos0: &'a [i64],
    pub state_src: &'a [i64],
    pub state_dst: &'a [i64],
    pub blocks_indptr: &'a [i64],
    pub blocks: &'a [i64],
}

impl Rows<'_> {
    pub fn len(&self) -> usize {
        self.pos0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos0.is_empty()
    }

    fn blocks(&self, row: usize) -> &[i64] {
        &self.blocks[self.blocks_indptr[row] as usize..self.blocks_indptr[row + 1] as usize]
    }

    fn validate(&self, mode: &Mode) -> Result<(), String> {
        let n = self.len();
        if n == 0 || n > mode.rows() {
            return Err(format!(
                "{n} rows for mode {} of {} rows",
                mode.name,
                mode.rows()
            ));
        }
        if self.state_src.len() != n || self.state_dst.len() != n {
            return Err("state_src and state_dst need one entry a row".into());
        }
        if self.blocks_indptr.len() != n + 1
            || self.blocks_indptr[0] != 0
            || self.blocks_indptr.windows(2).any(|pair| pair[0] > pair[1])
            || self.blocks_indptr[n] as usize != self.blocks.len()
        {
            return Err("blocks_indptr does not delimit the rows".into());
        }
        Ok(())
    }
}

/// Stored slot of logical index p within a page (`loading.convert.sigma_inv`).
pub fn sigma_inv(p: i64) -> i64 {
    let (block, r) = (p & !127, p & 127);
    block + if r < 64 { 2 * r } else { 2 * (r - 64) + 1 }
}

/// What a filled step needs from the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filled {
    /// narrow slot launches of each full-attention layer
    pub launches: usize,
    /// wide slot launches of each full-attention layer (made before the narrow ones)
    pub wide: usize,
    /// bytes of the image to upload
    pub nbytes: usize,
}

/// A target pass's inputs. `real`: a prefill's real tokens (of its one row), else the step's real rows.
/// `tokens`: the columns' tokens (None: the program binds another device buffer as the tokens). `params`: PARAMS
/// sampler values for each sampled column (`mode.sampled()` of them; fewer leave the rest zero).
#[derive(Clone, Copy)]
pub struct Step<'a> {
    pub rows: Rows<'a>,
    pub real: usize,
    pub tokens: Option<&'a [i64]>,
    pub params: &'a [f32],
}

/// Write a target pass's tables into `host` (a host copy of the mode's image, 4-byte aligned).
pub fn fill_target(
    model: &Model,
    mode: &Mode,
    layout: &Layout,
    host: &mut [u8],
    step: &Step<'_>,
) -> Result<Filled, String> {
    let Step {
        rows,
        real,
        tokens,
        params,
    } = *step;
    let rows = &rows;
    rows.validate(mode)?;
    let (t, n, q) = (mode.columns, rows.len(), mode.per_row);
    if host.len() < layout.nbytes || !(host.as_ptr() as usize).is_multiple_of(4) {
        return Err("the host copy is too small or misaligned".into());
    }
    if params.len() > PARAMS * mode.sampled() || !params.len().is_multiple_of(PARAMS) {
        return Err(format!(
            "{} sampler values for mode {}",
            params.len(),
            mode.name
        ));
    }
    if mode.prefill && (n != 1 || real == 0 || real > t) {
        return Err(format!(
            "a prefill of {real} tokens in {n} rows for mode {}",
            mode.name
        ));
    }
    if !mode.prefill && real != n {
        return Err(format!("{real} real rows of {n}"));
    }
    // SAFETY: the buffer is 4-byte aligned and holds nbytes bytes; every table write stays inside it.
    let words = unsafe {
        std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<i32>(), layout.nbytes / 4)
    };
    let mut w = Words(words);
    // ---- the columns
    w.fill(layout.pos, t, 0);
    if mode.prefill {
        // positions past the last one clamp (the padding's, and as the reference does, the real ones)
        let pos0 = rows.pos0[0];
        for c in 0..t {
            w.set(
                layout.pos,
                c,
                (pos0 + c as i64).min(model.max_positions - 1),
            );
        }
    } else {
        for b in 0..n {
            if rows.pos0[b] + q as i64 > model.max_positions {
                return Err(format!("position exceeds {}", model.max_positions));
            }
            for j in 0..q {
                w.set(layout.pos, b * q + j, rows.pos0[b] + j as i64);
            }
        }
    }
    let meta = [
        if mode.prefill { rows.pos0[0] } else { 0 },
        real as i64,
        0,
        0,
    ];
    for (k, &value) in meta.iter().enumerate() {
        w.set(layout.meta, k, value);
    }
    if mode.prefill {
        w.set(layout.last, 0, real as i64 - 1);
    }
    if let Some(tokens) = tokens {
        if tokens.len() != t {
            return Err(format!("{} tokens for {t} columns", tokens.len()));
        }
        for (c, &token) in tokens.iter().enumerate() {
            w.set(layout.tok, c, token);
        }
    }
    let samp = layout.samp_params / 4;
    w.fill(layout.samp_params, PARAMS * mode.sampled(), 0);
    for (k, &value) in params.iter().enumerate() {
        w.0[samp + k] = value.to_bits() as i32;
    }
    // ---- GDN states: [conv_src, ssm_src, conv_dst, ssm_dst] of each row, a layer apart
    let field = |base: i64, layer: usize, conv: bool| -> i64 {
        if base < 0 {
            -1
        } else {
            base + layer as i64 * model.state_layer_rows + if conv { model.ssm_rows } else { 0 }
        }
    };
    for l in 0..model.gdn_layers {
        let at = layout.state + l * layout.state_stride;
        if mode.prefill {
            w.fill(at, layout.state_stride / 4, 0);
            let (src, dst) = (rows.state_src[0], rows.state_dst[0]);
            let fields = [
                field(src, l, true),
                field(src, l, false),
                field(dst, l, true),
                field(dst, l, false),
            ];
            for (k, &value) in fields.iter().enumerate() {
                w.set(at + k * SMALL, 0, value);
            }
        } else {
            w.fill(at, layout.state_stride / 4, -1);
            for b in 0..n {
                let (src, dst) = (rows.state_src[b], rows.state_dst[b]);
                let fields = [
                    field(src, l, true),
                    field(src, l, false),
                    field(dst, l, true),
                    field(dst, l, false),
                ];
                for (k, &value) in fields.iter().enumerate() {
                    w.set(at, 4 * b + k, value);
                }
                w.set(at + layout.conv_dst, b, fields[2]);
            }
        }
    }
    // ---- attention: each column's K and V rows, then the slots of every (row, query block, key chunk, group)
    let runs: Vec<SlotRun<'_>> = (0..n)
        .map(|b| {
            let (column, real_b) = if mode.prefill { (0, real) } else { (b * q, q) };
            SlotRun {
                column,
                pos0: rows.pos0[b],
                real: real_b,
                blocks: rows.blocks(b),
            }
        })
        .collect();
    let attention = Attention::of(model, mode, layout);
    kv_rows_table(
        &mut w,
        layout.kv_rows,
        t,
        model.kv_groups,
        model.pages,
        &runs,
    )?;
    let (wide, launches) = slot_tables(&mut w, &attention, &mode.name, &runs)?;
    Ok(Filled {
        launches,
        wide,
        nbytes: layout.upload_bytes(launches, wide),
    })
}

/// A row's columns in a step's attention tables (`common.attention.SlotRow`): its first column, its first position,
/// its real columns, and the first arena row of each of its KV blocks (page p's at `blocks[p]`).
#[derive(Clone, Copy, Debug)]
pub struct SlotRun<'a> {
    pub column: usize,
    pub pos0: i64,
    pub real: usize,
    pub blocks: &'a [i64],
}

/// The shape of a step's slot launches and their tables' place in an image: the first launch's tables at
/// `at`, `stride` apart, each the slots (S, 8) then the K pages (S, 16) at `k_pages` and the V pages at `v_pages`.
/// With `wide`, the step's first slots go to its wide launches.
#[derive(Clone, Copy, Debug)]
pub struct Attention {
    pub kv_groups: i64,
    pub pages: Pages,
    pub hpg: i64,
    pub slots: usize,
    pub query_block: i64,
    pub max_launches: usize,
    pub at: usize,
    pub stride: usize,
    pub k_pages: usize,
    pub v_pages: usize,
    pub wide: Option<WideLaunches>,
}

/// Wide slot launches: `slots` slots each, room for `max`, the first one's tables at `at`, `stride` apart, with
/// their K and V pages at `k_pages` and `v_pages`. They take the step's first slots, or with `full` the slots
/// whose every key each query of the block sees.
#[derive(Clone, Copy, Debug)]
pub struct WideLaunches {
    pub full: bool,
    pub slots: usize,
    pub max: usize,
    pub at: usize,
    pub stride: usize,
    pub k_pages: usize,
    pub v_pages: usize,
}

/// (wide, narrow) launches of `total` slots: whole wide launches of `wide_slots` slots first, and the rest in
/// narrow launches of `slots`; one more wide launch instead when the rest needs more than `max_narrow`.
pub fn split_launches(
    total: usize,
    slots: usize,
    wide_slots: usize,
    max_narrow: usize,
) -> (usize, usize) {
    if wide_slots == 0 {
        return (0, total.div_ceil(slots));
    }
    let wide = total / wide_slots;
    let narrow = (total - wide * wide_slots).div_ceil(slots);
    if narrow > max_narrow {
        (wide + 1, 0)
    } else {
        (wide, narrow)
    }
}

impl Attention {
    /// A target pass's slot launches: the model's attention, the mode's shape and the layout's tables.
    pub fn of(model: &Model, mode: &Mode, layout: &Layout) -> Self {
        Self {
            kv_groups: model.kv_groups,
            pages: model.pages,
            hpg: model.hpg,
            slots: mode.slots,
            query_block: mode.query_block,
            max_launches: mode.max_launches,
            at: layout.slots,
            stride: layout.launch_stride,
            k_pages: layout.k_pages,
            v_pages: layout.v_pages,
            wide: (mode.wide_slots > 0).then_some(WideLaunches {
                full: mode.wide_full,
                slots: mode.wide_slots,
                max: mode.max_wide,
                at: layout.wide,
                stride: layout.wide_stride,
                k_pages: layout.wide_k_pages,
                v_pages: layout.wide_v_pages,
            }),
        }
    }

    /// (k_pages, v_pages, stride) of `slots` slots a launch: the offsets within a launch's tables and their size
    pub fn launch_tables(slots: usize) -> (usize, usize, usize) {
        let k_pages = round(4 * 8 * slots);
        let v_pages = k_pages + round(4 * SLOT_PAGES * slots);
        (k_pages, v_pages, v_pages + round(4 * SLOT_PAGES * slots))
    }
}

/// Each column's K rows (groups in entries 0 .. 3) and V rows (4 .. 7) of the runs (`kv_rows`): (t, 8) at `at`,
/// -1 for padding and for groups past `kv_groups`; an FP8 page's keys as 256-byte units (`Pages::key`).
pub(crate) fn kv_rows_table(
    w: &mut Words<'_>,
    at: usize,
    t: usize,
    kv_groups: i64,
    pages: Pages,
    runs: &[SlotRun<'_>],
) -> Result<(), String> {
    let g = kv_groups;
    w.fill(at, 8 * t, -1);
    for (b, run) in runs.iter().enumerate() {
        for j in 0..run.real {
            let pos = run.pos0 + j as i64;
            let page = (pos / PAGE_TOKENS) as usize;
            let block = *run
                .blocks
                .get(page)
                .ok_or_else(|| format!("row {b} has no KV block for position {pos}"))?;
            let r = pos % PAGE_TOKENS;
            for grp in 0..g {
                let (kp, vp) = pages.first(block, g, grp);
                w.set(
                    at,
                    8 * (run.column + j) + grp as usize,
                    Pages::key(pages.k8, kp, r),
                );
                w.set(
                    at,
                    8 * (run.column + j) + 4 + grp as usize,
                    Pages::key(pages.v8, vp, sigma_inv(r)),
                );
            }
        }
    }
    Ok(())
}

/// The slot tables of the runs (`plan_slots`): every (row, query block, key chunk, group) whose keys a query of
/// the block can see, in that order: the first ones in whole wide launches (`attention.wide`), the rest
/// `attention.slots` a launch, the last launch padded with group -1 slots. Returns (wide, narrow) launches.
pub(crate) fn slot_tables(
    w: &mut Words<'_>,
    attention: &Attention,
    name: &str,
    runs: &[SlotRun<'_>],
) -> Result<(usize, usize), String> {
    let a = attention;
    let g = a.kv_groups;
    let s_count = a.slots;
    let qb = a.query_block;
    let full = a.wide.is_some_and(|wide| wide.full);
    // every query of a block that starts at position qpos sees each key of the chunk at key0
    let all_seen =
        |key0: i64, qpos: i64, klen: i64| key0 + SLOT_KEYS - 1 <= qpos && key0 + SLOT_KEYS <= klen;
    // the step's slots (and with full wide launches, the slots that need no mask), then how many wide and
    // narrow launches hold them
    let (mut s_need, mut s_full) = (0usize, 0usize);
    for run in runs {
        let (pos0, real_b) = (run.pos0, run.real as i64);
        let klen = pos0 + real_b;
        let mut t0 = 0i64;
        while t0 < real_b {
            let visible = (pos0 + t0 + qb).min(klen);
            let chunks = (visible + SLOT_KEYS - 1) / SLOT_KEYS;
            s_need += chunks as usize * g as usize;
            if full {
                let seen = (0..chunks).filter(|&c| all_seen(c * SLOT_KEYS, pos0 + t0, klen));
                s_full += seen.count() * g as usize;
            }
            t0 += qb;
        }
    }
    let (w_count, max_wide) = a.wide.map_or((0, 0), |wide| (wide.slots, wide.max));
    let (wide, launches) = if full {
        (
            s_full.div_ceil(w_count),
            (s_need - s_full).div_ceil(s_count),
        )
    } else {
        split_launches(s_need, s_count, w_count, a.max_launches)
    };
    if launches > a.max_launches || wide > max_wide {
        return Err(format!(
            "{s_need} attention slots exceed the launches of mode {name} ({} of {s_count}, {max_wide} of {w_count})",
            a.max_launches
        ));
    }
    // slot k of the wide or the narrow launches: (table base, index in the launch, K pages offset, V pages
    // offset, the launch's table bytes)
    let place = |to_wide: bool, k: usize| -> (usize, usize, usize, usize, usize) {
        match a.wide {
            Some(wl) if to_wide => (
                wl.at + (k / w_count) * wl.stride,
                k % w_count,
                wl.k_pages,
                wl.v_pages,
                wl.stride,
            ),
            _ => (
                a.at + (k / s_count) * a.stride,
                k % s_count,
                a.k_pages,
                a.v_pages,
                a.stride,
            ),
        }
    };
    // slots written to the wide and the narrow launches, each in the step's order
    let (mut n_wide, mut n_narrow) = (0usize, 0usize);
    for (b, run) in runs.iter().enumerate() {
        let (column, pos0, real_b) = (run.column, run.pos0, run.real as i64);
        let blocks = run.blocks;
        let klen = pos0 + real_b;
        let mut t0 = 0i64;
        while t0 < real_b {
            let visible = (pos0 + t0 + qb).min(klen);
            let chunks = (visible + SLOT_KEYS - 1) / SLOT_KEYS;
            for c in 0..chunks {
                let key0 = c * SLOT_KEYS;
                let first = (key0 / PAGE_TOKENS) as usize;
                let live = (SLOT_PAGES as i64).min((visible - key0 + PAGE_TOKENS - 1) / PAGE_TOKENS)
                    as usize;
                if blocks.len() < first + live {
                    return Err(format!(
                        "row {b} has {} KV blocks, its slots need {}",
                        blocks.len(),
                        first + live
                    ));
                }
                let seen = all_seen(key0, pos0 + t0, klen);
                for grp in 0..g {
                    let to_wide = if full {
                        seen
                    } else {
                        n_wide + n_narrow < wide * w_count
                    };
                    let (base, k, k_off, v_off, stride) =
                        place(to_wide, if to_wide { n_wide } else { n_narrow });
                    if k == 0 {
                        w.fill(base, stride / 4, 0);
                    }
                    let fields = [
                        grp,
                        a.hpg * (column as i64 + t0),
                        pos0 + t0,
                        key0,
                        klen,
                        column as i64 + t0,
                        0,
                        0,
                    ];
                    for (f, &value) in fields.iter().enumerate() {
                        w.set(base, 8 * k + f, value);
                    }
                    for lane in 0..SLOT_PAGES {
                        let (kp, vp) = if lane < live {
                            a.pages.first(blocks[first + lane], g, grp)
                        } else {
                            (-1, -1)
                        };
                        w.set(base + k_off, SLOT_PAGES * k + lane, kp);
                        w.set(base + v_off, SLOT_PAGES * k + lane, vp);
                    }
                    if to_wide {
                        n_wide += 1;
                    } else {
                        n_narrow += 1;
                    }
                }
            }
            t0 += qb;
        }
    }
    debug_assert_eq!(n_wide + n_narrow, s_need);
    // pad the last launch of each kind with group -1 slots
    for (to_wide, mut n, end) in [
        (true, n_wide, wide * w_count),
        (false, n_narrow, launches * s_count),
    ] {
        while n < end {
            let (base, k, k_off, v_off, stride) = place(to_wide, n);
            if k == 0 {
                w.fill(base, stride / 4, 0);
            }
            w.set(base, 8 * k, -1);
            w.fill(base + k_off + 4 * SLOT_PAGES * k, SLOT_PAGES, -1);
            w.fill(base + v_off + 4 * SLOT_PAGES * k, SLOT_PAGES, -1);
            n += 1;
        }
    }
    Ok((wide, launches))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decode_step_fills_its_tables() {
        let model = Model {
            gdn_layers: 2,
            kv_groups: 4,
            pages: Pages::BF16,
            hpg: 6,
            state_layer_rows: 6400,
            ssm_rows: 6144,
            max_positions: 8192,
        };
        let mode = Mode {
            name: "d2".into(),
            columns: 2,
            per_row: 1,
            query_block: 1,
            slots: 16,
            prefill: false,
            max_launches: 8,
            wide_slots: 0,
            max_wide: 0,
            wide_full: false,
        };
        let layout = Layout::new(&model, &mode);
        let mut host = vec![0u32; layout.nbytes / 4];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), layout.nbytes)
        };
        let rows = Rows {
            pos0: &[5, 3000],
            state_src: &[100, -1],
            state_dst: &[100, 200],
            blocks_indptr: &[0, 1, 25],
            blocks: &[1024; 25],
        };
        let step = Step {
            rows,
            real: 2,
            tokens: Some(&[7, 8]),
            params: &[0.0; 16],
        };
        let filled = fill_target(&model, &mode, &layout, bytes, &step).unwrap();
        // row 0: one chunk x 4 groups; row 1 (position 3000): two chunks x 4 groups
        assert_eq!(filled.launches, 1);
        assert_eq!(filled.nbytes, layout.slots + layout.launch_stride);
        let word = |offset: usize, i: usize| host[offset / 4 + i] as i32;
        assert_eq!((word(layout.pos, 0), word(layout.pos, 1)), (5, 3000));
        assert_eq!((word(layout.tok, 0), word(layout.tok, 1)), (7, 8));
        // gdn bases of layer 1, row 0: conv_src, ssm_src, conv_dst, ssm_dst
        let l1 = layout.state + layout.state_stride;
        assert_eq!(
            (word(l1, 0), word(l1, 1), word(l1, 2), word(l1, 3)),
            (100 + 6400 + 6144, 100 + 6400, 100 + 6400 + 6144, 100 + 6400)
        );
        assert_eq!(word(l1, 4), -1); // row 1 has a fresh source
        assert_eq!(word(l1 + layout.conv_dst, 1), 200 + 6400 + 6144);
        // the ninth slot (row 1, chunk 1, group 0): key0 2048; twelve slots, then group -1 padding
        assert_eq!(word(layout.slots, 8 * 8 + 3), 2048);
        assert_eq!(word(layout.slots, 8 * 8), 0);
        assert_eq!(word(layout.slots, 8 * 12), -1);
    }

    #[test]
    fn launches_split_into_wide_then_narrow() {
        assert_eq!(split_launches(12, 16, 0, 8), (0, 1));
        assert_eq!(split_launches(12, 16, 64, 3), (0, 1));
        assert_eq!(split_launches(48, 16, 64, 3), (0, 3));
        assert_eq!(split_launches(49, 16, 64, 3), (1, 0)); // four narrow launches: one padded wide one
        assert_eq!(split_launches(64, 16, 64, 3), (1, 0));
        assert_eq!(split_launches(256 + 20, 16, 64, 3), (4, 2));
    }

    #[test]
    fn a_long_decode_step_takes_wide_launches_first() {
        let model = Model {
            gdn_layers: 1,
            kv_groups: 4,
            pages: Pages::BF16,
            hpg: 6,
            state_layer_rows: 6400,
            ssm_rows: 6144,
            max_positions: 32768,
        };
        let narrow = Mode {
            name: "d2".into(),
            columns: 2,
            per_row: 1,
            query_block: 1,
            slots: 16,
            prefill: false,
            max_launches: 8,
            wide_slots: 0,
            max_wide: 0,
            wide_full: false,
        };
        let wide = Mode {
            max_launches: 3,
            wide_slots: 64,
            max_wide: 2,
            ..narrow.clone()
        };
        // row 0: 9 chunks (36 slots), row 1: 12 chunks (48 slots): 84 slots = one wide launch and two narrow ones
        let blocks: Vec<i64> = (0..400).map(|i| 1024 * (i + 1)).collect();
        let rows = Rows {
            pos0: &[17000, 24000],
            state_src: &[100, 100],
            state_dst: &[100, 200],
            blocks_indptr: &[0, 200, 400],
            blocks: &blocks,
        };
        let step = Step {
            rows,
            real: 2,
            tokens: Some(&[7, 8]),
            params: &[0.0; 16],
        };
        let run = |mode: &Mode| {
            let layout = Layout::new(&model, mode);
            let mut host = vec![0u32; layout.nbytes / 4];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), layout.nbytes)
            };
            let filled = fill_target(&model, mode, &layout, bytes, &step).unwrap();
            (layout, host, filled)
        };
        let (ln, hn, fnarrow) = run(&narrow);
        let (lw, hw, fwide) = run(&wide);
        assert_eq!((fnarrow.wide, fnarrow.launches), (0, 6));
        assert_eq!((fwide.wide, fwide.launches), (1, 2));
        assert_eq!(fwide.nbytes, lw.wide + lw.wide_stride);
        // slot k of the narrow plan is slot k of the wide plan: the same fields and pages
        let entry = |layout: &Layout, host: &[u32], k: usize, wide: bool| -> Vec<u32> {
            let (base, i, kp, vp) = if wide {
                (lw.wide, k, lw.wide_k_pages, lw.wide_v_pages)
            } else {
                let base = layout.slots + (k / 16) * layout.launch_stride;
                (base, k % 16, layout.k_pages, layout.v_pages)
            };
            let at = |o: usize, n: usize, len: usize| {
                host[(base + o) / 4 + n * len..(base + o) / 4 + (n + 1) * len].to_vec()
            };
            [at(0, i, 8), at(kp, i, SLOT_PAGES), at(vp, i, SLOT_PAGES)].concat()
        };
        for k in 0..84 {
            let got = if k < 64 {
                entry(&lw, &hw, k, true)
            } else {
                entry(&lw, &hw, k - 64, false)
            };
            assert_eq!(got, entry(&ln, &hn, k, false), "slot {k}");
        }
        // the narrow launches' padding: slots 84..96 are group -1
        assert_eq!(entry(&lw, &hw, 95 - 64, false)[0] as i32, -1);
    }

    #[test]
    fn a_prefill_chunk_puts_its_slots_without_a_mask_first() {
        let model = Model {
            gdn_layers: 1,
            kv_groups: 4,
            pages: Pages::BF16,
            hpg: 6,
            state_layer_rows: 6400,
            ssm_rows: 6144,
            max_positions: 32768,
        };
        let masked = Mode {
            name: "p1024".into(),
            columns: 1024,
            per_row: 1024,
            query_block: 512,
            slots: 16,
            prefill: true,
            max_launches: 4,
            wide_slots: 0,
            max_wide: 0,
            wide_full: false,
        };
        let split = Mode {
            wide_slots: 16,
            max_wide: 4,
            wide_full: true,
            ..masked.clone()
        };
        // a chunk at 6144: each of its two query blocks sees chunks 0 .. 2 whole and chunk 3 in part (32 slots:
        // 24 without a mask, then 8 with one)
        let blocks: Vec<i64> = (0..56).map(|i| 1024 * (i + 1)).collect();
        let rows = Rows {
            pos0: &[6144],
            state_src: &[100],
            state_dst: &[100],
            blocks_indptr: &[0, 56],
            blocks: &blocks,
        };
        let tokens = vec![7i64; 1024];
        let step = Step {
            rows,
            real: 1024,
            tokens: Some(&tokens),
            params: &[0.0; 8],
        };
        let run = |mode: &Mode| {
            let layout = Layout::new(&model, mode);
            let mut host = vec![0u32; layout.nbytes / 4];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), layout.nbytes)
            };
            let filled = fill_target(&model, mode, &layout, bytes, &step).unwrap();
            (layout, host, filled)
        };
        let (lm, hm, fm) = run(&masked);
        let (ls, hs, fs) = run(&split);
        assert_eq!((fm.wide, fm.launches), (0, 2));
        assert_eq!((fs.wide, fs.launches), (2, 1));
        assert_eq!(fs.nbytes, ls.wide + 2 * ls.wide_stride);
        let entry = |layout: &Layout, host: &[u32], k: usize, wide: bool| -> Vec<u32> {
            let (base, kp, vp, stride) = if wide {
                (
                    layout.wide,
                    layout.wide_k_pages,
                    layout.wide_v_pages,
                    layout.wide_stride,
                )
            } else {
                (
                    layout.slots,
                    layout.k_pages,
                    layout.v_pages,
                    layout.launch_stride,
                )
            };
            let (base, i) = (base + (k / 16) * stride, k % 16);
            let at = |o: usize, n: usize, len: usize| {
                host[(base + o) / 4 + n * len..(base + o) / 4 + (n + 1) * len].to_vec()
            };
            [at(0, i, 8), at(kp, i, SLOT_PAGES), at(vp, i, SLOT_PAGES)].concat()
        };
        // the masked plan's slots in order: those whose keys every query sees, then the others
        let plan: Vec<Vec<u32>> = (0..32).map(|k| entry(&lm, &hm, k, false)).collect();
        let seen = |e: &Vec<u32>| e[3] as i64 + SLOT_KEYS - 1 <= e[2] as i64;
        let (whole, part): (Vec<_>, Vec<_>) = plan.iter().partition(|e| seen(e));
        assert_eq!((whole.len(), part.len()), (24, 8));
        for (k, e) in whole.iter().enumerate() {
            assert_eq!(&entry(&ls, &hs, k, true), *e, "slot {k} without a mask");
        }
        for (k, e) in part.iter().enumerate() {
            assert_eq!(&entry(&ls, &hs, k, false), *e, "slot {k} with a mask");
        }
        // padding: the second wide launch after its 8 slots, the narrow launch after its 8
        assert_eq!(entry(&ls, &hs, 24, true)[0] as i32, -1);
        assert_eq!(entry(&ls, &hs, 8, false)[0] as i32, -1);
    }
}
