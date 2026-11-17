//! The drafter side of a Qwen3.5/3.8 verify step (DFlash 2 and DSpine drafters), written into host copies of
//! device table images. Ports of `resin_models_gaudi/common/dflash.py` (`inject_rows`, `block_rows_table`,
//! `plan_block_slots`, `Drafter._tables`, `Drafter.prepare_device`), of `qwen3_5/model.py` (`choose_width`,
//! `SURVIVAL`, `step_ms` and the `spec_fix` tables of `Qwen35Gaudi._fix`) and of
//! `resin.backend.speculative.accept_speculative_tokens`. The Python code was the reference: the tables are the
//! same values. (The Python step code of `qwen3_5/model.py` is deleted; the CPU tests keep reference copies.)
//!
//! A block step has an image of its own (`BlockLayout`). Its slot tables are launch-major across the drafter's
//! layers (launch i of layer l at `tables + (i * layers + l) * stride`), so a step of n launches uploads one
//! contiguous range; `spec_fix`'s patch entries address them that way.

// Index loops over rows and layers keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

use crate::qwen::{PAGE_TOKENS, SLOT_KEYS, sigma_inv};
use crate::tables::{Words, round, words};

/// query heads per KV pair (two KV heads of four)
pub const PAIR_HEADS: i64 = 8;
const SLOT_PAGES: usize = (SLOT_KEYS / PAGE_TOKENS) as usize;

/// What a drafter's tables depend on (`DrafterConfig` and the drafter's scratch pages).
#[derive(Clone, Debug)]
pub struct Drafter {
    pub layers: usize,
    pub pairs: usize,
    /// arena rows of one drafter layer's pages in a DRAFT block: K and V pages per pair
    pub layer_rows: i64,
    /// a full block: the anchor and every draft
    pub block: usize,
    pub window: i64,
    /// a causal block (DSpine) runs only the positions a verify width needs
    pub causal: bool,
    pub mask_token: i64,
    /// the first arena row of the scratch pages: (rows, layers, pairs, [K, V]) pages
    pub scratch_rows: i64,
    /// rows of the widest block step: `spec_fix` declares its tables at that step's size
    pub max_rows: usize,
}

impl Drafter {
    /// the block of a block step for a verify of `width` columns (`Drafter.block_cfg`)
    pub fn block_of(&self, width: usize) -> usize {
        if self.causal && width < self.block {
            width
        } else {
            self.block
        }
    }

    /// row `b`'s scratch page of `layer`, `pair` and K (`kv` 0) or V (1)
    pub fn scratch(&self, b: usize, layer: usize, pair: usize, kv: usize) -> i64 {
        self.scratch_rows
            + ((((b * self.layers + layer) * self.pairs + pair) * 2 + kv) as i64) * PAGE_TOKENS
    }

    /// slots of one launch of a block step of `rows` rows (`slots_per_launch`)
    pub fn slots_per_launch(&self, rows: usize) -> usize {
        (rows * self.pairs).max(16)
    }

    /// per row and pair at most window / SLOT_KEYS + 1 context chunks and the block
    pub fn max_launches(&self) -> usize {
        (self.window / SLOT_KEYS) as usize + 2
    }
}

/// One row of a drafting step: its anchor's position, and its DRAFT block rows, `first` being the logical
/// block of `blocks[0]`.
#[derive(Clone, Copy, Debug)]
pub struct DraftRow<'a> {
    pub position: i64,
    pub blocks: &'a [i64],
    pub first: i64,
}

// ------------------------------------------------------------------ injection
/// Offsets of an injection's tables in a verify image: the columns' positions (`tokens` of them), then each
/// drafter layer's `kv_rows` (tokens, 8), `kv_stride` apart.
#[derive(Clone, Copy, Debug, Default)]
pub struct InjectLayout {
    pub tokens: usize,
    pub pos: usize,
    pub kv_rows: usize,
    pub kv_stride: usize,
}

impl InjectLayout {
    /// the bytes the tables take
    pub fn sizes(drafter: &Drafter, tokens: usize) -> [usize; 2] {
        [4 * tokens, drafter.layers * round(4 * 8 * tokens)]
    }
}

/// One injected run: `count` columns from `column`, at positions from `position`, of `row`.
#[derive(Clone, Copy, Debug)]
pub struct Injected<'a> {
    pub column: usize,
    pub position: i64,
    pub count: usize,
    pub row: DraftRow<'a>,
}

/// Write an injection's positions and `kv_rows` (`Drafter.inject`, `inject_rows`) into `host`, which holds
/// `nbytes` bytes from the image's start.
pub fn fill_inject(
    drafter: &Drafter,
    layout: &InjectLayout,
    host: &mut [u8],
    nbytes: usize,
    runs: &[Injected<'_>],
) -> Result<(), String> {
    let mut w = words(host, nbytes)?;
    let (t, pairs) = (layout.tokens, drafter.pairs as i64);
    w.fill(layout.pos, t, 0);
    for layer in 0..drafter.layers {
        w.fill(layout.kv_rows + layer * layout.kv_stride, 8 * t, -1);
    }
    for run in runs {
        if run.count == 0 {
            continue;
        }
        if run.column + run.count > t {
            return Err(format!("injected columns past {t}"));
        }
        let last_page = (run.position + run.count as i64 - 1) / PAGE_TOKENS - run.row.first;
        if last_page >= run.row.blocks.len() as i64 {
            return Err("positions past the row's DRAFT blocks".into());
        }
        for j in 0..run.count {
            let pos = run.position + j as i64;
            let column = run.column + j;
            w.set(layout.pos, column, pos);
            // positions before the retained blocks are older than the window: nothing reads them
            let page = pos / PAGE_TOKENS - run.row.first;
            if page < 0 {
                continue;
            }
            let (base, r) = (run.row.blocks[page as usize], pos % PAGE_TOKENS);
            for layer in 0..drafter.layers {
                let lb = base + layer as i64 * drafter.layer_rows;
                let at = layout.kv_rows + layer * layout.kv_stride;
                for pg in 0..pairs {
                    w.set(at, 8 * column + pg as usize, lb + pg * PAGE_TOKENS + r);
                    w.set(
                        at,
                        8 * column + 4 + pg as usize,
                        lb + (pairs + pg) * PAGE_TOKENS + sigma_inv(r),
                    );
                }
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ the block step
/// Offsets of a block step's tables in its image, for `bucket` rows.
#[derive(Clone, Debug, Default)]
pub struct BlockLayout {
    pub bucket: usize,
    /// slots per launch
    pub slots: usize,
    pub tok: usize,
    pub pos: usize,
    pub anchors: usize,
    /// per drafter layer `kv_rows` of the block's own K / V, `block_stride` apart
    pub block_rows: usize,
    pub block_stride: usize,
    /// slot tables, launch-major: launch i of layer l at `tables + (i * layers + l) * stride`
    pub tables: usize,
    pub stride: usize,
    pub k_pages: usize,
    pub v_pages: usize,
    pub layers: usize,
    pub max_launches: usize,
    /// the image's size with every launch
    pub nbytes: usize,
}

impl BlockLayout {
    /// The layout of a block step of `bucket` rows. `spec_fix` declares the tokens, positions and anchors at the
    /// widest step's size (`max_rows` rows), so those tables take that much room in every layout, and so does
    /// the image's extent of slot tables.
    pub fn new(drafter: &Drafter, bucket: usize) -> Self {
        let tb = drafter.block * bucket;
        let widest = drafter.block * drafter.max_rows;
        let mut at = 0usize;
        let mut take = |nbytes: usize| {
            let offset = at;
            at += round(nbytes);
            offset
        };
        let tok = take(4 * widest);
        let pos = take(4 * widest);
        let anchors = take(4 * drafter.max_rows);
        let block_stride = round(4 * 8 * tb);
        let block_rows = take(drafter.layers * block_stride);
        let s = drafter.slots_per_launch(bucket);
        let (k_pages, v_pages) = (
            round(4 * 8 * s),
            round(4 * 8 * s) + round(4 * SLOT_PAGES * s),
        );
        let stride = v_pages + round(4 * SLOT_PAGES * s);
        let tables = at;
        let max_launches = drafter.max_launches();
        // spec_fix's slot tables tensor: every layer's launches at the widest step's stride
        let s_widest = drafter.slots_per_launch(drafter.max_rows);
        let widest_stride = round(4 * 8 * s_widest) + 2 * round(4 * SLOT_PAGES * s_widest);
        BlockLayout {
            bucket,
            slots: s,
            tok,
            pos,
            anchors,
            block_rows,
            block_stride,
            tables,
            stride,
            k_pages,
            v_pages,
            layers: drafter.layers,
            max_launches,
            nbytes: tables + max_launches * drafter.layers * stride.max(widest_stride),
        }
    }

    /// launch `i`'s tables of drafter layer `layer`, from the image's start
    pub fn table(&self, i: usize, layer: usize) -> usize {
        self.tables + (i * self.layers + layer) * self.stride
    }

    pub fn upload_bytes(&self, launches: usize) -> usize {
        self.tables + launches * self.layers * self.stride
    }
}

/// A filled block step: its slot launches per layer, the bytes to upload, and with device anchors
/// (`low`) the (row, kind) of each slot in order (kind 0 a context chunk, 1 the block).
#[derive(Clone, Debug, Default)]
pub struct BlockFilled {
    pub launches: usize,
    pub nbytes: usize,
    pub records: Vec<(usize, i32)>,
}

/// Write a block step's tokens, positions, anchors, scratch rows and slot tables (`Drafter._tables`) for a
/// verify of `width` columns: row b drafts after its anchor `anchors[b]` at `rows[b].position`. With `anchors`
/// None the device decides them (`spec_fix`): each row's anchor at a position in [low[b], position].
#[allow(clippy::too_many_arguments)]
pub fn fill_block(
    drafter: &Drafter,
    layout: &BlockLayout,
    host: &mut [u8],
    rows: &[DraftRow<'_>],
    anchors: Option<&[i64]>,
    low: Option<&[i64]>,
    width: usize,
    max_positions: i64,
) -> Result<BlockFilled, String> {
    let bucket = layout.bucket;
    let n = rows.len();
    if n == 0 || n > bucket {
        return Err(format!("{n} rows in a block step of {bucket}"));
    }
    if anchors.is_some_and(|a| a.len() != n) || low.is_some_and(|l| l.len() != n) {
        return Err("anchors and low take one entry a row".into());
    }
    let mut w = words(host, layout.nbytes)?;
    let block = drafter.block_of(width);
    let t = block * bucket;
    w.fill(layout.tok, t, drafter.mask_token as i32);
    w.fill(layout.pos, t, 0);
    w.fill(layout.anchors, bucket, 0);
    let mut top = 0i64;
    for (b, row) in rows.iter().enumerate() {
        if let Some(anchors) = anchors {
            w.set(layout.tok, b * block, anchors[b]);
            w.set(layout.anchors, b, anchors[b]);
        }
        for j in 0..block {
            w.set(layout.pos, b * block + j, row.position + j as i64);
        }
        top = top.max(row.position + block as i64 - 1);
    }
    if top + 1 > max_positions + block as i64 {
        return Err(format!("position exceeds {max_positions}"));
    }
    // the block's own K / V rows in the scratch pages
    let pairs = drafter.pairs;
    for layer in 0..drafter.layers {
        let at = layout.block_rows + layer * layout.block_stride;
        w.fill(at, 8 * t, -1);
        for b in 0..n {
            for j in 0..block {
                let column = b * block + j;
                for pg in 0..pairs {
                    w.set(
                        at,
                        8 * column + pg,
                        drafter.scratch(b, layer, pg, 0) + j as i64,
                    );
                    w.set(
                        at,
                        8 * column + 4 + pg,
                        drafter.scratch(b, layer, pg, 1) + sigma_inv(j as i64),
                    );
                }
            }
        }
    }
    // slots: per row and pair one per context chunk, then one for the block
    let s_count = layout.slots;
    let mut slot = 0usize;
    let mut records = Vec::new();
    let put = |w: &mut Words<'_>,
               slot: &mut usize,
               fields: [i64; 8],
               kp: &[i64; SLOT_PAGES],
               vp: &[i64; SLOT_PAGES]|
     -> Result<(), String> {
        let (i, s) = (*slot / s_count, *slot % s_count);
        if i >= layout.max_launches {
            return Err(format!(
                "more than {} block-step slot launches",
                layout.max_launches
            ));
        }
        for layer in 0..drafter.layers {
            let base = layout.table(i, layer);
            if s == 0 {
                w.fill(base, layout.stride / 4, 0);
            }
            for (f, &value) in fields.iter().enumerate() {
                w.set(base, 8 * s + f, value);
            }
            let shift = layer as i64 * drafter.layer_rows;
            for lane in 0..SLOT_PAGES {
                let (k, v) = (kp[lane], vp[lane]);
                w.set(
                    base + layout.k_pages,
                    SLOT_PAGES * s + lane,
                    if k >= 0 { k + shift } else { -1 },
                );
                w.set(
                    base + layout.v_pages,
                    SLOT_PAGES * s + lane,
                    if v >= 0 { v + shift } else { -1 },
                );
            }
        }
        *slot += 1;
        Ok(())
    };
    let window = drafter.window;
    for (b, row) in rows.iter().enumerate() {
        let p = row.position;
        let lo = low.map_or(p, |low| low[b]);
        let kmin = (lo - window + 1).max(0);
        let qcol0 = PAIR_HEADS * block as i64 * b as i64;
        if p > 0 {
            for c in kmin / SLOT_KEYS..=(p - 1) / SLOT_KEYS {
                let key0 = c * SLOT_KEYS;
                let first_page = kmin.max(key0) / PAGE_TOKENS;
                let last_page = (p - 1) / PAGE_TOKENS;
                let mut base = [-1i64; SLOT_PAGES];
                for lane in 0..SLOT_PAGES {
                    let page = key0 / PAGE_TOKENS + lane as i64;
                    if page >= first_page && page <= last_page {
                        let index = page - row.first;
                        if index < 0 || index as usize >= row.blocks.len() {
                            return Err("the window reaches outside the row's DRAFT blocks".into());
                        }
                        base[lane] = row.blocks[index as usize];
                    }
                }
                for pg in 0..pairs as i64 {
                    let kp = base.map(|x| if x >= 0 { x + pg * PAGE_TOKENS } else { -1 });
                    let vp = base.map(|x| {
                        if x >= 0 {
                            x + (pairs as i64 + pg) * PAGE_TOKENS
                        } else {
                            -1
                        }
                    });
                    put(
                        &mut w,
                        &mut slot,
                        [pg, qcol0, p, key0, p, 0, kmin, window],
                        &kp,
                        &vp,
                    )?;
                    records.push((b, 0));
                }
            }
        }
        for pg in 0..pairs {
            let (mut kp, mut vp) = ([-1i64; SLOT_PAGES], [-1i64; SLOT_PAGES]);
            kp[0] = drafter.scratch(b, 0, pg, 0);
            vp[0] = drafter.scratch(b, 0, pg, 1);
            let fields = [pg as i64, qcol0, p, p, p + block as i64, 0, p, window];
            put(&mut w, &mut slot, fields, &kp, &vp)?;
            records.push((b, 1));
        }
    }
    let launches = slot.div_ceil(s_count);
    while slot < launches * s_count {
        put(
            &mut w,
            &mut slot,
            [-1, 0, 0, 0, 0, 0, 0, 0],
            &[-1; SLOT_PAGES],
            &[-1; SLOT_PAGES],
        )?;
    }
    Ok(BlockFilled {
        launches,
        nbytes: layout.upload_bytes(launches),
        records: if low.is_some() { records } else { Vec::new() },
    })
}

/// `spec_fix`'s patch entries (`Drafter.prepare_device`): per drafter layer and recorded slot, [its record's
/// offset in the block image's slot tables (i32 units), row, kind, 0].
pub fn patch_entries(layout: &BlockLayout, records: &[(usize, i32)]) -> Vec<[i32; 4]> {
    let mut out = Vec::with_capacity(layout.layers * records.len());
    for layer in 0..layout.layers {
        for (k, &(row, kind)) in records.iter().enumerate() {
            let (i, s) = (k / layout.slots, k % layout.slots);
            let offset = (layout.table(i, layer) - layout.tables) / 4 + 8 * s;
            out.push([offset as i32, row as i32, kind, 0]);
        }
    }
    out
}

// ------------------------------------------------------------------ spec_fix
/// Offsets of a verify step's `spec_fix` and commit tables in its image (`Qwen35Gaudi._fix`), for `bucket`
/// rows of `layers` GDN layers and room for `patches` patch entries.
#[derive(Clone, Debug, Default)]
pub struct FixLayout {
    pub bucket: usize,
    pub layers: usize,
    pub patches: usize,
    pub cb0: usize,
    pub cb1: usize,
    pub rows: usize,
    pub commit_bases: usize,
    pub commit_counts: usize,
    pub hidden_table: usize,
    pub patch: usize,
    pub meta: usize,
}

impl FixLayout {
    /// the bytes of each table, in the order of the fields
    pub fn sizes(bucket: usize, layers: usize, patches: usize) -> [usize; 8] {
        let bl = bucket * layers;
        [
            round(16 * bl),
            round(16 * bl),
            round(32 * bucket),
            round(16 * bl),
            round(4 * bl),
            round(8 * bucket),
            round(16 * patches),
            round(16),
        ]
    }

    /// the layout from the tables' offsets (`sizes` order)
    pub fn new(bucket: usize, layers: usize, patches: usize, offsets: &[usize]) -> Self {
        FixLayout {
            bucket,
            layers,
            patches,
            cb0: offsets[0],
            cb1: offsets[1],
            rows: offsets[2],
            commit_bases: offsets[3],
            commit_counts: offsets[4],
            hidden_table: offsets[5],
            patch: offsets[6],
            meta: offsets[7],
        }
    }
}

/// A verify row's state: its first position, its STATE source (-1: fresh) and its STATE block rows, `first`
/// being the logical block of `blocks[0]` (`RowCache.state_at`).
#[derive(Clone, Copy, Debug)]
pub struct StateRow<'a> {
    pub pos0: i64,
    pub src: i64,
    pub blocks: &'a [i64],
    pub first: i64,
}

impl StateRow<'_> {
    pub fn state_at(&self, position: i64, state_tokens: i64) -> Result<i64, String> {
        let index = position / state_tokens - self.first;
        if index < 0 || index as usize >= self.blocks.len() {
            return Err(format!("no STATE block for position {position}"));
        }
        Ok(self.blocks[index as usize])
    }
}

/// What the `spec_fix` tables need of the model's STATE geometry.
#[derive(Clone, Copy, Debug)]
pub struct StateGeometry {
    pub state_layer_rows: i64,
    pub ssm_rows: i64,
    pub state_tokens: i64,
    /// the hidden slot's row in a STATE block (after every GDN layer's state)
    pub hidden_offset: i64,
}

/// Write a verify step's `spec_fix` tables (`Qwen35Gaudi._fix`): the commit's state fields for the two STATE
/// blocks the accepted count may end in, each row's [pos0, 1, switch, hidden rows], the commit tables reset
/// (padding rows commit nothing), the block step's patch entries and [entries, block, window, 0].
#[allow(clippy::too_many_arguments)]
pub fn fill_fix(
    layout: &FixLayout,
    geometry: &StateGeometry,
    host: &mut [u8],
    nbytes: usize,
    rows: &[StateRow<'_>],
    width: usize,
    patch: &[[i32; 4]],
    block: usize,
    window: i64,
) -> Result<(), String> {
    let (bucket, layers) = (layout.bucket, layout.layers);
    if rows.is_empty() || rows.len() > bucket {
        return Err(format!("{} rows in a verify step of {bucket}", rows.len()));
    }
    if patch.len() > layout.patches {
        return Err(format!(
            "{} slot patches exceed {}",
            patch.len(),
            layout.patches
        ));
    }
    let mut w = words(host, nbytes)?;
    let g = geometry;
    let field = |base: i64, layer: usize, conv: bool| -> i64 {
        if base < 0 {
            -1
        } else {
            base + layer as i64 * g.state_layer_rows + if conv { g.ssm_rows } else { 0 }
        }
    };
    let hidden = |block: i64| {
        if block < 0 {
            -1
        } else {
            block + g.hidden_offset
        }
    };
    w.fill(layout.cb0, 4 * bucket * layers, -1);
    w.fill(layout.cb1, 4 * bucket * layers, -1);
    w.fill(layout.rows, 8 * bucket, 0);
    for (b, row) in rows.iter().enumerate() {
        let first = row.state_at(row.pos0, g.state_tokens)?;
        let last = row.state_at(row.pos0 + width as i64 - 1, g.state_tokens)?;
        for layer in 0..layers {
            for (table, dst) in [(layout.cb0, first), (layout.cb1, last)] {
                let fields = [
                    field(row.src, layer, true),
                    field(row.src, layer, false),
                    field(dst, layer, true),
                    field(dst, layer, false),
                ];
                for (k, &value) in fields.iter().enumerate() {
                    w.set(table, 4 * (layer * bucket + b) + k, value);
                }
            }
        }
        let mut switch = width as i64;
        for m in 0..width as i64 {
            if row.state_at(row.pos0 + m, g.state_tokens)? != first {
                switch = m;
                break;
            }
        }
        for (k, &value) in [row.pos0, 1, switch, hidden(first), hidden(last)]
            .iter()
            .enumerate()
        {
            w.set(layout.rows, 8 * b + k, value);
        }
    }
    w.fill(layout.commit_bases, 4 * bucket * layers, -1);
    w.fill(layout.commit_counts, bucket * layers, 0);
    w.fill(layout.hidden_table, 2 * bucket, -1);
    w.fill(layout.patch, 4 * layout.patches, 0);
    for (k, entry) in patch.iter().enumerate() {
        for (f, &value) in entry.iter().enumerate() {
            w.set(layout.patch, 4 * k + f, i64::from(value));
        }
    }
    for (k, &value) in [patch.len() as i64, block as i64, window, 0]
        .iter()
        .enumerate()
    {
        w.set(layout.meta, k, value);
    }
    Ok(())
}

// ------------------------------------------------------------------ verify widths
/// P(a step accepts at least j of its drafts | the row's last step), j = 0 .. 15, by class of the last step
/// (`qwen3_5.model.SURVIVAL`).
#[rustfmt::skip]
pub const SURVIVAL: [[f64; 16]; 10] = [
    [1.000, 0.695, 0.480, 0.329, 0.222, 0.151, 0.103, 0.065, 0.048, 0.036, 0.032, 0.028, 0.022, 0.017, 0.012, 0.012],
    [1.000, 0.710, 0.487, 0.339, 0.216, 0.144, 0.090, 0.063, 0.041, 0.031, 0.023, 0.018, 0.012, 0.007, 0.006, 0.003],
    [0.999, 0.747, 0.535, 0.346, 0.219, 0.151, 0.093, 0.069, 0.048, 0.035, 0.025, 0.015, 0.015, 0.010, 0.009, 0.008],
    [0.999, 0.735, 0.511, 0.373, 0.262, 0.197, 0.130, 0.105, 0.082, 0.064, 0.049, 0.031, 0.029, 0.024, 0.024, 0.014],
    [0.999, 0.735, 0.535, 0.387, 0.293, 0.226, 0.157, 0.114, 0.090, 0.060, 0.042, 0.023, 0.018, 0.013, 0.013, 0.013],
    [0.999, 0.802, 0.618, 0.490, 0.344, 0.242, 0.173, 0.132, 0.121, 0.109, 0.104, 0.083, 0.070, 0.055, 0.052, 0.050],
    [0.998, 0.852, 0.731, 0.593, 0.520, 0.439, 0.407, 0.346, 0.302, 0.277, 0.253, 0.229, 0.217, 0.168, 0.144, 0.128],
    [0.998, 0.983, 0.938, 0.879, 0.865, 0.820, 0.736, 0.682, 0.667, 0.653, 0.643, 0.628, 0.589, 0.549, 0.534, 0.515],
    [1.000, 0.785, 0.604, 0.472, 0.373, 0.301, 0.235, 0.195, 0.172, 0.150, 0.135, 0.116, 0.107, 0.092, 0.087, 0.080],
    [0.999, 0.876, 0.756, 0.643, 0.571, 0.497, 0.440, 0.388, 0.363, 0.347, 0.334, 0.315, 0.297, 0.257, 0.241, 0.227],
];
const SURVIVAL_CLASS: [usize; 16] = [0, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7];

/// A speculative step's time (ms) with the DSpine drafter, `bucket` rows of `width` columns (`step_ms`).
pub fn step_ms(width: usize, bucket: usize) -> f64 {
    let per_column = (0.71 - 0.026 * (bucket as f64 - 1.0)).max(0.4);
    22.3 + 1.44 * (bucket as f64 - 1.0) + per_column * (bucket * width) as f64
}

/// A plain decode step's time (ms), `bucket` rows (`plain_ms`).
pub fn plain_ms(bucket: usize) -> f64 {
    16.0 + 1.07 * bucket as f64
}

/// A speculative step's time (ms) with the MTP layer (`mtp_step_ms`).
pub fn mtp_step_ms(width: usize, bucket: usize) -> f64 {
    18.0 + 0.92 * (bucket * width) as f64
}

/// The next verify step's columns per row (`choose_width`): the width of the most expected tokens per ms, each
/// row's tokens expected from its accepted drafts of this step (of `width` - 1 verified) through SURVIVAL.
pub fn choose_width(widths: &[usize], accepted: &[i64], width: usize, bucket: usize) -> usize {
    if widths.len() == 1 {
        return widths[0];
    }
    let classes: Vec<usize> = accepted
        .iter()
        .map(|&a| {
            if a >= width as i64 - 1 && width < 16 {
                if width <= 4 { 8 } else { 9 }
            } else {
                SURVIVAL_CLASS[(a.max(0) as usize).min(15)]
            }
        })
        .collect();
    let mut sums = [[0f64; 16]; 10];
    for (c, row) in SURVIVAL.iter().enumerate() {
        let mut total = 0.0;
        for (j, &p) in row.iter().enumerate() {
            total += p;
            sums[c][j] = total;
        }
    }
    let (mut best, mut choice) = (-1.0f64, *widths.last().expect("widths"));
    for &w in widths {
        let mut tokens = 0.0;
        for &c in &classes {
            tokens += 1.0 + sums[c][(w - 1).min(15)] - SURVIVAL[c][0];
        }
        let rate = tokens / step_ms(w, bucket);
        if rate > best {
            best = rate;
            choice = w;
        }
    }
    choice
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_follow_the_survival_table() {
        // qwen3_5.model.choose_width((4, 8, 16), ...) of the same accepted counts
        assert_eq!(choose_width(&[16], &[0], 16, 1), 16);
        let all = [15];
        let w = choose_width(&[4, 8, 16], &all, 16, 1);
        assert!([4, 8, 16].contains(&w));
    }
}
