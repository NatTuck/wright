//! DeepSeek-V4.1's step tables, written into one host copy of the device table buffer: the port of `_fill_positions`
//! and `_fill_tokens` (`resin_models_gaudi/deepseek_v41/model.py`) for decode and prefill shapes, byte for byte
//! (the numpy code is the reference while it exists).

// Index loops over columns and rows keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

pub mod context;

use crate::tables::{Words, words};

pub const PAGE_TOKENS: i64 = 128;
pub const HISTORY_TOKENS: i64 = 256;
pub const PAGE_ROWS: i64 = 80; // fp4 compressed latent page rows (window pages stay 129, via layer_rows)
pub const INDEX_ROWS: i64 = 17; // fp4 index page rows
pub const PAIR_ROWS: i64 = PAGE_ROWS + INDEX_ROWS;
pub const INDEX_KEYS: i64 = 2048;
/// the rows of a merge group's running candidates (the top index entries; `INDEX_TOP` of the Python side)
pub const INDEX_RUN: i64 = 512;
/// pending-state rows per (WINDOW block, ratio-2 owner): slot (position / 2) % PENDING_SLOTS (cache.PENDING_SLOTS)
pub const PENDING_SLOTS: i64 = 4;
const ROPE: usize = 128;

/// Env-gated (`RESIN_DS41_FILL_PROF=1`) per-section fill timers (stderr, once per fill). Off: one branch.
struct FillProf {
    on: bool,
    last: std::time::Instant,
    acc: Vec<(&'static str, u64)>,
}
impl FillProf {
    fn new() -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("RESIN_DS41_FILL_PROF").map(|v| v == "1").unwrap_or(false));
        Self { on, last: std::time::Instant::now(), acc: Vec::new() }
    }
    fn mark(&mut self, name: &'static str) {
        if self.on {
            let now = std::time::Instant::now();
            self.acc.push((name, (now - self.last).as_nanos() as u64));
            self.last = now;
        }
    }
    fn report(&self, t: usize, prefill: bool) {
        if !self.on { return; }
        let mut order: Vec<&'static str> = Vec::new();
        let mut sums: Vec<u64> = Vec::new();
        for &(name, ns) in &self.acc {
            match order.iter().position(|&n| n == name) {
                Some(i) => sums[i] += ns,
                None => { order.push(name); sums.push(ns); }
            }
        }
        let total: u64 = sums.iter().sum();
        let parts: Vec<String> = order.iter().zip(&sums).map(|(n, ns)| format!("{n}={:.1}", *ns as f64 / 1e6)).collect();
        eprintln!("FILL_PROF t={t} prefill={prefill} total={:.1}ms {}", total as f64 / 1e6, parts.join(" "));
    }
}

/// A layer that owns a compressed cache: its ratio, its pages per HISTORY block and its row offset in a block.
#[derive(Clone, Debug)]
pub struct Owner {
    pub ratio: i64,
    pub pages: i64,
    pub offset: i64,
}

/// The host side of the Engram hash: compressed ids and the per-layer multipliers (layers x n).
#[derive(Clone, Debug)]
pub struct Engram {
    pub token_map: Vec<i64>,
    pub pad_id: i64,
    pub multipliers: Vec<i64>,
    pub n: usize,
    pub heads: usize,
    pub layers: usize,
    /// (layers, n - 1, heads): the moduli of each layer's hash columns
    pub primes: Vec<i64>,
    /// (layers, (n - 1) heads): each column's first row in its layer's table
    pub offsets: Vec<i64>,
}

/// One Engram layer's host-resident lookup table, in checkpoint format: `weight` is `rows x 256`
/// E4M3 codes and `scale` is `rows x 8` E8M0 bytes (one scale per 32 codes). Borrowed from the mmap
/// numpy views the fill holds for the step; never owned by the (cloned) `Model`, which would copy
/// the 188.8 GiB table. An empty slice means card mode (the device gathers from its own shard).
#[derive(Clone, Copy)]
pub struct HostTable<'a> {
    pub weight: &'a [u8],
    pub scale: &'a [u8],
}

/// The OCP E4M3 (bias 7) value of a code, as f64, exactly as the host oracle (`weights.py:OCP`):
/// `0x7F` / `0xFF` read as 0; the magnitudes reach 448 on the host (the device's 240 saturation is a
/// separate contract — the gather reproduces the host oracle, not the device decode).
fn ocp_e4m3(code: u8) -> f64 {
    let e = i32::from((code >> 3) & 15);
    let m = f64::from(code & 7);
    let mut mag = if e > 0 {
        (8.0 + m) * 2f64.powi(e - 10)
    } else {
        m * 2f64.powi(-9)
    };
    if (code & 0x7F) == 0x7F {
        mag = 0.0;
    }
    if code & 0x80 != 0 { -mag } else { mag }
}

/// `2^(s - 127)` formed in f32, exactly as the host oracle forms it
/// (`np.exp2(np.arange(256, f32) - 127)`): exact powers of two, the subnormal `2^-127` at s=0, and
/// overflow to inf at s=255. One documented deviation: numpy's f32 `exp2` ufunc rounds `2^127`
/// (s=254) UP by one ulp to `0x7F000001` (libc `exp2f` and the exact power give `0x7F000000`); the
/// oracle uses `np.exp2`, so the LUT reproduces that to be bit-identical at all 65536 pairs. Scale
/// 254 never occurs in a valid checkpoint (its ~2^127 block magnitude is rejected at load).
fn engram_scale_f32(s: usize) -> f32 {
    let e = s as i32 - 127;
    if e == 128 {
        f32::INFINITY
    } else if e == 127 {
        f32::from_bits(0x7F00_0001) // numpy exp2(f32(127)) = 2^127 + 1 ulp
    } else if e == -127 {
        f32::from_bits(0x0040_0000) // 2^-127, the smallest normal's half (subnormal)
    } else {
        f32::from_bits(((e + 127) as u32) << 23)
    }
}

fn build_engram_lut() -> [[f32; 256]; 256] {
    let mut ocp = [0f64; 256];
    for (c, value) in ocp.iter_mut().enumerate() {
        *value = ocp_e4m3(c as u8);
    }
    let mut lut = [[0f32; 256]; 256];
    for (s, row) in lut.iter_mut().enumerate() {
        // OCP (f64) times the f32 exponent promoted to f64, rounded once to f32 — the oracle's
        // `(OCP[None, :] * exp2_f32[:, None]).astype(f32)`.
        let scale = f64::from(engram_scale_f32(s));
        for (c, value) in row.iter_mut().enumerate() {
            *value = (ocp[c] * scale) as f32;
        }
    }
    lut
}

/// `LUT[scale][code] = OCP_E4M3[code] * 2^(scale - 127)` (f32), built once. Bit-identical to the host
/// gather oracle at all 65536 (scale, code) pairs; f32 output scope (scale 255 overflows to inf/NaN,
/// which a valid checkpoint never carries — E8M0 255 is NaN, rejected at load).
pub fn engram_lut() -> &'static [[f32; 256]; 256] {
    static LUT: std::sync::OnceLock<[[f32; 256]; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(build_engram_lut)
}

/// What every step's tables depend on besides the step itself.
#[derive(Clone, Debug)]
pub struct Model {
    pub heads: i64,
    pub index_heads: i64,
    pub sliding_window: i64,
    pub layer_rows: Vec<i64>,
    pub owners: Vec<Owner>,
    /// indices into `owners` of the ratio-2 owners (with pending tokens), in that order
    pub pending: Vec<usize>,
    /// HISTORY table width of a column (latent pages of a ratio-1 owner)
    pub pages: usize,
    pub kmax: i64,
    pub kc: i64,
    /// (ratio of the candidate source layer, candidate_topk_blocks), if there is one
    pub candidate: Option<(i64, i64)>,
    pub engram: Option<Engram>,
    /// decode steps whose columns see more index keys select by radix passes (`RADIX_MIN`)
    pub radix_min: i64,
    /// a prefill step whose every column sees more entries than `prefilter_min` (and 8 x the blocks) selects from
    /// its `prefilter_blocks` best blocks of 8 (`PREFILTER_BLOCKS`, `PREFILTER_MIN`; 0 blocks: never)
    pub prefilter_blocks: i64,
    pub prefilter_min: i64,
    /// a prefill step whose columns see at most this many compressed entries attends densely (`DENSE_KEYS`)
    pub dense_keys: i64,
}

/// Byte offsets of one shape's tables in a host copy.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    pub t: usize,
    pub prefill: bool,
    /// columns per window and index slot (`Shape.qb`)
    pub qb: usize,
    pub win_slots: usize,
    /// window pages per slot (`Shape.win_keys` / PAGE_TOKENS)
    pub win_pages: usize,
    /// ratio-2 groups a step can complete (`Shape.groups2`)
    pub groups2: usize,
    pub ix_slots: usize,
    /// tiles of `INDEX_KEYS` index keys per merge group, launches per group, the group's score rows, and the
    /// groups a step may need per ratio
    pub ix_tp: usize,
    pub ix_lpg: usize,
    /// tiles per bucketed index-chain recipe (the plain/candidate arms take ceil(chunks / ix_chain_g) launches)
    pub ix_chain_g: usize,
    pub ix_rows: usize,
    /// per ratio (2, T): per query block its first column's position and the entries its last column sees. The
    /// merge, the block kernels and `ds_ix_gslot` derive every per-group value from this and `ix_vis` on the
    /// device. One table PER RATIO: ratio-1 and ratio-2 `seen` differ, and each ratio's gslot launch reads its
    /// own (a shared table left ratio-1 records with ratio-2 seen/klen, corrupting ratio-1 top-512 at vis>512).
    pub blk: [usize; 2],
    /// per ratio: every merge group's slot records (8, groups * lpg * ix_slots) and the groups' first keys
    /// (groups,), both written on the device by `ds_ix_gslot`
    pub ix_rec: [usize; 2],
    pub ix_gb: [usize; 2],
    pub nbytes: usize,
    /// byte ranges zeroed before a fill (all but the tables written whole)
    pub zero: Vec<(usize, usize)>,
    pub tok: usize,
    pub meta: usize,
    pub cs: [usize; 2],
    pub cs_inv: [usize; 2],
    pub cs_g2: usize,
    pub cs_g1: usize,
    /// per owner
    pub cmp_store: Vec<usize>,
    /// per pending owner
    pub cmp_groups: Vec<usize>,
    pub cmp_rows: Vec<usize>,
    pub slots: usize,
    /// per layer (and DSpark stage)
    pub rows: Vec<usize>,
    pub lat: Vec<usize>,
    pub ix_visb: usize,
    /// the padded block counts of the ratio that is not the candidate source's (`_visb`)
    pub ix_visb2: usize,
    pub ix_visc: usize,
    /// the sparse arm's flat per-batch slot records (relative columns within each query-block batch)
    pub ixsc_rec: usize,
    pub sparse_batch: usize,
    pub ixsc_batches: usize,
    pub ixc_per_batch: usize,
    pub ix_vis: [usize; 2],
    pub cmp_blocks: [usize; 2],
    /// per Engram layer of a step that hashes on the device: (hi table, lo table, its index among the
    /// checkpoint's Engram layers)
    pub engram: Vec<(usize, usize, usize)>,
    /// per Engram layer of a step whose hash ids come from the host: (ids table, its Engram index)
    pub engram_ids: Vec<(usize, usize)>,
    /// per Engram layer of a step that gathers on the host: (f32 embedding table, its Engram index).
    /// The host reads the row from its mmap table, dequantizes it, and writes `eg_emb[256*j+i, c]`
    /// here; the device's `engram_graph` reads the embedding directly (no gather/hash kernel).
    pub engram_emb: Vec<(usize, usize)>,
    /// a prefill shape with the dense compressed attention: per ratio its slots, and per owner its latent pages
    pub cslots: Option<[usize; 2]>,
    pub cmp_lat: Vec<usize>,
    /// a decode shape with DSpark: its draft step's (block indices, block slots, anchors)
    pub draft: Option<(usize, usize, usize)>,
}

/// One step's rows (sequences), flat: per row its first column, first position and token count, its WINDOW
/// block rows and ids from logical block `window_first`, its HISTORY block rows from block 0, and its tokens
/// before pos0 (`*_indptr` delimit each row's part).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub column: &'a [i64],
    pub pos0: &'a [i64],
    pub real: &'a [i64],
    pub window_first: &'a [i64],
    pub window_indptr: &'a [i64],
    pub window: &'a [i64],
    pub window_ids: &'a [i64],
    pub history_indptr: &'a [i64],
    pub history: &'a [i64],
    pub before_indptr: &'a [i64],
    pub before: &'a [i64],
}

impl Rows<'_> {
    fn len(&self) -> usize {
        self.column.len()
    }

    fn part<'b>(values: &'b [i64], indptr: &[i64], row: usize) -> &'b [i64] {
        &values[indptr[row] as usize..indptr[row + 1] as usize]
    }

    fn validate(&self) -> Result<(), String> {
        let rows = self.len();
        if rows == 0 {
            return Err("a step needs at least one row".into());
        }
        for (name, len) in [
            ("pos0", self.pos0.len()),
            ("real", self.real.len()),
            ("window_first", self.window_first.len()),
        ] {
            if len != rows {
                return Err(format!("{name} has {len} entries for {rows} rows"));
            }
        }
        for (name, indptr, values) in [
            ("window", self.window_indptr, self.window.len()),
            ("history", self.history_indptr, self.history.len()),
            ("before", self.before_indptr, self.before.len()),
        ] {
            if indptr.len() != rows + 1
                || indptr[0] != 0
                || indptr.windows(2).any(|pair| pair[0] > pair[1])
                || indptr[rows] as usize != values
            {
                return Err(format!("{name}_indptr does not delimit {rows} rows"));
            }
        }
        if self.window_ids.len() != self.window.len() {
            return Err("window_ids and window differ in length".into());
        }
        Ok(())
    }
}

/// Which op variants a step needs (`counts` in the Python fill).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub sparse: bool,
    pub cand: bool,
    pub ixc: usize,
    pub radix: [bool; 2],
    /// per ratio: the ops a step's plain and candidate index arms take (their groups' launches and merges)
    pub ix: [usize; 2],
    pub pf: [usize; 2],
    /// per ratio: a prefill step's dense attention
    pub dense: [bool; 2],
}

/// NumPy indexing of a padded row: negative indices count from the end of the padded width; past a row's own
/// values, the padding.
fn padded(values: &[i64], width: usize, index: i64, pad: i64) -> Result<i64, String> {
    let index = if index < 0 {
        index + width as i64
    } else {
        index
    };
    if index < 0 || index as usize >= width {
        return Err(format!("index {index} is outside a row of {width}"));
    }
    Ok(values.get(index as usize).copied().unwrap_or(pad))
}

/// NumPy indexing of a row's own values (negative indices count from its end).
fn own(values: &[i64], index: i64) -> Result<i64, String> {
    let len = values.len() as i64;
    let at = if index < 0 { index + len } else { index };
    if at < 0 || at >= len {
        return Err(format!("index {index} is outside a row of {len}"));
    }
    Ok(values[at as usize])
}

fn rope(table: &[f32], position: i64) -> Result<&[f32], String> {
    let start = usize::try_from(position)
        .ok()
        .and_then(|row| row.checked_mul(ROPE))
        .filter(|&start| start + ROPE <= table.len())
        .ok_or_else(|| format!("position {position} has no rotation table row"))?;
    Ok(&table[start..start + ROPE])
}

/// Write a step's tables into `host` (a host copy of at least `tables.nbytes` bytes, 4-byte aligned).
/// `cs_all[r]` holds every position's rotation table for rope config r (positions x 128). Without `with_tokens`
/// this is `_fill_positions` (all but the tokens and their Engram rows); with it, `_fill`. `tokens` (None: the
/// device's) are the step's tokens, needed with them for rows of several columns. `floor`: no window reaches
/// before that position (`_stage`'s floor, slot lane 6).
#[allow(clippy::too_many_arguments)]
pub fn fill(
    model: &Model,
    tables: &Tables,
    host: &mut [u8],
    rows: &Rows<'_>,
    positions: &[i64],
    tokens: Option<&[i64]>,
    with_tokens: bool,
    floor: Option<i64>,
    cs_all: [&[f32]; 2],
    host_tables: &[HostTable],
) -> Result<Counts, String> {
    rows.validate()?;
    let t = tables.t;
    let n = positions.len();
    if n == 0 || n > t {
        return Err(format!("{n} positions for a step of {t} columns"));
    }
    if tables.prefill && rows.len() != 1 {
        return Err("a prefill step runs one row".into());
    }
    if host.len() < tables.nbytes || !tables.nbytes.is_multiple_of(4) {
        return Err("the host copy is too small or misaligned".into());
    }
    let mut prof = FillProf::new();
    for &(lo, hi) in &tables.zero {
        host[lo..hi].fill(0);
    }
    prof.mark("zero");
    let mut w = words(host, tables.nbytes)?;
    let qb = tables.qb.max(1);
    let r_count = rows.len();
    let page = PAGE_TOKENS;

    w.set(tables.meta, 0, positions[0]);
    w.set(tables.meta, 1, n as i64);
    w.set(tables.meta, 2, 0);
    w.set(tables.meta, 3, 0);
    let mut pos = vec![0i64; t];
    pos[..n].copy_from_slice(positions);
    for r in 0..2 {
        for (column, &position) in pos.iter().enumerate() {
            let cs = rope(cs_all[r], position)?;
            w.set_many_f32(tables.cs[r], column * ROPE, cs);
            // cs_inv is cs with its last 64 lanes negated: copy the row, then flip the tail's signs
            w.set_many_f32(tables.cs_inv[r], column * ROPE, cs);
            let inv_start = tables.cs_inv[r] / 4 + column * ROPE;
            for d in w.0[inv_start + 64..inv_start + ROPE].iter_mut() {
                *d = (-f32::from_bits(*d as u32)).to_bits() as i32;
            }
        }
    }

    prof.mark("rope");
    // each row's blocks: WINDOW rows and ids from logical block window_first (padded with 0), HISTORY rows from
    // block 0 (padded with -1 to at least the model's page count)
    let window_width = (0..r_count)
        .map(|k| Rows::part(rows.window, rows.window_indptr, k).len())
        .max()
        .unwrap_or(0);
    let history_width = (0..r_count)
        .map(|k| Rows::part(rows.history, rows.history_indptr, k).len())
        .max()
        .unwrap_or(0)
        .max(model.pages);
    let window = |k: usize| Rows::part(rows.window, rows.window_indptr, k);
    let window_ids = |k: usize| Rows::part(rows.window_ids, rows.window_indptr, k);
    let history = |k: usize| Rows::part(rows.history, rows.history_indptr, k);
    let mut prev_id = vec![0i64; r_count];
    let mut col_row = vec![-1i64; t];
    let mut slot_rows: Vec<(usize, i64)> = Vec::new();
    for k in 0..r_count {
        let (column, pos0, real) = (rows.column[k], rows.pos0[k], rows.real[k]);
        if real < 1 || column < 0 || (column + real) as usize > t {
            return Err(format!(
                "row {k} spans columns {column}..{} of {t}",
                column + real
            ));
        }
        if pos0 > 0 {
            prev_id[k] = own(window_ids(k), (pos0 - 1) / page - rows.window_first[k])?;
        }
        for c in column..column + real {
            col_row[c as usize] = k as i64;
        }
        for b0 in (0..real).step_by(qb) {
            slot_rows.push((k, b0));
        }
    }
    let cols: Vec<usize> = (0..t).filter(|&c| col_row[c] >= 0).collect();
    let live = |c: usize| col_row[c] >= 0;
    let mut win = vec![0i64; t];
    let mut win_id = vec![0i64; t];
    for &c in &cols {
        let k = col_row[c] as usize;
        let block = pos[c] / page - rows.window_first[k];
        win[c] = padded(window(k), window_width, block, 0)?;
        win_id[c] = padded(window_ids(k), window_width, block, 0)?;
    }
    let last: Vec<i64> = (0..r_count)
        .map(|k| rows.pos0[k] + rows.real[k] - 1)
        .collect();

    prof.mark("blocks_setup");
    // window slots, and every layer's window pages and store rows
    let sliding = model.sliding_window;
    let (win_slots, pages) = (tables.win_slots, tables.win_pages);
    if slot_rows.len() > win_slots {
        return Err(format!(
            "{} window slots for a step of {win_slots}",
            slot_rows.len()
        ));
    }
    for s in 0..win_slots {
        w.set(tables.slots, s * 8, -1);
    }
    let mut lat0 = vec![-1i64; win_slots * pages];
    // per slot: its row, first query position and first column
    let mut slot_q0 = Vec::with_capacity(slot_rows.len());
    for (s, &(k, b0)) in slot_rows.iter().enumerate() {
        let q0 = rows.pos0[k] + b0;
        let key0 = (q0 - (sliding - 1)).max(0) / page * page;
        let col0 = rows.column[k] + b0;
        slot_q0.push((k, q0, col0));
        let values = [
            0,
            model.heads * col0,
            q0,
            key0,
            last[k] + 1,
            0,
            floor.unwrap_or(0),
            sliding,
        ];
        for (lane, value) in values.into_iter().enumerate() {
            w.set(tables.slots, s * 8 + lane, value);
        }
        for p in 0..pages {
            let pg = key0 / page + p as i64;
            if pg <= last[k] / page {
                lat0[s * pages + p] =
                    padded(window(k), window_width, pg - rows.window_first[k], 0)?;
            }
        }
    }
    let mut lat_buf: Vec<i32> = Vec::with_capacity(lat0.len());
    prof.mark("slots_lat0");
    for (&lat, &offset) in tables.lat.iter().zip(&model.layer_rows) {
        lat_buf.clear();
        lat_buf.extend(lat0.iter().map(|&v| if v >= 0 { (v + offset) as i32 } else { -1 }));
        w.copy_words(lat, 0, &lat_buf);
    }
    prof.mark("lat");
    for (&table, &offset) in tables.rows.iter().zip(&model.layer_rows) {
        w.fill(table, t * 8, -1);
        for &c in &cols {
            let slot = pos[c] % page;
            w.set(table, c * 8, win[c] + slot + offset);
            w.set(table, c * 8 + 3, win[c] + page + offset);
            w.set(table, c * 8 + 4, slot);
        }
    }

    prof.mark("rows");
    // compressor: ratio-2 groups (completed at odd positions) and pending tokens, ratio-1 entries
    let odd: Vec<usize> = cols.iter().copied().filter(|&c| pos[c] % 2 == 1).collect();
    let groups2 = tables.groups2;
    if odd.len() > groups2 {
        return Err(format!(
            "{} ratio-2 groups for a step of {groups2}",
            odd.len()
        ));
    }
    let mut entries2 = vec![-1i64; groups2];
    for (g, &c) in odd.iter().enumerate() {
        entries2[g] = pos[c] / 2;
    }
    if !model.pending.is_empty() {
        for (g, &entry) in entries2.iter().enumerate() {
            let cs = rope(cs_all[1], if entry >= 0 { 2 * entry } else { 0 })?;
            for (lane, &value) in cs.iter().enumerate() {
                w.set_f32(tables.cs_g2, g * ROPE + lane, value);
            }
        }
    }
    let npend = model.pending.len() as i64;
    // the pending-state row of owner i for the token at `position` in WINDOW block `block`
    let pending_row = |block: i64, i: usize, position: i64| {
        (block * npend + i as i64) * PENDING_SLOTS + position / 2 % PENDING_SLOTS
    };
    for (i, (&groups, &pend)) in tables.cmp_groups.iter().zip(&tables.cmp_rows).enumerate() {
        for (g, &c) in odd.iter().enumerate() {
            let k = col_row[c] as usize;
            let inside = pos[c] > rows.pos0[k];
            w.set(groups, g * 4, if inside { c as i64 - 1 } else { -1 });
            w.set(groups, g * 4 + 1, c as i64);
            w.set(
                groups,
                g * 4 + 2,
                if inside {
                    0
                } else {
                    pending_row(prev_id[k], i, pos[c] - 1)
                },
            );
        }
        // the columns whose pending state is stored: a prefill row's last column when its position is even (the
        // next chunk or step completes the group); every even column of a decode or verify row, since a
        // rejected draft can end the committed range at any of them, each in the slot of its position
        w.fill(pend, t, -1);
        for &c in &cols {
            let k = col_row[c] as usize;
            let stored = if tables.prefill {
                c as i64 == rows.column[k] + rows.real[k] - 1 && last[k] % 2 == 0
            } else {
                pos[c] % 2 == 0
            };
            if stored {
                w.set(pend, c, pending_row(win_id[c], i, pos[c]));
            }
        }
    }
    if model.pending.len() < model.owners.len() {
        for (c, &position) in pos.iter().enumerate() {
            let cs = rope(cs_all[1], if live(c) { position } else { 0 })?;
            for (lane, &value) in cs.iter().enumerate() {
                w.set_f32(tables.cs_g1, c * ROPE + lane, value);
            }
        }
    }
    for (owner, &store) in model.owners.iter().zip(&tables.cmp_store) {
        let entries: Vec<(usize, i64, usize)> = if owner.ratio > 1 {
            w.fill(store, groups2 * 8, -1);
            odd.iter()
                .enumerate()
                .map(|(g, &c)| (g, entries2[g], col_row[c] as usize))
                .collect()
        } else {
            w.fill(store, t * 8, -1);
            cols.iter()
                .map(|&c| (c, pos[c], col_row[c] as usize))
                .collect()
        };
        for (g, j, k) in entries {
            let p = j / page;
            let lat0 = padded(history(k), history_width, p / owner.pages, -1)?
                + owner.offset
                + PAIR_ROWS * (p % owner.pages);
            let e = j % page;
            // fp4 store (ds_store_f4): lane 0 = the latent page's first row, lane 1 = the entry index;
            // the kernel derives the code/scale rows for the latent and the index page (lat0 + PAGE_ROWS).
            w.set(store, g * 8, lat0);
            w.set(store, g * 8 + 1, e);
        }
    }

    prof.mark("compressor");
    // indexer: the tiled chain's *group* tables are built on the device (``ds_ix_gslot``), not here: the fill
    // publishes only what the step itself fixes, each O(T) -- the columns' visible entries, each query block's
    // position and seen count, and the group count -- so its index work no longer grows with the number of
    // groups (it was groups x T entries per ratio, per layer).
    let mut counts = Counts {
        sparse: false,
        cand: false,
        ..Counts::default()
    };
    for (ri, ratio) in [(0usize, 1i64), (1usize, 2i64)] {
        let vis: Vec<i64> = (0..t)
            .map(|c| if live(c) { (pos[c] + 1) / ratio } else { 0 })
            .collect();
        if std::env::var("RESIN_DS41_DUMP").is_ok() && ri == 1 {
            let g = |c: usize| if c < t { vis[c] } else { -999 };
            let p = |c: usize| if c < t { pos[c] } else { -999 };
            eprintln!("[filldbg] t={t} n={n} pos0={} real={} vis[0,500,512,1535,1536,1746]=[{},{},{},{},{},{}] pos[500,1746]=[{},{}] live[500,1536,1746]=[{},{},{}]",
                rows.pos0[0], rows.real[0], g(0), g(500), g(512), g(1535), g(1536), g(1746), p(500), p(1746),
                live(500.min(t-1)), live(1536.min(t-1)), live(1746.min(t-1)));
        }
        for (c, &value) in vis.iter().enumerate() {
            w.set(tables.ix_vis[ri], c, value);
        }
        let most = vis.iter().copied().max().unwrap_or(0);
        // each query block: its first column's position and the entries its last column sees. ds_ix_gslot turns
        // these into every group's slot records on the device.
        let mut chunks = 0i64;
        for k in 0..r_count {
            let (pos0, real) = (rows.pos0[k], rows.real[k]);
            let seen_all = (pos0 + real) / ratio;
            for b0 in (0..real).step_by(qb) {
                let q = pos0 + b0;
                let seen = ((q + qb as i64) / ratio).min(seen_all);
                // the block is the one its columns fall in: b0 / qb inside a prefill chunk's single row, and
                // the column itself in a decode step (where every row is one column)
                let b = ((rows.column[k] + b0) / qb as i64) as usize;
                w.set(tables.blk[ri], 2 * b, q);
                w.set(tables.blk[ri], 2 * b + 1, seen);
                chunks = chunks.max((seen + INDEX_KEYS - 1) / INDEX_KEYS);
            }
        }
        let candidate_ratio = model.candidate.is_none_or(|(r, _)| r == ratio);
        for (c, &value) in vis.iter().enumerate() {
            let blocks_of = (value + 7) / 8;
            w.set(
                if candidate_ratio {
                    tables.ix_visb
                } else {
                    tables.ix_visb2
                },
                c,
                ((blocks_of + 63) / 64 * 64).min(model.kmax / 8),
            );
        }
        // a prefill step whose columns see few entries: a slot per query block over entries 0 .. klen - 1 (what
        // its last column sees), and the owners' latent pages of those entries
        counts.dense[ri] = tables.cslots.is_some()
            && model.dense_keys > 0
            && cols.iter().all(|&c| vis[c] <= model.dense_keys);
        if let Some(cslots) = tables.cslots.filter(|_| counts.dense[ri]) {
            let table = cslots[ri];
            w.fill(table, win_slots * 8, 0);
            for s in 0..win_slots {
                w.set(table, s * 8, -1);
            }
            let dense_pages = (model.dense_keys / page) as usize;
            let owners: Vec<(usize, &Owner)> = model
                .owners
                .iter()
                .enumerate()
                .filter(|(_, o)| o.ratio == ratio)
                .collect();
            for &(o, _) in &owners {
                w.fill(tables.cmp_lat[o], win_slots * dense_pages, -1);
            }
            for (s, &(k, q0, col0)) in slot_q0.iter().enumerate() {
                let klen = (q0 + qb as i64).min(last[k] + 1) / ratio;
                for (lane, value) in [0, model.heads * col0, q0, 0, klen, col0, ratio, 0]
                    .into_iter()
                    .enumerate()
                {
                    w.set(table, s * 8 + lane, value);
                }
                for &(o, owner) in &owners {
                    for pg in 0..dense_pages as i64 {
                        if pg * page < klen {
                            let value = padded(history(k), history_width, pg / owner.pages, -1)?
                                + owner.offset
                                + PAIR_ROWS * (pg % owner.pages);
                            w.set(tables.cmp_lat[o], s * dense_pages + pg as usize, value);
                        }
                    }
                }
            }
        }
        // The candidate prefilter and the sparse reindex arm belong to the candidate source's ratio only
        // (its top-2048 blocks gate the later layers of that same ratio). Drive them from that ratio's
        // visibility, not the last ratio's, and for prefill as well as decode.
        if candidate_ratio {
            counts.cand = model.candidate.is_some_and(|(_, topk)| most > topk * 8);
            counts.sparse =
                tables.ixsc_batches > 0 && cols.iter().all(|&c| vis[c] > model.kc);
            if counts.sparse {
                for (c, _) in vis.iter().enumerate() {
                    w.set(tables.ix_visc, c, if live(c) { model.kc } else { 0 });
                }
                // query-block batches: batch b scores its own columns (relative rc) into the bounded staging.
                // Each batch's records are rc*chunks+chunk packed into ixc_per_batch launches of ix_slots; the
                // step's native program runs batches*(per+1) ops (the per-batch selection follows its scores).
                let chunks = (model.kc / INDEX_KEYS) as usize;
                let (sb, batches, per) =
                    (tables.sparse_batch, tables.ixsc_batches, tables.ixc_per_batch);
                let total = batches * per * tables.ix_slots;
                let mut table = vec![[0i64; 8]; total];
                for row in &mut table {
                    row[0] = -1;
                }
                for b in 0..batches {
                    for rc in 0..sb {
                        let c = b * sb + rc;
                        if c >= t || !live(c) {
                            continue;
                        }
                        for chunk in 0..chunks {
                            let ri = rc * chunks + chunk;
                            let flat = (b * per + ri / tables.ix_slots) * tables.ix_slots
                                + ri % tables.ix_slots;
                            table[flat] = [
                                0,
                                model.index_heads * rc as i64,
                                model.kc - 1,
                                INDEX_KEYS * chunk as i64,
                                model.kc,
                                rc as i64,
                                1,
                                0,
                            ];
                        }
                    }
                }
                counts.ixc = batches * (per + 1);
                for (i, row) in table.iter().enumerate() {
                    for (lane, &value) in row.iter().enumerate() {
                        w.set(tables.ixsc_rec, i * 8 + lane, value);
                    }
                }
            }
        }
        // each live column's HISTORY rows, per page of this ratio. A column's row depends only on its
        // sequence's HISTORY (not the column): compute each row's page buffer once, copy to its columns.
        let per = (HISTORY_TOKENS / ratio / page) as usize;
        let blocks = tables.cmp_blocks[ri];
        let pages_n = model.pages;
        let mut row_bufs: Vec<Vec<i32>> = Vec::with_capacity(r_count);
        for k in 0..r_count {
            let hist = history(k);
            let mut buf = vec![0i32; pages_n];
            for (i, b) in buf.iter_mut().enumerate() {
                let value = padded(hist, history_width, (i / per) as i64, -1)?;
                *b = if value >= 0 {
                    (value + PAIR_ROWS * (i % per) as i64) as i32
                } else {
                    -1
                };
            }
            row_bufs.push(buf);
        }
        for c in 0..t {
            if !live(c) {
                w.fill(blocks + c * pages_n * 4, pages_n, -1);
                continue;
            }
            let k = col_row[c] as usize;
            w.copy_words(blocks, c * pages_n, &row_bufs[k]);
        }
        // the bucketed G-tile chain: each arm takes ceil(chunks / ix_chain_g) recipe launches (each ONE launch
        // carrying the device-resident running top-512; the candidate's block-max/mask is inside the recipe, and
        // the source's block-publish tail runs after the take). O(tiles/G) launches, O(1) host metadata.
        let g = tables.ix_chain_g.max(1) as i64;
        let used = ((chunks + g - 1) / g) as usize;
        counts.ix[ri] = used;
        counts.pf[ri] = used;
        if std::env::var("RESIN_DS41_DEBUG_COUNTS").is_ok() {
            eprintln!(
                "[fill ri={}] chunks={} chain_g={} used={} => counts.ix={} counts.pf={}",
                ri, chunks, g, used, counts.ix[ri], counts.pf[ri]
            );
        }
    }

    prof.mark("index");
    if with_tokens {
        fill_tokens(model, tables, host, rows, n, tokens, host_tables)?;
    }
    prof.mark("tokens");
    prof.report(t, tables.prefill);
    Ok(counts)
}

/// `_fill_tokens`: the tables of a host copy that need the step's `n` tokens (None: the device copies the
/// previous step's samples), the tokens and their Engram rows.
pub fn fill_tokens(
    model: &Model,
    tables: &Tables,
    host: &mut [u8],
    rows: &Rows<'_>,
    n: usize,
    tokens: Option<&[i64]>,
    host_tables: &[HostTable],
) -> Result<(), String> {
    rows.validate()?;
    let t = tables.t;
    if n == 0 || n > t {
        return Err(format!("{n} tokens for a step of {t} columns"));
    }
    let mut w = words(host, tables.nbytes)?;
    let r_count = rows.len();
    if let Some(tokens) = tokens {
        if tokens.len() < n {
            return Err(format!("{} tokens for {n} positions", tokens.len()));
        }
        for (c, &token) in tokens[..n].iter().enumerate() {
            w.set(tables.tok, c, token);
        }
    }
    if let Some(engram) = &model.engram
        && !tables.engram.is_empty()
    {
        let older_n = engram.n - 1;
        let mut older = vec![engram.pad_id; t * older_n];
        for k in 0..r_count {
            let (column, real) = (rows.column[k] as usize, rows.real[k] as usize);
            if real != 1 && tokens.is_none() {
                return Err(
                    "a step with several columns per row needs its tokens on the host".into(),
                );
            }
            let mut seen: Vec<i64> = Rows::part(rows.before, rows.before_indptr, k).to_vec();
            for j in 0..real {
                if j > 0 {
                    seen.push(tokens.expect("checked above")[column + j - 1]);
                }
                for (i, &token) in seen.iter().rev().take(older_n).enumerate() {
                    let id = usize::try_from(token)
                        .ok()
                        .and_then(|token| engram.token_map.get(token))
                        .ok_or_else(|| format!("token {token} is outside the Engram map"))?;
                    older[(column + j) * older_n + i] = *id;
                }
            }
        }
        let width = older_n * engram.heads;
        for &(hi, lo, e) in &tables.engram {
            let multipliers = &engram.multipliers[e * engram.n..(e + 1) * engram.n];
            for c in 0..t {
                let mut acc = 0i64;
                for i in 0..older_n {
                    acc ^= older[c * older_n + i].wrapping_mul(multipliers[i + 1]);
                    let bits = acc as u64;
                    for head in 0..engram.heads {
                        let lane = i * engram.heads + head;
                        debug_assert!(lane < width);
                        w.set(hi, c * 64 + lane, (bits >> 32) as u32 as i32 as i64);
                        w.set(lo, c * 64 + lane, (bits & 0xFFFF_FFFF) as u32 as i32 as i64);
                    }
                }
            }
        }
    }
    if let Some(engram) = &model.engram
        && !tables.engram_ids.is_empty()
    {
        let tokens = tokens.ok_or("a step that hashes on the host needs its tokens")?;
        let ids = engram_ids(engram, rows, tokens, t)?;
        let width = (engram.n - 1) * engram.heads;
        for &(table, e) in &tables.engram_ids {
            for c in 0..t {
                for j in 0..width {
                    w.set(
                        table,
                        c * width + j,
                        ids[(c * engram.layers + e) * width + j],
                    );
                }
            }
        }
    }
    if let Some(engram) = &model.engram
        && !tables.engram_emb.is_empty()
    {
        let tokens = tokens.ok_or("a step that gathers Engram on the host needs its tokens")?;
        let ids = engram_ids(engram, rows, tokens, t)?;
        engram_gather_host(engram, tables, &mut w, &ids, host_tables, t)?;
    }
    Ok(())
}

/// The shared host-gather core: dequantize `ids.len()` global rows of one host Engram table into
/// `out` (row-major `(n, 256)` f32): `out[r*256+i] = LUT[scale[row][i/32]][weight[row][i]]`. Both the
/// step fill (`engram_gather_host`) and the oracle gate call this, so they dequantize identically.
pub fn engram_gather_into(
    weight: &[u8],
    scale: &[u8],
    ids: &[i64],
    out: &mut [f32],
) -> Result<(), String> {
    let rows = weight.len() / 256;
    if weight.len() % 256 != 0 || scale.len() != rows * 8 {
        return Err("host Engram table is not (rows, 256) codes + (rows, 8) scales".into());
    }
    if out.len() != ids.len() * 256 {
        return Err("the gather output is not ids.len() x 256".into());
    }
    let lut = engram_lut();
    for (r, &id) in ids.iter().enumerate() {
        let row = usize::try_from(id)
            .ok()
            .filter(|&x| x < rows)
            .ok_or_else(|| format!("Engram row {id} is outside the host table"))?;
        let codes = &weight[row * 256..row * 256 + 256];
        let scales = &scale[row * 8..row * 8 + 8];
        let out_row = &mut out[r * 256..r * 256 + 256];
        // hoist the LUT row per 32-code block (one scale lookup, not one per code)
        for (blk, &sc) in scales.iter().enumerate() {
            let l = &lut[usize::from(sc)];
            let src = &codes[blk * 32..blk * 32 + 32];
            let dst = &mut out_row[blk * 32..blk * 32 + 32];
            for (d, &c) in dst.iter_mut().zip(src) {
                *d = l[usize::from(c)];
            }
        }
    }
    Ok(())
}

/// The host Engram gather: for each `(offset, e)` embedding table, every column `c` and lane `j`,
/// read global row `ids[(c*layers+e)*width+j]` from the layer's mmap table, dequantize its 256 codes
/// with the LUT (one E8M0 scale per 32 codes), and write `eg_emb[256*j+i, c]` — the f32 layout the
/// device's `engram_graph` reads. Runs inside the fill under `py.detach` (no GIL, no per-step Python).
///
/// Prefill shapes gather on scoped worker threads (columns are independent, the ids are a pure hash
/// of host-known prompt tokens, the LUT and the mmap tables are shared read-only); decode shapes
/// stay serial — their gather is a few dozen rows and the decode floor is not to be perturbed.
fn engram_gather_host(
    engram: &Engram,
    tables: &Tables,
    w: &mut Words<'_>,
    ids: &[i64],
    host_tables: &[HostTable],
    t: usize,
) -> Result<(), String> {
    let threads = gather_threads(tables.prefill, t);
    if threads <= 1 {
        engram_gather_serial(engram, tables, w, ids, host_tables, t)
    } else {
        engram_gather_threaded(engram, tables, w, ids, host_tables, t, threads)
    }
}

/// The gather's worker count: `RESIN_DS41_GATHER_THREADS` (default 8, 0/1: serial), capped by the
/// column count and the box's cores. Prefill only — a decode shape always returns 1.
fn gather_threads(prefill: bool, t: usize) -> usize {
    if !prefill {
        return 1;
    }
    static CFG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cfg = *CFG.get_or_init(|| {
        std::env::var("RESIN_DS41_GATHER_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
    });
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cfg.min(t).min(cores).max(1)
}

/// The serial gather: one column at a time through a scratch row buffer, then a bulk write.
fn engram_gather_serial(
    engram: &Engram,
    tables: &Tables,
    w: &mut Words<'_>,
    ids: &[i64],
    host_tables: &[HostTable],
    t: usize,
) -> Result<(), String> {
    let width = (engram.n - 1) * engram.heads;
    let mut col_ids = vec![0i64; width];
    let mut scratch = vec![0f32; width * 256];
    for &(offset, e) in &tables.engram_emb {
        let ht = host_tables
            .get(e)
            .ok_or_else(|| format!("no host Engram table for layer index {e}"))?;
        for c in 0..t {
            for j in 0..width {
                col_ids[j] = ids[(c * engram.layers + e) * width + j];
            }
            engram_gather_into(ht.weight, ht.scale, &col_ids, &mut scratch)?;
            let base = c * (256 * width);
            w.set_many_f32(offset, base, &scratch);
        }
    }
    Ok(())
}

/// The threaded gather: each layer's embedding region is split into contiguous column ranges, one
/// per scoped worker, and every worker dequantizes its columns straight into its own disjoint slice
/// of the host image (no scratch, no second copy). The mmap tables and the LUT are shared read-only;
/// the workers touch no Python and no device state, so this is safe under the fill's `py.detach`.
/// A worker's error (or panic) fails the fill; partial writes are harmless — an errored fill is
/// never uploaded.
fn engram_gather_threaded(
    engram: &Engram,
    tables: &Tables,
    w: &mut Words<'_>,
    ids: &[i64],
    host_tables: &[HostTable],
    t: usize,
    threads: usize,
) -> Result<(), String> {
    let width = (engram.n - 1) * engram.heads;
    let stride = 256 * width; // f32 words one column occupies in an embedding table
    let per = t.div_ceil(threads);
    let ranges: Vec<(usize, usize)> = (0..threads)
        .map(|k| (k * per, ((k + 1) * per).min(t)))
        .filter(|&(a, b)| a < b)
        .collect();
    let all: &mut [i32] = w.0;
    for &(offset, e) in &tables.engram_emb {
        let ht = host_tables
            .get(e)
            .ok_or_else(|| format!("no host Engram table for layer index {e}"))?;
        let layer = &mut all[offset / 4..offset / 4 + t * stride];
        // the workers' disjoint column slices of this layer's region
        let mut rest = layer;
        let mut slices = Vec::with_capacity(ranges.len());
        for &(c0, c1) in &ranges {
            let (mine, tail) = rest.split_at_mut((c1 - c0) * stride);
            slices.push(mine);
            rest = tail;
        }
        let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(ranges.len());
            for (&(c0, c1), slice) in ranges.iter().zip(slices) {
                handles.push(scope.spawn(move || -> Result<(), String> {
                    for c in c0..c1 {
                        let col = &ids[(c * engram.layers + e) * width..][..width];
                        let dst = &mut slice[(c - c0) * stride..][..stride];
                        // SAFETY: i32 and f32 have identical size and alignment; the host image is
                        // 4-byte aligned (words()) and every table offset is a multiple of 4; this
                        // worker exclusively owns its column range's slice for the whole gather.
                        let out = unsafe {
                            std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), stride)
                        };
                        engram_gather_into(ht.weight, ht.scale, col, out)?;
                    }
                    Ok(())
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("an Engram gather worker panicked".into()))
                })
                .collect()
        });
        for r in results {
            r?;
        }
    }
    Ok(())
}

/// Cross-chunk Engram prefetch: gather one prefill chunk's `eg_emb` into a caller-owned host buffer,
/// ahead of the step that will consume it, so the gather overlaps the previous chunk's device submit.
///
/// `before` are the `engram.n - 1` (or fewer) tokens immediately before `pos0` and `tokens` the chunk's
/// own `n` tokens — exactly what the consuming step's `Rows` carries (the context keeps only the newest
/// tokens, and `engram_ids` reads no further back). The buffer is layer-major: layer `e`'s `t` columns
/// of `256 * width` f32 at byte `e * t * 256 * width * 4`, the layout `engram_gather_host` writes. This
/// calls the SAME `engram_ids` + `engram_gather_host` as the step fill, so a matching chunk's bytes are
/// bit-identical to gathering in place; the caller copies them into the image and skips the fill's gather.
pub fn engram_prefetch_gather(
    engram: &Engram,
    host_tables: &[HostTable],
    before: &[i64],
    tokens: &[i64],
    pos0: i64,
    t: usize,
    out: &mut [u8],
) -> Result<(), String> {
    let n = tokens.len();
    let width = (engram.n - 1) * engram.heads;
    let per = t * 256 * width * 4;
    let column = [0i64];
    let pos = [pos0];
    let real = [n as i64];
    let before_indptr = [0i64, before.len() as i64];
    // engram_ids reads only column/pos0/real/before; the window and history parts are unused here.
    let empty = [0i64, 0];
    let rows = Rows {
        column: &column,
        pos0: &pos,
        real: &real,
        window_first: &column,
        window_indptr: &empty,
        window: &[],
        window_ids: &[],
        history_indptr: &empty,
        history: &[],
        before_indptr: &before_indptr,
        before,
    };
    let ids = engram_ids(engram, &rows, tokens, t)?;
    let tables = Tables {
        t,
        prefill: true,
        engram_emb: (0..engram.layers).map(|e| (e * per, e)).collect(),
        nbytes: engram.layers * per,
        ..Tables::default()
    };
    let mut w = words(out, tables.nbytes)?;
    engram_gather_host(engram, &tables, &mut w, &ids, host_tables, t)
}

/// A DSpark draft step's settings: the drafted block, its noise token, and the drafter's stages (the last layers
/// of `layer_rows`).
#[derive(Clone, Copy, Debug)]
pub struct Draft {
    pub block: usize,
    pub noise: i64,
    pub stages: usize,
}

/// Which decode steps draft (DSpark): with drafting on, a step of up to `rows` rows verifies each row's anchor
/// and `tokens` drafts, and the Engine leases `tokens` positions past each row's token. Other steps, and every
/// step while drafting is off, decode one token a row and lease no lookahead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Drafting {
    /// the drafts a speculative step verifies (0: no DSpark)
    pub tokens: i64,
    /// the most rows a speculative step takes
    pub rows: usize,
    /// the runner's switch
    pub on: bool,
    /// a follower rank: the driver's decision for the call it replays (it decides for every rank)
    pub decided: Option<bool>,
}

impl Drafting {
    /// Whether a decode step of `rows` rows drafts.
    pub fn drafts(&self, rows: usize) -> bool {
        self.decided
            .unwrap_or(self.on && self.tokens > 0 && rows <= self.rows)
    }

    /// The positions the Engine leases past a decode row's token in a step of `rows` rows.
    pub fn lookahead(&self, rows: usize) -> i64 {
        if self.drafts(rows) { self.tokens } else { 0 }
    }
}

/// `_fill_draft`: a draft step's tables. Row b's block is in columns b block .. b block + block - 1 at positions
/// pos0 .. (their RoPE), each column's window slot over the stages' pages of the target's last positions before
/// pos0 (the block's own keys are not stored), and the block keys' columns and slots. `anchors`: each row's
/// anchor token (at its pos0).
pub fn fill_draft(
    model: &Model,
    tables: &Tables,
    host: &mut [u8],
    rows: &Rows<'_>,
    anchors: &[i64],
    draft: Draft,
    cs_all: [&[f32]; 2],
) -> Result<(), String> {
    rows.validate()?;
    let (bidx, lslots, anchor) = tables.draft.ok_or("this shape has no draft tables")?;
    let (t, k, b) = (tables.t, draft.block, rows.len());
    let n = b * k;
    if n > t || anchors.len() != b || draft.stages > tables.lat.len() {
        return Err(format!("{b} rows of {k} drafts for a step of {t} columns"));
    }
    if host.len() < tables.nbytes || !tables.nbytes.is_multiple_of(4) {
        return Err("the host copy is too small or misaligned".into());
    }
    host[..tables.nbytes].fill(0);
    let mut w = words(host, tables.nbytes)?;
    let page = PAGE_TOKENS;
    let sliding = model.sliding_window;
    let mut pos = vec![0i64; t];
    for c in 0..n {
        let (r, i) = (c / k, c % k);
        w.set(tables.tok, c, if i == 0 { anchors[r] } else { draft.noise });
        pos[c] = rows.pos0[r] + i as i64;
    }
    w.set(tables.meta, 0, rows.pos0[0]);
    w.set(tables.meta, 1, n as i64);
    for r in 0..2 {
        for (column, &position) in pos.iter().enumerate() {
            let cs = rope(cs_all[r], position)?;
            w.set_many_f32(tables.cs[r], column * ROPE, cs);
            // cs_inv is cs with its last 64 lanes negated: copy the row, then flip the tail's signs
            w.set_many_f32(tables.cs_inv[r], column * ROPE, cs);
            let inv_start = tables.cs_inv[r] / 4 + column * ROPE;
            for d in w.0[inv_start + 64..inv_start + ROPE].iter_mut() {
                *d = (-f32::from_bits(*d as u32)).to_bits() as i32;
            }
        }
    }
    let pages = tables.win_pages;
    for s in 0..tables.win_slots {
        w.set(tables.slots, s * 8, -1);
    }
    let mut lat0 = vec![-1i64; tables.win_slots * pages];
    for c in 0..n {
        let r = c / k;
        let start = rows.pos0[r];
        // every column's window ends at the position before its anchor
        let q = start - 1;
        let key0 = (q - (sliding - 1)).max(0) / page * page;
        let values = [0, model.heads * c as i64, q, key0, start, 0, 0, sliding];
        for (lane, value) in values.into_iter().enumerate() {
            w.set(tables.slots, c * 8 + lane, value);
        }
        let window = Rows::part(rows.window, rows.window_indptr, r);
        let first = rows.window_first[r];
        for p in 0..pages {
            let idx = key0 / page + p as i64 - first;
            if q >= 0 && idx >= 0 && (idx as usize) < window.len() && idx + first <= q / page {
                lat0[c * pages + p] = window[idx as usize];
            }
        }
    }
    for &table in &tables.lat {
        w.fill(table, tables.win_slots * pages, -1);
    }
    for &table in &tables.rows {
        w.fill(table, t * 8, -1); // nothing stored
    }
    let layers = tables.lat.len();
    for (&table, &offset) in tables.lat[layers - draft.stages..]
        .iter()
        .zip(&model.layer_rows[layers - draft.stages..])
    {
        for (index, &value) in lat0.iter().enumerate() {
            w.set(table, index, if value >= 0 { value + offset } else { -1 });
        }
    }
    for c in 0..n {
        for j in 0..k {
            w.set(bidx, c * k + j, ((c / k) * k + j) as i64);
        }
    }
    for s in 0..t {
        w.set(lslots, s * 8, -1);
    }
    for c in 0..n {
        let values = [
            0,
            model.heads * c as i64,
            k as i64,
            0,
            k as i64,
            c as i64,
            0,
            0,
        ];
        for (lane, value) in values.into_iter().enumerate() {
            w.set(lslots, c * 8 + lane, value);
        }
    }
    for (r, &token) in anchors.iter().enumerate() {
        w.set(anchor, r, token);
    }
    Ok(())
}

/// `_engram_ids`: every column's hash ids, (t, layers, (n - 1) heads) (padding columns: row 0). Each column hashes
/// its token and the n - 1 before it (a row's `before`, then its own tokens), newest first, `pad_id` before
/// the sequence's start.
fn engram_ids(
    engram: &Engram,
    rows: &Rows<'_>,
    tokens: &[i64],
    t: usize,
) -> Result<Vec<i64>, String> {
    let (n, heads, layers) = (engram.n, engram.heads, engram.layers);
    let width = (n - 1) * heads;
    let mut out = vec![0i64; t * layers * width];
    let compressed = |token: i64| {
        usize::try_from(token)
            .ok()
            .and_then(|token| engram.token_map.get(token))
            .copied()
            .ok_or_else(|| format!("token {token} is outside the Engram map"))
    };
    let mut window = vec![0i64; n];
    for k in 0..rows.len() {
        let (column, pos0, real) = (rows.column[k] as usize, rows.pos0[k], rows.real[k] as usize);
        if tokens.len() < column + real {
            return Err(format!(
                "{} tokens for columns up to {}",
                tokens.len(),
                column + real
            ));
        }
        let before = Rows::part(rows.before, rows.before_indptr, k);
        let seq = |at: i64| -> i64 {
            let at = at.max(0) as usize;
            if at < before.len() {
                before[at]
            } else {
                tokens[column + at - before.len()]
            }
        };
        for j in 0..real {
            for (back, slot) in window.iter_mut().enumerate() {
                *slot = if pos0 + (j as i64) < back as i64 {
                    engram.pad_id
                } else {
                    compressed(seq(before.len() as i64 + j as i64 - back as i64))?
                };
            }
            let c = column + j;
            for e in 0..layers {
                let multipliers = &engram.multipliers[e * n..(e + 1) * n];
                let mut rolling = window[0].wrapping_mul(multipliers[0]);
                for i in 1..n {
                    rolling ^= window[i].wrapping_mul(multipliers[i]);
                    for head in 0..heads {
                        let prime = engram.primes[(e * (n - 1) + i - 1) * heads + head];
                        let lane = (i - 1) * heads + head;
                        out[(c * layers + e) * width + lane] =
                            rolling.rem_euclid(prime) + engram.offsets[e * width + lane];
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Whether a prefill row runs with bounded replay, the layers after the cut only on the prompt's positions from
/// `floor` on: with a cut and a replay span, when the prompt is longer than the span, and when the row scores no prompt
/// logprobs (a scored row needs every layer at every position, so it runs whole). The choice is the row's own, not its
/// step's: every chunk of a prompt must make the same one, because a chunk that runs whole after replayed chunks reads
/// windows of the layers after the cut that those chunks did not write.
pub fn replays(cut: bool, replay: i64, floor: i64, scored: bool) -> bool {
    cut && replay > 0 && floor > 0 && !scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexing_follows_numpy() {
        // a padded row: negative indices count from the padded end; past the row's values, the padding
        assert_eq!(padded(&[5, 6], 4, 1, -1), Ok(6));
        assert_eq!(padded(&[5, 6], 4, 3, -1), Ok(-1));
        assert_eq!(padded(&[5, 6], 4, -3, -1), Ok(6));
        assert!(padded(&[5, 6], 4, 4, -1).is_err());
        // a row's own values: negative indices count from its own end
        assert_eq!(own(&[5, 6, 7], -1), Ok(7));
        assert!(own(&[5, 6, 7], 3).is_err());
    }

    #[test]
    fn rows_must_delimit_their_parts() {
        let one = [0i64];
        let indptr = [0i64, 1];
        let values = [9i64];
        let rows = Rows {
            column: &one,
            pos0: &one,
            real: &[1],
            window_first: &one,
            window_indptr: &indptr,
            window: &values,
            window_ids: &values,
            history_indptr: &indptr,
            history: &values,
            before_indptr: &[0, 0],
            before: &[],
        };
        assert!(rows.validate().is_ok());
        let short = Rows {
            history_indptr: &[0, 2],
            ..rows
        };
        assert!(short.validate().is_err());
    }

    #[test]
    fn a_misaligned_or_short_host_copy_is_rejected() {
        let model = Model {
            heads: 8,
            index_heads: 32,
            sliding_window: 128,
            layer_rows: vec![],
            owners: vec![],
            pending: vec![],
            pages: 1,
            kmax: 4096,
            kc: 0,
            candidate: None,
            engram: None,
            radix_min: 20480,
            prefilter_blocks: 0,
            prefilter_min: 0,
            dense_keys: 0,
        };
        let tables = Tables {
            t: 1,
            ix_slots: 16,
            nbytes: 1024,
            ..Tables::default()
        };
        let one = [0i64];
        let rows = Rows {
            column: &one,
            pos0: &one,
            real: &[1],
            window_first: &one,
            window_indptr: &[0, 1],
            window: &[1],
            window_ids: &[1],
            history_indptr: &[0, 1],
            history: &[1],
            before_indptr: &[0, 0],
            before: &[],
        };
        let cs = vec![0f32; 128];
        let mut host = vec![0u8; 512];
        let error = fill(
            &model,
            &tables,
            &mut host,
            &rows,
            &[0],
            None,
            true,
            None,
            [&cs, &cs],
            &[],
        );
        assert!(error.is_err());
    }

    #[test]
    fn blk_is_per_ratio() {
        // The host fill must write each ratio's query-block `seen` to its OWN blk table: ratio-1 gslot reads
        // blk1, ratio-2 gslot reads blk2. A shared table (ratio-2 written last) left ratio-1 records with
        // ratio-2 seen/klen, corrupting ratio-1 top-512 selection at vis>512 (the 512k context goal).
        let model = Model {
            heads: 8,
            index_heads: 32,
            sliding_window: 128,
            layer_rows: vec![],
            owners: vec![],
            pending: vec![],
            pages: 1,
            kmax: 4096,
            kc: 0,
            candidate: None,
            engram: None,
            radix_min: 20480,
            prefilter_blocks: 0,
            prefilter_min: 0,
            dense_keys: 0,
        };
        let tables = Tables {
            t: 32,
            prefill: true,
            qb: 32,
            win_slots: 1,
            win_pages: 3,
            groups2: 17,
            ix_slots: 8,
            ix_tp: 1,
            ix_lpg: 1,
            ix_chain_g: 8,
            ix_rows: 2048,
            nbytes: 200000,
            meta: 0,
            cs: [1000, 20000],
            cs_inv: [40000, 60000],
            slots: 80000,
            ix_vis: [81000, 82000],
            ix_visb: 83000,
            ix_visb2: 84000,
            cmp_blocks: [85000, 86000],
            blk: [90000, 91000],
            ..Tables::default()
        };
        let zero = [0i64];
        let real = [32i64];
        let rows = Rows {
            column: &zero,
            pos0: &zero,
            real: &real,
            window_first: &zero,
            window_indptr: &[0, 1],
            window: &[0],
            window_ids: &[0],
            history_indptr: &[0, 1],
            history: &[0],
            before_indptr: &[0, 0],
            before: &[],
        };
        let cs = vec![0f32; 32 * 128];
        let mut host = vec![0u8; 200000];
        fill(&model, &tables, &mut host, &rows, &(0..32).collect::<Vec<i64>>(), None, false, None,
             [&cs, &cs], &[]).unwrap();
        let word = |off: usize| i32::from_le_bytes(host[off..off + 4].try_into().unwrap());
        // block 0: lane 0 the first position (0), lane 1 the entries its last column sees
        assert_eq!(word(tables.blk[0]), 0);
        assert_eq!(word(tables.blk[0] + 4), 32); // ratio 1: (0 + 32) / 1
        assert_eq!(word(tables.blk[1]), 0);
        assert_eq!(word(tables.blk[1] + 4), 16); // ratio 2: (0 + 32) / 2
    }

    #[test]
    fn drafts_only_on_narrow_steps_while_drafting_is_on() {
        let on = Drafting {
            tokens: 3,
            rows: 2,
            on: true,
            decided: None,
        };
        assert_eq!(
            (1..=4).map(|rows| on.lookahead(rows)).collect::<Vec<_>>(),
            [3, 3, 0, 0]
        );
        let off = Drafting { on: false, ..on };
        assert!((1..=64).all(|rows| !off.drafts(rows) && off.lookahead(rows) == 0));
        // a runner without DSpark never drafts
        let none = Drafting { tokens: 0, ..on };
        assert!(!none.drafts(1) && none.lookahead(1) == 0);
        // a follower takes the driver's decision, whatever its own switch says
        for decided in [false, true] {
            let follower = Drafting {
                on: !decided,
                decided: Some(decided),
                ..on
            };
            assert_eq!(follower.drafts(1), decided);
            assert_eq!(follower.lookahead(1), if decided { 3 } else { 0 });
        }
    }

    #[test]
    fn replay_is_the_rows_own_choice() {
        assert!(replays(true, 256, 2044, false));
        assert!(
            !replays(true, 256, 2044, true),
            "a row that scores prompt logprobs runs whole"
        );
        assert!(
            !replays(true, 256, 0, false),
            "a prompt within the replay span runs whole"
        );
        assert!(!replays(false, 256, 2044, false), "no cut");
        assert!(!replays(true, 0, 2044, false), "no replay span");
    }

    /// An independent transcription of the OCP E4M3 (bias 7) decode, for the LUT oracle (not the
    /// production helper): sign first, 0x7F/0xFF read as 0, subnormal exponent 0, magnitude to 448.
    fn ref_ocp(code: u8) -> f64 {
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((code >> 3) & 0xF);
        let mant = f64::from(code & 0x7);
        let mag = if (code & 0x7F) == 0x7F {
            0.0
        } else if exp == 0 {
            mant * 2f64.powi(-9)
        } else {
            (8.0 + mant) * 2f64.powi(exp - 10)
        };
        // sign * mag (not an early return): numpy zeros 0x7F/0xFF then negates, so OCP[0xFF] = -0.0
        sign * mag
    }

    /// The oracle the LUT must match: `OCP[code] * 2^(scale-127)`, the exponent formed in f32 (as
    /// `np.exp2(np.arange(256, f32) - 127)`), the product in f64, rounded once to f32.
    fn ref_lut(scale: usize, code: u8) -> f32 {
        let e = scale as i32 - 127;
        // numpy's f32 exp2 ufunc rounds 2^127 up by 1 ulp (the oracle's np.exp2(f32)); match it
        let exp = if e == 127 {
            f32::from_bits(0x7F00_0001)
        } else {
            2f32.powi(e)
        };
        (ref_ocp(code) * f64::from(exp)) as f32
    }

    #[test]
    fn engram_lut_matches_ocp_times_pow2_at_every_pair() {
        let lut = engram_lut();
        for s in 0..256 {
            for c in 0..256u16 {
                let got = lut[s][c as usize];
                let want = ref_lut(s, c as u8);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "LUT[{s}][{c:#04x}] = {got} ({:#010x}), oracle {want} ({:#010x})",
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
        // the f32 output scope: scale 255 overflows (inf, or NaN where OCP is 0), never a finite repair
        assert!(lut[255][1].is_infinite());
        assert!(lut[255][0x7F].is_nan());
        assert!(lut[254][0x7E].is_infinite()); // OCP[0x7E]=448, 448*2^127 overflows f32
    }

    /// A small synthetic host table and a naive reference gather, to check the production gather on
    /// the required case class: same shape, DIFFERENT data (different ids AND different table bytes).
    fn synthetic_table(rows: usize, seed: u8) -> (Vec<u8>, Vec<u8>) {
        let mut weight = vec![0u8; rows * 256];
        let mut scale = vec![0u8; rows * 8];
        for (i, b) in weight.iter_mut().enumerate() {
            *b = ((i as u32 * 31 + u32::from(seed) * 17 + 7) % 256) as u8;
        }
        for (i, b) in scale.iter_mut().enumerate() {
            // keep scales in a real range (no 254/255), varied per row/block
            *b = (100 + ((i as u32 * 13 + u32::from(seed) * 5) % 40)) as u8;
        }
        (weight, scale)
    }

    fn ref_gather(
        weight: &[u8],
        scale: &[u8],
        ids: &[i64],
        t: usize,
        layers: usize,
        width: usize,
        e: usize,
    ) -> Vec<u32> {
        let mut out = vec![0u32; t * 256 * width];
        for c in 0..t {
            for j in 0..width {
                let row = ids[(c * layers + e) * width + j] as usize;
                for i in 0..256 {
                    let code = weight[row * 256 + i];
                    let s = scale[row * 8 + i / 32];
                    let val = ref_lut(usize::from(s), code);
                    out[c * 256 * width + j * 256 + i] = val.to_bits();
                }
            }
        }
        out
    }

    #[test]
    fn engram_gather_same_shape_different_data() {
        let (n, heads, layers) = (4usize, 8usize, 1usize);
        let width = (n - 1) * heads; // 24
        let engram = Engram {
            token_map: vec![],
            pad_id: 0,
            multipliers: vec![],
            n,
            heads,
            layers,
            primes: vec![],
            offsets: vec![],
        };
        let rows = 64usize;
        for seed in [0u8, 9, 200] {
            let (weight, scale) = synthetic_table(rows, seed);
            let ht = HostTable {
                weight: &weight,
                scale: &scale,
            };
            for t in [1usize, 3, 8] {
                // three different id sets of the same shape (the required case class)
                for variant in 0..3 {
                    let ids: Vec<i64> = (0..t * layers * width)
                        .map(|k| ((k as u64 * 7 + variant * 11 + u64::from(seed)) % rows as u64) as i64)
                        .collect();
                    let nbytes = t * 256 * width * 4;
                    let mut buf = vec![0u8; nbytes];
                    let tables = Tables {
                        t,
                        engram_emb: vec![(0, 0)],
                        nbytes,
                        ..Tables::default()
                    };
                    {
                        let mut w = words(&mut buf, nbytes).unwrap();
                        engram_gather_host(&engram, &tables, &mut w, &ids, &[ht], t).unwrap();
                    }
                    let got: Vec<u32> = buf
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    let want = ref_gather(&weight, &scale, &ids, t, layers, width, 0);
                    assert_eq!(got, want, "seed {seed}, t {t}, variant {variant}");
                }
            }
        }
    }

    #[test]
    fn engram_gather_rejects_an_out_of_range_row() {
        let engram = Engram {
            token_map: vec![],
            pad_id: 0,
            multipliers: vec![],
            n: 4,
            heads: 8,
            layers: 1,
            primes: vec![],
            offsets: vec![],
        };
        let (weight, scale) = synthetic_table(8, 0);
        let ht = HostTable { weight: &weight, scale: &scale };
        let width = 24;
        let nbytes = 256 * width * 4;
        let mut buf = vec![0u8; nbytes];
        let tables = Tables { t: 1, engram_emb: vec![(0, 0)], nbytes, ..Tables::default() };
        let mut ids = vec![0i64; width];
        ids[5] = 8; // one past the last row
        let mut w = words(&mut buf, nbytes).unwrap();
        assert!(engram_gather_host(&engram, &tables, &mut w, &ids, &[ht], 1).is_err());
        // a missing host table for the layer index is an error too
        let mut w = words(&mut buf, nbytes).unwrap();
        assert!(engram_gather_host(&engram, &tables, &mut w, &vec![0i64; width], &[], 1).is_err());
    }

    /// The independent oracle LUT at all 65536 pairs, built once (the large-shape gates below do
    /// millions of lookups; `ref_lut` per element would be needlessly slow). Still shares no code
    /// with the production LUT: it is the `ref_ocp`/`ref_lut` transcription.
    fn ref_lut_table() -> Vec<[f32; 256]> {
        (0..256)
            .map(|s| {
                let mut row = [0f32; 256];
                for (c, v) in row.iter_mut().enumerate() {
                    *v = ref_lut(s, c as u8);
                }
                row
            })
            .collect()
    }

    /// One layer's oracle output with a prebuilt independent LUT: same indexing as `ref_gather`.
    #[allow(clippy::too_many_arguments)]
    fn ref_gather_lut(
        lut: &[[f32; 256]],
        weight: &[u8],
        scale: &[u8],
        ids: &[i64],
        t: usize,
        layers: usize,
        width: usize,
        e: usize,
    ) -> Vec<u32> {
        let mut out = vec![0u32; t * 256 * width];
        for c in 0..t {
            for j in 0..width {
                let row = ids[(c * layers + e) * width + j] as usize;
                for i in 0..256 {
                    let code = weight[row * 256 + i];
                    let s = scale[row * 8 + i / 32];
                    out[c * 256 * width + j * 256 + i] = lut[usize::from(s)][usize::from(code)].to_bits();
                }
            }
        }
        out
    }

    /// A multi-layer Engram and its per-layer host tables with DIFFERENT bytes per layer (a layer
    /// swap or a wrong `e` must be visible), `layers` embedding tables laid out back to back.
    #[allow(clippy::type_complexity)]
    fn threaded_fixture(
        layers: usize,
        rows: usize,
        seed: u8,
        t: usize,
    ) -> (Engram, Vec<(Vec<u8>, Vec<u8>)>, Tables, usize) {
        let (n, heads) = (4usize, 8usize);
        let engram = Engram {
            token_map: vec![],
            pad_id: 0,
            multipliers: vec![],
            n,
            heads,
            layers,
            primes: vec![],
            offsets: vec![],
        };
        let data: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|e| synthetic_table(rows, seed.wrapping_add(e as u8 * 31)))
            .collect();
        let width = (n - 1) * heads;
        let per = t * 256 * width * 4;
        let tables = Tables {
            t,
            prefill: true,
            engram_emb: (0..layers).map(|e| (e * per, e)).collect(),
            nbytes: layers * per,
            ..Tables::default()
        };
        (engram, data, tables, width)
    }

    fn ids_of(t: usize, layers: usize, width: usize, rows: usize, seed: u8, variant: u64) -> Vec<i64> {
        (0..t * layers * width)
            .map(|k| ((k as u64 * 7 + variant * 11 + u64::from(seed)) % rows as u64) as i64)
            .collect()
    }

    fn buf_words(buf: &[u8]) -> Vec<u32> {
        buf.chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// Bit-compare two word vectors; on mismatch report the FIRST differing word (a full `assert_eq`
    /// dump of a 12.6M-word vector is an 18 MB log line).
    fn assert_words_eq(got: &[u32], want: &[u32], ctx: &str) {
        assert_eq!(got.len(), want.len(), "{ctx}: length");
        if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
            panic!(
                "{ctx}: first mismatch at word {at}: got {:#010x}, want {:#010x}",
                got[at], want[at]
            );
        }
    }

    /// The THREADED gather on the required case class — same shape, DIFFERENT data (different ids,
    /// different table bytes, different bytes per layer) — at shapes where every worker owns several
    /// columns (the small-t oracle above gives each worker one column at most, so a within-range
    /// lane/order bug is invisible there). Includes the production 2048-column chunk and an odd
    /// column count (uneven ranges), one and two layers.
    #[test]
    fn engram_gather_threaded_same_shape_different_data() {
        let lut = ref_lut_table();
        let rows = 4096usize;
        for layers in [1usize, 2] {
            for seed in [0u8, 200] {
                for t in [64usize, 257, 2048] {
                    let (engram, data, tables, width) = threaded_fixture(layers, rows, seed, t);
                    let hts: Vec<HostTable> = data
                        .iter()
                        .map(|(w, s)| HostTable { weight: w, scale: s })
                        .collect();
                    for variant in 0..2u64 {
                        let ids = ids_of(t, layers, width, rows, seed, variant);
                        let mut buf = vec![0u8; tables.nbytes];
                        {
                            let mut w = words(&mut buf, tables.nbytes).unwrap();
                            engram_gather_host(&engram, &tables, &mut w, &ids, &hts, t).unwrap();
                        }
                        let got = buf_words(&buf);
                        let mut want = Vec::with_capacity(got.len());
                        for (e, (weight, scale)) in data.iter().enumerate() {
                            want.extend(ref_gather_lut(&lut, weight, scale, &ids, t, layers, width, e));
                        }
                        assert_words_eq(
                            &got,
                            &want,
                            &format!("layers {layers}, seed {seed}, t {t}, variant {variant}"),
                        );
                    }
                }
            }
        }
    }

    /// The threaded gather is bit-exact against the serial gather on the same inputs (both paths
    /// called directly, so the comparison does not depend on the dispatch or the environment).
    #[test]
    fn engram_gather_threaded_bit_exact_vs_serial() {
        let rows = 4096usize;
        for layers in [1usize, 2] {
            for t in [64usize, 257, 2048] {
                let (engram, data, tables, width) = threaded_fixture(layers, rows, 9, t);
                let hts: Vec<HostTable> = data
                    .iter()
                    .map(|(w, s)| HostTable { weight: w, scale: s })
                    .collect();
                for variant in 0..2u64 {
                    let ids = ids_of(t, layers, width, rows, 9, variant);
                    let mut serial = vec![0u8; tables.nbytes];
                    let mut threaded = vec![0u8; tables.nbytes];
                    {
                        let mut w = words(&mut serial, tables.nbytes).unwrap();
                        engram_gather_serial(&engram, &tables, &mut w, &ids, &hts, t).unwrap();
                    }
                    {
                        let mut w = words(&mut threaded, tables.nbytes).unwrap();
                        engram_gather_threaded(&engram, &tables, &mut w, &ids, &hts, t, 8).unwrap();
                    }
                    assert_words_eq(
                        &buf_words(&serial),
                        &buf_words(&threaded),
                        &format!("layers {layers}, t {t}, variant {variant}"),
                    );
                }
            }
        }
    }

    /// The threaded path keeps the serial path's error contract: an out-of-range row and a missing
    /// host table both fail the fill (here in a worker's late column, not the first).
    #[test]
    fn engram_gather_threaded_rejects_bad_rows_and_missing_tables() {
        let rows = 64usize;
        let t = 64usize;
        let (engram, data, tables, width) = threaded_fixture(1, rows, 0, t);
        let hts: Vec<HostTable> = data
            .iter()
            .map(|(w, s)| HostTable { weight: w, scale: s })
            .collect();
        let mut ids = ids_of(t, 1, width, rows, 0, 0);
        ids[40 * width + 5] = rows as i64; // one past the last row, in a late worker's range
        let mut buf = vec![0u8; tables.nbytes];
        let mut w = words(&mut buf, tables.nbytes).unwrap();
        assert!(engram_gather_threaded(&engram, &tables, &mut w, &ids, &hts, t, 8).is_err());
        // a missing host table for the layer index is an error too
        let ids = ids_of(t, 1, width, rows, 0, 0);
        let mut w = words(&mut buf, tables.nbytes).unwrap();
        assert!(engram_gather_threaded(&engram, &tables, &mut w, &ids, &[], t, 8).is_err());
    }

    /// Decode shapes never thread (the decode floor is not to be perturbed), and a one-column
    /// prefill has nothing to split.
    #[test]
    fn gather_threads_is_prefill_only() {
        assert_eq!(gather_threads(false, 2048), 1);
        assert_eq!(gather_threads(false, 64), 1);
        assert_eq!(gather_threads(true, 1), 1);
        let n = gather_threads(true, 2048);
        assert!((1..=8).contains(&n));
    }

    /// A HASHING Engram (real token_map / multipliers / primes / offsets, so `engram_ids` actually
    /// hashes), per-layer host tables with DIFFERENT bytes, and a prompt of in-map tokens. `rows` is
    /// large enough that every hashed id (prime modulus + offset) lands inside the table.
    #[allow(clippy::type_complexity)]
    fn hashing_fixture(
        layers: usize,
        rows: usize,
        seed: u8,
    ) -> (Engram, Vec<(Vec<u8>, Vec<u8>)>, Vec<i64>) {
        let (n, heads) = (4usize, 8usize);
        let width = (n - 1) * heads;
        let vocab = 4096i64;
        let token_map: Vec<i64> = (0..vocab).collect();
        // distinct non-zero multipliers per (layer, n-gram lane)
        let multipliers: Vec<i64> = (0..layers * n)
            .map(|k| (0x9E37_79B9u64.wrapping_mul(k as u64 + 1) ^ u64::from(seed)) as i64 | 1)
            .collect();
        // positive moduli, one per (layer, n-1, head); kept small so ids stay inside `rows`
        let primes: Vec<i64> = (0..layers * (n - 1) * heads)
            .map(|k| 1009 + (k as i64 * 7) % 512)
            .collect();
        let offsets: Vec<i64> = (0..layers * width).map(|k| (k as i64 * 3) % 256).collect();
        let engram = Engram {
            token_map,
            pad_id: 0,
            multipliers,
            n,
            heads,
            layers,
            primes,
            offsets,
        };
        let data: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|e| synthetic_table(rows, seed.wrapping_add(e as u8 * 31)))
            .collect();
        // a prompt of in-map tokens (varied by seed): the prefetch reconstructs ids from these
        let mut state = 0x1234_5678_9abc_def0u64 ^ u64::from(seed);
        let prompt: Vec<i64> = (0..8192)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % vocab as u64) as i64
            })
            .collect();
        (engram, data, prompt)
    }

    /// The prefetch gather is bit-identical to the production step gather on the required case class —
    /// same shape, DIFFERENT data (different prompts, pos0, chunk sizes, buckets, per-layer bytes). The
    /// production path uses the FULL prefix as `before` (what a naive reconstruction would pass); the
    /// prefetch uses only the newest `n - 1` tokens (what the context actually keeps). `engram_ids`
    /// reads no further back, so the two must agree byte for byte; a wrong `before` window or padding
    /// in the prefetch reconstruction shows up here.
    #[test]
    fn engram_prefetch_matches_step_gather_same_shape_different_data() {
        let rows = 4096usize;
        for layers in [1usize, 2] {
            for seed in [0u8, 200] {
                let (engram, data, prompt) = hashing_fixture(layers, rows, seed);
                let hts: Vec<HostTable> = data
                    .iter()
                    .map(|(w, s)| HostTable { weight: w, scale: s })
                    .collect();
                let width = (engram.n - 1) * engram.heads;
                // (pos0, n, t): a mid-prompt full chunk, an odd chunk, the small last-chunk bucket,
                // and a chunk right at the sequence start (padding before the first tokens)
                for &(pos0, n, t) in &[
                    (2048i64, 2048usize, 2048usize),
                    (4096, 93, 128),
                    (2048, 257, 512),
                    (0, 64, 128),
                    (3, 5, 32),
                ] {
                    let tokens = &prompt[pos0 as usize..pos0 as usize + n];
                    let per = t * 256 * width * 4;
                    // production: full-prefix before, the step's own ids + gather
                    let full_before = &prompt[..pos0 as usize];
                    let column = [0i64];
                    let pos = [pos0];
                    let real = [n as i64];
                    let bindptr = [0i64, full_before.len() as i64];
                    let empty = [0i64, 0];
                    let rows_full = Rows {
                        column: &column,
                        pos0: &pos,
                        real: &real,
                        window_first: &column,
                        window_indptr: &empty,
                        window: &[],
                        window_ids: &[],
                        history_indptr: &empty,
                        history: &[],
                        before_indptr: &bindptr,
                        before: full_before,
                    };
                    let ids = engram_ids(&engram, &rows_full, tokens, t).unwrap();
                    let tables = Tables {
                        t,
                        prefill: true,
                        engram_emb: (0..layers).map(|e| (e * per, e)).collect(),
                        nbytes: layers * per,
                        ..Tables::default()
                    };
                    let mut step = vec![0u8; tables.nbytes];
                    {
                        let mut w = words(&mut step, tables.nbytes).unwrap();
                        engram_gather_host(&engram, &tables, &mut w, &ids, &hts, t).unwrap();
                    }
                    // prefetch: only the newest n-1 tokens before pos0 (the context's window)
                    let keep = (engram.n - 1).min(pos0 as usize);
                    let short_before = &prompt[pos0 as usize - keep..pos0 as usize];
                    let mut pref = vec![0u8; tables.nbytes];
                    engram_prefetch_gather(
                        &engram, &hts, short_before, tokens, pos0, t, &mut pref,
                    )
                    .unwrap();
                    assert_words_eq(
                        &buf_words(&step),
                        &buf_words(&pref),
                        &format!("layers {layers}, seed {seed}, pos0 {pos0}, n {n}, t {t}"),
                    );
                }
            }
        }
    }

    /// The consume path: prefetch into a layer-major ring, then copy each layer's region into the image
    /// at its OWN (non-contiguous) `engram_emb` offset, must equal gathering straight into the image.
    /// Different per-layer bytes and a gap between the layers' regions make a swapped offset or a wrong
    /// per-layer stride visible. This is the copy/layout gate the fill's skip-gather path relies on.
    #[test]
    fn engram_prefetch_ring_copy_matches_in_place_gather() {
        let rows = 4096usize;
        let layers = 2usize;
        let (engram, data, prompt) = hashing_fixture(layers, rows, 9);
        let hts: Vec<HostTable> = data
            .iter()
            .map(|(w, s)| HostTable { weight: w, scale: s })
            .collect();
        let width = (engram.n - 1) * engram.heads;
        let (pos0, n, t) = (2048i64, 2048usize, 2048usize);
        let per = t * 256 * width * 4;
        let tokens = &prompt[pos0 as usize..pos0 as usize + n];
        // image: layer 0 at 1024, layer 1 after a 4096-byte gap (non-contiguous, distinct offsets)
        let off0 = 1024usize;
        let off1 = off0 + per + 4096;
        let nbytes = off1 + per + 1024;
        let image_tables = Tables {
            t,
            prefill: true,
            engram_emb: vec![(off0, 0), (off1, 1)],
            nbytes,
            ..Tables::default()
        };
        // in-place production gather into the image offsets
        let full_before = &prompt[..pos0 as usize];
        let column = [0i64];
        let pos = [pos0];
        let real = [n as i64];
        let bindptr = [0i64, full_before.len() as i64];
        let empty = [0i64, 0];
        let rows_full = Rows {
            column: &column,
            pos0: &pos,
            real: &real,
            window_first: &column,
            window_indptr: &empty,
            window: &[],
            window_ids: &[],
            history_indptr: &empty,
            history: &[],
            before_indptr: &bindptr,
            before: full_before,
        };
        let ids = engram_ids(&engram, &rows_full, tokens, t).unwrap();
        let mut in_place = vec![0u8; nbytes];
        {
            let mut w = words(&mut in_place, nbytes).unwrap();
            engram_gather_host(&engram, &image_tables, &mut w, &ids, &hts, t).unwrap();
        }
        // prefetch into a contiguous layer-major ring, then copy to the image offsets
        let keep = (engram.n - 1).min(pos0 as usize);
        let short_before = &prompt[pos0 as usize - keep..pos0 as usize];
        let mut ring = vec![0u8; layers * per];
        engram_prefetch_gather(&engram, &hts, short_before, tokens, pos0, t, &mut ring).unwrap();
        let mut copied = vec![0u8; nbytes];
        for &(offset, e) in &image_tables.engram_emb {
            copied[offset..offset + per].copy_from_slice(&ring[e * per..e * per + per]);
        }
        assert_words_eq(&buf_words(&in_place), &buf_words(&copied), "ring copy vs in-place");
    }

    /// Isolated gather timing against the REAL checkpoint mmap tables (no device, no boot): the
    /// production 2048-column chunk, random rows (a fresh prompt's case), serial vs threaded, with
    /// a bit-exact check of every arm against the serial output on the real bytes. Ignored by
    /// default; run with `cargo test -p resin-gaudi -- --ignored engram_gather_real_table_bench`.
    #[test]
    #[ignore]
    fn engram_gather_real_table_bench() {
        let ckpt = std::env::var("DS41_CHECKPOINT").unwrap_or_else(|_| {
            "/home/nat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017\
             /snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
                .into()
        });
        let index = std::fs::read_to_string(format!("{ckpt}/model.safetensors.index.json"))
            .expect("checkpoint index");
        // the two Engram layers' shards, out of the weight map
        let shard_of = |layer: usize| -> String {
            let key = format!("\"layers.{layer}.engram.embed.weight\": \"");
            let at = index.find(&key).expect("engram weight map entry") + key.len();
            index[at..index[at..].find('"').unwrap() + at].to_string()
        };
        // a safetensors tensor's (rows, data start, data end) out of the shard header
        let tensor = |header: &str, name: &str| -> (usize, usize, usize) {
            let at = header.find(&format!("\"{name}\"")).expect(name);
            let rest = &header[at..];
            let int_after = |key: &str| -> usize {
                let s = &rest[rest.find(key).unwrap() + key.len()..];
                let end = s.find(|c: char| !c.is_ascii_digit()).unwrap();
                s[..end].parse().unwrap()
            };
            let rows = int_after("\"shape\":[");
            let key = "\"data_offsets\":[";
            let lo = int_after(key);
            let hi = {
                let s = &rest[rest.find(key).unwrap() + key.len()..];
                let comma = s.find(',').unwrap();
                let s = &s[comma + 1..];
                let end = s.find(|c: char| !c.is_ascii_digit()).unwrap();
                s[..end].parse::<usize>().unwrap()
            };
            (rows, lo, hi)
        };
        let mut maps = Vec::new();
        let mut hts: Vec<HostTable> = Vec::new();
        let mut rows_n = Vec::new();
        for layer in [1usize, 14] {
            let path = format!("{ckpt}/{}", shard_of(layer));
            let len = std::fs::metadata(&path).unwrap().len() as usize;
            let cpath = std::ffi::CString::new(path.clone()).unwrap();
            // SAFETY: a read-only shared mapping of the whole shard, kept alive in `maps` until the
            // test ends; the HostTable slices borrow it exactly as the runner's numpy views do.
            let (ptr, fd) = unsafe {
                let fd = libc::open(cpath.as_ptr(), libc::O_RDONLY);
                assert!(fd >= 0, "open {path}");
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd,
                    0,
                );
                assert_ne!(ptr, libc::MAP_FAILED, "mmap {path}");
                (ptr as *const u8, fd)
            };
            let raw = unsafe { std::slice::from_raw_parts(ptr, len) };
            let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
            let header = std::str::from_utf8(&raw[8..8 + hlen]).unwrap();
            let base = 8 + hlen;
            let name = format!("layers.{layer}.engram.embed");
            let (rows, wlo, whi) = tensor(header, &format!("{name}.weight"));
            let (_, slo, shi) = tensor(header, &format!("{name}.scale"));
            assert_eq!(whi - wlo, rows * 256);
            assert_eq!(shi - slo, rows * 8);
            // residency of the table's own pages (the gather's read term depends on it); mincore
            // needs a page-aligned range
            let page = 4096usize;
            let lo = (base + wlo.min(slo)) & !(page - 1);
            let hi = (base + whi.max(shi) + page - 1) & !(page - 1);
            let pages = (hi - lo) / page;
            let mut vec = vec![0u8; pages];
            let rc = unsafe {
                libc::mincore(
                    ptr.add(lo) as *mut libc::c_void,
                    hi - lo,
                    vec.as_mut_ptr(),
                )
            };
            let resident = if rc == 0 {
                vec.iter().filter(|&&v| v & 1 == 1).count() as f64 / pages as f64
            } else {
                -1.0
            };
            eprintln!(
                "layer {layer}: {rows} rows from {}, table pages {resident:.1}% resident",
                path.rsplit('/').next().unwrap()
            );
            hts.push(HostTable {
                weight: &raw[base + wlo..base + whi],
                scale: &raw[base + slo..base + shi],
            });
            rows_n.push(rows);
            maps.push((ptr, len, fd));
        }
        let (t, layers, width) = (2048usize, 2usize, 24usize);
        let per = t * 256 * width * 4;
        let tables = Tables {
            t,
            prefill: true,
            engram_emb: vec![(0, 0), (per, 1)],
            nbytes: 2 * per,
            ..Tables::default()
        };
        let engram = Engram {
            token_map: vec![],
            pad_id: 0,
            multipliers: vec![],
            n: 4,
            heads: 8,
            layers,
            primes: vec![],
            offsets: vec![],
        };
        // random rows: a fresh prompt's scattered ids (no n-gram repetition)
        let mut state = 0x1234_5678_9abc_def0u64;
        let ids: Vec<i64> = (0..t * layers * width)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % rows_n[0] as u64) as i64
            })
            .collect();
        let mut serial = vec![0u8; tables.nbytes];
        let mut arm = vec![0u8; tables.nbytes];
        let time = |buf: &mut [u8], f: &dyn Fn(&mut Words<'_>) -> Result<(), String>| -> f64 {
            let start = std::time::Instant::now();
            let mut w = words(buf, tables.nbytes).unwrap();
            f(&mut w).unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        for rep in 0..3 {
            let ms = time(&mut serial, &|w| {
                engram_gather_serial(&engram, &tables, w, &ids, &hts, t)
            });
            eprintln!("BENCH serial rep {rep}: {ms:.1} ms");
        }
        for threads in [2usize, 4, 8, 16] {
            for rep in 0..3 {
                let ms = time(&mut arm, &|w| {
                    engram_gather_threaded(&engram, &tables, w, &ids, &hts, t, threads)
                });
                eprintln!("BENCH threaded-{threads} rep {rep}: {ms:.1} ms");
            }
            assert_words_eq(
                &buf_words(&serial),
                &buf_words(&arm),
                &format!("real tables, threaded-{threads}"),
            );
        }
        for (ptr, len, fd) in maps {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, len);
                libc::close(fd);
            }
        }
    }
}
