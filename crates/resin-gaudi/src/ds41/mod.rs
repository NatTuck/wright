//! DeepSeek-V4.1's step tables, written into one host copy of the device table buffer: the port of `_fill_positions`
//! and `_fill_tokens` (`resin_models_gaudi/deepseek_v41/model.py`) for decode and prefill shapes, byte for byte
//! (the numpy code is the reference while it exists).

// Index loops over columns and rows keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

pub mod context;

use crate::tables::{Words, words};

pub const PAGE_TOKENS: i64 = 128;
pub const HISTORY_TOKENS: i64 = 256;
pub const PAGE_ROWS: i64 = 129;
pub const INDEX_ROWS: i64 = 33;
pub const PAIR_ROWS: i64 = PAGE_ROWS + INDEX_ROWS;
pub const INDEX_KEYS: i64 = 2048;
/// pending-state rows per (WINDOW block, ratio-2 owner): slot (position / 2) % PENDING_SLOTS (cache.PENDING_SLOTS)
pub const PENDING_SLOTS: i64 = 4;
const LANES: usize = (INDEX_KEYS / PAGE_TOKENS) as usize;
const ROPE: usize = 128;

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
    /// per candidate launch
    pub ixc: Vec<usize>,
    pub ix_vis: [usize; 2],
    pub cmp_blocks: [usize; 2],
    /// per ratio and launch
    pub ix: [Vec<usize>; 2],
    /// per ratio, launch and owner of that ratio (in owner order)
    pub ix_owner: [Vec<Vec<usize>>; 2],
    /// per Engram layer of a step that hashes on the device: (hi table, lo table, its index among the
    /// checkpoint's Engram layers)
    pub engram: Vec<(usize, usize, usize)>,
    /// per Engram layer of a step whose hash ids come from the host: (ids table, its Engram index)
    pub engram_ids: Vec<(usize, usize)>,
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
    pub ix: [usize; 2],
    /// per ratio: a prefill step's prefiltered selection, its dense attention
    pub pf: [bool; 2],
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
    for &(lo, hi) in &tables.zero {
        host[lo..hi].fill(0);
    }
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
            for (lane, &value) in cs.iter().enumerate() {
                w.set_f32(tables.cs[r], column * ROPE + lane, value);
                w.set_f32(
                    tables.cs_inv[r],
                    column * ROPE + lane,
                    if lane < 64 { value } else { -value },
                );
            }
        }
    }

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
    for (&lat, &offset) in tables.lat.iter().zip(&model.layer_rows) {
        for (index, &value) in lat0.iter().enumerate() {
            w.set(lat, index, if value >= 0 { value + offset } else { -1 });
        }
    }
    for (&table, &offset) in tables.rows.iter().zip(&model.layer_rows) {
        w.fill(table, t * 8, -1);
        for &c in &cols {
            let slot = pos[c] % page;
            w.set(table, c * 8, win[c] + slot + offset);
            w.set(table, c * 8 + 3, win[c] + page + offset);
            w.set(table, c * 8 + 4, slot);
        }
    }

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
            let row = padded(history(k), history_width, p / owner.pages, -1)?
                + owner.offset
                + PAIR_ROWS * (p % owner.pages);
            let e = j % page;
            w.set(store, g * 8, row + e);
            w.set(store, g * 8 + 2, 4 * (row + PAGE_ROWS) + e);
            w.set(store, g * 8 + 5, row + PAGE_ROWS + INDEX_ROWS - 1);
            w.set(store, g * 8 + 6, e);
            w.set(store, g * 8 + 3, row + page);
            w.set(store, g * 8 + 4, e);
        }
    }

    // indexer: visible entries, compressed pages, index launches
    let mut counts = Counts {
        sparse: false,
        cand: true,
        ..Counts::default()
    };
    for (ri, ratio) in [(0usize, 1i64), (1usize, 2i64)] {
        let vis: Vec<i64> = (0..t)
            .map(|c| if live(c) { (pos[c] + 1) / ratio } else { 0 })
            .collect();
        for (c, &value) in vis.iter().enumerate() {
            w.set(tables.ix_vis[ri], c, value);
        }
        let most = vis.iter().copied().max().unwrap_or(0);
        counts.radix[ri] = !tables.prefill && most > model.radix_min;
        let visb = |w: &mut Words<'_>, table: usize| {
            for (c, &value) in vis.iter().enumerate() {
                let blocks = (value + 7) / 8;
                w.set(table, c, ((blocks + 63) / 64 * 64).min(model.kmax / 8));
            }
        };
        // a prefill step whose every column sees many entries: blocks of 8 entries, in whole groups of 64 for
        // ix_thr, and the packed entries
        counts.pf[ri] = tables.prefill
            && model.prefilter_blocks > 0
            && cols
                .iter()
                .all(|&c| vis[c] > model.prefilter_min.max(8 * model.prefilter_blocks));
        if counts.pf[ri] {
            let candidate_ratio = model.candidate.is_none_or(|(r, _)| r == ratio);
            visb(
                &mut w,
                if candidate_ratio {
                    tables.ix_visb
                } else {
                    tables.ix_visb2
                },
            );
            for c in 0..t {
                w.set(
                    tables.ix_visc,
                    c,
                    if live(c) {
                        8 * model.prefilter_blocks
                    } else {
                        0
                    },
                );
            }
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
        if let Some((candidate_ratio, topk)) = model.candidate
            && candidate_ratio == ratio
        {
            visb(&mut w, tables.ix_visb);
            counts.cand = most > topk * 8;
            let sparse = !tables.ixc.is_empty() && cols.iter().all(|&c| vis[c] > model.kc);
            counts.sparse = sparse;
            if sparse {
                for (c, _) in vis.iter().enumerate() {
                    w.set(tables.ix_visc, c, if live(c) { model.kc } else { 0 });
                }
                let chunks = (model.kc / INDEX_KEYS) as usize;
                let total = tables.ixc.len() * tables.ix_slots;
                let records = cols.len() * chunks;
                if records > total {
                    return Err("a step's candidate launches exceed the table capacity".into());
                }
                let mut table = vec![[0i64; 8]; total];
                for row in &mut table {
                    row[0] = -1;
                }
                for (i, &c) in cols.iter().enumerate() {
                    for chunk in 0..chunks {
                        table[i * chunks + chunk] = [
                            0,
                            model.index_heads * c as i64,
                            model.kc - 1,
                            INDEX_KEYS * chunk as i64,
                            model.kc,
                            c as i64,
                            1,
                            0,
                        ];
                    }
                }
                counts.ixc = records.div_ceil(tables.ix_slots);
                for (j, &launch) in tables.ixc.iter().enumerate() {
                    for (s, row) in table[j * tables.ix_slots..(j + 1) * tables.ix_slots]
                        .iter()
                        .enumerate()
                    {
                        for (lane, &value) in row.iter().enumerate() {
                            w.set(launch, s * 8 + lane, value);
                        }
                    }
                }
            }
        }
        // each live column's HISTORY rows, per page of this ratio
        let per = (HISTORY_TOKENS / ratio / page) as usize;
        let blocks = tables.cmp_blocks[ri];
        for c in 0..t {
            if !live(c) {
                w.fill(blocks + c * model.pages * 4, model.pages, -1);
                continue;
            }
            let k = col_row[c] as usize;
            let hist = history(k);
            for i in 0..model.pages {
                let value = padded(hist, history_width, (i / per) as i64, -1)?;
                w.set(
                    blocks,
                    c * model.pages + i,
                    if value >= 0 {
                        value + PAIR_ROWS * (i % per) as i64
                    } else {
                        -1
                    },
                );
            }
        }
        // index launches: per query column, its keys in chunks of INDEX_KEYS
        let mut records: Vec<(usize, i64, i64, i64, i64)> = Vec::new();
        for k in 0..r_count {
            let (pos0, real) = (rows.pos0[k], rows.real[k]);
            let seen_all = (pos0 + real) / ratio;
            for b0 in (0..real).step_by(qb) {
                let q = pos0 + b0;
                let seen = ((q + qb as i64) / ratio).min(seen_all);
                let mut key = 0;
                while key < seen {
                    records.push((k, b0, q, key, seen_all));
                    key += INDEX_KEYS;
                }
            }
        }
        let launches = records.len().div_ceil(tables.ix_slots);
        if launches > tables.ix[ri].len() {
            return Err("a step's index launches exceed the table capacity".into());
        }
        counts.ix[ri] = launches;
        if records.is_empty() {
            continue;
        }
        let owners: Vec<&Owner> = model.owners.iter().filter(|o| o.ratio == ratio).collect();
        for j in 0..launches {
            let launch = tables.ix[ri][j];
            for s in 0..tables.ix_slots {
                let Some(&(k, b0, q, key, len)) = records.get(j * tables.ix_slots + s) else {
                    w.set(launch, s * 8, -1);
                    for lane in 1..8 {
                        w.set(launch, s * 8 + lane, 0);
                    }
                    for &pages in &tables.ix_owner[ri][j] {
                        w.fill(pages + s * LANES * 4, LANES, -1);
                    }
                    continue;
                };
                let col = rows.column[k] + b0;
                let values = [0, model.index_heads * col, q, key, len, col, ratio, 0];
                for (lane, value) in values.into_iter().enumerate() {
                    w.set(launch, s * 8 + lane, value);
                }
                let valid = (LANES as i64).min((len - key + page - 1) / page);
                for (owner, &pages) in owners.iter().zip(&tables.ix_owner[ri][j]) {
                    for lane in 0..LANES {
                        let p = key / page + lane as i64;
                        let value = if (lane as i64) < valid {
                            padded(history(k), history_width, p / owner.pages, -1)?
                                + owner.offset
                                + PAIR_ROWS * (p % owner.pages)
                                + PAGE_ROWS
                        } else {
                            -1
                        };
                        w.set(pages, s * LANES + lane, value);
                    }
                }
            }
        }
    }

    if with_tokens {
        fill_tokens(model, tables, host, rows, n, tokens)?;
    }
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
    Ok(())
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
            for (lane, &value) in cs.iter().enumerate() {
                w.set_f32(tables.cs[r], column * ROPE + lane, value);
                w.set_f32(
                    tables.cs_inv[r],
                    column * ROPE + lane,
                    if lane < 64 { value } else { -value },
                );
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
        );
        assert!(error.is_err());
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
}
