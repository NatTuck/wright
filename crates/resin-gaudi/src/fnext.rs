//! Qwen3.8-Flash-Next's decode-step tables, written into one host copy of the device table image. This is the
//! port of `Qwen4ExpGaudi._fill_decode` (`resin_models_gaudi/qwen4_exp/model.py`) for decode shapes (one column
//! per row), byte for byte: the Python code stays the reference.

// Index loops over rows and layers keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

use crate::tables::Words;

pub const PAGE_TOKENS: i64 = 128;
pub const SLOT_KEYS: i64 = 2048;
pub const PARAMS: usize = 8;
const SLOT_PAGES: usize = (SLOT_KEYS / PAGE_TOKENS) as usize;

/// The host side of the PLE's n-gram hash: the multipliers of the n context positions, every head's prime table
/// size and first row, and this rank's slice of the rows (`table_first`, `table_rows` including the zero row).
#[derive(Clone, Debug)]
pub struct NGram {
    pub mult: Vec<u64>,
    pub sizes: Vec<u64>,
    pub offs: Vec<i64>,
    pub hpn: usize,
    pub eos: i64,
    pub table_first: i64,
    pub table_rows: i64,
}

impl NGram {
    fn n(&self) -> usize {
        self.mult.len()
    }

    fn heads(&self) -> usize {
        self.sizes.len()
    }

    /// This rank's table rows of a token after `history` (the n - 1 tokens before it, oldest first; -1: none,
    /// EOS stands in): the global rows made local, the other ranks' rows and the padding being the zero row.
    fn rows(&self, token: i64, history: &[i64], out: &mut [i64]) {
        let n = self.n();
        let zero = (self.table_rows - 1).max(0);
        // ctx[0] = the token, ctx[s] = the s-th predecessor, EOS at and behind an EOS
        let mut ctx = vec![0i64; n];
        ctx[0] = token;
        let mut cut = false;
        for s in 1..n {
            let p = history[history.len() - s];
            let p = if p < 0 { self.eos } else { p };
            cut |= p == self.eos;
            ctx[s] = if cut { self.eos } else { p };
        }
        let mut mixed = (ctx[0] as u64).wrapping_mul(self.mult[0]);
        for g in 2..=n {
            mixed ^= (ctx[g - 1] as u64).wrapping_mul(self.mult[g - 1]);
            let h0 = (g - 2) * self.hpn;
            for h in h0..h0 + self.hpn {
                let row = (mixed % self.sizes[h]) as i64 + self.offs[h];
                let local = row - self.table_first;
                out[h] = if local >= 0 && local < zero {
                    local
                } else {
                    zero
                };
            }
        }
    }
}

/// What every step's tables depend on besides the step itself.
#[derive(Clone, Debug)]
pub struct Model {
    pub linear_layers: usize,
    pub full_layers: usize,
    pub kv_groups: i64,
    pub hpg: i64,
    pub kv_layer_rows: i64,
    pub state_layer_rows: i64,
    pub ssm_rows: i64,
    /// the PLE conv state's row within a STATE block (None: no PLE layer)
    pub ple_offset: Option<i64>,
    pub max_positions: i64,
    /// the PLE's n-gram heads (the rows a column takes)
    pub ple_heads: usize,
    pub ngram: Option<NGram>,
}

/// A per-layer table: its first row's byte offset and the stride between layers' rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct Strided {
    pub offset: usize,
    pub stride: usize,
}

impl Strided {
    /// layer `layer`'s row of a per-layer table
    fn at(self, layer: usize) -> usize {
        self.offset + layer * self.stride
    }
}

/// Byte offsets of one decode shape's tables in a host copy.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    pub t: usize,
    pub nbytes: usize,
    pub slots: usize,
    pub tok: usize,
    pub pos: usize,
    pub meta: usize,
    pub cols: usize,
    pub samp_params: usize,
    pub prow: Option<usize>,
    pub ple_src: Option<usize>,
    pub ple_dst: Option<usize>,
    pub gdn_bases: Strided,
    pub conv_dst: Strided,
    pub kv_rows: Strided,
    pub ix_rows: Strided,
    pub kb_rows: Strided,
    pub kb_tail: Strided,
    pub slot_table: Strided,
    pub k_pages: Strided,
    pub v_pages: Strided,
}

/// One step's rows (sequences), flat: per row its position, its STATE block rows (-1: fresh), its KV block rows
/// (`blocks_indptr` delimits each row's part), its token and the two tokens before it (-1: none).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub pos0: &'a [i64],
    pub state_src: &'a [i64],
    pub state_dst: &'a [i64],
    pub blocks_indptr: &'a [i64],
    pub blocks: &'a [i64],
    pub token: &'a [i64],
    pub history: &'a [i64],
}

impl Rows<'_> {
    fn len(&self) -> usize {
        self.pos0.len()
    }

    fn blocks(&self, row: usize) -> &[i64] {
        &self.blocks[self.blocks_indptr[row] as usize..self.blocks_indptr[row + 1] as usize]
    }

    fn validate(&self, t: usize) -> Result<(), String> {
        let rows = self.len();
        if rows == 0 || rows > t {
            return Err(format!("{rows} rows for a step of {t} columns"));
        }
        for (name, len) in [
            ("state_src", self.state_src.len()),
            ("state_dst", self.state_dst.len()),
            ("token", self.token.len()),
        ] {
            if len != rows {
                return Err(format!("{name} has {len} entries for {rows} rows"));
            }
        }
        if self.history.len() != 2 * rows {
            return Err("history holds two tokens a row".into());
        }
        if self.blocks_indptr.len() != rows + 1
            || self.blocks_indptr[0] != 0
            || self.blocks_indptr.windows(2).any(|pair| pair[0] > pair[1])
            || self.blocks_indptr[rows] as usize != self.blocks.len()
        {
            return Err("blocks_indptr does not delimit the rows".into());
        }
        Ok(())
    }
}

/// Stored slot of logical index p within a page (`sigma_inv`).
fn sigma_inv(p: i64) -> i64 {
    let (block, r) = (p & !127, p & 127);
    block + if r < 64 { 2 * r } else { 2 * (r - 64) + 1 }
}

/// Write a decode step's tables into `host` (a host copy of at least `tables.nbytes` bytes, 4-byte aligned).
/// `samp` holds every row's PARAMS sampler values.
pub fn fill(
    model: &Model,
    tables: &Tables,
    host: &mut [u8],
    rows: &Rows<'_>,
    samp: &[f32],
) -> Result<(), String> {
    let t = tables.t;
    rows.validate(t)?;
    let n = rows.len();
    if samp.len() != n * PARAMS {
        return Err(format!("{} sampler values for {n} rows", samp.len()));
    }
    if host.len() < tables.nbytes
        || !(host.as_ptr() as usize).is_multiple_of(4)
        || !tables.nbytes.is_multiple_of(4)
    {
        return Err("the host copy is too small or misaligned".into());
    }
    // SAFETY: the buffer is 4-byte aligned and holds nbytes bytes; every table write stays inside it.
    let words = unsafe {
        std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<i32>(), tables.nbytes / 4)
    };
    let mut w = Words(words);
    let (nl, nf, g) = (model.linear_layers, model.full_layers, model.kv_groups);
    let page = PAGE_TOKENS;

    // ---- the columns
    w.fill(tables.tok, t, 0);
    w.fill(tables.pos, t, 0);
    for b in 0..n {
        w.set(tables.tok, b, rows.token[b]);
        w.set(tables.pos, b, rows.pos0[b]);
    }
    for (k, value) in [0, n as i64, 0, 0].into_iter().enumerate() {
        w.set(tables.meta, k, value);
    }
    w.fill(tables.cols, t, n as i64 - 1);
    w.fill(tables.samp_params, PARAMS * t, 0);
    for (k, &value) in samp.iter().enumerate() {
        w.set_f32(tables.samp_params, k, value);
    }
    // ---- the PLE: every column's n-gram rows and the conv state's rows
    if let Some(prow) = tables.prow {
        let ngram = model.ngram.as_ref();
        let heads = model.ple_heads;
        let zero = ngram.map(|ng| (ng.table_rows - 1).max(0)).unwrap_or(0);
        w.fill(prow, heads * t, zero);
        if let Some(ng) = ngram.filter(|ng| ng.table_rows > 0 && ng.heads() == heads) {
            let mut out = vec![0i64; heads];
            for b in 0..n {
                ng.rows(rows.token[b], &rows.history[2 * b..2 * b + 2], &mut out);
                for (h, &row) in out.iter().enumerate() {
                    w.set(prow, b * heads + h, row);
                }
            }
        }
    }
    if let (Some(src), Some(dst), Some(offset)) = (tables.ple_src, tables.ple_dst, model.ple_offset)
    {
        w.fill(src, t, -1);
        w.fill(dst, t, -1);
        let ple_row = |block: i64| if block < 0 { -1 } else { block + offset };
        for b in 0..n {
            w.set(src, b, ple_row(rows.state_src[b]));
            w.set(dst, b, ple_row(rows.state_dst[b]));
        }
    }
    // ---- delta-net states: [conv_src, ssm_src, conv_dst, ssm_dst] a column, every layer a stride further
    for nth in 0..nl {
        let bases = tables.gdn_bases.at(nth);
        let conv = tables.conv_dst.at(nth);
        w.fill(bases, 4 * t, -1);
        w.fill(conv, t, -1);
        let shift = nth as i64 * model.state_layer_rows;
        for b in 0..n {
            let (src, dst) = (rows.state_src[b], rows.state_dst[b]);
            let base = [
                if src >= 0 {
                    src + model.ssm_rows + shift
                } else {
                    -1
                },
                if src >= 0 { src + shift } else { -1 },
                dst + model.ssm_rows + shift,
                dst + shift,
            ];
            for (k, &value) in base.iter().enumerate() {
                w.set(bases, 4 * b + k, value);
            }
            w.set(conv, b, base[2]);
        }
    }
    if nf == 0 {
        return Ok(());
    }
    // ---- attention: each row's rows at its position and its slots, from the first attention layer's pages
    let (raw_at, kb_at) = (2 * g * page, (2 * g + 1) * page);
    let s_count = tables.slots;
    let chunks = (model.max_positions + SLOT_KEYS - 1) / SLOT_KEYS;
    let block_of = |b: usize, index: i64| -> Result<i64, String> {
        let blocks = rows.blocks(b);
        blocks.get(index as usize).copied().ok_or_else(|| {
            format!(
                "row {b} has no KV block {index} for position {}",
                rows.pos0[b]
            )
        })
    };
    for a in 0..nf {
        let shift = a as i64 * model.kv_layer_rows;
        let shifted = |row: i64| if row >= 0 { row + shift } else { -1 };
        let kv = tables.kv_rows.at(a);
        let ix = tables.ix_rows.at(a);
        let kb = tables.kb_rows.at(a);
        let tail = tables.kb_tail.at(a);
        let st = tables.slot_table.at(a);
        let kp = tables.k_pages.at(a);
        let vp = tables.v_pages.at(a);
        w.fill(kv, 8 * t, -1);
        w.fill(ix, t, -1);
        w.fill(kb, 4 * t, -1);
        w.fill(tail, t, -1);
        w.fill(st, 8 * s_count, -1);
        w.fill(kp, SLOT_PAGES * s_count, -1);
        w.fill(vp, SLOT_PAGES * s_count, -1);
        for b in 0..n {
            let pos = rows.pos0[b];
            let block = block_of(b, pos / page)?;
            let (off, soff) = (pos % page, sigma_inv(pos % page));
            for grp in 0..g {
                w.set(kv, 8 * b + grp as usize, shifted(block + grp * page + off));
                w.set(
                    kv,
                    8 * b + 4 + grp as usize,
                    shifted(block + g * page + grp * page + soff),
                );
            }
            w.set(ix, b, shifted(block + raw_at + off));
            let first = 4 * (pos / 4);
            let block_f = block_of(b, first / page)?;
            w.set(tail, b, shifted(block_f + raw_at + first % page));
            for k in 0..4 {
                w.set(
                    kb,
                    4 * b + k,
                    shifted(block_f + kb_at + first % page + k as i64),
                );
            }
        }
        // slots: row b, key chunk c, group grp -> [grp, hpg b, pos, 2048 c, pos + 1, b, 0, 0], chunk-major
        let mut s = 0usize;
        for c in 0..chunks {
            for grp in 0..g {
                for b in 0..n {
                    let pos = rows.pos0[b];
                    if pos < c * SLOT_KEYS {
                        continue;
                    }
                    if s >= s_count {
                        return Err(format!("more than {s_count} slots"));
                    }
                    let fields = [
                        grp,
                        model.hpg * b as i64,
                        pos,
                        c * SLOT_KEYS,
                        pos + 1,
                        b as i64,
                        0,
                        0,
                    ];
                    for (k, &value) in fields.iter().enumerate() {
                        w.set(st, 8 * s + k, value);
                    }
                    let blocks = rows.blocks(b);
                    let lo = (c as usize) * SLOT_PAGES;
                    for j in 0..SLOT_PAGES {
                        if let Some(&block) = blocks.get(lo + j).filter(|&&block| block >= 0) {
                            w.set(kp, SLOT_PAGES * s + j, block + grp * page + shift);
                            w.set(
                                vp,
                                SLOT_PAGES * s + j,
                                block + g * page + grp * page + shift,
                            );
                        }
                    }
                    s += 1;
                }
            }
        }
    }
    Ok(())
}
