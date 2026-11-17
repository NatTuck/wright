//! The MTP layer's side of a Qwen3.5/3.8 step, written into host copies of device table images. Ports of
//! `resin_models_gaudi/qwen3_5/model.py`: the tables of an MTP pass (`_mtp_pass`: the MTP layer over a step's
//! columns, its keys only or with its attention and head), of a verify step's commit (`_commit`) and of the draft
//! chain after it (`_chain`). The Python code was the reference: the tables are the same values. (That code is
//! deleted; `tests/test_qwen_native_mtp.py` keeps reference copies of its tables.)
//!
//! An MTP pass that runs with a target pass (a prefill chunk's keys, a plain decode step's keys, a verify step's
//! full pass, a prompt's first draft) has its tables in the target pass's image ([`PassLayout`], extras of the
//! mode's layout). The commit and the chain run after the host accepts the drafts, from an image of their own
//! ([`ChainLayout`]).

// Index loops over rows and layers keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

use crate::qwen::{Attention, Mode, Pages, SlotRun, kv_rows_table, slot_tables};
use crate::qwen_draft::{StateGeometry, StateRow};
use crate::sampling::PARAMS;
use crate::tables::{round, words};

/// The MTP layer's attention: its KV groups and query heads per group (a replicated MTP layer of a
/// tensor-parallel model keeps all of them).
#[derive(Clone, Copy, Debug)]
pub struct MtpAttention {
    pub kv_groups: i64,
    pub pages: Pages,
    pub hpg: i64,
    pub max_positions: i64,
}

/// Offsets of one MTP pass's tables in an image: the hidden column of each column (`mtp_cols`), the tokens
/// (`mtp_tok`, when the host gives them), the positions, the K and V rows, and for a full pass (attention and
/// head) the head's sampler inputs and the slot tables (`launches` of them, `stride` apart).
#[derive(Clone, Copy, Debug, Default)]
pub struct PassLayout {
    pub cols: usize,
    pub tok: usize,
    pub pos: usize,
    pub kv_rows: usize,
    pub params: usize,
    pub slots: usize,
    pub stride: usize,
    pub k_pages: usize,
    pub v_pages: usize,
}

impl PassLayout {
    /// The bytes of each table of a pass of `mode` (`cols`, `tok`, `pos`, `kv_rows`, `params`, `slots`); a keys
    /// pass (`full` false) takes no sampler inputs and no slot tables.
    pub fn sizes(mode: &Mode, full: bool) -> [usize; 6] {
        let t = mode.columns;
        let (_, _, stride) = Attention::launch_tables(mode.slots);
        [
            round(4 * t),
            round(4 * t),
            round(4 * t),
            round(4 * 8 * t),
            if full { round(4 * PARAMS * t) } else { 0 },
            if full { mode.max_launches * stride } else { 0 },
        ]
    }

    /// The layout from the tables' offsets (`sizes` order), for slot launches of `mode`.
    pub fn new(mode: &Mode, offsets: &[usize]) -> Self {
        let (k_pages, v_pages, stride) = Attention::launch_tables(mode.slots);
        Self {
            cols: offsets[0],
            tok: offsets[1],
            pos: offsets[2],
            kv_rows: offsets[3],
            params: offsets[4],
            slots: offsets[5],
            stride,
            k_pages,
            v_pages,
        }
    }

    /// the bytes from the image's start to the end of `launches` slot launches' tables
    pub fn upload_bytes(&self, launches: usize) -> usize {
        self.slots + launches * self.stride
    }
}

/// One MTP pass over a mode's columns: row i's columns sit at positions from `starts[i]` (a prefill's `real` of
/// them, else the mode's `per_row`), column c reads hidden column `cols[c]` of [prev | x] and token `tokens[c]`
/// (None: a device buffer), and a full pass samples with `params` (PARAMS a column; fewer leave the rest zero).
/// `blocks_indptr` delimits each row's MTP KV block rows in `blocks`.
#[derive(Clone, Copy)]
pub struct Pass<'a> {
    pub starts: &'a [i64],
    pub real: usize,
    pub blocks_indptr: &'a [i64],
    pub blocks: &'a [i64],
    pub cols: &'a [i64],
    pub tokens: Option<&'a [i64]>,
    pub params: Option<&'a [f32]>,
}

/// Write an MTP pass's tables (`_mtp_pass`) into `host`, which holds `nbytes` bytes from the image's start.
/// Returns the slot launches of a full pass (0 for a keys pass).
pub fn fill_pass(
    attention: &MtpAttention,
    mode: &Mode,
    layout: &PassLayout,
    host: &mut [u8],
    nbytes: usize,
    pass: &Pass<'_>,
) -> Result<usize, String> {
    let n = pass.starts.len();
    let (t, q) = (mode.columns, mode.per_row);
    if n == 0 || n > mode.rows() {
        return Err(format!(
            "{n} rows for an MTP pass of mode {} of {} rows",
            mode.name,
            mode.rows()
        ));
    }
    if pass.blocks_indptr.len() != n + 1
        || pass.blocks_indptr[0] != 0
        || pass.blocks_indptr.windows(2).any(|pair| pair[0] > pair[1])
        || pass.blocks_indptr[n] as usize != pass.blocks.len()
    {
        return Err("blocks_indptr does not delimit the rows".into());
    }
    if pass.cols.len() > t || pass.tokens.is_some_and(|tokens| tokens.len() > t) {
        return Err(format!("more than {t} columns for mode {}", mode.name));
    }
    if mode.prefill && pass.real > t {
        return Err(format!("{} columns of a {t}-column prefill", pass.real));
    }
    let mut w = words(host, nbytes)?;
    w.fill(layout.cols, t, 0);
    for (c, &col) in pass.cols.iter().enumerate() {
        w.set(layout.cols, c, col);
    }
    if let Some(tokens) = pass.tokens {
        w.fill(layout.tok, t, 0);
        for (c, &token) in tokens.iter().enumerate() {
            w.set(layout.tok, c, token);
        }
    }
    // positions: each row's columns from its start, the rest 0, all clamped to the rope table
    let count = if mode.prefill { pass.real } else { q };
    w.fill(layout.pos, t, 0);
    let mut runs = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..count {
            let pos = (pass.starts[i] + j as i64).min(attention.max_positions - 1);
            w.set(layout.pos, i * q + j, pos);
        }
        runs.push(SlotRun {
            column: i * q,
            pos0: pass.starts[i],
            real: count,
            blocks: &pass.blocks
                [pass.blocks_indptr[i] as usize..pass.blocks_indptr[i + 1] as usize],
        });
    }
    kv_rows_table(
        &mut w,
        layout.kv_rows,
        t,
        attention.kv_groups,
        attention.pages,
        &runs,
    )?;
    let Some(params) = pass.params else {
        return Ok(0);
    };
    if params.len() > PARAMS * t || !params.len().is_multiple_of(PARAMS) {
        return Err(format!(
            "{} sampler values for mode {}",
            params.len(),
            mode.name
        ));
    }
    w.fill(layout.params, PARAMS * t, 0);
    for (k, &value) in params.iter().enumerate() {
        w.0[layout.params / 4 + k] = value.to_bits() as i32;
    }
    let slots = Attention {
        kv_groups: attention.kv_groups,
        pages: attention.pages,
        hpg: attention.hpg,
        slots: mode.slots,
        query_block: mode.query_block,
        max_launches: mode.max_launches,
        at: layout.slots,
        stride: layout.stride,
        k_pages: layout.k_pages,
        v_pages: layout.v_pages,
        wide: None,
    };
    slot_tables(&mut w, &slots, &mode.name, &runs).map(|(_, launches)| launches)
}

/// A prefill chunk's hidden store (`_prefill_mtp`): its last real column `last` into the hidden slot of STATE
/// block `block` (-1: none), as the (2, 1) `hidden_table` at `at`.
pub fn fill_hidden(
    geometry: &StateGeometry,
    host: &mut [u8],
    nbytes: usize,
    at: usize,
    last: i64,
    block: i64,
) -> Result<(), String> {
    let mut w = words(host, nbytes)?;
    w.set(at, 0, last);
    w.set(at, 1, hidden(geometry, block));
    Ok(())
}

fn hidden(geometry: &StateGeometry, block: i64) -> i64 {
    if block < 0 {
        -1
    } else {
        block + geometry.hidden_offset
    }
}

/// Offsets of a verify step's commit tables (`commit_graph`): `commit_bases` (4, rows x layers), `commit_counts`
/// (rows x layers,) and `hidden_table` (2, rows), for `bucket` rows of `layers` GDN layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitLayout {
    pub bucket: usize,
    pub layers: usize,
    pub commit_bases: usize,
    pub commit_counts: usize,
    pub hidden_table: usize,
}

/// Write a verify step's commit (`_commit`): each row's first accepted + 1 tokens' GDN updates into the STATE
/// block of its last accepted position, and that column's residual into the block's hidden slot; padding rows
/// commit nothing. `width`: the step's columns a row.
pub fn fill_commit(
    layout: &CommitLayout,
    geometry: &StateGeometry,
    host: &mut [u8],
    nbytes: usize,
    rows: &[StateRow<'_>],
    accepted: &[i64],
    width: usize,
) -> Result<(), String> {
    let (bucket, layers) = (layout.bucket, layout.layers);
    if rows.is_empty() || rows.len() > bucket || accepted.len() != rows.len() {
        return Err(format!("{} rows in a commit of {bucket}", rows.len()));
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
    w.fill(layout.commit_bases, 4 * bucket * layers, -1);
    w.fill(layout.commit_counts, bucket * layers, 0);
    w.fill(layout.hidden_table, 2 * bucket, -1);
    for (b, row) in rows.iter().enumerate() {
        let m = accepted[b];
        if m < 0 || m as usize >= width {
            return Err(format!("row {b} accepted {m} drafts of {}", width - 1));
        }
        let dst = row.state_at(row.pos0 + m, g.state_tokens)?;
        for layer in 0..layers {
            let fields = [
                field(row.src, layer, true),
                field(row.src, layer, false),
                field(dst, layer, true),
                field(dst, layer, false),
            ];
            for (k, &value) in fields.iter().enumerate() {
                w.set(layout.commit_bases, 4 * (layer * bucket + b) + k, value);
            }
            w.set(layout.commit_counts, layer * bucket + b, m + 1);
        }
        w.set(layout.hidden_table, 2 * b, (b * width) as i64 + m);
        w.set(layout.hidden_table, 2 * b + 1, hidden(g, dst));
    }
    Ok(())
}

/// The image of a verify step's commit and the draft chain after it, for `bucket` rows of `width` columns (K + 1):
/// the commit tables, the next step's tokens [anchor, first draft, 0 ..] a row (`following`, (bucket, width)),
/// the chain's attention metadata [0, rows, 0, 0], and K - 1 links, each an MTP pass of the decode mode of
/// `bucket` rows (their fixed tables first, then each link's slot tables).
#[derive(Clone, Debug, Default)]
pub struct ChainLayout {
    pub bucket: usize,
    pub width: usize,
    pub commit: CommitLayout,
    pub following: usize,
    pub meta: usize,
    pub links: Vec<PassLayout>,
    /// the image's size with every link's every launch
    pub nbytes: usize,
}

impl ChainLayout {
    /// The layout after a verify step of `decode`'s rows (the chain's decode mode) at `width` columns a row.
    pub fn new(decode: &Mode, width: usize, layers: usize) -> Self {
        let bucket = decode.rows();
        let mut at = 0usize;
        let mut take = |nbytes: usize| {
            let offset = at;
            at += round(nbytes);
            offset
        };
        let commit = CommitLayout {
            bucket,
            layers,
            commit_bases: take(16 * bucket * layers),
            commit_counts: take(4 * bucket * layers),
            hidden_table: take(8 * bucket),
        };
        let following = take(4 * bucket * width);
        let meta = take(16);
        let sizes = PassLayout::sizes(decode, true);
        let fixed: Vec<[usize; 5]> = (1..width - 1)
            .map(|_| {
                [
                    take(sizes[0]),
                    take(sizes[1]),
                    take(sizes[2]),
                    take(sizes[3]),
                    take(sizes[4]),
                ]
            })
            .collect();
        let links: Vec<PassLayout> = fixed
            .iter()
            .map(|f| PassLayout::new(decode, &[f[0], f[1], f[2], f[3], f[4], take(sizes[5])]))
            .collect();
        Self {
            bucket,
            width,
            commit,
            following,
            meta,
            links,
            nbytes: at,
        }
    }
}

/// A verify step's accepted drafts, for its commit and chain: each row's state, first position, accepted drafts,
/// next anchor (its last emitted token), MTP draft at its last accepted column (`first`), and MTP KV block rows
/// (`blocks_indptr` delimits them); `params[j - 1]`: link j's sampler inputs (PARAMS a row).
pub struct Chain<'a> {
    pub rows: &'a [StateRow<'a>],
    pub accepted: &'a [i64],
    pub anchors: &'a [i64],
    pub first: &'a [i64],
    pub blocks_indptr: &'a [i64],
    pub blocks: &'a [i64],
    pub params: &'a [Vec<f32>],
}

/// A filled chain image: each link's slot launches and the bytes to upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainFilled {
    pub launches: Vec<usize>,
    pub nbytes: usize,
}

/// Write a verify step's commit and draft chain (`_commit`, the `following` tokens of `verify_finish` and
/// `_chain`): link j (1 .. K - 1) is the MTP layer at position anchor + j - 1 of each row (the anchor at pos0 +
/// accepted + 1), reading link j - 1's output (the first link: the verify pass's MTP output at the row's accepted
/// column) and its token (the first link: `first`).
pub fn fill_chain(
    attention: &MtpAttention,
    decode: &Mode,
    layout: &ChainLayout,
    geometry: &StateGeometry,
    host: &mut [u8],
    chain: &Chain<'_>,
) -> Result<ChainFilled, String> {
    let (bucket, width) = (layout.bucket, layout.width);
    let n = chain.rows.len();
    if [chain.accepted.len(), chain.anchors.len(), chain.first.len()] != [n; 3]
        || chain.params.len() != layout.links.len()
    {
        return Err("a chain takes one entry a row and sampler inputs a link".into());
    }
    let nbytes = layout.nbytes;
    fill_commit(
        &layout.commit,
        geometry,
        host,
        nbytes,
        chain.rows,
        chain.accepted,
        width,
    )?;
    {
        let mut w = words(host, nbytes)?;
        w.fill(layout.following, bucket * width, 0);
        for b in 0..n {
            w.set(layout.following, b * width, chain.anchors[b]);
            w.set(layout.following, b * width + 1, chain.first[b]);
        }
        for (k, value) in [0, n as i64, 0, 0].into_iter().enumerate() {
            w.set(layout.meta, k, value);
        }
    }
    let mut launches = Vec::with_capacity(layout.links.len());
    let mut first_cols = vec![0i64; bucket];
    let mut first_tokens = vec![0i64; bucket];
    for b in 0..n {
        first_cols[b] = (b * width) as i64 + chain.accepted[b] + 1;
        first_tokens[b] = chain.first[b];
    }
    let later_cols: Vec<i64> = (1..=bucket as i64).collect();
    for (i, link) in layout.links.iter().enumerate() {
        let j = i as i64 + 1;
        let starts: Vec<i64> = (0..n)
            .map(|b| chain.rows[b].pos0 + chain.accepted[b] + j)
            .collect();
        let pass = Pass {
            starts: &starts,
            real: 1,
            blocks_indptr: chain.blocks_indptr,
            blocks: chain.blocks,
            cols: if j == 1 { &first_cols } else { &later_cols },
            tokens: (j == 1).then_some(first_tokens.as_slice()),
            params: Some(&chain.params[i]),
        };
        launches.push(fill_pass(attention, decode, link, host, nbytes, &pass)?);
    }
    let upload = match (layout.links.last(), launches.last()) {
        (Some(link), Some(&last)) => link.upload_bytes(last),
        _ => layout.meta + round(16),
    };
    Ok(ChainFilled {
        launches,
        nbytes: upload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(rows: usize) -> Mode {
        Mode {
            name: format!("d{rows}"),
            columns: rows,
            per_row: 1,
            query_block: 1,
            slots: (4 * rows).max(16),
            prefill: false,
            max_launches: 8,
            wide_slots: 0,
            max_wide: 0,
            wide_full: false,
        }
    }

    #[test]
    fn a_chain_layout_puts_the_slot_tables_last() {
        let layout = ChainLayout::new(&decode(4), 4, 3);
        assert_eq!(layout.links.len(), 2);
        let (a, b) = (layout.links[0], layout.links[1]);
        assert!(a.params < b.cols && b.params < a.slots && a.slots < b.slots);
        assert_eq!(b.slots + 8 * b.stride, layout.nbytes);
        // K = 1: no links
        assert!(ChainLayout::new(&decode(4), 2, 3).links.is_empty());
    }

    #[test]
    fn the_commit_takes_each_row_s_accepted_tokens() {
        let layout = CommitLayout {
            bucket: 2,
            layers: 2,
            commit_bases: 0,
            commit_counts: 256,
            hidden_table: 512,
        };
        let geometry = StateGeometry {
            state_layer_rows: 1000,
            ssm_rows: 900,
            state_tokens: 4096,
            hidden_offset: 2000,
        };
        let mut host = vec![0u32; 192];
        let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), 768) };
        let blocks = [10_000i64, 20_000];
        // a row that crosses into the second STATE block at its last accepted position
        let rows = [StateRow {
            pos0: 4094,
            src: 10_000,
            blocks: &blocks,
            first: 0,
        }];
        fill_commit(&layout, &geometry, bytes, 768, &rows, &[2], 3).unwrap();
        let word = |i: usize| host[i] as i32;
        // layer 1, row 0: conv_src, ssm_src, conv_dst, ssm_dst
        let l1 = 4 * 2;
        assert_eq!(
            (word(l1), word(l1 + 1), word(l1 + 2), word(l1 + 3)),
            (11_900, 11_000, 21_900, 21_000)
        );
        assert_eq!(word(l1 + 4), -1); // the padding row
        assert_eq!((word(64), word(65), word(66), word(67)), (3, 0, 3, 0));
        assert_eq!((word(128), word(129), word(130)), (2, 22_000, -1));
    }
}
