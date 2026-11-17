//! GLM-5.3-Flash's step tables, written into one host copy of the device table buffer: the port of `_launch`'s
//! table writes, `_attention_tables`, `_store_rows`, `_index_launches`, `plan_slots` and `_kda_tables`
//! (`resin_models_gaudi/glm5_3_flash/model.py`), byte for byte (the numpy code is the reference while it exists).
//!
//! A step's tables (`_Tables`): a header (tokens, meta, the head's columns, per column visibility and key
//! length), per DSA layer the key-store and pool-store rows, the KDA records, the sparse block table, then the
//! slot and index records ordered by launch first, so a step uploads only the launches it uses. The MTP pass uses
//! the same tables with one DSA layer at offset 0 and no KDA layer.

// Index loops over columns and rows keep the code parallel to the numpy reference it must match.
#![allow(clippy::needless_range_loop)]

use crate::tables::{Words, words};

pub const PAGE_TOKENS: i64 = 128;
/// keys per pool of the DSA indexer
pub const POOL: i64 = 4;
/// pools of one KV block's index page
pub const PAGE_POOLS: i64 = 32;
/// pools per indexer slot (64 index pages)
pub const SLOT_POOLS: i64 = 2048;
/// keys per attention slot (16 pages)
pub const SLOT_KEYS: i64 = 2048;
pub const LATENT_PAGE_ROWS: i64 = 129;
pub const INDEX_PAGE_ROWS: i64 = 16;
/// a layer's tail rows, from its latent page's first row
pub const TAIL_OFFSET: i64 = LATENT_PAGE_ROWS + INDEX_PAGE_ROWS;
pub const ROWS_FIELDS: usize = 8;
/// a prefill chunk's head columns
pub const HEAD_COLS: usize = 16;
/// a slot record's slots, (S <= 16, 8) i32
pub const SLOTS_BYTES: usize = 512;
/// a slot record's latent pages, (S <= 16, 16) i32
pub const PAGES_BYTES: usize = 1024;
pub const SLOT_RECORD: usize = SLOTS_BYTES + PAGES_BYTES;
/// an index record: slots, then (S <= 16, 64) index pages
pub const INDEX_RECORD: usize = SLOTS_BYTES + 4 * PAGES_BYTES;
const KEY_LANES: usize = (SLOT_KEYS / PAGE_TOKENS) as usize;
const INDEX_LANES: usize = (SLOT_POOLS / PAGE_POOLS) as usize;

/// What every step's tables depend on besides the step itself.
#[derive(Clone, Debug)]
pub struct Model {
    /// MLA heads of the rank (a slot's query column stride)
    pub mla_heads: i64,
    pub index_heads: i64,
    /// pools a column attends to once the indexer runs
    pub top_pools: i64,
    /// consecutive pools whose tails a layer keeps apart (`cache.tail_rows`)
    pub tail_pools: i64,
    /// the sparse block table's width (`Glm53Gaudi.pages`)
    pub pages: usize,
    /// a STATE block's rows of one KDA layer, and of its recurrent state (the conv state follows)
    pub state_layer_rows: i64,
    pub ssm_rows: i64,
    /// decode steps have at most this many columns; wider steps are prefill chunks
    pub max_t: usize,
}

/// Byte offsets of one shape's tables in a host copy (`_Tables`), and its launch capacity.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    pub t: usize,
    pub tok: usize,
    pub meta: usize,
    pub head: usize,
    pub vis: usize,
    pub klen: usize,
    pub header: usize,
    pub rows_bytes: usize,
    pub store: usize,
    pub pool: usize,
    pub kda: usize,
    pub bases_bytes: usize,
    pub kda_record: usize,
    pub blocks: usize,
    pub slot_base: usize,
    pub index_base: usize,
    pub key_launches: usize,
    pub index_launches: usize,
    pub nbytes: usize,
    /// DSA layers of the tables (1 for the MTP pass)
    pub dsa: usize,
    /// KDA layers of the tables (0 for the MTP pass)
    pub kda_layers: usize,
    /// each DSA layer's page offset in a KV block (the MTP pass: [0])
    pub offsets: Vec<i64>,
}

impl Tables {
    fn slot(&self, launch: usize, nth: usize) -> usize {
        self.slot_base + (launch * self.dsa + nth) * SLOT_RECORD
    }

    fn index(&self, launch: usize, nth: usize) -> usize {
        self.index_base + (launch * self.dsa + nth) * INDEX_RECORD
    }
}

/// A step's rows, flat: per row its first column, first position, real tokens, STATE blocks (`state_src` holds
/// the state after pos0 - 1, -1 fresh; `state_dst` receives the state after the step's last token), and its KV
/// blocks' first arena rows (`blocks_indptr`).
pub struct Rows<'a> {
    pub column: &'a [i64],
    pub pos0: &'a [i64],
    pub real: &'a [i64],
    pub state_src: &'a [i64],
    pub state_dst: &'a [i64],
    pub blocks_indptr: &'a [i64],
    pub blocks: &'a [i64],
}

impl Rows<'_> {
    fn len(&self) -> usize {
        self.column.len()
    }

    fn blocks(&self, r: usize) -> &[i64] {
        &self.blocks[self.blocks_indptr[r] as usize..self.blocks_indptr[r + 1] as usize]
    }

    fn block(&self, r: usize, position: i64) -> Result<i64, String> {
        self.blocks(r)
            .get((position / PAGE_TOKENS) as usize)
            .copied()
            .ok_or_else(|| format!("row {r} has no KV block for position {position}"))
    }
}

/// What the step is: `columns` real columns (tokens of the step, None: a decode continuation's, on the device),
/// whose first position goes into the meta table.
pub struct Step<'a> {
    pub tokens: Option<&'a [i64]>,
    pub columns: usize,
    pub first_position: i64,
    /// write the head's columns (a prefill chunk's last real column; the MTP pass has no head table)
    pub head: bool,
    /// a verify step's columns per row (0: none): KDA records by row, every pool tail kept
    pub verify: usize,
    /// keep every tail (`keep_tails`: a verify step, and every MTP pass)
    pub keep_tails: bool,
    /// the meta table's real columns (default: `columns`; an MTP pass: its rows' real tokens)
    pub meta_columns: Option<usize>,
}

/// What a fill decided: the program's shape and launch counts, and the bytes to upload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// some column sees more pools than the indexer keeps
    pub index: bool,
    /// a decode step with the indexer on attends over the sparse block table
    pub sparse: bool,
    /// slot launches of each DSA layer (1 when sparse)
    pub slots: usize,
    /// index slot launches of each DSA layer
    pub index_launches: usize,
    /// the uploads: [0, blocks), the block table, the slot records, the index records (offset, bytes; 0: none)
    pub uploads: [(usize, usize); 4],
}

fn tail_rows(position: i64, pools: i64) -> i64 {
    let p = position.rem_euclid(pools * POOL);
    2 * (p - p / POOL)
}

/// Per column, a key store [latent row, scale row, scale index] and a pool store [pool unit, src0, src1, src2,
/// tail row].
type StoreRows = (Vec<[i64; 3]>, Vec<[i64; 5]>);

/// `_store_rows`: per column, relative to the layer's pages, `store` [latent row, scale row, scale index] and
/// `pool` [pool unit, src0, src1, src2, tail row] (-1 or the field's default for padding columns).
fn store_rows(
    t: usize,
    rows: &Rows<'_>,
    keep_tails: bool,
    tail_pools: i64,
) -> Result<StoreRows, String> {
    let mut store = vec![[-1i64; 3]; t];
    // sources default to column 0: a column completing no pool never reads them
    let mut pool = vec![[-1i64, 0, 0, 0, -1]; t];
    for r in 0..rows.len() {
        let (column, pos0, real) = (rows.column[r], rows.pos0[r], rows.real[r]);
        let last = pos0 + real - 1;
        for j in 0..real {
            let pos = pos0 + j;
            let col = column + j;
            let c = col as usize;
            if c >= t {
                return Err(format!(
                    "row {r}'s column {col} is past the step's {t} columns"
                ));
            }
            let b = rows.block(r, pos)?;
            let slot = pos % PAGE_TOKENS;
            store[c] = [b + slot, b + PAGE_TOKENS, slot];
            let p = pos / POOL;
            if pos % POOL == POOL - 1 {
                pool[c][0] = 2 * (b + LATENT_PAGE_ROWS) + p % PAGE_POOLS;
                for s in 0..3 {
                    let q = pos - 3 + s as i64;
                    pool[c][1 + s] = if q >= pos0 {
                        col - 3 + s as i64
                    } else {
                        -(b + TAIL_OFFSET + tail_rows(q, tail_pools) + 1)
                    };
                }
            } else if keep_tails || p * POOL + POOL - 1 > last {
                // the pool of an unfinished token: its last token is past this launch, so a later launch reads
                // it. A verify step keeps every tail: a rejected draft restarts the next step inside any pool
                pool[c][4] = b + TAIL_OFFSET + tail_rows(pos, tail_pools);
            }
        }
    }
    Ok((store, pool))
}

/// One slot (attention or index) of a launch: its 8 fields and its pages (-1: none).
struct Slot {
    fields: [i64; 8],
    pages: Vec<i64>,
}

/// `plan_slots` for one KV group: every (row, query block, key chunk) whose keys a query of the block can see.
fn key_slots(rows: &Rows<'_>, query_block: i64, heads: i64) -> Result<Vec<Slot>, String> {
    let mut out = Vec::new();
    for r in 0..rows.len() {
        let (column, pos0, real) = (rows.column[r], rows.pos0[r], rows.real[r]);
        let blocks = rows.blocks(r);
        let klen = pos0 + real;
        let mut t0 = 0;
        while t0 < real {
            let visible = (pos0 + t0 + query_block).min(klen);
            let chunks = (visible + SLOT_KEYS - 1) / SLOT_KEYS;
            for chunk in 0..chunks {
                let key0 = chunk * SLOT_KEYS;
                let first = (key0 / PAGE_TOKENS) as usize;
                let live = (KEY_LANES as i64).min((visible - key0 + PAGE_TOKENS - 1) / PAGE_TOKENS)
                    as usize;
                if blocks.len() < first + live {
                    return Err(format!(
                        "row {r} has {} KV blocks, its slots need {}",
                        blocks.len(),
                        first + live
                    ));
                }
                let mut pages = vec![-1i64; KEY_LANES];
                pages[..live].copy_from_slice(&blocks[first..first + live]);
                out.push(Slot {
                    fields: [
                        0,
                        heads * (column + t0),
                        pos0 + t0,
                        key0,
                        klen,
                        column + t0,
                        0,
                        0,
                    ],
                    pages,
                });
            }
            t0 += query_block;
        }
    }
    Ok(out)
}

/// `_index_launches`' slots: per row and query block, every 2048-pool chunk of the pools it may see.
fn index_slots(rows: &Rows<'_>, query_block: i64, index_heads: i64) -> Result<Vec<Slot>, String> {
    let mut out = Vec::new();
    for r in 0..rows.len() {
        let (column, pos0, real) = (rows.column[r], rows.pos0[r], rows.real[r]);
        let blocks = rows.blocks(r);
        let total = (pos0 + real) / POOL;
        let mut t0 = 0;
        while t0 < real {
            let seen = ((pos0 + t0 + query_block) / POOL).min(total);
            let mut key0 = 0;
            while key0 < seen {
                let first = (key0 / PAGE_POOLS) as usize;
                let live =
                    (INDEX_LANES as i64).min((seen - key0 + PAGE_POOLS - 1) / PAGE_POOLS) as usize;
                let available = blocks.len().saturating_sub(first).min(live);
                if available < live {
                    return Err(format!(
                        "row {r} has {} KV blocks, its index slots need {}",
                        blocks.len(),
                        first + live
                    ));
                }
                let mut pages = vec![-1i64; INDEX_LANES];
                pages[..live].copy_from_slice(&blocks[first..first + live]);
                out.push(Slot {
                    fields: [
                        0,
                        index_heads * (column + t0),
                        1 << 30,
                        key0,
                        seen,
                        column + t0,
                        0,
                        0,
                    ],
                    pages,
                });
                key0 += SLOT_POOLS;
            }
            t0 += query_block;
        }
    }
    Ok(out)
}

/// Write `slots` into the launch records at `record(launch, nth)` of every DSA layer, `per_launch` a launch, the
/// last launch padded with slot -1 (`fields`) and pages -1. A layer's pages are its offset plus `extra` further.
/// Returns the launches.
#[allow(clippy::too_many_arguments)]
fn write_launches(
    w: &mut Words<'_>,
    slots: &[Slot],
    per_launch: usize,
    lanes: usize,
    offsets: &[i64],
    extra: i64,
    record: impl Fn(usize, usize) -> usize,
    pad_fields: [i64; 8],
) -> usize {
    let launches = slots.len().div_ceil(per_launch);
    for j in 0..launches {
        for (nth, &off) in offsets.iter().enumerate() {
            let at = record(j, nth);
            for s in 0..per_launch {
                let slot = slots.get(j * per_launch + s);
                let fields = slot.map_or(pad_fields, |slot| slot.fields);
                for (f, &value) in fields.iter().enumerate() {
                    w.set(at, 8 * s + f, value);
                }
                let pages_at = at + SLOTS_BYTES;
                for lane in 0..lanes {
                    let page = slot.map_or(-1, |slot| slot.pages[lane]);
                    let value = if page >= 0 { page + off + extra } else { -1 };
                    w.set(pages_at, lanes * s + lane, value);
                }
            }
        }
    }
    launches
}

/// (ssm, conv) first rows of the `nth` KDA layer in a STATE block (-1: none).
fn state_rows(model: &Model, block: i64, nth: usize) -> (i64, i64) {
    if block < 0 {
        return (-1, -1);
    }
    let ssm = block + nth as i64 * model.state_layer_rows;
    (ssm, ssm + model.ssm_rows)
}

/// `_kda_tables`: a prefill chunk's four state rows (256 bytes apart), or a decode step's (T, 4) bases and (T,)
/// conv destinations, per KDA layer; a verify step's records are by row.
fn kda_tables(
    w: &mut Words<'_>,
    model: &Model,
    tables: &Tables,
    rows: &Rows<'_>,
    step: &Step<'_>,
) -> Result<(), String> {
    let record = tables.kda_record / 4;
    for nth in 0..tables.kda_layers {
        let at = tables.kda + nth * tables.kda_record;
        w.fill(at, record, -1);
        if tables.t > model.max_t {
            let r = 0;
            let (ss, cs) = state_rows(model, rows.state_src[r], nth);
            let (sd, cd) = state_rows(model, rows.state_dst[r], nth);
            w.set(at, 0, cs);
            w.set(at, 64, ss);
            w.set(at, 128, sd);
            w.set(at, 192, cd);
            continue;
        }
        let bases = tables.bases_bytes / 16;
        for r in 0..rows.len() {
            let column = rows.column[r];
            let c = if step.verify > 0 {
                column / step.verify as i64
            } else {
                column
            } as usize;
            if c >= bases {
                return Err(format!(
                    "row {r}'s KDA record {c} is past the table's {bases}"
                ));
            }
            let (ss, cs) = state_rows(model, rows.state_src[r], nth);
            let (sd, cd) = state_rows(model, rows.state_dst[r], nth);
            for (f, value) in [cs, ss, -1, sd].into_iter().enumerate() {
                w.set(at, 4 * c + f, value);
            }
            w.set(at + tables.bases_bytes, c, cd);
        }
    }
    Ok(())
}

/// Write a step's tables into `host` (one host copy, at least `tables.nbytes` bytes): the port of `_launch`'s table
/// writes. Padding words of the slot and index records (past a launch's slots) keep what the copy held.
pub fn fill(
    model: &Model,
    tables: &Tables,
    host: &mut [u8],
    rows: &Rows<'_>,
    step: &Step<'_>,
) -> Result<Counts, String> {
    let t = tables.t;
    let n = step.columns;
    if n > t {
        return Err(format!("{n} columns exceed the shape's {t}"));
    }
    if rows.len() == 0 {
        return Err("a step needs a row".into());
    }
    let mut w = words(host, tables.nbytes)?;
    w.fill(0, tables.header / 4, 0);
    if let Some(tokens) = step.tokens {
        if tokens.len() != n {
            return Err(format!("{} tokens for {n} columns", tokens.len()));
        }
        for (c, &token) in tokens.iter().enumerate() {
            w.set(tables.tok, c, token);
        }
    }
    let meta_columns = step.meta_columns.unwrap_or(n);
    for (f, value) in [step.first_position, meta_columns as i64, 0, 0]
        .into_iter()
        .enumerate()
    {
        w.set(tables.meta, f, value);
    }
    // a prefill shape: one query block of the step's width
    let prefill = t > model.max_t;
    if step.head {
        w.fill(tables.head, HEAD_COLS, (n as i64) - 1);
    }
    // visibility and key length; the indexer runs when some column sees more pools than it keeps
    let mut most = 0;
    for r in 0..rows.len() {
        for j in 0..rows.real[r] {
            let c = (rows.column[r] + j) as usize;
            if c >= t {
                return Err(format!(
                    "row {r}'s column {c} is past the step's {t} columns"
                ));
            }
            let p = rows.pos0[r] + j;
            let vis = (p + 1) / POOL;
            most = most.max(vis);
            w.set(tables.vis, c, vis);
            w.set(tables.klen, c, p + 1);
        }
    }
    let index = most > model.top_pools;
    let query_block = if prefill { t } else { 1 };
    let sparse = index && query_block == 1;
    let per_launch = if prefill { 1 } else { 16 };
    // the key and pool stores of every DSA layer
    let (store, pool) = store_rows(t, rows, step.keep_tails, model.tail_pools)?;
    let rows_words = tables.rows_bytes / 4;
    for (nth, &off) in tables.offsets.iter().enumerate() {
        let so = tables.store + nth * tables.rows_bytes;
        w.fill(so, rows_words, -1);
        let po = tables.pool + nth * tables.rows_bytes;
        w.fill(po, rows_words, -1);
        for c in 0..t {
            let [latent, scale, index] = store[c];
            if latent >= 0 {
                w.set(so, ROWS_FIELDS * c, latent + off);
                w.set(so, ROWS_FIELDS * c + 3, scale + off);
                w.set(so, ROWS_FIELDS * c + 4, index);
            }
            let unit = pool[c];
            if unit[0] >= 0 {
                w.set(po, ROWS_FIELDS * c, unit[0] + 2 * off);
            }
            for s in 1..4 {
                // a column >= 0, or -(arena row + 1): tails shift with the layer
                let source = if unit[s] < 0 { unit[s] - off } else { unit[s] };
                w.set(po, ROWS_FIELDS * c + s, source);
            }
            if unit[4] >= 0 {
                w.set(po, ROWS_FIELDS * c + 4, unit[4] + off);
            }
        }
    }
    let slots = if sparse {
        Vec::new()
    } else {
        key_slots(rows, query_block as i64, model.mla_heads)?
    };
    let ix = if index {
        index_slots(rows, query_block as i64, model.index_heads)?
    } else {
        Vec::new()
    };
    let (launches, ix_launches) = (
        slots.len().div_ceil(per_launch),
        ix.len().div_ceil(per_launch),
    );
    if launches > tables.key_launches || ix_launches > tables.index_launches {
        return Err("a step's slot launches exceed the table capacity".into());
    }
    let offsets = &tables.offsets;
    write_launches(
        &mut w,
        &slots,
        per_launch,
        KEY_LANES,
        offsets,
        0,
        |j, nth| tables.slot(j, nth),
        [-1, 0, 0, 0, 0, 0, 0, 0],
    );
    write_launches(
        &mut w,
        &ix,
        per_launch,
        INDEX_LANES,
        offsets,
        LATENT_PAGE_ROWS,
        |j, nth| tables.index(j, nth),
        [-1, 0, 0, 0, 0, 0, 0, 0],
    );
    if sparse {
        let width = model.pages;
        w.fill(tables.blocks, width * t, -1);
        for r in 0..rows.len() {
            let blocks = rows.blocks(r);
            if blocks.len() > width {
                return Err(format!(
                    "row {r} has {} KV blocks, the block table {width}",
                    blocks.len()
                ));
            }
            for j in 0..rows.real[r] {
                let c = (rows.column[r] + j) as usize;
                for (k, &block) in blocks.iter().enumerate() {
                    w.set(tables.blocks, width * c + k, block);
                }
            }
        }
    }
    if tables.kda_layers > 0 {
        kda_tables(&mut w, model, tables, rows, step)?;
    }
    let mut uploads = [(0, tables.blocks), (0, 0), (0, 0), (0, 0)];
    if sparse {
        uploads[1] = (tables.blocks, 4 * model.pages * t);
    }
    if launches > 0 {
        uploads[2] = (
            tables.slot(0, 0),
            tables.slot(launches, 0) - tables.slot(0, 0),
        );
    }
    if ix_launches > 0 {
        uploads[3] = (
            tables.index(0, 0),
            tables.index(ix_launches, 0) - tables.index(0, 0),
        );
    }
    Ok(Counts {
        index,
        sparse,
        slots: if sparse { 1 } else { launches },
        index_launches: ix_launches,
        uploads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables(t: usize, dsa: usize, kda: usize, key: usize, ix: usize, pages: usize) -> Tables {
        // `_Tables`
        let r = |n: usize| n.div_ceil(128) * 128;
        let max_t = 64;
        let meta = r(4 * t);
        let head = meta + 128;
        let vis = head + 128;
        let klen = vis + r(4 * t);
        let header = klen + r(4 * t);
        let rows_bytes = r(4 * ROWS_FIELDS * t);
        let store = header;
        let pool = store + dsa * rows_bytes;
        let kda_base = pool + dsa * rows_bytes;
        let bases_bytes = if t <= max_t { r(16 * t) } else { 1024 };
        let kda_record = bases_bytes + if t <= max_t { r(4 * t) } else { 0 };
        let blocks = r(kda_base + kda * kda_record);
        let slot_base = (blocks + 4 * pages * t).div_ceil(4096) * 4096;
        let index_base = slot_base + key * dsa * SLOT_RECORD;
        Tables {
            t,
            tok: 0,
            meta,
            head,
            vis,
            klen,
            header,
            rows_bytes,
            store,
            pool,
            kda: kda_base,
            bases_bytes,
            kda_record,
            blocks,
            slot_base,
            index_base,
            key_launches: key,
            index_launches: ix,
            nbytes: index_base + ix * dsa * INDEX_RECORD,
            dsa,
            kda_layers: kda,
            offsets: (0..dsa as i64).map(|i| 161 * i).collect(),
        }
    }

    fn model() -> Model {
        Model {
            mla_heads: 16,
            index_heads: 32,
            top_pools: 512,
            tail_pools: 2,
            pages: 64,
            state_layer_rows: 1024,
            ssm_rows: 512,
            max_t: 64,
        }
    }

    #[test]
    fn tails_take_two_rows_per_open_token() {
        assert_eq!(
            (0..9).map(|p| tail_rows(p, 2)).collect::<Vec<_>>(),
            [0, 2, 4, 6, 6, 8, 10, 12, 0]
        );
        assert_eq!(tail_rows(-1, 2), 12);
    }

    #[test]
    fn a_decode_step_stores_its_keys_and_pools() {
        let (m, tb) = (model(), tables(2, 2, 1, 4, 4, 64));
        let blocks = [1000i64, 2000];
        let rows = Rows {
            column: &[0, 1],
            pos0: &[130, 7],
            real: &[1, 1],
            state_src: &[5000, -1],
            state_dst: &[5000, 6000],
            blocks_indptr: &[0, 2, 3],
            blocks: &[blocks[0], blocks[1], 3000],
        };
        let step = Step {
            tokens: Some(&[11, 12]),
            columns: 2,
            first_position: 130,
            head: false,
            verify: 0,
            keep_tails: false,
            meta_columns: None,
        };
        let mut host = vec![0u8; tb.nbytes];
        let counts = fill(&m, &tb, &mut host, &rows, &step).unwrap();
        assert!(!counts.index && !counts.sparse);
        assert_eq!((counts.slots, counts.index_launches), (1, 0));
        let w = |at: usize| i64::from(i32::from_le_bytes(host[at..at + 4].try_into().unwrap()));
        assert_eq!((w(0), w(4), w(tb.meta), w(tb.meta + 4)), (11, 12, 130, 2));
        // row 0, position 130 of block 1: slot 2, an open pool (its tail kept: the pool ends after the step)
        let so = tb.store;
        assert_eq!((w(so), w(so + 12), w(so + 16)), (2000 + 2, 2000 + 128, 2));
        let po = tb.pool;
        assert_eq!(w(po), -1);
        assert_eq!(w(po + 16), 2000 + TAIL_OFFSET + 4);
        // row 1, position 7: completes pool 1, whose earlier tokens are tails of block 3000 (positions 4, 5, 6)
        let c1 = po + 4 * ROWS_FIELDS;
        assert_eq!(w(c1), 2 * (3000 + LATENT_PAGE_ROWS) + 1);
        let tail = |q: i64| -(3000 + TAIL_OFFSET + tail_rows(q, 2) + 1);
        assert_eq!(
            (w(c1 + 4), w(c1 + 8), w(c1 + 12)),
            (tail(4), tail(5), tail(6))
        );
        // layer 1's pages are 161 rows further
        let so1 = tb.store + tb.rows_bytes;
        assert_eq!(w(so1), 2000 + 2 + 161);
        let c1_1 = tb.pool + tb.rows_bytes + 4 * ROWS_FIELDS;
        assert_eq!(w(c1_1), 2 * (3000 + 161 + LATENT_PAGE_ROWS) + 1);
        assert_eq!(w(c1_1 + 4), tail(4) - 161);
        // one slot each: row 0 sees blocks 1000 and 2000 (131 keys), row 1 block 3000
        let s = tb.slot_base;
        assert_eq!(
            (0..8).map(|f| w(s + 4 * f)).collect::<Vec<_>>(),
            [0, 0, 130, 0, 131, 0, 0, 0]
        );
        assert_eq!(w(s + 32), 0);
        assert_eq!(w(s + 32 + 4), 16);
        assert_eq!((w(s + SLOTS_BYTES), w(s + SLOTS_BYTES + 4)), (1000, 2000));
        assert_eq!(w(s + SLOTS_BYTES + 8), -1);
        assert_eq!(w(s + SLOTS_BYTES + 4 * KEY_LANES), 3000);
        // padding slots
        assert_eq!(w(s + 4 * 8 * 2), -1);
        // KDA: row 0 reads and writes block 5000; row 1 starts fresh
        let k = tb.kda;
        assert_eq!(
            (0..8).map(|f| w(k + 4 * f)).collect::<Vec<_>>(),
            [5512, 5000, -1, 5000, -1, -1, -1, 6000]
        );
        assert_eq!(
            (w(k + tb.bases_bytes), w(k + tb.bases_bytes + 4)),
            (5512, 6512)
        );
        assert_eq!(counts.uploads[2], (tb.slot_base, 2 * SLOT_RECORD));
    }

    #[test]
    fn a_long_row_turns_the_indexer_on() {
        let (m, tb) = (model(), tables(1, 1, 0, 4, 4, 64));
        let blocks: Vec<i64> = (0..20).map(|b| 100 * b).collect();
        let rows = Rows {
            column: &[0],
            pos0: &[2100],
            real: &[1],
            state_src: &[-1],
            state_dst: &[-1],
            blocks_indptr: &[0, 20],
            blocks: &blocks,
        };
        let step = Step {
            tokens: None,
            columns: 1,
            first_position: 2100,
            head: false,
            verify: 0,
            keep_tails: false,
            meta_columns: None,
        };
        let mut host = vec![0u8; tb.nbytes];
        let counts = fill(&m, &tb, &mut host, &rows, &step).unwrap();
        assert!(counts.index && counts.sparse);
        assert_eq!((counts.slots, counts.index_launches), (1, 1));
        let w = |at: usize| i64::from(i32::from_le_bytes(host[at..at + 4].try_into().unwrap()));
        assert_eq!(w(tb.blocks), 0);
        assert_eq!(w(tb.blocks + 4 * 19), 1900);
        assert_eq!(w(tb.blocks + 4 * 20), -1);
        let ix = tb.index_base;
        assert_eq!(
            (0..8).map(|f| w(ix + 4 * f)).collect::<Vec<_>>(),
            [0, 0, 1 << 30, 0, 525, 0, 0, 0]
        );
        // 525 pools on 17 index pages, each a block's first row + LATENT_PAGE_ROWS
        assert_eq!(w(ix + SLOTS_BYTES), LATENT_PAGE_ROWS);
        assert_eq!(w(ix + SLOTS_BYTES + 4 * 16), 1600 + LATENT_PAGE_ROWS);
        assert_eq!(w(ix + SLOTS_BYTES + 4 * 17), -1);
    }

    #[test]
    fn a_prefill_chunk_writes_its_head_and_state_rows() {
        let (m, tb) = (model(), tables(128, 1, 2, 4, 4, 64));
        let blocks = [0i64, 1000];
        let rows = Rows {
            column: &[0],
            pos0: &[64],
            real: &[100],
            state_src: &[7000],
            state_dst: &[9000],
            blocks_indptr: &[0, 2],
            blocks: &blocks,
        };
        let tokens: Vec<i64> = (0..100).collect();
        let step = Step {
            tokens: Some(&tokens),
            columns: 100,
            first_position: 64,
            head: true,
            verify: 0,
            keep_tails: false,
            meta_columns: None,
        };
        let mut host = vec![0u8; tb.nbytes];
        let counts = fill(&m, &tb, &mut host, &rows, &step).unwrap();
        assert_eq!((counts.slots, counts.index_launches), (1, 0));
        let w = |at: usize| i64::from(i32::from_le_bytes(host[at..at + 4].try_into().unwrap()));
        assert_eq!((w(tb.head), w(tb.head + 60)), (99, 99));
        let k = tb.kda + tb.kda_record;
        assert_eq!(
            (w(k), w(k + 256), w(k + 512), w(k + 768), w(k + 4)),
            (8536, 8024, 10024, 10536, -1)
        );
        // one query block of the chunk, one key chunk
        let s = tb.slot_base;
        assert_eq!(
            (0..6).map(|f| w(s + 4 * f)).collect::<Vec<_>>(),
            [0, 0, 64, 0, 164, 0]
        );
        assert_eq!((w(s + SLOTS_BYTES), w(s + SLOTS_BYTES + 4)), (0, 1000));
    }
}
