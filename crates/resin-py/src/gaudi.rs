//! Host step code for the Gaudi runners (cargo feature `gaudi`): `Ds41Fill` writes a DeepSeek-V4.1 decode step's
//! tables straight into the runner's pinned host copy.

use std::collections::HashMap;

use numpy::{
    PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use resin_gaudi::{
    draft_choice, ds41, fnext, glm, glue, qwen, qwen_draft, qwen_mtp, sampling, speculative, step,
    tables,
};

fn slice<'a, T: numpy::Element>(
    name: &str,
    array: &'a PyReadonlyArray1<'_, T>,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous array")))
}

pub(crate) fn item<T>(layout: &Bound<'_, PyDict>, name: &str) -> PyResult<T>
where
    T: for<'a, 'py> FromPyObject<'a, 'py>,
{
    let value = layout
        .get_item(name)?
        .ok_or_else(|| PyKeyError::new_err(name.to_owned()))?;
    value.extract().map_err(Into::into)
}

/// (token_map, pad_id, multipliers (layers x n), n, heads, layers) of the Engram hash.
type EngramArgs = (
    Vec<i64>,
    i64,
    Vec<i64>,
    usize,
    usize,
    usize,
    Vec<i64>,
    Vec<i64>,
);
/// Bind the host Engram tables readonly for a fill call. The returned guards must outlive the
/// `ds41::HostTable` slices borrowed from them (both live to the end of the calling fill).
pub(crate) fn engram_host_guards<'py>(
    py: Python<'py>,
    tables: &[(Py<PyArray2<u8>>, Py<PyArray2<u8>>)],
) -> Vec<(PyReadonlyArray2<'py, u8>, PyReadonlyArray2<'py, u8>)> {
    tables
        .iter()
        .map(|(w, s)| (w.bind(py).readonly(), s.bind(py).readonly()))
        .collect()
}

/// Borrow the host Engram tables' bytes from their readonly guards (contiguous, as the mmap views are).
pub(crate) fn engram_host_slices<'a>(
    guards: &'a [(PyReadonlyArray2<'_, u8>, PyReadonlyArray2<'_, u8>)],
) -> PyResult<Vec<ds41::HostTable<'a>>> {
    guards
        .iter()
        .map(|(w, s)| {
            Ok(ds41::HostTable {
                weight: w
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("engram_host weight must be contiguous"))?,
                scale: s
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("engram_host scale must be contiguous"))?,
            })
        })
        .collect()
}

/// A fill's model, its shapes' tables by (columns, prefill), the rotation tables, and the host
/// Engram tables (per layer: (weight (rows,256) u8, scale (rows,8) u8) mmap views; empty in card mode).
pub(crate) type Ds41Parts = (
    ds41::Model,
    HashMap<(usize, bool), ds41::Tables>,
    [Py<PyArray2<f32>>; 2],
    Vec<(Py<PyArray2<u8>>, Py<PyArray2<u8>>)>,
);
/// (sparse, cand, ixc, radix1, radix2, ix1, ix2, pf1, pf2, dense1, dense2)
type Ds41Counts = (
    bool,
    bool,
    usize,
    bool,
    bool,
    usize,
    usize,
    usize,
    usize,
    bool,
    bool,
);

/// The native step fill of one DeepSeek-V4.1 model: constructed once with what the tables depend on, then
/// `add_shape` per step shape (table offsets) and `fill` per step.
#[pyclass(module = "resin._resin")]
pub(crate) struct Ds41Fill {
    model: ds41::Model,
    /// by (columns, prefill)
    shapes: HashMap<(usize, bool), ds41::Tables>,
    cs: [Py<PyArray2<f32>>; 2],
    /// host Engram lookup tables, per Engram layer index: (weight (rows,256) u8, scale (rows,8) u8),
    /// numpy views over the shared mmap, held alive like cs. Empty in card mode.
    engram_host: Vec<(Py<PyArray2<u8>>, Py<PyArray2<u8>>)>,
}

impl Ds41Fill {
    /// The fill's model, every shape's tables, the rotation tables and the host Engram tables (for
    /// the native executor).
    pub(crate) fn parts(&self, py: Python<'_>) -> Ds41Parts {
        (
            self.model.clone(),
            self.shapes.clone(),
            [self.cs[0].clone_ref(py), self.cs[1].clone_ref(py)],
            self.engram_host
                .iter()
                .map(|(w, s)| (w.clone_ref(py), s.clone_ref(py)))
                .collect(),
        )
    }

    fn shape(&self, t: usize, prefill: bool) -> PyResult<&ds41::Tables> {
        self.shapes.get(&(t, prefill)).ok_or_else(|| {
            let kind = if prefill { "prefill" } else { "decode" };
            PyKeyError::new_err(format!("no {kind} shape of {t} columns"))
        })
    }
}

#[pymethods]
impl Ds41Fill {
    #[new]
    #[pyo3(signature = (*, heads, index_heads, sliding_window, layer_rows, owners, pending, pages, kmax, kc, candidate, engram, cs0, cs1, radix_min, prefilter, dense_keys, geometry, engram_host=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        heads: i64,
        index_heads: i64,
        sliding_window: i64,
        layer_rows: Vec<i64>,
        owners: Vec<(i64, i64, i64)>,
        pending: Vec<usize>,
        pages: usize,
        kmax: i64,
        kc: i64,
        candidate: Option<(i64, i64)>,
        engram: Option<EngramArgs>,
        cs0: Bound<'_, PyArray2<f32>>,
        cs1: Bound<'_, PyArray2<f32>>,
        radix_min: i64,
        prefilter: (i64, i64),
        dense_keys: i64,
        geometry: (i64, i64, i64, i64, i64, i64),
        engram_host: Option<Vec<(Bound<'_, PyArray2<u8>>, Bound<'_, PyArray2<u8>>)>>,
    ) -> PyResult<Self> {
        // the cache geometry the Python side uses: (page tokens, history tokens, page rows, index rows, index
        // keys, pending slots)
        let native = (
            ds41::PAGE_TOKENS,
            ds41::HISTORY_TOKENS,
            ds41::PAGE_ROWS,
            ds41::INDEX_ROWS,
            ds41::INDEX_KEYS,
            ds41::PENDING_SLOTS,
        );
        if geometry != native {
            return Err(PyValueError::new_err(format!(
                "the cache geometry {geometry:?} is not the native fill's {native:?}"
            )));
        }
        for (name, table) in [("cs0", &cs0), ("cs1", &cs1)] {
            if table.shape()[1] != 128 || !table.is_c_contiguous() {
                return Err(PyValueError::new_err(format!(
                    "{name} must be a contiguous (positions, 128) table"
                )));
            }
        }
        if pending.iter().any(|&index| index >= owners.len()) {
            return Err(PyValueError::new_err("pending owners index the owners"));
        }
        let engram = match engram {
            Some((token_map, pad_id, multipliers, n, heads, layers, primes, offsets)) => {
                if n < 2
                    || multipliers.len() != layers * n
                    || primes.len() != layers * (n - 1) * heads
                    || offsets.len() != layers * (n - 1) * heads
                {
                    return Err(PyValueError::new_err(
                        "engram multipliers must hold layers x n values, primes and offsets layers x (n - 1) x heads",
                    ));
                }
                Some(ds41::Engram {
                    token_map,
                    pad_id,
                    multipliers,
                    n,
                    heads,
                    layers,
                    primes,
                    offsets,
                })
            }
            None => None,
        };
        let engram_host = match engram_host {
            None => Vec::new(),
            Some(tables) => {
                if engram.is_none() {
                    return Err(PyValueError::new_err(
                        "engram_host needs the engram hash parameters",
                    ));
                }
                let mut held = Vec::with_capacity(tables.len());
                for (w, s) in tables {
                    let (ws, ss) = (w.shape(), s.shape());
                    if ws.len() != 2 || ws[1] != 256 || ss.len() != 2 || ss[1] != 8 || ws[0] != ss[0]
                    {
                        return Err(PyValueError::new_err(
                            "engram_host needs (weight (rows,256) u8, scale (rows,8) u8) per layer",
                        ));
                    }
                    if !w.is_c_contiguous() || !s.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "engram_host weight and scale must be C-contiguous",
                        ));
                    }
                    held.push((w.unbind(), s.unbind()));
                }
                held
            }
        };
        Ok(Self {
            model: ds41::Model {
                heads,
                index_heads,
                sliding_window,
                layer_rows,
                owners: owners
                    .into_iter()
                    .map(|(ratio, pages, offset)| ds41::Owner {
                        ratio,
                        pages,
                        offset,
                    })
                    .collect(),
                pending,
                pages,
                kmax,
                kc,
                candidate,
                engram,
                radix_min,
                prefilter_blocks: prefilter.0,
                prefilter_min: prefilter.1,
                dense_keys,
            },
            shapes: HashMap::new(),
            cs: [cs0.unbind(), cs1.unbind()],
            engram_host,
        })
    }

    /// The host Engram dequant LUT (256,256) f32: `LUT[scale, code] = OCP[code] * 2^(scale-127)`.
    /// A verification surface for the host-gather oracle tests; not used in the step path.
    #[staticmethod]
    fn _engram_lut(py: Python<'_>) -> Bound<'_, PyArray2<f32>> {
        let rows: Vec<Vec<f32>> = ds41::engram_lut().iter().map(|r| r.to_vec()).collect();
        PyArray2::from_vec2(py, &rows).unwrap()
    }

    /// Gather + dequantize global row `ids` from host Engram table `layer` (verification surface for
    /// the oracle gate; the step path uses the same `engram_gather_into` core). Returns `(n, 256)` f32.
    fn _engram_gather<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        ids: PyReadonlyArray1<i64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (w, s) = self
            .engram_host
            .get(layer)
            .ok_or_else(|| PyValueError::new_err("no such host Engram layer"))?;
        let wg = w.bind(py).readonly();
        let sg = s.bind(py).readonly();
        let weight = wg
            .as_slice()
            .map_err(|_| PyValueError::new_err("engram_host weight must be contiguous"))?;
        let scale = sg
            .as_slice()
            .map_err(|_| PyValueError::new_err("engram_host scale must be contiguous"))?;
        let ids = ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("ids must be contiguous"))?;
        let mut out = vec![0f32; ids.len() * 256];
        ds41::engram_gather_into(weight, scale, ids, &mut out).map_err(PyValueError::new_err)?;
        let rows: Vec<Vec<f32>> = out.chunks(256).map(|c| c.to_vec()).collect();
        Ok(PyArray2::from_vec2(py, &rows).unwrap())
    }

    /// Byte offsets of a shape's tables: a dict of the `Tables` fields.
    fn add_shape(&mut self, layout: &Bound<'_, PyDict>) -> PyResult<()> {
        let tables = ds41::Tables {
            t: item(layout, "t")?,
            prefill: item(layout, "prefill")?,
            qb: item(layout, "qb")?,
            win_slots: item(layout, "win_slots")?,
            win_pages: item(layout, "win_pages")?,
            groups2: item(layout, "groups2")?,
            ix_visb2: item(layout, "ix_visb2")?,
            engram_ids: item(layout, "engram_ids")?,
            engram_emb: match layout.get_item("engram_emb")? {
                Some(value) => value.extract()?,
                None => Vec::new(),
            },
            cslots: item::<Option<(usize, usize)>>(layout, "cslots")?.map(Into::into),
            cmp_lat: item(layout, "cmp_lat")?,
            draft: item(layout, "draft")?,
            ix_slots: item(layout, "ix_slots")?,
            ix_tp: item(layout, "ix_tp")?,
            ix_lpg: item(layout, "ix_lpg")?,
            ix_rows: item(layout, "ix_rows")?,
            blk: item::<(usize, usize)>(layout, "blk")?.into(),
            ix_rec: item::<(usize, usize)>(layout, "ix_rec")?.into(),
            ix_gb: item::<(usize, usize)>(layout, "ix_gb")?.into(),
            nbytes: item(layout, "nbytes")?,
            zero: item(layout, "zero")?,
            tok: item(layout, "tok")?,
            meta: item(layout, "meta")?,
            cs: item::<(usize, usize)>(layout, "cs")?.into(),
            cs_inv: item::<(usize, usize)>(layout, "cs_inv")?.into(),
            cs_g2: item(layout, "cs_g2")?,
            cs_g1: item(layout, "cs_g1")?,
            cmp_store: item(layout, "cmp_store")?,
            cmp_groups: item(layout, "cmp_groups")?,
            cmp_rows: item(layout, "cmp_rows")?,
            slots: item(layout, "slots")?,
            rows: item(layout, "rows")?,
            lat: item(layout, "lat")?,
            ix_visb: item(layout, "ix_visb")?,
            ix_visc: item(layout, "ix_visc")?,
            ixsc_rec: item(layout, "ixsc_rec")?,
            sparse_batch: item(layout, "sparse_batch")?,
            ixsc_batches: item(layout, "ixsc_batches")?,
            ixc_per_batch: item(layout, "ixc_per_batch")?,
            ix_vis: item::<(usize, usize)>(layout, "ix_vis")?.into(),
            cmp_blocks: item::<(usize, usize)>(layout, "cmp_blocks")?.into(),
            engram: item(layout, "engram")?,
        };
        let layers = self.model.layer_rows.len();
        if tables.rows.len() != layers || tables.lat.len() != layers {
            return Err(PyValueError::new_err("rows and lat need a table per layer"));
        }
        if tables.cmp_store.len() != self.model.owners.len()
            || tables.cmp_groups.len() != self.model.pending.len()
            || tables.cmp_rows.len() != self.model.pending.len()
        {
            return Err(PyValueError::new_err(
                "compressor tables must follow the owners and pending owners",
            ));
        }
        if tables.cslots.is_some() && tables.cmp_lat.len() != self.model.owners.len() {
            return Err(PyValueError::new_err(
                "a dense shape needs each owner's latent pages",
            ));
        }
        self.shapes.insert((tables.t, tables.prefill), tables);
        Ok(())
    }

    /// Write step `t`'s tables (`prefill`: a prefill shape) into `host` (`with_tokens`: also the tokens and their
    /// Engram rows; `floor`: no window reaches before that position); returns (sparse, cand, ixc, radix1, radix2,
    /// ix1, ix2, pf1, pf2, dense1, dense2).
    #[pyo3(signature = (t, host, column, pos0, real, window_first, window_indptr, window, window_ids, history_indptr, history, before_indptr, before, positions, tokens=None, with_tokens=true, prefill=false, floor=None))]
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        py: Python<'_>,
        t: usize,
        mut host: PyReadwriteArray1<'_, u8>,
        column: PyReadonlyArray1<'_, i64>,
        pos0: PyReadonlyArray1<'_, i64>,
        real: PyReadonlyArray1<'_, i64>,
        window_first: PyReadonlyArray1<'_, i64>,
        window_indptr: PyReadonlyArray1<'_, i64>,
        window: PyReadonlyArray1<'_, i64>,
        window_ids: PyReadonlyArray1<'_, i64>,
        history_indptr: PyReadonlyArray1<'_, i64>,
        history: PyReadonlyArray1<'_, i64>,
        before_indptr: PyReadonlyArray1<'_, i64>,
        before: PyReadonlyArray1<'_, i64>,
        positions: PyReadonlyArray1<'_, i64>,
        tokens: Option<PyReadonlyArray1<'_, i64>>,
        with_tokens: bool,
        prefill: bool,
        floor: Option<i64>,
    ) -> PyResult<Ds41Counts> {
        let tables = self.shape(t, prefill)?;
        let cs0 = self.cs[0].bind(py).readonly();
        let cs1 = self.cs[1].bind(py).readonly();
        let cs_all = [
            cs0.as_slice()
                .map_err(|_| PyValueError::new_err("cs0 must be contiguous"))?,
            cs1.as_slice()
                .map_err(|_| PyValueError::new_err("cs1 must be contiguous"))?,
        ];
        let rows = ds41::Rows {
            column: slice("column", &column)?,
            pos0: slice("pos0", &pos0)?,
            real: slice("real", &real)?,
            window_first: slice("window_first", &window_first)?,
            window_indptr: slice("window_indptr", &window_indptr)?,
            window: slice("window", &window)?,
            window_ids: slice("window_ids", &window_ids)?,
            history_indptr: slice("history_indptr", &history_indptr)?,
            history: slice("history", &history)?,
            before_indptr: slice("before_indptr", &before_indptr)?,
            before: slice("before", &before)?,
        };
        let positions = slice("positions", &positions)?;
        let tokens = match &tokens {
            Some(tokens) => Some(slice("tokens", tokens)?),
            None => None,
        };
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let model = &self.model;
        let host_guards = engram_host_guards(py, &self.engram_host);
        let host_tables = engram_host_slices(&host_guards)?;
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
                    &host_tables,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            counts.sparse,
            counts.cand,
            counts.ixc,
            counts.radix[0],
            counts.radix[1],
            counts.ix[0],
            counts.ix[1],
            counts.pf[0],
            counts.pf[1],
            counts.dense[0],
            counts.dense[1],
        ))
    }

    /// Write the tables of step `t` (`prefill`) that need its `n` tokens into `host`: the tokens (None: the
    /// device's) and their Engram rows (`_fill_tokens`).
    #[pyo3(signature = (t, host, column, pos0, real, window_first, window_indptr, window, window_ids, history_indptr, history, before_indptr, before, n, tokens=None, prefill=false))]
    #[allow(clippy::too_many_arguments)]
    fn fill_tokens(
        &self,
        py: Python<'_>,
        t: usize,
        mut host: PyReadwriteArray1<'_, u8>,
        column: PyReadonlyArray1<'_, i64>,
        pos0: PyReadonlyArray1<'_, i64>,
        real: PyReadonlyArray1<'_, i64>,
        window_first: PyReadonlyArray1<'_, i64>,
        window_indptr: PyReadonlyArray1<'_, i64>,
        window: PyReadonlyArray1<'_, i64>,
        window_ids: PyReadonlyArray1<'_, i64>,
        history_indptr: PyReadonlyArray1<'_, i64>,
        history: PyReadonlyArray1<'_, i64>,
        before_indptr: PyReadonlyArray1<'_, i64>,
        before: PyReadonlyArray1<'_, i64>,
        n: usize,
        tokens: Option<PyReadonlyArray1<'_, i64>>,
        prefill: bool,
    ) -> PyResult<()> {
        let tables = self.shape(t, prefill)?;
        let rows = ds41::Rows {
            column: slice("column", &column)?,
            pos0: slice("pos0", &pos0)?,
            real: slice("real", &real)?,
            window_first: slice("window_first", &window_first)?,
            window_indptr: slice("window_indptr", &window_indptr)?,
            window: slice("window", &window)?,
            window_ids: slice("window_ids", &window_ids)?,
            history_indptr: slice("history_indptr", &history_indptr)?,
            history: slice("history", &history)?,
            before_indptr: slice("before_indptr", &before_indptr)?,
            before: slice("before", &before)?,
        };
        let tokens = match &tokens {
            Some(tokens) => Some(slice("tokens", tokens)?),
            None => None,
        };
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let model = &self.model;
        let host_guards = engram_host_guards(py, &self.engram_host);
        let host_tables = engram_host_slices(&host_guards)?;
        py.detach(|| ds41::fill_tokens(model, tables, host, &rows, n, tokens, &host_tables))
            .map_err(PyValueError::new_err)
    }

    /// Write decode shape `t`'s draft-step tables into `host` (`_fill_draft`): `anchors` per row, DSpark blocks of
    /// `block` tokens with `noise` after the anchor, through the last `stages` layers.
    #[pyo3(signature = (t, host, pos0, window_first, window_indptr, window, anchors, block, noise, stages))]
    #[allow(clippy::too_many_arguments)]
    fn fill_draft(
        &self,
        py: Python<'_>,
        t: usize,
        mut host: PyReadwriteArray1<'_, u8>,
        pos0: PyReadonlyArray1<'_, i64>,
        window_first: PyReadonlyArray1<'_, i64>,
        window_indptr: PyReadonlyArray1<'_, i64>,
        window: PyReadonlyArray1<'_, i64>,
        anchors: PyReadonlyArray1<'_, i64>,
        block: usize,
        noise: i64,
        stages: usize,
    ) -> PyResult<()> {
        let tables = self.shape(t, false)?;
        let pos0 = slice("pos0", &pos0)?;
        let b = pos0.len();
        let zeros = vec![0i64; b];
        let indptr = vec![0i64; b + 1];
        let window = slice("window", &window)?;
        let rows = ds41::Rows {
            column: &zeros,
            pos0,
            real: &zeros,
            window_first: slice("window_first", &window_first)?,
            window_indptr: slice("window_indptr", &window_indptr)?,
            window,
            window_ids: window,
            history_indptr: &indptr,
            history: &[],
            before_indptr: &indptr,
            before: &[],
        };
        let anchors = slice("anchors", &anchors)?;
        let cs0 = self.cs[0].bind(py).readonly();
        let cs1 = self.cs[1].bind(py).readonly();
        let cs_all = [
            cs0.as_slice()
                .map_err(|_| PyValueError::new_err("cs0 must be contiguous"))?,
            cs1.as_slice()
                .map_err(|_| PyValueError::new_err("cs1 must be contiguous"))?,
        ];
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let model = &self.model;
        let draft = ds41::Draft {
            block,
            noise,
            stages,
        };
        py.detach(|| ds41::fill_draft(model, tables, host, &rows, anchors, draft, cs_all))
            .map_err(PyValueError::new_err)
    }
}

/// DeepSeek-V4.1's Engram context (`resin_gaudi::ds41::context`): the tokens before each row's first position.
#[pyclass(module = "resin._resin")]
pub(crate) struct Ds41Context {
    pub(crate) context: ds41::context::Context,
}

#[pymethods]
impl Ds41Context {
    #[new]
    fn new(n: usize, window_blocks: usize) -> PyResult<Self> {
        Ok(Self {
            context: ds41::context::Context::new(n, window_blocks)
                .map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn n(&self) -> usize {
        self.context.n()
    }

    /// Whether `sequence` has a record (it is not released).
    fn has(&self, sequence: u64) -> bool {
        self.context.has(sequence)
    }

    /// The tokens at positions `pos0 - n` .. `pos0 - 1` that Engram hashes, for a row whose WINDOW table starts at
    /// logical block `first` with window block ids `ids`.
    fn before<'py>(
        &mut self,
        py: Python<'py>,
        sequence: u64,
        pos0: i64,
        first: i64,
        ids: Vec<i64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
        let before = self
            .context
            .before(sequence, pos0, first, &ids)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(numpy::PyArray1::from_vec(py, before))
    }

    /// Record `tokens`, committed at positions `pos0` and after, in place of any recorded there and after.
    fn seen(&mut self, sequence: u64, pos0: i64, tokens: Vec<i64>) -> PyResult<()> {
        self.context
            .seen(sequence, pos0, &tokens)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Forget released sequences (their blocks' records stay).
    fn release(&mut self, sequences: Vec<u64>) {
        self.context.release(&sequences);
    }
}

/// (mult, sizes, offs, hpn, eos, table_first, table_rows) of the PLE's n-gram hash on this rank.
type NGramArgs = (Vec<u64>, Vec<u64>, Vec<i64>, usize, i64, i64, i64);

fn strided(layout: &Bound<'_, PyDict>, name: &str) -> PyResult<fnext::Strided> {
    let (offset, stride) = item::<(usize, usize)>(layout, name)?;
    Ok(fnext::Strided { offset, stride })
}

/// The native decode-step fill of one Qwen3.8-Flash-Next model: constructed once with what the tables depend
/// on, then `add_shape` per decode bucket (table offsets) and `fill` per step.
#[pyclass(module = "resin._resin")]
pub(crate) struct FnextFill {
    model: fnext::Model,
    shapes: HashMap<usize, fnext::Tables>,
}

#[pymethods]
impl FnextFill {
    #[new]
    #[pyo3(signature = (*, linear_layers, full_layers, kv_groups, hpg, kv_layer_rows, state_layer_rows, ssm_rows, ple_offset, max_positions, ple_heads, ngram))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        linear_layers: usize,
        full_layers: usize,
        kv_groups: i64,
        hpg: i64,
        kv_layer_rows: i64,
        state_layer_rows: i64,
        ssm_rows: i64,
        ple_offset: Option<i64>,
        max_positions: i64,
        ple_heads: usize,
        ngram: Option<NGramArgs>,
    ) -> PyResult<Self> {
        let ngram = match ngram {
            Some((mult, sizes, offs, hpn, eos, table_first, table_rows)) => {
                if mult.len() < 2
                    || sizes.len() != offs.len()
                    || sizes.len() != (mult.len() - 1) * hpn
                {
                    return Err(PyValueError::new_err(
                        "the n-gram hash takes n multipliers and (n - 1) x hpn head sizes and offsets",
                    ));
                }
                Some(fnext::NGram {
                    mult,
                    sizes,
                    offs,
                    hpn,
                    eos,
                    table_first,
                    table_rows,
                })
            }
            None => None,
        };
        Ok(Self {
            model: fnext::Model {
                linear_layers,
                full_layers,
                kv_groups,
                hpg,
                kv_layer_rows,
                state_layer_rows,
                ssm_rows,
                ple_offset,
                max_positions,
                ple_heads,
                ngram,
            },
            shapes: HashMap::new(),
        })
    }

    /// Byte offsets of decode shape `t`'s tables: a dict of the `Tables` fields (per-layer tables as
    /// (offset, stride)).
    fn add_shape(&mut self, layout: &Bound<'_, PyDict>) -> PyResult<()> {
        let tables = fnext::Tables {
            t: item(layout, "t")?,
            nbytes: item(layout, "nbytes")?,
            slots: item(layout, "slots")?,
            tok: item(layout, "tok")?,
            pos: item(layout, "pos")?,
            meta: item(layout, "meta")?,
            cols: item(layout, "cols")?,
            samp_params: item(layout, "samp_params")?,
            prow: item(layout, "prow")?,
            ple_src: item(layout, "ple_src")?,
            ple_dst: item(layout, "ple_dst")?,
            gdn_bases: strided(layout, "gdn_bases")?,
            conv_dst: strided(layout, "conv_dst")?,
            kv_rows: strided(layout, "kv_rows")?,
            ix_rows: strided(layout, "ix_rows")?,
            kb_rows: strided(layout, "kb_rows")?,
            kb_tail: strided(layout, "kb_tail")?,
            slot_table: strided(layout, "slot_table")?,
            k_pages: strided(layout, "k_pages")?,
            v_pages: strided(layout, "v_pages")?,
        };
        self.shapes.insert(tables.t, tables);
        Ok(())
    }

    /// Write decode step `t`'s tables into `host`: every row's position, STATE block rows, KV block rows
    /// (`blocks_indptr` delimits them), token and the two tokens before it, and `samp` (rows x 8) f32.
    #[pyo3(signature = (t, host, pos0, state_src, state_dst, blocks_indptr, blocks, token, history, samp))]
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        py: Python<'_>,
        t: usize,
        mut host: PyReadwriteArray1<'_, u8>,
        pos0: PyReadonlyArray1<'_, i64>,
        state_src: PyReadonlyArray1<'_, i64>,
        state_dst: PyReadonlyArray1<'_, i64>,
        blocks_indptr: PyReadonlyArray1<'_, i64>,
        blocks: PyReadonlyArray1<'_, i64>,
        token: PyReadonlyArray1<'_, i64>,
        history: PyReadonlyArray1<'_, i64>,
        samp: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let tables = self
            .shapes
            .get(&t)
            .ok_or_else(|| PyKeyError::new_err(format!("no decode shape of {t} columns")))?;
        let rows = fnext::Rows {
            pos0: slice("pos0", &pos0)?,
            state_src: slice("state_src", &state_src)?,
            state_dst: slice("state_dst", &state_dst)?,
            blocks_indptr: slice("blocks_indptr", &blocks_indptr)?,
            blocks: slice("blocks", &blocks)?,
            token: slice("token", &token)?,
            history: slice("history", &history)?,
        };
        let samp = slice("samp", &samp)?;
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let model = &self.model;
        py.detach(|| fnext::fill(model, tables, host, &rows, samp))
            .map_err(PyValueError::new_err)
    }
}

/// The native step tables of one Qwen3.5/3.8 model (one rank's dimensions): constructed once, then `add_mode`
/// per step mode (it returns the mode's table offsets, which the device bindings take) and `fill` per step.
#[pyclass(module = "resin._resin")]
pub(crate) struct QwenFill {
    model: qwen::Model,
    modes: HashMap<String, (qwen::Mode, qwen::Layout)>,
}

impl QwenFill {
    /// The model and every added mode with its layout (for the native executor).
    pub(crate) fn parts(&self) -> (qwen::Model, &HashMap<String, (qwen::Mode, qwen::Layout)>) {
        (self.model.clone(), &self.modes)
    }
}

#[pymethods]
impl QwenFill {
    #[new]
    /// `kv8`: bit 0 the K pages FP8, bit 1 the V pages (`qwen3_5/cache.py`).
    #[pyo3(signature = (*, gdn_layers, kv_groups, hpg, state_layer_rows, ssm_rows, max_positions, kv8=0))]
    fn new(
        gdn_layers: usize,
        kv_groups: i64,
        hpg: i64,
        state_layer_rows: i64,
        ssm_rows: i64,
        max_positions: i64,
        kv8: u8,
    ) -> Self {
        Self {
            model: qwen::Model {
                gdn_layers,
                kv_groups,
                pages: qwen::Pages {
                    k8: kv8 & 1 != 0,
                    v8: kv8 & 2 != 0,
                },
                hpg,
                state_layer_rows,
                ssm_rows,
                max_positions,
            },
            modes: HashMap::new(),
        }
    }

    /// Add step mode `name` and return its image layout: a dict of byte offsets (`tok`, `pos`, `meta`, `last`,
    /// `samp_params`, `kv_rows`, `state`, `state_stride`, `conv_dst`, `slots`, `launch_stride`, `k_pages`,
    /// `v_pages`, and of the wide launches `wide`, `wide_stride`, `wide_k_pages`, `wide_v_pages`) and the image's
    /// size with every launch (`nbytes`). With `wide_slots` (a multiple of `slots`), a step's first slots go in
    /// up to `max_wide` launches of `wide_slots`, and `max_launches` (the narrow ones) is under
    /// `wide_slots / slots`. With `wide_full`, the wide launches take the slots whose every key each query of
    /// the block sees instead, and the narrow ones the others (`wide_slots` may then equal `slots`).
    #[pyo3(signature = (name, *, columns, per_row, query_block, slots, prefill, max_launches, extras=Vec::new(),
                        wide_slots=0, max_wide=0, wide_full=false))]
    #[allow(clippy::too_many_arguments)]
    fn add_mode<'py>(
        &mut self,
        py: Python<'py>,
        name: String,
        columns: usize,
        per_row: usize,
        query_block: i64,
        slots: usize,
        prefill: bool,
        max_launches: usize,
        extras: Vec<usize>,
        wide_slots: usize,
        max_wide: usize,
        wide_full: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if columns == 0 || per_row == 0 || !columns.is_multiple_of(per_row) || slots == 0 {
            return Err(PyValueError::new_err(format!("mode {name}: bad shape")));
        }
        if wide_slots > 0
            && (wide_slots < slots
                || (wide_slots == slots && !wide_full)
                || !wide_slots.is_multiple_of(slots)
                || (max_launches >= wide_slots / slots && !wide_full)
                || max_wide == 0)
        {
            return Err(PyValueError::new_err(format!(
                "mode {name}: wide launches of {wide_slots} slots need a multiple of {slots}, fewer than \
                 {} narrow launches and room for one",
                wide_slots / slots
            )));
        }
        let mode = qwen::Mode {
            name: name.clone(),
            columns,
            per_row,
            query_block,
            slots,
            prefill,
            max_launches,
            wide_slots,
            max_wide,
            wide_full,
        };
        let (layout, extra) = qwen::Layout::with_extras(&self.model, &mode, &extras);
        let out = PyDict::new(py);
        out.set_item("extras", extra)?;
        for (key, value) in [
            ("tok", layout.tok),
            ("pos", layout.pos),
            ("meta", layout.meta),
            ("last", layout.last),
            ("samp_params", layout.samp_params),
            ("kv_rows", layout.kv_rows),
            ("state", layout.state),
            ("state_stride", layout.state_stride),
            ("conv_dst", layout.conv_dst),
            ("slots", layout.slots),
            ("launch_stride", layout.launch_stride),
            ("k_pages", layout.k_pages),
            ("v_pages", layout.v_pages),
            ("launches", max_launches),
            ("wide", layout.wide),
            ("wide_slots", wide_slots),
            ("wide_launches", max_wide),
            ("wide_stride", layout.wide_stride),
            ("wide_k_pages", layout.wide_k_pages),
            ("wide_v_pages", layout.wide_v_pages),
            ("nbytes", layout.nbytes),
        ] {
            out.set_item(key, value)?;
        }
        self.modes.insert(name, (mode, layout));
        Ok(out)
    }

    /// The bytes of an MTP pass's tables (`cols`, `tok`, `pos`, `kv_rows`, `params`, `slots`; `resin_gaudi::
    /// qwen_mtp::PassLayout`) over a mode of `columns` columns, `slots` slots a launch and room for `max_launches`
    /// launches; a keys pass (`full` False) takes no sampler inputs and no slot tables.
    #[staticmethod]
    fn mtp_pass_sizes(columns: usize, slots: usize, max_launches: usize, full: bool) -> Vec<usize> {
        qwen_mtp::PassLayout::sizes(&decode_mode(columns, slots, max_launches), full).to_vec()
    }

    /// (k_pages, v_pages, stride) of a slot launch's tables with `slots` slots.
    #[staticmethod]
    fn launch_tables(slots: usize) -> (usize, usize, usize) {
        qwen::Attention::launch_tables(slots)
    }

    /// The chain image after an MTP verify step of `rows` rows at `width` columns a row (`qwen_mtp::ChainLayout`;
    /// its links run the decode mode of `rows` rows, `slots` slots a launch and room for `max_launches` launches)
    /// over `layers` GDN layers: a dict of the byte offsets (`commit_bases`, `commit_counts`, `hidden_table`,
    /// `following`, `meta`), `links` (each a dict of `cols`, `tok`, `pos`, `kv_rows`, `params`, `slots`,
    /// `stride`, `k_pages`, `v_pages`) and `nbytes`.
    #[staticmethod]
    fn mtp_chain<'py>(
        py: Python<'py>,
        rows: usize,
        slots: usize,
        max_launches: usize,
        width: usize,
        layers: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if width < 2 {
            return Err(PyValueError::new_err("a verify step of at least 2 columns"));
        }
        let layout =
            qwen_mtp::ChainLayout::new(&decode_mode(rows, slots, max_launches), width, layers);
        let out = PyDict::new(py);
        for (key, value) in [
            ("commit_bases", layout.commit.commit_bases),
            ("commit_counts", layout.commit.commit_counts),
            ("hidden_table", layout.commit.hidden_table),
            ("following", layout.following),
            ("meta", layout.meta),
            ("nbytes", layout.nbytes),
        ] {
            out.set_item(key, value)?;
        }
        let links = pyo3::types::PyList::empty(py);
        for link in &layout.links {
            let item = PyDict::new(py);
            for (key, value) in [
                ("cols", link.cols),
                ("tok", link.tok),
                ("pos", link.pos),
                ("kv_rows", link.kv_rows),
                ("params", link.params),
                ("slots", link.slots),
                ("stride", link.stride),
                ("k_pages", link.k_pages),
                ("v_pages", link.v_pages),
            ] {
                item.set_item(key, value)?;
            }
            links.append(item)?;
        }
        out.set_item("links", links)?;
        Ok(out)
    }

    /// Write an MTP pass of mode `mode` (`qwen_mtp::fill_pass`) into `host` at the tables `offsets`
    /// (`mtp_pass_sizes` order): row i's columns at positions from `starts[i]` (a prefill's `real` of them),
    /// its MTP KV block rows (`blocks_indptr` delimits them), each column's hidden column `cols` and token
    /// `tokens` (None: a device buffer), and for a full pass the sampler inputs `params`. The MTP layer's
    /// attention: `kv_groups` KV groups of `hpg` heads. Returns the slot launches.
    #[pyo3(signature = (mode, host, offsets, starts, real, blocks_indptr, blocks, cols, tokens, params, kv_groups, hpg))]
    #[allow(clippy::too_many_arguments)]
    fn fill_mtp(
        &self,
        mode: &str,
        mut host: PyReadwriteArray1<'_, u8>,
        offsets: Vec<usize>,
        starts: Vec<i64>,
        real: usize,
        blocks_indptr: Vec<i64>,
        blocks: Vec<i64>,
        cols: Vec<i64>,
        tokens: Option<Vec<i64>>,
        params: Option<Vec<f32>>,
        kv_groups: i64,
        hpg: i64,
    ) -> PyResult<usize> {
        let (mode, _) = self
            .modes
            .get(mode)
            .ok_or_else(|| PyKeyError::new_err(format!("no step mode {mode}")))?;
        if offsets.len() != 6 {
            return Err(PyValueError::new_err("an MTP pass takes 6 tables"));
        }
        let layout = qwen_mtp::PassLayout::new(mode, &offsets);
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let nbytes = host.len() / 4 * 4;
        let attention = qwen_mtp::MtpAttention {
            kv_groups,
            pages: self.model.pages,
            hpg,
            max_positions: self.model.max_positions,
        };
        let pass = qwen_mtp::Pass {
            starts: &starts,
            real,
            blocks_indptr: &blocks_indptr,
            blocks: &blocks,
            cols: &cols,
            tokens: tokens.as_deref(),
            params: params.as_deref(),
        };
        qwen_mtp::fill_pass(&attention, mode, &layout, host, nbytes, &pass)
            .map_err(PyValueError::new_err)
    }

    /// Write a verify step's commit and draft chain (`qwen_mtp::fill_chain`) into `host` (the chain image of
    /// `mtp_chain(rows, ...)`, `decode` naming the chain's decode mode): each row's first position, STATE source
    /// and STATE block rows (`state_first`: the logical block of the first), accepted drafts, next anchor, first
    /// next draft, MTP KV block rows (`blocks_indptr` delimits them) and each link's sampler inputs. Returns (each
    /// link's slot launches, the bytes to upload).
    #[pyo3(signature = (decode, width, host, pos0, state_src, states, state_first, accepted, anchors, first, blocks_indptr, blocks, params, kv_groups, hpg, state))]
    #[allow(clippy::too_many_arguments)]
    fn fill_chain(
        &self,
        decode: &str,
        width: usize,
        mut host: PyReadwriteArray1<'_, u8>,
        pos0: Vec<i64>,
        state_src: Vec<i64>,
        states: Vec<Vec<i64>>,
        state_first: Vec<i64>,
        accepted: Vec<i64>,
        anchors: Vec<i64>,
        first: Vec<i64>,
        blocks_indptr: Vec<i64>,
        blocks: Vec<i64>,
        params: Vec<Vec<f32>>,
        kv_groups: i64,
        hpg: i64,
        state: (i64, i64, i64, i64),
    ) -> PyResult<(Vec<usize>, usize)> {
        let (mode, _) = self
            .modes
            .get(decode)
            .ok_or_else(|| PyKeyError::new_err(format!("no step mode {decode}")))?;
        let layout = qwen_mtp::ChainLayout::new(mode, width, self.model.gdn_layers);
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        if host.len() < layout.nbytes || states.len() != pos0.len() {
            return Err(PyValueError::new_err(
                "a host copy too small, or rows unequal",
            ));
        }
        let rows: Vec<qwen_draft::StateRow<'_>> = (0..pos0.len())
            .map(|b| qwen_draft::StateRow {
                pos0: pos0[b],
                src: state_src[b],
                blocks: &states[b],
                first: state_first[b],
            })
            .collect();
        let geometry = qwen_draft::StateGeometry {
            state_layer_rows: state.0,
            ssm_rows: state.1,
            state_tokens: state.2,
            hidden_offset: state.3,
        };
        let attention = qwen_mtp::MtpAttention {
            kv_groups,
            pages: self.model.pages,
            hpg,
            max_positions: self.model.max_positions,
        };
        let chain = qwen_mtp::Chain {
            rows: &rows,
            accepted: &accepted,
            anchors: &anchors,
            first: &first,
            blocks_indptr: &blocks_indptr,
            blocks: &blocks,
            params: &params,
        };
        let filled = qwen_mtp::fill_chain(&attention, mode, &layout, &geometry, host, &chain)
            .map_err(PyValueError::new_err)?;
        Ok((filled.launches, filled.nbytes))
    }

    /// Write a target pass of mode `mode` into `host` (the mode's image): every row's first position, STATE block
    /// rows and KV block rows (`blocks_indptr` delimits them); `real` (a prefill's real tokens, else the real
    /// rows); the columns' `tokens` (None: another buffer holds them); `params` (PARAMS f32 a sampled column).
    /// Returns (slot launches per layer, bytes to upload).
    #[pyo3(signature = (mode, host, pos0, state_src, state_dst, blocks_indptr, blocks, real, tokens, params))]
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        py: Python<'_>,
        mode: &str,
        mut host: PyReadwriteArray1<'_, u8>,
        pos0: PyReadonlyArray1<'_, i64>,
        state_src: PyReadonlyArray1<'_, i64>,
        state_dst: PyReadonlyArray1<'_, i64>,
        blocks_indptr: PyReadonlyArray1<'_, i64>,
        blocks: PyReadonlyArray1<'_, i64>,
        real: usize,
        tokens: Option<PyReadonlyArray1<'_, i64>>,
        params: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<(usize, usize)> {
        let (mode, layout) = self
            .modes
            .get(mode)
            .ok_or_else(|| PyKeyError::new_err(format!("no step mode {mode}")))?;
        let step = qwen::Step {
            rows: qwen::Rows {
                pos0: slice("pos0", &pos0)?,
                state_src: slice("state_src", &state_src)?,
                state_dst: slice("state_dst", &state_dst)?,
                blocks_indptr: slice("blocks_indptr", &blocks_indptr)?,
                blocks: slice("blocks", &blocks)?,
            },
            real,
            tokens: match &tokens {
                Some(tokens) => Some(slice("tokens", tokens)?),
                None => None,
            },
            params: slice("params", &params)?,
        };
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let model = &self.model;
        let filled = py
            .detach(|| qwen::fill_target(model, mode, layout, host, &step))
            .map_err(PyValueError::new_err)?;
        Ok((filled.launches, filled.nbytes))
    }
}

/// A decode mode of `rows` rows (one column each), for the MTP layouts.
fn decode_mode(rows: usize, slots: usize, max_launches: usize) -> qwen::Mode {
    qwen::Mode {
        name: format!("d{rows}"),
        columns: rows,
        per_row: 1,
        query_block: 1,
        slots,
        prefill: false,
        max_launches,
        wide_slots: 0,
        max_wide: 0,
        wide_full: false,
    }
}

/// The native drafter tables of a Qwen3.5/3.8 verify step (`resin_gaudi::qwen_draft`), decoded into arrays for
/// the CPU checks against `common.dflash` and `qwen3_5.model` (the executor fills the images directly).
#[pyclass(module = "resin._resin")]
pub(crate) struct QwenDraftTables {
    drafter: qwen_draft::Drafter,
}

type DraftRowArg = (i64, Vec<i64>, i64);

fn draft_rows(rows: &[DraftRowArg]) -> Vec<qwen_draft::DraftRow<'_>> {
    rows.iter()
        .map(|(position, blocks, first)| qwen_draft::DraftRow {
            position: *position,
            blocks,
            first: *first,
        })
        .collect()
}

fn words_of(host: &[u32], offset: usize, count: usize) -> Vec<i32> {
    host[offset / 4..offset / 4 + count]
        .iter()
        .map(|&w| w as i32)
        .collect()
}

#[pymethods]
impl QwenDraftTables {
    #[new]
    #[pyo3(signature = (*, layers, pairs, layer_rows, block, window, causal, mask_token, scratch_rows, max_rows))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        layers: usize,
        pairs: usize,
        layer_rows: i64,
        block: usize,
        window: i64,
        causal: bool,
        mask_token: i64,
        scratch_rows: i64,
        max_rows: usize,
    ) -> Self {
        Self {
            drafter: qwen_draft::Drafter {
                layers,
                pairs,
                layer_rows,
                block,
                window,
                causal,
                mask_token,
                scratch_rows,
                max_rows,
            },
        }
    }

    /// An injection of `tokens` columns: (positions (tokens,), kv_rows (layers * tokens * 8,)). `runs`: (column,
    /// position, count, (position, blocks, first)) each.
    fn inject(
        &self,
        tokens: usize,
        runs: Vec<(usize, i64, usize, DraftRowArg)>,
    ) -> PyResult<(Vec<i32>, Vec<i32>)> {
        let d = &self.drafter;
        let kv_stride = tables::round(32 * tokens);
        let layout = qwen_draft::InjectLayout {
            tokens,
            pos: 0,
            kv_rows: tables::round(4 * tokens),
            kv_stride,
        };
        let nbytes = layout.kv_rows + d.layers * kv_stride;
        let mut host = vec![0xA5A5_A5A5u32; nbytes / 4];
        let rows: Vec<DraftRowArg> = runs.iter().map(|r| r.3.clone()).collect();
        let draft = draft_rows(&rows);
        let injected: Vec<qwen_draft::Injected<'_>> = runs
            .iter()
            .zip(&draft)
            .map(|(run, row)| qwen_draft::Injected {
                column: run.0,
                position: run.1,
                count: run.2,
                row: *row,
            })
            .collect();
        // SAFETY: a u32 buffer viewed as its bytes
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), nbytes) };
        qwen_draft::fill_inject(d, &layout, bytes, nbytes, &injected)
            .map_err(PyValueError::new_err)?;
        let mut kv = Vec::new();
        for layer in 0..d.layers {
            kv.extend(words_of(
                &host,
                layout.kv_rows + layer * kv_stride,
                8 * tokens,
            ));
        }
        Ok((words_of(&host, 0, tokens), kv))
    }

    /// A block step of `bucket` rows for a verify of `width` columns, decoded: a dict of `tok`, `pos`, `anchors`,
    /// `block_rows` (layers * T * 8), `slots` / `k_pages` / `v_pages` per (launch, layer), `slots_per_launch`,
    /// `launches`, `records` and `patch` (offsets in i32 units from the slot tables, launch-major).
    #[pyo3(signature = (bucket, rows, anchors, low, width, max_positions))]
    #[allow(clippy::too_many_arguments)]
    fn block<'py>(
        &self,
        py: Python<'py>,
        bucket: usize,
        rows: Vec<DraftRowArg>,
        anchors: Option<Vec<i64>>,
        low: Option<Vec<i64>>,
        width: usize,
        max_positions: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.drafter;
        let layout = qwen_draft::BlockLayout::new(d, bucket);
        let mut host = vec![0xA5A5_A5A5u32; layout.nbytes / 4];
        // SAFETY: a u32 buffer viewed as its bytes
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), layout.nbytes)
        };
        let draft = draft_rows(&rows);
        let filled = qwen_draft::fill_block(
            d,
            &layout,
            bytes,
            &draft,
            anchors.as_deref(),
            low.as_deref(),
            width,
            max_positions,
        )
        .map_err(PyValueError::new_err)?;
        let t = d.block_of(width) * bucket;
        let out = PyDict::new(py);
        out.set_item("tok", words_of(&host, layout.tok, t))?;
        out.set_item("pos", words_of(&host, layout.pos, t))?;
        out.set_item("anchors", words_of(&host, layout.anchors, bucket))?;
        let mut block_rows = Vec::new();
        for layer in 0..d.layers {
            block_rows.extend(words_of(
                &host,
                layout.block_rows + layer * layout.block_stride,
                8 * t,
            ));
        }
        out.set_item("block_rows", block_rows)?;
        let s = layout.slots;
        let (mut slots, mut kp, mut vp) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..filled.launches {
            for layer in 0..d.layers {
                let base = layout.table(i, layer);
                slots.extend(words_of(&host, base, 8 * s));
                kp.extend(words_of(&host, base + layout.k_pages, 16 * s));
                vp.extend(words_of(&host, base + layout.v_pages, 16 * s));
            }
        }
        out.set_item("slots", slots)?;
        out.set_item("k_pages", kp)?;
        out.set_item("v_pages", vp)?;
        out.set_item("slots_per_launch", s)?;
        out.set_item("stride", layout.stride)?;
        out.set_item("launches", filled.launches)?;
        out.set_item("nbytes", filled.nbytes)?;
        out.set_item("records", filled.records.clone())?;
        out.set_item("patch", qwen_draft::patch_entries(&layout, &filled.records))?;
        Ok(out)
    }

    /// A block step's image layout for `bucket` rows: a dict of byte offsets (`tok`, `pos`, `anchors`,
    /// `block_rows`, `block_stride`, `tables`, `stride`, `k_pages`, `v_pages`), `slots` a launch, `max_launches`
    /// and `nbytes`; launch i of layer l's tables at `tables + (i * layers + l) * stride`.
    fn layout<'py>(&self, py: Python<'py>, bucket: usize) -> PyResult<Bound<'py, PyDict>> {
        let l = qwen_draft::BlockLayout::new(&self.drafter, bucket);
        let out = PyDict::new(py);
        for (key, value) in [
            ("tok", l.tok),
            ("pos", l.pos),
            ("anchors", l.anchors),
            ("block_rows", l.block_rows),
            ("block_stride", l.block_stride),
            ("tables", l.tables),
            ("stride", l.stride),
            ("k_pages", l.k_pages),
            ("v_pages", l.v_pages),
            ("slots", l.slots),
            ("max_launches", l.max_launches),
            ("nbytes", l.nbytes),
        ] {
            out.set_item(key, value)?;
        }
        Ok(out)
    }

    /// The bytes of a verify step's spec_fix tables (`FixLayout` order) for `bucket` rows.
    #[staticmethod]
    fn fix_sizes(bucket: usize, layers: usize, patches: usize) -> Vec<usize> {
        qwen_draft::FixLayout::sizes(bucket, layers, patches).to_vec()
    }

    /// The bytes of an injection's tables of `tokens` columns: (positions, every layer's kv_rows).
    fn inject_sizes(&self, tokens: usize) -> Vec<usize> {
        qwen_draft::InjectLayout::sizes(&self.drafter, tokens).to_vec()
    }

    /// `qwen3_5.model.choose_width`
    #[staticmethod]
    fn choose_width(widths: Vec<usize>, accepted: Vec<i64>, width: usize, bucket: usize) -> usize {
        qwen_draft::choose_width(&widths, &accepted, width, bucket)
    }

    /// `accept_speculative_tokens` of rows x (width - 1) drafts and rows x width samples: (accepted, emitted).
    #[staticmethod]
    fn accept(
        drafts: Vec<i64>,
        samples: Vec<i64>,
        rows: usize,
        width: usize,
    ) -> (Vec<i64>, Vec<Vec<i64>>) {
        speculative::accept(&drafts, &samples, rows, width, None)
    }
}

/// `common.sampling.philox_uniform` of one seed and offset (for the CPU checks of the native sampling inputs).
#[pyfunction]
pub(crate) fn qwen_philox_uniform(seed: u64, offset: u64) -> f64 {
    sampling::philox_uniform(seed, offset)
}

/// The glue functions of the Gaudi runtime library the device loaded (`Device.libraries.runtime`).
#[pyclass(module = "resin._resin", frozen)]
pub(crate) struct GaudiGlue {
    pub(crate) glue: glue::Glue,
}

#[pymethods]
impl GaudiGlue {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            glue: glue::Glue::open(path).map_err(PyValueError::new_err)?,
        })
    }
}

/// An operand from Python: an int, or (base, register, scale) for `base + registers[register] * scale`.
#[derive(FromPyObject)]
enum OperandArg {
    Const(u64),
    Scaled(u64, step::Register, u64),
}

impl From<OperandArg> for step::Operand {
    fn from(operand: OperandArg) -> Self {
        match operand {
            OperandArg::Const(value) => Self::Const(value),
            OperandArg::Scaled(base, register, scale) => Self::Scaled {
                base,
                register,
                scale,
            },
        }
    }
}

/// A section by its number or its name.
#[derive(FromPyObject)]
enum SectionArg {
    Index(usize),
    Name(String),
}

fn section_of(program: &step::Program, section: &SectionArg) -> PyResult<usize> {
    match section {
        SectionArg::Index(index) if *index < program.sections().len() => Ok(*index),
        SectionArg::Index(index) => Err(PyValueError::new_err(format!(
            "section {index} of {}",
            program.sections().len()
        ))),
        SectionArg::Name(name) => program
            .section(name)
            .ok_or_else(|| PyKeyError::new_err(name.clone())),
    }
}

/// (kind, integers) of one stream call. A launch is (recipe, then slot and address of each binding); device
/// copies are (n, then the n sources, the n sizes and the n destinations).
type StepCall = (&'static str, Vec<u64>);

fn step_call(call: step::Call) -> StepCall {
    use step::Call;
    match call {
        Call::Launch { recipe, binds } => (
            "launch",
            std::iter::once(recipe as u64)
                .chain(
                    binds
                        .into_iter()
                        .flat_map(|(slot, address)| [slot as u64, address]),
                )
                .collect(),
        ),
        Call::AllGather {
            send,
            recv,
            count,
            bf16,
        } => ("allgather", vec![send, recv, count, u64::from(bf16)]),
        Call::AllReduce {
            address,
            count,
            bf16,
        } => ("allreduce", vec![address, count, u64::from(bf16)]),
        Call::Copy { dst, src, bytes } => ("copy", vec![dst, src, bytes]),
        Call::CopyMulti { src, bytes, dst } => (
            "copy_multi",
            std::iter::once(src.len() as u64)
                .chain(src)
                .chain(bytes)
                .chain(dst)
                .collect(),
        ),
        Call::Upload { dst, host, bytes } => ("upload", vec![dst, host, bytes]),
        Call::UploadRing { dst, host, bytes } => ("upload_ring", vec![dst, host, bytes]),
        Call::Download { host, src, bytes } => ("download", vec![host, src, bytes]),
        Call::Memset { dst, value, count } => ("memset", vec![dst, u64::from(value), count]),
        Call::Record { slot } => ("record", vec![slot as u64]),
    }
}

fn run_error(py: Python<'_>, error: step::RunError) -> PyErr {
    let raised = pyo3::exceptions::PyRuntimeError::new_err(error.to_string());
    // the number of calls the stream took before the failure: after any, device state may have changed
    match raised.value(py).setattr("issued", error.issued) {
        Ok(()) => raised,
        Err(failed) => failed,
    }
}

enum Stage {
    Building(step::Builder),
    Built(Arc<step::Program>),
}

/// A model step's stream calls as one native program (`resin_gaudi::step`). It is built once after the
/// recipes compile: `section`, then the ops in stream order, with `take` or `select` (and `arm`) opening a
/// nested list that `end` closes, then `finish`. A step then runs a list of sections with its register array
/// (`run`, without the GIL); `calls` gives the same walk's calls, recorded, for tests and check mode.
#[pyclass(module = "resin._resin")]
pub(crate) struct GaudiProgram {
    stage: Stage,
}

impl GaudiProgram {
    fn builder(&mut self) -> PyResult<&mut step::Builder> {
        match &mut self.stage {
            Stage::Building(builder) => Ok(builder),
            Stage::Built(_) => Err(PyValueError::new_err("the program is finished")),
        }
    }

    fn built(&self) -> PyResult<Arc<step::Program>> {
        match &self.stage {
            Stage::Built(program) => Ok(Arc::clone(program)),
            Stage::Building(_) => Err(PyValueError::new_err("the program is not finished")),
        }
    }

    /// The finished program (for the native executors).
    pub(crate) fn program(&self) -> PyResult<Arc<step::Program>> {
        self.built()
    }

    fn push(&mut self, op: step::Op) -> PyResult<()> {
        self.builder()?.push(op).map_err(PyValueError::new_err)
    }

    fn steps(
        steps: &[(Bound<'_, GaudiProgram>, SectionArg)],
    ) -> PyResult<Vec<(Arc<step::Program>, usize)>> {
        steps
            .iter()
            .map(|(program, section)| {
                let program = program.borrow().built()?;
                let section = section_of(&program, section)?;
                Ok((program, section))
            })
            .collect()
    }
}

#[pymethods]
impl GaudiProgram {
    #[new]
    fn new(registers: usize) -> Self {
        Self {
            stage: Stage::Building(step::Builder::new(registers)),
        }
    }

    /// Start a section; returns its number.
    fn section(&mut self, name: &str) -> PyResult<usize> {
        self.builder()?.section(name).map_err(PyValueError::new_err)
    }

    /// A graph launch: the fixed bindings, then `dynamic` (slot, operand) bindings, in order.
    #[pyo3(signature = (recipe, slots, addresses, dynamic=Vec::new()))]
    fn launch(
        &mut self,
        recipe: i32,
        slots: Vec<i32>,
        addresses: Vec<u64>,
        dynamic: Vec<(i32, OperandArg)>,
    ) -> PyResult<()> {
        self.push(step::Op::Launch(step::Launch {
            recipe,
            slots,
            addresses,
            dynamic: dynamic
                .into_iter()
                .map(|(slot, address)| (slot, address.into()))
                .collect(),
        }))
    }

    #[pyo3(signature = (send, recv, count, bf16=false))]
    fn allgather(
        &mut self,
        send: OperandArg,
        recv: OperandArg,
        count: OperandArg,
        bf16: bool,
    ) -> PyResult<()> {
        self.push(step::Op::AllGather {
            send: send.into(),
            recv: recv.into(),
            count: count.into(),
            bf16,
        })
    }

    #[pyo3(signature = (address, count, bf16=false))]
    fn allreduce(&mut self, address: OperandArg, count: OperandArg, bf16: bool) -> PyResult<()> {
        self.push(step::Op::AllReduce {
            address: address.into(),
            count: count.into(),
            bf16,
        })
    }

    fn copy(&mut self, dst: OperandArg, src: OperandArg, bytes: OperandArg) -> PyResult<()> {
        self.push(step::Op::Copy {
            dst: dst.into(),
            src: src.into(),
            bytes: bytes.into(),
        })
    }

    fn copy_multi(&mut self, src: Vec<u64>, bytes: Vec<u64>, dst: Vec<u64>) -> PyResult<()> {
        self.push(step::Op::CopyMulti { src, bytes, dst })
    }

    /// From pinned host memory, which must stay alive until the stream has passed the copy.
    fn upload(&mut self, dst: OperandArg, host: OperandArg, bytes: OperandArg) -> PyResult<()> {
        self.push(step::Op::Upload {
            dst: dst.into(),
            host: host.into(),
            bytes: bytes.into(),
        })
    }

    /// Through the runtime's ring of pinned slots: the host bytes are read when the step gets here.
    fn upload_ring(
        &mut self,
        dst: OperandArg,
        host: OperandArg,
        bytes: OperandArg,
    ) -> PyResult<()> {
        self.push(step::Op::UploadRing {
            dst: dst.into(),
            host: host.into(),
            bytes: bytes.into(),
        })
    }

    /// To pinned host memory, which must stay alive until the stream has passed the copy.
    fn download(&mut self, host: OperandArg, src: OperandArg, bytes: OperandArg) -> PyResult<()> {
        self.push(step::Op::Download {
            host: host.into(),
            src: src.into(),
            bytes: bytes.into(),
        })
    }

    fn memset(&mut self, dst: OperandArg, value: u32, count: OperandArg) -> PyResult<()> {
        self.push(step::Op::Memset {
            dst: dst.into(),
            value,
            count: count.into(),
        })
    }

    fn record(&mut self, slot: i32) -> PyResult<()> {
        self.push(step::Op::Record { slot })
    }

    /// Open a list of which a step makes the first `registers[register]` ops (no collectives in it).
    fn take(&mut self, register: step::Register) -> PyResult<()> {
        self.builder()?
            .take(register)
            .map_err(PyValueError::new_err)
    }

    /// Open a choice of arms: `table[index]`, where `index` combines the (register, radix) keys with the first
    /// key the most significant; -1 is no arm. `arm` starts each arm.
    fn select(&mut self, keys: Vec<(step::Register, u64)>, table: Vec<i32>) -> PyResult<()> {
        self.builder()?
            .select(keys, table)
            .map_err(PyValueError::new_err)
    }

    fn arm(&mut self) -> PyResult<usize> {
        self.builder()?.arm().map_err(PyValueError::new_err)
    }

    fn end(&mut self) -> PyResult<()> {
        self.builder()?.end().map_err(PyValueError::new_err)
    }

    fn finish(&mut self) -> PyResult<()> {
        let program = self.builder()?.finish().map_err(PyValueError::new_err)?;
        self.stage = Stage::Built(Arc::new(program));
        Ok(())
    }

    #[getter]
    fn registers(&self) -> PyResult<usize> {
        Ok(self.built()?.registers())
    }

    #[getter]
    fn sections(&self) -> PyResult<Vec<String>> {
        Ok(self
            .built()?
            .sections()
            .iter()
            .map(|section| section.name.clone())
            .collect())
    }

    /// Whether a section makes no collective on any path.
    fn collective_free(&self, section: SectionArg) -> PyResult<bool> {
        let program = self.built()?;
        let section = section_of(&program, &section)?;
        Ok(program.collective_free(section))
    }

    /// The calls a step with these registers makes, recorded: (kind, integers) per call.
    fn calls(
        &self,
        py: Python<'_>,
        registers: Vec<u64>,
        sections: Vec<SectionArg>,
    ) -> PyResult<Vec<StepCall>> {
        let program = self.built()?;
        let sections = sections
            .iter()
            .map(|section| section_of(&program, section))
            .collect::<PyResult<Vec<_>>>()?;
        program
            .calls(&sections, &registers)
            .map(|calls| calls.into_iter().map(step_call).collect())
            .map_err(|error| run_error(py, error))
    }

    /// Make the step's calls on the card through `glue`, without the GIL. Returns the number of calls. On a
    /// failure the RuntimeError's `issued` is the number of calls the stream took first.
    fn run(
        &self,
        py: Python<'_>,
        glue: &GaudiGlue,
        registers: PyReadonlyArray1<'_, u64>,
        sections: Vec<SectionArg>,
    ) -> PyResult<usize> {
        let program = self.built()?;
        let sections = sections
            .iter()
            .map(|section| section_of(&program, section))
            .collect::<PyResult<Vec<_>>>()?;
        let registers = slice("registers", &registers)?;
        let glue = &glue.glue;
        py.detach(|| {
            // SAFETY: the Python builder binds device allocations and pinned host buffers that the model keeps
            // for the life of its programs, and recipes and event slots of the loaded runtime.
            let mut stream = unsafe { step::CardStream::new(glue) };
            program.run(&sections, registers, &mut stream)
        })
        .map_err(|error| run_error(py, error))
    }

    /// The calls of several programs' sections in order, recorded (see `calls`).
    #[staticmethod]
    fn sequence_calls(
        py: Python<'_>,
        steps: Vec<(Bound<'_, GaudiProgram>, SectionArg)>,
        registers: Vec<u64>,
    ) -> PyResult<Vec<StepCall>> {
        let steps = Self::steps(&steps)?;
        let steps: Vec<(&step::Program, usize)> = steps
            .iter()
            .map(|(program, section)| (program.as_ref(), *section))
            .collect();
        let mut recorder = step::Recorder::default();
        step::run_sequence(&steps, &registers, &mut recorder)
            .map_err(|error| run_error(py, error))?;
        Ok(recorder.calls.into_iter().map(step_call).collect())
    }

    /// Make several programs' sections in order as one step, without the GIL (see `run`).
    #[staticmethod]
    fn run_sequence(
        py: Python<'_>,
        glue: &GaudiGlue,
        steps: Vec<(Bound<'_, GaudiProgram>, SectionArg)>,
        registers: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<usize> {
        let steps = Self::steps(&steps)?;
        let registers = slice("registers", &registers)?;
        let glue = &glue.glue;
        py.detach(|| {
            let steps: Vec<(&step::Program, usize)> = steps
                .iter()
                .map(|(program, section)| (program.as_ref(), *section))
                .collect();
            // SAFETY: as in `run`.
            let mut stream = unsafe { step::CardStream::new(glue) };
            step::run_sequence(&steps, registers, &mut stream)
        })
        .map_err(|error| run_error(py, error))
    }
}

/// The native step fill of one GLM-5.3-Flash model (`resin_gaudi::glm`): constructed once with what the tables
/// depend on, then `add_shape` per table shape (the step's and the MTP pass's) and `fill` per step.
#[pyclass(module = "resin._resin")]
pub(crate) struct GlmFill {
    pub(crate) model: glm::Model,
    /// by (columns, MTP pass)
    pub(crate) shapes: HashMap<(usize, bool), glm::Tables>,
}

impl GlmFill {
    pub(crate) fn tables(&self, t: usize, mtp: bool) -> PyResult<&glm::Tables> {
        self.shapes.get(&(t, mtp)).ok_or_else(|| {
            let kind = if mtp { "MTP" } else { "step" };
            PyKeyError::new_err(format!("no {kind} tables of {t} columns"))
        })
    }
}

/// (index on, sparse, slot launches, index launches, uploads (offset, bytes) x 4)
type GlmCounts = (bool, bool, usize, usize, [(usize, usize); 4]);

#[pymethods]
impl GlmFill {
    #[new]
    #[pyo3(signature = (*, mla_heads, index_heads, top_pools, tail_pools, pages, state_layer_rows, ssm_rows, max_t, geometry))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mla_heads: i64,
        index_heads: i64,
        top_pools: i64,
        tail_pools: i64,
        pages: usize,
        state_layer_rows: i64,
        ssm_rows: i64,
        max_t: usize,
        geometry: (i64, i64, i64, i64, i64, i64),
    ) -> PyResult<Self> {
        // the cache geometry the Python side uses: (page tokens, pool keys, page pools, latent page rows, index
        // page rows, slot pools)
        let native = (
            glm::PAGE_TOKENS,
            glm::POOL,
            glm::PAGE_POOLS,
            glm::LATENT_PAGE_ROWS,
            glm::INDEX_PAGE_ROWS,
            glm::SLOT_POOLS,
        );
        if geometry != native {
            return Err(PyValueError::new_err(format!(
                "the cache geometry {geometry:?} is not the native fill's {native:?}"
            )));
        }
        Ok(Self {
            model: glm::Model {
                mla_heads,
                index_heads,
                top_pools,
                tail_pools,
                pages,
                state_layer_rows,
                ssm_rows,
                max_t,
            },
            shapes: HashMap::new(),
        })
    }

    /// Byte offsets of a shape's tables (a dict of the `_Tables` fields, `offsets` its DSA layers' page offsets);
    /// `mtp`: the MTP pass's tables.
    fn add_shape(&mut self, layout: &Bound<'_, PyDict>, mtp: bool) -> PyResult<()> {
        let tables = glm::Tables {
            t: item(layout, "T")?,
            tok: item(layout, "TOK")?,
            meta: item(layout, "META")?,
            head: item(layout, "HEAD")?,
            vis: item(layout, "VIS")?,
            klen: item(layout, "KLEN")?,
            header: item(layout, "HEADER")?,
            rows_bytes: item(layout, "ROWS_BYTES")?,
            store: item(layout, "STORE")?,
            pool: item(layout, "POOL")?,
            kda: item(layout, "KDA")?,
            bases_bytes: item(layout, "BASES_BYTES")?,
            kda_record: item(layout, "KDA_RECORD")?,
            blocks: item(layout, "BLOCKS")?,
            slot_base: item(layout, "slot_base")?,
            index_base: item(layout, "index_base")?,
            key_launches: item(layout, "key_launches")?,
            index_launches: item(layout, "index_launches")?,
            nbytes: item(layout, "nbytes")?,
            dsa: item(layout, "dsa")?,
            kda_layers: item(layout, "kda_layers")?,
            offsets: item(layout, "offsets")?,
        };
        if tables.offsets.len() != tables.dsa {
            return Err(PyValueError::new_err(
                "offsets need one page offset per DSA layer",
            ));
        }
        if tables.slot_base < tables.blocks || tables.nbytes < tables.index_base {
            return Err(PyValueError::new_err(
                "the tables' offsets are out of order",
            ));
        }
        self.shapes.insert((tables.t, mtp), tables);
        Ok(())
    }

    /// Write a step's tables (shape `t`, `mtp`: the MTP pass's) into `host`: the rows (column, pos0, real,
    /// state_src, state_dst, KV blocks by `blocks_indptr`), the step's `columns` real columns with `tokens` (None:
    /// the device's), its first position, and whether it writes the head's columns (a prefill chunk), is a verify
    /// step of `verify` columns a row, keeps every pool tail, and the meta table's real columns. Returns (index on, sparse, slot launches,
    /// index launches, uploads).
    #[pyo3(signature = (t, mtp, host, column, pos0, real, state_src, state_dst, blocks_indptr, blocks, columns, first_position, tokens=None, head=false, verify=0, keep_tails=false, meta_columns=None))]
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        py: Python<'_>,
        t: usize,
        mtp: bool,
        mut host: PyReadwriteArray1<'_, u8>,
        column: PyReadonlyArray1<'_, i64>,
        pos0: PyReadonlyArray1<'_, i64>,
        real: PyReadonlyArray1<'_, i64>,
        state_src: PyReadonlyArray1<'_, i64>,
        state_dst: PyReadonlyArray1<'_, i64>,
        blocks_indptr: PyReadonlyArray1<'_, i64>,
        blocks: PyReadonlyArray1<'_, i64>,
        columns: usize,
        first_position: i64,
        tokens: Option<PyReadonlyArray1<'_, i64>>,
        head: bool,
        verify: usize,
        keep_tails: bool,
        meta_columns: Option<usize>,
    ) -> PyResult<GlmCounts> {
        let tables = self.tables(t, mtp)?;
        let rows = glm::Rows {
            column: slice("column", &column)?,
            pos0: slice("pos0", &pos0)?,
            real: slice("real", &real)?,
            state_src: slice("state_src", &state_src)?,
            state_dst: slice("state_dst", &state_dst)?,
            blocks_indptr: slice("blocks_indptr", &blocks_indptr)?,
            blocks: slice("blocks", &blocks)?,
        };
        let n = rows.column.len();
        if [rows.pos0, rows.real, rows.state_src, rows.state_dst]
            .iter()
            .any(|a| a.len() != n)
            || rows.blocks_indptr.len() != n + 1
        {
            return Err(PyValueError::new_err(
                "every row array needs one value a row, blocks_indptr one more",
            ));
        }
        let tokens = match &tokens {
            Some(tokens) => Some(slice("tokens", tokens)?),
            None => None,
        };
        let host = host
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("host must be a contiguous uint8 array"))?;
        let step = glm::Step {
            tokens,
            columns,
            first_position,
            head,
            verify,
            keep_tails,
            meta_columns,
        };
        let model = &self.model;
        let counts = py
            .detach(|| glm::fill(model, tables, host, &rows, &step))
            .map_err(PyValueError::new_err)?;
        Ok((
            counts.index,
            counts.sparse,
            counts.slots,
            counts.index_launches,
            counts.uploads,
        ))
    }
}

/// `resin_gaudi::draft_choice::DraftChoice` for tests: the native draft policy, call for call with
/// `draft_choice.DraftChoice`.
#[pyclass(module = "resin._resin")]
pub(crate) struct GlmDraftPolicy {
    choice: draft_choice::DraftChoice,
}

#[pymethods]
impl GlmDraftPolicy {
    #[new]
    #[pyo3(signature = (k, costs, *, max_rows, prior, buckets, shared, mode))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k: usize,
        costs: &Bound<'_, PyDict>,
        max_rows: usize,
        prior: f64,
        buckets: Vec<usize>,
        shared: bool,
        mode: &str,
    ) -> PyResult<Self> {
        let costs = draft_choice::StepCosts {
            plain_0: item(costs, "plain_0")?,
            plain_row: item(costs, "plain_row")?,
            launch_0: item(costs, "launch_0")?,
            launch_row: item(costs, "launch_row")?,
            scale: item(costs, "scale")?,
            group: item::<usize>(costs, "group")?.max(1),
        };
        let mode = draft_choice::Mode::of(mode).map_err(PyValueError::new_err)?;
        Ok(Self {
            choice: draft_choice::DraftChoice::new(
                k, costs, max_rows, prior, &buckets, shared, mode,
            ),
        })
    }

    fn speculate(&mut self, sequences: Vec<u64>) -> bool {
        self.choice.speculate(&sequences)
    }

    fn sync_due(&self) -> bool {
        self.choice.sync_due()
    }

    fn table(&self) -> HashMap<(String, usize), f64> {
        self.choice
            .times
            .table()
            .into_iter()
            .map(|((kind, bucket), ms)| ((kind.name().to_owned(), bucket), ms))
            .collect()
    }

    fn set_shared(&mut self, table: HashMap<(String, usize), f64>) {
        self.choice.set_shared(
            table
                .into_iter()
                .filter_map(|((kind, bucket), ms)| {
                    draft_choice::Kind::of(&kind).map(|k| ((k, bucket), ms))
                })
                .collect(),
        );
    }

    fn measured(&mut self, kind: &str, rows: usize, ms: f64) -> PyResult<()> {
        let kind = draft_choice::Kind::of(kind)
            .ok_or_else(|| PyValueError::new_err(format!("no step kind {kind}")))?;
        self.choice.measured(kind, rows, ms);
        Ok(())
    }

    #[pyo3(signature = (sequences, accepted, own=None))]
    fn verified(&mut self, sequences: Vec<u64>, accepted: Vec<i64>, own: Option<Vec<bool>>) {
        self.choice.verified(&sequences, &accepted, own.as_deref());
    }

    fn forget(&mut self, sequences: Vec<u64>) {
        self.choice.forget(&sequences);
    }

    fn ms(&self, kind: &str, rows: usize) -> PyResult<f64> {
        let kind = draft_choice::Kind::of(kind)
            .ok_or_else(|| PyValueError::new_err(format!("no step kind {kind}")))?;
        Ok(self.choice.ms(kind, rows))
    }

    fn expected(&self, sequence: u64) -> f64 {
        self.choice.acceptance.expected(sequence)
    }
}
