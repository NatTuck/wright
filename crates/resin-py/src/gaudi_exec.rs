//! Pieces every native Gaudi step executor shares: pinned host images and staging buffers, the host timings,
//! the logprob columns of a step's output and of a prefill's prompt scores, and the STEP frames that a
//! tensor-parallel driver sends its followers.

use std::time::Instant;

use numpy::PyReadwriteArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use resin_gaudi::sampling::{self, PARAMS, TOP_MAX as TOP};

use resin_core::resources::kv::{KvGroupFamily, KvLayout};

use crate::{BatchPhase, BatchPlan};

/// A runner error after the step's device work began: the Engine must not reuse the step's pages.
pub(crate) fn post_accept(py: Python<'_>, message: String) -> PyErr {
    let error = PyRuntimeError::new_err(message);
    let _ = error.value(py).setattr("_resin_post_accept", true);
    error
}

pub(crate) fn since(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

/// Pinned host memory of a Python array (a flat uint8 view), kept alive with it.
pub(crate) struct Pinned {
    pub(crate) address: usize,
    pub(crate) bytes: usize,
    pub(crate) _owner: Py<PyAny>,
}

impl Pinned {
    pub(crate) fn new(name: &str, mut array: PyReadwriteArray1<'_, u8>) -> PyResult<Self> {
        let owner = array.as_any().clone().unbind();
        let slice = array.as_slice_mut().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a contiguous uint8 array"))
        })?;
        if !(slice.as_ptr() as usize).is_multiple_of(4) {
            return Err(PyValueError::new_err(format!(
                "{name} must be 4-byte aligned"
            )));
        }
        Ok(Self {
            address: slice.as_mut_ptr() as usize,
            bytes: slice.len(),
            _owner: owner,
        })
    }

    pub(crate) fn at_least(
        name: &str,
        array: PyReadwriteArray1<'_, u8>,
        bytes: usize,
    ) -> PyResult<Self> {
        let pinned = Self::new(name, array)?;
        if pinned.bytes < bytes {
            return Err(PyValueError::new_err(format!(
                "{name} holds {} bytes, not {bytes}",
                pinned.bytes
            )));
        }
        Ok(pinned)
    }

    /// `count` 32-bit words at byte `offset`.
    pub(crate) fn words<T: Copy>(&self, offset: usize, count: usize) -> Vec<T> {
        assert!(offset + 4 * count <= self.bytes && size_of::<T>() == 4);
        // SAFETY: inside the pinned allocation (asserted), 4-byte aligned; the device's copies into it are
        // complete once the staging event is done, which the caller waited for.
        unsafe { std::slice::from_raw_parts((self.address + offset) as *const T, count).to_vec() }
    }

    /// The `len` bytes at byte `offset`, to write.
    ///
    /// # Safety
    /// No device copy reads or writes the range until the caller's writes are done (its event was waited for).
    pub(crate) unsafe fn bytes_mut<'a>(&self, offset: usize, len: usize) -> &'a mut [u8] {
        assert!(offset + len <= self.bytes);
        unsafe { std::slice::from_raw_parts_mut((self.address + offset) as *mut u8, len) }
    }
}

/// A device table image with two pinned host copies: a step fills the copy whose last upload is done (its
/// event), and its program uploads that copy.
pub(crate) struct Images {
    pub(crate) device: u64,
    pub(crate) host: Pinned,
    pub(crate) stride: usize,
    pub(crate) events: i32,
    pub(crate) used: [bool; 2],
    pub(crate) next: usize,
}

impl Images {
    pub(crate) fn new(device: u64, host: Pinned, stride: usize, events: i32) -> PyResult<Self> {
        if host.bytes < 2 * stride || !stride.is_multiple_of(4) {
            return Err(PyValueError::new_err(
                "an image's host copies must hold two images",
            ));
        }
        Ok(Self {
            device,
            host,
            stride,
            events,
            used: [false; 2],
            next: 0,
        })
    }
}

/// Host nanoseconds of the native steps, cumulative: filling the tables (`fill`), submitting the programs'
/// calls (`submit`), waiting for a host image to be free (`image_wait`) and for the outputs (`output_wait`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Timings {
    pub(crate) fill: u64,
    pub(crate) submit: u64,
    pub(crate) image_wait: u64,
    pub(crate) output_wait: u64,
    /// at the gates: waiting until every rank is ready
    pub(crate) gate: u64,
    /// the driver sending each call to the followers
    pub(crate) announce: u64,
}

/// Row `r`'s sampler settings in `plan` (effective seeds are required).
pub(crate) fn row_sample(plan: &BatchPlan, r: usize) -> PyResult<sampling::Sample> {
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
pub(crate) fn plan_params(plan: &BatchPlan, vocab: u64) -> PyResult<Vec<[f32; PARAMS]>> {
    (0..plan.sequence_ids.len())
        .map(|r| {
            let query = u64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
            let offset = (u64::from(plan.context_lengths[r]) + query) * sampling::OFFSET_STRIDE;
            Ok(sampling::row_params(&row_sample(plan, r)?, vocab, offset))
        })
        .collect()
}

/// `Sampling(plan_params, seeds, (positions + 1) x 32).at(rows, [0] x rows)`: each row's sampler inputs with
/// its uniform at its exact next position (a continuation's plan only bounds the positions).
pub(crate) fn params_at(
    plan: &BatchPlan,
    vocab: u64,
    positions: &[i64],
) -> PyResult<Vec<[f32; PARAMS]>> {
    (0..plan.sequence_ids.len())
        .map(|r| {
            let offset = (positions[r] + 1) as u64 * sampling::OFFSET_STRIDE;
            Ok(sampling::row_params(&row_sample(plan, r)?, vocab, offset))
        })
        .collect()
}

/// Row `r`'s blocks of cache group `index` as first arena rows (slab x `slab_rows` + slot x `block_rows`), and the
/// logical block of the first.
pub(crate) fn group_blocks(
    plan: &BatchPlan,
    index: usize,
    r: usize,
    slab_rows: i64,
    block_rows: i64,
) -> PyResult<(Vec<i64>, i64)> {
    let group = plan
        .cache_groups
        .get(index)
        .map(|group| group.inner_native())
        .ok_or_else(|| PyValueError::new_err(format!("the plan has no cache group {index}")))?;
    let (lo, hi) = (
        group.block_indptr[r] as usize,
        group.block_indptr[r + 1] as usize,
    );
    let blocks = (lo..hi)
        .map(|i| {
            i64::from(group.block_slab_ids[i]) * slab_rows
                + i64::from(group.block_slotnums[i]) * block_rows
        })
        .collect();
    Ok((blocks, i64::from(group.block_table_start_indices[r])))
}

/// The smallest of `buckets` that holds `n` columns.
pub(crate) fn bucket_of(buckets: &[usize], n: usize, prefill: bool) -> PyResult<usize> {
    buckets.iter().copied().find(|&b| n <= b).ok_or_else(|| {
        let kind = if prefill { "prefill" } else { "decode" };
        PyValueError::new_err(format!("{n} columns exceed the {kind} buckets"))
    })
}

/// The sampler's outputs in a staging set: byte offsets of the tokens, logprobs, top ids and top logprobs.
#[derive(Clone, Copy)]
pub(crate) struct Sampled {
    pub(crate) tok: usize,
    pub(crate) lp: usize,
    pub(crate) top_idx: usize,
    pub(crate) top_lp: usize,
}

/// A step's outputs read from a staging set: per column its token, logprob and top list.
pub(crate) struct Outputs {
    pub(crate) tokens: Vec<i32>,
    pub(crate) lp: Vec<f32>,
    pub(crate) top_ids: Vec<i32>,
    pub(crate) top_lp: Vec<f32>,
}

impl Outputs {
    pub(crate) fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            lp: Vec::new(),
            top_ids: Vec::new(),
            top_lp: Vec::new(),
        }
    }

    pub(crate) fn extend(&mut self, other: Outputs) {
        self.tokens.extend(other.tokens);
        self.lp.extend(other.lp);
        self.top_ids.extend(other.top_ids);
        self.top_lp.extend(other.top_lp);
    }

    /// The first `n` columns' samples of a staging set at byte `base` of `staging`.
    pub(crate) fn read(staging: &Pinned, base: usize, s: Sampled, n: usize) -> Self {
        Self {
            tokens: staging.words(base + s.tok, n),
            lp: staging.words(base + s.lp, n),
            top_ids: staging.words(base + s.top_idx, n * TOP),
            top_lp: staging.words(base + s.top_lp, n * TOP),
        }
    }

    /// The output of a step that sampled one token a row (`plan_output`): with logprobs when the plan wants them.
    pub(crate) fn output(
        &self,
        py: Python<'_>,
        plan_logprobs: bool,
        params: &[[f32; PARAMS]],
        prompt: Option<crate::output::PromptColumns>,
    ) -> PyResult<Py<crate::output::ForwardOutput>> {
        let n = self.tokens.len();
        let token_ids = self.tokens.iter().map(|&t| t as u32).collect();
        let indptr = (0..=n as u32).collect();
        let (scores, top) = if plan_logprobs {
            let (s, t) = score_columns(params, &vec![1; n], &self.lp, &self.top_ids, &self.top_lp);
            (Some(s), t)
        } else {
            (None, None)
        };
        Py::new(
            py,
            crate::output::ForwardOutput::native_with_prompt(
                py, token_ids, indptr, scores, top, prompt,
            )?,
        )
    }
}

/// Write the sampler's table into `table` (a pinned copy that only a later ring upload reads): the rows'
/// parameters, then `last` at byte `last_at` (`_sample_launch`).
pub(crate) fn write_sample_table(
    table: &mut [u8],
    params: &[[f32; PARAMS]],
    last_at: usize,
    last: i64,
) {
    table.fill(0);
    for (r, row) in params.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let at = 4 * (r * PARAMS + j);
            table[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    table[last_at..last_at + 4].copy_from_slice(&(last as i32).to_le_bytes());
}

pub(crate) type ScoreColumns = ((Vec<f32>, Vec<u32>), Option<(Vec<u32>, Vec<u32>, Vec<f32>)>);

/// The logprob columns of `resin.backend.sampling.format_output` (`common.scoring.score_columns`): `params`
/// per row, `counts` output tokens a row, and per output token its logprob and top list.
pub(crate) fn score_columns(
    params: &[[f32; PARAMS]],
    counts: &[usize],
    lp: &[f32],
    top_ids: &[i32],
    top_lp: &[f32],
) -> ScoreColumns {
    let mut values = Vec::new();
    let mut indptr = vec![0u32];
    let (mut tindptr, mut tids, mut tvalues) = (vec![0u32], Vec::new(), Vec::new());
    let mut any_top = false;
    let mut token = 0usize;
    for (p, &count) in params.iter().zip(counts) {
        let scored = p[6] > 0.5;
        for _ in 0..count {
            if scored {
                values.push(lp[token]);
                let ntop = p[5] as usize;
                any_top |= ntop > 0;
                for j in 0..ntop.min(TOP) {
                    tids.push(top_ids[token * TOP + j] as u32);
                    tvalues.push(top_lp[token * TOP + j]);
                }
                tindptr.push(tids.len() as u32);
            }
            token += 1;
        }
        indptr.push(values.len() as u32);
    }
    let top = any_top.then_some((tindptr, tids, tvalues));
    ((values, indptr), top)
}

/// A prefill row's prompt positions to score (`common.scoring.ScoreRequest`): indices into its tokens, and each
/// one's target tokens (the next prompt token first) and top-list length.
pub(crate) struct ScoreRequest {
    pub(crate) positions: Vec<usize>,
    pub(crate) targets: Vec<Vec<u32>>,
    pub(crate) tops: Vec<usize>,
}

/// A scored position (`common.scoring.Scores`): its targets' logprobs, and its top ids and logprobs.
#[derive(Clone, Default)]
pub(crate) struct Scored {
    pub(crate) targets: Vec<f32>,
    pub(crate) top_ids: Vec<u32>,
    pub(crate) top_lp: Vec<f32>,
}

/// Each row's prompt positions to score (`common.scoring.score_requests`).
pub(crate) fn score_requests(plan: &BatchPlan, vocab: u64) -> Vec<Option<ScoreRequest>> {
    let indices = &plan.prompt_score_indices;
    let extra_ptr = &plan.prompt_target_indptr;
    (0..plan.sequence_ids.len())
        .map(|r| {
            let (lo, hi) = (plan.token_indptr[r], plan.token_indptr[r + 1]);
            let inside: Vec<usize> = (0..indices.len())
                .filter(|&k| indices[k] >= lo && indices[k] < hi)
                .collect();
            if inside.is_empty() {
                return None;
            }
            Some(ScoreRequest {
                positions: inside.iter().map(|&k| (indices[k] - lo) as usize).collect(),
                targets: inside
                    .iter()
                    .map(|&k| {
                        let mut targets = vec![plan.prompt_score_token_ids[k]];
                        if extra_ptr.len() > 1 {
                            targets.extend_from_slice(
                                &plan.prompt_target_token_ids
                                    [extra_ptr[k] as usize..extra_ptr[k + 1] as usize],
                            );
                        }
                        targets
                    })
                    .collect(),
                tops: inside
                    .iter()
                    .map(|&k| u64::from(plan.prompt_score_top_counts[k]).min(vocab) as usize)
                    .collect(),
            })
        })
        .collect()
}

/// NumPy's pairwise sum of f64 values (`pairwise_sum`: eight accumulators below 128 values, halves above), so
/// that host scores round as the numpy scorer's did.
fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |acc, &v| acc + v);
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                r[j] += values[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[i..] {
            res += v;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// One prompt position's scores from its logits `z` over the whole vocabulary (`_Scorer.add`): the f64 log-softmax
/// at `targets`, and the `top` largest logprobs' ids (descending, ties by id) and values, as f32.
pub(crate) fn score_position(z: &[f32], targets: &[u32], top: usize) -> Scored {
    let m = z
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    let shifted: Vec<f64> = z.iter().map(|&v| f64::from(v) - m).collect();
    let exps: Vec<f64> = shifted.iter().map(|&v| v.exp()).collect();
    let log_sum = pairwise_sum(&exps).ln();
    let logp = |i: usize| shifted[i] - log_sum;
    let top = top.min(z.len());
    let mut ids: Vec<u32> = if top > 0 {
        (0..z.len() as u32).collect()
    } else {
        Vec::new()
    };
    // a stable descending order by logprob (logp's order is z's)
    let order = |a: &u32, b: &u32| {
        let (x, y) = (shifted[*a as usize], shifted[*b as usize]);
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if top > 0 && top < ids.len() {
        ids.select_nth_unstable_by(top - 1, order);
        ids.truncate(top);
    }
    ids.sort_by(order);
    ids.truncate(top);
    Scored {
        targets: targets.iter().map(|&t| logp(t as usize) as f32).collect(),
        top_lp: ids.iter().map(|&i| logp(i as usize) as f32).collect(),
        top_ids: ids,
    }
}

/// The prompt score columns of a prefill (`common.scoring.prompt_columns`): every row's scored positions in order.
pub(crate) fn prompt_columns(
    plan: &BatchPlan,
    scored: &[Vec<Scored>],
) -> crate::output::PromptColumns {
    let all: Vec<&Scored> = scored.iter().flatten().collect();
    let offsets = |counts: &mut dyn Iterator<Item = usize>| -> Vec<u32> {
        let mut out = vec![0u32];
        for count in counts {
            out.push(out.last().copied().unwrap_or(0) + count as u32);
        }
        out
    };
    let top = all.iter().any(|s| !s.top_ids.is_empty()).then(|| {
        (
            offsets(&mut all.iter().map(|s| s.top_ids.len())),
            all.iter().flat_map(|s| s.top_ids.iter().copied()).collect(),
            all.iter().flat_map(|s| s.top_lp.iter().copied()).collect(),
        )
    });
    let targets = all.iter().any(|s| s.targets.len() > 1).then(|| {
        (
            offsets(&mut all.iter().map(|s| s.targets.len() - 1)),
            plan.prompt_target_token_ids.clone(),
            all.iter()
                .flat_map(|s| s.targets[1..].iter().copied())
                .collect(),
        )
    });
    crate::output::PromptColumns {
        logprobs: all.iter().map(|s| s.targets[0]).collect(),
        indptr: plan.prompt_score_indptr.clone(),
        top,
        targets,
    }
}

pub(crate) const STEP_HEAD: usize = 16;
/// The start of the error that a gate gives when another rank failed.
pub(crate) const RANK_FAILED: &str = "a rank failed: ";
pub(crate) const SAMPLE_BYTES: usize = 56;

/// A STEP frame's payload: the call, whether it continues the last step, the driver's draft decision and fixed
/// acceptance, the plan's
/// wire archive with each row's sampling inputs, and the sequences to release.
pub(crate) fn encode_step(
    op: u8,
    continues: bool,
    speculating: bool,
    fixed: Option<usize>,
    plan: Option<&BatchPlan>,
    ids: &[u64],
) -> PyResult<Vec<u8>> {
    let fixed = match fixed {
        None => 0,
        Some(f) if f < 255 => f as u8 + 1,
        Some(f) => {
            return Err(PyValueError::new_err(format!(
                "a fixed acceptance of {f} drafts does not fit a step frame"
            )));
        }
    };
    let wire = match plan {
        Some(plan) => plan.encode_wire()?,
        None => Vec::new(),
    };
    let rows = plan.map_or(0, |p| p.sampling_params.len());
    let mut out = Vec::with_capacity(STEP_HEAD + wire.len() + rows * SAMPLE_BYTES + 8 * ids.len());
    out.extend_from_slice(&[op, u8::from(continues), u8::from(speculating), fixed]);
    for value in [rows as u32, wire.len() as u32, ids.len() as u32] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&wire);
    if let Some(plan) = plan {
        for p in &plan.sampling_params {
            out.extend_from_slice(&p.temperature.to_le_bytes());
            out.extend_from_slice(&p.top_p.to_le_bytes());
            out.extend_from_slice(&p.top_k.map_or(-1i64, i64::from).to_le_bytes());
            out.extend_from_slice(&p.seed.unwrap_or(0).to_le_bytes());
            for value in [
                u32::from(p.logprobs),
                p.top_logprobs as u32,
                u32::from(p.seed.is_some()),
                u32::from(p.prompt_logprobs),
                p.prompt_top_logprobs as u32,
                0,
            ] {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    Ok(out)
}

/// A decoded STEP frame.
pub(crate) struct StepFrame {
    pub(crate) op: u8,
    pub(crate) continues: bool,
    pub(crate) speculating: bool,
    /// the driver's benchmark acceptance (`fixed_acceptance`), which every rank must accept alike
    pub(crate) fixed: Option<usize>,
    pub(crate) plan: Option<BatchPlan>,
    pub(crate) ids: Vec<u64>,
}

pub(crate) fn decode_step(bytes: &[u8]) -> PyResult<StepFrame> {
    let bad = || PyValueError::new_err("a malformed step frame");
    if bytes.len() < STEP_HEAD {
        return Err(bad());
    }
    let word =
        |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes")) as usize;
    let (rows, wire, ids) = (word(4), word(8), word(12));
    if bytes.len() != STEP_HEAD + wire + rows * SAMPLE_BYTES + 8 * ids {
        return Err(bad());
    }
    let mut at = STEP_HEAD + wire;
    let plan = if wire > 0 {
        let mut plan = BatchPlan::decode_wire(&bytes[STEP_HEAD..STEP_HEAD + wire])?;
        let f64_at = |at: usize| f64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
        let i64_at = |at: usize| i64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
        let mut params = Vec::with_capacity(rows);
        for _ in 0..rows {
            let top_k = i64_at(at + 16);
            params.push(crate::SamplingParams {
                temperature: f64_at(at),
                top_p: f64_at(at + 8),
                top_k: (top_k >= 0).then_some(top_k as u32),
                seed: (word(at + 40) != 0).then_some(i64_at(at + 24)),
                logprobs: word(at + 32) != 0,
                top_logprobs: word(at + 36),
                prompt_logprobs: word(at + 44) != 0,
                prompt_top_logprobs: word(at + 48),
                ..crate::SamplingParams::default()
            });
            at += SAMPLE_BYTES;
        }
        plan.media = vec![None; params.len()];
        plan.sampling_params = params;
        Some(BatchPlan::checked(plan)?)
    } else {
        None
    };
    let ids = (0..ids)
        .map(|i| {
            u64::from_le_bytes(
                bytes[at + 8 * i..at + 8 * i + 8]
                    .try_into()
                    .expect("8 bytes"),
            )
        })
        .collect();
    Ok(StepFrame {
        op: bytes[0],
        continues: bytes[1] != 0,
        speculating: bytes[2] != 0,
        fixed: bytes[3].checked_sub(1).map(usize::from),
        plan,
        ids,
    })
}

/// The checks of `resin.backend.plans.PagedCachePlan.from_plan`: the plan's cache groups follow `layout` (slab
/// packing, physical bounds), every row's block table is its exact active range (a decode row's leased past its
/// token by `decode_lookahead`), a WINDOW group's write span fits its read and write capacity, and every append
/// goes to the row's own block at the token's offset. The first failing row raises, as there.
pub(crate) fn check_paged_plan(
    plan: &BatchPlan,
    layout: &KvLayout,
    decode_lookahead: u64,
) -> Result<(), String> {
    let groups = layout.groups();
    if plan.cache_groups.len() != groups.len() {
        return Err("BatchPlan cache groups do not match CacheLayout".into());
    }
    let rows = plan.sequence_ids.len();
    let lookahead = if plan.phase == BatchPhase::Decode {
        decode_lookahead
    } else {
        0
    };
    let capacity = layout.parent_capacity().get();
    let query = |r: usize| u64::from(plan.token_indptr[r + 1] - plan.token_indptr[r]);
    for (index, (descriptor, native)) in groups.iter().zip(&plan.cache_groups).enumerate() {
        let group = native.inner_native();
        let block_tokens = u64::from(descriptor.block_tokens().get());
        if group.blocks_per_slab != descriptor.blocks_per_slab().get() {
            return Err(format!("cache group {index} has the wrong slab packing"));
        }
        if group
            .append_block_offsets
            .iter()
            .any(|&o| u64::from(o) >= block_tokens)
            || group.block_slab_ids.iter().any(|&s| s > capacity)
            || group.append_slab_ids.iter().any(|&s| s > capacity)
        {
            return Err(format!(
                "cache group {index} has a physical address outside its layout"
            ));
        }
        let window_blocks = descriptor
            .window_resident_blocks()
            .map(|b| u64::from(b.get()));
        let maximum = match window_blocks {
            Some(_) => descriptor
                .max_write_tokens()
                .map_or(0, |m| u64::from(m.get())),
            None => 0,
        };
        for r in 0..rows {
            let write = query(r) + lookahead;
            let leased = u64::from(plan.context_lengths[r]) + write;
            let span = window_blocks.is_some() && write > maximum;
            let last = (leased.max(1) - 1) / block_tokens;
            let first = if descriptor.family() == KvGroupFamily::State {
                last.saturating_sub(1)
            } else if let Some(blocks) = window_blocks {
                (last + 1).saturating_sub(blocks)
            } else {
                0
            };
            let start = u64::from(group.block_table_start_indices[r]);
            let (lo, hi) = (
                group.block_indptr[r] as usize,
                group.block_indptr[r + 1] as usize,
            );
            let range = start != first || (hi - lo) as u64 != last - first + 1;
            let mut append = false;
            if !range {
                let context = u64::from(plan.context_lengths[r]);
                for (j, token) in (plan.token_indptr[r]..plan.token_indptr[r + 1]).enumerate() {
                    let position = context + j as u64;
                    let block = position / block_tokens;
                    if block < first {
                        continue;
                    }
                    let (t, at) = (token as usize, lo + (block - first) as usize);
                    append |= group.append_slab_ids[t] != group.block_slab_ids[at]
                        || group.append_slotnums[t] != group.block_slotnums[at]
                        || u64::from(group.append_block_offsets[t]) != position % block_tokens;
                }
            }
            if span {
                return Err(format!(
                    "cache group {index} row {r} write span {write} exceeds {maximum}-token Window read/write capacity"
                ));
            }
            if range {
                return Err(format!(
                    "cache group {index} row {r} is not its exact active range"
                ));
            }
            if append {
                return Err(format!(
                    "cache group {index} row {r} append target is not row-owned"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaudi_ranks::{OP_ENQUEUE, OP_FORWARD, OP_RELEASE};

    #[test]
    fn step_frames_round_trip() {
        Python::initialize();
        let mut plan = crate::tests::wire_plan(0);
        plan.sampling_params = vec![crate::SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            seed: Some(123),
            logprobs: true,
            top_logprobs: 5,
            prompt_logprobs: true,
            prompt_top_logprobs: 2,
            ..crate::SamplingParams::default()
        }];
        let bytes = encode_step(OP_ENQUEUE, true, true, Some(0), Some(&plan), &[]).unwrap();
        let frame = decode_step(&bytes).unwrap();
        assert_eq!(
            (frame.op, frame.continues, frame.speculating, frame.fixed),
            (OP_ENQUEUE, true, true, Some(0))
        );
        let back = frame.plan.unwrap();
        assert_eq!(back.token_ids, plan.token_ids);
        assert_eq!(back.cache_groups, plan.cache_groups);
        let (a, b) = (&back.sampling_params[0], &plan.sampling_params[0]);
        assert_eq!(
            (
                a.temperature,
                a.top_p,
                a.top_k,
                a.seed,
                a.logprobs,
                a.top_logprobs
            ),
            (
                b.temperature,
                b.top_p,
                b.top_k,
                b.seed,
                b.logprobs,
                b.top_logprobs
            )
        );
        assert_eq!((a.prompt_logprobs, a.prompt_top_logprobs), (true, 2));
        assert_eq!(back.media.len(), 1);
        // no top-k
        plan.sampling_params[0].top_k = None;
        let frame =
            decode_step(&encode_step(OP_FORWARD, false, false, Some(7), Some(&plan), &[]).unwrap())
                .unwrap();
        assert_eq!(frame.fixed, Some(7));
        assert_eq!(frame.plan.unwrap().sampling_params[0].top_k, None);
        assert!(encode_step(OP_FORWARD, false, false, Some(255), Some(&plan), &[]).is_err());
        // a release: ids, no plan
        let frame =
            decode_step(&encode_step(OP_RELEASE, false, false, None, None, &[4, 1 << 40]).unwrap())
                .unwrap();
        assert_eq!(
            (frame.op, frame.ids, frame.fixed),
            (OP_RELEASE, vec![4, 1 << 40], None)
        );
        assert!(frame.plan.is_none());
        assert!(decode_step(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn score_columns_keep_scored_rows() {
        let mut params = [[0f32; PARAMS]; 2];
        params[1][6] = 1.0;
        params[1][5] = 2.0;
        // row 0 one token (not scored), row 1 two tokens (scored, 2 top entries each)
        let lp = [0.5, -1.0, -2.0];
        let mut ids = vec![0i32; 3 * TOP];
        ids[TOP] = 7;
        ids[TOP + 1] = 9;
        ids[2 * TOP] = 3;
        ids[2 * TOP + 1] = 4;
        let values = vec![-0.25f32; 3 * TOP];
        let ((v, indptr), top) = score_columns(&params, &[1, 2], &lp, &ids, &values);
        assert_eq!((v, indptr), (vec![-1.0, -2.0], vec![0, 0, 2]));
        let (tindptr, tids, _) = top.unwrap();
        assert_eq!((tindptr, tids), (vec![0, 2, 4], vec![7, 9, 3, 4]));
    }
}
