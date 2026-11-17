//! The host side of the device sampler, shared by every model: each row's sampler inputs
//! (`resin_models_gaudi/common/sampling.py`: `philox_uniform`, `row_params` with `plan_params`' semantics).

/// sampler inputs per row: temperature, top-p, top-k, uniform, mode, top logprobs, scored, (unused)
pub const PARAMS: usize = 8;
/// the longest top list the sampler returns
pub const TOP_MAX: usize = 64;
/// Philox offsets per position (`resin.backend.sampling.SAMPLING_OFFSET_STRIDE`)
pub const OFFSET_STRIDE: u64 = 32;

/// f64 uniform (r + 0.5) * 2^-32 of Philox4x32-10, key = seed, counter = (0, 0, offset).
pub fn philox_uniform(seed: u64, offset: u64) -> f64 {
    const MASK: u64 = 0xFFFF_FFFF;
    let (mut c0, mut c1, mut c2, mut c3) = (0u64, 0u64, offset & MASK, offset >> 32);
    let (mut k0, mut k1) = (seed & MASK, seed >> 32);
    for _ in 0..10 {
        let a = 0xD251_1F53u64 * c0;
        let b = 0xCD9E_8D57u64 * c2;
        (c0, c1, c2, c3) = ((b >> 32) ^ c1 ^ k0, b & MASK, (a >> 32) ^ c3 ^ k1, a & MASK);
        k0 = (k0 + 0x9E37_79B9) & MASK;
        k1 = (k1 + 0xBB67_AE85) & MASK;
    }
    (c0 as f64 + 0.5) * 2f64.powi(-32)
}

/// One row's sampling request (`resin` `SamplingParams` with its effective seed).
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub temperature: f64,
    pub top_p: f64,
    /// None: the whole vocabulary
    pub top_k: Option<u64>,
    pub seed: u64,
    /// the row samples (the plan's sampling mask)
    pub sample: bool,
    pub logprobs: bool,
    pub top_logprobs: u64,
}

/// A row's PARAMS sampler inputs (`row_params` with `plan_params`' semantics) at Philox offset `offset`.
pub fn row_params(sample: &Sample, vocab: u64, offset: u64) -> [f32; PARAMS] {
    let mut out = [0f32; PARAMS];
    let t = sample.temperature;
    out[0] = t as f32;
    out[1] = sample.top_p as f32;
    let k = sample.top_k.unwrap_or(vocab).min(vocab);
    out[2] = if k >= vocab { 0.0 } else { k as f32 };
    let mode = sample.sample && t > 0.0 && t.is_finite();
    if mode {
        out[3] = philox_uniform(sample.seed, offset) as f32;
    }
    out[4] = mode as u8 as f32;
    let scored = sample.sample && sample.logprobs;
    if scored {
        out[5] = sample.top_logprobs.min(vocab).min(TOP_MAX as u64) as f32;
        out[6] = 1.0;
    }
    out
}

/// The PARAMS of the `width` columns of a verify row whose first column is at position `start`: column j samples the
/// token at position start + 1 + j with that position's uniform (offset (start + 1 + j) x OFFSET_STRIDE), as a plain
/// decode step at position start + j does. So seeded output with drafts is the seeded output without drafts, and the
/// tokens a step emits are sampled with independent uniforms. With width 1 this is a plain decode row at `start`.
pub fn column_params(sample: &Sample, vocab: u64, start: i64, width: usize) -> Vec<[f32; PARAMS]> {
    (0..width as i64)
        .map(|j| {
            let position = u64::try_from(start + 1 + j).expect("positions are not negative");
            row_params(sample, vocab, position * OFFSET_STRIDE)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_matches_the_reference_values() {
        // common.sampling.philox_uniform of the same seeds and offsets
        assert_eq!(philox_uniform(1234, 160), 0.44745189289096743);
        assert_eq!(
            philox_uniform((1 << 40) + 7, (1 << 33) + 5),
            0.005502560990862548
        );
        assert_eq!(philox_uniform(0, 0), 0.3990464707603678);
    }

    #[test]
    fn a_verify_column_samples_as_the_plain_step_at_its_position() {
        let sample = Sample {
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(20),
            seed: 1234,
            sample: true,
            logprobs: false,
            top_logprobs: 0,
        };
        let columns = column_params(&sample, 1000, 100, 4);
        assert_eq!(columns.len(), 4);
        for (j, column) in columns.iter().enumerate() {
            // the plain decode step at position 100 + j: its uniform at offset (100 + j + 1) x 32
            assert_eq!(column, &column_params(&sample, 1000, 100 + j as i64, 1)[0]);
            assert_eq!(
                column,
                &row_params(&sample, 1000, (101 + j as u64) * OFFSET_STRIDE)
            );
        }
        // each emitted token has its own uniform
        let uniforms: Vec<f32> = columns.iter().map(|c| c[3]).collect();
        for a in 0..4 {
            for b in a + 1..4 {
                assert_ne!(uniforms[a], uniforms[b]);
            }
        }
        // greedy rows have no uniform
        let greedy = Sample {
            temperature: 0.0,
            ..sample
        };
        assert!(
            column_params(&greedy, 1000, 100, 4)
                .iter()
                .all(|c| c[3] == 0.0 && c[4] == 0.0)
        );
    }
}
