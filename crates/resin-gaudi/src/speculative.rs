//! Acceptance of a verify step's drafts, shared by every model's native executor
//! (`resin.backend.speculative`: `accept_speculative_tokens`, `SpeculativeStats`).

/// Each row's accepted drafts (the leading drafts its samples repeat, or `fixed` of them: the benchmark's
/// synthetic acceptance) and emitted tokens (them, then the sample after them), from `drafts`
/// (rows x (width - 1)) and `samples` (rows x width).
pub fn accept(
    drafts: &[i64],
    samples: &[i64],
    rows: usize,
    width: usize,
    fixed: Option<usize>,
) -> (Vec<i64>, Vec<Vec<i64>>) {
    let k = width - 1;
    let mut accepted = Vec::with_capacity(rows);
    let mut emitted = Vec::with_capacity(rows);
    for b in 0..rows {
        let (d, y) = (
            &drafts[b * k..(b + 1) * k],
            &samples[b * width..(b + 1) * width],
        );
        let a = match fixed {
            Some(fixed) => fixed.min(k),
            None => d.iter().zip(y).take_while(|(d, y)| d == y).count(),
        };
        let mut tokens = d[..a].to_vec();
        tokens.push(y[a]);
        accepted.push(a as i64);
        emitted.push(tokens);
    }
    (accepted, emitted)
}

/// The verify steps' acceptance counters (`SpeculativeStats`), and the benchmark's fixed acceptance.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub blocks: u64,
    pub drafts: u64,
    pub accepted: u64,
    pub emitted: u64,
    /// accept this many drafts of every row, whatever the samples (`benchmark_accepted_drafts`)
    pub fixed: Option<usize>,
}

impl Stats {
    /// `accept`, counted.
    pub fn accept(
        &mut self,
        drafts: &[i64],
        samples: &[i64],
        rows: usize,
        width: usize,
    ) -> (Vec<i64>, Vec<Vec<i64>>) {
        let (accepted, emitted) = accept(drafts, samples, rows, width, self.fixed);
        self.blocks += rows as u64;
        self.drafts += (rows * (width - 1)) as u64;
        self.accepted += accepted.iter().sum::<i64>() as u64;
        self.emitted += emitted.iter().map(Vec::len).sum::<usize>() as u64;
        (accepted, emitted)
    }

    /// (blocks, drafts, accepted, emitted)
    pub fn counts(&self) -> (u64, u64, u64, u64) {
        (self.blocks, self.drafts, self.accepted, self.emitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_takes_the_matching_prefix() {
        let (accepted, emitted) =
            accept(&[5, 6, 7, 1, 2, 3], &[5, 6, 9, 0, 4, 4, 4, 4], 2, 4, None);
        assert_eq!(accepted, vec![2, 0]);
        assert_eq!(emitted, vec![vec![5, 6, 9], vec![4]]);
    }

    #[test]
    fn fixed_acceptance_keeps_the_target_s_correction() {
        // accept_speculative_tokens(..., benchmark_accepted_drafts=2)
        let mut stats = Stats {
            fixed: Some(2),
            ..Stats::default()
        };
        let (accepted, emitted) =
            stats.accept(&[5, 6, 7, 1, 2, 3], &[0, 0, 9, 0, 4, 4, 4, 4], 2, 4);
        assert_eq!(accepted, vec![2, 2]);
        assert_eq!(emitted, vec![vec![5, 6, 9], vec![1, 2, 4]]);
        assert_eq!(stats.counts(), (2, 6, 4, 6));
    }
}
