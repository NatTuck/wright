//! Whether a decode step verifies drafts, from each row's recent acceptance and measured step times: the port of
//! `resin_models_gaudi/draft_choice.py` (`DraftChoice`, `RowAcceptance`, `StepTimes`), decision for decision.
//!
//! A step verifies drafts when its expected tokens per ms beat a plain step's: each row is expected to commit one
//! token and its average accepted drafts, over the verify step's time, against one token a row over the plain
//! step's time. A row with no verified step yet counts at the average of all rows. Once a bucket decodes plainly,
//! its rows' averages do not change, so after a run of plain steps a few steps verify drafts again (a probe). A probe
//! costs the tokens that plain steps would commit in its time, less the tokens it commits: its first step verifies
//! zeros (a plain step made no drafts), and the others accept the rows' averages. The run before a probe is long
//! enough that probes cost at most PROBE_SHARE of the run's tokens. So a bucket whose rows are near the break-even
//! point probes often (a probe costs little there, and the text can change to text that the drafter predicts
//! well), and a bucket far below it probes seldom. The probes do not back off. A bucket probes only where drafts
//! could pay: where a verify step that accepts every draft beats a plain step.
//!
//! Tensor-parallel ranks must decide the same. With a native rank group the driver decides and sends the
//! decision with the step; otherwise each rank decides with the first rank's times, which `share` gives every rank
//! at the same steps (SYNC_FIRST, then every SYNC_STEPS decode steps).

use std::collections::{BTreeMap, HashMap, VecDeque};

/// "plain" or "verify"
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Plain,
    Verify,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Plain => "plain",
            Kind::Verify => "verify",
        }
    }

    pub fn of(name: &str) -> Option<Self> {
        match name {
            "plain" => Some(Kind::Plain),
            "verify" => Some(Kind::Verify),
            _ => None,
        }
    }
}

/// Each row's drafts accepted a verify step (`RowAcceptance`): an average that starts at the average of all rows
/// and weights the latest step by BETA; the average of all rows weights each row's step by PRIOR_BETA.
#[derive(Clone, Debug)]
pub struct RowAcceptance {
    pub prior: f64,
    rows: HashMap<u64, f64>,
    /// the rows in insertion order (a record moves its row to the end), for the LIMIT
    order: VecDeque<u64>,
}

impl RowAcceptance {
    pub const BETA: f64 = 0.2;
    pub const PRIOR_BETA: f64 = 0.02;
    pub const LIMIT: usize = 4096;

    pub fn new(prior: f64) -> Self {
        Self {
            prior,
            rows: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn expected(&self, sequence: u64) -> f64 {
        self.rows.get(&sequence).copied().unwrap_or(self.prior)
    }

    pub fn record(&mut self, sequence: u64, accepted: i64) {
        let old = match self.rows.remove(&sequence) {
            Some(old) => {
                self.order.retain(|&s| s != sequence);
                old
            }
            None => self.prior,
        };
        let accepted = accepted as f64;
        self.rows
            .insert(sequence, old + Self::BETA * (accepted - old));
        self.order.push_back(sequence);
        self.prior += Self::PRIOR_BETA * (accepted - self.prior);
        if self.rows.len() > Self::LIMIT
            && let Some(oldest) = self.order.pop_front()
        {
            self.rows.remove(&oldest);
        }
    }

    pub fn forget(&mut self, sequences: &[u64]) {
        for sequence in sequences {
            if self.rows.remove(sequence).is_some() {
                self.order.retain(|s| s != sequence);
            }
        }
    }
}

/// Measured decode step times (ms) by kind and bucket (`StepTimes`): averages that take the first steps equally
/// and then weight the latest by BETA; a step SPIKE times its average is not counted unless SPIKE_RUN come in a
/// row.
#[derive(Clone, Debug, Default)]
pub struct StepTimes {
    /// [steps, average ms, spikes in a row]
    stats: BTreeMap<(Kind, usize), (u64, f64, u64)>,
}

impl StepTimes {
    pub const BETA: f64 = 0.05;
    pub const SPIKE: f64 = 3.0;
    pub const SPIKE_RUN: u64 = 3;

    pub fn record(&mut self, kind: Kind, bucket: usize, ms: f64) {
        // not a positive time (NaN too)
        if ms.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return;
        }
        let stats = self.stats.entry((kind, bucket)).or_insert((0, 0.0, 0));
        if stats.0 > 0 && ms > Self::SPIKE * stats.1 {
            stats.2 += 1;
            if stats.2 < Self::SPIKE_RUN {
                return;
            }
        }
        stats.2 = 0;
        stats.0 += 1;
        stats.1 += Self::BETA.max(1.0 / stats.0 as f64) * (ms - stats.1);
    }

    pub fn table(&self) -> BTreeMap<(Kind, usize), f64> {
        self.stats.iter().map(|(k, s)| (*k, s.1)).collect()
    }
}

/// The fitted step times (ms) by rows a policy starts from (`step_costs`): plain steps `plain_0 + plain_row x
/// rows`, verify steps `scale x launches x (launch_0 + launch_row x min(rows, group))` in launches of `group` rows.
#[derive(Clone, Copy, Debug)]
pub struct StepCosts {
    pub plain_0: f64,
    pub plain_row: f64,
    pub launch_0: f64,
    pub launch_row: f64,
    pub scale: f64,
    pub group: usize,
}

impl StepCosts {
    pub fn plain(&self, rows: usize) -> f64 {
        self.plain_0 + self.plain_row * rows as f64
    }

    pub fn verify(&self, rows: usize) -> f64 {
        let launches = rows.div_ceil(self.group);
        self.scale
            * launches as f64
            * (self.launch_0 + self.launch_row * rows.min(self.group) as f64)
    }

    fn of(&self, kind: Kind, rows: usize) -> f64 {
        match kind {
            Kind::Plain => self.plain(rows),
            Kind::Verify => self.verify(rows),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Always,
    Never,
}

impl Mode {
    pub fn of(name: &str) -> Result<Self, String> {
        match name {
            "auto" => Ok(Mode::Auto),
            "always" => Ok(Mode::Always),
            "never" => Ok(Mode::Never),
            _ => Err(format!("draft policy {name:?}: one of auto, always, never")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Auto => "auto",
            Mode::Always => "always",
            Mode::Never => "never",
        }
    }
}

/// Whether each decode step verifies drafts (`DraftChoice`). `speculate` must run once a decode step, on every
/// deciding rank in the same order, after the last step's `verified`.
#[derive(Clone, Debug)]
pub struct DraftChoice {
    pub k: usize,
    pub costs: StepCosts,
    pub max_rows: usize,
    pub mode: Mode,
    buckets: Vec<usize>,
    pub acceptance: RowAcceptance,
    pub times: StepTimes,
    /// with a share: the times every rank decides with (None: decide with this rank's own measurements)
    pub shared: Option<BTreeMap<(Kind, usize), f64>>,
    steps: u64,
    drafting: HashMap<usize, bool>,
    /// plain decisions in a row, by bucket
    plain_run: HashMap<usize, u64>,
    /// verify steps left in a probe
    probe: u64,
}

impl DraftChoice {
    pub const MARGIN: f64 = 1.1;
    /// the most plain steps in a bucket before a probe
    pub const PROBE_MAX: u64 = 512;
    /// verify steps of a probe; the first verifies zeros (a plain step made no drafts)
    pub const PROBE_STEPS: u64 = 4;
    /// the most of a plain run's tokens that its probe costs. The probes do not back off: text changes, and a
    /// back-off would keep rows near the break-even point on plain steps after their text changes to text that
    /// the drafter predicts well.
    pub const PROBE_SHARE: f64 = 0.05;
    pub const SYNC_FIRST: [u64; 3] = [32, 64, 128];
    pub const SYNC_STEPS: u64 = 256;

    /// `shared`: the ranks share the first rank's times (`share` in the Python policy); the caller gives them at
    /// the steps `sync_due` names.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: usize,
        costs: StepCosts,
        max_rows: usize,
        prior: f64,
        buckets: &[usize],
        shared: bool,
        mode: Mode,
    ) -> Self {
        let mut buckets = buckets.to_vec();
        buckets.sort_unstable();
        Self {
            k,
            costs,
            max_rows,
            mode,
            buckets,
            acceptance: RowAcceptance::new(prior),
            times: StepTimes::default(),
            shared: shared.then(BTreeMap::new),
            steps: 0,
            drafting: HashMap::new(),
            plain_run: HashMap::new(),
            probe: 0,
        }
    }

    pub fn bucket(&self, rows: usize) -> usize {
        self.buckets
            .iter()
            .copied()
            .find(|&b| b >= rows)
            .unwrap_or(*self.buckets.last().expect("buckets"))
    }

    /// A step's time: the shared (or own) measurement of its bucket, else the fit scaled by the nearest measured
    /// bucket of the kind (the fit when the kind has none).
    pub fn ms(&self, kind: Kind, rows: usize) -> f64 {
        let own;
        let table = match &self.shared {
            Some(shared) => shared,
            None => {
                own = self.times.table();
                &own
            }
        };
        let b = self.bucket(rows);
        if let Some(&measured) = table.get(&(kind, b)) {
            return measured;
        }
        let fit = self.costs.of(kind, rows);
        let bits = |x: usize| (usize::BITS - x.leading_zeros()) as i64;
        let near = table
            .keys()
            .filter(|key| key.0 == kind)
            .map(|key| key.1)
            .min_by_key(|&n| ((bits(n) - bits(b)).abs(), n));
        match near {
            None => fit,
            Some(n) => fit * table[&(kind, n)] / self.costs.of(kind, n),
        }
    }

    /// The expected tokens per ms of a verify step and of a plain step of these rows.
    pub fn rates(&self, sequences: &[u64]) -> (f64, f64) {
        let rows = sequences.len();
        let tokens: f64 = sequences
            .iter()
            .map(|&s| 1.0 + self.acceptance.expected(s).min(self.k as f64))
            .sum();
        (
            tokens / self.ms(Kind::Verify, rows),
            rows as f64 / self.ms(Kind::Plain, rows),
        )
    }

    /// Whether this decode step shares the first rank's times first (call before `speculate`, auto mode only).
    pub fn sync_due(&self) -> bool {
        let next = self.steps + 1;
        self.mode == Mode::Auto
            && self.k > 0
            && self.shared.is_some()
            && (Self::SYNC_FIRST.contains(&next) || next.is_multiple_of(Self::SYNC_STEPS))
    }

    /// Install the first rank's times (at a step `sync_due` named).
    pub fn set_shared(&mut self, table: BTreeMap<(Kind, usize), f64>) {
        self.shared = Some(table);
    }

    /// Decide with this rank's own measured times from now on, and name no sync steps: a native rank group's
    /// driver decides for every rank, and the other ranks take its decision with the step.
    pub fn decide_alone(&mut self) {
        self.shared = None;
    }

    pub fn speculate(&mut self, sequences: &[u64]) -> bool {
        let rows = sequences.len();
        if self.k == 0 || self.mode != Mode::Auto {
            return self.k > 0 && self.mode == Mode::Always && 0 < rows && rows <= self.max_rows;
        }
        self.steps += 1;
        if rows == 0 || rows > self.max_rows {
            return false;
        }
        if self.probe > 0 {
            self.probe -= 1;
            return true;
        }
        let b = self.bucket(rows);
        let (spec, plain) = self.rates(sequences);
        let drafting = self.drafting.get(&b).copied().unwrap_or(true);
        let decide = if drafting {
            spec * Self::MARGIN >= plain
        } else {
            spec > plain
        };
        self.drafting.insert(b, decide);
        if decide {
            self.plain_run.insert(b, 0);
            return true;
        }
        let run = self.plain_run.get(&b).copied().unwrap_or(0) + 1;
        if run as f64 >= self.probe_after(sequences) {
            self.plain_run.insert(b, 0);
            self.probe = Self::PROBE_STEPS - 1;
            return true;
        }
        self.plain_run.insert(b, run);
        false
    }

    /// The plain steps of these rows before a probe (infinity: drafts cannot pay, even when the rows accept every
    /// draft). A verify step takes `ratio` plain steps' time: a probe's first step costs ratio - 1 tokens a row (it
    /// verifies zeros), and each other step ratio - 1 - the rows' average accepted drafts, at least 0.
    pub fn probe_after(&self, sequences: &[u64]) -> f64 {
        let rows = sequences.len();
        let ratio = self.ms(Kind::Verify, rows) / self.ms(Kind::Plain, rows);
        if (1 + self.k) as f64 <= ratio {
            return f64::INFINITY;
        }
        let accepted = sequences
            .iter()
            .map(|&s| self.acceptance.expected(s).min(self.k as f64))
            .sum::<f64>()
            / rows as f64;
        let cost = ratio - 1.0 + (Self::PROBE_STEPS - 1) as f64 * (ratio - 1.0 - accepted).max(0.0);
        (cost / Self::PROBE_SHARE)
            .ceil()
            .min(Self::PROBE_MAX as f64)
    }

    /// A decode step of `rows` rows took `ms` (on this rank).
    pub fn measured(&mut self, kind: Kind, rows: usize, ms: f64) {
        let b = self.bucket(rows);
        self.times.record(kind, b, ms);
    }

    /// A finished verify step: each row's accepted drafts; `own`: the rows that verified drafts of their own (None:
    /// all).
    pub fn verified(&mut self, sequences: &[u64], accepted: &[i64], own: Option<&[bool]>) {
        for (r, &sequence) in sequences.iter().enumerate() {
            if own.is_none_or(|own| own[r]) {
                self.acceptance.record(sequence, accepted[r]);
            }
        }
    }

    pub fn forget(&mut self, sequences: &[u64]) {
        self.acceptance.forget(sequences);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs() -> StepCosts {
        // step_costs(4, 3, False, 4): MTP of 3 drafts on 4 cards
        StepCosts {
            plain_0: 8.8,
            plain_row: 1.53,
            launch_0: 12.7,
            launch_row: 6.4,
            scale: 1.0,
            group: 4,
        }
    }

    #[test]
    fn modes_decide_without_measurements() {
        let buckets = [1, 2, 4, 8, 16, 32, 64];
        let mut always = DraftChoice::new(3, costs(), 4, 1.2, &buckets, false, Mode::Always);
        assert!(always.speculate(&[1, 2]));
        assert!(!always.speculate(&[1, 2, 3, 4, 5]));
        let mut never = DraftChoice::new(3, costs(), 4, 1.2, &buckets, false, Mode::Never);
        assert!(!never.speculate(&[1]));
    }

    #[test]
    fn a_row_that_accepts_nothing_turns_drafts_off_and_probes_later() {
        let buckets = [1, 2, 4, 8, 16, 32, 64];
        let mut choice = DraftChoice::new(3, costs(), 4, 1.2, &buckets, false, Mode::Auto);
        // prior 1.2: (2.2 / 31.9) * 1.1 >= 1 / 10.33
        assert!(choice.speculate(&[7]));
        for _ in 0..40 {
            choice.verified(&[7], &[0], None);
        }
        assert!(choice.acceptance.expected(7) < 0.01);
        // a verify step takes 19.1 / 10.33 plain steps' time: a probe costs 0.85 tokens for its first step and
        // 0.85 for each other step (no drafts accepted), so 3.4 / PROBE_SHARE plain steps come before it
        let interval = choice.probe_after(&[7]);
        assert_eq!(interval, 68.0);
        let decisions: Vec<bool> = (0..300).map(|_| choice.speculate(&[7])).collect();
        assert!(!decisions[0]);
        // probes of PROBE_STEPS verify steps, each after the same run of plain steps (no back-off). The decision
        // that reaches the interval is the probe's first step, so interval - 1 plain steps come before it
        let steps = DraftChoice::PROBE_STEPS as usize;
        let starts: Vec<usize> = (0..decisions.len())
            .filter(|&i| decisions[i] && (i == 0 || !decisions[i - 1]))
            .collect();
        assert_eq!(starts[0], interval as usize - 1);
        assert!(starts.len() >= 3);
        for pair in starts.windows(2) {
            assert!(decisions[pair[0]..pair[0] + steps].iter().all(|&d| d));
            assert_eq!(pair[1] - pair[0] - steps, interval as usize - 1);
        }
    }

    #[test]
    fn probes_come_often_near_the_break_even_point() {
        let buckets = [1, 2, 4, 8, 16, 32, 64];
        let mut choice = DraftChoice::new(3, costs(), 4, 1.2, &buckets, false, Mode::Auto);
        let ratio = costs().verify(1) / costs().plain(1);
        for _ in 0..40 {
            choice.verified(&[7], &[0], None);
            choice.verified(&[8], &[1], None);
        }
        let far = choice.probe_after(&[7]);
        let near = choice.probe_after(&[8]);
        assert!(near < far);
        let accepted = choice.acceptance.expected(8);
        let cost = ratio - 1.0 + 3.0 * (ratio - 1.0 - accepted).max(0.0);
        assert_eq!(near, (cost / DraftChoice::PROBE_SHARE).ceil());
        // 16 rows: a verify step of 4 launches loses even with every draft accepted, so no probe
        let rows: Vec<u64> = (10..26).collect();
        assert_eq!(choice.probe_after(&rows), f64::INFINITY);
    }

    #[test]
    fn step_times_skip_spikes_until_they_last() {
        let mut times = StepTimes::default();
        times.record(Kind::Plain, 1, 10.0);
        times.record(Kind::Plain, 1, 50.0);
        assert_eq!(times.table()[&(Kind::Plain, 1)], 10.0);
        times.record(Kind::Plain, 1, 50.0);
        times.record(Kind::Plain, 1, 50.0);
        assert!(times.table()[&(Kind::Plain, 1)] > 10.0);
    }

    #[test]
    fn the_nearest_measured_bucket_scales_the_fit() {
        let buckets = [1, 2, 4, 8, 16, 32, 64];
        let mut choice = DraftChoice::new(3, costs(), 4, 1.2, &buckets, false, Mode::Auto);
        choice.measured(Kind::Plain, 1, 2.0 * costs().plain(1));
        assert!((choice.ms(Kind::Plain, 8) - 2.0 * costs().plain(8)).abs() < 1e-9);
        assert_eq!(choice.ms(Kind::Verify, 2), costs().verify(2));
    }

    #[test]
    fn a_policy_that_decides_alone_uses_its_own_times() {
        // a native rank group's driver: built to share the first rank's times, then deciding for every rank
        let buckets = [1, 2, 4, 8, 16, 32, 64];
        let mut choice = DraftChoice::new(3, costs(), 4, 1.2, &buckets, true, Mode::Auto);
        for _ in 0..200 {
            choice.measured(Kind::Verify, 1, 60.0); // three times the fit
            choice.measured(Kind::Plain, 1, 10.0);
        }
        // sharing, before the first shared table: the fits
        assert_eq!(choice.ms(Kind::Verify, 1), costs().verify(1));
        choice.decide_alone();
        assert_eq!(choice.ms(Kind::Verify, 1), 60.0);
        assert_eq!(choice.ms(Kind::Plain, 1), 10.0);
        // prior 1.2: 2.2 tokens in 60 ms lose to 1 in 10 ms, and a verify step of six plain steps' time cannot pay
        // even when every draft is accepted, so no probe; no step shares times
        for _ in 0..300 {
            assert!(!choice.sync_due());
            assert!(!choice.speculate(&[7]));
        }
    }
}
