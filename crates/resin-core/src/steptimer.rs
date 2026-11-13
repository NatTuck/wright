//! Per-step timing of the engine loop, behind `RESIN_ENGINE_STEP_TIMERS`.
//!
//! Off by default: [`span`] returns `None` after one relaxed load of a cached
//! flag and reads no clock, so a disabled process runs the same instructions,
//! allocates nothing and prints nothing.
//!
//! Enabled (`RESIN_ENGINE_STEP_TIMERS=1`), each slot keeps a count and a ring of
//! its last [`RING`] samples, and one summary is logged to stderr every
//! `RESIN_ENGINE_STEP_TIMERS_EVERY` steps (default 100) and once more at
//! [`flush`]. The counters are per thread, which is enough because one engine
//! steps on one thread.
//!
//! # One statistic, one window
//!
//! Every slot reports **one** figure: the median of its last `min(n, RING)`
//! samples. There is deliberately no lifetime mean. A mean over the whole
//! process carries the boot's compile-thread wind-down into every number it
//! reports — the same effect that made a warmup window read 21.57 ms against a
//! 15.9 ms steady state, which is a property of the boot and not of the
//! context. A window median drops that tail once the window has rolled past it,
//! and it is the *same estimator* at one row and at eight, which is what makes
//! a ratio between two batch widths mean anything at all.
//!
//! The window is rolling: it holds the last [`RING`] samples at the moment of
//! the print, not the samples since the previous print. With the default report
//! interval of 100 steps consecutive windows are disjoint, and a window lags the
//! step count on its line by up to `every - RING` steps. `n` is printed per slot
//! where a slot's sample count differs from the step count, so a window that is
//! not yet full cannot be read as a full one.
//!
//! # Slots are grouped by what one sample counts
//!
//! `step`, `prep` and `comp` record once per engine step. `call` records once
//! per HTTP session call and `request` once per request chunk, which is *not*
//! once per step — `process` runs hundreds of times per step. Those slots are
//! printed on their own lines with their own counts and a per-step rate, and
//! they are never a step total: a step figure is `prepare` + `forward` +
//! `extract` + `complete` and nothing else.
//!
//! The window is logged at the *start* of the following step, so every span it
//! reports has already been recorded; a step's `prepare`/`forward`/`complete`
//! counts therefore agree exactly.
//!
//! These numbers are a partition of the engine loop only. They are not
//! comparable with the Python-side `RESIN_GAUDI_PROFILE_STEP` counters, which
//! time the model step between `prepare_step` and `complete_step_with`.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::sync::OnceLock;
use std::time::Instant;

/// Samples kept per slot. The reporting window is the last `min(n, RING)` of them.
const RING: usize = 64;

/// What one recorded sample counts, which decides how it may be read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unit {
    /// One sample per engine step. These slots partition a step.
    Step,
    /// One sample per HTTP session call. Never a per-step figure.
    Call,
    /// One sample per request chunk. Never a per-step figure.
    Request,
}

impl Unit {
    /// The trailing note that stops a non-step group being read as a step total.
    fn note(self) -> &'static str {
        match self {
            Unit::Step => "",
            Unit::Call => "  -- per call, not per step; never a step total",
            Unit::Request => "  -- per request chunk, not per step; never a step total",
        }
    }
}

/// The printed groups, in order. A slot's own group label must be one of these.
const GROUPS: [(&str, Unit); 5] = [
    ("step", Unit::Step),
    ("prep", Unit::Step),
    ("comp", Unit::Step),
    ("call", Unit::Call),
    ("request", Unit::Request),
];

macro_rules! slots {
    ($($(#[$meta:meta])* $variant:ident => $name:literal, $group:literal, $unit:ident);* $(;)?) => {
        /// One timed region.
        #[derive(Clone, Copy, Debug)]
        #[repr(usize)]
        pub enum Slot {
            $($(#[$meta])* $variant),*
        }

        impl Slot {
            /// Every slot, in declaration order.
            pub const ALL: &'static [Slot] = &[$(Slot::$variant),*];
            /// Number of slots.
            pub const COUNT: usize = Self::ALL.len();
            const NAMES: [&'static str; Self::COUNT] = [$($name),*];
            const GROUPS: [&'static str; Self::COUNT] = [$($group),*];
            const UNITS: [Unit; Self::COUNT] = [$(Unit::$unit),*];
        }
    };
}

// Why one statistic, and why this one.
//
// This table used to be printed as `median/mean`: a median over the last 64
// samples and a mean over every sample since the process started, on one line,
// under a header that named the aggregations and never the windows. Two numbers
// that read as a pair then get divided across batch widths, and the result is a
// ratio of two different quantities.
//
// The pair is not harmless, because the mean carries the boot's compile-thread
// wind-down for the whole life of the process. On the boot that produced the
// published b1/b8 figures, the warmup window read 21.57 ms against a 15.9 ms
// steady state for the same span at the same context — a property of the boot,
// not of the workload. Replaying that log through the old code, the *same* last-64
// window reports 44.600 ms as a median and 34.304 or 216.568 ms as a mean,
// differing only in what happened before the window; a ratio taken across those
// means reads 1.43x where the medians read 2.71x.
//
// So: one figure per slot, the median of its last `min(n, RING)` samples, and no
// running sum exists to compute a mean from. A published corroboration survived
// the mixed columns by luck — both of its quoted figures happened to be the
// median — and the tests below now pin the property that made it luck: two
// histories with the same last 64 samples must print the same figure.
slots! {
    /// `EngineCore::prepare_step` whole: raise checks, batch preparation, launch accounting.
    Prepare => "prepare", "step", Step;
    /// `runner.forward(plan)`: the model step, timed at its Python call site.
    Forward => "forward", "step", Step;
    /// `EngineCore::complete_step_with` whole (and its continuation twin).
    Complete => "complete", "step", Step;
    /// The Python adapter's runner-output extraction and logprob validation.
    Extract => "extract", "step", Step;
    /// `prepare_batch`: scheduler selection and the admission estimate.
    PrepCohort => "cohort", "prep", Step;
    /// `prepare_batch`: row planning (`schedule_requests`).
    PrepSchedule => "schedule", "prep", Step;
    /// `prepare_batch`: selection commit and KV context growth.
    PrepKv => "kv", "prep", Step;
    /// `prepare_batch`: block/KV allocation, writer claims, batch view construction.
    PrepAcquire => "acquire", "prep", Step;
    /// `prepare_batch`: `build_plan` (CSR construction and payload cloning).
    PrepBuild => "build", "prep", Step;
    /// `complete_batch`: output and logprob validation, prompt score preparation.
    CompValidate => "validate", "comp", Step;
    /// `complete_batch`: the per-row token loop, including `make_event`.
    CompEvents => "events", "comp", Step;
    /// `complete_batch`: `prepare_scheduled_progress`.
    CompProgress => "progress", "comp", Step;
    /// `complete_batch`: `runner.release`, KV lease release, prefix publication.
    CompRelease => "release", "comp", Step;
    /// `complete_batch`: `retire_request_cache`.
    CompRetire => "retire", "comp", Step;
    /// `complete_batch`: metrics commit, progress apply, requeue, lease completion.
    CompCommit => "commit", "comp", Step;
    /// The HTTP session's `admit`: ingress parsing into engine submissions, once per job.
    Admit => "admit", "request", Request;
    /// The HTTP session's `process` whole: event conversion, detokenisation, emission.
    Process => "process", "call", Call;
    /// `process`: parser/detokeniser work and event construction, once per request chunk.
    ProcessParse => "parse", "request", Request;
    /// `process`: stream-interval buffering and the channel send, once per request chunk.
    ProcessFlush => "flush", "request", Request;
    /// `process`: draining the output channel into the HTTP packet.
    ProcessDrain => "drain", "call", Call;
}

/// Whether the counters are enabled. Read once, then one relaxed load.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RESIN_ENGINE_STEP_TIMERS")
            .is_ok_and(|value| !matches!(value.trim(), "" | "0" | "false" | "off"))
    })
}

/// One slot's samples: a lifetime count and a ring of the most recent [`RING`].
///
/// There is no running sum, so no lifetime mean can be reported by accident.
struct Series {
    count: u64,
    ring: [u32; RING],
    next: usize,
}

impl Series {
    const fn new() -> Self {
        Self {
            count: 0,
            ring: [0; RING],
            next: 0,
        }
    }

    fn record(&mut self, ns: u64) {
        self.count += 1;
        // A sample longer than 4.29 s saturates; no engine span is that long.
        self.ring[self.next] = ns.min(u64::from(u32::MAX)) as u32;
        self.next = (self.next + 1) % RING;
    }

    fn samples(&self) -> usize {
        self.count.min(RING as u64) as usize
    }

    /// The median of the last `min(count, RING)` samples, in milliseconds.
    fn median_ms(&self) -> f64 {
        let samples = self.samples();
        if samples == 0 {
            return 0.0;
        }
        let mut sorted = self.ring;
        sorted[..samples].sort_unstable();
        let middle = if samples % 2 == 1 {
            f64::from(sorted[samples / 2])
        } else {
            (f64::from(sorted[samples / 2 - 1]) + f64::from(sorted[samples / 2])) / 2.0
        };
        middle / 1e6
    }
}

/// A non-step sample count as a per-step rate, for the groups that are not per step.
fn per_step(count: u64, steps: u64) -> String {
    if steps == 0 {
        return "0".to_owned();
    }
    let rate = count as f64 / steps as f64;
    if rate >= 1.0 {
        format!("{rate:.1}")
    } else {
        format!("{rate:.3}")
    }
}

struct Timers {
    series: [Series; Slot::COUNT],
    steps: u64,
    decode_steps: u64,
    prefill_steps: u64,
    every: u64,
}

impl Timers {
    fn new() -> Self {
        let every = std::env::var("RESIN_ENGINE_STEP_TIMERS_EVERY")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|&count| count > 0)
            .unwrap_or(100);
        Self {
            series: std::array::from_fn(|_| Series::new()),
            steps: 0,
            decode_steps: 0,
            prefill_steps: 0,
            every,
        }
    }

    /// One printed line for one group: every slot of that group, one figure each.
    fn group(&self, out: &mut String, label: &str, unit: Unit, steps: u64) {
        let _ = write!(out, "[resin-step]   {label}:");
        for index in 0..Slot::COUNT {
            if Slot::GROUPS[index] != label || Slot::UNITS[index] != unit {
                continue;
            }
            let series = &self.series[index];
            let _ = write!(out, " {} {:.3}", Slot::NAMES[index], series.median_ms());
            if unit == Unit::Step {
                if series.count != steps {
                    let _ = write!(out, "[n={}]", series.count);
                }
            } else {
                let _ = write!(
                    out,
                    "[n={}, {}/step]",
                    series.count,
                    per_step(series.count, steps)
                );
            }
        }
        let _ = writeln!(out, "{}", unit.note());
    }

    /// `steps` is the number of steps the reported samples cover.
    fn summary(&self, steps: u64) -> String {
        let mut out = String::with_capacity(900);
        let _ = writeln!(
            out,
            "[resin-step] {steps} steps (cumulative decode {}, prefill {}) -- window median ms over each slot's last min(n,{RING}) samples; n printed where a slot's sample count differs from the step count",
            self.decode_steps, self.prefill_steps
        );
        for (label, unit) in GROUPS {
            self.group(&mut out, label, unit, steps);
        }
        out
    }
}

thread_local! {
    static TIMERS: RefCell<Timers> = RefCell::new(Timers::new());
}

/// A timed region; records on drop, so early returns are timed too.
pub struct Span {
    slot: Slot,
    started: Instant,
}

impl Drop for Span {
    fn drop(&mut self) {
        let ns = self.started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let slot = self.slot;
        let _ = TIMERS.try_with(|timers| {
            if let Ok(mut timers) = timers.try_borrow_mut() {
                timers.series[slot as usize].record(ns);
            }
        });
    }
}

/// Start a region, or return `None` when the counters are off.
#[must_use]
pub fn span(slot: Slot) -> Option<Span> {
    if !enabled() {
        return None;
    }
    Some(Span {
        slot,
        started: Instant::now(),
    })
}

/// Count one dispatched step, and log the fully recorded window when it closes.
///
/// Called before the step's own spans start, so the window it prints holds
/// every sample of the steps before this one.
pub fn begin_step() {
    if !enabled() {
        return;
    }
    let line = TIMERS
        .try_with(|timers| {
            let mut timers = timers.try_borrow_mut().ok()?;
            let finished = timers.steps;
            let line =
                (finished > 0 && finished % timers.every == 0).then(|| timers.summary(finished));
            timers.steps = finished + 1;
            line
        })
        .ok()
        .flatten();
    if let Some(line) = line {
        eprint!("{line}");
    }
}

/// Count the step just prepared as decode or prefill.
pub fn note_phase(decode: bool) {
    if !enabled() {
        return;
    }
    let _ = TIMERS.try_with(|timers| {
        if let Ok(mut timers) = timers.try_borrow_mut() {
            if decode {
                timers.decode_steps += 1;
            } else {
                timers.prefill_steps += 1;
            }
        }
    });
}

/// Log the summary of everything recorded so far. Called when an engine closes.
pub fn flush() {
    if !enabled() {
        return;
    }
    let line = TIMERS
        .try_with(|timers| {
            let timers = timers.try_borrow().ok()?;
            Some(timers.summary(timers.steps))
        })
        .ok()
        .flatten();
    if let Some(line) = line {
        eprint!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A timer set whose `forward` slot holds `history`, one sample per step.
    fn with_forward(history: &[u64]) -> Timers {
        let mut timers = Timers::new();
        for &ns in history {
            timers.series[Slot::Forward as usize].record(ns);
        }
        timers.steps = history.len() as u64;
        timers
    }

    fn group_line<'a>(summary: &'a str, label: &str) -> &'a str {
        let needle = format!(" {label}:");
        summary
            .lines()
            .find(|line| line.contains(&needle))
            .unwrap_or_else(|| panic!("no `{label}:` group in:\n{summary}"))
    }

    /// The single figure printed for `slot` on the `label` group's line, without
    /// any `[n=…]` suffix a non-step group appends to it.
    fn figure(summary: &str, label: &str, slot: &str) -> String {
        let line = group_line(summary, label);
        let mut words = line.split_whitespace();
        while let Some(word) = words.next() {
            if word == slot {
                let value = words.next().expect("a value after the slot name");
                return value.split('[').next().unwrap_or(value).to_owned();
            }
        }
        panic!("no `{slot}` on the `{label}:` line: {line}");
    }

    /// The regression this file exists for: a 1-row figure and an 8-row figure may
    /// only be divided if they are the same statistic.
    ///
    /// Both arms hold the *same* last 64 samples (a b8 steady state at 44.6 ms), so
    /// the reported figure must be identical. One arm reaches the window after a
    /// clean boot, the other after a 60-step compile-thread wind-down. A lifetime
    /// mean would differ between the arms by more than a third, and any reader
    /// comparing that column across two batch widths would be dividing two
    /// different quantities.
    #[test]
    fn reported_figure_does_not_depend_on_history_before_the_window() {
        let steady = [44_600_000u64; RING];

        let mut clean_boot = vec![16_000_000u64; 36];
        clean_boot.extend_from_slice(&steady);

        let mut after_wind_down = vec![400_000_000u64; 60];
        after_wind_down.extend_from_slice(&steady);

        let clean = with_forward(&clean_boot);
        let warm = with_forward(&after_wind_down);
        let clean_figure = figure(&clean.summary(clean.steps), "step", "forward");
        let warm_figure = figure(&warm.summary(warm.steps), "step", "forward");
        assert_eq!(
            clean_figure, warm_figure,
            "forward must report one statistic over one window; got {clean_figure} against \
             {warm_figure} for the same last {RING} samples"
        );
    }

    /// The figure is the window median, and it is the only number on the line: a
    /// second column would be the lifetime mean, which the boot contaminates.
    #[test]
    fn the_window_median_is_the_only_figure_reported() {
        let mut history = vec![400_000_000u64; 40];
        history.extend(std::iter::repeat_n(16_000_000u64, 60));
        let timers = with_forward(&history);
        let summary = timers.summary(timers.steps);
        let step_line = group_line(&summary, "step");

        assert_eq!(
            figure(&summary, "step", "forward"),
            "16.000",
            "expected the median of the last {RING} samples, not a lifetime mean: {step_line}"
        );
        assert!(
            !step_line.contains('/'),
            "one statistic per slot, no `median/mean` pair: {step_line}"
        );
    }

    /// A per-call span is not a step: it must not sit on the `step:` line, and it
    /// must carry its own count and per-step rate.
    #[test]
    fn per_call_slots_are_not_a_step_total() {
        let mut timers = with_forward(&[16_000_000u64; 8]);
        for _ in 0..551 {
            timers.series[Slot::Process as usize].record(1_000_000);
        }
        let summary = timers.summary(timers.steps);

        let step_line = group_line(&summary, "step");
        assert!(
            !step_line.contains("process"),
            "`process` is per call, not per step, and must not sit on the step line: {step_line}"
        );
        let call_line = group_line(&summary, "call");
        assert!(
            call_line.contains("process 1.000[n=551, 68.9/step]"),
            "a per-call slot needs its count and its rate: {call_line}"
        );
        assert!(
            call_line.contains("not per step"),
            "the call line must say it is not a step total: {call_line}"
        );
    }

    /// Every slot is printed exactly once, in the group its unit declares, as the
    /// median of its own window — so no slot can be printed under another's name
    /// or aggregated some other way.
    #[test]
    fn every_slot_is_printed_once_in_its_group_as_the_window_median() {
        let mut timers = Timers::new();
        for index in 0..Slot::COUNT {
            for _ in 0..RING {
                timers.series[index].record(1_000_000 * (index as u64 + 1));
            }
        }
        timers.steps = RING as u64;
        let summary = timers.summary(timers.steps);

        for (index, slot) in Slot::ALL.iter().enumerate() {
            let label = Slot::GROUPS[index];
            let expected = format!("{:.3}", timers.series[index].median_ms());
            assert_eq!(
                figure(&summary, label, Slot::NAMES[index]),
                expected,
                "slot {slot:?} is not reported as its own window median"
            );
        }
        let printed = summary
            .lines()
            .filter(|line| line.starts_with("[resin-step]   "))
            .flat_map(|line| line.split_whitespace().skip(1))
            .filter(|word| Slot::NAMES.contains(word))
            .count();
        assert_eq!(printed, Slot::COUNT, "a slot is printed twice or not at all:\n{summary}");
    }

    /// The two tables a slot is printed by cannot disagree, or a slot would be
    /// silently dropped from the summary.
    #[test]
    fn every_slot_belongs_to_a_declared_group() {
        for (index, slot) in Slot::ALL.iter().enumerate() {
            let label = Slot::GROUPS[index];
            let unit = Slot::UNITS[index];
            assert!(
                GROUPS.contains(&(label, unit)),
                "slot {slot:?} is labelled `{label}` with a unit the print order does not carry"
            );
        }
    }
}
