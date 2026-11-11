//! Per-step timing of the engine loop, behind `RESIN_ENGINE_STEP_TIMERS`.
//!
//! Off by default: [`span`] returns `None` after one relaxed load of a cached
//! flag and reads no clock, so a disabled process runs the same instructions,
//! allocates nothing and prints nothing.
//!
//! Enabled (`RESIN_ENGINE_STEP_TIMERS=1`), each slot keeps a count, a mean and
//! a ring of the last [`RING`] samples, and one summary of medians is logged to
//! stderr every `RESIN_ENGINE_STEP_TIMERS_EVERY` steps (default 100) and once
//! more at [`flush`]. The counters are per thread, which is enough because one
//! engine steps on one thread.
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

/// Samples kept per slot for the median.
const RING: usize = 64;

macro_rules! slots {
    ($($(#[$meta:meta])* $variant:ident => $name:literal),* $(,)?) => {
        /// One timed region.
        #[derive(Clone, Copy)]
        #[repr(usize)]
        pub enum Slot {
            $($(#[$meta])* $variant),*
        }

        impl Slot {
            /// Number of slots.
            pub const COUNT: usize = [$(Slot::$variant),*].len();
            const NAMES: [&'static str; Self::COUNT] = [$($name),*];
        }
    };
}

slots! {
    /// `EngineCore::prepare_step` whole: raise checks, batch preparation, launch accounting.
    Prepare => "prepare",
    /// `runner.forward(plan)`: the model step, timed at its Python call site.
    Forward => "forward",
    /// `EngineCore::complete_step_with` whole (and its continuation twin).
    Complete => "complete",
    /// The Python adapter's runner-output extraction and logprob validation.
    Extract => "extract",
    /// `prepare_batch`: scheduler selection and the admission estimate.
    PrepCohort => "cohort",
    /// `prepare_batch`: row planning (`schedule_requests`).
    PrepSchedule => "schedule",
    /// `prepare_batch`: selection commit and KV context growth.
    PrepKv => "kv",
    /// `prepare_batch`: block/KV allocation, writer claims, batch view construction.
    PrepAcquire => "acquire",
    /// `prepare_batch`: `build_plan` (CSR construction and payload cloning).
    PrepBuild => "build",
    /// `complete_batch`: output and logprob validation, prompt score preparation.
    CompValidate => "validate",
    /// `complete_batch`: the per-row token loop, including `make_event`.
    CompEvents => "events",
    /// `complete_batch`: `prepare_scheduled_progress`.
    CompProgress => "progress",
    /// `complete_batch`: `runner.release`, KV lease release, prefix publication.
    CompRelease => "release",
    /// `complete_batch`: `retire_request_cache`.
    CompRetire => "retire",
    /// `complete_batch`: metrics commit, progress apply, requeue, lease completion.
    CompCommit => "commit",
    /// The HTTP session's `admit`: ingress parsing into engine submissions.
    Admit => "admit",
    /// The HTTP session's `process` whole: event conversion, detokenisation, emission.
    Process => "process",
    /// `process`: parser/detokeniser work and event construction, per request.
    ProcessParse => "parse",
    /// `process`: stream-interval buffering and the channel send.
    ProcessFlush => "flush",
    /// `process`: draining the output channel into the HTTP packet.
    ProcessDrain => "drain",
}

/// Whether the counters are enabled. Read once, then one relaxed load.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RESIN_ENGINE_STEP_TIMERS")
            .is_ok_and(|value| !matches!(value.trim(), "" | "0" | "false" | "off"))
    })
}

struct Series {
    count: u64,
    sum_ns: u64,
    ring: [u32; RING],
    next: usize,
}

impl Series {
    const fn new() -> Self {
        Self {
            count: 0,
            sum_ns: 0,
            ring: [0; RING],
            next: 0,
        }
    }

    fn record(&mut self, ns: u64) {
        self.count += 1;
        self.sum_ns = self.sum_ns.saturating_add(ns);
        self.ring[self.next] = ns.min(u64::from(u32::MAX)) as u32;
        self.next = (self.next + 1) % RING;
    }

    fn samples(&self) -> usize {
        self.count.min(RING as u64) as usize
    }

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

    fn mean_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ns as f64 / self.count as f64 / 1e6
        }
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

    fn group(&self, out: &mut String, label: &str, slots: &[Slot], steps: u64) {
        let _ = write!(out, "[resin-step]   {label}:");
        for &slot in slots {
            let series = &self.series[slot as usize];
            let name = Slot::NAMES[slot as usize];
            let _ = write!(
                out,
                " {name} {:.3}/{:.3}",
                series.median_ms(),
                series.mean_ms()
            );
            if series.count != steps {
                let _ = write!(out, "[{}]", series.count);
            }
        }
        out.push('\n');
    }

    /// `steps` is the number of steps the reported samples cover.
    fn summary(&self, steps: u64) -> String {
        let mut out = String::with_capacity(700);
        let _ = writeln!(
            out,
            "[resin-step] {steps} steps (cumulative decode {}, prefill {}) -- median/mean ms per call, [n] where n differs from the step count",
            self.decode_steps, self.prefill_steps
        );
        self.group(
            &mut out,
            "step",
            &[
                Slot::Prepare,
                Slot::Forward,
                Slot::Extract,
                Slot::Complete,
                Slot::Admit,
                Slot::Process,
            ],
            steps,
        );
        self.group(
            &mut out,
            "prep",
            &[
                Slot::PrepCohort,
                Slot::PrepSchedule,
                Slot::PrepKv,
                Slot::PrepAcquire,
                Slot::PrepBuild,
            ],
            steps,
        );
        self.group(
            &mut out,
            "comp",
            &[
                Slot::CompValidate,
                Slot::CompEvents,
                Slot::CompProgress,
                Slot::CompRelease,
                Slot::CompRetire,
                Slot::CompCommit,
            ],
            steps,
        );
        self.group(
            &mut out,
            "sess",
            &[Slot::ProcessParse, Slot::ProcessFlush, Slot::ProcessDrain],
            steps,
        );
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
