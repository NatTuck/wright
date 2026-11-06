//! Time sharing between prefill and decode launches.
//!
//! A launch never mixes phases, so a long prompt stops every decoding stream
//! while it prefills. With a [`DecodeShare`], the Engine interleaves the two
//! phases while both have work. After a prefill launch that took `P` seconds,
//! decode receives `P * s / (1 - s)` seconds before the next prefill launch,
//! where `s` is the target decode share. Without contention, the phase that
//! has work runs and nothing is throttled.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::Phase;

/// Bounds on the share of contended execution time that decode receives.
///
/// The target moves linearly from `min` with no decoding rows to `max` when
/// every running slot decodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeShare {
    min: f64,
    max: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeShareError;

impl fmt::Display for DecodeShareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("decode_share must satisfy 0 <= min <= max < 1")
    }
}

impl std::error::Error for DecodeShareError {}

impl DecodeShare {
    pub fn new(min: f64, max: f64) -> Result<Self, DecodeShareError> {
        if min.is_finite() && max.is_finite() && 0.0 <= min && min <= max && max < 1.0 {
            Ok(Self { min, max })
        } else {
            Err(DecodeShareError)
        }
    }

    pub fn min(self) -> f64 {
        self.min
    }

    pub fn max(self) -> f64 {
        self.max
    }

    /// The target decode share with `decoding` of `slots` running rows in decode.
    pub fn target(self, decoding: usize, slots: usize) -> f64 {
        let fill = if slots == 0 {
            1.0
        } else {
            (decoding as f64 / slots as f64).min(1.0)
        };
        self.min + (self.max - self.min) * fill
    }
}

/// A monotonic time source. Tests substitute a manual clock.
pub type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Cumulative host-observed execution time of committed launches.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BusySeconds {
    pub prefill: f64,
    pub decode: f64,
}

#[derive(Clone, Copy, Debug)]
struct Launch {
    phase: Phase,
    at: Instant,
    /// The target share when a prefill launch ran against decoding rows.
    share: Option<f64>,
}

pub(crate) struct TimeShare {
    share: Option<DecodeShare>,
    clock: Clock,
    /// Launches whose results the Engine has not collected, oldest first.
    launches: VecDeque<Launch>,
    last_return: Option<Instant>,
    /// Decode seconds still owed for earlier prefill launches.
    debt: f64,
    contended: bool,
    busy: BusySeconds,
}

impl TimeShare {
    pub(crate) fn new(share: Option<DecodeShare>) -> Self {
        Self {
            // A zero maximum is the prefill-first order.
            share: share.filter(|share| share.max > 0.0),
            clock: Arc::new(Instant::now),
            launches: VecDeque::with_capacity(2),
            last_return: None,
            debt: 0.0,
            contended: false,
            busy: BusySeconds::default(),
        }
    }

    pub(crate) fn set_clock(&mut self, clock: Clock) {
        self.clock = clock;
    }

    pub(crate) fn share(&self) -> Option<DecodeShare> {
        self.share
    }

    pub(crate) fn busy(&self) -> BusySeconds {
        self.busy
    }

    /// Whether decode still has time owed from earlier prefill launches.
    pub(crate) fn decode_owed(&self) -> bool {
        self.debt > 0.0
    }

    /// Record whether both phases have work. The debt belongs to one period of
    /// contention, so it resets when that period ends.
    pub(crate) fn observe(&mut self, contended: bool) {
        if !contended {
            self.debt = 0.0;
        }
        self.contended = contended;
    }

    /// Stamp a plan handed to the runner. `share` is the target for a prefill
    /// launch that runs while rows decode.
    pub(crate) fn launched(&mut self, phase: Phase, share: Option<f64>) {
        let at = (self.clock)();
        self.launches.push_back(Launch { phase, at, share });
    }

    /// Stamp the collection of the oldest launch. A step that overlaps its
    /// predecessor is charged only from the predecessor's return.
    pub(crate) fn returned(&mut self, executed: bool) {
        let now = (self.clock)();
        let Some(launch) = self.launches.pop_front() else {
            return;
        };
        let start = self
            .last_return
            .map_or(launch.at, |previous| previous.max(launch.at));
        self.last_return = Some(now);
        if !executed {
            return;
        }
        let busy = now.saturating_duration_since(start).as_secs_f64();
        match launch.phase {
            Phase::Prefill => {
                self.busy.prefill += busy;
                if let Some(share) = launch.share {
                    self.debt += busy * share / (1.0 - share);
                }
            }
            Phase::Decode => {
                self.busy.decode += busy;
                if self.contended {
                    self.debt -= busy;
                }
            }
        }
    }

    /// Forget launches that will never be collected.
    pub(crate) fn clear_launches(&mut self) {
        self.launches.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use super::{DecodeShare, TimeShare};
    use crate::Phase;

    struct ManualClock(Arc<Mutex<Instant>>);

    impl ManualClock {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Instant::now())))
        }

        fn advance(&self, seconds: f64) {
            *self.0.lock().unwrap() += Duration::from_secs_f64(seconds);
        }

        fn install(&self, share: &mut TimeShare) {
            let time = Arc::clone(&self.0);
            share.set_clock(Arc::new(move || *time.lock().unwrap()));
        }
    }

    fn run(share: &mut TimeShare, clock: &ManualClock, phase: Phase, seconds: f64, target: f64) {
        share.launched(phase, (phase == Phase::Prefill).then_some(target));
        clock.advance(seconds);
        share.returned(true);
    }

    #[test]
    fn decode_share_bounds_are_validated() {
        assert!(DecodeShare::new(0.0, 0.0).is_ok());
        assert!(DecodeShare::new(0.5, 0.9).is_ok());
        for (min, max) in [(0.6, 0.5), (-0.1, 0.5), (0.5, 1.0), (f64::NAN, 0.5)] {
            assert!(DecodeShare::new(min, max).is_err(), "{min} {max}");
        }
    }

    #[test]
    fn target_scales_with_decoding_rows() {
        let share = DecodeShare::new(0.5, 0.9).unwrap();
        assert_eq!(share.target(0, 16), 0.5);
        assert!((share.target(8, 16) - 0.7).abs() < 1e-12);
        assert!((share.target(16, 16) - 0.9).abs() < 1e-12);
        assert!((share.target(32, 16) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn zero_maximum_is_prefill_first() {
        assert!(
            TimeShare::new(DecodeShare::new(0.0, 0.0).ok())
                .share()
                .is_none()
        );
    }

    #[test]
    fn prefill_launch_buys_decode_time() {
        let clock = ManualClock::new();
        let mut share = TimeShare::new(DecodeShare::new(0.8, 0.8).ok());
        clock.install(&mut share);
        share.observe(true);
        assert!(!share.decode_owed(), "prefill goes first");
        run(&mut share, &clock, Phase::Prefill, 1.0, 0.8);
        let mut decode = 0.0_f64;
        while share.decode_owed() {
            run(&mut share, &clock, Phase::Decode, 0.05, 0.8);
            decode += 0.05;
        }
        assert!((decode - 4.0).abs() < 0.05 + 1e-9, "{decode}");
    }

    #[test]
    fn overshoot_carries_and_the_long_run_share_matches() {
        for target in [0.5, 0.9] {
            let clock = ManualClock::new();
            let mut share = TimeShare::new(DecodeShare::new(target, target).ok());
            clock.install(&mut share);
            share.observe(true);
            for _ in 0..200 {
                if share.decode_owed() {
                    run(&mut share, &clock, Phase::Decode, 0.07, target);
                } else {
                    run(&mut share, &clock, Phase::Prefill, 0.3, target);
                }
            }
            let busy = share.busy();
            let measured = busy.decode / (busy.decode + busy.prefill);
            assert!((measured - target).abs() < 0.02, "{target}: {measured}");
        }
    }

    #[test]
    fn debt_resets_when_contention_ends() {
        let clock = ManualClock::new();
        let mut share = TimeShare::new(DecodeShare::new(0.5, 0.5).ok());
        clock.install(&mut share);
        share.observe(true);
        run(&mut share, &clock, Phase::Prefill, 1.0, 0.5);
        assert!(share.decode_owed());
        share.observe(false);
        assert!(!share.decode_owed());
        share.observe(true);
        assert!(!share.decode_owed());
    }

    #[test]
    fn decode_without_contention_does_not_build_credit() {
        let clock = ManualClock::new();
        let mut share = TimeShare::new(DecodeShare::new(0.5, 0.5).ok());
        clock.install(&mut share);
        share.observe(false);
        for _ in 0..10 {
            run(&mut share, &clock, Phase::Decode, 0.1, 0.5);
        }
        share.observe(true);
        run(&mut share, &clock, Phase::Prefill, 1.0, 0.5);
        assert!(share.decode_owed());
    }

    #[test]
    fn overlapped_steps_are_charged_from_the_previous_return() {
        let clock = ManualClock::new();
        let mut share = TimeShare::new(None);
        clock.install(&mut share);
        share.launched(Phase::Decode, None);
        clock.advance(0.01);
        share.launched(Phase::Decode, None);
        clock.advance(0.04);
        share.returned(true);
        clock.advance(0.05);
        share.returned(true);
        assert!((share.busy().decode - 0.1).abs() < 1e-9);
    }

    #[test]
    fn rejected_launches_charge_nothing() {
        let clock = ManualClock::new();
        let mut share = TimeShare::new(DecodeShare::new(0.5, 0.5).ok());
        clock.install(&mut share);
        share.observe(true);
        share.launched(Phase::Prefill, Some(0.5));
        clock.advance(1.0);
        share.returned(false);
        assert_eq!(share.busy().prefill, 0.0);
        assert!(!share.decode_owed());
    }
}
