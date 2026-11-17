//! One card shared by a rank of a tensor-parallel model and a one-card replica in the same process (the port of
//! `resin_models_gaudi/synapse/card_gate.py`'s `CardGate`).
//!
//! The TP model goes first. A replica step starts when the TP model has had no step on this card for `holdoff`
//! and its last step is done on the device. With a `share` above 0, a replica step can also start while the TP
//! model is busy, as long as the replica has used less than its part of the card's time. A replica step in the
//! TP model's busy time is charged its device time, which timed events measure.
//!
//! The two models alternate whole steps, and events order them on the device: a replica step that starts while
//! the TP model is busy waits on the device for the TP step before it, and the next TP step waits on the device
//! for that replica step. Each side calls the gate on its own thread, with its own runtime context current.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Seconds with no TP step on the card before a replica step starts in the TP model's idle time.
pub const HOLDOFF: f64 = 0.005;
/// The most card time (seconds) that a replica with a share saves up while it has no work.
pub const CREDIT_CAP: f64 = 0.05;
const POLL: f64 = 0.0005;

/// The device calls the gate makes, on the calling thread's current context.
pub trait CardOps {
    fn record(&self, slot: i32) -> Result<(), String>;
    fn stream_wait(&self, slot: i32) -> Result<(), String>;
    /// whether what `slot` recorded is done
    fn done(&self, slot: i32) -> Result<bool, String>;
    /// device seconds from timed slot `start`'s record to timed slot `end`'s, None while either is pending
    fn elapsed(&self, start: i32, end: i32) -> Result<Option<f64>, String>;
}

impl CardOps for crate::glue::Glue {
    fn record(&self, slot: i32) -> Result<(), String> {
        crate::glue::Glue::record(self, slot)
    }
    fn stream_wait(&self, slot: i32) -> Result<(), String> {
        crate::glue::Glue::stream_wait(self, slot)
    }
    fn done(&self, slot: i32) -> Result<bool, String> {
        crate::glue::Glue::done(self, slot)
    }
    fn elapsed(&self, start: i32, end: i32) -> Result<Option<f64>, String> {
        crate::glue::Glue::elapsed(self, start, end)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// replica steps in the TP model's idle time
    pub idle_steps: u64,
    /// replica steps while the TP model was busy (its share)
    pub shared_steps: u64,
    /// TP steps that waited on the device for a replica step
    pub tp_waits: u64,
}

struct State {
    tp_busy: bool,
    tp_recorded: bool,
    tp_last: Option<Instant>,
    replica_pending: bool,
    shared: bool,
    /// a shared step whose device time is not charged yet
    unpaid: bool,
    last_cost: f64,
    credit: f64,
    credit_at: Instant,
    counts: Counts,
}

/// The order of the TP model's and the replica's steps on one card.
pub struct CardGate<O> {
    pub ops: O,
    /// event slots: the TP model's last step, the replica's last step, a shared replica step's start
    pub tp_slot: i32,
    pub replica_slot: i32,
    pub replica_start: i32,
    /// the replica's fraction of the card's time while both models have work (`share` / (1 + `share`))
    share: f64,
    holdoff: f64,
    state: Mutex<State>,
    ended: Condvar,
}

impl<O: CardOps> CardGate<O> {
    /// `first`: three event slots (the replica's two timed); `share`: the replica's weight against the TP model's 1.
    pub fn new(ops: O, first: i32, share: f64, holdoff: f64) -> Result<Self, String> {
        if share < 0.0 || share.is_nan() {
            return Err("share takes 0 or more".into());
        }
        Ok(Self {
            ops,
            tp_slot: first,
            replica_slot: first + 1,
            replica_start: first + 2,
            share: share / (1.0 + share),
            holdoff,
            state: Mutex::new(State {
                tp_busy: false,
                tp_recorded: false,
                tp_last: None,
                replica_pending: false,
                shared: false,
                unpaid: false,
                last_cost: 0.0,
                credit: 0.0,
                credit_at: Instant::now(),
                counts: Counts::default(),
            }),
            ended: Condvar::new(),
        })
    }

    pub fn counts(&self) -> Counts {
        self.state.lock().expect("the gate's lock").counts
    }

    pub fn tp_busy(&self) -> bool {
        self.state.lock().expect("the gate's lock").tp_busy
    }

    /// A TP step starts on this card. It waits on the device for a replica step that the card has not run yet.
    pub fn tp_begin(&self) -> Result<(), String> {
        let mut s = self.state.lock().expect("the gate's lock");
        s.tp_busy = true;
        s.tp_last = Some(Instant::now());
        if s.replica_pending {
            s.replica_pending = false;
            s.counts.tp_waits += 1;
            self.ops.stream_wait(self.replica_slot)?;
        }
        Ok(())
    }

    /// The TP step's work is on the device: record it, so that a replica step can wait for it.
    pub fn tp_end(&self) -> Result<(), String> {
        let mut s = self.state.lock().expect("the gate's lock");
        let recorded = self.ops.record(self.tp_slot);
        s.tp_recorded = true;
        s.tp_busy = false;
        s.tp_last = Some(Instant::now());
        drop(s);
        self.ended.notify_all();
        recorded
    }

    /// Wait up to `timeout` for the replica's turn. False: not this time (the replica then reads its requests and
    /// asks again).
    pub fn acquire(&self, timeout: Duration) -> Result<bool, String> {
        let deadline = Instant::now() + timeout;
        let mut s = self.state.lock().expect("the gate's lock");
        loop {
            let now = Instant::now();
            self.accrue(&mut s, now)?;
            if self.tp_idle(&s, now)? {
                s.counts.idle_steps += 1;
                s.shared = false;
                return Ok(true);
            }
            if self.share > 0.0 && s.credit >= 0.0 {
                s.counts.shared_steps += 1;
                s.shared = true;
                if s.unpaid {
                    // still on the device: charge what the last one took
                    s.unpaid = false;
                    s.credit -= s.last_cost;
                }
                if s.tp_recorded {
                    self.ops.stream_wait(self.tp_slot)?;
                }
                self.ops.record(self.replica_start)?;
                return Ok(true);
            }
            if now >= deadline {
                return Ok(false);
            }
            let wait = if !s.tp_busy {
                // the holdoff's rest, then poll the TP step's event
                POLL.max(self.holdoff - since(s.tp_last, now))
            } else if self.share > 0.0 {
                POLL.max(-s.credit / self.share)
            } else {
                (deadline - now).as_secs_f64() // tp_end wakes it
            };
            let wait = Duration::from_secs_f64(wait).min(deadline - now);
            s = self.ended.wait_timeout(s, wait).expect("the gate's lock").0;
        }
    }

    /// The replica's step is on the device: record it, so that the next TP step waits for it.
    pub fn release(&self) -> Result<(), String> {
        let mut s = self.state.lock().expect("the gate's lock");
        self.ops.record(self.replica_slot)?;
        s.replica_pending = true;
        // the step's own time counts in the time the share is of
        self.accrue(&mut s, Instant::now())?;
        s.unpaid = s.shared;
        Ok(())
    }

    fn tp_idle(&self, s: &State, now: Instant) -> Result<bool, String> {
        if s.tp_busy || since(s.tp_last, now) < self.holdoff {
            return Ok(false);
        }
        Ok(!s.tp_recorded || self.ops.done(self.tp_slot)?)
    }

    /// The replica earns its part of the time that passes while the TP model is busy, and pays the device time of
    /// its last shared step once the device has run it.
    fn accrue(&self, s: &mut State, now: Instant) -> Result<(), String> {
        if s.tp_busy || since(s.tp_last, now) < self.holdoff {
            let earned = (now - s.credit_at).as_secs_f64() * self.share;
            s.credit = CREDIT_CAP.min(s.credit + earned);
        }
        s.credit_at = now;
        if s.unpaid {
            // accounting must not stop the replica: an error charges the last cost again
            let cost = match self.ops.elapsed(self.replica_start, self.replica_slot) {
                Ok(cost) => cost,
                Err(_) => Some(s.last_cost),
            };
            if let Some(cost) = cost {
                s.unpaid = false;
                s.last_cost = cost;
                s.credit -= cost;
            }
        }
        Ok(())
    }
}

/// Seconds from `last` to `now` (infinitely long without a last).
fn since(last: Option<Instant>, now: Instant) -> f64 {
    last.map_or(f64::INFINITY, |last| (now - last).as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Records every call with the calling thread's "context" (its name); every event is done at once, and a
    /// timed pair took the host time between its records.
    #[derive(Default)]
    struct Calls(
        Mutex<Vec<(String, i32, String)>>,
        Mutex<std::collections::HashMap<i32, Instant>>,
    );

    impl Calls {
        fn push(&self, call: &str, slot: i32) {
            let context = std::thread::current().name().unwrap_or("main").to_string();
            self.0.lock().unwrap().push((call.into(), slot, context));
            if call == "record" {
                self.1.lock().unwrap().insert(slot, Instant::now());
            }
        }

        fn all(&self) -> Vec<(String, i32, String)> {
            self.0.lock().unwrap().clone()
        }
    }

    impl CardOps for Arc<Calls> {
        fn record(&self, slot: i32) -> Result<(), String> {
            self.push("record", slot);
            Ok(())
        }
        fn stream_wait(&self, slot: i32) -> Result<(), String> {
            self.push("stream_wait", slot);
            Ok(())
        }
        fn done(&self, _slot: i32) -> Result<bool, String> {
            Ok(true)
        }
        fn elapsed(&self, start: i32, end: i32) -> Result<Option<f64>, String> {
            let times = self.1.lock().unwrap();
            Ok(match (times.get(&start), times.get(&end)) {
                (Some(a), Some(b)) if b >= a => Some((*b - *a).as_secs_f64()),
                _ => None,
            })
        }
    }

    /// A TP model busy for `seconds`: one step after another, `step` each, on thread "tp".
    fn tp_steps(
        gate: Arc<CardGate<Arc<Calls>>>,
        seconds: f64,
        step: f64,
    ) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("tp".into())
            .spawn(move || {
                let end = Instant::now() + Duration::from_secs_f64(seconds);
                while Instant::now() < end {
                    gate.tp_begin().unwrap();
                    std::thread::sleep(Duration::from_secs_f64(step));
                    gate.tp_end().unwrap();
                }
            })
            .unwrap()
    }

    #[test]
    fn the_tp_model_goes_first_and_the_replica_steps_when_it_is_idle() {
        let calls = Arc::new(Calls::default());
        let gate = Arc::new(CardGate::new(Arc::clone(&calls), 10, 0.0, 0.01).unwrap());
        let tp = tp_steps(Arc::clone(&gate), 0.3, 0.004);
        std::thread::sleep(Duration::from_millis(20));
        let g = Arc::clone(&gate);
        let replica = std::thread::Builder::new()
            .name("replica".into())
            .spawn(move || {
                let started = Instant::now();
                assert!(!g.acquire(Duration::from_millis(100)).unwrap()); // busy: no turn, and the call returns
                while !g.acquire(Duration::from_millis(100)).unwrap() {}
                let waited = started.elapsed().as_secs_f64();
                g.release().unwrap();
                waited
            })
            .unwrap();
        let waited = replica.join().unwrap();
        tp.join().unwrap();
        assert!(waited >= 0.25, "{waited}");
        assert_eq!(
            gate.counts(),
            Counts {
                idle_steps: 1,
                shared_steps: 0,
                tp_waits: 0
            }
        );
        // the replica's step needed no wait for the TP step: the TP step was done
        let all = calls.all();
        assert!(
            !all.iter()
                .any(|(c, s, _)| c == "stream_wait" && *s == gate.tp_slot)
        );
        assert_eq!(
            all.last().unwrap(),
            &("record".into(), gate.replica_slot, "replica".into())
        );
        // the next TP step waits on the device for the replica's step
        gate.tp_begin().unwrap();
        gate.tp_end().unwrap();
        let all = calls.all();
        let tail: Vec<(String, i32)> = all[all.len() - 2..]
            .iter()
            .map(|(c, s, _)| (c.clone(), *s))
            .collect();
        assert_eq!(
            tail,
            vec![
                ("stream_wait".into(), gate.replica_slot),
                ("record".into(), gate.tp_slot)
            ]
        );
        assert_eq!(gate.counts().tp_waits, 1);
    }

    #[test]
    fn a_replica_with_a_share_steps_while_the_tp_model_is_busy() {
        let calls = Arc::new(Calls::default());
        let gate = Arc::new(CardGate::new(Arc::clone(&calls), 10, 1.0, 0.01).unwrap()); // an even split
        let tp = tp_steps(Arc::clone(&gate), 0.6, 0.004);
        let alive = Arc::new(AtomicBool::new(true));
        std::thread::sleep(Duration::from_millis(20));
        let (g, a) = (Arc::clone(&gate), Arc::clone(&alive));
        let replica = std::thread::Builder::new()
            .name("replica".into())
            .spawn(move || {
                let (mut used, started) = (0.0, Instant::now());
                while a.load(Ordering::SeqCst) {
                    if g.acquire(Duration::from_millis(50)).unwrap() {
                        let begin = Instant::now();
                        std::thread::sleep(Duration::from_millis(3));
                        used += begin.elapsed().as_secs_f64();
                        g.release().unwrap();
                    }
                }
                used / started.elapsed().as_secs_f64()
            })
            .unwrap();
        tp.join().unwrap();
        alive.store(false, Ordering::SeqCst);
        let part = replica.join().unwrap();
        assert!(gate.counts().shared_steps > 10, "{:?}", gate.counts());
        assert!(part > 0.2 && part < 0.65, "{part}");
        // a shared step waits on the device for the TP step before it, on the replica's context
        let waits: Vec<String> = calls
            .all()
            .into_iter()
            .filter(|(c, s, _)| c == "stream_wait" && *s == gate.tp_slot)
            .map(|(_, _, context)| context)
            .collect();
        assert!(!waits.is_empty() && waits.iter().all(|c| c == "replica"));
    }

    #[test]
    fn share_must_not_be_negative() {
        assert!(CardGate::new(Arc::new(Calls::default()), 0, -1.0, HOLDOFF).is_err());
    }
}
