//! A step trace for host-path studies (`RESIN_STEP_TRACE=<dir>`): each process appends `(CLOCK_MONOTONIC ns,
//! event, argument)` lines to `<dir>/trace.<pid>`. The clock is the same in every process of the host, so the
//! ranks' traces line up. Off (one relaxed load an event) unless the variable is set.

use std::io::Write;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

struct Trace {
    file: std::fs::File,
    events: Vec<(u64, &'static str, i64)>,
}

static STATE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 off, 2 on
static TRACE: OnceLock<Mutex<Trace>> = OnceLock::new();

fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime writes the timespec it is given.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn trace() -> Option<&'static Mutex<Trace>> {
    match STATE.load(Ordering::Relaxed) {
        1 => return None,
        2 => return TRACE.get(),
        _ => {}
    }
    let opened = std::env::var_os("RESIN_STEP_TRACE").and_then(|dir| {
        let path = std::path::Path::new(&dir).join(format!("trace.{}", std::process::id()));
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
    });
    match opened {
        Some(file) => {
            let _ = TRACE.set(Mutex::new(Trace {
                file,
                events: Vec::with_capacity(4096),
            }));
            STATE.store(2, Ordering::Relaxed);
            TRACE.get()
        }
        None => {
            STATE.store(1, Ordering::Relaxed);
            None
        }
    }
}

/// Record `event` with `arg` now (a no-op unless the trace is on).
pub(crate) fn mark(event: &'static str, arg: i64) {
    let Some(trace) = trace() else {
        return;
    };
    let t = now_ns();
    let mut trace = trace.lock().unwrap_or_else(|e| e.into_inner());
    trace.events.push((t, event, arg));
    if trace.events.len() >= 512 {
        flush_locked(&mut trace);
    }
}

fn flush_locked(trace: &mut Trace) {
    let mut text = String::with_capacity(trace.events.len() * 32);
    for (t, event, arg) in trace.events.drain(..) {
        text.push_str(&format!("{t} {event} {arg}\n"));
    }
    let _ = trace.file.write_all(text.as_bytes());
}

/// Write the buffered events out.
pub(crate) fn flush() {
    if let Some(trace) = trace() {
        flush_locked(&mut trace.lock().unwrap_or_else(|e| e.into_inner()));
    }
}
