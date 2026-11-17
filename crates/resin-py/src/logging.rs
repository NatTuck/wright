//! Native log records reach Python's `logging` module: a record with target
//! `wright.http` goes to the `wright.http` logger, which the server's log
//! setup configures through its `wright` parent. Nothing is logged until
//! Python sets a level with [`set_log_level`], so the facade costs one atomic
//! load per call site while it is off.
//!
//! With a sink ([`set_log_sink`]), native code writes its lines itself, in the
//! format of the server's handlers (stderr, and a log file if any), and calls no
//! Python: a serving thread logs each step without the GIL.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, RwLock};

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;

/// The frontend's logger: request lifecycle lines.
pub(crate) const HTTP: &str = "wright.http";

struct PythonLogger;

static LOGGER: PythonLogger = PythonLogger;
static LOGGING: OnceLock<Py<PyAny>> = OnceLock::new();

/// Where native lines go without Python: `{time} {LEVEL} {prefix}{message}`.
struct Sink {
    prefix: String,
    /// coloredlogs' default styles on stderr
    colors: bool,
    file: Option<Mutex<File>>,
}

static SINK: RwLock<Option<Sink>> = RwLock::new(None);

/// Python's `levelname` of a level.
fn level_name(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Warn => "WARNING",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "Level 5",
    }
}

/// coloredlogs' default message style of a level (none for info).
fn level_style(level: Level) -> Option<&'static str> {
    match level {
        Level::Error => Some("31"),
        Level::Warn => Some("33"),
        Level::Info => None,
        Level::Debug => Some("32"),
        Level::Trace => Some("2;32"),
    }
}

/// The local time as `%H:%M:%S` (`logging.Formatter`'s `asctime` with the server's date format).
fn clock() -> String {
    // SAFETY: localtime_r writes only the `tm` it is given
    unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&now, &mut tm).is_null() {
            return "??:??:??".into();
        }
        format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

/// The record's line as the sink's handlers format it: (stderr, the log file).
fn lines(sink: &Sink, level: Level, message: &str) -> (String, String) {
    let time = clock();
    let name = level_name(level);
    let plain = format!("{time} {name} {}{message}\n", sink.prefix);
    if !sink.colors {
        return (plain.clone(), plain);
    }
    let body = match level_style(level) {
        Some(style) => format!("\x1b[{style}m{message}\x1b[0m"),
        None => message.to_string(),
    };
    let colored = format!(
        "\x1b[32m{time}\x1b[0m \x1b[1;30m{name}\x1b[0m {}{body}\n",
        sink.prefix
    );
    (colored, plain)
}

fn python_level(level: Level) -> u8 {
    match level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

impl Log for PythonLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let target = record.target();
        let name = if target.starts_with("wright") {
            target
        } else {
            "wright.native"
        };
        let message = record.args().to_string();
        if let Ok(guard) = SINK.read()
            && let Some(sink) = guard.as_ref()
        {
            let (terminal, file) = lines(sink, record.level(), &message);
            let _ = std::io::stderr().lock().write_all(terminal.as_bytes());
            if let Some(out) = &sink.file {
                let mut out = out.lock().unwrap_or_else(|error| error.into_inner());
                let _ = out.write_all(file.as_bytes());
            }
            return;
        }
        let level = python_level(record.level());
        Python::attach(|py| {
            let logging = match LOGGING.get() {
                Some(logging) => logging.bind(py).clone(),
                None => {
                    let Ok(logging) = py.import("logging") else {
                        return;
                    };
                    let logging = logging.into_any();
                    let _ = LOGGING.set(logging.clone().unbind());
                    logging
                }
            };
            let _ = logging
                .call_method1("getLogger", (name,))
                .and_then(|logger| logger.call_method1("log", (level, message)));
        });
    }

    fn flush(&self) {}
}

/// Route the native crates' log records to Python, off until a level is set.
pub(crate) fn install() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Off);
    }
}

/// Let native code log at and above a Python logging level (10 debug, 20 info,
/// 30 warning, 40 error). Levels above error silence it; `logging.NOTSET` (0)
/// and other levels at most 5 enable trace output.
#[pyfunction]
pub(crate) fn set_log_level(level: i64) {
    log::set_max_level(match level {
        ..=5 => LevelFilter::Trace,
        6..=10 => LevelFilter::Debug,
        11..=20 => LevelFilter::Info,
        21..=30 => LevelFilter::Warn,
        31..=40 => LevelFilter::Error,
        _ => LevelFilter::Off,
    });
}

/// Write native lines without Python, as the server's handlers do: `prefix` before each message
/// (a rank's or a replica's), coloredlogs' colors on stderr when `colors`, and each line also
/// appended to `path`. `None` sends native records to Python's `logging` again.
#[pyfunction]
#[pyo3(signature = (prefix=None, colors=false, path=None))]
pub(crate) fn set_log_sink(
    prefix: Option<String>,
    colors: bool,
    path: Option<PathBuf>,
) -> PyResult<()> {
    let sink = match prefix {
        None => None,
        Some(prefix) => {
            let file = match path {
                Some(path) => Some(Mutex::new(
                    OpenOptions::new().create(true).append(true).open(path)?,
                )),
                None => None,
            };
            Some(Sink {
                prefix,
                colors,
                file,
            })
        }
    };
    *SINK.write().unwrap_or_else(|error| error.into_inner()) = sink;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_match_the_python_handlers() {
        let sink = Sink {
            prefix: "rank 1: ".into(),
            colors: true,
            file: None,
        };
        let (terminal, file) = lines(&sink, Level::Warn, "slow");
        let time = &file[..8];
        assert_eq!(file, format!("{time} WARNING rank 1: slow\n"));
        assert_eq!(
            terminal,
            format!("\x1b[32m{time}\x1b[0m \x1b[1;30mWARNING\x1b[0m rank 1: \x1b[33mslow\x1b[0m\n")
        );
        let (terminal, _) = lines(&sink, Level::Info, "step");
        assert!(terminal.ends_with("\x1b[1;30mINFO\x1b[0m rank 1: step\n"));
    }
}
