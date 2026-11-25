//! One step program for every Gaudi model: the stream calls of a model step (graph launches with their
//! bindings, collectives, device copies, uploads, downloads, memsets, event records), made in one native call.
//!
//! A step's fill decides a few values per step: counts, lengths, addresses, flags. They go in an array of `u64`
//! registers, and an operand is a constant or `base + register * scale`. Two ops give control: `Take` makes the
//! first n ops of a list (n from a register), and `Select` makes one of several arms, chosen by a table over
//! key registers. A program has named sections, and one call runs a sequence of (program, section).
//!
//! `Program::run` walks the ops and makes each call on a `Stream`: the card (`CardStream`, through the glue of
//! the runtime library) or a `Recorder`, which keeps the calls. Tests and check mode read the recorded calls,
//! so what they compare is the same walk that runs on a card.

use std::ffi::{c_int, c_void};
use std::fmt;

use crate::glue::Glue;

pub type Register = u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Const(u64),
    /// `base + registers[register] * scale`
    Scaled {
        base: u64,
        register: Register,
        scale: u64,
    },
}

impl From<u64> for Operand {
    fn from(value: u64) -> Self {
        Self::Const(value)
    }
}

impl Operand {
    fn value(self, registers: &[u64]) -> Result<u64, String> {
        match self {
            Self::Const(value) => Ok(value),
            Self::Scaled {
                base,
                register,
                scale,
            } => {
                let value = registers[usize::from(register)];
                value
                    .checked_mul(scale)
                    .and_then(|scaled| scaled.checked_add(base))
                    .ok_or_else(|| {
                        format!("{base} + register {register} ({value}) * {scale} overflows")
                    })
            }
        }
    }

    fn register(self) -> Option<Register> {
        match self {
            Self::Const(_) => None,
            Self::Scaled { register, .. } => Some(register),
        }
    }
}

/// One graph launch and the bindings it sets first. Bindings stay on the recipe between launches, so a launch
/// sets only the slots that change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Launch {
    pub recipe: i32,
    pub slots: Vec<c_int>,
    pub addresses: Vec<u64>,
    /// Bindings whose address the step decides. They are set after the fixed ones and in order, so the last
    /// binding of a slot wins.
    pub dynamic: Vec<(c_int, Operand)>,
}

impl Launch {
    pub fn new(recipe: i32) -> Self {
        Self {
            recipe,
            ..Self::default()
        }
    }

    pub fn bind(mut self, slot: c_int, address: u64) -> Self {
        self.slots.push(slot);
        self.addresses.push(address);
        self
    }

    pub fn bind_dynamic(mut self, slot: c_int, address: Operand) -> Self {
        self.dynamic.push((slot, address));
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    Launch(Launch),
    AllGather {
        send: Operand,
        recv: Operand,
        count: Operand,
        bf16: bool,
    },
    /// in place
    AllReduce {
        address: Operand,
        count: Operand,
        bf16: bool,
    },
    /// device to device (`rg_d2d_async`)
    Copy {
        dst: Operand,
        src: Operand,
        bytes: Operand,
    },
    /// several device copies in one call (`rg_d2d_multi`)
    CopyMulti {
        src: Vec<u64>,
        bytes: Vec<u64>,
        dst: Vec<u64>,
    },
    /// from pinned host memory (`rg_h2d_async`)
    Upload {
        dst: Operand,
        host: Operand,
        bytes: Operand,
    },
    /// through the runtime's ring of pinned slots (`rg_h2d_nb`): the host bytes are read when the walk gets here
    UploadRing {
        dst: Operand,
        host: Operand,
        bytes: Operand,
    },
    /// to pinned host memory (`rg_d2h_async`)
    Download {
        host: Operand,
        src: Operand,
        bytes: Operand,
    },
    /// `count` 32-bit words of `value` (`rg_memset32`)
    Memset {
        dst: Operand,
        value: u32,
        count: Operand,
    },
    /// record the stream event of `slot`
    Record {
        slot: i32,
    },
    /// the first `registers[register]` ops
    Take {
        register: Register,
        ops: Vec<Op>,
    },
    /// The arm `table[index]`, where `index` combines the key registers' values with the first key the most
    /// significant (each value below its radix). A table entry of -1 means no arm, which is an error.
    Select {
        keys: Vec<(Register, u64)>,
        table: Vec<i32>,
        arms: Vec<Vec<Op>>,
    },
}

impl Op {
    fn is_collective(&self) -> bool {
        matches!(self, Self::AllGather { .. } | Self::AllReduce { .. })
    }

    /// Whether this op, or any op inside it, is a collective.
    fn has_collective(&self) -> bool {
        match self {
            Self::Take { ops, .. } => ops.iter().any(Self::has_collective),
            Self::Select { arms, .. } => arms.iter().flatten().any(Self::has_collective),
            op => op.is_collective(),
        }
    }
}

/// A call on the stream, with every operand evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Call {
    Launch {
        recipe: i32,
        binds: Vec<(c_int, u64)>,
    },
    AllGather {
        send: u64,
        recv: u64,
        count: u64,
        bf16: bool,
    },
    AllReduce {
        address: u64,
        count: u64,
        bf16: bool,
    },
    Copy {
        dst: u64,
        src: u64,
        bytes: u64,
    },
    CopyMulti {
        src: Vec<u64>,
        bytes: Vec<u64>,
        dst: Vec<u64>,
    },
    Upload {
        dst: u64,
        host: u64,
        bytes: u64,
    },
    UploadRing {
        dst: u64,
        host: u64,
        bytes: u64,
    },
    Download {
        host: u64,
        src: u64,
        bytes: u64,
    },
    Memset {
        dst: u64,
        value: u32,
        count: u64,
    },
    Record {
        slot: i32,
    },
}

/// Where a program's calls go.
pub trait Stream {
    /// Set the bindings in order, then launch.
    fn launch(&mut self, recipe: i32, slots: &[c_int], addresses: &[u64]) -> Result<(), String>;
    fn all_gather(&mut self, send: u64, recv: u64, count: u64, bf16: bool) -> Result<(), String>;
    fn all_reduce(&mut self, address: u64, count: u64, bf16: bool) -> Result<(), String>;
    fn copy(&mut self, dst: u64, src: u64, bytes: u64) -> Result<(), String>;
    fn copy_multi(&mut self, src: &[u64], bytes: &[u64], dst: &[u64]) -> Result<(), String>;
    fn upload(&mut self, dst: u64, host: u64, bytes: u64) -> Result<(), String>;
    fn upload_ring(&mut self, dst: u64, host: u64, bytes: u64) -> Result<(), String>;
    fn download(&mut self, host: u64, src: u64, bytes: u64) -> Result<(), String>;
    fn memset(&mut self, dst: u64, value: u32, count: u64) -> Result<(), String>;
    fn record(&mut self, slot: i32) -> Result<(), String>;
}

/// A stream that keeps the calls. With `fail_at = Some(n)`, call n fails (for tests of partial steps).
#[derive(Clone, Debug, Default)]
pub struct Recorder {
    pub calls: Vec<Call>,
    pub fail_at: Option<usize>,
}

impl Recorder {
    fn keep(&mut self, call: Call) -> Result<(), String> {
        if self.fail_at == Some(self.calls.len()) {
            return Err(format!("call {} fails", self.calls.len()));
        }
        self.calls.push(call);
        Ok(())
    }
}

impl Stream for Recorder {
    fn launch(&mut self, recipe: i32, slots: &[c_int], addresses: &[u64]) -> Result<(), String> {
        let binds = slots
            .iter()
            .copied()
            .zip(addresses.iter().copied())
            .collect();
        self.keep(Call::Launch { recipe, binds })
    }

    fn all_gather(&mut self, send: u64, recv: u64, count: u64, bf16: bool) -> Result<(), String> {
        self.keep(Call::AllGather {
            send,
            recv,
            count,
            bf16,
        })
    }

    fn all_reduce(&mut self, address: u64, count: u64, bf16: bool) -> Result<(), String> {
        self.keep(Call::AllReduce {
            address,
            count,
            bf16,
        })
    }

    fn copy(&mut self, dst: u64, src: u64, bytes: u64) -> Result<(), String> {
        self.keep(Call::Copy { dst, src, bytes })
    }

    fn copy_multi(&mut self, src: &[u64], bytes: &[u64], dst: &[u64]) -> Result<(), String> {
        self.keep(Call::CopyMulti {
            src: src.to_vec(),
            bytes: bytes.to_vec(),
            dst: dst.to_vec(),
        })
    }

    fn upload(&mut self, dst: u64, host: u64, bytes: u64) -> Result<(), String> {
        self.keep(Call::Upload { dst, host, bytes })
    }

    fn upload_ring(&mut self, dst: u64, host: u64, bytes: u64) -> Result<(), String> {
        self.keep(Call::UploadRing { dst, host, bytes })
    }

    fn download(&mut self, host: u64, src: u64, bytes: u64) -> Result<(), String> {
        self.keep(Call::Download { host, src, bytes })
    }

    fn memset(&mut self, dst: u64, value: u32, count: u64) -> Result<(), String> {
        self.keep(Call::Memset { dst, value, count })
    }

    fn record(&mut self, slot: i32) -> Result<(), String> {
        self.keep(Call::Record { slot })
    }
}

/// The card's stream, through the glue of the runtime library the Python device loaded.
pub struct CardStream<'a> {
    glue: &'a Glue,
    debug: &'static Debug,
    calls: usize,
}

/// Debug aids for ordering problems, read from the environment once per process (all off by default).
#[derive(Debug, Default)]
struct Debug {
    /// `RESIN_STEP_SYNC`: wait for the stream after each call of these kinds: `launch`, `collective`, `copy`
    /// (uploads, downloads, copies, memsets) or `all`, comma-separated
    sync_after: u8,
    /// `RESIN_STEP_FENCE`: record an event and make the stream wait for it `before`, `after` or `both` sides of
    /// each collective (event slot 63, which nothing else may use then)
    fence: u8,
    /// `RESIN_STEP_SYNC_EVERY=N`: wait for the stream after every N calls
    every: usize,
    /// `RESIN_STEP_SYNC_RECIPES`: wait for the stream after each launch of these recipe handles (comma-separated)
    recipes: Vec<i32>,
    /// `RESIN_STEP_PROF=1`: time each launch on the device (now -> launch -> sync -> elapsed) and print
    /// `PROF\t<recipe_handle>\t<us>`; the caller maps the handle to a name via the compile order
    prof: bool,
}

fn debug() -> &'static Debug {
    static DEBUG: std::sync::OnceLock<Debug> = std::sync::OnceLock::new();
    DEBUG.get_or_init(|| Debug {
        sync_after: sync_kinds(),
        fence: fence_sides(),
        every: std::env::var("RESIN_STEP_SYNC_EVERY")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0),
        recipes: std::env::var("RESIN_STEP_SYNC_RECIPES")
            .map(|list| {
                list.split(',')
                    .filter_map(|r| r.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default(),
        prof: std::env::var("RESIN_STEP_PROF").map(|v| v == "1").unwrap_or(false),
    })
}

const FENCE_SLOT: c_int = 63;

fn fence_sides() -> u8 {
    match std::env::var("RESIN_STEP_FENCE").as_deref() {
        Ok("before") => 1,
        Ok("after") => 2,
        Ok("both") => 3,
        _ => 0,
    }
}

const SYNC_LAUNCH: u8 = 1;
const SYNC_COLLECTIVE: u8 = 2;
const SYNC_COPY: u8 = 4;

fn sync_kinds() -> u8 {
    let Ok(value) = std::env::var("RESIN_STEP_SYNC") else {
        return 0;
    };
    value
        .split(',')
        .map(|kind| match kind.trim() {
            "launch" => SYNC_LAUNCH,
            "collective" => SYNC_COLLECTIVE,
            "copy" => SYNC_COPY,
            "all" => SYNC_LAUNCH | SYNC_COLLECTIVE | SYNC_COPY,
            _ => 0,
        })
        .fold(0, |kinds, kind| kinds | kind)
}

impl<'a> CardStream<'a> {
    /// # Safety
    /// For every call of the programs run on this stream: recipes and event slots are valid; bound and copied
    /// device addresses are allocations of the loaded runtime with room for the call; host addresses point to
    /// live memory of at least `bytes`, pinned for `upload` and `download`, which stays alive until the stream
    /// has passed the copy.
    pub unsafe fn new(glue: &'a Glue) -> Self {
        Self {
            glue,
            debug: debug(),
            calls: 0,
        }
    }

    fn check(&self, status: c_int, what: impl FnOnce() -> String) -> Result<(), String> {
        if status < 0 {
            Err(self.glue.error(&what()))
        } else {
            Ok(())
        }
    }

    /// With `RESIN_STEP_FENCE` side `side`: an event on the stream, which the stream waits for.
    fn fence(&self, side: u8) -> Result<(), String> {
        if self.debug.fence & side != 0 {
            // SAFETY: the runtime checks the slot.
            let status = unsafe { (self.glue.record)(FENCE_SLOT) };
            self.check(status, || "fence record".to_owned())?;
            self.glue.stream_wait(FENCE_SLOT)?;
        }
        Ok(())
    }

    /// A call of `kind` is issued: with `RESIN_STEP_SYNC` (or every `RESIN_STEP_SYNC_EVERY` calls), wait until the
    /// stream has done it.
    fn done(&mut self, kind: u8, result: Result<(), String>) -> Result<(), String> {
        result?;
        self.calls += 1;
        let d = self.debug;
        if d.sync_after & kind != 0 || (d.every > 0 && self.calls.is_multiple_of(d.every)) {
            self.glue.sync()?;
        }
        Ok(())
    }
}

impl Stream for CardStream<'_> {
    fn launch(&mut self, recipe: i32, slots: &[c_int], addresses: &[u64]) -> Result<(), String> {
        debug_assert_eq!(slots.len(), addresses.len());
        if !slots.is_empty() {
            let count = c_int::try_from(slots.len()).map_err(|_| "too many bindings".to_owned())?;
            // SAFETY: both arrays hold `count` entries; the addresses are valid by `CardStream::new`.
            let status = unsafe {
                (self.glue.bind_slots)(recipe, slots.as_ptr(), addresses.as_ptr(), count)
            };
            self.check(status, || format!("bindings of recipe {recipe}"))?;
        }
        // SAFETY: the recipe is valid by `CardStream::new`.
        let t0 = if self.debug.prof { Some(std::time::Instant::now()) } else { None };
        let status = unsafe { (self.glue.launch)(recipe) };
        self.check(status, || format!("launch of recipe {recipe}"))?;
        if let Some(t0) = t0 {
            // the sync waits for this (and only this, since the stream is empty before t0) launch: the device time
            self.glue.sync()?;
            eprintln!("PROF\t{recipe}\t{:.1}", t0.elapsed().as_secs_f64() * 1e6);
        }
        if self.debug.recipes.contains(&recipe) {
            self.glue.sync()?;
        }
        self.done(SYNC_LAUNCH, Ok(()))
    }

    fn all_gather(&mut self, send: u64, recv: u64, count: u64, bf16: bool) -> Result<(), String> {
        self.fence(1)?;
        // SAFETY: device addresses, valid by `CardStream::new`.
        let status = unsafe { (self.glue.allgather)(send, recv, count, c_int::from(bf16)) };
        self.check(status, || "allgather".to_owned())?;
        self.fence(2)?;
        self.done(SYNC_COLLECTIVE, Ok(()))
    }

    fn all_reduce(&mut self, address: u64, count: u64, bf16: bool) -> Result<(), String> {
        self.fence(1)?;
        // SAFETY: a device address, valid by `CardStream::new`.
        let status = unsafe { (self.glue.allreduce)(address, address, count, c_int::from(bf16)) };
        self.check(status, || "allreduce".to_owned())?;
        self.fence(2)?;
        self.done(SYNC_COLLECTIVE, Ok(()))
    }

    fn copy(&mut self, dst: u64, src: u64, bytes: u64) -> Result<(), String> {
        // SAFETY: device addresses, valid by `CardStream::new`.
        let status = unsafe { (self.glue.copy)(dst, src, bytes) };
        self.done(SYNC_COPY, self.check(status, || "device copy".to_owned()))
    }

    fn copy_multi(&mut self, src: &[u64], bytes: &[u64], dst: &[u64]) -> Result<(), String> {
        if src.len() != bytes.len() || src.len() != dst.len() {
            return Err("device copies with unequal lists".to_owned());
        }
        // SAFETY: three arrays of `src.len()` device addresses and sizes, valid by `CardStream::new`.
        let status = unsafe {
            (self.glue.copy_multi)(src.as_ptr(), bytes.as_ptr(), dst.as_ptr(), src.len() as u64)
        };
        self.done(SYNC_COPY, self.check(status, || "device copies".to_owned()))
    }

    fn upload(&mut self, dst: u64, host: u64, bytes: u64) -> Result<(), String> {
        // SAFETY: pinned host memory of `bytes`, alive until the copy is done, by `CardStream::new`.
        let status = unsafe { (self.glue.upload)(dst, host as *const c_void, bytes) };
        self.done(SYNC_COPY, self.check(status, || "upload".to_owned()))
    }

    fn upload_ring(&mut self, dst: u64, host: u64, bytes: u64) -> Result<(), String> {
        // SAFETY: host memory of `bytes`, read before the call returns, by `CardStream::new`.
        let status = unsafe { (self.glue.upload_ring)(dst, host as *const c_void, bytes) };
        self.done(SYNC_COPY, self.check(status, || "ring upload".to_owned()))
    }

    fn download(&mut self, host: u64, src: u64, bytes: u64) -> Result<(), String> {
        // SAFETY: pinned host memory of `bytes`, alive until the copy is done, by `CardStream::new`.
        let status = unsafe { (self.glue.download)(host as *mut c_void, src, bytes) };
        self.done(SYNC_COPY, self.check(status, || "download".to_owned()))
    }

    fn memset(&mut self, dst: u64, value: u32, count: u64) -> Result<(), String> {
        // SAFETY: a device address with room for `count` words, by `CardStream::new`.
        let status = unsafe { (self.glue.memset32)(dst, value, count) };
        self.done(SYNC_COPY, self.check(status, || "memset".to_owned()))
    }

    fn record(&mut self, slot: i32) -> Result<(), String> {
        // SAFETY: the runtime checks the slot.
        let status = unsafe { (self.glue.record)(slot) };
        self.check(status, || format!("record of event {slot}"))
    }
}

/// A failed run: the message, and how many calls the stream took before it. A run that failed after a call
/// may have changed device state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunError {
    pub message: String,
    pub issued: usize,
}

impl fmt::Display for RunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} (after {} stream calls)",
            self.message, self.issued
        )
    }
}

impl std::error::Error for RunError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub ops: Vec<Op>,
}

/// A built program: made by `Builder`, which checks it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    registers: usize,
    sections: Vec<Section>,
}

impl Program {
    /// The number of registers a run must give.
    pub fn registers(&self) -> usize {
        self.registers
    }

    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    pub fn section(&self, name: &str) -> Option<usize> {
        self.sections
            .iter()
            .position(|section| section.name == name)
    }

    /// Whether a section makes no collective, on any path. Sections that read results after a step (a
    /// harvest) must not, so that a rank never waits on a collective that another rank skipped.
    pub fn collective_free(&self, section: usize) -> bool {
        !self.sections[section].ops.iter().any(Op::has_collective)
    }

    /// Make the calls of `sections`, in that order, on `stream`. Returns the number of calls made.
    pub fn run<S: Stream>(
        &self,
        sections: &[usize],
        registers: &[u64],
        stream: &mut S,
    ) -> Result<usize, RunError> {
        let steps: Vec<(&Program, usize)> =
            sections.iter().map(|&section| (self, section)).collect();
        run_sequence(&steps, registers, stream)
    }

    /// The calls that `run` makes, recorded.
    pub fn calls(&self, sections: &[usize], registers: &[u64]) -> Result<Vec<Call>, RunError> {
        let mut recorder = Recorder::default();
        self.run(sections, registers, &mut recorder)?;
        Ok(recorder.calls)
    }
}

/// Make the calls of several programs' sections in order, with one register array. Returns the number of calls.
pub fn run_sequence<S: Stream>(
    steps: &[(&Program, usize)],
    registers: &[u64],
    stream: &mut S,
) -> Result<usize, RunError> {
    for &(program, section) in steps {
        if registers.len() < program.registers {
            return Err(RunError {
                message: format!(
                    "{} registers for a program of {}",
                    registers.len(),
                    program.registers
                ),
                issued: 0,
            });
        }
        if section >= program.sections.len() {
            return Err(RunError {
                message: format!("section {section} of {}", program.sections.len()),
                issued: 0,
            });
        }
    }
    let mut walk = Walk {
        registers,
        stream,
        slots: Vec::new(),
        addresses: Vec::new(),
        issued: 0,
    };
    for &(program, section) in steps {
        walk.ops(&program.sections[section].ops)
            .map_err(|message| RunError {
                message: format!("{}: {message}", program.sections[section].name),
                issued: walk.issued,
            })?;
    }
    Ok(walk.issued)
}

struct Walk<'a, S: Stream> {
    registers: &'a [u64],
    stream: &'a mut S,
    /// a launch's fixed and step bindings together
    slots: Vec<c_int>,
    addresses: Vec<u64>,
    issued: usize,
}

impl<S: Stream> Walk<'_, S> {
    fn ops(&mut self, ops: &[Op]) -> Result<(), String> {
        ops.iter().try_for_each(|op| self.op(op))
    }

    fn op(&mut self, op: &Op) -> Result<(), String> {
        let registers = self.registers;
        let value = |operand: &Operand| operand.value(registers);
        match op {
            Op::Launch(launch) => {
                if launch.dynamic.is_empty() {
                    self.stream
                        .launch(launch.recipe, &launch.slots, &launch.addresses)?;
                } else {
                    self.slots.clear();
                    self.addresses.clear();
                    self.slots.extend_from_slice(&launch.slots);
                    self.addresses.extend_from_slice(&launch.addresses);
                    for (slot, address) in &launch.dynamic {
                        self.slots.push(*slot);
                        self.addresses.push(value(address)?);
                    }
                    self.stream
                        .launch(launch.recipe, &self.slots, &self.addresses)?;
                }
            }
            Op::AllGather {
                send,
                recv,
                count,
                bf16,
            } => self
                .stream
                .all_gather(value(send)?, value(recv)?, value(count)?, *bf16)?,
            Op::AllReduce {
                address,
                count,
                bf16,
            } => self
                .stream
                .all_reduce(value(address)?, value(count)?, *bf16)?,
            Op::Copy { dst, src, bytes } => {
                self.stream.copy(value(dst)?, value(src)?, value(bytes)?)?
            }
            Op::CopyMulti { src, bytes, dst } => self.stream.copy_multi(src, bytes, dst)?,
            Op::Upload { dst, host, bytes } => {
                self.stream
                    .upload(value(dst)?, value(host)?, value(bytes)?)?
            }
            Op::UploadRing { dst, host, bytes } => {
                self.stream
                    .upload_ring(value(dst)?, value(host)?, value(bytes)?)?
            }
            Op::Download { host, src, bytes } => {
                self.stream
                    .download(value(host)?, value(src)?, value(bytes)?)?
            }
            Op::Memset {
                dst,
                value: word,
                count,
            } => self.stream.memset(value(dst)?, *word, value(count)?)?,
            Op::Record { slot } => self.stream.record(*slot)?,
            Op::Take { register, ops } => {
                let count = registers[usize::from(*register)];
                let taken = usize::try_from(count)
                    .ok()
                    .and_then(|count| ops.get(..count))
                    .ok_or_else(|| {
                        format!("register {register} takes {count} ops of {}", ops.len())
                    })?;
                return self.ops(taken);
            }
            Op::Select { keys, table, arms } => {
                let mut index = 0u64;
                for &(register, radix) in keys {
                    let key = registers[usize::from(register)];
                    if key >= radix {
                        return Err(format!("register {register} is {key}, not below {radix}"));
                    }
                    // below the table length, which the builder checked
                    index = index * radix + key;
                }
                let arm = table[index as usize];
                let arm = usize::try_from(arm).map_err(|_| {
                    let values: Vec<u64> = keys
                        .iter()
                        .map(|&(register, _)| registers[usize::from(register)])
                        .collect();
                    format!("no arm for key values {values:?}")
                })?;
                return self.ops(&arms[arm]);
            }
        }
        self.issued += 1;
        Ok(())
    }
}

enum Frame {
    Take {
        register: Register,
        ops: Vec<Op>,
    },
    /// ops go to the last arm
    Select {
        keys: Vec<(Register, u64)>,
        table: Vec<i32>,
        arms: Vec<Vec<Op>>,
    },
}

/// Builds a program one op at a time, the way a Python builder walks a model step: `section`, then ops, with
/// `take` or `select` (and `arm` for each arm) opening a nested list that `end` closes. Every op is checked as it
/// is added.
pub struct Builder {
    registers: usize,
    sections: Vec<Section>,
    open: Vec<Frame>,
}

impl Builder {
    pub fn new(registers: usize) -> Self {
        Self {
            registers,
            sections: Vec::new(),
            open: Vec::new(),
        }
    }

    /// Start a section; the ops that follow go in it.
    pub fn section(&mut self, name: &str) -> Result<usize, String> {
        if !self.open.is_empty() {
            return Err(format!(
                "section {name} starts inside an open take or select"
            ));
        }
        if self.sections.iter().any(|section| section.name == name) {
            return Err(format!("two sections named {name}"));
        }
        self.sections.push(Section {
            name: name.to_owned(),
            ops: Vec::new(),
        });
        Ok(self.sections.len() - 1)
    }

    pub fn push(&mut self, op: Op) -> Result<(), String> {
        let in_take = self
            .open
            .iter()
            .any(|frame| matches!(frame, Frame::Take { .. }));
        self.check(&op, in_take)?;
        self.target()?.push(op);
        Ok(())
    }

    /// Open a list of ops of which a step makes the first `registers[register]`.
    pub fn take(&mut self, register: Register) -> Result<(), String> {
        self.check_register(register)?;
        self.target()?;
        self.open.push(Frame::Take {
            register,
            ops: Vec::new(),
        });
        Ok(())
    }

    /// Open a choice of arms by `table` over the key registers (see `Op::Select`); `arm` starts each arm.
    pub fn select(&mut self, keys: Vec<(Register, u64)>, table: Vec<i32>) -> Result<(), String> {
        self.check_keys(&keys, &table)?;
        self.target()?;
        self.open.push(Frame::Select {
            keys,
            table,
            arms: Vec::new(),
        });
        Ok(())
    }

    /// Start the next arm of the open select; returns its number.
    pub fn arm(&mut self) -> Result<usize, String> {
        match self.open.last_mut() {
            Some(Frame::Select { arms, .. }) => {
                arms.push(Vec::new());
                Ok(arms.len() - 1)
            }
            _ => Err("an arm outside a select".to_owned()),
        }
    }

    /// Close the innermost open take or select.
    pub fn end(&mut self) -> Result<(), String> {
        let op = match self.open.pop() {
            None => return Err("end without an open take or select".to_owned()),
            Some(Frame::Take { register, ops }) => Op::Take { register, ops },
            Some(Frame::Select { keys, table, arms }) => {
                if let Some(arm) = table
                    .iter()
                    .find(|&&arm| arm >= 0 && arm as usize >= arms.len())
                {
                    return Err(format!("a select table names arm {arm} of {}", arms.len()));
                }
                Op::Select { keys, table, arms }
            }
        };
        // the target was there when the take or select opened
        self.target()?.push(op);
        Ok(())
    }

    /// The program built so far; the builder is then empty. On an error the builder is unchanged.
    pub fn finish(&mut self) -> Result<Program, String> {
        if !self.open.is_empty() {
            return Err(format!("{} takes or selects not ended", self.open.len()));
        }
        Ok(Program {
            registers: self.registers,
            sections: std::mem::take(&mut self.sections),
        })
    }

    fn target(&mut self) -> Result<&mut Vec<Op>, String> {
        match self.open.last_mut() {
            Some(Frame::Take { ops, .. }) => Ok(ops),
            Some(Frame::Select { arms, .. }) => arms
                .last_mut()
                .ok_or_else(|| "an op in a select before its first arm".to_owned()),
            None => self
                .sections
                .last_mut()
                .map(|section| &mut section.ops)
                .ok_or_else(|| "an op before the first section".to_owned()),
        }
    }

    fn check_register(&self, register: Register) -> Result<(), String> {
        if usize::from(register) < self.registers {
            Ok(())
        } else {
            Err(format!("register {register} of {}", self.registers))
        }
    }

    fn check_keys(&self, keys: &[(Register, u64)], table: &[i32]) -> Result<(), String> {
        let mut size = 1u64;
        for &(register, radix) in keys {
            self.check_register(register)?;
            if radix == 0 {
                return Err(format!("register {register} has a radix of 0"));
            }
            size = size
                .checked_mul(radix)
                .ok_or_else(|| "a select table too large".to_owned())?;
        }
        if size != table.len() as u64 {
            return Err(format!(
                "a select table of {} entries for {size} key values",
                table.len()
            ));
        }
        if let Some(arm) = table.iter().find(|&&arm| arm < -1) {
            return Err(format!("a select table entry of {arm}"));
        }
        Ok(())
    }

    fn check(&self, op: &Op, in_take: bool) -> Result<(), String> {
        if in_take && op.is_collective() {
            return Err(
                "a collective inside a take: every rank must make the same collectives".to_owned(),
            );
        }
        let operands: Vec<Operand> = match op {
            Op::Launch(launch) => {
                if launch.slots.len() != launch.addresses.len() {
                    return Err(format!(
                        "recipe {}: {} slots and {} addresses",
                        launch.recipe,
                        launch.slots.len(),
                        launch.addresses.len()
                    ));
                }
                launch.dynamic.iter().map(|&(_, address)| address).collect()
            }
            Op::AllGather {
                send, recv, count, ..
            } => vec![*send, *recv, *count],
            Op::AllReduce { address, count, .. } => vec![*address, *count],
            Op::Copy { dst, src, bytes } => vec![*dst, *src, *bytes],
            Op::CopyMulti { src, bytes, dst } => {
                if src.len() != bytes.len() || src.len() != dst.len() {
                    return Err("device copies with unequal lists".to_owned());
                }
                Vec::new()
            }
            Op::Upload { dst, host, bytes } | Op::UploadRing { dst, host, bytes } => {
                vec![*dst, *host, *bytes]
            }
            Op::Download { host, src, bytes } => vec![*host, *src, *bytes],
            Op::Memset { dst, count, .. } => vec![*dst, *count],
            Op::Record { .. } => Vec::new(),
            Op::Take { register, ops } => {
                self.check_register(*register)?;
                return ops.iter().try_for_each(|op| self.check(op, true));
            }
            Op::Select { keys, table, arms } => {
                self.check_keys(keys, table)?;
                if let Some(arm) = table
                    .iter()
                    .find(|&&arm| arm >= 0 && arm as usize >= arms.len())
                {
                    return Err(format!("a select table names arm {arm} of {}", arms.len()));
                }
                return arms
                    .iter()
                    .flatten()
                    .try_for_each(|op| self.check(op, in_take));
            }
        };
        operands
            .into_iter()
            .filter_map(Operand::register)
            .try_for_each(|register| self.check_register(register))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaled(base: u64, register: Register, scale: u64) -> Operand {
        Operand::Scaled {
            base,
            register,
            scale,
        }
    }

    fn launch(recipe: i32) -> Op {
        Op::Launch(Launch::new(recipe))
    }

    fn recipes(calls: &[Call]) -> Vec<i32> {
        calls
            .iter()
            .filter_map(|call| match call {
                Call::Launch { recipe, .. } => Some(*recipe),
                _ => None,
            })
            .collect()
    }

    fn one_section(registers: usize, build: impl FnOnce(&mut Builder)) -> Program {
        let mut builder = Builder::new(registers);
        builder.section("step").unwrap();
        build(&mut builder);
        builder.finish().unwrap()
    }

    #[test]
    fn operands_read_the_registers() {
        let program = one_section(2, |b| {
            b.push(Op::Copy {
                dst: scaled(1000, 0, 64),
                src: 7.into(),
                bytes: scaled(0, 1, 2),
            })
            .unwrap();
        });
        let calls = program.calls(&[0], &[3, 5]).unwrap();
        assert_eq!(
            calls,
            [Call::Copy {
                dst: 1000 + 3 * 64,
                src: 7,
                bytes: 10
            }]
        );
        let error = program.calls(&[0], &[u64::MAX, 1]).unwrap_err();
        assert!(error.message.contains("overflows"), "{error}");
        assert_eq!(error.issued, 0);
    }

    #[test]
    fn step_bindings_follow_the_fixed_ones_and_the_last_one_wins() {
        let program = one_section(1, |b| {
            b.push(Op::Launch(
                Launch::new(4)
                    .bind(0, 100)
                    .bind(1, 101)
                    .bind_dynamic(1, scaled(2000, 0, 8))
                    .bind_dynamic(2, 9.into()),
            ))
            .unwrap();
        });
        let calls = program.calls(&[0], &[3]).unwrap();
        assert_eq!(
            calls,
            [Call::Launch {
                recipe: 4,
                binds: vec![(0, 100), (1, 101), (1, 2024), (2, 9)]
            }]
        );
    }

    #[test]
    fn a_take_makes_the_first_n_ops() {
        let program = one_section(1, |b| {
            b.push(launch(1)).unwrap();
            b.take(0).unwrap();
            for recipe in 10..13 {
                b.push(launch(recipe)).unwrap();
            }
            b.end().unwrap();
            b.push(launch(2)).unwrap();
        });
        for n in 0..=3 {
            let calls = program.calls(&[0], &[n]).unwrap();
            let expected: Vec<i32> = [1]
                .into_iter()
                .chain(10..10 + n as i32)
                .chain([2])
                .collect();
            assert_eq!(recipes(&calls), expected);
        }
        let error = program.calls(&[0], &[4]).unwrap_err();
        assert_eq!(error.issued, 1, "the first launch was made: {error}");
        assert!(error.message.contains("takes 4 ops of 3"), "{error}");
    }

    #[test]
    fn a_select_makes_the_arm_its_table_names() {
        // keys (r0 below 2, r1 below 3): index = r0 * 3 + r1
        let table = vec![0, 1, 1, 2, -1, 0];
        let program = one_section(2, |b| {
            b.select(vec![(0, 2), (1, 3)], table.clone()).unwrap();
            for recipe in [10, 20, 30] {
                b.arm().unwrap();
                b.push(launch(recipe)).unwrap();
            }
            b.end().unwrap();
        });
        for r0 in 0..2u64 {
            for r1 in 0..3u64 {
                let arm = table[(r0 * 3 + r1) as usize];
                let result = program.calls(&[0], &[r0, r1]);
                if arm < 0 {
                    assert!(result.unwrap_err().message.contains("no arm"));
                } else {
                    assert_eq!(recipes(&result.unwrap()), [10 * (arm + 1)]);
                }
            }
        }
        assert!(
            program
                .calls(&[0], &[2, 0])
                .unwrap_err()
                .message
                .contains("not below 2")
        );
    }

    #[test]
    fn a_collective_inside_a_take_is_refused() {
        let reduce = Op::AllReduce {
            address: 1.into(),
            count: 2.into(),
            bf16: false,
        };
        let mut builder = Builder::new(2);
        builder.section("step").unwrap();
        builder.take(0).unwrap();
        assert!(
            builder
                .push(reduce.clone())
                .unwrap_err()
                .contains("collective")
        );
        // nor in a select inside the take
        builder.select(vec![(1, 1)], vec![0]).unwrap();
        builder.arm().unwrap();
        assert!(
            builder
                .push(reduce.clone())
                .unwrap_err()
                .contains("collective")
        );
        builder.end().unwrap();
        builder.end().unwrap();
        // nor in a take pushed whole
        let take = Op::Take {
            register: 0,
            ops: vec![reduce.clone()],
        };
        assert!(builder.push(take).unwrap_err().contains("collective"));
        // a collective in a select arm outside a take is allowed: every rank has the same key values
        builder.select(vec![(1, 1)], vec![0]).unwrap();
        builder.arm().unwrap();
        builder.push(reduce).unwrap();
        builder.end().unwrap();
        builder.finish().unwrap();
    }

    #[test]
    fn the_builder_checks_what_it_is_given() {
        let mut builder = Builder::new(1);
        assert!(
            builder
                .push(launch(1))
                .unwrap_err()
                .contains("before the first section")
        );
        builder.section("a").unwrap();
        assert!(builder.section("a").unwrap_err().contains("two sections"));
        assert!(builder.take(1).unwrap_err().contains("register 1 of 1"));
        let copy = Op::Copy {
            dst: scaled(0, 3, 1),
            src: 0.into(),
            bytes: 1.into(),
        };
        assert!(builder.push(copy).unwrap_err().contains("register 3"));
        let mut uneven = Launch::new(1).bind(0, 1);
        uneven.slots.push(1);
        assert!(
            builder
                .push(Op::Launch(uneven))
                .unwrap_err()
                .contains("2 slots and 1 addresses")
        );
        assert!(
            builder
                .select(vec![(0, 2)], vec![0])
                .unwrap_err()
                .contains("1 entries for 2")
        );
        assert!(
            builder
                .select(vec![(0, 0)], vec![])
                .unwrap_err()
                .contains("radix of 0")
        );
        assert!(
            builder
                .select(vec![(0, 1)], vec![-2])
                .unwrap_err()
                .contains("entry of -2")
        );
        builder.select(vec![(0, 2)], vec![0, 1]).unwrap();
        assert!(
            builder
                .push(launch(1))
                .unwrap_err()
                .contains("before its first arm")
        );
        builder.arm().unwrap();
        // the table names arm 1, which was never started
        assert!(builder.end().unwrap_err().contains("names arm 1 of 1"));
        assert!(
            builder.arm().is_err(),
            "the select is gone after a failed end"
        );
        assert!(builder.end().unwrap_err().contains("without an open"));
        builder.take(0).unwrap();
        assert!(builder.section("b").unwrap_err().contains("inside an open"));
        assert!(builder.finish().unwrap_err().contains("not ended"));
    }

    #[test]
    fn sections_and_programs_run_in_the_order_given() {
        let mut builder = Builder::new(1);
        let first = builder.section("first").unwrap();
        builder.push(launch(1)).unwrap();
        builder
            .push(Op::AllGather {
                send: 1.into(),
                recv: 2.into(),
                count: 3.into(),
                bf16: true,
            })
            .unwrap();
        let harvest = builder.section("harvest").unwrap();
        builder
            .push(Op::Download {
                host: 5.into(),
                src: 6.into(),
                bytes: scaled(0, 0, 4),
            })
            .unwrap();
        builder.push(Op::Record { slot: 3 }).unwrap();
        let a = builder.finish().unwrap();
        assert_eq!(a.section("harvest"), Some(harvest));
        assert!(!a.collective_free(first));
        assert!(a.collective_free(harvest));

        let b = one_section(0, |b| b.push(launch(9)).unwrap());
        let mut recorder = Recorder::default();
        let made =
            run_sequence(&[(&a, harvest), (&b, 0), (&a, first)], &[2], &mut recorder).unwrap();
        assert_eq!(made, 5);
        assert_eq!(
            recorder.calls,
            [
                Call::Download {
                    host: 5,
                    src: 6,
                    bytes: 8
                },
                Call::Record { slot: 3 },
                Call::Launch {
                    recipe: 9,
                    binds: vec![]
                },
                Call::Launch {
                    recipe: 1,
                    binds: vec![]
                },
                Call::AllGather {
                    send: 1,
                    recv: 2,
                    count: 3,
                    bf16: true
                },
            ]
        );
        let error = a.calls(&[first], &[]).unwrap_err();
        assert!(
            error.message.contains("0 registers for a program of 1"),
            "{error}"
        );
        assert!(
            a.calls(&[2], &[0])
                .unwrap_err()
                .message
                .contains("section 2 of 2")
        );
    }

    #[test]
    fn a_failed_call_reports_the_calls_made_before_it() {
        let program = one_section(0, |b| {
            for recipe in 0..4 {
                b.push(launch(recipe)).unwrap();
            }
        });
        let mut recorder = Recorder {
            fail_at: Some(2),
            ..Recorder::default()
        };
        let error = program.run(&[0], &[], &mut recorder).unwrap_err();
        assert_eq!(error.issued, 2);
        assert_eq!(error.message, "step: call 2 fails");
        assert_eq!(recipes(&recorder.calls), [0, 1]);
    }

    /// The DeepSeek decode index op that this program replaced (`Op::Index` of the DeepSeek-only program),
    /// written as a select over (sparse, cand, radix) whose arms take the index launches, makes the same
    /// launches for every count. `_decode_program` in deepseek_v41/model.py builds the same select.
    #[test]
    fn an_index_op_as_select_and_take_makes_the_ds41_launches() {
        use crate::ds41::Counts;

        /// The replaced op's launches between a launch (1) and an all-gather (-1): by candidates in a sparse
        /// step, else the first index launches and the all or radix selection. Err where it failed.
        fn reference(
            counts: &Counts,
            ratio: usize,
            ix: &[i32],
            ixc: Option<&[i32]>,
            sparse: Option<i32>,
            all: Option<i32>,
            radix: [Option<i32>; 2],
        ) -> Result<Vec<i32>, ()> {
            let mut out = vec![1];
            if let Some(ixc) = ixc.filter(|_| counts.sparse) {
                out.extend(ixc.get(..counts.ixc).ok_or(())?);
                out.push(sparse.ok_or(())?);
            } else {
                let ri = ratio - 1;
                out.extend(ix.get(..counts.ix[ri]).ok_or(())?);
                out.push(match all {
                    Some(all) if !counts.cand => all,
                    _ => radix[usize::from(counts.radix[ri])].ok_or(())?,
                });
            }
            out.push(-1);
            Ok(out)
        }

        // registers: 0 sparse, 1 cand, 2 ixc, 3 radix (ratio 1), 4 radix (ratio 2), 5 ix (ratio 1), 6 ix (ratio 2)
        fn as_select(
            ratio: usize,
            ix: &[i32],
            ixc: Option<&[i32]>,
            sparse: Option<i32>,
            all: Option<i32>,
            radix: [Option<i32>; 2],
        ) -> Program {
            let ri = ratio - 1;
            let mut arms: Vec<(Register, Vec<i32>, i32)> = Vec::new();
            let mut arm_of = |register: Register, launches: &[i32], select: i32| {
                arms.push((register, launches.to_vec(), select));
                arms.len() as i32 - 1
            };
            let sparse_arm = match (ixc, sparse) {
                (Some(ixc), Some(sparse)) => Some(arm_of(2, ixc, sparse)),
                _ => None,
            };
            let all_arm = all.map(|all| arm_of(5 + ri as Register, ix, all));
            let radix_arms =
                radix.map(|radix| radix.map(|radix| arm_of(5 + ri as Register, ix, radix)));
            let mut table = Vec::new();
            for s in 0..2 {
                for c in 0..2 {
                    for radix_arm in radix_arms {
                        let arm = if ixc.is_some() && s == 1 {
                            sparse_arm
                        } else if all.is_some() && c == 0 {
                            all_arm
                        } else {
                            radix_arm
                        };
                        table.push(arm.unwrap_or(-1));
                    }
                }
            }
            let mut builder = Builder::new(7);
            builder.section("layer").unwrap();
            builder.push(launch(1)).unwrap();
            builder
                .select(vec![(0, 2), (1, 2), (3 + ri as Register, 2)], table)
                .unwrap();
            for (register, launches, select) in arms {
                builder.arm().unwrap();
                builder.take(register).unwrap();
                for recipe in launches {
                    builder.push(launch(recipe)).unwrap();
                }
                builder.end().unwrap();
                builder.push(launch(select)).unwrap();
            }
            builder.end().unwrap();
            builder
                .push(Op::AllGather {
                    send: 1.into(),
                    recv: 2.into(),
                    count: 3.into(),
                    bf16: false,
                })
                .unwrap();
            builder.finish().unwrap()
        }

        let ix = [10, 11, 12];
        let ixc = [20, 21];
        let mut compared = 0;
        for ratio in [1, 2] {
            for with_ixc in [false, true] {
                for sparse in [None, Some(30)] {
                    for all in [None, Some(31)] {
                        for radix in [[Some(32), Some(33)], [Some(32), None], [None, Some(33)]] {
                            let select = as_select(
                                ratio,
                                &ix,
                                with_ixc.then_some(&ixc[..]),
                                sparse,
                                all,
                                radix,
                            );
                            for bits in 0..8u8 {
                                for n_ixc in 0..=3 {
                                    for n_ix in 0..=4 {
                                        // the other ratio's registers differ, so a wrong register read shows
                                        let counts = Counts {
                                            sparse: bits & 1 != 0,
                                            cand: bits & 2 != 0,
                                            ixc: n_ixc,
                                            radix: [bits & 4 != 0, bits & 4 == 0],
                                            ix: [n_ix, 4 - n_ix],
                                            ..Counts::default()
                                        };
                                        let registers = [
                                            u64::from(counts.sparse),
                                            u64::from(counts.cand),
                                            counts.ixc as u64,
                                            u64::from(counts.radix[0]),
                                            u64::from(counts.radix[1]),
                                            counts.ix[0] as u64,
                                            counts.ix[1] as u64,
                                        ];
                                        let expected = reference(
                                            &counts,
                                            ratio,
                                            &ix,
                                            with_ixc.then_some(&ixc[..]),
                                            sparse,
                                            all,
                                            radix,
                                        );
                                        let got = select.calls(&[0], &registers).map(|calls| {
                                            calls
                                                .iter()
                                                .map(|call| match call {
                                                    Call::Launch { recipe, .. } => *recipe,
                                                    _ => -1,
                                                })
                                                .collect::<Vec<_>>()
                                        });
                                        match (expected, got) {
                                            (Ok(expected), Ok(got)) => {
                                                assert_eq!(expected, got, "{counts:?}")
                                            }
                                            (Err(_), Err(_)) => {}
                                            (expected, got) => panic!(
                                                "ratio {ratio} ixc {with_ixc} sparse {sparse:?} all {all:?} \
                                                 radix {radix:?} {counts:?}: {expected:?} vs {got:?}"
                                            ),
                                        }
                                        compared += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 2 * 2 * 2 * 2 * 3 * 8 * 4 * 5);
    }
}
