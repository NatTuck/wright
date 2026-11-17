//! The stream functions of the Gaudi runtime library (`libresin_gaudi_rt.so`) that the Python device loaded. The
//! library's process globals hold the stream, the recipes and the communicator, so these functions are resolved
//! in that same loaded copy, never in a second one.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

pub(crate) type BindSlots = unsafe extern "C" fn(c_int, *const c_int, *const u64, c_int) -> c_int;
pub(crate) type LaunchFn = unsafe extern "C" fn(c_int) -> c_int;
pub(crate) type Collective = unsafe extern "C" fn(u64, u64, u64, c_int) -> c_int;
/// (device, host, bytes) or (host, device, bytes); the host side is a pointer
pub(crate) type HostCopy = unsafe extern "C" fn(u64, *const c_void, u64) -> c_int;
pub(crate) type HostRead = unsafe extern "C" fn(*mut c_void, u64, u64) -> c_int;
pub(crate) type DeviceCopy = unsafe extern "C" fn(u64, u64, u64) -> c_int;
pub(crate) type DeviceCopies =
    unsafe extern "C" fn(*const u64, *const u64, *const u64, u64) -> c_int;
pub(crate) type Memset32 = unsafe extern "C" fn(u64, u32, u64) -> c_int;
pub(crate) type EventFn = unsafe extern "C" fn(c_int) -> c_int;
pub(crate) type EventSpin = unsafe extern "C" fn(c_int, i64) -> c_int;
pub(crate) type Status = unsafe extern "C" fn() -> c_int;
pub(crate) type SlotFn = unsafe extern "C" fn(c_int, *const c_char) -> c_int;
pub(crate) type CclId = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type CclInit = unsafe extern "C" fn(*const c_void, c_int, c_int) -> c_int;
pub(crate) type NowNs = unsafe extern "C" fn() -> i64;
pub(crate) type EventElapsed = unsafe extern "C" fn(c_int, c_int) -> i64;
pub(crate) type LastError = unsafe extern "C" fn() -> *const c_char;

#[derive(Clone, Copy)]
pub struct Glue {
    pub(crate) bind_slots: BindSlots,
    pub(crate) launch: LaunchFn,
    pub(crate) allgather: Collective,
    pub(crate) allreduce: Collective,
    /// `rg_h2d_async`: from pinned host memory
    pub(crate) upload: HostCopy,
    /// `rg_h2d_nb`: through the library's ring of pinned slots
    pub(crate) upload_ring: HostCopy,
    /// `rg_d2h_async`: to pinned host memory
    pub(crate) download: HostRead,
    /// `rg_d2d_async`: (dst, src, bytes)
    pub(crate) copy: DeviceCopy,
    /// `rg_d2d_multi`: (src, bytes, dst, n)
    pub(crate) copy_multi: DeviceCopies,
    pub(crate) memset32: Memset32,
    pub(crate) record: EventFn,
    /// `rg_event_wait`: block until the slot's work is done
    pub(crate) event_wait: EventFn,
    /// `rg_event_query`: 1 done, 0 pending
    pub(crate) event_query: EventFn,
    /// `rg_event_spin`: poll the slot for up to some nanoseconds, then block
    pub(crate) event_spin: EventSpin,
    /// `rg_event_stream_wait`: the current context's later work waits on the device for the slot
    pub(crate) stream_wait: EventFn,
    /// `rg_context_use`: make a context the calling thread's current one; returns the one it replaces
    pub(crate) context_use: EventFn,
    pub(crate) context_current: Status,
    /// `rg_sync`: wait for the current context's work
    pub(crate) sync: Status,
    /// `rg_slot`: the slot of a persistent tensor name in a recipe, or -1
    pub(crate) slot: SlotFn,
    pub(crate) ccl_id_size: Status,
    pub(crate) ccl_unique_id: CclId,
    /// `rg_ccl_init`: a communicator for the current context
    pub(crate) ccl_init: CclInit,
    pub(crate) now_ns: NowNs,
    /// `rg_event_elapsed`: device nanoseconds between two timed slots' records, -2 while either is pending
    pub(crate) event_elapsed: EventElapsed,
    pub(crate) last_error: LastError,
}

impl Glue {
    /// Resolve the glue of the already loaded library at `path` (it is never loaded a second time).
    pub fn open(path: &str) -> Result<Self, String> {
        let name = CString::new(path).map_err(|_| "a path with a NUL byte".to_owned())?;
        // SAFETY: RTLD_NOLOAD only returns a handle of a library already loaded, and the handle is never closed,
        // so the resolved functions stay valid for the life of the process. Each symbol is transmuted to the
        // signature that runtime.cpp declares for it (and runtime.py's _SIGNATURES repeats).
        unsafe {
            let handle = libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD);
            if handle.is_null() {
                return Err(format!("{path} is not loaded"));
            }
            let symbol = |name: &CStr| -> Result<*mut c_void, String> {
                let pointer = libc::dlsym(handle, name.as_ptr());
                if pointer.is_null() {
                    Err(format!("{path} has no {}", name.to_string_lossy()))
                } else {
                    Ok(pointer)
                }
            };
            use std::mem::transmute;
            Ok(Self {
                bind_slots: transmute::<*mut c_void, BindSlots>(symbol(c"rg_bind_slots")?),
                launch: transmute::<*mut c_void, LaunchFn>(symbol(c"rg_launch")?),
                allgather: transmute::<*mut c_void, Collective>(symbol(c"rg_ccl_allgather")?),
                allreduce: transmute::<*mut c_void, Collective>(symbol(c"rg_ccl_allreduce")?),
                upload: transmute::<*mut c_void, HostCopy>(symbol(c"rg_h2d_async")?),
                upload_ring: transmute::<*mut c_void, HostCopy>(symbol(c"rg_h2d_nb")?),
                download: transmute::<*mut c_void, HostRead>(symbol(c"rg_d2h_async")?),
                copy: transmute::<*mut c_void, DeviceCopy>(symbol(c"rg_d2d_async")?),
                copy_multi: transmute::<*mut c_void, DeviceCopies>(symbol(c"rg_d2d_multi")?),
                memset32: transmute::<*mut c_void, Memset32>(symbol(c"rg_memset32")?),
                record: transmute::<*mut c_void, EventFn>(symbol(c"rg_event_record")?),
                event_wait: transmute::<*mut c_void, EventFn>(symbol(c"rg_event_wait")?),
                event_query: transmute::<*mut c_void, EventFn>(symbol(c"rg_event_query")?),
                event_spin: transmute::<*mut c_void, EventSpin>(symbol(c"rg_event_spin")?),
                stream_wait: transmute::<*mut c_void, EventFn>(symbol(c"rg_event_stream_wait")?),
                context_use: transmute::<*mut c_void, EventFn>(symbol(c"rg_context_use")?),
                context_current: transmute::<*mut c_void, Status>(symbol(c"rg_context_current")?),
                sync: transmute::<*mut c_void, Status>(symbol(c"rg_sync")?),
                slot: transmute::<*mut c_void, SlotFn>(symbol(c"rg_slot")?),
                ccl_id_size: transmute::<*mut c_void, Status>(symbol(c"rg_ccl_id_size")?),
                ccl_unique_id: transmute::<*mut c_void, CclId>(symbol(c"rg_ccl_unique_id")?),
                ccl_init: transmute::<*mut c_void, CclInit>(symbol(c"rg_ccl_init")?),
                now_ns: transmute::<*mut c_void, NowNs>(symbol(c"rg_now_ns")?),
                event_elapsed: transmute::<*mut c_void, EventElapsed>(symbol(c"rg_event_elapsed")?),
                last_error: transmute::<*mut c_void, LastError>(symbol(c"rg_last_error")?),
            })
        }
    }

    fn checked(&self, rc: c_int, what: &str) -> Result<c_int, String> {
        if rc < 0 {
            Err(self.error(what))
        } else {
            Ok(rc)
        }
    }

    /// Wait for what event `slot` recorded: poll it for up to `spin_ns` nanoseconds, then block.
    pub fn wait(&self, slot: c_int, spin_ns: i64) -> Result<(), String> {
        // SAFETY: a plain call into the loaded runtime; the slot is range-checked there.
        self.checked(unsafe { (self.event_spin)(slot, spin_ns) }, "event wait")
            .map(drop)
    }

    /// Block until the work that event `slot` recorded is done (no polling).
    pub fn block(&self, slot: c_int) -> Result<(), String> {
        // SAFETY: as in `wait`.
        self.checked(unsafe { (self.event_wait)(slot) }, "event wait")
            .map(drop)
    }

    /// The current context's later work waits on the device for what event `slot` recorded.
    /// Record event `slot` on the current context's stream.
    pub fn record(&self, slot: c_int) -> Result<(), String> {
        // SAFETY: a plain call into the loaded runtime; the slot is range-checked there.
        self.checked(unsafe { (self.record)(slot) }, "event record")
            .map(drop)
    }

    /// Device seconds from what timed slot `start` recorded to what timed slot `end` recorded, or None while
    /// either is pending.
    pub fn elapsed(&self, start: c_int, end: c_int) -> Result<Option<f64>, String> {
        // SAFETY: a plain call into the loaded runtime; the slots are range-checked there.
        let ns = unsafe { (self.event_elapsed)(start, end) };
        match ns {
            -2 => Ok(None),
            ns if ns < 0 => Err(self.error("event elapsed")),
            ns => Ok(Some(ns as f64 / 1e9)),
        }
    }

    pub fn stream_wait(&self, slot: c_int) -> Result<(), String> {
        // SAFETY: as in `wait`.
        self.checked(unsafe { (self.stream_wait)(slot) }, "stream wait")
            .map(drop)
    }

    /// Whether the work that event `slot` recorded is done.
    pub fn done(&self, slot: c_int) -> Result<bool, String> {
        // SAFETY: as in `wait`.
        self.checked(unsafe { (self.event_query)(slot) }, "event query")
            .map(|rc| rc == 1)
    }

    /// Make context `id` the calling thread's current context; returns the context it replaces.
    pub fn use_context(&self, id: c_int) -> Result<c_int, String> {
        // SAFETY: the runtime checks the id; the current context is thread-local there.
        self.checked(unsafe { (self.context_use)(id) }, "context")
    }

    /// The calling thread's current context.
    pub fn current_context(&self) -> c_int {
        // SAFETY: reads a thread-local of the runtime.
        unsafe { (self.context_current)() }
    }

    /// Wait for the current context's work.
    pub fn sync(&self) -> Result<(), String> {
        // SAFETY: a plain call into the loaded runtime.
        self.checked(unsafe { (self.sync)() }, "sync").map(drop)
    }

    /// The slot of persistent tensor `name` in recipe `recipe`.
    pub fn slot(&self, recipe: c_int, name: &str) -> Result<c_int, String> {
        let name = CString::new(name).map_err(|_| "a tensor name with a NUL byte".to_owned())?;
        // SAFETY: the runtime reads the NUL-terminated name during the call only.
        self.checked(unsafe { (self.slot)(recipe, name.as_ptr()) }, "slot")
    }

    /// A new HCCL unique id (on the first rank of a group).
    pub fn ccl_unique_id(&self) -> Result<Vec<u8>, String> {
        // SAFETY: the id buffer has the size the runtime reports.
        unsafe {
            let size = self.checked((self.ccl_id_size)(), "ccl id size")? as usize;
            let mut id = vec![0u8; size];
            self.checked(
                (self.ccl_unique_id)(id.as_mut_ptr().cast()),
                "ccl unique id",
            )?;
            Ok(id)
        }
    }

    /// A communicator of `ranks` ranks for the current context, as rank `rank`, from the group's unique id.
    pub fn ccl_init(&self, id: &[u8], ranks: c_int, rank: c_int) -> Result<(), String> {
        // SAFETY: the runtime reads the id's size bytes during the call.
        unsafe {
            let size = self.checked((self.ccl_id_size)(), "ccl id size")? as usize;
            if id.len() != size {
                return Err(format!("an HCCL id has {size} bytes, not {}", id.len()));
            }
            self.checked((self.ccl_init)(id.as_ptr().cast(), ranks, rank), "ccl init")
                .map(drop)
        }
    }

    /// The runtime's monotonic clock, in nanoseconds.
    pub fn now_ns(&self) -> i64 {
        // SAFETY: reads the clock.
        unsafe { (self.now_ns)() }
    }

    /// `what` and the library's last error message.
    pub(crate) fn error(&self, what: &str) -> String {
        // SAFETY: rg_last_error returns the library's NUL-terminated message buffer.
        let message = unsafe { CStr::from_ptr((self.last_error)()) };
        format!("{what}: {}", message.to_string_lossy())
    }
}
