//! A slab table in shared memory: one KV arena layout that a tensor-parallel model's pool (side 0, in rank 0's
//! process) and one-card replicas' pools (side k, the replica on card k, in that card's process) share across one
//! boundary (`resin_core::resources::paged_kv::SlabBoundary`).
//!
//! Every card has the same arena of `capacity` slabs, split at the same boundary slab B: the tensor-parallel
//! model holds slabs 1 to B - 1 on every card (a slab is a column of the cards), each replica slabs B to
//! `capacity` of its card. The boundary moves one slab at a time:
//! - down (the replicas gain slab B - 1) when the tensor-parallel model gives its highest slab;
//! - up (the tensor-parallel model gains slab B) once every replica has given slab B of its card. A replica that
//!   gave it while another replica still holds it takes it back when the tensor-parallel model stops wanting it.
//!
//! The table is a file that every process maps (MAP_SHARED). A lock word guards the moves; the boundary, each
//! side's record (floor, the edge slab it gave, oldest prefix stamp, whether it wants slabs) and the shared clock
//! are atomics.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use resin_core::resources::paged_kv::{OtherSide, SlabBoundary};

const MAGIC: u64 = u64::from_le_bytes(*b"WRSLAB02");
const HEADER: usize = 64;
const SIDE: usize = 32;
const NONE: u64 = u64::MAX;

/// The mapped table.
#[derive(Debug)]
struct Mapping {
    base: *mut u8,
    len: usize,
    capacity: u32,
    cards: u32,
    path: PathBuf,
}

// The mapping is shared memory accessed only through atomics and under the table's lock.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base.cast(), self.len) };
    }
}

fn size(cards: u32) -> usize {
    HEADER + SIDE * (cards as usize + 1)
}

impl Mapping {
    fn map(path: &Path, len: usize, create: bool) -> std::io::Result<(*mut u8, usize)> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(create)
            .open(path)?;
        if create {
            file.set_len(len as u64)?;
        }
        let len = if create {
            len
        } else {
            file.metadata()?.len() as usize
        };
        use std::os::fd::AsRawFd;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok((base.cast(), len))
    }

    fn u32_at(&self, offset: usize) -> &AtomicU32 {
        debug_assert!(offset + 4 <= self.len && offset.is_multiple_of(4));
        unsafe { &*(self.base.add(offset) as *const AtomicU32) }
    }

    fn u64_at(&self, offset: usize) -> &AtomicU64 {
        debug_assert!(offset + 8 <= self.len && offset.is_multiple_of(8));
        unsafe { &*(self.base.add(offset) as *const AtomicU64) }
    }

    fn lock_word(&self) -> &AtomicU32 {
        self.u32_at(16)
    }

    fn clock(&self) -> &AtomicU64 {
        self.u64_at(24)
    }

    fn boundary(&self) -> &AtomicU32 {
        self.u32_at(32)
    }

    fn side(&self, side: usize) -> SideRecord<'_> {
        SideRecord {
            map: self,
            base: HEADER + SIDE * side,
        }
    }

    fn replicas(&self) -> std::ops::RangeInclusive<usize> {
        1..=self.cards as usize
    }

    /// The largest floor of the replicas: they share one region size.
    fn replica_floor(&self) -> u32 {
        self.replicas()
            .map(|card| self.side(card).floor().load(Ordering::Acquire))
            .max()
            .unwrap_or(0)
    }

    /// Slabs below the boundary (the tensor-parallel model's side) and from it (each replica's side).
    fn sizes(&self) -> (u32, u32) {
        let boundary = self.boundary().load(Ordering::Acquire);
        (boundary - 1, self.capacity + 1 - boundary)
    }

    /// Run `body` under the table's lock: a spin lock with short sleeps (every critical section is short) whose
    /// word holds the owner's process id. A process that waits long checks the owner, and takes the lock from a
    /// process that died holding it.
    fn locked<T>(&self, body: impl FnOnce() -> T) -> T {
        let lock = self.lock_word();
        let me = std::process::id();
        let mut spins = 0u32;
        loop {
            match lock.compare_exchange_weak(0, me, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => break,
                Err(owner) => {
                    spins += 1;
                    if spins < 64 {
                        std::hint::spin_loop();
                        continue;
                    }
                    std::thread::sleep(Duration::from_micros(20));
                    // about every 100 ms of waiting
                    if spins.is_multiple_of(5000)
                        && owner != 0
                        && !process_alive(owner)
                        && lock
                            .compare_exchange(owner, me, Ordering::Acquire, Ordering::Relaxed)
                            .is_ok()
                    {
                        log::warn!("slab table: process {owner} died holding the table's lock");
                        break;
                    }
                }
            }
        }
        let result = body();
        lock.store(0, Ordering::Release);
        result
    }
}

/// Whether process `pid` exists (a process of another user counts).
fn process_alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    let signalled = unsafe { libc::kill(pid, 0) } == 0;
    signalled || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

struct SideRecord<'a> {
    map: &'a Mapping,
    base: usize,
}

impl SideRecord<'_> {
    fn floor(&self) -> &AtomicU32 {
        self.map.u32_at(self.base)
    }
    /// The edge slab this side gave and the boundary has not passed yet (0: none).
    fn given(&self) -> &AtomicU32 {
        self.map.u32_at(self.base + 4)
    }
    fn oldest(&self) -> &AtomicU64 {
        self.map.u64_at(self.base + 8)
    }
    fn wants(&self) -> &AtomicU32 {
        self.map.u32_at(self.base + 16)
    }
    /// The side's edge slab is empty, above its floor and headroom (`SlabBoundary::publish`).
    fn spare(&self) -> &AtomicU32 {
        self.map.u32_at(self.base + 20)
    }

    fn oldest_stamp(&self) -> Option<u64> {
        let oldest = self.oldest().load(Ordering::Acquire);
        (oldest != NONE).then_some(oldest)
    }

    fn wanting(&self) -> bool {
        self.wants().load(Ordering::Acquire) != 0
    }
}

/// One side of a slab table: the tensor-parallel model (0) or the replica of card k (k).
#[derive(Debug)]
struct Side {
    map: Arc<Mapping>,
    side: u32,
}

impl SlabBoundary for Side {
    fn region(&self) -> (u32, u32) {
        let map = &*self.map;
        map.locked(|| {
            let boundary = map.boundary().load(Ordering::Acquire);
            let given = map.side(self.side as usize).given().load(Ordering::Acquire);
            if self.side == 0 {
                (1, boundary - 1)
            } else if given == boundary {
                (boundary + 1, map.capacity)
            } else {
                (boundary, map.capacity)
            }
        })
    }

    fn low(&self) -> bool {
        self.side == 0
    }

    fn want(&self, wanting: bool) {
        self.map
            .side(self.side as usize)
            .wants()
            .store(u32::from(wanting), Ordering::Release);
    }

    fn give(&self, slab: u32) -> bool {
        let map = &*self.map;
        map.locked(|| {
            let boundary = map.boundary().load(Ordering::Acquire);
            let (low, high) = map.sizes();
            if self.side == 0 {
                // down: the replicas gain slab B - 1, unless a move up is under way
                let moving_up = map
                    .replicas()
                    .any(|card| map.side(card).given().load(Ordering::Acquire) != 0);
                if slab != boundary - 1
                    || low <= map.side(0).floor().load(Ordering::Acquire)
                    || moving_up
                {
                    return false;
                }
                map.boundary().store(boundary - 1, Ordering::Release);
                return true;
            }
            if slab != boundary || high <= map.replica_floor() {
                return false;
            }
            map.side(self.side as usize)
                .given()
                .store(slab, Ordering::Release);
            // up, once every replica gave slab B
            if map
                .replicas()
                .all(|card| map.side(card).given().load(Ordering::Acquire) == boundary)
            {
                for card in map.replicas() {
                    map.side(card).given().store(0, Ordering::Release);
                }
                map.boundary().store(boundary + 1, Ordering::Release);
            }
            true
        })
    }

    fn gave(&self) -> bool {
        self.side != 0
            && self
                .map
                .side(self.side as usize)
                .given()
                .load(Ordering::Acquire)
                != 0
    }

    fn withdraw(&self) -> bool {
        if self.side == 0 {
            return false;
        }
        let map = &*self.map;
        map.locked(|| {
            let record = map.side(self.side as usize);
            if record.given().load(Ordering::Acquire) == 0 {
                return false;
            }
            record.given().store(0, Ordering::Release);
            true
        })
    }

    fn tick(&self) -> u64 {
        self.map.clock().fetch_add(1, Ordering::AcqRel) + 1
    }

    fn publish(&self, oldest: Option<u64>, spare: bool) {
        let record = self.map.side(self.side as usize);
        record
            .oldest()
            .store(oldest.unwrap_or(NONE), Ordering::Release);
        record.spare().store(u32::from(spare), Ordering::Release);
    }

    fn other(&self) -> OtherSide {
        let map = &*self.map;
        let (low, high) = map.sizes();
        if self.side != 0 {
            let first = map.side(0);
            return OtherSide {
                oldest: first.oldest_stamp(),
                wants: first.wanting(),
                can_give: low > first.floor().load(Ordering::Acquire),
                spare: first.spare().load(Ordering::Acquire) != 0,
            };
        }
        // the replicas together: the oldest prefix of any, wanting while any wants
        OtherSide {
            oldest: map
                .replicas()
                .filter_map(|card| map.side(card).oldest_stamp())
                .min(),
            wants: map.replicas().any(|card| map.side(card).wanting()),
            can_give: high > map.replica_floor(),
            // a move up needs every replica's slab B
            spare: map
                .replicas()
                .all(|card| map.side(card).spare().load(Ordering::Acquire) != 0),
        }
    }

    fn floor(&self) -> u32 {
        if self.side == 0 {
            self.map.side(0).floor().load(Ordering::Acquire)
        } else {
            self.map.replica_floor()
        }
    }

    fn other_floor(&self) -> u32 {
        if self.side == 0 {
            self.map.replica_floor()
        } else {
            self.map.side(0).floor().load(Ordering::Acquire)
        }
    }
}

/// A slab table that pools in several processes share (see the module docs).
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct SlabTable {
    map: Arc<Mapping>,
}

/// One side of a `SlabTable`, for `CachePool(..., boundary=side)`.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct SlabSide {
    pub(crate) boundary: Arc<dyn SlabBoundary>,
    #[pyo3(get)]
    side: u32,
}

#[pymethods]
impl SlabTable {
    /// Make the table at `path` (a new file): `capacity` slabs, `cards` replica cards, and the floor of the
    /// tensor-parallel model and of each replica (slabs each keeps). The boundary starts halfway between the
    /// floors.
    #[staticmethod]
    fn create(
        path: PathBuf,
        capacity: u32,
        cards: u32,
        tp_floor: u32,
        replica_floors: Vec<u32>,
    ) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("a slab table needs slabs"));
        }
        if replica_floors.len() != cards as usize {
            return Err(PyValueError::new_err(
                "give one floor for each replica card",
            ));
        }
        let replica_floor = replica_floors.iter().copied().max().unwrap_or(0);
        if tp_floor + replica_floor > capacity {
            return Err(PyValueError::new_err(format!(
                "the floors do not fit in {capacity} slabs: {tp_floor} for the first model and {replica_floor} \
                 for a replica"
            )));
        }
        let (base, len) = Mapping::map(&path, size(cards), true).map_err(|error| {
            PyOSError::new_err(format!("slab table {}: {error}", path.display()))
        })?;
        let map = Mapping {
            base,
            len,
            capacity,
            cards,
            path,
        };
        unsafe {
            std::ptr::write(map.base as *mut u64, MAGIC);
            std::ptr::write(map.base.add(8) as *mut u32, capacity);
            std::ptr::write(map.base.add(12) as *mut u32, cards);
        }
        let spare = capacity - tp_floor - replica_floor;
        map.boundary()
            .store(1 + tp_floor + spare / 2, Ordering::Release);
        map.side(0).floor().store(tp_floor, Ordering::Release);
        for (card, &floor) in replica_floors.iter().enumerate() {
            map.side(card + 1).floor().store(floor, Ordering::Release);
        }
        for side in 0..=cards as usize {
            map.side(side).oldest().store(NONE, Ordering::Release);
        }
        Ok(Self { map: Arc::new(map) })
    }

    /// Map the table that `create` made at `path`.
    #[staticmethod]
    fn open(path: PathBuf) -> PyResult<Self> {
        let (base, len) = Mapping::map(&path, 0, false).map_err(|error| {
            PyOSError::new_err(format!("slab table {}: {error}", path.display()))
        })?;
        let (magic, capacity, cards) = if len >= HEADER {
            unsafe {
                (
                    std::ptr::read(base as *const u64),
                    std::ptr::read(base.add(8) as *const u32),
                    std::ptr::read(base.add(12) as *const u32),
                )
            }
        } else {
            (0, 0, 0)
        };
        let map = Mapping {
            base,
            len,
            capacity,
            cards,
            path: path.clone(),
        };
        if magic != MAGIC || len != size(cards) {
            return Err(PyValueError::new_err(format!(
                "{} is not a slab table",
                path.display()
            )));
        }
        Ok(Self { map: Arc::new(map) })
    }

    #[getter]
    fn capacity(&self) -> u32 {
        self.map.capacity
    }

    #[getter]
    fn cards(&self) -> u32 {
        self.map.cards
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.map.path.clone()
    }

    /// The first slab of the replicas' side; the tensor-parallel model holds the slabs below it.
    #[getter]
    fn boundary(&self) -> u32 {
        self.map.boundary().load(Ordering::Acquire)
    }

    /// The side of the tensor-parallel model (0) or of the replica on card `side` (1 to `cards`).
    fn side(&self, side: u32) -> PyResult<SlabSide> {
        if side > self.map.cards {
            return Err(PyValueError::new_err(format!(
                "side {side}: the table has {} replica cards",
                self.map.cards
            )));
        }
        Ok(SlabSide {
            boundary: Arc::new(Side {
                map: self.map.clone(),
                side,
            }),
            side,
        })
    }

    /// Each side: (floor, slabs on its side of the boundary, the edge slab it gave or 0, oldest prefix stamp or
    /// None, whether it wants slabs).
    fn sides(&self) -> Vec<(u32, u32, u32, Option<u64>, bool)> {
        let (low, high) = self.map.sizes();
        (0..=self.map.cards as usize)
            .map(|side| {
                let record = self.map.side(side);
                (
                    record.floor().load(Ordering::Acquire),
                    if side == 0 { low } else { high },
                    record.given().load(Ordering::Acquire),
                    record.oldest_stamp(),
                    record.wanting(),
                )
            })
            .collect()
    }
}
