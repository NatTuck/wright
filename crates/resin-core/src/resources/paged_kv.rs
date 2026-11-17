use std::collections::HashSet;
use std::fmt;
use std::num::NonZeroU32;
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use crate::{RequestId, SequenceStep};

use super::kv::{KvGroupLayout, KvGroupPlan, KvLayout, KvPlanError, KvRetention};
use super::paged::{
    CachePool, CachePoolError, CachedPages, FreeOrder, GroupDemand, GroupId, PageAccessClaim,
    PageHandle, RequestPages, RetireRequestFailure, SequenceId, WriterAccessLease,
};
use super::prefix::PrefixCache;
use super::{BatchAccess, RequestResourceRef};

/// A point-in-time inventory of the shared physical cache pool.
///
/// Slabs are fungible across groups; group block counts are not independent
/// capacity partitions. Reclaimable storage excludes live requests and all
/// outstanding device access, including retired overlapped executions.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct CacheMetrics {
    pub capacity_slabs: u32,
    /// The slabs that this pool holds: all of them, or its side of a slab boundary.
    pub held_slabs: u32,
    pub free_slabs: u32,
    pub occupied_slabs: u32,
    pub reclaimable_slabs: u32,
    pub groups: Vec<CacheGroupMetrics>,
    pub prefix: Option<PrefixCacheMetrics>,
}

/// Physical block usage for an ordered runner-declared cache group.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct CacheGroupMetrics {
    pub group: u32,
    /// Full history (`kv`), sliding-window history (`swa`), or `recurrent`.
    pub pool: &'static str,
    pub block_tokens: u32,
    pub blocks_per_slab: u32,
    pub occupied_slabs: u32,
    pub reclaimable_slabs: u32,
    pub occupied_blocks: u64,
    /// Blocks held by at least one prefix segment, including live requests.
    pub cached_blocks: u64,
    /// Blocks freed by retiring every cached prefix, with requests unchanged.
    pub reclaimable_blocks: u64,
    /// Blocks with outstanding device read or write leases.
    pub pinned_blocks: u64,
    /// Unoccupied slots in slabs currently bound to this group.
    pub free_blocks: u64,
    /// Block-rounded token capacity; recurrent checkpoints are not tokens.
    pub occupied_tokens: Option<u64>,
    pub reclaimable_tokens: Option<u64>,
}

/// Prefix LRU counters. Reclaimed storage counts actual physical frees during
/// eviction; evicting an ancestor or a live request's prefix may free nothing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct PrefixCacheMetrics {
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reclaimed_blocks: u64,
    pub reclaimed_slabs: u64,
}

/// A pool of equal-sized cache slabs that one Engine allocates from, or
/// several: one Engine for each model loaded in the process.
///
/// Each Engine joins with its own cache layout (`PagedKvResources::join`). Its
/// groups are added to the pool's group list, so a slab bound to one of them
/// belongs to that Engine until it is free again. The cached prefixes of all
/// Engines form one least-recently-used order: when an allocation finds no
/// free slab, the oldest prefix of any Engine is evicted first. Live requests
/// are never evicted. A floor reserves slabs for an Engine: the other Engines
/// do not take the free slabs that its floor still needs, and do not evict its
/// prefixes while it holds no more than its floor.
#[derive(Clone, Debug)]
pub struct SharedCachePool {
    state: Arc<Mutex<SharedState>>,
}

/// One side of a KV arena that pools in several processes share (resin-py's shared-memory slab table). The arena
/// is split at one boundary slab, the same on every card: the first side (the multi-card model, `low`) holds the
/// slabs below it, the other side (each card's replica) the boundary slab and above. A side gives up its edge slab
/// (the first side its highest, the other side its lowest) when the other side wants slabs, and the boundary moves
/// one slab once every pool across it has given that slab. The sides' prefix entries share one clock, so the
/// older one goes first.
pub trait SlabBoundary: Send + Sync + fmt::Debug {
    /// This side's slabs now, the first and the last (from 1; empty when the first is after the last): its side of
    /// the boundary, less an edge slab that it gave and that the boundary has not passed yet.
    fn region(&self) -> (u32, u32);
    /// Whether this is the first side (the slabs below the boundary; it goes first).
    fn low(&self) -> bool;
    /// Ask the other side for slabs, or stop asking.
    fn want(&self, wanting: bool);
    /// Give up edge slab `slab`, in which this side's pool holds no block. False when the table refuses (the
    /// floor, or a move the other way in progress): the slab stays this side's.
    fn give(&self, slab: u32) -> bool;
    /// Take back an edge slab that this side gave and the boundary has not passed (the other side no longer
    /// wants it, and other pools on this side have not given theirs). Returns whether it did.
    fn withdraw(&self) -> bool {
        false
    }
    /// Whether this side gave an edge slab that the boundary has not passed yet (other pools on this side have not
    /// given theirs): it has no edge slab to give until then.
    fn gave(&self) -> bool {
        false
    }
    /// The next stamp of the clock that every side shares.
    fn tick(&self) -> u64;
    /// Publish this side's oldest prefix stamp, and whether it is `spare`: its edge slab is empty and it holds
    /// more than its floor and headroom, so the other side takes that slab before it evicts its own prefixes.
    fn publish(&self, oldest: Option<u64>, spare: bool);
    /// The other side as it last published.
    fn other(&self) -> OtherSide;
    /// Slabs this side keeps however much the other side wants.
    fn floor(&self) -> u32;
    /// Slabs the other side keeps, which this side never gets.
    fn other_floor(&self) -> u32;
}

/// Clears slabs on this side's card before they change sides of a slab boundary, both ways.
pub trait SlabScrub: Send + Sync + fmt::Debug {
    /// Clear `slabs` and wait until they are clear.
    fn scrub(&self, slabs: &[u32]) -> Result<(), String>;
}

/// Boundary work that a pool with a `SlabScrub` claims under its lock and does without it.
#[derive(Debug, Default)]
struct BoundaryWork {
    /// Slabs that the boundary gave this pool: free once clear.
    gained: Vec<u32>,
    /// This side's edge slab, out of the free list: it goes to the other side once clear.
    edge: Option<u32>,
}

impl BoundaryWork {
    fn slabs(&self) -> Vec<u32> {
        self.gained.iter().copied().chain(self.edge).collect()
    }
}

fn out_of_patience(boundary: &BoundaryState) -> bool {
    boundary
        .asked
        .is_some_and(|since| since.elapsed() > BOUNDARY_PATIENCE)
}

/// A `SharedCachePool` that may be gone.
#[derive(Clone, Debug)]
pub struct WeakCachePool(Weak<Mutex<SharedState>>);

impl WeakCachePool {
    pub fn upgrade(&self) -> Option<SharedCachePool> {
        self.0.upgrade().map(|state| SharedCachePool { state })
    }
}

/// The other side of a slab boundary, as it last published.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OtherSide {
    pub oldest: Option<u64>,
    /// It asks for slabs.
    pub wants: bool,
    /// It holds slabs above its floor, so the boundary can move toward it.
    pub can_give: bool,
    /// Its edge slab is empty (above its floor and headroom): it gives it at once.
    pub spare: bool,
}

/// What `SharedState::reclaim_for` did for an allocation that found no free slab.
enum Reclaim {
    /// A slab may be free now: try again.
    Freed,
    /// The other side of the boundary gives a slab later.
    Pending,
    /// Nothing left to reclaim.
    Exhausted,
}

/// A pool's side of a slab boundary.
#[derive(Debug)]
struct BoundaryState {
    side: Arc<dyn SlabBoundary>,
    /// The slabs the pool holds (first, last): its free list and its bound slabs are these.
    owned: (u32, u32),
    /// The pool asked the other side for slabs and has not gained one since: when it last asked.
    wanting: Option<Instant>,
    /// When the pool first asked, of the asks that `wanting` renews.
    asked: Option<Instant>,
    /// Clears the slabs that cross the boundary, if this side must.
    scrub: Option<Arc<dyn SlabScrub>>,
    /// Slabs that the boundary gave this pool and that are not clear yet: not in the free list.
    unscrubbed: Vec<u32>,
}

/// A want that the pool's allocations do not renew for this long ends (their request finished another way).
const WANT_LIFETIME: Duration = Duration::from_millis(20);

/// How long a side waits for the other side's slabs before it runs on in its own: a live block of the other side
/// can hold the edge slab until its request ends. Then the Engine retracts its own requests, as with a pool of its
/// own, and admits only what its own slabs hold until a slab comes.
pub const BOUNDARY_PATIENCE: Duration = Duration::from_millis(100);

#[derive(Debug)]
pub struct SharedState {
    pool: CachePool,
    /// Orders the prefix entries of every Engine.
    clock: u64,
    tenants: Vec<Tenant>,
    boundary: Option<BoundaryState>,
}

#[derive(Debug)]
struct Tenant {
    groups: Range<u32>,
    floor: u32,
    prefix_cache: Option<PrefixCache<CachedPages>>,
    reclaimed_blocks: u64,
    reclaimed_slabs: u64,
    /// The Engine's last step found no slab, and waits for other Engines.
    waiting: bool,
}

/// What an Engine does when an allocation fails after every eviction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityWait {
    /// No other Engine holds live slabs: the request cannot fit.
    Fail,
    /// Other Engines' live requests hold slabs that they will release.
    Wait,
    /// Every other Engine that holds live slabs also waits: give back slabs.
    Yield,
}

impl SharedCachePool {
    pub fn new(parent_capacity: u32) -> Result<Self, CachePoolError> {
        Ok(Self {
            state: Arc::new(Mutex::new(SharedState {
                pool: CachePool::with_capacity(parent_capacity)?,
                clock: 0,
                tenants: Vec::new(),
                boundary: None,
            })),
        })
    }

    /// A pool of `parent_capacity` slabs that it shares with pools in other processes across `boundary`: it
    /// places blocks only in its side's slabs. With `scrub`, every slab that crosses the boundary from or to this
    /// side is clear first (see `SlabScrub`).
    pub fn with_boundary(
        parent_capacity: u32,
        boundary: Arc<dyn SlabBoundary>,
        scrub: Option<Arc<dyn SlabScrub>>,
    ) -> Result<Self, CachePoolError> {
        let order = if boundary.low() {
            FreeOrder::Low
        } else {
            FreeOrder::High
        };
        let mut pool = CachePool::with_leased_capacity(parent_capacity, order)?;
        let (first, last) = boundary.region();
        let slabs: Vec<u32> = (first..=last).collect();
        pool.add_free_parents(&slabs)?;
        boundary.publish(None, false);
        Ok(Self {
            state: Arc::new(Mutex::new(SharedState {
                pool,
                clock: 0,
                tenants: Vec::new(),
                boundary: Some(BoundaryState {
                    side: boundary,
                    owned: (first, last),
                    wanting: None,
                    asked: None,
                    scrub,
                    unscrubbed: Vec::new(),
                }),
            })),
        })
    }

    /// A handle that does not keep the pool alive (a thread that serves its boundary).
    pub fn downgrade(&self) -> WeakCachePool {
        WeakCachePool(Arc::downgrade(&self.state))
    }

    /// Whether the pool shares its slabs with pools in other processes.
    pub fn has_boundary(&self) -> bool {
        self.lock().boundary.is_some()
    }

    /// Slabs this pool holds: all of them, or its side of the boundary.
    pub fn held_slabs(&self) -> usize {
        self.lock().pool.leased_parent_count()
    }

    /// The work of a pool with a boundary between steps (any thread may call it): take the slabs the boundary
    /// gave it; when the other side wants slabs, give it this side's edge slab, after evicting the prefix entries
    /// in it (a slab with live blocks waits for them); publish this side's oldest prefix. A pool with a
    /// `SlabScrub` clears those slabs first, without its lock.
    pub fn serve_boundary(&self) -> Result<(), CachePoolError> {
        let (scrub, work) = {
            let mut state = self.lock();
            let scrub = state
                .boundary
                .as_ref()
                .and_then(|boundary| boundary.scrub.clone());
            (scrub, state.claim_boundary_work()?)
        };
        let (Some(scrub), slabs) = (scrub, work.slabs()) else {
            return Ok(());
        };
        if slabs.is_empty() {
            return Ok(());
        }
        let result = scrub.scrub(&slabs);
        let mut state = self.lock();
        state.finish_boundary_work(work, result.is_ok())?;
        result.map_err(|message| {
            log::error!("slab boundary: clearing slabs {slabs:?} failed: {message}");
            CachePoolError::ScrubFailed
        })
    }

    fn lock(&self) -> MutexGuard<'_, SharedState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn id(&self) -> u64 {
        self.lock().pool.id()
    }

    pub fn parent_capacity(&self) -> u32 {
        self.lock().pool.parent_capacity()
    }

    pub fn free_parent_count(&self) -> usize {
        self.lock().pool.free_parent_count()
    }

    /// The Engines that joined.
    pub fn engines(&self) -> usize {
        self.lock().tenants.len()
    }

    /// Slabs bound to each Engine's groups, in join order.
    pub fn bound_slabs(&self) -> Vec<u64> {
        let state = self.lock();
        state
            .tenants
            .iter()
            .map(|tenant| state.pool.bound_slab_count(tenant.groups.clone()))
            .collect()
    }
}

impl SharedState {
    fn tick(&mut self) -> u64 {
        if let Some(boundary) = &self.boundary {
            return boundary.side.tick();
        }
        self.clock += 1;
        self.clock
    }

    /// The oldest prefix stamp of any Engine of this pool.
    fn oldest_stamp(&self) -> Option<u64> {
        self.tenants
            .iter()
            .filter_map(|tenant| tenant.prefix_cache.as_ref()?.oldest_stamp())
            .min()
    }

    /// Take the slabs that the boundary gave this pool since the last time. Returns whether it gained any.
    fn sync_boundary(&mut self) -> Result<bool, CachePoolError> {
        let Some(boundary) = self.boundary.as_mut() else {
            return Ok(false);
        };
        let (first, last) = boundary.side.region();
        let (owned_first, owned_last) = boundary.owned;
        let gained: Vec<u32> = if boundary.side.low() {
            (owned_last + 1..=last).collect()
        } else {
            (first..owned_first).collect()
        };
        boundary.owned = if boundary.side.low() {
            (owned_first, last.max(owned_last))
        } else {
            (first.min(owned_first), owned_last)
        };
        if gained.is_empty() {
            return Ok(false);
        }
        boundary.asked = None;
        if boundary.wanting.take().is_some() {
            boundary.side.want(false);
        }
        if boundary.scrub.is_some() {
            // free once clear (serve_boundary)
            boundary.unscrubbed.extend(gained);
            return Ok(false);
        }
        self.pool.add_free_parents(&gained)?;
        Ok(true)
    }

    /// Evict every prefix entry of this pool that has a page in slab `parent`.
    fn evict_slab(&mut self, parent: u32) -> Result<(), CachePoolError> {
        let in_slab = |pages: &CachedPages| {
            pages.groups().iter().any(|group| {
                group
                    .segments()
                    .any(|(_, refs)| refs.iter().any(|page| page.handle().parent() == parent))
            })
        };
        for tenant in 0..self.tenants.len() {
            let Some(cache) = self.tenants[tenant].prefix_cache.as_mut() else {
                continue;
            };
            for snapshot in cache.evict_matching(in_slab) {
                self.pool.retire_cached_prefix(snapshot)?;
            }
        }
        Ok(())
    }

    /// Whether this side's edge slab is empty and the side holds more than its floor and headroom: the other side
    /// takes such a slab before it evicts its own prefixes. The first side keeps a headroom of one more floor (a
    /// request that arrives finds its slabs at once); the other side keeps none (the first side goes first).
    fn spare(&self) -> bool {
        let Some(boundary) = self.boundary.as_ref() else {
            return false;
        };
        let side = &boundary.side;
        let (first, last) = boundary.owned;
        if first > last || side.gave() {
            return false;
        }
        let keep = if side.low() {
            2 * side.floor()
        } else {
            side.floor()
        };
        let edge = if side.low() { last } else { first };
        last - first + 1 > keep && self.pool.parent_is_free(edge)
    }

    /// Take this side's edge slab out of the free list, when the floor allows and the slab is free or holds only
    /// prefix entries (they go).
    fn take_edge(&mut self) -> Result<Option<u32>, CachePoolError> {
        let Some(boundary) = self.boundary.as_ref() else {
            return Ok(None);
        };
        let side = boundary.side.clone();
        let (first, last) = boundary.owned;
        if first > last || last - first < side.floor() || side.gave() {
            return Ok(None);
        }
        let edge = if side.low() { last } else { first };
        if !self.pool.take_free_parent(edge) {
            if self.pool.parent_has_live_blocks(edge) {
                return Ok(None); // live blocks: later (its prefixes stay until then)
            }
            self.evict_slab(edge)?;
            if !self.pool.take_free_parent(edge) {
                return Ok(None); // live blocks: later
            }
        }
        Ok(Some(edge))
    }

    /// Give the other side `edge`, which `take_edge` took; it comes back to the free list when the table refuses.
    /// Returns whether it went.
    fn pass_edge(&mut self, edge: u32) -> Result<bool, CachePoolError> {
        let boundary = self
            .boundary
            .as_mut()
            .expect("a pool that took an edge slab");
        if !boundary.side.give(edge) {
            self.pool.add_free_parents(&[edge])?;
            return Ok(false);
        }
        let (first, last) = boundary.owned;
        boundary.owned = if boundary.side.low() {
            (first, last - 1)
        } else {
            (first + 1, last)
        };
        Ok(true)
    }

    /// The boundary work between steps (see `SharedCachePool::serve_boundary`). Without a `SlabScrub` it does all
    /// of it; with one, it returns the slabs that must be clear first: the gained ones and an edge slab to give.
    fn claim_boundary_work(&mut self) -> Result<BoundaryWork, CachePoolError> {
        let mut work = BoundaryWork::default();
        if self.boundary.is_none() {
            return Ok(work);
        }
        self.sync_boundary()?;
        let boundary = self.boundary.as_mut().expect("checked above");
        if boundary
            .wanting
            .is_some_and(|since| since.elapsed() > WANT_LIFETIME)
        {
            boundary.wanting = None;
            boundary.asked = None;
            boundary.side.want(false);
        }
        let side = boundary.side.clone();
        let other = side.other();
        // the first side goes first: it gives only when it does not want slabs itself
        let gives = other.wants && (!side.low() || boundary.wanting.is_none());
        let scrubs = boundary.scrub.is_some();
        if gives {
            if let Some(edge) = self.take_edge()? {
                if scrubs {
                    work.edge = Some(edge);
                } else {
                    self.pass_edge(edge)?;
                }
            }
        } else if !other.wants && side.withdraw() {
            self.sync_boundary()?;
        }
        side.publish(self.oldest_stamp(), self.spare());
        if let Some(boundary) = self.boundary.as_mut() {
            work.gained = std::mem::take(&mut boundary.unscrubbed);
        }
        Ok(work)
    }

    /// After `claim_boundary_work`'s slabs are clear (`scrubbed`): the gained ones become free, the edge slab goes.
    /// Else they stay out of use: the gained ones for the next try, the edge slab with this side.
    fn finish_boundary_work(
        &mut self,
        work: BoundaryWork,
        scrubbed: bool,
    ) -> Result<(), CachePoolError> {
        if !scrubbed {
            if let Some(boundary) = self.boundary.as_mut() {
                boundary.unscrubbed.extend(work.gained);
            }
            if let Some(edge) = work.edge {
                self.pool.add_free_parents(&[edge])?;
            }
            return Ok(());
        }
        if !work.gained.is_empty() {
            self.pool.add_free_parents(&work.gained)?;
        }
        if let Some(edge) = work.edge {
            self.pass_edge(edge)?;
        }
        Ok(())
    }

    /// An allocation of `tenant` found no free slab: take slabs the boundary gave; else evict the least recently
    /// used prefix, this side's, or (asking the other side for slabs) the other side's when that one is older.
    fn reclaim_for(&mut self, tenant: usize) -> Result<Reclaim, CachePoolError> {
        if self.boundary.is_none() {
            return Ok(if self.evict_for(tenant)? {
                Reclaim::Freed
            } else {
                Reclaim::Exhausted
            });
        }
        if self.sync_boundary()? {
            return Ok(Reclaim::Freed);
        }
        if self
            .boundary
            .as_ref()
            .is_some_and(|boundary| !boundary.unscrubbed.is_empty())
        {
            return Ok(Reclaim::Pending); // gained slabs, free once clear
        }
        let side = self.boundary.as_ref().expect("checked above").side.clone();
        let other = side.other();
        let own = self.oldest_stamp();
        let theirs_older = match (other.oldest, own) {
            (Some(theirs), Some(mine)) => theirs < mine,
            (Some(_), None) => true,
            (None, _) => false,
        };
        // The other side's empty slab goes first, then its older prefix, while it can give and has not kept this side
        // waiting too long.
        let waited = self
            .boundary
            .as_ref()
            .and_then(|boundary| boundary.asked)
            .is_some_and(|since| since.elapsed() > WANT_LIFETIME);
        let defer = (other.spare || theirs_older) && other.can_give && !waited;
        if !defer && self.evict_for(tenant)? {
            return Ok(Reclaim::Freed);
        }
        if other.can_give {
            let boundary = self.boundary.as_mut().expect("checked above");
            let now = Instant::now();
            boundary.wanting = Some(now);
            boundary.asked.get_or_insert(now);
            side.want(true);
            return Ok(Reclaim::Pending);
        }
        Ok(Reclaim::Exhausted)
    }

    /// Free slabs that `tenant` must leave for the floors of the other Engines.
    fn reserve_for(&self, tenant: usize) -> usize {
        self.tenants
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != tenant)
            .map(|(_, other)| {
                u64::from(other.floor)
                    .saturating_sub(self.pool.bound_slab_count(other.groups.clone()))
                    as usize
            })
            .sum()
    }

    /// Evict the least recently used prefix of any Engine, except Engines other
    /// than `tenant` that hold no more than their floor. Returns false when
    /// none is left.
    fn evict_for(&mut self, tenant: usize) -> Result<bool, CachePoolError> {
        let oldest = self
            .tenants
            .iter()
            .enumerate()
            .filter(|&(index, other)| {
                index == tenant
                    || self.pool.bound_slab_count(other.groups.clone()) > u64::from(other.floor)
            })
            .filter_map(|(index, other)| {
                Some((other.prefix_cache.as_ref()?.oldest_stamp()?, index))
            })
            .min();
        match oldest {
            Some((_, index)) => self.evict_from(index),
            None => Ok(false),
        }
    }

    /// Evict the least recently used prefix of `tenant`.
    fn evict_from(&mut self, tenant: usize) -> Result<bool, CachePoolError> {
        let Some(snapshot) = self.tenants[tenant]
            .prefix_cache
            .as_mut()
            .and_then(PrefixCache::evict_lru)
        else {
            return Ok(false);
        };
        let occupied = self.pool.occupied_block_count();
        let free = self.pool.free_parent_count();
        self.pool.retire_cached_prefix(snapshot)?;
        let owner = &mut self.tenants[tenant];
        owner.reclaimed_blocks = owner
            .reclaimed_blocks
            .saturating_add((occupied - self.pool.occupied_block_count()) as u64);
        owner.reclaimed_slabs = owner
            .reclaimed_slabs
            .saturating_add((self.pool.free_parent_count() - free) as u64);
        Ok(true)
    }

    /// Slabs of each other Engine that hold live or leased blocks (a scan of
    /// the pool's metadata).
    fn live_slabs_of_others(&self, tenant: usize) -> Vec<(usize, u64)> {
        let occupancy = self.pool.group_occupancy();
        self.tenants
            .iter()
            .enumerate()
            .filter(|&(index, _)| index != tenant)
            .map(|(index, other)| {
                let live = other
                    .groups
                    .clone()
                    .map(|group| {
                        let group = &occupancy[group as usize];
                        u64::from(group.occupied_slabs - group.reclaimable_slabs)
                    })
                    .sum();
                (index, live)
            })
            .collect()
    }
}

fn prefix_metrics_of(tenant: &Tenant) -> Option<PrefixCacheMetrics> {
    tenant
        .prefix_cache
        .as_ref()
        .map(|cache| PrefixCacheMetrics {
            entries: cache.len() as u64,
            hits: cache.hits(),
            misses: cache.misses(),
            evictions: cache.evictions(),
            reclaimed_blocks: tenant.reclaimed_blocks,
            reclaimed_slabs: tenant.reclaimed_slabs,
        })
}

/// The pool for reads, under the shared lock (`PagedKvResources::pool`). Drop
/// it before the next call into the same resources.
pub struct PoolGuard<'a>(MutexGuard<'a, SharedState>);

impl Deref for PoolGuard<'_> {
    type Target = CachePool;

    fn deref(&self) -> &CachePool {
        &self.0.pool
    }
}

#[cfg(test)]
impl std::ops::DerefMut for PoolGuard<'_> {
    fn deref_mut(&mut self) -> &mut CachePool {
        &mut self.0.pool
    }
}

/// One Engine's view of a cache pool: its layout and its prefix cache.
#[derive(Debug)]
pub struct PagedKvResources {
    layout: KvLayout,
    shared: SharedCachePool,
    tenant: usize,
    group_offset: u32,
    prefix_alignment: Option<u32>,
    batch_active: bool,
}

impl PagedKvResources {
    /// A pool of its own for one Engine.
    pub fn new(layout: KvLayout) -> Result<Self, CachePoolError> {
        let shared = SharedCachePool::new(layout.parent_capacity().get())?;
        Self::join(&shared, layout, false, 0)
    }

    pub fn with_prefix_cache(layout: KvLayout) -> Result<Self, CachePoolError> {
        let shared = SharedCachePool::new(layout.parent_capacity().get())?;
        Self::join(&shared, layout, true, 0)
    }

    /// Join `shared` with `layout`, whose capacity must be the pool's.
    /// `floor`: slabs reserved for this Engine (0: none).
    pub fn join(
        shared: &SharedCachePool,
        layout: KvLayout,
        prefix_cache: bool,
        floor: u32,
    ) -> Result<Self, CachePoolError> {
        let prefix_alignment = if prefix_cache {
            Some(prefix_alignment(&layout)?)
        } else {
            None
        };
        let mut state = shared.lock();
        let capacity = state.pool.parent_capacity();
        if layout.parent_capacity().get() != capacity {
            return Err(CachePoolError::PoolCapacityMismatch {
                pool: capacity,
                layout: layout.parent_capacity().get(),
            });
        }
        let packings = layout.packings().collect::<Vec<_>>();
        let group_offset = state.pool.register_groups(&packings)?;
        state.tenants.push(Tenant {
            groups: group_offset..group_offset + packings.len() as u32,
            floor,
            prefix_cache: prefix_alignment.map(PrefixCache::new),
            reclaimed_blocks: 0,
            reclaimed_slabs: 0,
            waiting: false,
        });
        let tenant = state.tenants.len() - 1;
        drop(state);
        Ok(Self {
            layout,
            shared: shared.clone(),
            tenant,
            group_offset,
            prefix_alignment,
            batch_active: false,
        })
    }

    /// The shared state, with this Engine's floor reserve applied to planning.
    fn lock(&self) -> MutexGuard<'_, SharedState> {
        let mut state = self.shared.lock();
        let reserve = state.reserve_for(self.tenant);
        state.pool.set_free_reserve(reserve);
        state
    }

    fn group(&self, index: usize) -> GroupId {
        GroupId::new(self.group_offset + index as u32)
    }

    pub fn shared_pool(&self) -> &SharedCachePool {
        &self.shared
    }

    /// Whether other Engines share this pool.
    pub fn shares_pool(&self) -> bool {
        self.shared.engines() > 1
    }

    pub fn pool(&self) -> PoolGuard<'_> {
        PoolGuard(self.shared.lock())
    }

    pub fn parent_capacity(&self) -> u32 {
        self.layout.parent_capacity().get()
    }

    /// Slabs this Engine may plan for: the pool's capacity, less the live
    /// slabs or the floor (whichever is more) of every other Engine.
    pub fn admission_capacity(&self) -> u64 {
        let state = self.shared.lock();
        let mut capacity = u64::from(state.pool.parent_capacity());
        if let Some(boundary) = &state.boundary {
            if out_of_patience(boundary) {
                // the other side does not give now: this side's own slabs
                return state.pool.leased_parent_count() as u64;
            }
            // the other side of the boundary keeps its floor
            capacity = capacity.saturating_sub(u64::from(boundary.side.other_floor()));
        }
        if state.tenants.len() == 1 {
            return capacity;
        }
        let reserved: u64 = state
            .live_slabs_of_others(self.tenant)
            .into_iter()
            .map(|(index, live)| live.max(u64::from(state.tenants[index].floor)))
            .sum();
        capacity.saturating_sub(reserved)
    }

    /// Whether this side has asked the other side for slabs for longer than `BOUNDARY_PATIENCE`.
    pub fn out_of_patience(&self) -> bool {
        self.shared
            .lock()
            .boundary
            .as_ref()
            .is_some_and(out_of_patience)
    }

    /// The slab this side gives next across its slab boundary, when it has one and its floor lets the slab go.
    pub fn edge_slab(&self) -> Option<u32> {
        let state = self.shared.lock();
        let boundary = state.boundary.as_ref()?;
        let (first, last) = boundary.owned;
        (first <= last && last - first >= boundary.side.floor() && !boundary.side.gave())
            .then(|| if boundary.side.low() { last } else { first })
    }

    /// Whether the pool shares its slabs across a slab boundary.
    pub fn has_boundary(&self) -> bool {
        self.shared.has_boundary()
    }

    /// Serve the boundary now (after this Engine gave way: the slab of the request it retracted goes to the side
    /// that waits, before this Engine's next step can take it again).
    pub fn serve_boundary(&self) -> Result<(), CachePoolError> {
        self.shared.serve_boundary()
    }

    /// After an allocation failed with every prefix evicted: whether to fail,
    /// to wait for other Engines' live requests, or to give back slabs because
    /// those Engines wait too. Records whether this Engine waits.
    pub fn capacity_wait(&self) -> CapacityWait {
        let mut state = self.shared.lock();
        let holders = state
            .live_slabs_of_others(self.tenant)
            .into_iter()
            .filter(|&(_, live)| live > 0)
            .map(|(index, _)| state.tenants[index].waiting)
            .collect::<Vec<_>>();
        let mut holders = holders;
        let mut first = false;
        if let Some(boundary) = &state.boundary {
            // the other side of the boundary holds slabs above its floor
            let other = boundary.side.other();
            if other.can_give {
                holders.push(other.wants);
            }
            first = boundary.side.low();
        }
        let decision = if holders.is_empty() {
            CapacityWait::Fail
        } else if holders.iter().all(|&waiting| waiting) && !first {
            // Sides in two processes that wait for each other: one gives way, always the same one (the replicas;
            // the first model of the layout goes first), or both would give back and take again.
            CapacityWait::Yield
        } else {
            CapacityWait::Wait
        };
        state.tenants[self.tenant].waiting = decision == CapacityWait::Wait;
        decision
    }

    /// Collect a bounded-cadence snapshot, scanning physical block metadata.
    /// No GPU synchronization or token-history traversal is needed. Slab counts
    /// are the pool's; groups and prefixes are this Engine's.
    pub fn metrics(&self) -> CacheMetrics {
        let state = self.shared.lock();
        let occupancy = state.pool.group_occupancy();
        let own = &occupancy[self.group_offset as usize..][..self.layout.groups().len()];
        let groups = self
            .layout
            .groups()
            .iter()
            .zip(own)
            .enumerate()
            .map(|(group, (layout, occupancy))| {
                let pool = if layout.is_recurrent() {
                    "recurrent"
                } else if layout.retention().window_tokens().is_some() {
                    "swa"
                } else {
                    "kv"
                };
                let block_tokens = layout.block_tokens().get();
                CacheGroupMetrics {
                    group: group as u32,
                    pool,
                    block_tokens,
                    blocks_per_slab: layout.blocks_per_slab().get(),
                    occupied_slabs: occupancy.occupied_slabs,
                    reclaimable_slabs: occupancy.reclaimable_slabs,
                    occupied_blocks: occupancy.occupied_blocks,
                    cached_blocks: occupancy.cached_blocks,
                    reclaimable_blocks: occupancy.reclaimable_blocks,
                    pinned_blocks: occupancy.pinned_blocks,
                    free_blocks: occupancy.free_blocks,
                    occupied_tokens: (!layout.is_recurrent())
                        .then_some(occupancy.occupied_blocks * u64::from(block_tokens)),
                    reclaimable_tokens: (!layout.is_recurrent())
                        .then_some(occupancy.reclaimable_blocks * u64::from(block_tokens)),
                }
            })
            .collect::<Vec<_>>();
        let free_slabs = state.pool.free_parent_count() as u32;
        let held_slabs = state.pool.leased_parent_count() as u32;
        CacheMetrics {
            capacity_slabs: self.parent_capacity(),
            held_slabs,
            free_slabs,
            occupied_slabs: held_slabs - free_slabs,
            reclaimable_slabs: groups.iter().map(|group| group.reclaimable_slabs).sum(),
            groups,
            prefix: prefix_metrics_of(&state.tenants[self.tenant]),
        }
    }

    /// A batch must preserve the earliest query's read window while writing
    /// every scheduled token (including speculative lookahead).
    pub fn max_write_tokens(&self) -> Option<NonZeroU32> {
        self.layout
            .groups()
            .iter()
            .copied()
            .filter_map(KvGroupLayout::max_write_tokens)
            .min()
    }

    fn validate_write_span(&self, token_start: u32, context: u32) -> Result<(), CachePoolError> {
        let requested = context
            .checked_sub(token_start)
            .ok_or(CachePoolError::DemandOverflow)?;
        if let Some(maximum) = self.max_write_tokens()
            && requested > maximum.get()
        {
            return Err(CachePoolError::InvalidWriteSpan {
                requested,
                maximum: maximum.get(),
            });
        }
        Ok(())
    }

    pub fn attach(
        &mut self,
        request_id: RequestId,
        maximum_context: u32,
    ) -> Result<RequestPages, CachePoolError> {
        let demands = self.demands(maximum_context);
        let mut shared = self.lock();
        let pending = shared
            .pool
            .reserve_request(SequenceId::new(request_id.get()), &demands)?;
        Ok(shared.pool.promote_request(pending))
    }

    /// Grow one request just enough to address `context` tokens.
    pub fn ensure_context(
        &mut self,
        state: &mut RequestPages,
        token_start: u32,
        context: u32,
    ) -> Result<(), CachePoolError> {
        self.validate_write_span(token_start, context)?;
        // Never below what the request holds: a step that leased lookahead past a block end keeps that page when a
        // later step of the same request leases less (fewer tokens of lookahead).
        let demands: Vec<GroupDemand> = self
            .demands(context)
            .into_iter()
            .zip(state.groups())
            .map(|(demand, held)| demand.at_least(held.pages().len()))
            .collect();
        let mut shared = self.lock();
        loop {
            match shared.pool.grow_request(state, &demands) {
                Ok(()) => break,
                Err(error @ CachePoolError::InsufficientCapacity { .. }) => {
                    match shared.reclaim_for(self.tenant)? {
                        Reclaim::Freed => {}
                        Reclaim::Pending => return Err(CachePoolError::LeasePending),
                        Reclaim::Exhausted => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            }
        }

        let mut replacements = vec![Vec::new(); self.layout.groups().len()];
        for (group_index, group) in self.layout.groups().iter().copied().enumerate() {
            let table = &state.groups()[group_index];
            let first = token_start / group.block_tokens().get();
            let count = context
                .div_ceil(group.block_tokens().get())
                .saturating_sub(first);
            // A cyclic table can be visited at most once. In particular, a
            // long prefill must not scan every token to discover COW pages.
            let count = if group.is_recurrent() || group.max_write_tokens().is_some() {
                count.min(table.pages().len() as u32)
            } else {
                count
            };
            for logical in first..first + count {
                let index = page_index(group, table.pages().len(), logical).ok_or(
                    CachePoolError::InsufficientCapacity {
                        group: self.group(group_index),
                        requested: logical as usize + 1,
                        available: table.pages().len(),
                    },
                )?;
                let page = &table.pages()[index];
                if !page.is_writable() {
                    if group.retention() == KvRetention::FullContext && !group.is_recurrent() {
                        return Err(CachePoolError::PageNotWritable);
                    }
                    replacements[group_index].push(index);
                }
            }
            replacements[group_index].sort_unstable();
            replacements[group_index].dedup();
        }
        // grow_request already authenticated the whole idle owner above. An
        // unchanged ring needs no second full-history validation or mutation.
        if replacements.iter().all(Vec::is_empty) {
            return Ok(());
        }
        loop {
            match shared.pool.replace_shared_pages(state, &replacements) {
                Ok(()) => return Ok(()),
                Err(error @ CachePoolError::InsufficientCapacity { .. }) => {
                    match shared.reclaim_for(self.tenant)? {
                        Reclaim::Freed => {}
                        Reclaim::Pending => return Err(CachePoolError::LeasePending),
                        Reclaim::Exhausted => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Attach the longest aligned cached prefix and return its resident token
    /// count. A miss leaves the empty request untouched.
    pub fn restore_prefix(
        &mut self,
        state: &mut RequestPages,
        tokens: &[u32],
    ) -> Result<u32, CachePoolError> {
        let maximum = tokens
            .len()
            .saturating_sub(self.layout.replay_suffix_tokens() as usize);
        let mut guard = self.lock();
        let shared = &mut *guard;
        let stamp = shared.tick();
        let (pool, cache) = (
            &mut shared.pool,
            &mut shared.tenants[self.tenant].prefix_cache,
        );
        let Some((length, snapshot)) = cache
            .as_mut()
            .and_then(|cache| cache.find(tokens, maximum, stamp))
        else {
            return Ok(0);
        };
        pool.attach_shared_prefix(state, snapshot)?;
        Ok(length)
    }

    /// Publish a completed aligned resident boundary. Returns false when the
    /// current progress cannot form a safe block-complete snapshot.
    pub fn publish_prefix(
        &mut self,
        state: &mut RequestPages,
        tokens: &[u32],
        resident_tokens: u32,
    ) -> Result<bool, CachePoolError> {
        let Some(alignment) = self.prefix_alignment else {
            return Ok(false);
        };
        let length = resident_tokens as usize;
        if length == 0 || length > tokens.len() || !resident_tokens.is_multiple_of(alignment) {
            return Ok(false);
        }
        let demands = self.demands(resident_tokens);
        let mut shared = self.lock();
        let pages = shared.pool.share_request_prefix(state, &demands)?;
        let stamp = shared.tick();
        let replaced = shared.tenants[self.tenant]
            .prefix_cache
            .as_mut()
            .expect("cache enablement was checked")
            .insert(&tokens[..length], pages, stamp);
        if let Some(replaced) = replaced {
            shared.pool.retire_cached_prefix(replaced)?;
        }
        Ok(true)
    }

    /// Evict the least recently used cached prefix of any Engine of the pool
    /// (see `SharedCachePool`). Returns false when none is left.
    pub fn evict_one_prefix(&mut self) -> Result<bool, CachePoolError> {
        self.lock().evict_for(self.tenant)
    }

    /// Evict every cached prefix of this Engine (and none of the others').
    pub fn clear_prefix_cache(&mut self) -> Result<(), CachePoolError> {
        let mut shared = self.lock();
        while shared.evict_from(self.tenant)? {}
        Ok(())
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.prefix_alignment.is_some()
    }

    pub fn prefix_metrics(&self) -> Option<PrefixCacheMetrics> {
        prefix_metrics_of(&self.shared.lock().tenants[self.tenant])
    }

    pub fn prefix_alignment(&self) -> Option<u32> {
        self.prefix_alignment
    }

    /// The final prefill tokens that `restore_prefix` always leaves to compute.
    pub fn replay_suffix_tokens(&self) -> u32 {
        self.layout.replay_suffix_tokens()
    }

    /// Drop resident KV while retaining the request's allocator identity.
    pub fn reset(&mut self, state: &mut RequestPages) -> Result<(), CachePoolError> {
        self.lock().pool.clear_request(state)
    }

    /// Physical-parent budget for a set of estimated request contexts.
    pub fn estimated_parent_count(&self, contexts: &[u32]) -> Result<u64, CachePoolError> {
        self.layout
            .estimated_parent_count(contexts)
            .map_err(|_| CachePoolError::DemandOverflow)
    }

    /// Accumulate admission demand, crediting only pages physically shared by
    /// the selected requests. Merely equal token prefixes receive no credit.
    pub(crate) fn admission_estimate(&self) -> AdmissionEstimate<'_> {
        AdmissionEstimate {
            layout: &self.layout,
            blocks: vec![0; self.layout.groups().len()],
            occupied_parents: vec![HashSet::new(); self.layout.groups().len()],
            shared: HashSet::new(),
        }
    }

    pub fn retire(&mut self, state: RequestPages) -> Result<(), RetireRequestFailure> {
        self.lock().pool.retire_request(state)
    }

    fn demands(&self, context: u32) -> Vec<GroupDemand> {
        self.layout
            .groups()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, group)| {
                let (_, blocks) = group.active_blocks(context);
                GroupDemand::new(self.group(index), blocks as usize)
            })
            .collect()
    }

    pub fn release(&mut self, lease: WriterAccessLease) {
        assert!(self.batch_active, "released a nonexistent KV batch");
        self.lock().pool.release_writer_access(lease);
        self.batch_active = false;
    }

    fn invalid(message: impl Into<String>) -> PagedKvError {
        PagedKvError::Invalid(message.into())
    }

    pub fn acquire(
        &mut self,
        requests: &[RequestResourceRef<'_, RequestPages>],
        steps: &[SequenceStep],
        decode_write_lookahead: u32,
    ) -> Result<BatchAccess<PagedKvBatchView, WriterAccessLease>, PagedKvError> {
        if self.batch_active {
            return Err(Self::invalid(
                "cannot acquire a KV batch before the current batch is released",
            ));
        }
        if requests.is_empty() || requests.len() != steps.len() {
            return Err(Self::invalid(
                "KV requests and steps must have equal nonzero lengths",
            ));
        }
        for step in steps {
            let post_length = step
                .token_end()
                .ok_or_else(|| Self::invalid("scheduled token range overflows u32"))?;
            let leased_length = post_length
                .checked_add(if step.phase == crate::Phase::Decode {
                    decode_write_lookahead
                } else {
                    0
                })
                .ok_or_else(|| Self::invalid("decode cache write lookahead overflows u32"))?;
            self.validate_write_span(step.token_start(), leased_length)?;
        }
        let mut shared = self.lock();
        for request in requests {
            shared.pool.validate_request(request.state)?;
            if request.state.sequence().get() != request.request_id.get() {
                return Err(Self::invalid(format!(
                    "request {} cache pages belong to sequence {}",
                    request.request_id.get(),
                    request.state.sequence().get()
                )));
            }
            if request.state.groups().len() != self.layout.groups().len() {
                return Err(Self::invalid(format!(
                    "request {} has the wrong cache group count",
                    request.request_id.get()
                )));
            }
            for (index, table) in request.state.groups().iter().enumerate() {
                if table.group() != self.group(index) {
                    return Err(Self::invalid(format!(
                        "request {} cache groups are out of order",
                        request.request_id.get()
                    )));
                }
            }
        }

        let append_count = steps
            .iter()
            .map(|step| step.token_count().get() as usize)
            .sum();
        let mut groups = Vec::with_capacity(self.layout.groups().len());
        let mut claims = Vec::<PageAccessClaim>::new();
        let mut shared_reads = HashSet::new();
        for (group_index, group) in self.layout.groups().iter().copied().enumerate() {
            let mut indptr = Vec::with_capacity(requests.len() + 1);
            let mut block_slab_ids = Vec::new();
            let mut block_slotnums = Vec::new();
            let mut first_blocks = Vec::with_capacity(requests.len());
            let mut append_slab_ids = Vec::with_capacity(append_count);
            let mut append_slotnums = Vec::with_capacity(append_count);
            let mut append_offsets = Vec::with_capacity(append_count);
            indptr.push(0);

            for (request, step) in requests.iter().zip(steps) {
                let table = &request.state.groups()[group_index];
                let post_length = step
                    .token_end()
                    .ok_or_else(|| Self::invalid("scheduled token range overflows u32"))?;
                let leased_length = if step.phase == crate::Phase::Decode {
                    post_length
                        .checked_add(decode_write_lookahead)
                        .ok_or_else(|| {
                            Self::invalid("decode cache write lookahead overflows u32")
                        })?
                } else {
                    post_length
                };
                let (first, count) = group.active_blocks(leased_length);
                if count as usize > table.pages().len() {
                    return Err(Self::invalid(format!(
                        "request {} cache group {group_index} needs {count} blocks but reserved {}",
                        request.request_id.get(),
                        table.pages().len()
                    )));
                }
                first_blocks.push(first);
                for logical in first..first + count {
                    let page = page_for(group, table.pages(), logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "request {} cache group {group_index} is missing logical block {logical}",
                            request.request_id.get()
                        ))
                    })?;
                    let (slab_id, slotnum) = shared.pool.physical_address(page.handle())?;
                    block_slab_ids.push(slab_id);
                    block_slotnums.push(slotnum);
                    // Every request was authenticated above. A shared prefix
                    // needs one read pin for the batch, even when hundreds of
                    // rows expose that same history in their device tables.
                    // Keep every write claim so conflicting writers still fail
                    // prepare_writer_access's transactional duplicate check.
                    if page.is_writable() || shared_reads.insert((slab_id, slotnum)) {
                        claims.push(request.state.access_claim(page));
                    }
                }
                indptr.push(
                    u32::try_from(block_slab_ids.len()).map_err(|_| {
                        Self::invalid("cache block table exceeds the u32 CSR range")
                    })?,
                );

                for position in step.token_start()..post_length {
                    let logical = position / group.block_tokens().get();
                    let page = page_for(group, table.pages(), logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "request {} cache group {group_index} is missing logical block {logical}",
                            request.request_id.get()
                        ))
                    })?;
                    let (slab_id, slotnum) = shared.pool.physical_address(page.handle())?;
                    append_slab_ids.push(slab_id);
                    append_slotnums.push(slotnum);
                    append_offsets.push(position % group.block_tokens().get());
                }
            }
            groups.push(KvGroupPlan::new(
                group_index as u32,
                indptr,
                block_slab_ids,
                block_slotnums,
                first_blocks,
                append_slab_ids,
                append_slotnums,
                append_offsets,
                group.blocks_per_slab(),
            )?);
        }

        let prepared = shared.pool.prepare_writer_access(claims)?;
        let lease = shared.pool.acquire_writer_access(prepared)?;
        shared.tenants[self.tenant].waiting = false;
        let pool_id = shared.pool.id();
        drop(shared);
        self.batch_active = true;
        Ok(BatchAccess {
            view: PagedKvBatchView { pool_id, groups },
            lease,
        })
    }

    /// Make a decode continuation's block tables cover its next step: `next_leased[r]` tokens of row r (the next
    /// step's conservative end plus its lookahead). Full-context groups grow, window rings rotate onto the page of
    /// their oldest block (replacing it when shared: that block is outside every decode read), and spare pages the
    /// request already holds are write-claimed, all inside `lease` while its batch runs. Then the block tables of
    /// `groups` (not their appends) are rebuilt. Returns false with `groups` unchanged when a recurrent group would
    /// move, a full-context page ahead is shared, or capacity runs out after evicting cached prefixes; rows grown
    /// before that keep their pages as spare capacity.
    pub fn extend_continuation(
        &mut self,
        requests: &mut [&mut RequestPages],
        lease: &mut WriterAccessLease,
        groups: &mut [KvGroupPlan],
        next_leased: &[u32],
    ) -> Result<bool, PagedKvError> {
        let layout = self.layout.clone();
        if groups.len() != layout.groups().len()
            || requests.len() != next_leased.len()
            || groups
                .iter()
                .any(|plan| plan.block_table_start_indices.len() != requests.len())
        {
            return Err(Self::invalid(
                "a continuation needs one table per group and one row per request",
            ));
        }
        let mut changed = vec![false; groups.len()];
        let mut shared = self.lock();
        for (row, (request, &leased)) in requests.iter_mut().zip(next_leased).enumerate() {
            let mut demands = Vec::with_capacity(groups.len());
            let mut claims = vec![Vec::new(); groups.len()];
            let mut replacements = vec![Vec::new(); groups.len()];
            let mut needed = false;
            for (index, (group, plan)) in layout.groups().iter().copied().zip(&*groups).enumerate()
            {
                let held = request.groups()[index].pages();
                let current_first = plan.block_table_start_indices[row];
                let current_count = plan.block_indptr[row + 1] - plan.block_indptr[row];
                let (first, count) = group.active_blocks(leased);
                demands.push(GroupDemand::new(self.group(index), held.len()));
                if (first, count) == (current_first, current_count) {
                    continue;
                }
                if group.is_recurrent() || first + count < current_first + current_count {
                    return Ok(false);
                }
                changed[index] = true;
                let window = matches!(group.retention(), KvRetention::SlidingWindow { .. });
                // Blocks past the current table. A ring that is not yet full grows; a full one rotates.
                let grown = if window && count as usize > held.len() {
                    count as usize
                } else if window {
                    held.len()
                } else {
                    (first + count) as usize
                };
                for logical in current_first + current_count..first + count {
                    let index_in = page_index(group, grown, logical)
                        .ok_or_else(|| Self::invalid("a continuation table has no pages"))?;
                    if index_in >= held.len() {
                        continue; // grown below
                    }
                    let page = &held[index_in];
                    if !page.is_writable() {
                        if !window {
                            return Ok(false);
                        }
                        replacements[index].push(index_in);
                    } else if !lease.has_write_claim(page.handle()) {
                        claims[index].push(index_in);
                    }
                }
                claims[index].sort_unstable();
                claims[index].dedup();
                replacements[index].sort_unstable();
                replacements[index].dedup();
                let demand = demands.last_mut().expect("pushed above");
                *demand = demand.at_least(grown);
                needed |= grown > held.len()
                    || !claims[index].is_empty()
                    || !replacements[index].is_empty();
            }
            if !needed {
                continue;
            }
            loop {
                match shared.pool.extend_leased_request(
                    request,
                    lease,
                    &demands,
                    &claims,
                    &replacements,
                ) {
                    Ok(()) => break,
                    Err(CachePoolError::InsufficientCapacity { .. }) => {
                        match shared.reclaim_for(self.tenant)? {
                            Reclaim::Freed => {}
                            Reclaim::Pending => return Err(CachePoolError::LeasePending.into()),
                            Reclaim::Exhausted => return Ok(false),
                        }
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
        for (index, group) in layout.groups().iter().copied().enumerate() {
            if !changed[index] {
                continue;
            }
            let plan = &mut groups[index];
            let mut indptr = Vec::with_capacity(requests.len() + 1);
            let mut slab_ids = Vec::with_capacity(plan.block_slab_ids.len() + requests.len());
            let mut slotnums = Vec::with_capacity(slab_ids.capacity());
            let mut starts = Vec::with_capacity(requests.len());
            indptr.push(0);
            for (request, &leased) in requests.iter().zip(next_leased) {
                let pages = request.groups()[index].pages();
                let (first, count) = group.active_blocks(leased);
                starts.push(first);
                for logical in first..first + count {
                    let page = page_for(group, pages, logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "continuation cache group {index} is missing logical block {logical}"
                        ))
                    })?;
                    let (slab_id, slotnum) = shared.pool.physical_address(page.handle())?;
                    slab_ids.push(slab_id);
                    slotnums.push(slotnum);
                }
                indptr.push(
                    u32::try_from(slab_ids.len()).map_err(|_| {
                        Self::invalid("cache block table exceeds the u32 CSR range")
                    })?,
                );
            }
            plan.block_indptr = indptr;
            plan.block_slab_ids = slab_ids;
            plan.block_slotnums = slotnums;
            plan.block_table_start_indices = starts;
        }
        Ok(true)
    }
}

pub(crate) struct AdmissionEstimate<'a> {
    layout: &'a KvLayout,
    blocks: Vec<u64>,
    occupied_parents: Vec<HashSet<u32>>,
    shared: HashSet<PageHandle>,
}

impl AdmissionEstimate<'_> {
    pub(crate) fn include(
        &mut self,
        pages: Option<&RequestPages>,
        context: u32,
    ) -> Result<u64, CachePoolError> {
        let mut parents = 0u64;
        for (index, group) in self.layout.groups().iter().copied().enumerate() {
            let count = group.active_blocks(context).1;
            let mut added = u64::from(count);
            if let Some(pages) = pages {
                for page in pages.groups()[index].pages().iter().take(count as usize) {
                    self.occupied_parents[index].insert(page.handle().parent());
                    if group.retention() == KvRetention::FullContext
                        && !group.is_recurrent()
                        && !page.is_writable()
                        && !self.shared.insert(page.handle())
                    {
                        added -= 1;
                    }
                }
            }
            self.blocks[index] = self.blocks[index]
                .checked_add(added)
                .ok_or(CachePoolError::DemandOverflow)?;
            // Live pages cannot relocate to achieve the ideal packing. Their
            // parent identities provide a floor even for fragmented prefixes.
            let group_parents = self.blocks[index]
                .div_ceil(u64::from(group.blocks_per_slab().get()))
                .max(self.occupied_parents[index].len() as u64);
            parents = parents
                .checked_add(group_parents)
                .ok_or(CachePoolError::DemandOverflow)?;
        }
        Ok(parents)
    }
}

fn prefix_alignment(layout: &KvLayout) -> Result<u32, CachePoolError> {
    fn gcd(mut left: u32, mut right: u32) -> u32 {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }
    layout.groups().iter().try_fold(1u32, |alignment, group| {
        alignment
            .checked_mul(group.block_tokens().get() / gcd(alignment, group.block_tokens().get()))
            .ok_or(CachePoolError::DemandOverflow)
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedKvBatchView {
    pub pool_id: u64,
    pub groups: Vec<KvGroupPlan>,
}

#[derive(Debug)]
pub enum PagedKvError {
    Cache(CachePoolError),
    Invalid(String),
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for PagedKvError {}

impl From<CachePoolError> for PagedKvError {
    fn from(error: CachePoolError) -> Self {
        Self::Cache(error)
    }
}

impl From<KvPlanError> for PagedKvError {
    fn from(error: KvPlanError) -> Self {
        Self::Invalid(error.to_string())
    }
}

fn page_for<R>(group: KvGroupLayout, pages: &[R], logical: u32) -> Option<&R> {
    page_index(group, pages.len(), logical).and_then(|index| pages.get(index))
}

fn page_index(group: KvGroupLayout, page_count: usize, logical: u32) -> Option<usize> {
    if page_count == 0 {
        return None;
    }
    let logical = logical as usize;
    let index =
        if group.is_recurrent() || matches!(group.retention(), KvRetention::SlidingWindow { .. }) {
            logical % page_count
        } else {
            logical
        };
    (index < page_count).then_some(index)
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroU64};

    use crate::{Phase, TokenRange};

    use super::*;

    #[test]
    fn ancestor_eviction_counts_no_reclaimed_storage_until_the_last_checkpoint() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(2, 2, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut request = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut request, 0, 2).unwrap();
        resources.publish_prefix(&mut request, &[1, 2], 2).unwrap();
        resources.ensure_context(&mut request, 2, 4).unwrap();
        resources
            .publish_prefix(&mut request, &[1, 2, 3, 4], 4)
            .unwrap();
        assert_eq!(resources.metrics().groups[0].reclaimable_blocks, 0);
        resources.retire(request).unwrap();

        let cached = resources.metrics();
        assert_eq!(cached.reclaimable_slabs, 1);
        assert_eq!(cached.groups[0].reclaimable_blocks, 2);
        assert!(resources.evict_one_prefix().unwrap());
        let ancestor = resources.metrics();
        assert_eq!(ancestor.groups[0].occupied_blocks, 2);
        assert_eq!(ancestor.prefix.unwrap().reclaimed_blocks, 0);

        assert!(resources.evict_one_prefix().unwrap());
        let empty = resources.metrics();
        assert_eq!(empty.free_slabs, 2);
        assert_eq!(empty.prefix.unwrap().reclaimed_blocks, 2);
        assert_eq!(empty.prefix.unwrap().reclaimed_slabs, 1);
    }

    #[test]
    fn layout_estimate_matches_fresh_pool_allocation() {
        let layout = KvLayout::new(
            20,
            vec![
                KvGroupLayout::new(4, 2, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(
                    4,
                    4,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(5).unwrap(),
                    },
                )
                .unwrap(),
                KvGroupLayout::new_with_family(
                    8,
                    2,
                    KvRetention::FullContext,
                    super::super::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let contexts = [0, 1, 4, 5, 17];
        let expected = layout.estimated_parent_count(&contexts).unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        assert_eq!(
            resources.estimated_parent_count(&contexts).unwrap(),
            expected
        );
        let states: Vec<_> = contexts
            .into_iter()
            .enumerate()
            .map(|(index, context)| {
                resources
                    .attach(
                        RequestId::new(NonZeroU64::new(index as u64 + 1).unwrap()),
                        context,
                    )
                    .unwrap()
            })
            .collect();
        assert_eq!(
            u64::from(resources.parent_capacity()) - resources.pool().free_parent_count() as u64,
            expected
        );
        for state in states {
            resources.retire(state).unwrap();
        }
        assert_eq!(resources.pool().free_parent_count(), 20);
    }

    #[test]
    fn admission_preserves_fragmented_shared_prefix_parents() {
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(1, 8, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(8, 1, KvRetention::FullContext).unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = |value| RequestId::new(NonZeroU64::new(value).unwrap());
        let demands = |blocks| {
            [
                GroupDemand::new(GroupId::new(0), blocks),
                GroupDemand::new(GroupId::new(1), 0),
            ]
        };
        let mut seed = resources.attach(id(1), 0).unwrap();
        let mut fillers = Vec::new();
        for blocks in 1..=3 {
            resources
                .pool()
                .grow_request(&mut seed, &demands(blocks))
                .unwrap();
            // Fill each parent before growing the prefix into another parent.
            let mut filler = resources.attach(id(10 + blocks as u64), 0).unwrap();
            resources
                .pool()
                .grow_request(&mut filler, &demands(7))
                .unwrap();
            fillers.push(filler);
        }
        let prefix = resources
            .pool()
            .share_request_prefix(&mut seed, &demands(3))
            .unwrap();
        for filler in fillers {
            resources.retire(filler).unwrap();
        }
        resources.retire(seed).unwrap();

        let mut requests = Vec::new();
        for index in 0..2 {
            let mut request = resources.attach(id(20 + index), 0).unwrap();
            resources
                .pool()
                .attach_shared_prefix(&mut request, &prefix)
                .unwrap();
            resources.ensure_context(&mut request, 3, 4).unwrap();
            requests.push(request);
        }
        let mut estimate = resources.admission_estimate();
        assert_eq!(estimate.include(Some(&requests[0]), 4).unwrap(), 4);
        assert_eq!(estimate.include(Some(&requests[1]), 4).unwrap(), 5);
        // Three immutable child handles count once, but pin three parents.
        // Each request also owns one private child and a second-group parent.
        assert_eq!(estimate.blocks, [5, 2]);
        assert_eq!(resources.estimated_parent_count(&[4, 4]).unwrap(), 3);
        assert_eq!(8 - resources.pool().free_parent_count(), 5);
        drop(estimate);
        for request in requests {
            resources.retire(request).unwrap();
        }
        resources.pool().retire_cached_prefix(prefix).unwrap();
        assert_eq!(resources.pool().free_parent_count(), 8);
    }

    #[test]
    fn full_and_sliding_groups_share_one_current_batch_lease() {
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(
                    2,
                    2,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(7).unwrap(),
                    },
                )
                .unwrap()
                .with_read_window_tokens(4)
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let state = resources.attach(id, 12).unwrap();
        let step = SequenceStep {
            phase: Phase::Prefill,
            tokens: TokenRange {
                start: 0,
                count: NonZeroU32::new(4).unwrap(),
            },
            sample: false,
        };
        let requests = [RequestResourceRef {
            request_id: id,
            state: &state,
        }];
        let access = resources.acquire(&requests, &[step], 0).unwrap();
        assert_eq!(access.view.groups.len(), 2);
        assert!(resources.acquire(&requests, &[step], 0).is_err());
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }

    #[test]
    fn request_pages_grow_on_demand_and_reset_for_replay() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);

        resources.ensure_context(&mut state, 0, 5).unwrap();
        assert_eq!(state.groups()[0].pages().len(), 2);
        assert_eq!(resources.pool().occupied_block_count(), 2);
        assert!(matches!(
            resources.ensure_context(&mut state, 5, 9),
            Err(CachePoolError::InsufficientCapacity { .. })
        ));
        assert_eq!(state.groups()[0].pages().len(), 2);

        resources.reset(&mut state).unwrap();
        assert!(state.groups()[0].pages().is_empty());
        assert_eq!(resources.pool().occupied_block_count(), 0);
        resources.ensure_context(&mut state, 0, 4).unwrap();
        assert_eq!(state.groups()[0].pages().len(), 1);
        resources.retire(state).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
    }

    #[test]
    fn decode_write_lookahead_is_leased_but_not_reported_as_input_append() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut state, 4, 8).unwrap();
        let step = SequenceStep {
            phase: Phase::Decode,
            tokens: TokenRange {
                start: 4,
                count: NonZeroU32::new(1).unwrap(),
            },
            sample: true,
        };
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: id,
                    state: &state,
                }],
                &[step],
                3,
            )
            .unwrap();
        let group = &access.view.groups[0];
        assert_eq!(group.block_slab_ids.len(), 2);
        assert_eq!(group.block_slotnums.len(), 2);
        assert_eq!(group.append_slab_ids, vec![group.block_slab_ids[1]]);
        assert_eq!(group.append_slotnums, vec![group.block_slotnums[1]]);
        assert_eq!(group.append_block_offsets, vec![0]);
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }

    #[test]
    fn immutable_history_prefix_is_shared_and_suffix_stays_writable() {
        let layout = KvLayout::new(
            4,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        resources.ensure_context(&mut first, 0, 4).unwrap();
        let write = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: first_id,
                    state: &first,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 0,
                        count: NonZeroU32::new(4).unwrap(),
                    },
                    sample: false,
                }],
                0,
            )
            .unwrap();
        resources.release(write.lease);
        assert!(
            resources
                .publish_prefix(&mut first, &[10, 11, 12, 13], 4)
                .unwrap()
        );

        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut second, &[10, 11, 12, 13, 14])
                .unwrap(),
            4
        );
        assert_eq!(resources.pool().occupied_block_count(), 2);
        resources.ensure_context(&mut second, 4, 5).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 3);
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: second_id,
                    state: &second,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 4,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap();
        assert_eq!(access.view.groups[0].block_slab_ids.len(), 3);
        resources.release(access.lease);
        assert_eq!(
            resources.prefix_metrics(),
            Some(PrefixCacheMetrics {
                entries: 1,
                hits: 1,
                ..PrefixCacheMetrics::default()
            })
        );
        resources.retire(second).unwrap();
        resources.retire(first).unwrap();
    }

    #[test]
    fn shared_batch_read_claims_keep_all_tables_and_pin_retired_owners() {
        let layout = KvLayout::new(
            4,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let seed_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut seed = resources.attach(seed_id, 0).unwrap();
        resources.ensure_context(&mut seed, 0, 4).unwrap();
        resources
            .publish_prefix(&mut seed, &[10, 11, 12, 13], 4)
            .unwrap();
        resources.retire(seed).unwrap();

        let mut owners = Vec::new();
        for value in 2..=3 {
            let id = RequestId::new(NonZeroU64::new(value).unwrap());
            let mut state = resources.attach(id, 0).unwrap();
            assert_eq!(
                resources
                    .restore_prefix(&mut state, &[10, 11, 12, 13, 20])
                    .unwrap(),
                4
            );
            resources.ensure_context(&mut state, 4, 5).unwrap();
            owners.push((id, state));
        }
        let steps = [SequenceStep {
            phase: Phase::Decode,
            tokens: TokenRange {
                start: 4,
                count: NonZeroU32::new(1).unwrap(),
            },
            sample: true,
        }; 2];
        let requests = owners
            .iter()
            .map(|(id, state)| RequestResourceRef {
                request_id: *id,
                state,
            })
            .collect::<Vec<_>>();
        let access = resources.acquire(&requests, &steps, 0).unwrap();
        assert_eq!(access.view.groups[0].block_indptr, [0, 3, 6]);
        let addresses = &access.view.groups[0].block_slab_ids;
        assert_eq!(&addresses[..2], &addresses[3..5]);
        assert_ne!(addresses[2], addresses[5]);
        assert_eq!(access.lease.len(), 4); // Two shared reads, two distinct writers.
        drop(requests);
        for (_, state) in owners {
            resources.retire(state).unwrap();
        }
        let pinned = resources.metrics();
        assert_eq!(pinned.reclaimable_slabs, 0);
        assert_eq!(pinned.groups[0].cached_blocks, 2);
        assert_eq!(pinned.groups[0].reclaimable_blocks, 0);
        assert_eq!(pinned.groups[0].pinned_blocks, 4);
        resources.clear_prefix_cache().unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 4);
        let evicted = resources.metrics();
        assert_eq!(evicted.groups[0].cached_blocks, 0);
        assert_eq!(evicted.groups[0].pinned_blocks, 4);
        assert_eq!(evicted.prefix.unwrap().reclaimed_blocks, 0);
        assert_eq!(evicted.prefix.unwrap().reclaimed_slabs, 0);
        resources.release(access.lease);
        assert_eq!(resources.pool().free_parent_count(), 4);
        assert_eq!(resources.metrics().groups[0].pinned_blocks, 0);
    }

    #[test]
    fn batch_shared_read_deduplication_never_hides_duplicate_writers() {
        let layout = KvLayout::new(
            1,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let state = resources.attach(id, 2).unwrap();
        let requests = [
            RequestResourceRef {
                request_id: id,
                state: &state,
            },
            RequestResourceRef {
                request_id: id,
                state: &state,
            },
        ];
        let steps = [SequenceStep {
            phase: Phase::Decode,
            tokens: TokenRange {
                start: 0,
                count: NonZeroU32::new(1).unwrap(),
            },
            sample: true,
        }; 2];
        assert!(matches!(
            resources.acquire(&requests, &steps, 0),
            Err(PagedKvError::Cache(CachePoolError::DuplicateAccessClaim))
        ));
        assert!(!resources.batch_active);
        resources.retire(state).unwrap();
        assert_eq!(resources.pool().free_parent_count(), 1);
    }

    #[test]
    fn shared_sliding_snapshot_is_copy_on_write_when_ring_advances() {
        let layout = KvLayout::new(
            5,
            vec![
                KvGroupLayout::new(
                    2,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        for position in 0..6 {
            resources
                .ensure_context(&mut first, position, position + 1)
                .unwrap();
        }
        let write = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: first_id,
                    state: &first,
                }],
                &[SequenceStep {
                    phase: Phase::Decode,
                    tokens: TokenRange {
                        start: 5,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: false,
                }],
                0,
            )
            .unwrap();
        resources.release(write.lease);
        resources
            .publish_prefix(&mut first, &[1, 2, 3, 4, 5, 6], 6)
            .unwrap();

        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut second, &[1, 2, 3, 4, 5, 6, 7])
                .unwrap(),
            6
        );
        resources.ensure_context(&mut second, 6, 7).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 4);
        assert!(second.groups()[0].pages()[0].is_writable());
        resources.retire(second).unwrap();
        resources.retire(first).unwrap();
    }

    #[test]
    fn unsafe_window_write_is_rejected_before_copy_on_write() {
        let layout = KvLayout::new(
            20,
            vec![
                KvGroupLayout::new(
                    2,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        for position in 0..6 {
            resources
                .ensure_context(&mut state, position, position + 1)
                .unwrap();
        }
        resources
            .publish_prefix(&mut state, &(0..6).collect::<Vec<_>>(), 6)
            .unwrap();
        let original = state.groups()[0]
            .pages()
            .iter()
            .map(|page| page.handle())
            .collect::<Vec<_>>();
        assert_eq!(
            resources.ensure_context(&mut state, 6, 12),
            Err(CachePoolError::InvalidWriteSpan {
                requested: 6,
                maximum: 1
            })
        );
        assert_eq!(
            state.groups()[0]
                .pages()
                .iter()
                .map(|page| page.handle())
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(resources.pool().occupied_block_count(), 3);
        resources.retire(state).unwrap();
    }

    #[test]
    fn chunked_window_prefill_keeps_early_query_history_after_prefix_cow() {
        let group = KvGroupLayout::new(
            2,
            1,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(9).unwrap(),
            },
        )
        .unwrap()
        .with_read_window_tokens(4)
        .unwrap();
        let mut resources =
            PagedKvResources::with_prefix_cache(KvLayout::new(20, vec![group]).unwrap()).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut state, 0, 6).unwrap();
        let history = state.groups()[0]
            .pages()
            .iter()
            .map(|page| resources.pool().physical_address(page.handle()).unwrap())
            .collect::<Vec<_>>();
        resources
            .publish_prefix(&mut state, &(0..6).collect::<Vec<_>>(), 6)
            .unwrap();
        resources.ensure_context(&mut state, 6, 12).unwrap();
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: id,
                    state: &state,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 6,
                        count: NonZeroU32::new(6).unwrap(),
                    },
                    sample: false,
                }],
                0,
            )
            .unwrap();
        let plan = &access.view.groups[0];
        assert_eq!(plan.block_table_start_indices, vec![1]);
        let addresses = plan
            .block_slab_ids
            .iter()
            .copied()
            .zip(plan.block_slotnums.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(&addresses[..2], &history[1..]);
        assert_eq!(addresses.len(), 5);
        for (offset, (&slab, &slot)) in plan
            .append_slab_ids
            .iter()
            .zip(&plan.append_slotnums)
            .enumerate()
        {
            assert_eq!((slab, slot), addresses[(6 + offset) / 2 - 1]);
            assert!(!history[1..].contains(&(slab, slot)));
        }
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }

    #[test]
    fn live_allocation_evicts_lru_prefix_pages_under_pressure() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let cached_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut cached = resources.attach(cached_id, 0).unwrap();
        resources.ensure_context(&mut cached, 0, 4).unwrap();
        resources
            .publish_prefix(&mut cached, &[1, 2, 3, 4], 4)
            .unwrap();
        resources.retire(cached).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 2);

        let live_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut live = resources.attach(live_id, 0).unwrap();
        resources.ensure_context(&mut live, 0, 4).unwrap();
        let prefix_metrics = resources.prefix_metrics().unwrap();
        assert_eq!(prefix_metrics.entries, 0);
        assert_eq!(prefix_metrics.evictions, 1);
        assert_eq!(resources.pool().occupied_block_count(), 2);
        resources.retire(live).unwrap();
    }

    #[test]
    fn recurrent_prefix_keeps_input_shared_and_allocates_a_writable_output() {
        let layout = KvLayout::new(
            10,
            vec![
                KvGroupLayout::new(16, 1, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new_with_family(
                    128,
                    6,
                    KvRetention::FullContext,
                    crate::resources::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        resources.ensure_context(&mut first, 0, 128).unwrap();
        assert!(
            resources
                .publish_prefix(&mut first, &(0..128).collect::<Vec<_>>(), 128)
                .unwrap()
        );
        resources.retire(first).unwrap();

        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut second, &(0..129).collect::<Vec<_>>())
                .unwrap(),
            128
        );
        resources.ensure_context(&mut second, 128, 129).unwrap();
        let state_pages = second.groups()[1].pages();
        assert_eq!(state_pages.len(), 2);
        assert!(!state_pages[0].is_writable());
        assert!(state_pages[1].is_writable());

        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: second_id,
                    state: &second,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 128,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap();
        let state = &access.view.groups[1];
        assert_eq!(state.block_slab_ids.len(), 2);
        assert_eq!(state.block_slotnums.len(), 2);
        assert_eq!(state.append_slab_ids, vec![state.block_slab_ids[1]]);
        assert_eq!(state.append_slotnums, vec![state.block_slotnums[1]]);
        resources.release(access.lease);
        resources.retire(second).unwrap();
    }

    #[test]
    fn recurrent_context_wraps_across_the_third_logical_block() {
        let layout = KvLayout::new(
            2,
            vec![
                KvGroupLayout::new_with_family(
                    128,
                    6,
                    KvRetention::FullContext,
                    crate::resources::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();

        resources.ensure_context(&mut state, 0, 257).unwrap();

        assert_eq!(state.groups()[0].pages().len(), 2);
        assert!(
            state.groups()[0]
                .pages()
                .iter()
                .all(|page| page.is_writable())
        );
        resources.retire(state).unwrap();
    }

    fn decode_access(
        resources: &mut PagedKvResources,
        id: RequestId,
        state: &RequestPages,
        start: u32,
    ) -> BatchAccess<PagedKvBatchView, WriterAccessLease> {
        resources
            .acquire(
                &[RequestResourceRef {
                    request_id: id,
                    state,
                }],
                &[SequenceStep {
                    phase: Phase::Decode,
                    tokens: TokenRange {
                        start,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap()
    }

    #[test]
    fn continuation_extension_grows_inside_the_lease_or_falls_back_unchanged() {
        for (slabs, extended) in [(2, true), (1, false)] {
            let layout = KvLayout::new(
                slabs,
                vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
            )
            .unwrap();
            let mut resources = PagedKvResources::new(layout).unwrap();
            let id = RequestId::new(NonZeroU64::new(1).unwrap());
            let mut state = resources.attach(id, 0).unwrap();
            resources.ensure_context(&mut state, 0, 4).unwrap();
            let mut access = decode_access(&mut resources, id, &state, 3);
            let mut groups = access.view.groups.clone();
            let claims = access.lease.len();
            let ok = resources
                .extend_continuation(&mut [&mut state], &mut access.lease, &mut groups, &[5])
                .unwrap();
            assert_eq!(ok, extended);
            if extended {
                assert_eq!(groups[0].block_indptr, vec![0, 2]);
                assert_eq!(groups[0].block_table_start_indices, vec![0]);
                assert_eq!(access.lease.len(), claims + 1);
                assert_eq!(state.groups()[0].pages().len(), 2);
            } else {
                assert_eq!(groups, access.view.groups);
                assert_eq!(access.lease.len(), claims);
                assert_eq!(state.groups()[0].pages().len(), 1);
            }
            // The extension page is an ordinary claim: it returns with the lease.
            resources.release(access.lease);
            resources.retire(state).unwrap();
            assert_eq!(resources.pool().occupied_block_count(), 0);
        }
    }

    #[test]
    fn continuation_extension_copies_a_shared_ring_page_on_write() {
        // A 3-page ring of 2-token blocks. The second request restores the first one's 6 tokens, so its ring
        // pages are shared; rotating the table onto block 4 replaces the shared page of block 1 in the lease.
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(
                    2,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        for position in 0..6 {
            resources
                .ensure_context(&mut first, position, position + 1)
                .unwrap();
        }
        resources
            .publish_prefix(&mut first, &[1, 2, 3, 4, 5, 6], 6)
            .unwrap();
        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        resources
            .restore_prefix(&mut second, &[1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        resources.ensure_context(&mut second, 6, 7).unwrap();
        assert!(!second.groups()[0].pages()[1].is_writable());
        let mut access = decode_access(&mut resources, second_id, &second, 6);
        assert_eq!(access.view.groups[0].block_table_start_indices, vec![1]);
        let mut groups = access.view.groups.clone();
        let claims = access.lease.len();
        assert!(
            resources
                .extend_continuation(&mut [&mut second], &mut access.lease, &mut groups, &[9])
                .unwrap()
        );
        assert_eq!(groups[0].block_table_start_indices, vec![2]);
        assert!(second.groups()[0].pages()[1].is_writable());
        assert_eq!(access.lease.len(), claims + 1);
        resources.release(access.lease);
        resources.retire(second).unwrap();
        resources.retire(first).unwrap();
    }
}

#[cfg(test)]
mod boundary_tests {
    use std::num::NonZeroU64;

    use super::*;

    /// One card's arena in memory, split at `boundary`: side 0 (the first) holds [1, boundary), side 1
    /// [boundary, capacity]. With one card a given edge moves the boundary at once.
    #[derive(Debug)]
    struct Table {
        capacity: u32,
        boundary: u32,
        wants: [bool; 2],
        oldest: [Option<u64>; 2],
        spare: [bool; 2],
        floors: [u32; 2],
        clock: u64,
    }

    impl Table {
        fn region(&self, side: usize) -> (u32, u32) {
            if side == 0 {
                (1, self.boundary - 1)
            } else {
                (self.boundary, self.capacity)
            }
        }

        fn size(&self, side: usize) -> u32 {
            let (first, last) = self.region(side);
            (last + 1).saturating_sub(first)
        }
    }

    #[derive(Debug)]
    struct Side {
        table: Arc<Mutex<Table>>,
        side: usize,
    }

    impl Side {
        fn table(&self) -> MutexGuard<'_, Table> {
            self.table.lock().unwrap()
        }
    }

    impl SlabBoundary for Side {
        fn region(&self) -> (u32, u32) {
            self.table().region(self.side)
        }

        fn low(&self) -> bool {
            self.side == 0
        }

        fn want(&self, wanting: bool) {
            self.table().wants[self.side] = wanting;
        }

        fn give(&self, slab: u32) -> bool {
            let mut table = self.table();
            if table.size(self.side) <= table.floors[self.side] {
                return false;
            }
            let edge = if self.side == 0 {
                table.boundary - 1
            } else {
                table.boundary
            };
            assert_eq!(slab, edge, "a side gives its edge slab");
            if self.side == 0 {
                table.boundary -= 1;
            } else {
                table.boundary += 1;
            }
            true
        }

        fn tick(&self) -> u64 {
            let mut table = self.table();
            table.clock += 1;
            table.clock
        }

        fn publish(&self, oldest: Option<u64>, spare: bool) {
            let mut table = self.table();
            table.oldest[self.side] = oldest;
            table.spare[self.side] = spare;
        }

        fn other(&self) -> OtherSide {
            let table = self.table();
            let other = 1 - self.side;
            OtherSide {
                oldest: table.oldest[other],
                wants: table.wants[other],
                can_give: table.size(other) > table.floors[other],
                spare: table.spare[other],
            }
        }

        fn floor(&self) -> u32 {
            self.table().floors[self.side]
        }

        fn other_floor(&self) -> u32 {
            self.table().floors[1 - self.side]
        }
    }

    fn sides(
        capacity: u32,
        boundary: u32,
        floors: [u32; 2],
    ) -> (SharedCachePool, SharedCachePool, Arc<Mutex<Table>>) {
        let table = Arc::new(Mutex::new(Table {
            capacity,
            boundary,
            wants: [false; 2],
            oldest: [None; 2],
            spare: [false; 2],
            floors,
            clock: 0,
        }));
        let pool = |side| {
            let boundary: Arc<dyn SlabBoundary> = Arc::new(Side {
                table: table.clone(),
                side,
            });
            SharedCachePool::with_boundary(capacity, boundary, None).unwrap()
        };
        (pool(0), pool(1), table)
    }

    fn engine(pool: &SharedCachePool) -> PagedKvResources {
        // one 2-token block a slab
        let layout = KvLayout::new(
            pool.parent_capacity(),
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        PagedKvResources::join(pool, layout, true, 0).unwrap()
    }

    fn id(value: u64) -> RequestId {
        RequestId::new(NonZeroU64::new(value).unwrap())
    }

    /// A request of `tokens` tokens that stays live.
    fn live(engine: &mut PagedKvResources, value: u64, tokens: u32) -> RequestPages {
        let mut pages = engine.attach(id(value), 0).unwrap();
        engine.ensure_context(&mut pages, 0, tokens).unwrap();
        pages
    }

    /// A finished request whose `tokens` tokens stay as a cached prefix.
    fn cached(engine: &mut PagedKvResources, value: u64, tokens: u32) {
        let mut pages = live(engine, value, tokens);
        let prompt: Vec<u32> = (0..tokens)
            .map(|token| token + 100 * value as u32)
            .collect();
        assert!(engine.publish_prefix(&mut pages, &prompt, tokens).unwrap());
        engine.retire(pages).unwrap();
    }

    fn entries(engine: &PagedKvResources) -> u64 {
        engine.prefix_metrics().unwrap().entries
    }

    /// Each pool holds exactly its side of the boundary, and neither is below its floor.
    fn check(table: &Arc<Mutex<Table>>, pools: [&SharedCachePool; 2]) {
        let table = table.lock().unwrap();
        for (side, pool) in pools.iter().enumerate() {
            assert_eq!(pool.held_slabs() as u32, table.size(side), "side {side}");
            assert!(
                table.size(side) >= table.floors[side],
                "side {side} below its floor"
            );
        }
    }

    /// Between syncs: no pool holds more than its side of the boundary (a slab the other side gave arrives at the
    /// next sync), and none holds less than its floor.
    fn check_between(table: &Arc<Mutex<Table>>, pools: [&SharedCachePool; 2]) {
        let table = table.lock().unwrap();
        for (side, pool) in pools.iter().enumerate() {
            let held = pool.held_slabs() as u32;
            assert!(
                held <= table.size(side) && held >= table.floors[side],
                "side {side}: {held}"
            );
        }
    }

    /// Serve both sides until neither moves the boundary.
    fn settle(a: &SharedCachePool, b: &SharedCachePool) {
        for _ in 0..20 {
            a.serve_boundary().unwrap();
            b.serve_boundary().unwrap();
        }
    }

    #[test]
    fn each_side_holds_its_side_of_the_boundary() {
        let (a, b, table) = sides(8, 4, [1, 2]);
        assert_eq!((a.held_slabs(), b.held_slabs()), (3, 5));
        check(&table, [&a, &b]);
        // the other side's floor is never this side's
        assert_eq!(engine(&a).admission_capacity(), 6);
        assert_eq!(engine(&b).admission_capacity(), 7);
    }

    #[test]
    fn the_older_prefix_goes_first_on_either_side() {
        // A's prefix is the older one: B asks, and A gives its edge slabs with the prefix in them
        let (a, b, table) = sides(8, 5, [1, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        cached(&mut engine_a, 1, 4);
        cached(&mut engine_b, 2, 4);
        settle(&a, &b);
        let mut need = engine_b.attach(id(3), 0).unwrap();
        while engine_b.ensure_context(&mut need, 0, 10) == Err(CachePoolError::LeasePending) {
            a.serve_boundary().unwrap();
        }
        assert_eq!((entries(&engine_a), entries(&engine_b)), (0, 1));
        engine_b.retire(need).unwrap();
        check(&table, [&a, &b]);

        // B's prefix is the older one, and A has no empty slab: B evicts its own and asks A for nothing
        let (a, b, table) = sides(8, 5, [1, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        cached(&mut engine_b, 1, 4);
        cached(&mut engine_a, 2, 8); // all four of A's slabs
        settle(&a, &b);
        let mut need = engine_b.attach(id(3), 0).unwrap();
        engine_b.ensure_context(&mut need, 0, 8).unwrap();
        assert_eq!((entries(&engine_a), entries(&engine_b)), (1, 0));
        assert_eq!(table.lock().unwrap().wants, [false, false]);
        engine_b.retire(need).unwrap();
        check(&table, [&a, &b]);
    }

    #[test]
    fn a_side_takes_the_other_sides_empty_slabs_before_it_evicts_its_own_prefix() {
        // A is idle with empty slabs and no prefix; B's prefix fills its slabs, and B needs two more
        let (a, b, table) = sides(8, 5, [1, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        cached(&mut engine_b, 1, 8);
        settle(&a, &b);
        assert_eq!(table.lock().unwrap().spare, [true, false]);
        let mut need = engine_b.attach(id(2), 0).unwrap();
        while engine_b.ensure_context(&mut need, 0, 4) == Err(CachePoolError::LeasePending) {
            a.serve_boundary().unwrap();
        }
        assert_eq!(entries(&engine_b), 1, "B kept its prefix");
        assert_eq!(
            table.lock().unwrap().boundary,
            3,
            "A gave its slabs 4 and 3"
        );
        // A keeps a headroom of one more floor: with 2 slabs it is not spare, and B's next slab is its prefix's
        settle(&a, &b);
        assert!(!table.lock().unwrap().spare[0], "A is at its headroom");
        engine_b.ensure_context(&mut need, 0, 6).unwrap();
        assert_eq!(entries(&engine_b), 0);
        assert_eq!(table.lock().unwrap().boundary, 3);
        engine_b.retire(need).unwrap();
        let mut held = engine_a.attach(id(3), 0).unwrap();
        engine_a.ensure_context(&mut held, 0, 2).unwrap();
        engine_a.retire(held).unwrap();
        settle(&a, &b);
        check(&table, [&a, &b]);
    }

    #[test]
    fn live_requests_are_never_evicted_and_the_waiter_gets_their_slabs_later() {
        let (a, b, table) = sides(4, 4, [1, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        let held = live(&mut engine_a, 1, 6); // all three of A's slabs
        let mut need = engine_b.attach(id(2), 0).unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine_b.ensure_context(&mut need, 0, 4),
                Err(CachePoolError::LeasePending)
            );
            assert_eq!(engine_b.capacity_wait(), CapacityWait::Wait);
            a.serve_boundary().unwrap(); // A's edge holds a live block: it stays
            assert_eq!(a.held_slabs(), 3);
        }
        engine_a.retire(held).unwrap();
        a.serve_boundary().unwrap();
        engine_b.ensure_context(&mut need, 0, 4).unwrap();
        engine_b.retire(need).unwrap();
        check(&table, [&a, &b]);
    }

    #[test]
    fn a_side_gives_no_slab_of_its_floor() {
        let (a, b, table) = sides(8, 7, [3, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        cached(&mut engine_a, 1, 12); // all six of A's slabs
        settle(&a, &b);
        let mut need = engine_b.attach(id(2), 0).unwrap();
        // B asks for far more than A can give: A gives down to its floor, and B cannot have it all
        for _ in 0..20 {
            if engine_b.ensure_context(&mut need, 0, 16).is_ok() {
                panic!("B took A's floor");
            }
            a.serve_boundary().unwrap();
        }
        assert_eq!(a.held_slabs(), 3, "A keeps its floor, empty or not");
        check(&table, [&a, &b]);
        engine_b.ensure_context(&mut need, 0, 10).unwrap();
        engine_b.retire(need).unwrap();
    }

    /// Records the slabs it clears; fails while `fail` is set.
    #[derive(Debug, Default)]
    struct Recorder {
        cleared: Mutex<Vec<u32>>,
        fail: std::sync::atomic::AtomicBool,
    }

    impl SlabScrub for Recorder {
        fn scrub(&self, slabs: &[u32]) -> Result<(), String> {
            if self.fail.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("no device".into());
            }
            self.cleared.lock().unwrap().extend_from_slice(slabs);
            Ok(())
        }
    }

    #[test]
    fn a_side_with_a_scrub_clears_every_slab_that_crosses_before_either_side_uses_it() {
        let table = Arc::new(Mutex::new(Table {
            capacity: 4,
            boundary: 3,
            wants: [false; 2],
            oldest: [None; 2],
            spare: [false; 2],
            floors: [1, 1],
            clock: 0,
        }));
        let side = |side| -> Arc<dyn SlabBoundary> {
            Arc::new(Side {
                table: table.clone(),
                side,
            })
        };
        let recorder = Arc::new(Recorder::default());
        let a = SharedCachePool::with_boundary(4, side(0), None).unwrap();
        let b = SharedCachePool::with_boundary(4, side(1), Some(recorder.clone())).unwrap();
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        // B needs a third slab: A gives slab 2, and B clears it before it places a block there
        let mut need = engine_b.attach(id(1), 0).unwrap();
        assert_eq!(
            engine_b.ensure_context(&mut need, 0, 6),
            Err(CachePoolError::LeasePending)
        );
        a.serve_boundary().unwrap();
        assert_eq!(table.lock().unwrap().boundary, 2);
        assert_eq!(
            engine_b.ensure_context(&mut need, 0, 6),
            Err(CachePoolError::LeasePending),
            "slab 2 is not clear yet"
        );
        recorder
            .fail
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(b.serve_boundary(), Err(CachePoolError::ScrubFailed));
        assert_eq!(
            b.held_slabs(),
            2,
            "a slab that is not clear stays out of use"
        );
        recorder
            .fail
            .store(false, std::sync::atomic::Ordering::Relaxed);
        b.serve_boundary().unwrap();
        assert_eq!(*recorder.cleared.lock().unwrap(), [2]);
        engine_b.ensure_context(&mut need, 0, 6).unwrap();
        engine_b.retire(need).unwrap();
        // A needs slab 2 back: B clears it before it gives it
        let mut need = engine_a.attach(id(2), 0).unwrap();
        assert_eq!(
            engine_a.ensure_context(&mut need, 0, 4),
            Err(CachePoolError::LeasePending)
        );
        b.serve_boundary().unwrap();
        assert_eq!(*recorder.cleared.lock().unwrap(), [2, 2]);
        assert_eq!(table.lock().unwrap().boundary, 3);
        engine_a.ensure_context(&mut need, 0, 4).unwrap();
        engine_a.retire(need).unwrap();
        settle(&a, &b);
        check(&table, [&a, &b]);
    }

    #[test]
    fn a_side_evicts_its_own_prefix_when_the_older_one_cannot_go() {
        // A is at its floor: its older prefix stays, and B evicts its own
        let (a, b, table) = sides(8, 5, [4, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        cached(&mut engine_a, 1, 8);
        cached(&mut engine_b, 2, 8);
        settle(&a, &b);
        let mut need = engine_b.attach(id(3), 0).unwrap();
        engine_b.ensure_context(&mut need, 0, 4).unwrap();
        assert_eq!((entries(&engine_a), entries(&engine_b)), (1, 0));
        engine_b.retire(need).unwrap();
        check(&table, [&a, &b]);

        // A can give, but a live block holds its edge slab: B waits a short time, then evicts its own
        let (a, b, table) = sides(8, 5, [1, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        cached(&mut engine_a, 1, 6); // slabs 1 to 3
        let edge = live(&mut engine_a, 2, 2); // slab 4
        cached(&mut engine_b, 3, 8);
        settle(&a, &b);
        let mut need = engine_b.attach(id(4), 0).unwrap();
        assert_eq!(
            engine_b.ensure_context(&mut need, 0, 2),
            Err(CachePoolError::LeasePending)
        );
        a.serve_boundary().unwrap();
        assert_eq!(a.held_slabs(), 4, "the live block keeps A's edge slab");
        std::thread::sleep(WANT_LIFETIME + Duration::from_millis(5));
        engine_b.ensure_context(&mut need, 0, 2).unwrap();
        assert_eq!((entries(&engine_a), entries(&engine_b)), (1, 0));
        engine_b.retire(need).unwrap();
        engine_a.retire(edge).unwrap();
        settle(&a, &b);
        check(&table, [&a, &b]);
    }

    #[test]
    fn the_first_side_waits_and_the_other_gives_way_when_both_wait() {
        let (a, b, _) = sides(4, 3, [1, 1]);
        let (mut engine_a, mut engine_b) = (engine(&a), engine(&b));
        let mut held_a = live(&mut engine_a, 1, 4);
        let mut held_b = live(&mut engine_b, 2, 4);
        // each needs one more slab, and the other holds it with a live request
        assert_eq!(
            engine_a.ensure_context(&mut held_a, 0, 6),
            Err(CachePoolError::LeasePending)
        );
        assert_eq!(
            engine_b.ensure_context(&mut held_b, 0, 6),
            Err(CachePoolError::LeasePending)
        );
        assert_eq!(engine_b.capacity_wait(), CapacityWait::Yield);
        assert_eq!(
            engine_a.capacity_wait(),
            CapacityWait::Wait,
            "the first side never gives way"
        );
        engine_a.retire(held_a).unwrap();
        engine_b.retire(held_b).unwrap();
    }

    #[test]
    fn the_boundary_never_leaks_a_slab_under_random_churn() {
        let (a, b, table) = sides(12, 6, [2, 1]);
        let pools = [&a, &b];
        let mut engines = [engine(&a), engine(&b)];
        let mut held: [Vec<RequestPages>; 2] = [Vec::new(), Vec::new()];
        let mut seed: u64 = 0x5eed;
        let mut next = |range: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % range
        };
        let (mut value, mut pending, mut served) = (0, 0, 0);
        for _ in 0..800 {
            let side = next(2) as usize;
            value += 1;
            match next(4) {
                0 | 1 => {
                    let tokens = 2 + 2 * next(4) as u32;
                    let mut pages = engines[side].attach(id(value), 0).unwrap();
                    match engines[side].ensure_context(&mut pages, 0, tokens) {
                        Ok(()) => {
                            served += 1;
                            if next(2) == 0 {
                                let prompt: Vec<u32> =
                                    (0..tokens).map(|token| token + value as u32 * 50).collect();
                                engines[side]
                                    .publish_prefix(&mut pages, &prompt, tokens)
                                    .unwrap();
                                engines[side].retire(pages).unwrap();
                            } else {
                                held[side].push(pages);
                            }
                        }
                        Err(
                            CachePoolError::LeasePending
                            | CachePoolError::InsufficientCapacity { .. },
                        ) => {
                            pending += 1;
                            engines[side].retire(pages).unwrap();
                        }
                        Err(error) => panic!("{error}"),
                    }
                }
                2 => {
                    if !held[side].is_empty() {
                        let index = next(held[side].len() as u64) as usize;
                        let pages = held[side].swap_remove(index);
                        engines[side].retire(pages).unwrap();
                    }
                }
                _ => pools[side].serve_boundary().unwrap(),
            }
            check_between(&table, pools);
        }
        assert!(
            served > 100 && pending > 10,
            "the churn met pressure: {served} served, {pending} waited"
        );
        let moves = table.lock().unwrap().boundary;
        assert!((3..=11).contains(&moves));
        for side in 0..2 {
            for pages in held[side].drain(..) {
                engines[side].retire(pages).unwrap();
            }
        }
        settle(&a, &b);
        check(&table, pools);
    }
}
