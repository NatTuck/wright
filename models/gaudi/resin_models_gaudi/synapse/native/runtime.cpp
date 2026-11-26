// Thin C API over Synapse (runtime) for Gaudi2, called from Python through ctypes and from resin's native step
// programs. Resin plans every step on the host. Graph building and compiling are in graph.cpp.
//
// One device per process; multi-card serving runs one process per card. Work issues on a context: a stream on
// its own queue slot, with its own workspace and pinned staging buffers. Each thread issues on its current
// context (rg_context_use); a thread starts on context 0, which rg_init creates. A process that serves several
// models gives each model its own context.
#include "graph.h"
#include <hccl.h>
#include <synapse_common_types.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unistd.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

thread_local char g_err[1024];

struct Recipe {
    synRecipeHandle h = nullptr;
    uint64_t workspace_size = 0;
    std::vector<synLaunchTensorInfoExt> binds;   // one per persistent tensor
    std::vector<std::string> names;
    std::unordered_map<std::string, int> index;
    std::string graph_name;                      // rg_recipe_name: every recipe must have one
};

static const int kRing = 32;              // pinned slots for small non-blocking uploads (rg_h2d_nb)
static const size_t kSlot = 64 << 10;
static const size_t kStage = 64ull << 20; // pinned staging buffer of rg_h2d and rg_d2h

struct Context {
    synStreamHandle stream = nullptr;
    uint64_t affinity = 0;                // the stream's queue slot (a mask)
    uint64_t ws = 0, ws_size = 0;         // workspace of the recipes launched on this context
    void* stage = nullptr;
    synEventHandle h2d_ev[2] = {nullptr, nullptr};  // rg_h2d staging halves
    char* ring = nullptr;
    synEventHandle ring_ev[kRing] = {};
    bool ring_used[kRing] = {};
    int ring_i = 0;
    hcclComm_t comm = nullptr;            // this context's communicator (rg_ccl_init), else the process default
};

static synDeviceId g_dev;
static bool g_init = false;
static int g_card = -1;
static unsigned g_module = 0;
static const int kContexts = 8;
static std::atomic<Context*> g_contexts[kContexts] = {};
static int g_context_count = 0;
static thread_local int t_context = 0;
// Fixed storage: recipes load on some threads while others launch.
static const int kRecipes = 1 << 16;
static Recipe* g_recipes[kRecipes] = {};
static int g_recipe_count = 0;
static const int kEvents = 64;
static synEventHandle g_ev[kEvents] = {};       // rg_event_* slots
static int g_ev_context[kEvents] = {};          // the context that recorded each slot last
static bool g_ev_timed[kEvents] = {};           // slots whose events collect device time
// Guards the recipe and context tables and the workspaces.
static std::mutex g_mu;
// Device allocations: the address rg_malloc returned -> the raw address.
static std::unordered_map<uint64_t, uint64_t> g_alloc;
static std::mutex g_alloc_mu;

static Context* current() {
    Context* c = g_contexts[t_context].load(std::memory_order_relaxed);
    if (!c) snprintf(g_err, sizeof(g_err), "no Gaudi device is open");
    return c;
}
#define CTX(c) Context* c = current(); if (!c) return -1

RG_API const char* rg_last_error() { return g_err; }

static std::string read_sys(int card, const char* f) {
    std::string p = "/sys/class/accel/accel" + std::to_string(card) + "/device/" + f;
    FILE* fp = fopen(p.c_str(), "r");
    if (!fp) return "";
    char buf[128] = {0};
    if (!fgets(buf, sizeof(buf), fp)) buf[0] = 0;
    fclose(fp);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// A new stream on queue slot `affinity` (0: the slot Synapse picks).
static Context* new_context(uint64_t affinity) {
    auto* c = new Context();
    synStatus s = synStreamCreateGeneric(&c->stream, g_dev, 0);
    if (s != synSuccess) { delete c; fail(s, "synStreamCreateGeneric"); return nullptr; }
    const char* what = "synStreamSetAffinity";
    if (affinity) s = synStreamSetAffinity(g_dev, c->stream, affinity);
    if (s == synSuccess) { what = "synStreamGetAffinity"; s = synStreamGetAffinity(g_dev, c->stream, &c->affinity); }
    if (s != synSuccess) { synStreamDestroy(c->stream); delete c; fail(s, what); return nullptr; }
    return c;
}

static void destroy_context(Context* c) {
    synStreamSynchronize(c->stream);
    if (c->ws) synDeviceFree(g_dev, c->ws, 0);
    if (c->stage) synHostFree(g_dev, c->stage, 0);
    if (c->ring) synHostFree(g_dev, c->ring, 0);
    for (auto& e : c->h2d_ev) if (e) synEventDestroy(e);
    for (auto& e : c->ring_ev) if (e) synEventDestroy(e);
    synStreamDestroy(c->stream);
    delete c;
}

// Acquire accel `card` (its /sys/class/accel index). A busy card returns -1.
RG_API int rg_init(int card) {
    if (g_init) return g_card;
    CHK(synInitialize());
    if (read_sys(card, "status") != "Operational") {
        synDestroy();
        snprintf(g_err, sizeof(g_err), "Gaudi card %d is not operational", card);
        return -1;
    }
    unsigned module = unsigned(std::stoul(read_sys(card, "module_id")));
    if (synDeviceAcquireByModuleId(&g_dev, module) != synSuccess) {
        synDestroy();
        snprintf(g_err, sizeof(g_err), "Gaudi card %d is busy", card);
        return -1;
    }
    g_module = module;
    Context* c = new_context(0);
    if (!c) return -1;
    CHK(synHostMalloc(g_dev, kStage, 0, &c->stage));
    g_contexts[0].store(c);
    g_context_count = 1;
    g_card = card;
    g_init = true;
    return card;
}

RG_API int rg_module_id() { return g_init ? int(g_module) : -1; }

RG_API int rg_driver_version(char* out, int size) {
    CHK(synDriverGetVersion(out, unsigned(size)));
    return 0;
}

// ---------------------------------------------------------------- contexts
// A new context on queue slot `affinity` (a mask; 0: the lowest slot that no context uses). Returns its id.
RG_API int rg_context_create(uint64_t affinity) {
    if (!g_init) { snprintf(g_err, sizeof(g_err), "no Gaudi device is open"); return -1; }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_context_count == kContexts) { snprintf(g_err, sizeof(g_err), "context: all %d are in use", kContexts); return -1; }
    if (!affinity) {
        uint64_t range = 0, used = 0;
        CHK(synDeviceGetAffinityMaskRange(g_dev, &range));
        for (int i = 0; i < g_context_count; i++) used |= g_contexts[i].load()->affinity;
        const uint64_t free = range & ~used;
        if (!free) {
            snprintf(g_err, sizeof(g_err), "context: every queue slot (mask %#lx) has a stream", (unsigned long)range);
            return -1;
        }
        affinity = free & (~free + 1);
    }
    Context* c = new_context(affinity);
    if (!c) return -1;
    g_contexts[g_context_count].store(c);
    return g_context_count++;
}

// Make context `id` this thread's current context; returns the one it replaces, or -1.
RG_API int rg_context_use(int id) {
    if (id < 0 || id >= kContexts || !g_contexts[id].load()) { snprintf(g_err, sizeof(g_err), "context: no context %d", id); return -1; }
    const int previous = t_context;
    t_context = id;
    return previous;
}

RG_API int rg_context_current() { return t_context; }

// The queue slot (mask) of context `id`, or 0.
RG_API uint64_t rg_context_affinity(int id) {
    if (id < 0 || id >= kContexts) return 0;
    Context* c = g_contexts[id].load();
    return c ? c->affinity : 0;
}

// Wait for the current context's work.
RG_API int rg_sync() { CTX(c); CHK(synStreamSynchronize(c->stream)); return 0; }

// Wait for the work of every context.
RG_API int rg_sync_all() {
    for (auto& slot : g_contexts) if (Context* c = slot.load()) CHK(synStreamSynchronize(c->stream));
    return 0;
}

RG_API int rg_shutdown() {
    if (!g_init) return 0;
    rg_sync_all();
    for (int i = 0; i < g_recipe_count; i++) { synRecipeDestroy(g_recipes[i]->h); delete g_recipes[i]; g_recipes[i] = nullptr; }
    g_recipe_count = 0;
    rg_destroy_graphs();
    for (auto& kv : g_alloc) synDeviceFree(g_dev, kv.second, 0);
    g_alloc.clear();
    for (auto& e : g_ev) { if (e) synEventDestroy(e); e = nullptr; }
    for (auto& t : g_ev_timed) t = false;
    for (auto& slot : g_contexts) if (Context* c = slot.exchange(nullptr)) destroy_context(c);
    g_context_count = 0;
    synDeviceRelease(g_dev);
    synDestroy();
    g_card = -1;
    g_init = false;
    return 0;
}

// ---------------------------------------------------------------- memory
RG_API uint64_t rg_malloc(uint64_t bytes) {
    uint64_t raw = 0;
    synStatus s = synDeviceMalloc(g_dev, (bytes ? bytes : 256) + 256, 0, 0, &raw);
    if (s != synSuccess) { fail(s, "synDeviceMalloc"); return 0; }
    // 256-byte alignment: recipe sections bound at these addresses are validated at 256 (a 128-aligned
    // section fails synLaunchWithExternalEventsExt status=30 at the larger replay-kept group counts).
    const uint64_t aligned = (raw + 255) & ~uint64_t(255);
    std::lock_guard<std::mutex> lock(g_alloc_mu);
    g_alloc[aligned] = raw;
    return aligned;
}
RG_API int rg_free(uint64_t addr) {
    uint64_t raw;
    {
        std::lock_guard<std::mutex> lock(g_alloc_mu);
        auto it = g_alloc.find(addr);
        if (it == g_alloc.end()) { snprintf(g_err, sizeof(g_err), "free: %#lx was not allocated here", (unsigned long)addr); return -1; }
        raw = it->second;
        g_alloc.erase(it);
    }
    CHK(synDeviceFree(g_dev, raw, 0));
    return 0;
}

RG_API int rg_mem_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    CHK(synDeviceGetMemoryInfo(g_dev, free_bytes, total_bytes));
    return 0;
}

RG_API void* rg_host_malloc(uint64_t bytes) {
    void* p = nullptr;
    synStatus s = synHostMalloc(g_dev, bytes, 0, &p);
    if (s != synSuccess) { fail(s, "synHostMalloc"); return nullptr; }
    return p;
}
RG_API int rg_host_free(void* p) { CHK(synHostFree(g_dev, p, 0)); return 0; }

static int stage(Context* c) {
    if (!c->stage) CHK(synHostMalloc(g_dev, kStage, 0, &c->stage));
    return 0;
}

// Synchronous host->device copy.
RG_API int rg_h2d(uint64_t dst, const void* src, uint64_t bytes) {
    CTX(c);
    if (stage(c)) return -1;
    const char* s = (const char*)src;
    const size_t half = kStage / 2;
    if (!c->h2d_ev[0]) for (auto& e : c->h2d_ev) CHK(synEventCreate(&e, g_dev, 0));
    bool pending[2] = {false, false};
    for (int b = 0; bytes; b ^= 1) {
        size_t n = bytes < half ? bytes : half;
        char* buf = (char*)c->stage + b * half;
        if (pending[b]) CHK(synEventSynchronize(c->h2d_ev[b]));
        memcpy(buf, s, n);
        CHK(synMemCopyAsync(c->stream, (uint64_t)buf, n, dst, HOST_TO_DRAM));
        CHK(synEventRecord(c->h2d_ev[b], c->stream));
        pending[b] = true;
        s += n; dst += n; bytes -= n;
    }
    CHK(synStreamSynchronize(c->stream));
    return 0;
}
// Small non-blocking uploads (per-launch tokens / metadata): a ring of pinned slots.
RG_API int rg_h2d_nb(uint64_t dst, const void* src, uint64_t bytes) {
    if (bytes > kSlot) return rg_h2d(dst, src, bytes);
    CTX(c);
    if (!c->ring) {
        CHK(synHostMalloc(g_dev, kRing * kSlot, 0, (void**)&c->ring));
        for (auto& e : c->ring_ev) CHK(synEventCreate(&e, g_dev, 0));
    }
    const int i = c->ring_i;
    c->ring_i = (c->ring_i + 1) % kRing;
    if (c->ring_used[i]) CHK(synEventSynchronize(c->ring_ev[i]));
    memcpy(c->ring + i * kSlot, src, bytes);
    CHK(synMemCopyAsync(c->stream, (uint64_t)(c->ring + i * kSlot), bytes, dst, HOST_TO_DRAM));
    CHK(synEventRecord(c->ring_ev[i], c->stream));
    c->ring_used[i] = true;
    return 0;
}

RG_API int rg_d2h(void* dst, uint64_t src, uint64_t bytes) {
    CTX(c);
    if (stage(c)) return -1;
    char* d = (char*)dst;
    while (bytes) {
        size_t n = bytes < kStage ? bytes : kStage;
        CHK(synMemCopyAsync(c->stream, src, n, (uint64_t)c->stage, DRAM_TO_HOST));
        CHK(synStreamSynchronize(c->stream));
        memcpy(d, c->stage, n);
        d += n; src += n; bytes -= n;
    }
    return 0;
}
// Async copies between pinned host memory and device (the caller keeps pinned memory alive).
RG_API int rg_h2d_async(uint64_t dst, const void* pinned, uint64_t bytes) {
    CTX(c); CHK(synMemCopyAsync(c->stream, (uint64_t)pinned, bytes, dst, HOST_TO_DRAM)); return 0;
}
RG_API int rg_d2h_async(void* pinned, uint64_t src, uint64_t bytes) {
    CTX(c); CHK(synMemCopyAsync(c->stream, src, bytes, (uint64_t)pinned, DRAM_TO_HOST)); return 0;
}
RG_API int rg_d2d_async(uint64_t dst, uint64_t src, uint64_t bytes) {
    CTX(c); CHK(synMemCopyAsync(c->stream, src, bytes, dst, DRAM_TO_DRAM)); return 0;
}
// Batched device-to-device copies.
RG_API int rg_d2d_multi(const uint64_t* src, const uint64_t* size, const uint64_t* dst, uint64_t n) {
    if (!n) return 0;
    CTX(c);
    CHK(synMemCopyAsyncMultiple(c->stream, src, size, dst, DRAM_TO_DRAM, n));
    return 0;
}

RG_API int rg_memset32(uint64_t dst, uint32_t value, uint64_t count) {
    CTX(c); CHK(synMemsetD32Async(dst, value, count, c->stream)); return 0;
}

// ---------------------------------------------------------------- events
// A small pool of reusable stream events (g_ev), for completions that wait on one step only. A slot records
// the work issued so far on the current context.
RG_API int rg_event_record(int slot) {
    if (slot < 0 || slot >= kEvents) { snprintf(g_err, sizeof(g_err), "event: bad slot %d", slot); return -1; }
    CTX(c);
    if (!g_ev[slot]) CHK(synEventCreate(&g_ev[slot], g_dev, g_ev_timed[slot] ? EVENT_COLLECT_TIME : 0));
    CHK(synEventRecord(g_ev[slot], c->stream));
    g_ev_context[slot] = t_context;
    return 0;
}
RG_API int rg_event_wait(int slot) {
    if (slot < 0 || slot >= kEvents || !g_ev[slot]) { snprintf(g_err, sizeof(g_err), "event: slot %d not recorded", slot); return -1; }
    CHK(synEventSynchronize(g_ev[slot]));
    return 0;
}
RG_API int64_t rg_now_ns();

// Wait for what slot `slot` recorded: poll it for up to `spin_ns`, then block. Returns 0, or -1 on an error.
RG_API int rg_event_spin(int slot, int64_t spin_ns) {
    if (slot < 0 || slot >= kEvents || !g_ev[slot]) { snprintf(g_err, sizeof(g_err), "event: slot %d not recorded", slot); return -1; }
    const int64_t deadline = rg_now_ns() + spin_ns;
    for (;;) {
        synStatus s = synEventQuery(g_ev[slot]);
        if (s == synSuccess) return 0;
        if (s != synBusy) { fail(s, "synEventQuery"); return -1; }
        if (rg_now_ns() >= deadline) break;
    }
    CHK(synEventSynchronize(g_ev[slot]));
    return 0;
}
// 1 when complete, 0 when pending.
RG_API int rg_event_query(int slot) {
    if (slot < 0 || slot >= kEvents || !g_ev[slot]) { snprintf(g_err, sizeof(g_err), "event: slot %d not recorded", slot); return -1; }
    synStatus s = synEventQuery(g_ev[slot]);
    if (s == synSuccess) return 1;
    if (s == synBusy) return 0;
    fail(s, "synEventQuery");
    return -1;
}
// Event slot `slot` collects device time for rg_event_elapsed; before its first record.
RG_API int rg_event_timed(int slot) {
    if (slot < 0 || slot >= kEvents || g_ev[slot]) { snprintf(g_err, sizeof(g_err), "event: slot %d is in use", slot); return -1; }
    g_ev_timed[slot] = true;
    return 0;
}
// Device nanoseconds from what timed slot `start` recorded to what timed slot `end` recorded; -2 while either is
// pending or its time is gone, -1 on an error.
RG_API int64_t rg_event_elapsed(int start, int end) {
    for (int slot : {start, end})
        if (slot < 0 || slot >= kEvents || !g_ev[slot] || !g_ev_timed[slot]) {
            snprintf(g_err, sizeof(g_err), "event: slot %d is not a recorded timed slot", slot);
            return -1;
        }
    uint64_t ns = 0;
    synStatus s = synEventElapsedTime(&ns, g_ev[start], g_ev[end]);
    if (s == synBusy || s == synUnavailable) return -2;
    if (s != synSuccess) { fail(s, "synEventElapsedTime"); return -1; }
    // An end that reads earlier than its start has no usable time either.
    return ns > uint64_t(INT64_MAX) ? -2 : int64_t(ns);
}
// The current context's later work waits on the device (not on the host) for what event `slot` recorded. A slot
// that the current context recorded needs no wait.
RG_API int rg_event_stream_wait(int slot) {
    if (slot < 0 || slot >= kEvents || !g_ev[slot]) { snprintf(g_err, sizeof(g_err), "event: slot %d not recorded", slot); return -1; }
    CTX(c);
    if (g_ev_context[slot] == t_context) return 0;
    CHK(synStreamWaitEvent(c->stream, g_ev[slot], 0));
    return 0;
}

// ---------------------------------------------------------------- recipes
// Grow the workspace of context `c` to `size` bytes; the caller holds g_mu.
static int grow_workspace(Context* c, uint64_t size) {
    if (size <= c->ws_size) return 0;
    CHK(synStreamSynchronize(c->stream));
    if (c->ws) { CHK(synDeviceFree(g_dev, c->ws, 0)); c->ws = c->ws_size = 0; }
    CHK(synDeviceMalloc(g_dev, size, 0, 0, &c->ws));
    c->ws_size = size;
    return 0;
}

// The current context's workspace grows to fit the recipe here, while the model loads, so that the cache sizing
// that follows sees the memory it takes.
int rg_register_recipe(synRecipeHandle h, const std::vector<std::string>& persistent) {
    CTX(c);
    auto* r = new Recipe();
    r->h = h;
    r->names = persistent;
    std::vector<const char*> nm;
    for (auto& n : r->names) nm.push_back(n.c_str());
    std::vector<uint64_t> ids(nm.size());
    synStatus s = synWorkspaceGetSize(&r->workspace_size, r->h);
    if (s != synSuccess) { delete r; fail(s, "synWorkspaceGetSize"); return -1; }
    if (!nm.empty()) s = synTensorRetrieveIds(r->h, nm.data(), ids.data(), nm.size());
    if (s != synSuccess) { delete r; fail(s, "synTensorRetrieveIds"); return -1; }
    for (size_t i = 0; i < nm.size(); i++) {
        synLaunchTensorInfoExt b;
        memset(&b, 0, sizeof(b));
        b.tensorName = r->names[i].c_str();
        b.tensorType = DATA_TENSOR;
        b.tensorId = ids[i];
        r->binds.push_back(b);
        r->index[r->names[i]] = int(i);
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_recipe_count == kRecipes) { delete r; snprintf(g_err, sizeof(g_err), "recipes: all %d are in use", kRecipes); return -1; }
    if (grow_workspace(c, r->workspace_size)) { delete r; return -1; }
    g_recipes[g_recipe_count] = r;
    return g_recipe_count++;
}

RG_API uint64_t rg_recipe_workspace(int ri) { return g_recipes[ri]->workspace_size; }

// ---------------------------------------------------------------- recipe names
// Every recipe must have a name before it is launched: `rg_launch` aborts otherwise, so a timeline or a
// profile can never hold an anonymous launch. The name is the graph's (``RecipeCache.compile`` sets it as
// soon as the index exists, for a fresh compile and for one loaded from the disk cache alike).
RG_API int rg_recipe_name(int ri, const char* name) {
    if (ri < 0 || ri >= g_recipe_count || !g_recipes[ri]) {
        snprintf(g_err, sizeof(g_err), "name: recipe %d does not exist", ri);
        return -1;
    }
    if (!name || !*name) {
        snprintf(g_err, sizeof(g_err), "name: recipe %d given an empty name", ri);
        return -1;
    }
    g_recipes[ri]->graph_name = name;
    return 0;
}

// ---------------------------------------------------------------- launch log
// ``RESIN_LAUNCH_LOG=<dir>``: one line per launch, ``<ns> <ticks> <recipe> <name>``, from a preallocated
// buffer dumped at exit. Per launch this is a TSC read and one relaxed store — no sync, no device query, no
// allocation, no lock — so a logged run keeps its overlap and its timing (unlike ``RESIN_STEP_PROF``, which
// syncs after every launch). ``RESIN_LAUNCH_LOG_CAPACITY`` sets the entry count (default 1<<22, ~48 MB);
// entries past it are counted and dropped, never written out of bounds.
struct LaunchEntry {
    uint64_t ticks;
    uint32_t ri;
};

static std::vector<LaunchEntry> g_launch_log;
static std::atomic<size_t> g_launch_written{0};
static std::atomic<int> g_launch_state{0};  // 0: not tried, 1: off, 2: on
static double g_launch_ns_per_tick = 1.0;
static uint64_t g_launch_origin = 0;
static char g_launch_dir[4096];

static inline uint64_t launch_ticks() {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
#endif
}

static void launch_log_dump() {
    if (g_launch_state.load(std::memory_order_acquire) != 2) return;
    const size_t written = g_launch_written.load(std::memory_order_relaxed);
    const size_t n = written < g_launch_log.size() ? written : g_launch_log.size();
    char path[4200];
    snprintf(path, sizeof(path), "%s/launches.%d.tsv", g_launch_dir, int(getpid()));
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[launch-log] cannot write %s\n", path);
        return;
    }
    fprintf(f, "# ns\tticks\trecipe\tname\n");
    for (size_t i = 0; i < n; i++) {
        const LaunchEntry& e = g_launch_log[i];
        const char* name = "";
        if (e.ri < uint32_t(g_recipe_count) && g_recipes[e.ri]) name = g_recipes[e.ri]->graph_name.c_str();
        fprintf(f, "%.0f\t%llu\t%u\t%s\n", double(e.ticks - g_launch_origin) * g_launch_ns_per_tick,
                (unsigned long long)e.ticks, e.ri, name);
    }
    fclose(f);
    fprintf(stderr, "[launch-log] %zu launches (%zu requested) -> %s\n", n, written, path);
}

// Start the log if the environment asks for it; idempotent. Called from Python once the model has loaded, so
// the TSC calibration is not charged to a step, and lazily from `rg_launch` if it was not.
RG_API int rg_launch_log_start() {
    int state = g_launch_state.load(std::memory_order_acquire);
    if (state != 0) return state == 2 ? 0 : 1;
    const char* dir = getenv("RESIN_LAUNCH_LOG");
    if (!dir || !*dir) {
        g_launch_state.store(1, std::memory_order_release);
        return 1;
    }
    snprintf(g_launch_dir, sizeof(g_launch_dir), "%s", dir);
    const char* cap = getenv("RESIN_LAUNCH_LOG_CAPACITY");
    size_t want = (cap && *cap) ? size_t(strtoull(cap, nullptr, 10)) : (size_t(1) << 22);
    if (want < 1024) want = 1024;
    g_launch_log.assign(want, LaunchEntry{0, 0});
    // ticks -> ns, from two clock samples around a short sleep; the log is relative, so only the slope matters
    timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    uint64_t t0 = launch_ticks();
    usleep(5000);
    clock_gettime(CLOCK_MONOTONIC, &b);
    uint64_t t1 = launch_ticks();
    const long long ns = (long long)(b.tv_sec - a.tv_sec) * 1000000000LL + (long long)(b.tv_nsec - a.tv_nsec);
    if (t1 > t0 && ns > 0) g_launch_ns_per_tick = double(ns) / double(t1 - t0);
    g_launch_origin = launch_ticks();
    atexit(launch_log_dump);
    g_launch_state.store(2, std::memory_order_release);
    return 0;
}

// Append one launch. A hot-path branch on a load of `g_launch_state` (off in production) plus, when on, one
// relaxed fetch_add and one 16-byte store.
static inline void launch_log_note(int ri) {
    int state = g_launch_state.load(std::memory_order_relaxed);
    if (state == 0) {
        rg_launch_log_start();
        state = g_launch_state.load(std::memory_order_relaxed);
    }
    if (state != 2) return;
    const size_t i = g_launch_written.fetch_add(1, std::memory_order_relaxed);
    if (i < g_launch_log.size()) g_launch_log[i] = LaunchEntry{launch_ticks(), uint32_t(ri)};
}

// Slot of a persistent tensor name in recipe `ri`, or -1.
RG_API int rg_slot(int ri, const char* name) {
    Recipe* r = g_recipes[ri];
    auto it = r->index.find(name);
    if (it == r->index.end()) { snprintf(g_err, sizeof(g_err), "bind: tensor %s not in recipe", name); return -1; }
    return it->second;
}
RG_API int rg_bind_slots(int ri, const int* slots, const uint64_t* addrs, int n) {
    auto& b = g_recipes[ri]->binds;
    for (int i = 0; i < n; i++) b[slots[i]].pTensorAddress = addrs[i];
    return 0;
}

RG_API int rg_launch(int ri) {
    CTX(c);
    Recipe* r = g_recipes[ri];
    if (r->graph_name.empty()) {
        // No launch may be anonymous: a timeline, a profile or a rate that names nothing cannot be acted on.
        fprintf(stderr,
                "FATAL: launching recipe %d, which has no name; rg_recipe_name was never called for it "
                "(RecipeCache.compile sets it for every recipe, compiled or loaded)\n",
                ri);
        fflush(stderr);
        abort();
    }
    for (auto& b : r->binds) if (!b.pTensorAddress) { snprintf(g_err, sizeof(g_err), "launch: %s unbound", b.tensorName); return -1; }
    if (r->workspace_size > c->ws_size) {
        // a recipe that loaded on another context
        std::lock_guard<std::mutex> lock(g_mu);
        if (grow_workspace(c, r->workspace_size)) return -1;
    }
    launch_log_note(ri);
    synStatus ls = synLaunchWithExternalEventsExt(c->stream, r->binds.data(), r->binds.size(), c->ws, r->h, nullptr, 0, 0);
    if (ls != synSuccess) {
        // Name every bound section and its alignment so a sections-validation failure (status=30) identifies
        // the culprit instead of a bare status code. Diagnostic only; does not change the launch.
        fprintf(stderr, "RG_LAUNCH_FAIL recipe=%d status=%d nbinds=%zu\n", ri, int(ls), r->binds.size());
        uint64_t attrs[2] = {0, 0};
        synRecipeAttribute qa[2] = {RECIPE_ATTRIBUTE_PERSISTENT_TENSORS_SIZE, RECIPE_ATTRIBUTE_DEVICE_MEMORY_SIZE};
        synRecipeGetAttribute(attrs, qa, 2, r->h);
        fprintf(stderr, "  persistent_tensors_size=%llu device_memory_size=%llu workspace=%llu\n",
                (unsigned long long)attrs[0], (unsigned long long)attrs[1], (unsigned long long)r->workspace_size);
        for (auto& b : r->binds) {
            fprintf(stderr, "  section %-22s addr=0x%012llx mod128=%3llu mod256=%3llu tsize=[%llu,%llu,%llu,%llu]\n",
                    b.tensorName, (unsigned long long)b.pTensorAddress,
                    (unsigned long long)(b.pTensorAddress % 128), (unsigned long long)(b.pTensorAddress % 256),
                    (unsigned long long)b.tensorSize[0], (unsigned long long)b.tensorSize[1],
                    (unsigned long long)b.tensorSize[2], (unsigned long long)b.tensorSize[3]);
        }
        // Per-tensor declared max dims + section id + offset: identifies a section whose declared size exceeds
        // its bound buffer, or a bad section grouping/offset (the cause of a sections-validation failure).
        {
            std::vector<synRetrievedLaunchTensorInfoExt> infos(r->binds.size());
            memset(infos.data(), 0, infos.size() * sizeof(infos[0]));
            for (size_t i = 0; i < r->binds.size(); i++) infos[i].tensorId = r->binds[i].tensorId;
            synStatus qs = synTensorRetrieveLaunchInfoByIdExt(r->h, (uint32_t)infos.size(), infos.data());
            fprintf(stderr, "  launchinfo_query_status=%d\n", int(qs));
            for (size_t i = 0; i < infos.size(); i++) {
                auto& ti = infos[i];
                fprintf(stderr, "  tensor %-22s dtype=%d dims=%u max=[%llu,%llu,%llu,%llu,%llu] sect=%u off=%llu addr=0x%012llx\n",
                        ti.tensorName, int(ti.tensorDataType), ti.tensorDims,
                        (unsigned long long)ti.tensorMaxSize[0], (unsigned long long)ti.tensorMaxSize[1],
                        (unsigned long long)ti.tensorMaxSize[2], (unsigned long long)ti.tensorMaxSize[3],
                        (unsigned long long)ti.tensorMaxSize[4],
                        ti.tensorSectionId, (unsigned long long)ti.tensorOffsetInSection,
                        (unsigned long long)r->binds[i].pTensorAddress);
            }
        }
        fflush(stderr);
        fail(ls, "synLaunchWithExternalEventsExt");
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------- profiling
RG_API int rg_prof_start() {
    uint32_t bytes = 0;
    synStatus s = synProfilerQueryRequiredMemory(g_dev, &bytes);
    if (s != synSuccess && s != synUnsupported) { fail(s, "synProfilerQueryRequiredMemory"); return -1; }
    if (bytes) {
        uint64_t buf = 0;
        CHK(synDeviceMalloc(g_dev, bytes, 0, 0, &buf));
        CHK(synProfilerSetUserBuffer(g_dev, (void*)buf));
    }
    CHK(synProfilerStart(synTraceAll, g_dev));
    return 0;
}
RG_API int rg_prof_stop(const char* tsv_path) {
    if (rg_sync_all()) return -1;
    CHK(synProfilerStop(synTraceAll, g_dev));
    size_t bytes = 0, entries = 0;
    CHK(synProfilerGetTrace(synTraceAll, g_dev, synTraceFormatTEF, nullptr, &bytes, &entries));
    std::vector<unsigned char> buf(bytes);
    CHK(synProfilerGetTrace(synTraceAll, g_dev, synTraceFormatTEF, buf.data(), &bytes, &entries));
    FILE* f = fopen(tsv_path, "w");
    if (!f) { snprintf(g_err, sizeof(g_err), "prof_stop: cannot write %s", tsv_path); return -1; }
    fprintf(f, "ph\tengine_type\tengine_index\tts\tdur\tname\tcategory\tmeta\n");
    auto* ev = reinterpret_cast<synTraceEvent*>(buf.data());
    for (size_t i = 0; i < entries; i++) {
        auto& e = ev[i];
        fprintf(f, "%c\t%u\t%u\t%.3Lf\t%.3Lf\t%s\t%s\t%s\n", e.type ? e.type : '?', e.engineType, e.engineIndex,
                e.timestamp, e.duration, e.name ? e.name : "", e.category ? e.category : "",
                (e.type == 'M' && e.arguments.name) ? e.arguments.name : "");
    }
    fclose(f);
    return int(entries);
}

RG_API int64_t rg_now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

// ---------------------------------------------------------------- collectives (HCCL)
// rg_ccl_init makes a communicator for the current context. Collectives issue on the current context with its
// communicator, or with the first communicator of the process (the default) when the context has none, so a
// process with one communicator uses it on every context. A second model of the process (a replica that spans
// cards) makes its own on its own context. Ranks exchange the unique id out of band.
static hcclComm_t g_comm = nullptr;  // the process default: the first communicator made

static hcclComm_t comm_of(Context* c) {
    hcclComm_t comm = c->comm ? c->comm : g_comm;
    if (!comm) snprintf(g_err, sizeof(g_err), "no HCCL communicator on context %d", t_context);
    return comm;
}

RG_API int rg_ccl_id_size() { return int(sizeof(hcclUniqueId)); }

RG_API int rg_ccl_unique_id(void* out) {
    hcclUniqueId id;
    hcclResult_t r = hcclGetUniqueId(&id);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclGetUniqueId failed: %d", int(r)); return -1; }
    memcpy(out, &id, sizeof(id));
    return 0;
}

RG_API int rg_ccl_init(const void* id_bytes, int nranks, int rank) {
    CTX(c);
    if (c->comm) { snprintf(g_err, sizeof(g_err), "context %d has a communicator", t_context); return -1; }
    hcclUniqueId id;
    memcpy(&id, id_bytes, sizeof(id));
    hcclComm_t comm = nullptr;
    hcclResult_t r = hcclCommInitRank(&comm, nranks, id, rank);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclCommInitRank failed: %d", int(r)); return -1; }
    c->comm = comm;
    if (!g_comm) g_comm = comm;
    return 0;
}

// dtype: 0 = f32, 1 = bf16. In place when send == recv.
RG_API int rg_ccl_allreduce(uint64_t send, uint64_t recv, uint64_t count, int dtype) {
    CTX(c);
    hcclComm_t comm = comm_of(c);
    if (!comm) return -1;
    hcclResult_t r = hcclAllReduce((const void*)send, (void*)recv, count, dtype ? hcclBfloat16 : hcclFloat32,
                                   hcclSum, comm, c->stream);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclAllReduce failed: %d", int(r)); return -1; }
    return 0;
}

RG_API int rg_ccl_allgather(uint64_t send, uint64_t recv, uint64_t count, int dtype) {
    CTX(c);
    hcclComm_t comm = comm_of(c);
    if (!comm) return -1;
    hcclResult_t r = hcclAllGather((const void*)send, (void*)recv, count, dtype ? hcclBfloat16 : hcclFloat32,
                                   comm, c->stream);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclAllGather failed: %d", int(r)); return -1; }
    return 0;
}

// Destroy every communicator of the process.
RG_API int rg_ccl_destroy() {
    for (auto& slot : g_contexts)
        if (Context* c = slot.load(); c && c->comm) { hcclCommDestroy(c->comm); c->comm = nullptr; }
    g_comm = nullptr;
    return 0;
}
