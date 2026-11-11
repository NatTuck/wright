// Thin C API over Synapse (graph compiler + runtime) for Gaudi2, called from Python
// through ctypes. Resin plans every step on the host.
//
// One device and one stream per process. Multi-card serving runs one process per card.
#include <synapse_api.h>
#include <hccl.h>
#include <synapse_common_types.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <unistd.h>
#include <time.h>

#define RG_API extern "C" __attribute__((visibility("default")))

// Per thread: graphs may compile concurrently, and the caller reads the error on its own thread.
static thread_local char g_err[1024];
static bool fail(synStatus s, const char* what) {
    char desc[STATUS_DESCRIPTION_MAX_SIZE] = {0};
    synStatusGetBriefDescription(s, desc, sizeof(desc));
    snprintf(g_err, sizeof(g_err), "%s failed: status=%d (%s)", what, int(s), desc);
    return false;
}
#define CHK(call) do { synStatus _s = (call); if (_s != synSuccess) { fail(_s, #call); return -1; } } while (0)

struct Graph {
    synGraphHandle h = nullptr;
    std::vector<synTensor> tensors;
    std::vector<synSectionHandle> sections;
    std::vector<std::string> names;      // stable storage for tensor names
    std::vector<bool> persistent;
    std::vector<synNodeId> nodes;
};

struct Recipe {
    synRecipeHandle h = nullptr;
    uint64_t workspace_size = 0;
    std::vector<synLaunchTensorInfoExt> binds;   // one per persistent tensor
    std::vector<std::string> names;
    std::unordered_map<std::string, int> index;
};

static synDeviceId g_dev;
static synStreamHandle g_stream = nullptr;
static bool g_init = false;
static int g_card = -1;
static unsigned g_module = 0;
static std::vector<Graph*> g_graphs;
static std::vector<Recipe*> g_recipes;
static void* g_stage = nullptr;           // pinned staging buffer
static size_t g_stage_size = 0;
// ring of pinned slots for small non-blocking uploads (rg_h2d_nb)
static const int kRing = 32;
static const size_t kSlot = 64 << 10;
static char* g_ring = nullptr;
static synEventHandle g_ring_ev[kRing] = {};
static bool g_ring_used[kRing] = {};
static int g_ring_i = 0;
static synEventHandle g_h2d_ev[2] = {nullptr, nullptr};  // rg_h2d staging halves
static const int kEvents = 64;
static synEventHandle g_ev[kEvents] = {};       // rg_event_* slots
// Workspace shared by all recipes.
static uint64_t g_ws = 0, g_ws_size = 0;
// Graph compilations in this process (deserialized recipes do not count).
static std::atomic<uint64_t> g_compiles{0};
// Guards the graph and recipe tables and the shared workspace.
static std::mutex g_mu;
// Device allocations: the address rg_malloc returned -> the raw address.
static std::unordered_map<uint64_t, uint64_t> g_alloc;

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
    int chosen = card;
    CHK(synStreamCreateGeneric(&g_stream, g_dev, 0));
    g_stage_size = 64ull << 20;
    CHK(synHostMalloc(g_dev, g_stage_size, 0, &g_stage));
    g_card = chosen;
    g_init = true;
    return chosen;
}

RG_API int rg_module_id() { return g_init ? int(g_module) : -1; }

RG_API int rg_driver_version(char* out, int size) {
    CHK(synDriverGetVersion(out, unsigned(size)));
    return 0;
}

RG_API int rg_sync() { CHK(synStreamSynchronize(g_stream)); return 0; }

RG_API int rg_shutdown() {
    if (!g_init) return 0;
    synStreamSynchronize(g_stream);
    for (auto* r : g_recipes) if (r) { synRecipeDestroy(r->h); delete r; }
    g_recipes.clear();
    if (g_ws) synDeviceFree(g_dev, g_ws, 0);
    g_ws = g_ws_size = 0;
    for (auto* g : g_graphs) if (g) {
        for (auto t : g->tensors) synTensorDestroy(t);
        for (auto s : g->sections) synSectionDestroy(s);
        synGraphDestroy(g->h);
        delete g;
    }
    g_graphs.clear();
    for (auto& kv : g_alloc) synDeviceFree(g_dev, kv.second, 0);
    g_alloc.clear();
    if (g_stage) { synHostFree(g_dev, g_stage, 0); g_stage = nullptr; }
    if (g_ring) { synHostFree(g_dev, g_ring, 0); g_ring = nullptr; for (auto& u : g_ring_used) u = false; }
    // events are created lazily per device
    for (auto& e : g_h2d_ev) { if (e) synEventDestroy(e); e = nullptr; }
    for (auto& e : g_ring_ev) { if (e) synEventDestroy(e); e = nullptr; }
    for (auto& e : g_ev) { if (e) synEventDestroy(e); e = nullptr; }
    g_ring_i = 0;
    synStreamDestroy(g_stream);
    synDeviceRelease(g_dev);
    synDestroy();
    g_card = -1;
    g_init = false;
    return 0;
}

// ---------------------------------------------------------------- memory
RG_API uint64_t rg_malloc(uint64_t bytes) {
    uint64_t raw = 0;
    synStatus s = synDeviceMalloc(g_dev, (bytes ? bytes : 256) + 128, 0, 0, &raw);
    if (s != synSuccess) { fail(s, "synDeviceMalloc"); return 0; }
    const uint64_t aligned = (raw + 127) & ~uint64_t(127);
    g_alloc[aligned] = raw;
    return aligned;
}
RG_API int rg_free(uint64_t addr) {
    auto it = g_alloc.find(addr);
    if (it == g_alloc.end()) { snprintf(g_err, sizeof(g_err), "free: %#lx was not allocated here", (unsigned long)addr); return -1; }
    CHK(synDeviceFree(g_dev, it->second, 0));
    g_alloc.erase(it);
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

// Synchronous host->device copy.
RG_API int rg_h2d(uint64_t dst, const void* src, uint64_t bytes) {
    const char* s = (const char*)src;
    const size_t half = g_stage_size / 2;
    if (!g_h2d_ev[0]) for (auto& e : g_h2d_ev) CHK(synEventCreate(&e, g_dev, 0));
    bool pending[2] = {false, false};
    for (int b = 0; bytes; b ^= 1) {
        size_t n = bytes < half ? bytes : half;
        char* buf = (char*)g_stage + b * half;
        if (pending[b]) CHK(synEventSynchronize(g_h2d_ev[b]));
        memcpy(buf, s, n);
        CHK(synMemCopyAsync(g_stream, (uint64_t)buf, n, dst, HOST_TO_DRAM));
        CHK(synEventRecord(g_h2d_ev[b], g_stream));
        pending[b] = true;
        s += n; dst += n; bytes -= n;
    }
    CHK(synStreamSynchronize(g_stream));
    return 0;
}
// Small non-blocking uploads (per-launch tokens / metadata): a ring of pinned slots.
RG_API int rg_h2d_nb(uint64_t dst, const void* src, uint64_t bytes) {
    if (bytes > kSlot) return rg_h2d(dst, src, bytes);
    if (!g_ring) {
        CHK(synHostMalloc(g_dev, kRing * kSlot, 0, (void**)&g_ring));
        for (auto& e : g_ring_ev) CHK(synEventCreate(&e, g_dev, 0));
    }
    const int i = g_ring_i;
    g_ring_i = (g_ring_i + 1) % kRing;
    if (g_ring_used[i]) CHK(synEventSynchronize(g_ring_ev[i]));
    memcpy(g_ring + i * kSlot, src, bytes);
    CHK(synMemCopyAsync(g_stream, (uint64_t)(g_ring + i * kSlot), bytes, dst, HOST_TO_DRAM));
    CHK(synEventRecord(g_ring_ev[i], g_stream));
    g_ring_used[i] = true;
    return 0;
}

RG_API int rg_d2h(void* dst, uint64_t src, uint64_t bytes) {
    char* d = (char*)dst;
    while (bytes) {
        size_t n = bytes < g_stage_size ? bytes : g_stage_size;
        CHK(synMemCopyAsync(g_stream, src, n, (uint64_t)g_stage, DRAM_TO_HOST));
        CHK(synStreamSynchronize(g_stream));
        memcpy(d, g_stage, n);
        d += n; src += n; bytes -= n;
    }
    return 0;
}
// Async copies between pinned host memory and device (the caller keeps pinned memory alive).
RG_API int rg_h2d_async(uint64_t dst, const void* pinned, uint64_t bytes) {
    CHK(synMemCopyAsync(g_stream, (uint64_t)pinned, bytes, dst, HOST_TO_DRAM)); return 0;
}
RG_API int rg_d2h_async(void* pinned, uint64_t src, uint64_t bytes) {
    CHK(synMemCopyAsync(g_stream, src, bytes, (uint64_t)pinned, DRAM_TO_HOST)); return 0;
}
RG_API int rg_d2d_async(uint64_t dst, uint64_t src, uint64_t bytes) {
    CHK(synMemCopyAsync(g_stream, src, bytes, dst, DRAM_TO_DRAM)); return 0;
}
// Batched device-to-device copies.
RG_API int rg_d2d_multi(const uint64_t* src, const uint64_t* size, const uint64_t* dst, uint64_t n) {
    if (!n) return 0;
    CHK(synMemCopyAsyncMultiple(g_stream, src, size, dst, DRAM_TO_DRAM, n));
    return 0;
}

RG_API int rg_memset32(uint64_t dst, uint32_t value, uint64_t count) {
    CHK(synMemsetD32Async(dst, value, count, g_stream)); return 0;
}

// ---------------------------------------------------------------- events
// A small pool of reusable stream events (g_ev), for completions that wait on one step only.
RG_API int rg_event_record(int slot) {
    if (slot < 0 || slot >= kEvents) { snprintf(g_err, sizeof(g_err), "event: bad slot %d", slot); return -1; }
    if (!g_ev[slot]) CHK(synEventCreate(&g_ev[slot], g_dev, 0));
    CHK(synEventRecord(g_ev[slot], g_stream));
    return 0;
}
RG_API int rg_event_wait(int slot) {
    if (slot < 0 || slot >= kEvents || !g_ev[slot]) { snprintf(g_err, sizeof(g_err), "event: slot %d not recorded", slot); return -1; }
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

// ---------------------------------------------------------------- graphs
RG_API int rg_graph_new() {
    auto* g = new Graph();
    synStatus s = synGraphCreate(&g->h, synDeviceGaudi2);
    if (s != synSuccess) { delete g; fail(s, "synGraphCreate"); return -1; }
    std::lock_guard<std::mutex> lock(g_mu);
    g_graphs.push_back(g);
    return int(g_graphs.size() - 1);
}

static int add_tensor(Graph* g, const char* name, int dtype, int ndims, const uint64_t* sizes,
                      synSectionHandle sec, uint64_t offset, bool persistent) {
    // Graph-scoped handles.
    g->names.emplace_back(name);
    synTensor t;
    CHK(synTensorHandleCreate(&t, g->h, DATA_TENSOR, name));
    synTensorGeometry geo = {};
    geo.dims = unsigned(ndims);
    for (int i = 0; i < ndims; i++) geo.sizes[i] = sizes[i];
    CHK(synTensorSetGeometry(t, &geo, synGeometrySizes));
    CHK(synTensorSetDeviceDataType(t, synDataType(dtype)));
    if (sec) CHK(synTensorAssignToSection(t, sec, offset));
    g->tensors.push_back(t);
    g->persistent.push_back(persistent);
    return int(g->tensors.size() - 1);
}

// dtype: synDataType value. sizes: FCD-first (sizes[0] is the fastest dim).
// persistent: the tensor gets its own persistent section and is bound at launch.
RG_API int rg_tensor(int gi, const char* name, int dtype, int ndims, const uint64_t* sizes, int persistent) {
    Graph* g = g_graphs[gi];
    synSectionHandle sec = nullptr;
    if (persistent) {
        CHK(synSectionCreate(&sec, 0, g->h));
        CHK(synSectionSetPersistent(sec, true));
        g->sections.push_back(sec);
    }
    return add_tensor(g, name, dtype, ndims, sizes, sec, 0, persistent != 0);
}

// A persistent tensor placed at byte `offset` inside the section of tensor `other`.
RG_API int rg_tensor_in_section(int gi, const char* name, int dtype, int ndims, const uint64_t* sizes,
                                int other, uint64_t offset) {
    Graph* g = g_graphs[gi];
    synSectionHandle sec = nullptr;
    uint64_t off0 = 0;
    CHK(synTensorGetSection(g->tensors[other], &sec, &off0));
    return add_tensor(g, name, dtype, ndims, sizes, sec, offset, true);
}

// Mark the section of tensor `t` read-modify-write.
RG_API int rg_section_rmw(int gi, int t) {
    Graph* g = g_graphs[gi];
    synSectionHandle sec = nullptr;
    uint64_t off0 = 0;
    CHK(synTensorGetSection(g->tensors[t], &sec, &off0));
    CHK(synSectionSetRMW(sec, true));
    return 0;
}

RG_API int rg_node(int gi, const char* guid, const char* name,
                   const int* ins, int nin, const int* outs, int nout,
                   const void* params, int params_size) {
    Graph* g = g_graphs[gi];
    std::vector<synTensor> ti, to;
    for (int i = 0; i < nin; i++) ti.push_back(ins[i] < 0 ? nullptr : g->tensors[ins[i]]);
    for (int i = 0; i < nout; i++) to.push_back(g->tensors[outs[i]]);
    synNodeId id = 0;
    synStatus s = synNodeCreateWithId(g->h, ti.data(), to.data(), nin, nout, params, params_size, guid, name, &id, nullptr, nullptr);
    if (s != synSuccess) { fail(s, (std::string("synNodeCreate ") + guid + " " + name).c_str()); return -1; }
    g->nodes.push_back(id);
    return int(g->nodes.size() - 1);
}

// Control dependency: node `blocked` runs after node `blocking` (indices from rg_node).
RG_API int rg_dep(int gi, int blocking, int blocked) {
    Graph* g = g_graphs[gi];
    synNodeId a = g->nodes[blocking], b = g->nodes[blocked];
    CHK(synNodeDependencySet(g->h, &a, &b, 1, 1));
    return 0;
}

static int finish_recipe(Graph* g, Recipe* r) {
    std::lock_guard<std::mutex> lock(g_mu);
    CHK(synWorkspaceGetSize(&r->workspace_size, r->h));
    if (r->workspace_size > g_ws_size) {
        CHK(synStreamSynchronize(g_stream));
        if (g_ws) CHK(synDeviceFree(g_dev, g_ws, 0));
        CHK(synDeviceMalloc(g_dev, r->workspace_size, 0, 0, &g_ws));
        g_ws_size = r->workspace_size;
    }
    // Persistent tensors of the graph and their launch ids.
    for (size_t i = 0; i < g->tensors.size(); i++) if (g->persistent[i]) r->names.push_back(g->names[i]);
    std::vector<const char*> nm;
    for (auto& n : r->names) nm.push_back(n.c_str());
    std::vector<uint64_t> ids(nm.size());
    if (!nm.empty()) CHK(synTensorRetrieveIds(r->h, nm.data(), ids.data(), nm.size()));
    for (size_t i = 0; i < nm.size(); i++) {
        synLaunchTensorInfoExt b;
        memset(&b, 0, sizeof(b));
        b.tensorName = r->names[i].c_str();
        b.tensorType = DATA_TENSOR;
        b.tensorId = ids[i];
        r->binds.push_back(b);
        r->index[r->names[i]] = int(i);
    }
    g_recipes.push_back(r);
    return int(g_recipes.size() - 1);
}

// Compile graph `gi`; with a non-empty `out_path` also serialize the recipe there. Publication
// into the durable cache is the caller's job.
RG_API int rg_compile(int gi, const char* name, const char* out_path) {
    Graph* g;
    { std::lock_guard<std::mutex> lock(g_mu); g = g_graphs[gi]; }
    auto* r = new Recipe();
    synStatus s = synGraphCompile(&r->h, g->h, name, nullptr);
    if (s != synSuccess) { delete r; fail(s, "synGraphCompile"); return -1; }
    g_compiles++;
    if (out_path && out_path[0]) {
        s = synRecipeSerialize(r->h, out_path);
        if (s != synSuccess) { synRecipeDestroy(r->h); delete r; fail(s, "synRecipeSerialize"); return -1; }
    }
    return finish_recipe(g, r);
}

// Load a serialized recipe of graph `gi` (the graph supplies the persistent tensor names).
RG_API int rg_recipe_load(int gi, const char* path) {
    Graph* g;
    { std::lock_guard<std::mutex> lock(g_mu); g = g_graphs[gi]; }
    auto* r = new Recipe();
    synStatus s = synRecipeDeSerialize(&r->h, path);
    if (s != synSuccess) { delete r; fail(s, "synRecipeDeSerialize"); return -1; }
    return finish_recipe(g, r);
}

RG_API uint64_t rg_compile_count() { return g_compiles.load(); }

RG_API int rg_graph_destroy(int gi) {
    Graph* g = g_graphs[gi];
    if (!g) return 0;
    for (auto t : g->tensors) synTensorDestroy(t);
    for (auto s : g->sections) synSectionDestroy(s);
    synGraphDestroy(g->h);
    delete g;
    g_graphs[gi] = nullptr;
    return 0;
}

RG_API uint64_t rg_recipe_workspace(int ri) { return g_recipes[ri]->workspace_size; }

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
    Recipe* r = g_recipes[ri];
    for (auto& b : r->binds) if (!b.pTensorAddress) { snprintf(g_err, sizeof(g_err), "launch: %s unbound", b.tensorName); return -1; }
    CHK(synLaunchWithExternalEventsExt(g_stream, r->binds.data(), r->binds.size(), g_ws, r->h, nullptr, 0, 0));
    return 0;
}

// ---------------------------------------------------------------- profiling
// The trace type is selectable because the plugin phase and the requested type must agree:
// profileApi + synTraceAll collects host API events only, and the device phase rejects a
// mismatched type with a generic status 26. RESIN_PROF_TRACE_TYPE=all|device|host.
static synTraceType rg_trace_type() {
    const char* t = getenv("RESIN_PROF_TRACE_TYPE");
    if (!t) return synTraceAll;
    if (!strcmp(t, "device")) return synTraceDevice;
    if (!strcmp(t, "host")) return synTraceHost;
    return synTraceAll;
}
RG_API int rg_prof_start() {
    uint32_t bytes = 0;
    synStatus s = synProfilerQueryRequiredMemory(g_dev, &bytes);
    if (s != synSuccess && s != synUnsupported) { fail(s, "synProfilerQueryRequiredMemory"); return -1; }
    fprintf(stderr, "[prof] required_bytes=%u type=%d\n", bytes, (int)rg_trace_type());
    if (bytes) {
        uint64_t buf = 0;
        CHK(synDeviceMalloc(g_dev, bytes, 0, 0, &buf));
        CHK(synProfilerSetUserBuffer(g_dev, (void*)buf));
    }
    CHK(synProfilerStart(rg_trace_type(), g_dev));
    return 0;
}
// RESIN_PROF_DUMP_UNION=<path> writes the raw bytes of every event's `arguments` union beside
// the TSV, with the buffer bounds. Which member of that union is live depends on `ph`, and the
// header does not say which; this dump is how the mapping was read off a real trace instead of
// guessed, and it is how a later trace can be re-checked.
static void rg_dump_union(const char* path, synTraceEvent* ev, size_t entries,
                          const std::vector<unsigned char>& buf) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[prof] cannot write %s\n", path); return; }
    fprintf(f, "# buf_base=0x%llx buf_bytes=%zu entries=%zu\n",
            (unsigned long long)reinterpret_cast<uintptr_t>(buf.data()), buf.size(), entries);
    fprintf(f, "ph\tengine_type\tengine_index\tname\tcategory\tunion_hex\n");
    for (size_t i = 0; i < entries; i++) {
        auto& e = ev[i];
        const unsigned char* raw = reinterpret_cast<const unsigned char*>(&e.arguments);
        char hex[129];
        for (int b = 0; b < 64; b++) snprintf(hex + 2 * b, 3, "%02x", raw[b]);
        fprintf(f, "%c\t%u\t%u\t%s\t%s\t%s\n", e.type ? e.type : '?', e.engineType,
                e.engineIndex, e.name ? e.name : "", e.category ? e.category : "", hex);
    }
    fclose(f);
}

RG_API int rg_prof_stop(const char* tsv_path) {
    CHK(synStreamSynchronize(g_stream));
    CHK(synProfilerStop(rg_trace_type(), g_dev));
    size_t bytes = 0, entries = 0;
    CHK(synProfilerGetTrace(rg_trace_type(), g_dev, synTraceFormatTEF, nullptr, &bytes, &entries));
    std::vector<unsigned char> buf(bytes);
    CHK(synProfilerGetTrace(rg_trace_type(), g_dev, synTraceFormatTEF, buf.data(), &bytes, &entries));
    FILE* f = fopen(tsv_path, "w");
    if (!f) { snprintf(g_err, sizeof(g_err), "prof_stop: cannot write %s", tsv_path); return -1; }
    fprintf(f, "ph\tengine_type\tengine_index\tts\tdur\tname\tcategory\tmeta\trecipe_id\trecipe_name\n");
    auto* ev = reinterpret_cast<synTraceEvent*>(buf.data());
    // Columns 9-10 come from the union's event-args member (synapse_common_types.h:451-461), which
    // is what carries recipeId (uint16, line 455) and recipeName (line 456). The union is tagged by
    // `ph` and the header never spells the tag out: a metadata event ('M') stores {name, group} at
    // the same offset (synapse_common_types.h:478-482), so reading recipeId/recipeName off an M row
    // prints a truncated pointer as the id and raw bytes as the name -- measured on a device+host
    // trace, the M rows are the only rows with non-UTF-8 bytes, which is what broke parsers. Every
    // other phase seen in a trace ('X' device and host spans, 'B'/'E' pairs, 'i' instants, 'f'
    // flow) reads a plausible recipeId and a name that is either empty or a recipe name. So the two
    // columns are emitted for every phase except 'M', and a literal '-' marks the M rows.
    for (size_t i = 0; i < entries; i++) {
        auto& e = ev[i];
        fprintf(f, "%c\t%u\t%u\t%.3Lf\t%.3Lf\t%s\t%s\t%s", e.type ? e.type : '?', e.engineType, e.engineIndex,
                e.timestamp, e.duration, e.name ? e.name : "", e.category ? e.category : "",
                (e.type == 'M' && e.arguments.name) ? e.arguments.name : "");
        if (e.type == 'M') {
            fprintf(f, "\t-\t-\n");
        } else {
            fprintf(f, "\t%u\t%s\n", (unsigned)e.arguments.recipeId,
                    e.arguments.recipeName ? e.arguments.recipeName : "");
        }
    }
    fclose(f);
    const char* dump = getenv("RESIN_PROF_DUMP_UNION");
    if (dump && *dump) rg_dump_union(dump, ev, entries, buf);
    return int(entries);
}

RG_API int64_t rg_now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

// ---------------------------------------------------------------- collectives (HCCL)
// One communicator per process. Ranks exchange the unique id out of band (Resin's Gloo
// control groups).
static hcclComm_t g_comm = nullptr;

RG_API int rg_ccl_id_size() { return int(sizeof(hcclUniqueId)); }

RG_API int rg_ccl_unique_id(void* out) {
    hcclUniqueId id;
    hcclResult_t r = hcclGetUniqueId(&id);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclGetUniqueId failed: %d", int(r)); return -1; }
    memcpy(out, &id, sizeof(id));
    return 0;
}

RG_API int rg_ccl_init(const void* id_bytes, int nranks, int rank) {
    hcclUniqueId id;
    memcpy(&id, id_bytes, sizeof(id));
    hcclResult_t r = hcclCommInitRank(&g_comm, nranks, id, rank);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclCommInitRank failed: %d", int(r)); return -1; }
    return 0;
}

// dtype: 0 = f32, 1 = bf16. In place when send == recv.
RG_API int rg_ccl_allreduce(uint64_t send, uint64_t recv, uint64_t count, int dtype) {
    hcclResult_t r = hcclAllReduce((const void*)send, (void*)recv, count, dtype ? hcclBfloat16 : hcclFloat32,
                                   hcclSum, g_comm, g_stream);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclAllReduce failed: %d", int(r)); return -1; }
    return 0;
}

RG_API int rg_ccl_allgather(uint64_t send, uint64_t recv, uint64_t count, int dtype) {
    hcclResult_t r = hcclAllGather((const void*)send, (void*)recv, count, dtype ? hcclBfloat16 : hcclFloat32,
                                   g_comm, g_stream);
    if (r != hcclSuccess) { snprintf(g_err, sizeof(g_err), "hcclAllGather failed: %d", int(r)); return -1; }
    return 0;
}

// Launch recipes ris[k] in order, each followed by an in-place all-reduce of ar[k] (count[k]
// values, 0 = none; bit 63 set = bf16).
static const uint64_t AR_BF16 = 1ull << 63;
RG_API int rg_launch_segs(int n, const int* ris, const uint64_t* ar, const uint64_t* count) {
    for (int k = 0; k < n; k++) {
        if (rg_launch(ris[k])) return -1;
        if (count[k] && rg_ccl_allreduce(ar[k], ar[k], count[k] & ~AR_BF16, count[k] & AR_BF16 ? 1 : 0)) return -1;
    }
    return 0;
}

RG_API int rg_ccl_destroy() {
    if (g_comm) { hcclCommDestroy(g_comm); g_comm = nullptr; }
    return 0;
}
