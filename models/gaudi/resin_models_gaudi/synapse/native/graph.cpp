// Graph building and compiling: what decides a recipe's contents. The recipe cache keys recipes by the digest of
// this file and graph.h (see build.py), not by runtime.cpp.
#include "graph.h"
#include <cstdint>
#include <atomic>
#include <mutex>

struct Graph {
    synGraphHandle h = nullptr;
    std::vector<synTensor> tensors;
    std::vector<synSectionHandle> sections;
    std::vector<std::string> names;      // stable storage for tensor names
    std::vector<bool> persistent;
    std::vector<synNodeId> nodes;
};

static std::vector<Graph*> g_graphs;
// Guards the graph table: graphs are built and compiled on several threads.
static std::mutex g_graphs_mu;
// Graph compilations in this process (deserialized recipes do not count).
static std::atomic<uint64_t> g_compiles{0};

static Graph* graph(int gi) {
    std::lock_guard<std::mutex> lock(g_graphs_mu);
    return g_graphs[gi];
}

RG_API int rg_graph_new() {
    auto* g = new Graph();
    synStatus s = synGraphCreate(&g->h, synDeviceGaudi2);
    if (s != synSuccess) { delete g; fail(s, "synGraphCreate"); return -1; }
    std::lock_guard<std::mutex> lock(g_graphs_mu);
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
    Graph* g = graph(gi);
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
    Graph* g = graph(gi);
    synSectionHandle sec = nullptr;
    uint64_t off0 = 0;
    CHK(synTensorGetSection(g->tensors[other], &sec, &off0));
    return add_tensor(g, name, dtype, ndims, sizes, sec, offset, true);
}

// Mark the section of tensor `t` read-modify-write.
RG_API int rg_section_rmw(int gi, int t) {
    Graph* g = graph(gi);
    synSectionHandle sec = nullptr;
    uint64_t off0 = 0;
    CHK(synTensorGetSection(g->tensors[t], &sec, &off0));
    CHK(synSectionSetRMW(sec, true));
    return 0;
}

RG_API int rg_node(int gi, const char* guid, const char* name,
                   const int* ins, int nin, const int* outs, int nout,
                   const void* params, int params_size) {
    Graph* g = graph(gi);
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
    Graph* g = graph(gi);
    synNodeId a = g->nodes[blocking], b = g->nodes[blocked];
    CHK(synNodeDependencySet(g->h, &a, &b, 1, 1));
    return 0;
}

static int keep(Graph* g, synRecipeHandle h) {
    std::vector<std::string> names;
    for (size_t i = 0; i < g->tensors.size(); i++) if (g->persistent[i]) names.push_back(g->names[i]);
    const int ri = rg_register_recipe(h, names);
    if (ri < 0) synRecipeDestroy(h);
    return ri;
}

// Compile graph `gi`; with a non-empty `out_path` also serialize the recipe there. Publication
// into the durable cache is the caller's job.
RG_API int rg_compile(int gi, const char* name, const char* out_path) {
    Graph* g = graph(gi);
    synRecipeHandle h = nullptr;
    synStatus s = synGraphCompile(&h, g->h, name, nullptr);
    if (s != synSuccess) { fail(s, "synGraphCompile"); return -1; }
    g_compiles++;
    if (out_path && out_path[0]) {
        s = synRecipeSerialize(h, out_path);
        if (s != synSuccess) { synRecipeDestroy(h); fail(s, "synRecipeSerialize"); return -1; }
    }
    return keep(g, h);
}

// Load a serialized recipe of graph `gi` (the graph supplies the persistent tensor names).
RG_API int rg_recipe_load(int gi, const char* path) {
    Graph* g = graph(gi);
    synRecipeHandle h = nullptr;
    synStatus s = synRecipeDeSerialize(&h, path);
    if (s != synSuccess) { fail(s, "synRecipeDeSerialize"); return -1; }
    return keep(g, h);
}

RG_API uint64_t rg_compile_count() { return g_compiles.load(); }

static void destroy(Graph* g) {
    for (auto t : g->tensors) synTensorDestroy(t);
    for (auto s : g->sections) synSectionDestroy(s);
    synGraphDestroy(g->h);
    delete g;
}

RG_API int rg_graph_destroy(int gi) {
    Graph* g;
    {
        std::lock_guard<std::mutex> lock(g_graphs_mu);
        g = g_graphs[gi];
        g_graphs[gi] = nullptr;
    }
    if (g) destroy(g);
    return 0;
}

void rg_destroy_graphs() {
    std::lock_guard<std::mutex> lock(g_graphs_mu);
    for (auto* g : g_graphs) if (g) destroy(g);
    g_graphs.clear();
}
