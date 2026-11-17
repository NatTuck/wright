// What graph.cpp (graph building and compiling) and runtime.cpp (streams, memory, launches) share.
//
// Recipes are keyed by the digest of graph.h and graph.cpp only, so a change of runtime.cpp alone does not
// make the recipe cache compile again.
#pragma once
#include <synapse_api.h>
#include <cstdio>
#include <string>
#include <vector>

#define RG_API extern "C" __attribute__((visibility("default")))

// Per thread: graphs may compile concurrently, and the caller reads the error on its own thread.
extern thread_local char g_err[1024];

inline bool fail(synStatus s, const char* what) {
    char desc[STATUS_DESCRIPTION_MAX_SIZE] = {0};
    synStatusGetBriefDescription(s, desc, sizeof(desc));
    snprintf(g_err, sizeof(g_err), "%s failed: status=%d (%s)", what, int(s), desc);
    return false;
}
#define CHK(call) do { synStatus _s = (call); if (_s != synSuccess) { fail(_s, #call); return -1; } } while (0)

// runtime.cpp: keep recipe `h` (compiled or loaded) with the names of its graph's persistent tensors, in their
// order; returns the recipe index, or -1.
int rg_register_recipe(synRecipeHandle h, const std::vector<std::string>& persistent);
// graph.cpp: destroy every graph (rg_shutdown).
void rg_destroy_graphs();
