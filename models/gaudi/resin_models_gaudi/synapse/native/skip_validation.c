/* Preloaded (LD_PRELOAD) into every rank by the skipvalidation feature (-f skipvalidation): Synapse then does not
 * check the tensor bindings of each launch. If the symbol below does not match the installed Synapse, the check stays
 * on (resin_gaudi_skipped_validations stays 0). */
#include <stdatomic.h>

static atomic_long skipped;

int _ZN14RecipeLauncher20validateSectionsInfoERK10LaunchInfo(void *self, const void *info) {
    (void)self;
    (void)info;
    atomic_fetch_add_explicit(&skipped, 1, memory_order_relaxed);
    return 0; /* synSuccess */
}

/* How many launches skipped the check in this process. */
long resin_gaudi_skipped_validations(void) { return atomic_load(&skipped); }
