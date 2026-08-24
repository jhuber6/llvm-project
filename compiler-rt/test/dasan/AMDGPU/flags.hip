// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %env_dasan_opts=alias_globals=0 %run %t global 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNALIASED
// RUN: %env_dasan_opts=alias_globals=0 %run %t.opt global 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNALIASED
// RUN: %env_dasan_opts=halt_on_error=0 %run %t heap 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CARRYON
// RUN: %env_dasan_opts=halt_on_error=0 %run %t.opt heap 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CARRYON
// RUN: %env_dasan_opts=print_stats=1 %run %t none 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS
// RUN: %env_dasan_opts=print_stats=1 %run %t.opt none 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS

#include "hip.h"

// alias_globals is what puts images somewhere the check can see them; with it
// off the instrumented code still runs and has to treat those accesses as none
// of its business. Heap is always placed. Whether the program dies is decided
// by the host at the report, separately from whether the wave does.
__device__ int Table[16];

__global__ void write_heap(int *P) { P[64] = 1; }

__global__ void write_global(int I) { Table[I] = 1; }

int main(int argc, char **argv) {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  if (argv[1][0] == 'h')
    write_heap<<<1, 1>>>(P);
  else if (argv[1][0] == 'g')
    write_global<<<1, 1>>>(16);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 0;
}

// UNALIASED-NOT: ERROR: DeviceAddressSanitizer
// UNALIASED: survived

// CARRYON: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CARRYON-NOT: HINT
// CARRYON-NOT: ABORTING
// CARRYON: survived

// Printed as the device runtime shuts down, so this run has to reach the end.
// STATS-NOT: ERROR: DeviceAddressSanitizer
// STATS: stats: {{[0-9]+}} placed ({{[0-9]+}} above their own class), {{[0-9]+}} freed
// STATS: stats: {{[0-9]+}} code objects aliased over {{[0-9]+}} chunks
