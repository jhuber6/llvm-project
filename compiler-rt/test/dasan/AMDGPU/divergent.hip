// RUN: %clang_dasan_hip -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt -fsanitize-recover=daddress %s -o %t.opt %hip_libs
// RUN: %run %t wave 2>&1 | FileCheck %s --check-prefix=WAVE
// RUN: %run %t.opt wave 2>&1 | FileCheck %s --check-prefix=WAVE
// RUN: %run %t wave 2>&1 | grep -c "ERROR: DeviceAddressSanitizer" \
// RUN:   | FileCheck %s --check-prefix=ONCE
// RUN: %run %t.opt wave 2>&1 | grep -c "ERROR: DeviceAddressSanitizer" \
// RUN:   | FileCheck %s --check-prefix=ONCE
// RUN: %run %t one 2>&1 | FileCheck %s --check-prefix=ONE
// RUN: %run %t.opt one 2>&1 | FileCheck %s --check-prefix=ONE

#include "hip.h"

// Who speaks when a wave fails together, and who is named when only one lane
// does.
//
// Every lane of `wave` leaves the object at the same access. They all failed on
// the same thing, so one report describes it and the rest would only bury it: a
// single lane is elected from the check's own ballot, which is not the same set
// as the lanes still executing. Thirty-two threads is one wave wherever this
// runs.
//
// In `one` a single lane in a single block is out of bounds, and the report has
// to name that lane and that block rather than the first of either.
__global__ void every_lane(int *P) { P[64 + threadIdxX()] = 1; }

__global__ void one_lane(int *P) {
  if (blockIdxX() == 2 && threadIdxX() == 33)
    P[64] = 1;
}

int main(int argc, char **argv) {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  if (argv[1][0] == 'w')
    every_lane<<<1, 32>>>(P);
  else
    one_lane<<<4, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());

  printf("kept going\n");
  hipFree(P);
  return 0;
}

// WAVE: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// WAVE: WRITE of size 4 at {{0x[0-9a-f]+}} thread ({{[0-9]+}},0,0) of block (0,0,0)
// WAVE: in every_lane{{.*}}divergent.hip
// WAVE: kept going

// ONCE: {{^}}1{{$}}

// ONE: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// ONE: WRITE of size 4 at {{0x[0-9a-f]+}} thread (33,0,0) of block (2,0,0)
// ONE: in one_lane{{.*}}divergent.hip
// ONE: is located 0 bytes after 256-byte region
// ONE: kept going
