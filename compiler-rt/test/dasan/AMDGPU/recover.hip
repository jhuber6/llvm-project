// RUN: %clang_dasan_hip -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt -fsanitize-recover=daddress %s -o %t.opt %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Built to recover, so a wave reports and carries on rather than trapping, and
// the process lives long enough to say so itself.
__global__ void write_past(int *P, int I) { P[I] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  write_past<<<1, 1>>>(P, 64);
  CHECK_HIP(hipDeviceSynchronize());

  printf("kept going\n");
  hipFree(P);
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK-NOT: ABORTING
// CHECK: kept going
