// RUN: %clang_dasan_hip -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt -fsanitize-recover=daddress %s -o %t.opt %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s
// RUN: %run %t 2>&1 | grep -c "ERROR: DeviceAddressSanitizer" \
// RUN:   | FileCheck %s --check-prefix=ONCE
// RUN: %run %t.opt 2>&1 | grep -c "ERROR: DeviceAddressSanitizer" \
// RUN:   | FileCheck %s --check-prefix=ONCE

#include "hip.h"

// A loop that leaves its object every iteration is one bug, and a run that
// keeps going has to say so once. Reports are deduplicated on the device by the
// program counter that raised them, so the cost of a faulting loop is one round
// trip and not one per iteration -- without which a wave could spend the whole
// kernel talking to the host.
//
// The trip count comes from the host and the loop is kept rolled, so all two
// hundred accesses really do share one program counter.
__global__ void write_past(int *P, int N) {
#pragma clang loop unroll(disable)
  for (int I = 0; I < N; ++I)
    P[64 + I] = I;
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  write_past<<<1, 1>>>(P, 200);
  CHECK_HIP(hipDeviceSynchronize());

  printf("kept going\n");
  hipFree(P);
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at
// CHECK: in write_past{{.*}}dedup.hip
// CHECK: kept going

// ONCE: {{^}}1{{$}}
