// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

__global__ void write_one(int *P) { P[0] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  CHECK_HIP(hipFree(P));

  // The allocation is gone but the address still decodes to its chunk, which
  // no longer describes anything.
  write_one<<<1, 1>>>(P);
  hipDeviceSynchronize();

  printf("survived\n");
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_one{{.*}}use-after-free.hip
// CHECK-NOT: survived
