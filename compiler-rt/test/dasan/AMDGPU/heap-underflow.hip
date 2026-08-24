// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

__global__ void write_before(int *P, int I) { P[I] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  write_before<<<1, 1>>>(P, -1);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_before{{.*}}heap-underflow.hip
// CHECK: [[ADDR]] is located 4 bytes before 256-byte region
// CHECK-NOT: survived
