// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

__global__ void write_past(int *P, int I) { P[I] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  write_past<<<1, 1>>>(P, 64);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread ({{.*}}) of block ({{.*}})
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}heap-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK-NOT: survived
