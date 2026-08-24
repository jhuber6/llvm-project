// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Placed on a chunk of its own by the pass, so the space past its end is a
// redzone the host can describe.
__device__ int Table[16];

__global__ void write_past(int I) { Table[I] = 1; }

int main() {
  write_past<<<1, 1>>>(16);
  hipDeviceSynchronize();

  printf("survived\n");
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}global-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Table' {{.*}} of size 64
// CHECK-NOT: survived
