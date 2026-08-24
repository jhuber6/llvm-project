// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// The other direction. Load and store reach different handlers, chosen when the
// program was compiled rather than asked of the host, so a read past the end
// exercises a path a write never touches. The value has to leave the kernel or
// there is no load left to check.
__global__ void read_past(int *P, int *Out, int I) { *Out = P[I]; }

int main() {
  int *P = nullptr, *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  read_past<<<1, 1>>>(P, Out, 64);
  hipDeviceSynchronize();

  printf("survived\n");
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in read_past{{.*}}heap-read-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK-NOT: survived
