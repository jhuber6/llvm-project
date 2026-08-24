// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Group memory has no host to second-guess the report: the extent is the
// object the pointer was derived from, so an index below zero is measured
// against that object and not against the neighbour the address landed in.
__global__ void write_before(int *Out, int I) {
  __shared__ int First[16];
  __shared__ int Second[16];
  unsigned T = threadIdxX();
  First[T] = T;
  Second[T] = T;
  syncThreads();
  if (T == 0)
    Second[I] = 0x41;
  syncThreads();
  *Out = First[0] + Second[0];
}

int main() {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  write_before<<<1, 16>>>(Out, -1);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(Out);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_before{{.*}}shared-underflow.hip
// CHECK: [[ADDR]] is located 4 bytes before 64-byte region
// CHECK: the region is group memory
// CHECK-NOT: survived
