// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Group objects keep the addresses the program gave them; a write that leaves
// one object is measured against that object's extent, not against the whole
// segment. The result has to leave the kernel or there is no access left to
// catch.
__global__ void write_past(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = 0;
  syncThreads();
  if (T == 0)
    Buf[I] = 0x41;
  syncThreads();
  *Out = Buf[0] + Other[0];
}

int main() {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  write_past<<<1, 16>>>(Out, 16);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(Out);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}shared-overflow.hip
// CHECK-NOT: survived
