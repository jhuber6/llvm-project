// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Dynamic group memory has no extent in the object at all: how much of it a
// launch asked for is known only from the dispatch packet, so the bound has to
// be read from there rather than from anything the pass can see.
__global__ void write_past(int *Out, int I) {
  extern __shared__ int Dyn[];
  unsigned T = threadIdxX();
  Dyn[T] = T;
  syncThreads();
  if (T == 0)
    Dyn[I] = 0x41;
  syncThreads();
  *Out = Dyn[0];
}

int main() {
  const int N = 8;
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  write_past<<<1, N, N * sizeof(int)>>>(Out, N);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(Out);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}dynamic-shared-overflow.hip
// CHECK-NOT: survived
