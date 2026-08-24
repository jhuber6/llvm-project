// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Every kind of memory the tool covers, touched only where it should be. A
// report here is a false positive, which is worse than a missed one.
__device__ int Global[16];

__global__ void touch_all(int *Heap, int N) {
  __shared__ int Shared[16];
  extern __shared__ int Dyn[];
  unsigned T = threadIdxX();

  if (T < 16) {
    Shared[T] = T;
    Global[T] = T;
  }
  Dyn[T] = T;
  syncThreads();

  for (int I = T; I < N; I += 16)
    Heap[I] = Shared[I % 16] + Global[I % 16] + Dyn[T];
}

int main() {
  const int N = 256;
  int *Heap = nullptr;
  CHECK_HIP(hipMalloc((void **)&Heap, N * sizeof(int)));

  touch_all<<<1, 16, 16 * sizeof(int)>>>(Heap, N);
  CHECK_HIP(hipDeviceSynchronize());

  printf("clean\n");
  hipFree(Heap);
  return 0;
}

// CHECK-NOT: ERROR: DeviceAddressSanitizer
// CHECK: clean
