// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// The access is in a function the kernel called and the compiler did not
// inline, which is where the reported program counter comes from: the handler
// takes its own return address, so what is named is the innermost function that
// went wrong and not the kernel that got there. There is no walk behind it --
// nothing on the device kept a frame pointer for one -- so this is the whole of
// the stack, and it has to be the useful line rather than the launch.
__attribute__((noinline)) __device__ void store(int *P, int I) { P[I] = 1; }

__global__ void call_store(int *P, int I) { store(P, I); }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  call_store<<<1, 1>>>(P, 64);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in store{{.*}}device-function.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow {{.*}}device-function.hip
// CHECK-NOT: survived
