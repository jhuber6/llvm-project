// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 31 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 31 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: not %run %t 32 2>&1 | FileCheck %s
// RUN: not %run %t.opt 32 2>&1 | FileCheck %s

#include "hip.h"

// A kernel handed a pointer into the middle of an allocation. The check is
// rooted at the pointer the access was derived from, so everything here is
// measured from a base the allocator never handed out -- including the indices
// below zero, which walk back to the start of the allocation and are perfectly
// legal. An entry describes the whole allocation from anywhere in its chunk, so
// the answer does not depend on where inside it the pointer sat.
__global__ void touch(int *Q, int Hi) {
#pragma clang loop unroll(disable)
  for (int I = -32; I <= Hi; ++I)
    Q[I] = I;
}

extern "C" int atoi(const char *);

int main(int argc, char **argv) {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  const int Hi = atoi(argv[1]);
  touch<<<1, 1>>>(P + 32, Hi);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return Hi < 32 ? 0 : 3;
}

// INSIDE-NOT: ERROR: DeviceAddressSanitizer
// INSIDE: survived

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}interior-base.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK-NOT: survived
