// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t rmw 2>&1 | FileCheck %s --check-prefix=RMW
// RUN: not %run %t.opt rmw 2>&1 | FileCheck %s --check-prefix=RMW
// RUN: not %run %t cas 2>&1 | FileCheck %s --check-prefix=CAS
// RUN: not %run %t.opt cas 2>&1 | FileCheck %s --check-prefix=CAS

#include "hip.h"

// An atomic is neither a load nor a store as far as the IR is concerned, so it
// is reached by a branch of the pass of its own, and a counter one thread past
// the end of a shared array is exactly the bug that branch is there for. Both
// flavours count as writes: the access changes the location whether or not the
// value it read is used.
__global__ void bump(int *P, int I) {
  __atomic_fetch_add(P + I, 1, __ATOMIC_RELAXED);
}

__global__ void claim(int *P, int I) {
  int Want = 0;
  __atomic_compare_exchange_n(P + I, &Want, 1, false, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
}

int main(int argc, char **argv) {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  if (argv[1][0] == 'r')
    bump<<<1, 1>>>(P, 64);
  else
    claim<<<1, 1>>>(P, 64);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// RMW: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// RMW: WRITE of size 4 at
// RMW: in bump{{.*}}atomic-overflow.hip
// RMW: is located 0 bytes after 256-byte region
// RMW-NOT: survived

// CAS: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CAS: WRITE of size 4 at
// CAS: in claim{{.*}}atomic-overflow.hip
// CAS: is located 0 bytes after 256-byte region
// CAS-NOT: survived
