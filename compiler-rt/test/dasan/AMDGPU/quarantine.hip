// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=HELD
// RUN: not %run %t.opt 2>&1 | FileCheck %s --check-prefix=HELD
// RUN: %env_dasan_opts=quarantine_size_mb=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REUSED
// RUN: %env_dasan_opts=quarantine_size_mb=0 %run %t.opt 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REUSED

#include "hip.h"

// What a use after free is reported against is the chunk the freed allocation
// held, and that chunk stops describing anything until it is handed out again.
// Holding it back is the whole of the window in which a stale pointer is a
// diagnostic rather than a silent write into somebody else's object, so the
// quarantine is what decides the answer here and the test asks it both ways.
//
// With the default budget the freed chunk is held and the second allocation is
// somewhere else, so the stale pointer still names nothing. With no budget at
// all it is handed straight back, the second allocation is the same memory, and
// the stale write is an ordinary in-bounds write to a live object -- which is
// exactly what the tool cannot tell apart, and says so by staying quiet.
__global__ void write_one(int *P) { P[0] = 1; }

int main() {
  void *A = nullptr, *B = nullptr;
  CHECK_HIP(hipMalloc(&A, 256));
  CHECK_HIP(hipFree(A));
  CHECK_HIP(hipMalloc(&B, 256));
  printf(A == B ? "reused\n" : "held\n");
  fflush(nullptr);

  write_one<<<1, 1>>>((int *)A);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(B);
  return 0;
}

// HELD: held
// HELD: ERROR: DeviceAddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// HELD: WRITE of size 4 at [[ADDR]]
// HELD: #0 {{0x[0-9a-f]+}} in write_one{{.*}}quarantine.hip
// HELD: the region was freed by the host before this access
// HELD-NOT: survived

// REUSED: reused
// REUSED-NOT: ERROR: DeviceAddressSanitizer
// REUSED: survived
