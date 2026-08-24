// RUN: %clang_dasan_omp %s -o %t
// RUN: %clang_dasan_omp_O1 %s -o %t.opt
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

// The other offload path. It reaches the device runtime by dlopen rather than
// by linking it, and it runs its own RPC server that the tool has to be handed
// a turn on, so almost nothing it depends on is shared with the HIP tests.

#include <cstdio>
#include <omp.h>

int main() {
  int Dev = omp_get_default_device();
  int *P = (int *)omp_target_alloc(64 * sizeof(int), Dev);
  if (!P) {
    printf("setup failed\n");
    return 2;
  }

#pragma omp target is_device_ptr(P) device(Dev)
  {
    P[64] = 1;
  }

  printf("survived\n");
  omp_target_free(P, Dev);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}openmp-heap-overflow.cpp
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK-NOT: survived
