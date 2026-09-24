// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

__global__ void oob(int *P) { P[4] = 0; }

int main() {
  int *P = (int *)malloc(4 * sizeof(int));
  oob<<<1, 1>>>(P);
  hipDeviceSynchronize();
  free(P);
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at {{.*}} by device
// CHECK: #0 {{.*}}oob{{.*}}host-malloc-oob.hip:7
// CHECK: is located 0 bytes after 16-byte region
// CHECK: allocated by thread T0 here:
// CHECK: #{{[0-9]+}} {{.*}}main{{.*}}host-malloc-oob.hip:10
