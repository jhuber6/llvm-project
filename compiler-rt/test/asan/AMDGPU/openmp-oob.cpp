// REQUIRES: asan-openmp-offload
// RUN: %clang_asan_omp_offload %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <omp.h>

int main() {
  int *P = (int *)omp_target_alloc(4 * sizeof(int), omp_get_default_device());
#pragma omp target is_device_ptr(P)
  P[4] = 1;
  omp_target_free(P, omp_get_default_device());
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at {{.*}} by device
// CHECK: #0 {{.*}}openmp-oob.cpp:10
// CHECK: is located 0 bytes after 16-byte region
// CHECK: allocated by host thread here:
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow
