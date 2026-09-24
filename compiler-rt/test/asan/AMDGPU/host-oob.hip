// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

// Host accesses past a pinned HSA allocation hit the redzone on the host.

int main() {
  char *P = nullptr;
  CHECK_HIP(hipHostMalloc((void **)&P, 100));
  P[99] = 0;
  ((volatile char *)P)[100] = 1;
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 1
// CHECK: #0 {{.*}} in main {{.*}}host-oob.hip:11
