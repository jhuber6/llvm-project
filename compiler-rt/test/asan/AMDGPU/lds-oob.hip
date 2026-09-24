// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void oob(int N) {
  __shared__ int L[16];
  L[N] = 1;
  __builtin_amdgcn_s_barrier();
  if (L[0] == 42)
    L[1] = 0;
}

int main() {
  oob<<<1, 1>>>(16);
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at {{.*}} by device
// CHECK: #0 {{.*}}oob{{.*}}lds-oob.hip:7
// CHECK: is located {{[0-9]+}} bytes inside of {{[0-9]+}}-byte LDS block of a work-group
// CHECK-NOT: allocated by
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow
