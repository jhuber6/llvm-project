// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void twice() {
  char *P = (char *)malloc(8);
  free(P);
  free(P);
}

int main() {
  twice<<<1, 1>>>();
  hipDeviceSynchronize();
  return 3;
}

// CHECK: ERROR: AddressSanitizer: attempting double-free: [[ADDR:0x[0-9a-f]+]]
// CHECK: FREE of [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: #0 {{.*}}twice{{.*}}double-free.hip:8
// CHECK: SUMMARY: AddressSanitizer: attempting double-free
