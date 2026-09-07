// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void oob(int *P) { P[1] = 0; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  oob<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 3;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: #0 {{.*}}oob{{.*}}oob.hip:5
// CHECK: [[ADDR]] is located 0 bytes after a 4-byte host region
// CHECK: allocated by host thread here:
// CHECK: #{{[0-9]+}} {{.*}}main{{.*}}oob.hip:10
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow
// CHECK: Shadow bytes around the buggy address:
