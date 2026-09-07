// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__device__ char Buf[5];

__global__ void oob(int I) { Buf[I] = 1; }

int main() {
  oob<<<1, 1>>>(5);
  hipDeviceSynchronize();
  return 3;
}

// CHECK: ERROR: AddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: #0 {{.*}}oob{{.*}}global.hip:7
// CHECK: [[ADDR]] is located 0 bytes after device global 'Buf' of size 5
// CHECK: SUMMARY: AddressSanitizer: global-buffer-overflow
