// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__device__ int Array[8];

__global__ void oob(int N) { Array[N] = 1; }

int main() {
  oob<<<1, 1>>>(8);
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: global-buffer-overflow
// CHECK: WRITE of size 4 at {{.*}} by device
// CHECK: is located 0 bytes after global variable 'Array' of size 32
