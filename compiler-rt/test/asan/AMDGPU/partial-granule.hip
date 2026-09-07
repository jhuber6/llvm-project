// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

// An overflow that stays inside the last shadow granule is only caught if the
// redzone records how many bytes of that granule are addressable.

__global__ void oob(char *P) { P[5] = 1; }

int main() {
  char *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 5));
  oob<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 3;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: [[ADDR]] is located 0 bytes after a 5-byte host region
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow
