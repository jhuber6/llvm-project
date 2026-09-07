// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

// The device allocator gets its memory over RPC and redzones it in the same
// encoding the host uses.

__global__ void oob() {
  char *P = (char *)malloc(5);
  P[8] = 1;
  free(P);
}

int main() {
  oob<<<1, 1>>>();
  hipDeviceSynchronize();
  return 3;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: #0 {{.*}}oob{{.*}}device-malloc.hip:10
// CHECK: [[ADDR]] is located {{.*}} a 5-byte device region
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow
