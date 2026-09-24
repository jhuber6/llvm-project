// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void oob(int *P) { P[4] = 0; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 4 * sizeof(int)));
  oob<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at {{.*}} by device thread (0,0,0) block (0,0,0)
// CHECK: #0 {{.*}}oob{{.*}}heap-oob.hip:5
// CHECK: is located 0 bytes after 16-byte region
// CHECK: allocated by host thread here:
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow
