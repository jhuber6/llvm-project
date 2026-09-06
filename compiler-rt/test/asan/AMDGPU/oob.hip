// REQUIRES: asan-hip-xnack
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

// Device loads of host shadow need XNACK. Held until an XNACK machine is
// available.

__global__ void oob(int *P) { P[1] = 0; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  oob<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 3;
}

// CHECK: ERROR: AddressSanitizer:
// CHECK: WRITE of size 4
