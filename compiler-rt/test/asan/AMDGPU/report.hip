// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

extern "C" __device__ void __asan_report_store4(unsigned long);

__global__ void force_report(int *Out) {
  __asan_report_store4((unsigned long)Out);
  *Out = 1;
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  force_report<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 3;
}

// CHECK: ERROR: AddressSanitizer:
// CHECK: SUMMARY: AddressSanitizer:
