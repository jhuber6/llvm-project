// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

// Host interceptors poison GPU allocation redzones. A too-large copy is
// diagnosed from the host shadow and does not need XNACK.

int main() {
  char *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 8));
  char Src[16] = {};
  CHECK_HIP(hipMemcpy(P, Src, 16, hipMemcpyHostToDevice));
  hipFree(P);
  return 0;
}

// CHECK: ERROR: AddressSanitizer:
// CHECK: WRITE of size 16
