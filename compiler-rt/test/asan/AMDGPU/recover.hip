// REQUIRES: asan-hip
// RUN: %clang_asan_hip -fsanitize-recover=address %s -o %t %hip_libs
// RUN: %env_asan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s

__global__ void oob(int *P) { P[4] = 0; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 4 * sizeof(int)));
  oob<<<1, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  oob<<<1, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  hipFree(P);
  printf("done\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK-NOT: ERROR: AddressSanitizer
// CHECK: done
