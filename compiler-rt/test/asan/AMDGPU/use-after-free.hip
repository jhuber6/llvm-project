// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void load(int *P, int *Out) { *Out = *P; }

int main() {
  int *P = nullptr, *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  CHECK_HIP(hipFree(P));
  load<<<1, 1>>>(P, Out);
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-use-after-free
// CHECK: READ of size 4 at {{.*}} by device
// CHECK: freed by host thread here:
// CHECK: previously allocated by host thread here:
