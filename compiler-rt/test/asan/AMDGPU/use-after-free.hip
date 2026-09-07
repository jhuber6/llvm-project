// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void read(int *P, int *Out) { *Out = *P; }

int main() {
  int *P = nullptr;
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  CHECK_HIP(hipFree(P));
  read<<<1, 1>>>(P, Out);
  hipDeviceSynchronize();
  return 3;
}

// CHECK: ERROR: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: #0 {{.*}}read{{.*}}use-after-free.hip:5
// CHECK: [[ADDR]] is located 0 bytes inside a 4-byte host region
// CHECK: freed by host thread here:
// CHECK: previously allocated by host thread here:
// CHECK: SUMMARY: AddressSanitizer: heap-use-after-free
