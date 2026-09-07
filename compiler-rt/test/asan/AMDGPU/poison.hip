// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

// The manual poisoning interface works on the device and shares the host's
// shadow encoding, including for ranges that do not end on a granule boundary.

extern "C" __device__ void __asan_poison_memory_region(void const volatile *,
                                                       unsigned long);

__global__ void poison(char *P) {
  __asan_poison_memory_region(P + 8, 8);
  P[9] = 1;
}

int main() {
  char *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 16));
  poison<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 3;
}

// CHECK: ERROR: AddressSanitizer: use-after-poison on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] in block (0,0,0) thread (0,0,0) lane 0:
// CHECK: #0 {{.*}}poison{{.*}}poison.hip:13
// CHECK: [[ADDR]] is located 9 bytes inside a 16-byte host region
// CHECK: SUMMARY: AddressSanitizer: use-after-poison
