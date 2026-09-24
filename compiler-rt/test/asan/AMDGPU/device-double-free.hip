// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

extern "C" __device__ unsigned long __asan_malloc_impl(unsigned long,
                                                       unsigned long);
extern "C" __device__ void __asan_free_impl(unsigned long, unsigned long);

// Mirrors the device 'malloc' and 'free' of the HIP headers.
__device__ __attribute__((noinline)) void *dmalloc(unsigned long N) {
  return (void *)__asan_malloc_impl(
      N, (unsigned long)__builtin_return_address(0));
}
__device__ __attribute__((noinline)) void dfree(void *P) {
  __asan_free_impl((unsigned long)P,
                   (unsigned long)__builtin_return_address(0));
}

__global__ void twice() {
  void *P = dmalloc(16);
  dfree(P);
  dfree(P);
}

int main() {
  twice<<<1, 1>>>();
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: attempting double-free
// CHECK-SAME: by device
// CHECK-NEXT: #0 {{.*}}twice{{.*}}device-double-free.hip:22
// CHECK: freed by device here:
// CHECK-NEXT: #0 {{.*}}twice{{.*}}device-double-free.hip:21
// CHECK: previously allocated by device here:
// CHECK-NEXT: #0 {{.*}}twice{{.*}}device-double-free.hip:20
// CHECK: SUMMARY: AddressSanitizer: double-free
