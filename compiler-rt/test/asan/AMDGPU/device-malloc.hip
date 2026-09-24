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

__global__ void oob() {
  char *P = (char *)dmalloc(10);
  P[10] = 1;
  dfree(P);
}

int main() {
  oob<<<1, 1>>>();
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 1 at {{.*}} by device
// CHECK: is located 0 bytes after 10-byte region
// CHECK: allocated by device here:
// CHECK-NEXT: #0 {{.*}}oob{{.*}}device-malloc.hip:20
