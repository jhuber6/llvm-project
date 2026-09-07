//===-- hip.h -------------------------------------------------------------===//
//
// Minimal HIP headers so the tests can be compiled without a ROCm installation.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_TEST_HIP_H
#define ASAN_TEST_HIP_H

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
      : x(x), y(y), z(z) {}
};

enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
};

extern "C" {
typedef struct ihipStream_t *hipStream_t;

int hipMalloc(void **Ptr, unsigned long Size);
int hipFree(void *Ptr);
int hipDeviceSynchronize(void);
int hipMemcpy(void *Dst, const void *Src, unsigned long Size,
              hipMemcpyKind Kind);

int __hipPushCallConfiguration(dim3 GridDim, dim3 BlockDim,
                               unsigned long SharedMem = 0,
                               hipStream_t Stream = 0);
int __hipPopCallConfiguration(dim3 *GridDim, dim3 *BlockDim,
                              unsigned long *SharedMem, hipStream_t *Stream);
int hipLaunchKernel(const void *Func, dim3 GridDim, dim3 BlockDim, void **Args,
                    unsigned long SharedMem, hipStream_t Stream);

int printf(const char *, ...);
}

// Mirrors the device malloc/free that __clang_hip_runtime_wrapper.h provides;
// the tests run with -nogpuinc so the real wrapper is not available.
#if __has_feature(address_sanitizer)
extern "C" {
__device__ unsigned long long __asan_malloc_impl(unsigned long long, unsigned long long);
__device__ void __asan_free_impl(unsigned long long, unsigned long long);
}

__attribute__((noinline)) __device__ inline void *malloc(unsigned long Size) {
  return (void *)__asan_malloc_impl(
      Size, (unsigned long long)__builtin_return_address(0));
}

__attribute__((noinline)) __device__ inline void free(void *Ptr) {
  __asan_free_impl((unsigned long long)Ptr,
                   (unsigned long long)__builtin_return_address(0));
}
#endif

#define CHECK_HIP(Expr)                                                        \
  do {                                                                         \
    if ((Expr) != 0) {                                                         \
      printf("setup failed: %s\n", #Expr);                                     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

#endif // ASAN_TEST_HIP_H
