//===-- hip.h - what these tests use of HIP, declared here ---------------===//
//
// Compiled with -nogpuinc -nogpulib, so ROCm's headers are never read and the
// suite is not tied to the version of one that happens to be installed. Only
// the runtime library is still needed, to link and load against.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_TEST_HIP_H
#define DASAN_TEST_HIP_H

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

extern "C" {
typedef struct ihipStream_t *hipStream_t;

int hipMalloc(void **Ptr, unsigned long Size);
int hipFree(void *Ptr);
int hipMemcpy(void *Dst, const void *Src, unsigned long Size, int Kind);
int hipDeviceSynchronize(void);
int hipGetLastError(void);
int hipMemset(void *Ptr, int Value, unsigned long Size);
int hipGetDeviceCount(int *Count);
int hipSetDevice(int Device);
int hipGetDevice(int *Device);

// A device variable is reached from the host by the address of its host shadow,
// which is what the runtime registered it under.
int hipGetSymbolSize(unsigned long *Size, const void *Symbol);
int hipGetSymbolAddress(void **Ptr, const void *Symbol);
int hipMemcpyToSymbol(const void *Symbol, const void *Src, unsigned long Size,
                      unsigned long Offset = 0, int Kind = 1);
int hipMemcpyFromSymbol(void *Dst, const void *Symbol, unsigned long Size,
                        unsigned long Offset = 0, int Kind = 2);

// The defaults are what let <<<Grid, Block>>> leave the last two out.
int __hipPushCallConfiguration(dim3 GridDim, dim3 BlockDim,
                               unsigned long SharedMem = 0,
                               hipStream_t Stream = 0);
int __hipPopCallConfiguration(dim3 *GridDim, dim3 *BlockDim,
                              unsigned long *SharedMem, hipStream_t *Stream);
int hipLaunchKernel(const void *Func, dim3 GridDim, dim3 BlockDim, void **Args,
                    unsigned long SharedMem, hipStream_t Stream);

int printf(const char *, ...);

// A report reaches the terminal unbuffered, so anything a test prints that has
// to be read against one has to be pushed out first.
int fflush(void *Stream);
}

enum hipMemcpyKind { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };

// What the tests use of the launch geometry, taken from the builtins the real
// headers wrap rather than from the structs they declare around them.
__device__ inline unsigned threadIdxX() {
  return __builtin_amdgcn_workitem_id_x();
}
__device__ inline unsigned blockIdxX() {
  return __builtin_amdgcn_workgroup_id_x();
}
__device__ inline void syncThreads() { __builtin_amdgcn_s_barrier(); }

// A four-float vector, for an access as wide as one instruction can make it.
typedef float float4 __attribute__((ext_vector_type(4)));

// A test that reaches the end without dying has found nothing, which is a
// failure everywhere except the case that expects a clean run.
#define CHECK_HIP(Expr)                                                        \
  do {                                                                         \
    if ((Expr) != 0) {                                                         \
      printf("setup failed: %s\n", #Expr);                                     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

#endif // DASAN_TEST_HIP_H
