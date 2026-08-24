// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t arg 2>&1 | FileCheck %s
// RUN: %run %t.opt arg 2>&1 | FileCheck %s
// RUN: %run %t decay 2>&1 | FileCheck %s
// RUN: %run %t.opt decay 2>&1 | FileCheck %s

#include "hip.h"

// TODO: poke's argument has no getUnderlyingObject global, so the store is
// checked as heap. A group-segment pointer is outside the placed space and
// the access is not reported. Object-precise bounds on escaped LDS pointers
// need a capability scheme that does not invent uses of sibling tiles.

__device__ __attribute__((noinline)) void poke(int *P, int I) { P[I] = 0x41; }
__device__ __attribute__((noinline)) void poke0(int *P) { *P = 0x41; }

__global__ void write_arg(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = 0;
  syncThreads();
  if (T == 0)
    poke(Buf, I);
  syncThreads();
  *Out = Buf[0] + Other[0];
}

__global__ void write_decay(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = 0;
  syncThreads();
  if (T == 0)
    poke0(Buf + I);
  syncThreads();
  *Out = Buf[0] + Other[0];
}

int main(int argc, char **argv) {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  if (argv[1][0] == 'd')
    write_decay<<<1, 16>>>(Out, 16);
  else
    write_arg<<<1, 16>>>(Out, 16);
  CHECK_HIP(hipDeviceSynchronize());

  printf("survived\n");
  hipFree(Out);
  return 0;
}

// CHECK: survived
