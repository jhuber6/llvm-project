// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t.opt 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: not %run %t trip 2>&1 | FileCheck %s
// RUN: not %run %t.opt trip 2>&1 | FileCheck %s
// RUN: %clang_dasan_hip -fsanitize-recover=daddress %s -o %t.recover %hip_libs
// RUN: %clang_dasan_hip_opt -fsanitize-recover=daddress %s -o %t.recover.opt %hip_libs
// RUN: %run %t.recover storm 2>&1 | FileCheck %s --check-prefix=STORM
// RUN: %run %t.recover.opt storm 2>&1 | FileCheck %s --check-prefix=STORM

#include "hip.h"

// Allocator churn across size classes, with enough recycle that a chunk has
// been freed and handed out again while a neighbour is still live. A false
// report is a failure. After that, one byte past the end still has to be
// caught, and a recover storm still has to land.

__device__ int Global[1024];
__constant__ int Konst[64] = {1};

// Every byte of the allocation, written and read back, spread over the whole
// grid. The atomic is on the host's own allocation and is checked too.
__global__ void sweep(char *P, unsigned long N, unsigned long Stride,
                      unsigned long long *Sum) {
  unsigned long T = threadIdxX() + (unsigned long)blockIdxX() * 256;
  unsigned long long Acc = 0;
  for (unsigned long I = T; I < N; I += Stride) {
    P[I] = (char)(I + 1);
    Acc += (unsigned char)P[I];
  }
  __atomic_fetch_add(Sum, Acc, __ATOMIC_RELAXED);
}

// The intrinsics, at the full width of the allocation, both operands in bounds.
__global__ void shuffle(char *Dst, const char *Src, unsigned long N) {
  if (threadIdxX() == 0) {
    __builtin_memset(Dst, 0x41, N);
    __builtin_memcpy(Dst, Src, N);
  }
}

// The kinds of memory that are not the heap: group memory static and dynamic,
// a placed global, a placed constant. Every index is inside.
__global__ void locals(unsigned long long *Sum) {
  __shared__ int Buf[256];
  extern __shared__ int Dyn[];
  unsigned T = threadIdxX();

  Buf[T] = T;
  Dyn[T] = T;
  Global[T + 256 * blockIdxX()] = T;
  syncThreads();

  unsigned long long Acc = Buf[(T + 1) % 256] + Dyn[(T + 1) % 256] +
                           Global[T + 256 * blockIdxX()] + Konst[T % 64];
  __atomic_fetch_add(Sum, Acc, __ATOMIC_RELAXED);
}

// Handed a pointer into the middle of the allocation, and reaching both ways
// from it. Correct, and rooted at an address the allocator never handed out.
__global__ void around(int *Q, int Half) {
#pragma clang loop unroll(disable)
  for (int I = -Half; I < Half; ++I)
    Q[I] = I;
}

__global__ void poke(char *P, long I) { P[I] = 1; }

static const unsigned long Sizes[] = {16, 64, 252, 256, 1000, 4096, 33000,
                                      100000};
static const int NumSizes = sizeof(Sizes) / sizeof(Sizes[0]);
// Two rounds recycle the odd sizes once; a few more is enough for the clean
// path without turning this file into the suite's wall clock.
static const int SoakRounds = 4;
static const int ShortRounds = 2;

int main(int argc, char **argv) {
  const bool Trip = argc > 1 && argv[1][0] == 't';
  const bool Storm = argc > 1 && argv[1][0] == 's';

  unsigned long long *Sum = nullptr;
  CHECK_HIP(hipMalloc((void **)&Sum, sizeof(*Sum)));
  CHECK_HIP(hipMemset(Sum, 0, sizeof(*Sum)));

  char *Live[NumSizes] = {nullptr};
  unsigned long long Want = 0;
  const int Rounds = (Trip || Storm) ? ShortRounds : SoakRounds;

  for (int R = 0; R < Rounds; ++R) {
    for (int S = 0; S < NumSizes; ++S) {
      const unsigned long N = Sizes[S];

      // A fresh allocation every round. The one from last round is still live
      // for the odd sizes, so the two coexist and the freed chunks in between
      // have to come back without taking a live neighbour with them.
      char *P = nullptr;
      CHECK_HIP(hipMalloc((void **)&P, N));
      if ((unsigned long)P % 256) {
        printf("setup failed: alignment\n");
        return 2;
      }

      const unsigned Blocks = N > 4096 ? 8 : 2;
      const unsigned long Stride = 256UL * Blocks;
      sweep<<<Blocks, 256>>>(P, N, Stride, Sum);
      for (unsigned long I = 0; I < N; ++I)
        Want += (unsigned char)(I + 1);

      if (Live[S]) {
        shuffle<<<1, 1>>>(P, Live[S], N);
        CHECK_HIP(hipFree(Live[S]));
        Live[S] = nullptr;
      }

      if (N >= 256) {
        const int Half = (int)(N / 8);
        around<<<1, 1>>>((int *)P + Half, Half);
      }

      // Kept for the next round only for half of them, so that every round both
      // frees chunks and holds some back.
      if (S & 1)
        Live[S] = P;
      else
        CHECK_HIP(hipFree(P));
    }

    locals<<<4, 256, 256 * sizeof(int)>>>(Sum);
    for (int T = 0; T < 4 * 256; ++T)
      Want += (T % 256) + (T % 256) + (T % 256) + (T % 64 ? 0 : 1);
  }

  CHECK_HIP(hipDeviceSynchronize());

  unsigned long long Got = 0;
  CHECK_HIP(hipMemcpy(&Got, Sum, sizeof(Got), hipMemcpyDeviceToHost));
  printf("checksum %d\n", (int)(Got == Want));
  fflush(nullptr);

  // A hundred kilobytes, one byte past the end, after everything above.
  if (Trip) {
    poke<<<1, 1>>>(Live[7], (long)Sizes[7]);
    hipDeviceSynchronize();
  }

  // Many waves failing at once, over and over. Each launch is a dispatch of its
  // own, so the deduplication that silences a repeat within one does not
  // silence these, and the transport has to carry them all while the program
  // keeps running.
  if (Storm) {
    for (int R = 0; R < 4; ++R)
      poke<<<8, 64>>>(Live[5], (long)Sizes[5]);
    CHECK_HIP(hipDeviceSynchronize());
  }

  printf("finished\n");
  for (int S = 0; S < NumSizes; ++S)
    if (Live[S])
      hipFree(Live[S]);
  hipFree(Sum);
  return Trip ? 3 : 0;
}

// CLEAN-NOT: ERROR: DeviceAddressSanitizer
// CLEAN: checksum 1
// CLEAN: finished

// CHECK: checksum 1
// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in poke{{.*}}stress.hip
// CHECK: [[ADDR]] is located 0 bytes after 100000-byte region
// CHECK-NOT: finished

// STORM: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// STORM: is located 0 bytes after 4096-byte region
// STORM: finished
