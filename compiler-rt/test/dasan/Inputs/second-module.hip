//===-- second-module.hip - the other half of two-modules.hip ------------===//
//
// A second instrumented translation unit. Everything the pass adds to a module
// is added to this one too: its own module marker, its own group blob and table,
// its own placed globals.
//
//===----------------------------------------------------------------------===//

#include "hip.h"

__device__ int Other[16];

__global__ void write_other(int I) { Other[I] = 1; }

__global__ void write_other_shared(int *Out, int I) {
  __shared__ int Buf[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  syncThreads();
  if (T == 0)
    Buf[I] = 0x41;
  syncThreads();
  *Out = Buf[0];
}
