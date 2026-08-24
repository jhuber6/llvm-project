// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// What the host is told about a placed global. The pass wraps each one in a
// larger allocation to get a redzone behind it, and the symbol has to go on
// describing the object alone: everything the host does with a device variable
// by name -- its size, its address, a copy into part of it -- comes out of the
// symbol table, so a symbol grown to cover the redzone would hand every one of
// those the wrong answer, and a copy sized by it would write into the redzone
// and be reported.
//
// The kernel reading back what the host wrote is the other half: the symbol has
// to name the object itself and not the front of the allocation it sits in,
// which are the same address only because the object is placed first.
__device__ int Table[16];

__global__ void sum(int *Out) {
  int S = 0;
  for (int I = 0; I < 16; ++I)
    S += Table[I];
  *Out = S;
}

int main() {
  unsigned long Size = 0;
  CHECK_HIP(hipGetSymbolSize(&Size, (const void *)&Table));
  printf("size %lu\n", Size);

  void *Addr = nullptr;
  CHECK_HIP(hipGetSymbolAddress(&Addr, (const void *)&Table));
  printf("aligned %d\n", (int)((unsigned long)Addr % 256 == 0));

  int Host[16];
  for (int I = 0; I < 16; ++I)
    Host[I] = I + 1;

  // The whole object, then the back half of it at an offset, then a copy one
  // byte too long -- which the runtime refuses, because the size it has is the
  // object's.
  CHECK_HIP(hipMemcpyToSymbol((const void *)&Table, Host, sizeof(Host)));
  CHECK_HIP(hipMemcpyToSymbol((const void *)&Table, Host, sizeof(Host) / 2,
                              sizeof(Host) / 2));
  printf("refused %d\n",
         (int)(hipMemcpyToSymbol((const void *)&Table, Host, Size + 1) != 0));
  hipGetLastError();

  int Back[8] = {0};
  CHECK_HIP(hipMemcpyFromSymbol(Back, (const void *)&Table, sizeof(Back),
                                sizeof(Back)));
  printf("offset %d\n", (int)(Back[0] == 1 && Back[7] == 8));

  // 1..8 twice over, from the second copy landing on top of the first.
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  sum<<<1, 1>>>(Out);
  CHECK_HIP(hipDeviceSynchronize());

  int Got = 0;
  CHECK_HIP(hipMemcpy(&Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  printf("sum %d\n", Got);

  hipFree(Out);
  return 0;
}

// CHECK-NOT: ERROR: DeviceAddressSanitizer
// CHECK: size 64
// CHECK: aligned 1
// CHECK: refused 1
// CHECK: offset 1
// CHECK: sum 72
