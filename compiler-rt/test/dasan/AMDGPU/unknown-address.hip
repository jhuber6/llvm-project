// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// An address inside the instrumented space that names a chunk nobody allocated.
// Every chunk a pointer can form has an entry reserved for it, so this is not a
// fault and not a missing entry -- it is an entry saying nothing is there, and
// the tool can say so by name rather than leaving a bare page fault behind. A
// pointer that ran far enough off its own object to leave the chunk entirely
// arrives here the same way.
//
// Derived from a real allocation so the test needs no constant from the mapping
// header: a megabyte on is four thousand chunks away, and nothing in a program
// this small has been placed there.
__global__ void write_one(char *P) { *P = 1; }

int main() {
  char *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 256));

  write_one<<<1, 1>>>(P + (1 << 20));
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: unknown-device-address on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_one{{.*}}unknown-address.hip
// CHECK: SUMMARY: DeviceAddressSanitizer: unknown-device-address
// CHECK-NOT: survived
