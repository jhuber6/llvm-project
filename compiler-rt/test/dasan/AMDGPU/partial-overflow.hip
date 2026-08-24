// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// An access that begins inside the object and ends outside it. One compare
// decides both ends, so this is caught by the same check as any other overflow,
// but the address it reports is a legitimate one: the report says the access is
// *inside* the region rather than after it, and it is the width that took it
// out. The allocation is 252 bytes so that an eight byte store at offset 248,
// aligned as its own width, straddles the end by four.
__global__ void write_across(void *P, int I) {
  static_cast<double *>(P)[I] = 1.0;
}

int main() {
  void *P = nullptr;
  CHECK_HIP(hipMalloc(&P, 252));

  write_across<<<1, 1>>>(P, 31);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 8 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_across{{.*}}partial-overflow.hip
// CHECK: [[ADDR]] is located 248 bytes inside of 252-byte region
// CHECK-NOT: survived
