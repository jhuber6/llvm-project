// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Below the object rather than past it. The ground under a global is the pad of
// whatever the linker laid out before it, which that object's entry still
// describes, so the access is caught by an entry naming a neighbour. Which of
// the two ran into the padding is then settled on the host by whichever edge is
// nearer -- four bytes below this object against a thousand past the other --
// and the answer is the one that says something about the program.
__device__ int Before[256];
__device__ int Table[256];

__global__ void read_before(int I) { Before[0] = Table[I]; }

int main() {
  read_before<<<1, 1>>>(-1);
  hipDeviceSynchronize();

  printf("survived\n");
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in read_before{{.*}}global-underflow.hip
// CHECK: [[ADDR]] is located 4 bytes before global variable 'Table' {{.*}} of size 1024
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-underflow
// CHECK-NOT: survived
