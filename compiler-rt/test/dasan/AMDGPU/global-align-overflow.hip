// A global asking for more alignment than a chunk still starts one, but the
// slack a stricter alignment leaves behind it is the linker's to hand to
// whatever it lays out next, so none of it is a redzone: the host describes the
// chunks an object covers and nothing past them. What the byte after the last
// of those chunks lands in is the pad the pass put in the same allocation,
// which is there whatever the alignment was and whatever the size rounded to.
//
// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 3 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 3 | FileCheck %s --check-prefix=INSIDE
// RUN: not %run %t 96 2>&1 | FileCheck %s
// RUN: not %run %t.opt 96 2>&1 | FileCheck %s

#include "hip.h"

extern "C" int atoi(const char *);

__device__ __attribute__((aligned(1024))) int Small[4];

__global__ void touch(int I) { Small[I] = 1; }

int main(int argc, char **argv) {
  touch<<<1, 1>>>(atoi(argv[1]));
  hipDeviceSynchronize();

  printf("survived\n");
  return 0;
}

// INSIDE: survived

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}global-align-overflow.hip
// CHECK: [[ADDR]] is located 368 bytes after global variable 'Small' {{.*}} of size 16
// CHECK-NOT: survived
