// An object wider than a chunk is described by every chunk it covers, the later
// ones naming a base below themselves, so the check reaches the same extent
// from any of them. Its size here is a whole number of chunks, so the alignment
// leaves no redzone inside the last one and the byte past the end would be the
// next global's first. The guard the pass emits behind it is what that byte
// lands in instead, and the host describes the guard as part of this object, so
// the access is measured against the thing it left.
//
// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 1023 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 1023 | FileCheck %s --check-prefix=INSIDE
// RUN: not %run %t 1024 2>&1 | FileCheck %s
// RUN: not %run %t.opt 1024 2>&1 | FileCheck %s

#include "hip.h"

extern "C" int atoi(const char *);

__device__ int Wide[1024];

__global__ void touch(int I) { Wide[I] = 1; }

int main(int argc, char **argv) {
  touch<<<1, 1>>>(atoi(argv[1]));
  hipDeviceSynchronize();

  printf("survived\n");
  return 0;
}

// INSIDE: survived

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}global-span-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Wide' {{.*}} of size 4096
// CHECK-NOT: survived
