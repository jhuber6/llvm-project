// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t 16 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 16 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t 1000 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 1000 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t 4096 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 4096 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t 100000 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt 100000 in 2>&1 | FileCheck %s --check-prefix=INSIDE
// RUN: not %run %t 16 out 2>&1 | FileCheck %s -DSIZE=16
// RUN: not %run %t.opt 16 out 2>&1 | FileCheck %s -DSIZE=16
// RUN: not %run %t 1000 out 2>&1 | FileCheck %s -DSIZE=1000
// RUN: not %run %t.opt 1000 out 2>&1 | FileCheck %s -DSIZE=1000
// RUN: not %run %t 4096 out 2>&1 | FileCheck %s -DSIZE=4096
// RUN: not %run %t.opt 4096 out 2>&1 | FileCheck %s -DSIZE=4096
// RUN: not %run %t 100000 out 2>&1 | FileCheck %s -DSIZE=100000
// RUN: not %run %t.opt 100000 out 2>&1 | FileCheck %s -DSIZE=100000

#include "hip.h"

extern "C" int atoi(const char *);

// The same object at four sizes, spanning the classes an allocation can land
// in: one smaller than the leading redzone, one that shares its chunk with a
// redzone at each end, one a whole page, and one large enough that its class is
// backed a chunk at a time rather than in runs. In every case the extent the
// tool holds the access against is the size that was asked for and not the size
// of the chunk it was placed in, and the address it was placed at is aligned
// the way a device runtime promises, which the check itself leans on.
__global__ void touch(char *P, long I) { P[I] = 1; }

int main(int argc, char **argv) {
  const long N = atoi(argv[1]);
  const bool Out = argv[2][0] == 'o';

  char *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, N));
  if ((unsigned long)P % 256) {
    printf("setup failed: alignment\n");
    return 2;
  }

  // The last byte of the object, or the first one past it.
  touch<<<1, 1>>>(P, Out ? N : N - 1);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return Out ? 3 : 0;
}

// INSIDE-NOT: ERROR: DeviceAddressSanitizer
// INSIDE: survived

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}size-classes.hip
// CHECK: [[ADDR]] is located 0 bytes after [[SIZE]]-byte region
// CHECK-NOT: survived
