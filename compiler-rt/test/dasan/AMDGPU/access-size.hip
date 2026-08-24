// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 1 2>&1 | FileCheck %s --check-prefix=BYTE
// RUN: not %run %t.opt 1 2>&1 | FileCheck %s --check-prefix=BYTE
// RUN: not %run %t 2 2>&1 | FileCheck %s --check-prefix=SHORT
// RUN: not %run %t.opt 2 2>&1 | FileCheck %s --check-prefix=SHORT
// RUN: not %run %t 8 2>&1 | FileCheck %s --check-prefix=LONG
// RUN: not %run %t.opt 8 2>&1 | FileCheck %s --check-prefix=LONG
// RUN: not %run %t 16 2>&1 | FileCheck %s --check-prefix=VECTOR
// RUN: not %run %t.opt 16 2>&1 | FileCheck %s --check-prefix=VECTOR

#include "hip.h"

extern "C" int atoi(const char *);

// Every width the hardware can move in one instruction, each one byte past the
// end of the same allocation. The width is what the check adds to the offset
// before comparing, and it is what the report has to name: a one byte overflow
// reported as sixteen would send someone looking for the wrong access.
__global__ void write_byte(char *P) { P[256] = 1; }
__global__ void write_short(short *P) { P[128] = 1; }
__global__ void write_long(long *P) { P[32] = 1; }
__global__ void write_vector(float4 *P) { P[16] = float4(1.0f); }

int main(int argc, char **argv) {
  void *P = nullptr;
  CHECK_HIP(hipMalloc(&P, 256));

  switch (atoi(argv[1])) {
  case 1:
    write_byte<<<1, 1>>>((char *)P);
    break;
  case 2:
    write_short<<<1, 1>>>((short *)P);
    break;
  case 8:
    write_long<<<1, 1>>>((long *)P);
    break;
  default:
    write_vector<<<1, 1>>>((float4 *)P);
    break;
  }
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(P);
  return 3;
}

// BYTE: WRITE of size 1 at
// BYTE: in write_byte{{.*}}access-size.hip
// BYTE: is located 0 bytes after 256-byte region
// BYTE-NOT: survived

// SHORT: WRITE of size 2 at
// SHORT: in write_short{{.*}}access-size.hip
// SHORT: is located 0 bytes after 256-byte region
// SHORT-NOT: survived

// LONG: WRITE of size 8 at
// LONG: in write_long{{.*}}access-size.hip
// LONG: is located 0 bytes after 256-byte region
// LONG-NOT: survived

// VECTOR: WRITE of size 16 at
// VECTOR: in write_vector{{.*}}access-size.hip
// VECTOR: is located 0 bytes after 256-byte region
// VECTOR-NOT: survived
