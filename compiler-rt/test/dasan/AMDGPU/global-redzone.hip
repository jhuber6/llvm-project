// The redzone behind a global is sized the way asan sizes one, a quarter of the
// object floored at a chunk, so how far an overflow is caught scales with the
// object it came off. A stride is as easy to get wrong as an index, and a stride
// scales with what it strides over: a wrong one off a large array clears more
// ground than a fixed redzone would have covered.
//
// The host describes the redzone as part of the object in front of it, so an
// access anywhere in it is measured against that object however deep it lands.
//
// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: %run %t k 255 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt k 255 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t b 4095 | FileCheck %s --check-prefix=INSIDE
// RUN: %run %t.opt b 4095 | FileCheck %s --check-prefix=INSIDE
// RUN: not %run %t k 288 2>&1 | FileCheck %s --check-prefix=KILO
// RUN: not %run %t.opt k 288 2>&1 | FileCheck %s --check-prefix=KILO
// RUN: not %run %t b 4608 2>&1 | FileCheck %s --check-prefix=BIG
// RUN: not %run %t.opt b 4608 2>&1 | FileCheck %s --check-prefix=BIG

#include "hip.h"

extern "C" int atoi(const char *);

// A kilobyte, a quarter of which is under the floor, so it gets the floor.
__device__ int Kilo[256];

// Sixteen kilobytes, a quarter of which is four, so the redzone is four.
__device__ int Big[4096];

__global__ void kilo(int I) { Kilo[I] = 1; }

__global__ void big(int I) { Big[I] = 1; }

int main(int argc, char **argv) {
  if (argv[1][0] == 'k')
    kilo<<<1, 1>>>(atoi(argv[2]));
  else
    big<<<1, 1>>>(atoi(argv[2]));
  hipDeviceSynchronize();

  printf("survived\n");
  return 0;
}

// INSIDE: survived

// A hundred and twenty-eight bytes past the end, halfway into a redzone that is
// only as wide as the floor.
// KILO: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// KILO: WRITE of size 4 at [[ADDR]]
// KILO: #0 {{0x[0-9a-f]+}} in kilo{{.*}}global-redzone.hip
// KILO: [[ADDR]] is located 128 bytes after global variable 'Kilo' {{.*}} of size 1024
// KILO-NOT: survived

// Two kilobytes past the end, which a redzone of one chunk would have missed
// entirely and this one catches with half of itself to spare.
// BIG: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// BIG: WRITE of size 4 at [[ADDR]]
// BIG: #0 {{0x[0-9a-f]+}} in big{{.*}}global-redzone.hip
// BIG: [[ADDR]] is located 2048 bytes after global variable 'Big' {{.*}} of size 16384
// BIG-NOT: survived
