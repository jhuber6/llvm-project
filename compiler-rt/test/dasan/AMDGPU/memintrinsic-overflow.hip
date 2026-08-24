// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t copy 2>&1 | FileCheck %s --check-prefix=DEST
// RUN: not %run %t.opt copy 2>&1 | FileCheck %s --check-prefix=DEST
// RUN: not %run %t read 2>&1 | FileCheck %s --check-prefix=SRC
// RUN: not %run %t.opt read 2>&1 | FileCheck %s --check-prefix=SRC
// RUN: not %run %t fill 2>&1 | FileCheck %s --check-prefix=FILL
// RUN: not %run %t.opt fill 2>&1 | FileCheck %s --check-prefix=FILL

#include "hip.h"

// A memcpy or memset is checked as one access covering the whole interval,
// which is cheaper than walking it and catches the same thing: a run that
// starts inside the object and ends past its end. The report names the length,
// not the width of whatever the backend lowered it into, so a 200-byte copy
// reads as a 200-byte access. Both operands of a transfer are checked, and the
// source is a read.
//
// The length is taken from the host so the backend cannot lower these into
// plain stores whose sizes would be the ones reported.
__global__ void copy_into(char *Dst, const char *Src, unsigned long N) {
  __builtin_memcpy(Dst + 128, Src, N);
}

__global__ void copy_out_of(char *Dst, const char *Src, unsigned long N) {
  __builtin_memcpy(Dst, Src + 128, N);
}

__global__ void fill(char *Dst, unsigned long N) {
  __builtin_memset(Dst + 128, 0x41, N);
}

int main(int argc, char **argv) {
  char *A = nullptr, *B = nullptr;
  CHECK_HIP(hipMalloc((void **)&A, 256));
  CHECK_HIP(hipMalloc((void **)&B, 256));

  switch (argv[1][0]) {
  case 'c':
    copy_into<<<1, 1>>>(A, B, 200);
    break;
  case 'r':
    copy_out_of<<<1, 1>>>(A, B, 200);
    break;
  default:
    fill<<<1, 1>>>(A, 200);
    break;
  }
  hipDeviceSynchronize();

  printf("survived\n");
  return 3;
}

// DEST: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// DEST: WRITE of size 200 at
// DEST: in copy_into{{.*}}memintrinsic-overflow.hip
// DEST: is located 128 bytes inside of 256-byte region
// DEST-NOT: survived

// SRC: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// SRC: READ of size 200 at
// SRC: in copy_out_of{{.*}}memintrinsic-overflow.hip
// SRC: is located 128 bytes inside of 256-byte region
// SRC-NOT: survived

// FILL: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// FILL: WRITE of size 200 at
// FILL: in fill{{.*}}memintrinsic-overflow.hip
// FILL: is located 128 bytes inside of 256-byte region
// FILL-NOT: survived
