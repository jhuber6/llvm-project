// RUN: %clang_dasan_omp %s -o %t
// RUN: %clang_dasan_omp_O1 %s -o %t.opt
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s
// RUN: not %run %t overflow 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// RUN: not %run %t.opt overflow 2>&1 | FileCheck %s --check-prefix=OVERFLOW
//
// OpenMP maps an array section as `section - host_offset` so Data[P][I] still
// type-checks. Clang must not mark GEPs from that pointer inbounds.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1000
static int Data[8][M];

int main(int argc, char **argv) {
  int Overflow = argc > 1;

#pragma omp parallel num_threads(8)
  {
    int P = omp_get_thread_num();
    int N = Overflow ? M + 4096 : M;
#pragma omp target map(tofrom : Data[P : 1][0 : M]) firstprivate(P, N)
    {
      for (int I = 0; I < N; ++I)
        Data[P][I] = P + 1;
    }
  }

  for (int P = 0; P < 8; ++P)
    for (int I = 0; I < M; ++I)
      if (Data[P][I] != P + 1) {
        printf("wrong value at [%d][%d]\n", P, I);
        return 2;
      }

  printf("all in bounds\n");
  return 0;
}

// An in-bounds program must run clean and get the right answer.
// CHECK-NOT: ERROR: DeviceAddressSanitizer
// CHECK: all in bounds

// Running off the end of a section must still be caught.
// OVERFLOW: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// OVERFLOW: WRITE of size 4
// OVERFLOW-NOT: all in bounds
