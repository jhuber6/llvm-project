// RUN: %clang_dasan_omp %s -o %t
// RUN: %clang_dasan_omp_O1 %s -o %t.opt
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s

// Placing an allocation must not hand back a weaker address than the runtime
// standing aside for it would have. HSA aligns anything a granule or larger to
// a granule, and llvm-libc's device allocator builds on that: it recovers a
// slab's base by masking a pointer down to the granule, and tells a slab apart
// from a direct allocation by the same test. Insetting such an allocation by a
// leading redzone made both readings name the wrong object, and freeing then
// wrote through an index read out of the redzone. That landed outside anything
// mapped, so it arrived as a bare page fault rather than as a report.
//
// One allocation and its free is the whole of it: the allocation is what makes
// the device allocator ask the system for a slab, and the free is what masks a
// pointer back down to one.

#include <cstdio>
#include <cstdlib>

int main() {
  int Bad = 0;

#pragma omp target map(tofrom : Bad)
  {
    const int N = 256;
    int *P = (int *)malloc(N * sizeof(int));
    if (!P) {
      Bad = 1;
    } else {
      for (int I = 0; I < N; ++I)
        P[I] = I;
      for (int I = 0; I < N; ++I)
        if (P[I] != I)
          Bad = 2;
      free(P);
    }
  }

  if (Bad) {
    printf("allocation failed (%d)\n", Bad);
    return 2;
  }

  printf("allocator survived\n");
  return 0;
}

// CHECK-NOT: ERROR: DeviceAddressSanitizer
// CHECK: allocator survived
