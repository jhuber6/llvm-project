// RUN: %clang_dasan_omp %s -o %t
// RUN: %clang_dasan_omp_O1 %s -o %t.opt
// RUN: %run %t 15 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t.opt 15 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: not %run %t 16 2>&1 | FileCheck %s
// RUN: not %run %t.opt 16 2>&1 | FileCheck %s

// A declare-target global on the other offload path. Placing globals is where
// the two paths differ most: this one's device image carries the OpenMP runtime
// and the state it introspects by name, some of which it casts to a variable
// outright, so a pass that rewrites every global into an alias takes the link
// out rather than the program. The ones it must leave alone are named by the
// implementation; this one is the program's, and gets the treatment.

#include <cstdio>
#include <cstdlib>

#pragma omp begin declare target
int Table[16];
#pragma omp end declare target

int main(int argc, char **argv) {
  int I = atoi(argv[1]);

#pragma omp target firstprivate(I)
  {
    Table[I] = 1;
  }

  printf("survived\n");
  return I < 16 ? 0 : 3;
}

// CLEAN-NOT: ERROR: DeviceAddressSanitizer
// CLEAN: survived

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}openmp-global-overflow.cpp
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Table' {{.*}} of size 64
// CHECK-NOT: survived
