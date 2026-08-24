// RUN: %clang_dasan_hip %s %S/../Inputs/second-module.hip -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s %S/../Inputs/second-module.hip -o %t.opt %hip_libs
// RUN: %run %t clean 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t.opt clean 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: not %run %t mine 2>&1 | FileCheck %s --check-prefix=MINE
// RUN: not %run %t.opt mine 2>&1 | FileCheck %s --check-prefix=MINE
// RUN: not %run %t theirs 2>&1 | FileCheck %s --check-prefix=THEIRS
// RUN: not %run %t.opt theirs 2>&1 | FileCheck %s --check-prefix=THEIRS
// RUN: not %run %t shared 2>&1 | FileCheck %s --check-prefix=SHARED
// RUN: not %run %t.opt shared 2>&1 | FileCheck %s --check-prefix=SHARED

#include "hip.h"

// Two instrumented translation units linked into one image. Everything the pass
// adds is per module -- the marker that says the image has instrumented code in
// it, the table of group-object extents, the placed globals -- so linking two
// of them together is the first time any of it has to coexist. A duplicate
// marker or a merged table would take the whole image out.
//
// Each module's own objects are checked against their own extents. Group
// memory of a kernel in the other module is described by that module's table.
extern __global__ void write_other(int I);
extern __global__ void write_other_shared(int *Out, int I);

__device__ int Mine[16];

__global__ void write_mine(int I) { Mine[I] = 1; }

int main(int argc, char **argv) {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  switch (argv[1][0]) {
  case 'c':
    write_mine<<<1, 1>>>(15);
    write_other<<<1, 1>>>(15);
    write_other_shared<<<1, 16>>>(Out, 15);
    break;
  case 'm':
    write_mine<<<1, 1>>>(16);
    break;
  case 't':
    write_other<<<1, 1>>>(16);
    break;
  default:
    write_other_shared<<<1, 16>>>(Out, 16);
    break;
  }
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(Out);
  return argv[1][0] == 'c' ? 0 : 3;
}

// CLEAN-NOT: ERROR: DeviceAddressSanitizer
// CLEAN: survived

// MINE: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// MINE: in write_mine{{.*}}two-modules.hip
// MINE: is located 0 bytes after global variable 'Mine' {{.*}} of size 64
// MINE-NOT: survived

// THEIRS: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// THEIRS: in write_other{{.*}}second-module.hip
// THEIRS: is located 0 bytes after global variable 'Other' {{.*}} of size 64
// THEIRS-NOT: survived

// SHARED: ERROR: DeviceAddressSanitizer: shared-buffer-overflow
// SHARED: in write_other_shared{{.*}}second-module.hip
// SHARED: is located 0 bytes after 64-byte region
// SHARED-NOT: survived
