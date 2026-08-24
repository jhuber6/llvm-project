// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// Constant memory is a read-only window onto the same device memory, so it is
// checked like anything else -- but it reaches the device through a different
// address space and a different section, and the pass has to keep the two apart
// or an initialized read-only object ends up somewhere the linker cannot write
// it. The index comes from the host so the read cannot be folded away.
__constant__ int Table[16] = {0};

__global__ void read_past(int *Out, int I) { *Out = Table[I]; }

int main() {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  read_past<<<1, 1>>>(Out, 16);
  hipDeviceSynchronize();

  printf("survived\n");
  hipFree(Out);
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in read_past{{.*}}constant-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Table' {{.*}} of size 64
// CHECK-NOT: survived
