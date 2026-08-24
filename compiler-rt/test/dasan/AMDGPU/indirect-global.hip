// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t.opt 2>&1 | FileCheck %s

#include "hip.h"

// The pointer is loaded from memory rather than named, so nothing in the kernel
// says which object it belongs to and the check has to recover that from the
// address at runtime -- per lane, since a loaded base is not uniform. The
// report still names the global, because naming happens on the host from the
// address rather than in the module from the pointer.
__device__ int Table[16];
__device__ int *Ptr;

__global__ void publish() { Ptr = Table; }

__global__ void write_past(int I) { Ptr[I] = 1; }

int main() {
  publish<<<1, 1>>>();
  CHECK_HIP(hipDeviceSynchronize());

  write_past<<<1, 1>>>(16);
  hipDeviceSynchronize();

  printf("survived\n");
  return 3;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}indirect-global.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Table' {{.*}} of size 64
// CHECK-NOT: survived
