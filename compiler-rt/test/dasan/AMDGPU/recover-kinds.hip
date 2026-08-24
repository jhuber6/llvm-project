// RUN: %clang_dasan_hip -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt -fsanitize-recover=daddress %s -o %t.opt %hip_libs
// RUN: %run %t global 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: %run %t.opt global 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: %run %t shared 2>&1 | FileCheck %s --check-prefix=SHARED
// RUN: %run %t.opt shared 2>&1 | FileCheck %s --check-prefix=SHARED

#include "hip.h"

// The two kinds of report are separate entry points, and each comes in a
// halting and a recovering flavour picked when the program was compiled. In a
// halting build either ends the wave, so returning to the access is something
// only the recovering pair does, and each of the two has to be shown to do it:
// the kernel must finish and hand back what it computed. Without LDS redzones
// the recovered shared write still lands in the neighbouring object, so the
// value that comes back is the overflow (0x41) rather than the in-bounds 1.
__device__ int Table[16];

__global__ void write_global(int *Out, int I) {
  Table[I] = 1;
  *Out = Table[0] + 1;
}

__global__ void write_shared(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = T;
  syncThreads();
  if (T == 0)
    Buf[I] = 0x41;
  syncThreads();
  *Out = Buf[0] + Other[0] + 1;
}

int main(int argc, char **argv) {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  CHECK_HIP(hipMemset(Out, 0, sizeof(int)));

  if (argv[1][0] == 'g')
    write_global<<<1, 16>>>(Out, 16);
  else
    write_shared<<<1, 16>>>(Out, 16);
  CHECK_HIP(hipDeviceSynchronize());

  int Got = 0;
  CHECK_HIP(hipMemcpy(&Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  printf("kept going, wrote %d\n", Got);
  hipFree(Out);
  return 0;
}

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// GLOBAL: in write_global{{.*}}recover-kinds.hip
// GLOBAL: is located 0 bytes after global variable 'Table' {{.*}} of size 64
// GLOBAL-NOT: ABORTING
// GLOBAL: kept going, wrote 1

// SHARED: ERROR: DeviceAddressSanitizer: shared-buffer-overflow
// SHARED: in write_shared{{.*}}recover-kinds.hip
// SHARED: is located 0 bytes after 64-byte region
// SHARED-NOT: ABORTING
// SHARED: kept going, wrote 66
