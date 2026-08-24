// RUN: %clang_dasan_hip -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %clang_dasan_hip_opt -fsanitize-recover=daddress %s -o %t.opt %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s
// RUN: %run %t 2>&1 | grep -c "ERROR: DeviceAddressSanitizer" \
// RUN:   | FileCheck %s --check-prefix=TWICE
// RUN: %run %t.opt 2>&1 | grep -c "ERROR: DeviceAddressSanitizer" \
// RUN:   | FileCheck %s --check-prefix=TWICE

#include "hip.h"

// A recovering build has to survive the access it reported, and the access is
// what happens next: the check only reports, so the store still lands wherever
// the pointer said. An overflow that clears its chunk's own redzone lands in the
// next chunk, and that chunk has to be memory that exists, or the program dies
// on a page fault immediately after a report that was perfectly good -- which
// looks like the tool killing a program it had just diagnosed.
//
// So a fresh allocation is placed at the bottom of the run backing its class,
// leaving the rest of the run mapped above it. Only the highest chunk a class
// has ever handed out has nothing above it, and it stops being the highest as
// soon as the class grows again.
//
// Two bugs, in that order: one that walks eight hundred bytes off the end, and
// one that steps a single word off it afterwards. The second report is the
// point -- it can only be printed if the first one did not take the program
// down with it.
__global__ void walk(int *P, int N) {
#pragma clang loop unroll(disable)
  for (int I = 0; I < N; ++I)
    P[64 + I] = I;
}

__global__ void step(int *P) { P[64] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  walk<<<1, 1>>>(P, 200);
  CHECK_HIP(hipDeviceSynchronize());

  step<<<1, 1>>>(P);
  CHECK_HIP(hipDeviceSynchronize());

  printf("kept going\n");
  hipFree(P);
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: in walk{{.*}}recover-continues.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: in step{{.*}}recover-continues.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: kept going

// TWICE: {{^}}2{{$}}
