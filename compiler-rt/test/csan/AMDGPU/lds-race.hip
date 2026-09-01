// REQUIRES: csan-hip
// RUN: %clang_csan_hip %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__global__ void kernel() {
  __shared__ volatile int Shared[64];
  RACE_UNTIL_FOUND(i)
    Shared[0] += i;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}lds-race.hip:{{[0-9]+}}
// CHECK-NOT: Location is global
