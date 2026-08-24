// RUN: %clang_dasan_hip %s -o %t %hip_libs
// RUN: llvm-objcopy --dump-section=.hip_fatbin=%t.fatbin %t /dev/null
// RUN: %offload_bundler --unbundle --type=o \
// RUN:   --targets=hipv4-amdgcn-amd-amdhsa--%amdgpu_arch \
// RUN:   --input=%t.fatbin --output=%t.elf
// RUN: llvm-readelf --notes %t.elf | FileCheck %s \
// RUN:   --implicit-check-not='group_segment_fixed_size: {{[23][0-9]{4}}}'
// RUN: %run %t

#include "hip.h"

// Two kernels, each with a 16 KiB tile. Checks must not mention the other
// kernel's global, or occupancy falls off a cliff (HeCBench minimod).

#define N 16

__global__ void k1(float *Out, int I) {
  __shared__ float S[N][N][N];
  float *P = &S[0][0][0];
  P[I] = 1.f;
  *Out = S[0][0][0];
}

__global__ void k2(float *Out, int I) {
  __shared__ float T[N][N][N];
  float *P = &T[0][0][0];
  P[I] = 2.f;
  *Out = T[0][0][0];
}

int main() {
  float *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(float)));
  k1<<<1, 1>>>(Out, 0);
  CHECK_HIP(hipDeviceSynchronize());
  k2<<<1, 1>>>(Out, 0);
  CHECK_HIP(hipDeviceSynchronize());
  hipFree(Out);
  return 0;
}

// CHECK: .group_segment_fixed_size: 16384
// CHECK: .group_segment_fixed_size: 16384
