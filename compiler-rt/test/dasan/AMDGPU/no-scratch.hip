// RUN: %clang_dasan_hip_opt %s -o %t %hip_libs
// RUN: llvm-objcopy --dump-section=.hip_fatbin=%t.fatbin %t /dev/null
// RUN: %offload_bundler --unbundle --type=o \
// RUN:   --targets=hipv4-amdgcn-amd-amdhsa--%amdgpu_arch \
// RUN:   --input=%t.fatbin --output=%t.elf
// RUN: llvm-readelf --notes %t.elf | FileCheck %s \
// RUN:   --implicit-check-not='private_segment_fixed_size: {{[1-9]}}' \
// RUN:   --implicit-check-not='uses_dynamic_stack: true'

// An instrumented kernel must not acquire scratch. Checked at -O1: at -O0 an
// alloca in the kernel is a private array and this note would fail for the
// wrong reason. A kernel's private segment has to cover its whole call graph,
// so a report handler needing a frame of its own hands every kernel in the
// module scratch it never touches, which can cost an occupancy tier before a
// single check has failed.
//
// The handlers are leaves for exactly this reason: everything they would call is
// inlined into them instead, so none of them has to preserve a return address
// across a call. A helper call added to one would show up right here.
//
// A kernel must not claim a dynamic stack either, which it does as soon as a
// callee is not `norecurse`. The handlers end their wave rather than trapping so
// that every call they make is to something `nocallback`, which is what lets
// that be inferred; `llvm.trap` is not, and reintroducing it would show up here
// too.

#include "hip.h"

__global__ void touch(int *P) { P[0] = P[1]; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  touch<<<1, 1>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  hipFree(P);
  return 0;
}

// CHECK: .private_segment_fixed_size: 0
// CHECK: .uses_dynamic_stack: false
