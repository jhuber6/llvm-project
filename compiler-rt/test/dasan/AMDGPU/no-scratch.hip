// RUN: %clang_hip_dasan -O1 --offload-device-only --no-gpu-bundle-output \
// RUN:   -foffload-lto %s -o %t.elf
// RUN: llvm-readelf --notes %t.elf | FileCheck %s \
// RUN:   --implicit-check-not='private_segment_fixed_size: {{[1-9]}}' \
// RUN:   --implicit-check-not='uses_dynamic_stack: true'

// The report handlers are leaves that end the wave rather than trap, so they
// add neither scratch nor a dynamic stack to the kernels that call them.

__global__ void touch(int *P) { P[0] = P[1]; }

// CHECK: .private_segment_fixed_size: 0
// CHECK: .uses_dynamic_stack: false
