// Which report entry point the instrumentation calls is how a report being fatal
// is expressed: the plain one may end the program, and asking to recover picks
// the one that never does.  The choice is recorded in the module, because the
// post-link half of the pass runs from the backend, long after the request.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 \
// RUN:   -fsanitize=gpuasan -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 \
// RUN:   -fsanitize=gpuasan -fsanitize-recover=gpuasan -emit-llvm -o - %s \
// RUN:   | FileCheck --check-prefix=RECOVER %s

// CHECK: call void @__gpuasan_report_store(
// CHECK-NOT: !"gpuasan.recover"

// RECOVER: call void @__gpuasan_report_store_noabort(
// RECOVER: !{i32 4, !"gpuasan.recover", i32 1}

__attribute__((amdgpu_kernel)) void kernel(int *p, int i) { p[i] = 1; }
