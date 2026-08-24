// Test that DASan runs with the new pass manager. We run it under different
// optimizations to ensure the IR is still being instrumented properly.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - -fsanitize=daddress %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - -O1 -fsanitize=daddress %s | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -fsanitize=daddress %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress %s | FileCheck %s

// Abort/trap share a fail block per report kind; recover keeps a diamond
// per access so the load can still execute. Both pass an explicit PC.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress %s | FileCheck %s --check-prefix=ABORT
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress -mllvm -dasan-trap-on-error %s | FileCheck %s --check-prefix=TRAP
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress -fsanitize-recover=daddress %s | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -fsanitize=daddress %s | FileCheck %s --check-prefix=ABORT-SHARED

int foo(int *a, int *b) { return *a + *b; }
int bar(int __attribute__((address_space(3))) *s) { return *s; }

// CHECK: sanitize_device_address
// CHECK: !{i32 4, !"nosanitize_device_address", i32 1}

// ABORT:      br i1 {{.*}}, label %dasan.fail,
// ABORT:      br i1 {{.*}}, label %dasan.fail,
// ABORT:      dasan.fail:
// ABORT:      phi i64
// ABORT:      call void @__dasan_report_load(
// ABORT-NEXT: unreachable
// ABORT:      declare void @__dasan_report_load(ptr, i64, i64, ptr) [[ATTR:#[0-9]+]]
// ABORT:      attributes [[ATTR]] = { cold noreturn nounwind }

// TRAP:       br i1 {{.*}}, label %dasan.trap,
// TRAP:       br i1 {{.*}}, label %dasan.trap,
// TRAP:       dasan.trap:
// TRAP:       call void @llvm.trap()
// TRAP-NEXT:  unreachable

// Recovering is the only build in which the access after a report happens.
// RECOVER:     call void @__dasan_report_load_noabort(
// RECOVER-NOT: unreachable
// RECOVER:     call void @__dasan_report_load_noabort(
// RECOVER:     declare void @__dasan_report_load_noabort(ptr, i64, i64, ptr) [[RATTR:#[0-9]+]]
// RECOVER:     attributes [[RATTR]] = { cold nounwind }

// Unsized LDS needs the dispatch pointer, which -O1 infers away on this TU.
// ABORT-SHARED:      define {{.*}} @bar(
// ABORT-SHARED:      dasan.fail:
// ABORT-SHARED:      call void @__dasan_report_shared_load(
// ABORT-SHARED-NEXT: unreachable
