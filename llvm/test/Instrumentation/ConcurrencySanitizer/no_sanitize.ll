; The concurrency pass keys off sanitize_concurrency and ignores
; sanitize_thread, so the two never instrument the same function.
; RUN: opt < %s -passes='function(csan)' -S | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

define i32 @unattributed(ptr addrspace(1) %a) {
entry:
  %v = load i32, ptr addrspace(1) %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @unattributed(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr addrspace(1) %a, align 4
; CHECK-NEXT: ret i32 %v

define i32 @thread_only(ptr addrspace(1) %a) sanitize_thread {
entry:
  %v = load i32, ptr addrspace(1) %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @thread_only(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr addrspace(1) %a, align 4
; CHECK-NEXT: ret i32 %v

define i32 @instrumented(ptr addrspace(1) %a) sanitize_concurrency {
entry:
  %v = load i32, ptr addrspace(1) %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @instrumented(
; CHECK: call void @__csan_read4(ptr %{{.*}}, i32 0)

define i32 @disabled(ptr addrspace(1) %a) sanitize_concurrency disable_sanitizer_instrumentation {
entry:
  %v = load i32, ptr addrspace(1) %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @disabled(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr addrspace(1) %a, align 4
; CHECK-NEXT: ret i32 %v
