; A vtable pointer is a store the language treats specially, so it keeps its own
; callbacks rather than going through the sized access path. This is a host C++
; concern; GPU kernels do not use Itanium vtables.
; RUN: opt < %s -passes='function(csan)' -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @vptr_update(ptr %obj, ptr %vptr) sanitize_concurrency {
entry:
  store ptr %vptr, ptr %obj, align 8, !tbaa !0
  ret void
}
; CHECK-LABEL: @vptr_update(
; CHECK: call void @__csan_vptr_update(ptr %obj, ptr %vptr)
; CHECK-NEXT: store ptr %vptr, ptr %obj, align 8

define ptr @vptr_read(ptr %obj) sanitize_concurrency {
entry:
  %v = load ptr, ptr %obj, align 8, !tbaa !0
  ret ptr %v
}
; CHECK-LABEL: @vptr_read(
; CHECK: call void @__csan_vptr_read(ptr %obj)
; CHECK-NEXT: %v = load ptr, ptr %obj, align 8

!0 = !{!1, !1, i64 0}
!1 = !{!"vtable pointer", !2}
!2 = !{!"Simple C++ TBAA"}
