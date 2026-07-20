; AMDGPU is the primary target. Probes go in before the access; the original
; instruction is left in place. Device builds do not emit func_entry or a
; module ctor (those are host runtime).
; RUN: opt < %s -passes='module(csan-module),function(csan)' -S | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

define void @read_write(ptr addrspace(1) %a, ptr addrspace(1) %b) sanitize_concurrency {
entry:
  %v = load i32, ptr addrspace(1) %a, align 4
  store i32 %v, ptr addrspace(1) %b, align 4
  ret void
}
; CHECK-LABEL: @read_write(
; CHECK: %[[A:.*]] = addrspacecast ptr addrspace(1) %a to ptr
; CHECK: call void @__csan_read4(ptr %[[A]], i32 0)
; CHECK-NEXT: %v = load i32, ptr addrspace(1) %a, align 4
; CHECK: %[[B:.*]] = addrspacecast ptr addrspace(1) %b to ptr
; CHECK: call void @__csan_write4(ptr %[[B]], i32 0)
; CHECK-NEXT: store i32 %v, ptr addrspace(1) %b, align 4
; CHECK-NOT: __csan_func_

define i16 @unaligned(ptr addrspace(1) %a) sanitize_concurrency {
entry:
  %v = load i16, ptr addrspace(1) %a, align 1
  ret i16 %v
}
; CHECK-LABEL: @unaligned(
; CHECK: call void @__csan_unaligned_read2(ptr %{{.*}}, i32 0)
; CHECK-NEXT: %v = load i16, ptr addrspace(1) %a, align 1

define void @memintrinsics(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n, i1 false)
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n, i1 false)
  call void @llvm.memset.p1.i64(ptr addrspace(1) %dst, i8 0, i64 %n, i1 false)
  ret void
}
; CHECK-LABEL: @memintrinsics(
; CHECK: call void @__csan_read_range(ptr %{{.*}}, i64 %n, i32 0)
; CHECK: call void @__csan_write_range(ptr %{{.*}}, i64 %n, i32 0)
; CHECK: call void @llvm.memcpy
; CHECK: call void @llvm.memmove
; CHECK: call void @llvm.memset

; CHECK-NOT: csan.module_ctor
; CHECK-NOT: @__tsan

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
