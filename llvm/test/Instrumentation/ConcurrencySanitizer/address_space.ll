; Address spaces that can race are a target property. AMDGPU is primary:
; global (1) and LDS (3) race; private (5) does not. The callback argument
; is generic; the original access keeps its space. Host (AS 0 only) is a
; secondary check.
; RUN: opt < %s -passes='function(csan)' -S -mtriple=amdgcn-amd-amdhsa | FileCheck %s --check-prefixes=CHECK,GCN
; RUN: opt < %s -passes='function(csan)' -S -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefixes=CHECK,CPU

define void @load_as1(ptr addrspace(1) %p) sanitize_concurrency {
entry:
  %v = load i32, ptr addrspace(1) %p, align 4
  ret void
}
; GCN-LABEL: @load_as1(
; GCN: %[[GEN:.*]] = addrspacecast ptr addrspace(1) %p to ptr
; GCN: call void @__csan_read4(ptr %[[GEN]], i32 0)
; GCN: %v = load i32, ptr addrspace(1) %p, align 4
; CPU-LABEL: @load_as1(
; CPU-NEXT: entry:
; CPU-NEXT: %v = load i32, ptr addrspace(1) %p, align 4
; CPU-NEXT: ret void

define void @load_lds(ptr addrspace(3) %p) sanitize_concurrency {
entry:
  %v = load i32, ptr addrspace(3) %p, align 4
  ret void
}
; GCN-LABEL: @load_lds(
; GCN: %[[GEN:.*]] = addrspacecast ptr addrspace(3) %p to ptr
; GCN: call void @__csan_read4(ptr %[[GEN]], i32 0)
; GCN: %v = load i32, ptr addrspace(3) %p, align 4
; CPU-LABEL: @load_lds(
; CPU-NEXT: entry:
; CPU-NEXT: %v = load i32, ptr addrspace(3) %p, align 4
; CPU-NEXT: ret void

define void @load_private(ptr addrspace(5) %p) sanitize_concurrency {
entry:
  %v = load i32, ptr addrspace(5) %p, align 4
  ret void
}
; CHECK-LABEL: @load_private(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr addrspace(5) %p, align 4
; CHECK-NEXT: ret void

define void @load_buffer_fat(ptr addrspace(7) %p) sanitize_concurrency {
entry:
  %v = load i32, ptr addrspace(7) %p, align 4
  ret void
}
; CHECK-LABEL: @load_buffer_fat(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr addrspace(7) %p, align 4
; CHECK-NEXT: ret void

define void @memcpy_private(ptr addrspace(5) %dst, ptr addrspace(5) %src, i64 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) %dst, ptr addrspace(5) %src, i64 %n, i1 false)
  ret void
}
; GCN-LABEL: @memcpy_private(
; GCN-NOT: __csan_read
; GCN-NOT: __csan_write
; GCN: call void @llvm.memcpy
; CPU-LABEL: @memcpy_private(
; CPU-NOT: __csan_read
; CPU-NOT: __csan_write
; CPU: call void @llvm.memcpy

define i32 @scoped_atomic(ptr addrspace(1) %p) sanitize_concurrency {
entry:
  %v = load atomic i32, ptr addrspace(1) %p syncscope("agent") seq_cst, align 4
  ret i32 %v
}
; GCN-LABEL: @scoped_atomic(
; GCN: call void @__csan_read4(ptr %{{.*}}, i32 1)
; GCN: %v = load atomic i32, ptr addrspace(1) %p syncscope("agent") seq_cst, align 4
; CPU-LABEL: @scoped_atomic(
; CPU-NEXT: entry:
; CPU-NEXT: %v = load atomic i32, ptr addrspace(1) %p syncscope("agent") seq_cst, align 4
; CPU-NEXT: ret i32 %v

declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5), ptr addrspace(5), i64, i1)
