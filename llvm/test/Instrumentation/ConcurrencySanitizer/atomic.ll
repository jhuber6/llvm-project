; Atomics use the same __csan_readN / __csan_writeN callbacks as plain
; accesses, with AF_Atomic (1) and AF_Compound (2) in the flags argument.
; The original instruction keeps its ordering, syncscope, and address space.
; RUN: opt < %s -passes='function(csan)' -S | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

define i32 @atomic_load(ptr addrspace(1) %a) sanitize_concurrency {
entry:
  %v = load atomic i32, ptr addrspace(1) %a seq_cst, align 4
  ret i32 %v
}
; CHECK-LABEL: @atomic_load(
; CHECK: call void @__csan_read4(ptr %{{.*}}, i32 1)
; CHECK-NEXT: %v = load atomic i32, ptr addrspace(1) %a seq_cst, align 4

define void @atomic_store(ptr addrspace(1) %a, i32 %v) sanitize_concurrency {
entry:
  store atomic i32 %v, ptr addrspace(1) %a syncscope("agent") release, align 4
  ret void
}
; CHECK-LABEL: @atomic_store(
; CHECK: call void @__csan_write4(ptr %{{.*}}, i32 1)
; CHECK-NEXT: store atomic i32 %v, ptr addrspace(1) %a syncscope("agent") release, align 4

define i32 @atomic_rmw(ptr addrspace(1) %a, i32 %v) sanitize_concurrency {
entry:
  %old = atomicrmw add ptr addrspace(1) %a, i32 %v syncscope("workgroup") seq_cst
  ret i32 %old
}
; CHECK-LABEL: @atomic_rmw(
; CHECK: call void @__csan_write4(ptr %{{.*}}, i32 3)
; CHECK-NEXT: %old = atomicrmw add ptr addrspace(1) %a, i32 %v syncscope("workgroup") seq_cst

define i32 @atomic_cas(ptr addrspace(1) %a, i32 %cmp, i32 %new) sanitize_concurrency {
entry:
  %pair = cmpxchg ptr addrspace(1) %a, i32 %cmp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}
; CHECK-LABEL: @atomic_cas(
; CHECK: call void @__csan_write4(ptr %{{.*}}, i32 3)
; CHECK-NEXT: %pair = cmpxchg ptr addrspace(1) %a, i32 %cmp, i32 %new seq_cst seq_cst

define void @atomic_fence() sanitize_concurrency {
entry:
  fence seq_cst
  ret void
}
; CHECK-LABEL: @atomic_fence(
; CHECK-NOT: __csan_read
; CHECK-NOT: __csan_write
; CHECK: fence seq_cst

define i32 @suppressed(ptr addrspace(1) %a) {
entry:
  %v = load atomic i32, ptr addrspace(1) %a seq_cst, align 4
  ret i32 %v
}
; CHECK-LABEL: @suppressed(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load atomic i32, ptr addrspace(1) %a seq_cst, align 4
; CHECK-NEXT: ret i32 %v

; CHECK-NOT: @__tsan
; CHECK-NOT: @__csan_atomic
