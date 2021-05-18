; RUN: opt -passes=openmp-opt-cgscc -debug-only=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; ModuleID = 'single_threaded_exeuction.c'

; CHECK: [openmp-opt] Basic block @bar entry is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define internal void @bar() {
entry:
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @foo entry is executed by a single thread.
; CHECK: [openmp-opt] Basic block @foo if.then is executed by a single thread.
; CHECK-NOT: [openmp-opt] Basic block @foo if.end is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define dso_local void @foo() {
entry:
  %dummy = call i32 @omp_get_thread_num()
  %call = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @bar()
  br label %if.end

if.end:
  ret void
}

declare dso_local i32 @omp_get_thread_num()

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 13.0.0"}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
