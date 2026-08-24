; RUN: opt < %s -passes=dasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; CHECK-LABEL: define i32 @located
; CHECK: call void @__dasan_report_load({{.*}}), !dbg [[LOC:![0-9]+]]
define i32 @located(ptr addrspace(1) %p) sanitize_device_address !dbg !4 {
  %v = load i32, ptr addrspace(1) %p, align 4, !dbg !8
  ret i32 %v, !dbg !8
}

; CHECK-LABEL: define i32 @unlocated
; CHECK: call void @__dasan_report_load({{.*}}), !dbg [[SCOPE:![0-9]+]]
define i32 @unlocated(ptr addrspace(1) %p) sanitize_device_address !dbg !9 {
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}

; CHECK-LABEL: define i32 @nodebug
; CHECK: ret i32
; CHECK: call void @__dasan_report_load({{.*}}), !nosanitize
define i32 @nodebug(ptr addrspace(1) %p) sanitize_device_address {
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}

; CHECK-DAG: [[LOC]] = !DILocation(line: 6, scope: !{{[0-9]+}})
; CHECK-DAG: [[SCOPE]] = !DILocation(line: 11, scope: !{{[0-9]+}})

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "located", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 7, column: 3, scope: !4)
!9 = distinct !DISubprogram(name: "unlocated", scope: !1, file: !1, line: 10, type: !5, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0)
