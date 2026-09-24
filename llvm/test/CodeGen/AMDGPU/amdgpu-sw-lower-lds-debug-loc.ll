; RUN: opt < %s -passes=amdgpu-sw-lower-lds,verify -S -amdgpu-asan-instrument-lds=false -mtriple=amdgcn-amd-amdhsa | FileCheck %s

; Calls into a linked runtime with debug info must carry a location even if the
; kernel's first instruction has none.
@lds = internal addrspace(3) global [4 x i32] poison, align 4

; CHECK-LABEL: define amdgpu_kernel void @k0(
; CHECK: call i64 @__asan_malloc_impl({{.*}}), !dbg [[DL:![0-9]+]]
; CHECK: call void @__asan_free_impl({{.*}}), !dbg [[DL]]
; CHECK: [[DL]] = !DILocation(line: 1, column: 1, scope: [[SP:![0-9]+]])
define amdgpu_kernel void @k0() sanitize_address !dbg !5 {
  store i32 1, ptr addrspace(3) @lds, align 4
  ret void
}

define i64 @__asan_malloc_impl(i64 %size, i64 %pc) !dbg !7 {
  ret i64 0
}

define void @__asan_free_impl(i64 %ptr, i64 %pc) !dbg !8 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.hip", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 4, !"nosanitize_address", i32 1}
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "k0", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "__asan_malloc_impl", scope: !1, file: !1, line: 2, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "__asan_free_impl", scope: !1, file: !1, line: 3, type: !4, spFlags: DISPFlagDefinition, unit: !0)
