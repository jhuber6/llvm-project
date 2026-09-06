; RUN: opt -O0 -S -mtriple=amdgpu-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=OPTNONE %s
; RUN: opt -passes='default<O0>' -S -mtriple=amdgpu-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=OPTNONE %s
; RUN: opt -O1 -S -mtriple=amdgpu-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=ASAN_DCE %s
; RUN: opt -passes='default<O1>' -S -mtriple=amdgpu-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=ASAN_DCE %s

; Unused sanitizer runtime shims may be internalized and dropped. The device
; archive is linked after instrumentation, so LTO DCE is required.

; OPTNONE: define void @__asan_no_explicit_linkage(
define void @__asan_no_explicit_linkage() {
entry:
  ret void
}

; OPTNONE: define weak void @__asan_weak_linkage(
define weak void @__asan_weak_linkage() {
entry:
  ret void
}

; OPTNONE: define void @__sanitizer_no_explicit_linkage(
define void @__sanitizer_no_explicit_linkage() {
entry:
  ret void
}

; OPTNONE: define weak void @__sanitizer_weak_linkage(
define weak void @__sanitizer_weak_linkage() {
entry:
  ret void
}

; ASAN_DCE-NOT: @__asan_no_explicit_linkage
; ASAN_DCE-NOT: @__asan_weak_linkage
; ASAN_DCE-NOT: @__sanitizer_no_explicit_linkage
; ASAN_DCE-NOT: @__sanitizer_weak_linkage
