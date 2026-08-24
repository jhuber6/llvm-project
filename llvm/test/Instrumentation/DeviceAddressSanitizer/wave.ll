; RUN: opt < %s -passes=dasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: define i32 @load4(
; CHECK-NEXT:    %v = load i32, ptr addrspace(1) %a, align 4
; CHECK-NEXT:    ret i32 %v
; CHECK-NOT:     __dasan
define i32 @load4(ptr addrspace(1) %a) sanitize_device_address {
  %v = load i32, ptr addrspace(1) %a, align 4
  ret i32 %v
}
