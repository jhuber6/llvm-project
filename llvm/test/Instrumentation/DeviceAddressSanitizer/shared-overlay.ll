; RUN: opt < %s -passes=dasan -S | FileCheck %s
; RUN: opt < %s -passes=dasan,amdgpu-lower-module-lds -S | FileCheck %s --check-prefix=LDS

target datalayout = "e-p:64:64:64-p3:32:32-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; Two kernels, each with a 16 KiB tile. Instrumentation must not mention the
; other kernel's global, or amdgpu-lower-module-lds cannot overlay them.

@a = internal addrspace(3) global [4096 x float] poison
@b = internal addrspace(3) global [4096 x float] poison

define amdgpu_kernel void @k1(i32 %i) sanitize_device_address {
; CHECK-LABEL: define amdgpu_kernel void @k1(
; CHECK-NOT: ptrtoint {{.*}} @b
; CHECK: ret void
  %s = alloca ptr addrspace(3), addrspace(5)
  %p = getelementptr [4096 x float], ptr addrspace(3) @a, i32 0, i32 %i
  store ptr addrspace(3) %p, ptr addrspace(5) %s
  %q = load ptr addrspace(3), ptr addrspace(5) %s
  store float 1.0, ptr addrspace(3) %q
  ret void
}

define amdgpu_kernel void @k2(i32 %i) sanitize_device_address {
; CHECK-LABEL: define amdgpu_kernel void @k2(
; CHECK-NOT: ptrtoint {{.*}} @a
; CHECK: ret void
  %s = alloca ptr addrspace(3), addrspace(5)
  %p = getelementptr [4096 x float], ptr addrspace(3) @b, i32 0, i32 %i
  store ptr addrspace(3) %p, ptr addrspace(5) %s
  %q = load ptr addrspace(3), ptr addrspace(5) %s
  store float 1.0, ptr addrspace(3) %q
  ret void
}

; LDS: "amdgpu-lds-size"="16384"
; LDS-NOT: "amdgpu-lds-size"="32768"
