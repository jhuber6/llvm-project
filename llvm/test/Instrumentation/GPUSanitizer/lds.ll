; RUN: opt < %s -passes=gpuasan,amdgpu-lower-module-lds,gpuasan-lds -S | FileCheck %s

; REQUIRES: amdgpu-registered-target

; LDS is checked in two halves either side of the pass that decides its layout.
; The first widens each object so a granule belongs to exactly one of them and
; an overflow has somewhere poisoned to land; the second, running once the
; addresses are known, builds the shadow and emits the checks.

target datalayout = "e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-n32:64"
target triple = "amdgcn-amd-amdhsa"

@A = internal addrspace(3) global [4 x i32] poison, align 4
@B = internal addrspace(3) global [4 x i32] poison, align 4

; Two 16-byte objects, each followed by a 16-byte redzone. One shadow byte per
; granule holds the distance to the end of the object occupying it, so the
; redzones read as zero and reject any access. The trailing entry is what a
; pointer past the end of LDS clamps onto.
;
; CHECK: @gpuasan.lds.shadow = private unnamed_addr addrspace(4) constant [5 x i8] c"\10\00\10\00\00"

; CHECK-LABEL: define amdgpu_kernel void @k(
; CHECK: %gpuasan.ldsoff = ptrtoint ptr addrspace(3) %P to i32
; CHECK: %gpuasan.granule = lshr i32 %gpuasan.ldsoff, 4
; CHECK: call i32 @llvm.umin.i32(i32 %gpuasan.granule, i32 4)
; CHECK: %gpuasan.ldsentry = load i8, ptr addrspace(4) %{{[0-9]+}}, align 1
; CHECK: %gpuasan.rem = and i32 %gpuasan.ldsoff, 15
; CHECK: %gpuasan.ldsbad = icmp ugt i32 %{{[0-9]+}}, %{{[0-9]+}}
; CHECK: br i1 %gpuasan.ldsbad
; CHECK: call { i32, i32 } @gpuasan.lds.object(i32 %gpuasan.ldsoff)
; CHECK: call void @__gpuasan_report_store(ptr %{{[0-9]+}}, i64 4, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 3)
; CHECK: store i32 1, ptr addrspace(3) %P
define amdgpu_kernel void @k(i32 %I) {
  %P = getelementptr [4 x i32], ptr addrspace(3) @B, i32 0, i32 %I
  store i32 1, ptr addrspace(3) %P, align 4
  %Q = getelementptr [4 x i32], ptr addrspace(3) @A, i32 0, i32 %I
  store i32 2, ptr addrspace(3) %Q, align 4
  ret void
}

; The shadow cannot name the object it rejected an access against, so a report
; recovers it from the offset. Each redzone is split down the middle: a hit in
; the lower half is an overflow of the object below it and one in the upper
; half is an underflow of the object above, which is the only way the second
; object in a block is ever named for an access below its base.
;
; CHECK-LABEL: define internal { i32, i32 } @gpuasan.lds.object(
; CHECK: %[[HIT:[0-9]+]] = icmp uge i32 %0, 24
; CHECK: select i1 %[[HIT]], i32 32, i32 0
; CHECK: select i1 %[[HIT]], i32 16, i32 16
