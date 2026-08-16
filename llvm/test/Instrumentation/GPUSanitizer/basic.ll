; RUN: opt < %s -passes=gpuasan -S | FileCheck %s
; RUN: opt < %s -passes='gpuasan,gpuasan-lds' -S | FileCheck --check-prefix=POST %s
; RUN: opt < %s -passes='gpuasan,gpuasan-lds' -gpuasan-recover -S \
; RUN:   | FileCheck --check-prefix=RECOVER %s

; The placed-memory check: recover the slot from the pointer, read the extent
; from the table, and call the handler only when the access leaves it.

target datalayout = "e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-n32:64"
target triple = "amdgcn-amd-amdhsa"

@Global = protected addrspace(1) global [16 x i32] zeroinitializer, align 4
@Shared = internal addrspace(3) global [8 x i32] poison, align 4

; An LDS object is only widened here; the checks come once the layout is
; decided.
;
; CHECK: %gpuasan.lds = type { [8 x i32], [16 x i8] }

; A global this module could check is only marked here.  Its redzone and its
; table entry depend on where the linker puts it, so both have to wait until
; every module's globals are in one image.
;
; CHECK: @Global = protected addrspace(1) global [16 x i32] zeroinitializer, align 4, !gpuasan.checkable

; The handlers must survive whatever decides which symbols are live, because the
; second half may add the only calls to them.
;
; CHECK: @llvm.compiler.used = {{.*}}@__gpuasan_report_load{{.*}}@__gpuasan_report_store
; CHECK: @Shared = internal addrspace(3) global %gpuasan.lds poison, align 16

; CHECK-LABEL: define amdgpu_kernel void @global_access(
; CHECK: getelementptr [16 x i32], ptr addrspace(1) @Global
define amdgpu_kernel void @global_access(i32 %I) {
  %P = getelementptr [16 x i32], ptr addrspace(1) @Global, i32 0, i32 %I
  store i32 1, ptr addrspace(1) %P, align 4
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @heap_access(
; CHECK: %gpuasan.addr = ptrtoint ptr addrspace(1) %P to i64
; CHECK: %gpuasan.delta = sub i64 %gpuasan.addr, 35184372088832
; CHECK: %gpuasan.class = lshr i64 %gpuasan.delta, 41
; CHECK: %gpuasan.inregion = icmp ult i64 %gpuasan.class, 25
; CHECK: br i1 %gpuasan.inregion
; CHECK: %gpuasan.k = add i64 %gpuasan.class, 12
; CHECK: %gpuasan.slotoff = and i64 %gpuasan.addr, %gpuasan.slotmask
; CHECK: %gpuasan.index = lshr i64 %{{[0-9]+}}, %gpuasan.k
; The table is constant memory: only the host writes it, and a uniform index can
; then be loaded into a scalar register.  The load is marked so that the
; post-link half does not go on to check the check.
; CHECK: %gpuasan.entryptr = getelementptr i64, ptr addrspace(4) inttoptr (i64 105553116266496 to ptr addrspace(4)), i64 %gpuasan.id
; CHECK: %gpuasan.entry = load i64, ptr addrspace(4) %gpuasan.entryptr, align 8, !nosanitize
; CHECK: %gpuasan.off = sub i64 %gpuasan.slotoff, %gpuasan.color
; Not `off + size > extent`: an access below the base wraps, and the additive
; form would wrap it back into bounds.
; CHECK: %gpuasan.pastend = icmp uge i64 %gpuasan.off, %gpuasan.size
; CHECK: %gpuasan.remain = sub i64 %gpuasan.size, %gpuasan.off
; CHECK: %gpuasan.toowide = icmp ult i64 %gpuasan.remain, 4
; CHECK: %gpuasan.bad = or i1 %gpuasan.pastend, %gpuasan.toowide
; A freed slot keeps its extent and sets the top bit, so one signed compare
; separates a use-after-free from a slot that was never handed out -- and the
; entry still says how big the dead allocation was.
; CHECK: %gpuasan.poisoned = icmp slt i64 %gpuasan.entry, 0
; CHECK: %gpuasan.bad.poison = or i1 %gpuasan.bad.range, %gpuasan.poisoned
; CHECK: br i1 %gpuasan.bad.poison
; CHECK: %gpuasan.flags = shl i32 %{{[0-9]+}}, 8
; CHECK: call void @__gpuasan_report_store(ptr %{{[0-9]+}}, i64 4, ptr %{{[0-9]+}}, i64 %gpuasan.size, i32 %gpuasan.flags)
; CHECK: store i32 1, ptr addrspace(1) %P
define amdgpu_kernel void @heap_access(ptr addrspace(1) %P) {
  store i32 1, ptr addrspace(1) %P, align 4
  ret void
}

; Nothing outside the region can name a slot, so scratch is left alone.
;
; CHECK-LABEL: define amdgpu_kernel void @private_access(
; CHECK-NOT: gpuasan
; CHECK: ret void
define amdgpu_kernel void @private_access() {
  %A = alloca i32, align 4, addrspace(5)
  store i32 1, ptr addrspace(5) %A, align 4
  ret void
}

; Neither handler is `noreturn`: whether a report ends the program is the
; runtime's answer to it, not a property of this module, so the access the check
; guards stays in the code and the stop happens inside the handler.
;
; CHECK: declare void @__gpuasan_report_load(ptr, i64, ptr, i64, i32) #[[HANDLER:[0-9]+]]
; CHECK: declare void @__gpuasan_report_store(ptr, i64, ptr, i64, i32) #[[HANDLER]]
; CHECK: attributes #[[HANDLER]] = { cold disable_sanitizer_instrumentation nounwind }

; Asking to recover picks the entry points that never stop, and the choice is
; recorded so that the post-link half calls the same ones.
;
; RECOVER: call void @__gpuasan_report_store_noabort(
; RECOVER: call void @__gpuasan_report_store_noabort(
; RECOVER: !{i32 4, !"gpuasan.recover", i32 1}

; Post-link, the marked global keeps its address and its name.  The storage is
; the variable followed by a 16-byte redzone, private so that nothing links
; against it, and the name becomes an alias to its base -- which is what keeps
; the symbol's address, linkage, visibility and size the same as they were.
;
; POST: @Global.gpuasan = private addrspace(1) global { [16 x i32], [16 x i8] } zeroinitializer, section "__gpuasan_globals", align 16

; The linker's own bounds on the section it laid out.  Nothing is loaded to find
; the section: both are addresses in this image, so they materialize into scalar
; registers, and their difference is its extent.
;
; POST: @__start___gpuasan_globals = external hidden addrspace(1) constant i8
; POST: @__stop___gpuasan_globals = external hidden addrspace(1) constant i8

; One entry per granule of the section, filled by the host, plus a descriptor
; per variable telling it which granules belong to what.  The table is poisoned
; until the host says otherwise, so a granule that is only padding rejects
; everything.
;
; POST: @gpuasan.globals.table = private addrspace(1) externally_initialized global [5 x i64] zeroinitializer, align 16
; POST: @gpuasan.globals.name = private unnamed_addr addrspace(4) constant [6 x i8] c"Global"
; POST: @gpuasan.globals.desc = private addrspace(1) global [1 x { i64, i64, i64, i64 }] [{ i64, i64, i64, i64 } { i64 ptrtoint (ptr addrspace(1) @Global.gpuasan to i64), i64 64, i64 ptrtoint (ptr addrspace(4) @gpuasan.globals.name to i64), i64 6 }]

; The image's one externally visible symbol, which only the host reads: it finds
; it by name and everything else it needs hangs off it, starting with the origin
; the table is indexed from.  Weak, because every image linked from these modules
; carries one and only one survives.
;
; POST: @__gpuasan_globals_info.{{[0-9a-f]+}} = weak_odr protected addrspace(1) global [5 x i64] [i64 ptrtoint (ptr addrspace(1) @__start___gpuasan_globals to i64), i64 ptrtoint (ptr addrspace(1) @gpuasan.globals.table to i64), i64 5, i64 1, i64 ptrtoint (ptr addrspace(1) @gpuasan.globals.desc to i64)]

; POST: @Global = protected alias [16 x i32], ptr addrspace(1) @Global.gpuasan

; POST-LABEL: define amdgpu_kernel void @global_access(
; Clamped to what the table can name, which is what keeps a check from indexing
; off the end of its own table when an image is several modules sharing the
; section.
; POST: %gpuasan.globals.span = call i64 @llvm.umin.i64(i64 sub (i64 ptrtoint (ptr addrspace(1) @__stop___gpuasan_globals to i64), i64 ptrtoint (ptr addrspace(1) @__start___gpuasan_globals to i64)), i64 80)
; POST: %gpuasan.globals.delta = sub i64 %gpuasan.globals.addr, ptrtoint (ptr addrspace(1) @__start___gpuasan_globals to i64)
; POST: %gpuasan.globals.insection = icmp ult i64 %gpuasan.globals.delta, %gpuasan.globals.span
; POST: br i1 %gpuasan.globals.insection
; POST: %gpuasan.granule = lshr i64 %gpuasan.globals.delta, 4
; POST: %gpuasan.globals.entryptr = getelementptr i64, ptr addrspace(1) @gpuasan.globals.table, i64 %gpuasan.granule
; POST: %gpuasan.globals.entry = load i64, ptr addrspace(1) %gpuasan.globals.entryptr, align 8, !invariant.load
; The end is stored complemented, so the zero entry a table starts out as is the
; one that permits nothing and the host does not have to fill the padding.
; POST: %gpuasan.globals.begin = and i64 %gpuasan.globals.entry, 4294967295
; POST: %gpuasan.globals.end = lshr i64 %{{[0-9]+}}, 32
; POST: %gpuasan.globals.under = icmp ult i64 %gpuasan.globals.delta, %gpuasan.globals.begin
; POST: %gpuasan.globals.over = icmp ugt i64 %{{[0-9]+}}, %gpuasan.globals.end
; POST: br i1 %gpuasan.globals.bad
; POST: call void @__gpuasan_report_store(ptr %{{[0-9]+}}, i64 4, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 1)

; A pointer of unknown provenance could be either, so it is checked against both
; the heap table and the globals table.
;
; POST-LABEL: define amdgpu_kernel void @heap_access(
; POST: call void @__gpuasan_report_store(ptr %{{[0-9]+}}, i64 4, ptr %{{[0-9]+}}, i64 %gpuasan.size, i32 %gpuasan.flags)
; POST: %gpuasan.globals.insection = icmp ult i64 %gpuasan.globals.delta, %gpuasan.globals.span
; POST: call void @__gpuasan_report_store(ptr %{{[0-9]+}}, i64 4, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 1)
