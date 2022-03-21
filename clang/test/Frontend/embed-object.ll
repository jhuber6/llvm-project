; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-offload-object=file=%S/Inputs/empty.h -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK

; CHECK: @[[OBJECT:.+]] = private constant [104 x i8] c"\10\FF\10\AD{{.*}}\00", section ".llvm.offloading", align 8
; CHECK: @llvm.compiler.used = appending global [2 x i8*] [i8* @x, i8* getelementptr inbounds ([104 x i8], [104 x i8]* @[[OBJECT]], i32 0, i32 0)], section "llvm.metadata"

@x = private constant i8 1
@llvm.compiler.used = appending global [1 x i8*] [i8* @x], section "llvm.metadata"

define i32 @foo() {
  ret i32 0
}
