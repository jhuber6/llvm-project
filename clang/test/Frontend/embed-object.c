// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -emit-llvm -fembed-offload-object=file=%S/Inputs/empty.h -o - | FileCheck %s

// CHECK: @[[OBJECT:.+]] = private constant [104 x i8] c"\10\FF\10\AD\01{{.*}}\00\00", section ".llvm.offloading", align 8
// CHECK: @llvm.compiler.used = appending global [1 x i8*] [i8* getelementptr inbounds ([104 x i8], [104 x i8]* @[[OBJECT]], i32 0, i32 0)], section "llvm.metadata"

void foo(void) {}
