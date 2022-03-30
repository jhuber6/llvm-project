// RUN: not %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - 2>&1 | FileCheck %s

// CHECK: definition with same mangled name '_Z3foov' as another definition

#ifndef HEADER
#define HEADER

int foo() { return 0; }

#pragma omp begin declare variant match( \
    device = {arch(ppc64le, ppc64)},     \
    implementation = {extension(match_any, keep_original_name)})

int foo() { return 1; }

#pragma omp end declare variant

#endif
