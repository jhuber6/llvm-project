//===-- csan_defs.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared CSan constants: probe flags and report kinds. The pass's AccessFlags
// (ATOMIC, COMPOUND) must match the trailing i32 on __csan_{read,write}*.
// WRITE is implied by the write callbacks and set by the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CSAN_DEFS_H
#define CSAN_DEFS_H

enum {
  CSAN_ACCESS_ATOMIC = 1u << 0,
  CSAN_ACCESS_COMPOUND = 1u << 1,
  CSAN_ACCESS_WRITE = 1u << 2
};

enum {
  CSAN_RACE_DATA = 0,
  CSAN_RACE_UNKNOWN_ORIGIN = 1,
  CSAN_RACE_INTRA_WAVE = 2
};

#endif // CSAN_DEFS_H
