//===-- dasan_report.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Failed-access report. ABI between the device runtime and the host.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_REPORT_H
#define DASAN_REPORT_H

#include <stdint.h>

#define DASAN_RPC_BASE 'd'
#define DASAN_OPCODE(n) ((DASAN_RPC_BASE << 24) | (n))

typedef enum {
  DASAN_REPORT = DASAN_OPCODE(0),
} dasan_opcode_t;

typedef enum {
  DASAN_KIND_SPACE = 0,
  DASAN_KIND_SHARED = 1,
} dasan_kind_t;

typedef struct dasan_report_t {
  uint64_t Addr;       // Address accessed.
  uint64_t Base;       // Base of the allocation its chunk holds.
  uint64_t AllocSize;  // That allocation's size, zero once it is freed.
  uint64_t Pc;         // Where the access was, as the wave saw it.
  uint32_t AccessSize;
  uint32_t Block[3];
  uint16_t Thread[3];
  uint8_t Lane;
  uint8_t IsWrite;
  uint8_t Freed;
  uint8_t Kind;  // dasan_kind_t.
  uint8_t Recover;
} dasan_report_t;

#endif  // DASAN_REPORT_H
