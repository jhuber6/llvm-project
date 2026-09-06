//===-- asan_offload_packet.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offload/host RPC packet for ASan reports.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_PACKET_H
#define ASAN_OFFLOAD_PACKET_H

#include <stdint.h>

#define ASAN_OFFLOAD_REPORT_OPCODE (('a' << 24) | 0)

struct __asan_offload_report {
  uint64_t pc;
  uint64_t addr;
  uint64_t size;
  uint8_t is_write;
  uint8_t fatal;
  uint8_t reserved[6];
  uint8_t pad[32];
};

static_assert(sizeof(__asan_offload_report) == 64,
              "Offload ASan report must fit one RPC packet");

#endif // ASAN_OFFLOAD_PACKET_H
