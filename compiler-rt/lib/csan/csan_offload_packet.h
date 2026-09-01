//===-- csan_offload_packet.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offload/host RPC packet for CSan reports.
//
//===----------------------------------------------------------------------===//

#ifndef CSAN_OFFLOAD_PACKET_H
#define CSAN_OFFLOAD_PACKET_H

#include "csan_defs.h"
#include <stdint.h>

/// RPC opcode for a GPU race report. The high byte tags the sanitizer family
/// ('s') so distinct GPU tools can share the RPC channel. UBSan uses |0.
#define CSAN_OFFLOAD_REPORT_OPCODE (('s' << 24) | 1)

struct __csan_gpu_race {
  uint64_t pc;
  uint64_t peer_pc;
  uint64_t addr;
  uint32_t size;
  uint32_t access_type;
  uint32_t kind;
  uint32_t block[3];
  uint16_t thread[3];
  uint8_t lane;
  uint8_t peer_lane;
  uint8_t peer_access_type;
  uint8_t peer_size;
};

static_assert(sizeof(__csan_gpu_race) == 64,
              "Offload CSan report must fit one RPC packet");

#endif // CSAN_OFFLOAD_PACKET_H
