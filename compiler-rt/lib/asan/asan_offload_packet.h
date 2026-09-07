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

/// RPC opcode for a GPU ASan report. The high byte tags the sanitizer family
/// ('s') so distinct GPU tools can share the RPC channel. UBSan uses |0.
#define ASAN_OFFLOAD_REPORT_OPCODE (('s' << 24) | 2)

enum {
  ASAN_ACCESS_READ = 0,
  ASAN_ACCESS_WRITE = 1,
};

enum {
  /// A bad memory access; the host names the bug from the shadow byte.
  ASAN_REPORT_ACCESS = 0,
  /// 'free' of a pointer the device allocator never handed out.
  ASAN_REPORT_INVALID_FREE = 1,
  /// 'free' of a pointer that is already free.
  ASAN_REPORT_DOUBLE_FREE = 2,
};

/// Size of the redzone the device allocator puts on either side of a block.
#define ASAN_OFFLOAD_BLOCK_REDZONE 32

/// Header the device allocator keeps at the start of the left redzone. The host
/// reads it to describe the region an offending address falls in.
struct __asan_offload_block {
  uint64_t magic;
  uint64_t size;
};

#define ASAN_OFFLOAD_BLOCK_MAGIC 0x8badf00da5a50fdbULL

struct __asan_offload_report {
  uint64_t pc;
  uint64_t addr;
  uint32_t size;
  uint32_t access_type;
  uint32_t block[3];
  uint16_t thread[3];
  uint8_t lane;
  uint8_t fatal;
  uint8_t kind;
  uint8_t reserved[19];
};

static_assert(sizeof(__asan_offload_report) == 64,
              "Offload ASan report must fit one RPC packet");

#endif // ASAN_OFFLOAD_PACKET_H
