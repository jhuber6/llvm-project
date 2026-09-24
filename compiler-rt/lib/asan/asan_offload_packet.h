//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// RPC packet shared by the device and host AddressSanitizer runtimes.
///
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_PACKET_H
#define ASAN_OFFLOAD_PACKET_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_offload_opcodes.h"

enum : __sanitizer::u8 {
  ASAN_OFFLOAD_REPORT = 0,
  ASAN_OFFLOAD_MALLOC = 1,
  ASAN_OFFLOAD_FREE = 2,
};

// Every request is answered in the first word of the lane's buffer: the block
// for a malloc, non-zero for a free the host rejected, zero otherwise.
struct __asan_offload_packet {
  __sanitizer::u64 pc;
  __sanitizer::u64 addr;
  __sanitizer::u64 size;
  __sanitizer::u32 block[3];
  __sanitizer::u16 thread[3];
  __sanitizer::u8 op;
  __sanitizer::u8 lane;
  __sanitizer::u8 is_write;
  __sanitizer::u8 fatal;
  __sanitizer::u8 reserved[18];
};

static_assert(sizeof(__asan_offload_packet) == 64,
              "Offload ASan request must fit one RPC packet");

#endif  // ASAN_OFFLOAD_PACKET_H
