//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal declarations for host-side AddressSanitizer offload support.
///
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_H
#define ASAN_OFFLOAD_H

#include "asan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __asan {

enum OffloadChunkKind : u8 {
  // Allocated through an HSA memory pool by the host.
  kOffloadChunkPool,
  // Allocated on behalf of device 'malloc'.
  kOffloadChunkDevice,
  // System memory reserved for HMM, e.g. managed memory.
  kOffloadChunkReserve,
};

// A block handed out through HSA. The user pointer is the block's base and the
// redzone trails the user bytes, so pointers passed back to HSA need no
// translation.
struct OffloadChunk {
  uptr Beg;
  uptr Size;
  uptr Total;
  u64 AllocPc;
  u64 FreePc;
  u32 AllocStack;
  u32 FreeStack;
  OffloadChunkKind Kind;
};

// Lays out a block of Total bytes at Beg with Size bytes addressable.
void PoisonOffloadChunk(uptr Beg, uptr Size, uptr Total);
uptr OffloadChunkTotal(uptr Size);

// Starts tracking a block. Returns false if it has no shadow.
bool TrackOffloadChunk(const OffloadChunk& C);

enum class OffloadFreeResult { kFreed, kUntracked, kDoubleFree, kWrongKind };

// Moves a live block into quarantine. Returns the tracked record in Out.
OffloadFreeResult FreeOffloadChunk(uptr Beg, OffloadChunkKind Kind, u64 Pc,
                                   u32 Stack, OffloadChunk* Out);

// Finds the live or quarantined block that contains Addr.
bool FindOffloadChunk(uptr Addr, OffloadChunk* Out, bool* Freed);

// Returns the block size HSA knows for a live block starting at Beg.
bool OffloadChunkTotalFor(uptr Beg, uptr Size, uptr* Total);

u32 HandleOffloadRequest(void* Port, u32 Lanes);

// Reports a rejected free from the host (Stack) or the device (Pkt). Dies
// if halt_on_error is set.
void ReportOffloadFreeError(OffloadFreeResult R, uptr Addr, u32 Stack,
                            const __asan_offload_packet* Pkt);

}  // namespace __asan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __asan_offload_init();
}

#endif  // ASAN_OFFLOAD_H
