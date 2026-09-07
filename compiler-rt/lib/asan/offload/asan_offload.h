//===-- asan_offload.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host offload reporting runtime for ASan.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_H
#define ASAN_OFFLOAD_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "asan_offload_packet.h"

namespace __asan {

void Initialize();
void PrintOffloadReport(const __asan_offload_report &R);

// Lock order is HsaLifecycleMutex, then RpcMutex, then AsanOffloadMutex; the
// reverse deadlocks the report thread.
extern __sanitizer::Mutex HsaLifecycleMutex;
extern __sanitizer::Mutex AsanOffloadMutex;

// Where an allocation came from, which decides how the report describes it.
enum RegionOrigin {
  // 'hipMalloc' and friends, tracked by the HSA interceptors.
  kRegionHost,
  // The device allocator, tracked when it asked the host for memory.
  kRegionDevice,
};

// A device allocation containing some address, live or recently freed.
struct OffloadRegion {
  uptr Beg;
  uptr Size;
  u32 AllocStack;
  // Zero while the region is still live.
  u32 FreeStack;
  RegionOrigin Origin;
};

// Finds the allocation an address falls in or just outside of.
bool FindOffloadRegion(uptr Addr, OffloadRegion *Out);

// Tracking for the memory the device allocator requests over RPC.
void RecordDeviceHeap(uptr Base, uptr Size);
void ForgetDeviceHeap(uptr Base);

} // namespace __asan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __asan_offload_init();
}

#endif // ASAN_OFFLOAD_H
