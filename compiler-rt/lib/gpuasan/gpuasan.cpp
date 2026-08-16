//===-- gpuasan.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device runtime for the GPU address sanitizer. The pass emits the bounds check
// inline and calls in here only when one fails, so everything below is cold.
//
// The check itself is deliberately not here: recovering the allocation from a
// pointer is a dozen scalar instructions the optimizer can hoist and common out
// across accesses in the same kernel, which it cannot do across a call. What
// belongs on this side is what the compiler cannot express -- the transport to
// the host, deduplication, and the error budget.
//
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>
#include <stdint.h>

#include "sanitizer/gpuasan_interface.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client client asm("__llvm_rpc_client");

// Running total for this device, readable by the host after a kernel.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE uint64_t __gpuasan_error_count = 0;

// Stop after this many distinct reports so a diverged wavefront cannot spend
// the whole kernel talking to the host.
static constexpr uint64_t kMaxErrors = 64;

// Names the current dispatch. Folding it into the deduplication key keeps a bug
// that fires in every launch from being silenced by the first one, since the
// table lives in device memory and outlives any single kernel. The packet
// address is unique among the packets in flight, which is all that is needed.
static uint64_t dispatchToken() {
#if defined(__AMDGPU__)
  return (uint64_t)(uintptr_t)__builtin_amdgcn_dispatch_ptr();
#else
  return 0;
#endif
}

// Shallow deduplication keyed on the PC, so a loop that faults every iteration
// costs one report rather than one per iteration. Approximate on purpose: a
// collision drops a report, which is preferable to stalling every lane.
static bool shouldReport(uintptr_t Pc) {
  static uint64_t Seen[64] = {};
  const uint64_t Token = ((Pc >> 2) ^ dispatchToken()) * 0x9E3779B97F4A7C15ull;
  const uint64_t Idx = Token >> 58;
  uint64_t Last = __scoped_atomic_exchange_n(
      &Seen[Idx], Token, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return Last != Token;
}

// Extra bits above the address space in the handler's `flags` argument. Keep in
// sync with llvm/Transforms/Instrumentation/GPUSanitizer.h.
static constexpr uint32_t kFlagAddrSpaceMask = 0xff;
static constexpr uint32_t kFlagFreed = 1u << 8;

/// Send one report and return whether the program should stop.
///
/// The reply is what makes this a round trip rather than a store into a mailbox
/// the host reads whenever it gets around to it.  Two things need it.  Stopping
/// is the host's decision, since that is where the environment is; and a trap
/// takes the queue down with it, which can leave the host tearing down a dying
/// context with the report still sitting unread in the buffer.  Waiting for the
/// answer means the report has been printed before anything else happens.
///
/// The wait is only ever paid by an access that is already broken: the
/// deduplication and the error budget below bound how many of those there can
/// be.
[[gnu::cold]] static bool report(uintptr_t Addr, uint32_t AccessSize,
                                 uintptr_t Base, uint64_t AllocSize,
                                 uint32_t Flags, bool IsWrite, bool Recover,
                                 uintptr_t Pc) {
  if (!shouldReport(Pc))
    return false;
  if (__scoped_atomic_fetch_add(&__gpuasan_error_count, 1, __ATOMIC_RELAXED,
                                __MEMORY_SCOPE_DEVICE) >= kMaxErrors)
    return false;

  // Zeroed: the whole struct, tail padding included, is copied into the packet.
  gpuasan_report_t Rep{};
  Rep.pc = Pc;
  Rep.addr = Addr;
  Rep.base = Base;
  Rep.alloc_size = AllocSize;
  Rep.access_size = AccessSize;
  Rep.block[0] = __gpu_block_id(__GPU_X_DIM);
  Rep.block[1] = __gpu_block_id(__GPU_Y_DIM);
  Rep.block[2] = __gpu_block_id(__GPU_Z_DIM);
  Rep.thread[0] = __gpu_thread_id(__GPU_X_DIM);
  Rep.thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  Rep.thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  Rep.lane = __gpu_lane_id();
  Rep.is_write = IsWrite;
  Rep.addr_space = Flags & kFlagAddrSpaceMask;
  Rep.freed = (Flags & kFlagFreed) != 0;
  Rep.recover = Recover;

  static_assert(sizeof(gpuasan_report_t) <= sizeof(rpc::Buffer),
                "A report must fit in a single packet");

  bool Halt = false;
  rpc::Client::Port Port = client.open<GPUASAN_REPORT>();
  Port.send_and_recv(
      [&](rpc::Buffer *Buffer, uint32_t) {
        __builtin_memcpy(Buffer->data, &Rep, sizeof(Rep));
      },
      [&](rpc::Buffer *Buffer, uint32_t) { Halt = Buffer->data[0] & 1; });
  return Halt;
}

#define GPUASAN_INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE

// The pass hands over the allocation it already derived, rather than only the
// address, because the slot can be recycled before the host drains the report
// and because it keeps the geometry in one place. `flags` carries the address
// space of the object, which the host cannot recover from the address alone,
// and whether the metadata was poisoned, which is what separates a use after
// free from an overflow.
//
// Two entry points, named as the CPU sanitizers name theirs: the plain one
// stops the program when the host says to, which is what it says by default,
// and the
// `_noabort` one is what `-fsanitize-recover=gpuasan` asks for and never stops.
// Neither is `noreturn`: whether this returns is not a property of the program
// being compiled, so the access after the check stays in the code either way
// and the trap happens here.
#define GPUASAN_HANDLER(name, is_write)                                        \
  GPUASAN_INTERFACE void __gpuasan_report_##name(                              \
      void *addr, uint64_t access_size, void *base, uint64_t alloc_size,       \
      uint32_t flags) {                                                        \
    if (report(reinterpret_cast<uintptr_t>(addr), access_size,                 \
               reinterpret_cast<uintptr_t>(base), alloc_size, flags, is_write, \
               /*Recover=*/false,                                              \
               reinterpret_cast<uintptr_t>(__builtin_return_address(0))))      \
      __builtin_trap();                                                        \
  }                                                                            \
  GPUASAN_INTERFACE void __gpuasan_report_##name##_noabort(                    \
      void *addr, uint64_t access_size, void *base, uint64_t alloc_size,       \
      uint32_t flags) {                                                        \
    report(reinterpret_cast<uintptr_t>(addr), access_size,                     \
           reinterpret_cast<uintptr_t>(base), alloc_size, flags, is_write,     \
           /*Recover=*/true,                                                   \
           reinterpret_cast<uintptr_t>(__builtin_return_address(0)));          \
  }

GPUASAN_HANDLER(load, false)
GPUASAN_HANDLER(store, true)
