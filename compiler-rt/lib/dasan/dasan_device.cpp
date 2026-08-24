//===-- dasan_device.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device report path. Checks are emitted inline by the pass.
//
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>
#include <stdint.h>

#include "dasan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client __dasan_rpc_client asm("__llvm_rpc_client");

// Shallow deduplication table to avoid flooding the RPC channel on recover.
static bool ShouldReport(uintptr_t Pc) {
  static uint64_t Seen[64];
  const uint64_t Token = ((Pc >> 2)) * 0x9E3779B97F4A7C15ULL;
  const uint64_t Idx = Token >> 58;
  uint64_t Last = __scoped_atomic_exchange_n(
      &Seen[Idx], Token, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return Last != Token;
}

template <bool Abort>
[[clang::always_inline]] static void Send(uintptr_t Addr, uint64_t AccessSize,
                                          bool IsWrite, uintptr_t Pc,
                                          uint8_t Kind, uint64_t Base,
                                          uint64_t Size) {
  if (!__gpu_is_first_in_lane(__gpu_lane_mask()))
    return;
  if constexpr (!Abort) {
    if (!ShouldReport(Pc))
      return;
  }

  __dasan_offload_report R{};
  R.Addr = Addr;
  R.Base = Base;
  R.AllocSize = Size;
  R.Pc = Pc;
  R.Recover = !Abort;
  R.Kind = Kind;
  R.AccessSize = AccessSize;
  R.Block[0] = __gpu_block_id(__GPU_X_DIM);
  R.Block[1] = __gpu_block_id(__GPU_Y_DIM);
  R.Block[2] = __gpu_block_id(__GPU_Z_DIM);
  R.Thread[0] = __gpu_thread_id(__GPU_X_DIM);
  R.Thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  R.Thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  R.Lane = __gpu_lane_id();
  R.IsWrite = IsWrite;

  static_assert(sizeof(__dasan_offload_report) <= sizeof(rpc::Buffer),
                "a report must fit in a single packet");

  rpc::Client::Port Port = __dasan_rpc_client.open<SANITIZER_OFFLOAD_DASAN>();
  Port.send([&](rpc::Buffer* Buffer, uint32_t) {
    __builtin_memcpy(Buffer->data, &R, sizeof(R));
  });

  // Fatal reports are synchronous and wait on the host server.
  if constexpr (Abort)
    Port.recv([](rpc::Buffer*, uint32_t) {});
}

// The return address is the faulting access, so these must stay real calls.
#define DASAN_HANDLER(Name, Kind, IsWrite)                                   \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE                                   \
      [[gnu::cold, gnu::noinline, gnu::noreturn]] void                       \
      __dasan_report_##Name(void* Addr, uint64_t AccessSize, uint64_t Base,  \
                            uint64_t Size) {                                 \
    Send<true>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite,       \
               GET_CALLER_PC(), Kind, Base, Size);                           \
    __gpu_exit();                                                            \
  }                                                                          \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE [[gnu::cold, gnu::noinline]] void \
  __dasan_report_##Name##_noabort(void* Addr, uint64_t AccessSize,           \
                                  uint64_t Base, uint64_t Size) {            \
    Send<false>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite,      \
                GET_CALLER_PC(), Kind, Base, Size);                          \
  }

DASAN_HANDLER(load, DASAN_KIND_SPACE, false)
DASAN_HANDLER(store, DASAN_KIND_SPACE, true)
DASAN_HANDLER(shared_load, DASAN_KIND_SHARED, false)
DASAN_HANDLER(shared_store, DASAN_KIND_SHARED, true)
