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

#include "dasan_mapping.h"
#include "dasan_report.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client Client asm("__llvm_rpc_client");

// Shallow deduplication table to avoid flooding the RPC channel on recover.
static bool ShouldReport(uintptr_t Pc) {
  static uint64_t Seen[64];
  const uint64_t Token = ((Pc >> 2)) * 0x9E3779B97F4A7C15ULL;
  const uint64_t Idx = Token >> 58;
  uint64_t Last = __scoped_atomic_exchange_n(
      &Seen[Idx], Token, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return Last != Token;
}

namespace {

struct Failure {
  uint64_t Entry;
  uint64_t ChunkOffset;
};
}  // namespace

static Failure Recover(uintptr_t Addr) {
  using namespace __dasan;
  const uint64_t Class = GetSizeClass(Addr);
  const uint64_t Idx = GetChunkIdx(Addr, Class);
  return {*reinterpret_cast<uint64_t*>(GetMetadata(Addr, Idx)),
          (Addr & kRegionMask) & (ClassIdToSize(Class) - 1)};
}

template <bool Abort>
[[clang::always_inline]] static void Send(uintptr_t Addr, uint64_t AccessSize,
                                          bool IsWrite, uintptr_t Pc,
                                          uint8_t Kind, uintptr_t Base,
                                          uint64_t Size, bool Freed) {
  if constexpr (!Abort) {
    if (!ShouldReport(Pc))
      return;
  }

  dasan_report_t R{};
  R.Addr = Addr;
  R.Base = Base;
  R.AllocSize = Size;
  R.Pc = Pc;
  R.Recover = !Abort;
  R.Kind = Kind;
  R.Freed = Freed;
  R.AccessSize = AccessSize;
  R.Block[0] = __gpu_block_id(__GPU_X_DIM);
  R.Block[1] = __gpu_block_id(__GPU_Y_DIM);
  R.Block[2] = __gpu_block_id(__GPU_Z_DIM);
  R.Thread[0] = __gpu_thread_id(__GPU_X_DIM);
  R.Thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  R.Thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  R.Lane = __gpu_lane_id();
  R.IsWrite = IsWrite;

  static_assert(sizeof(dasan_report_t) <= sizeof(rpc::Buffer),
                "a report must fit in a single packet");

  rpc::Client::Port Port = Client.open<DASAN_REPORT>();
  Port.send([&](rpc::Buffer* Buffer, uint32_t) {
    __builtin_memcpy(Buffer->data, &R, sizeof(R));
  });
}

template <bool Abort>
[[clang::always_inline]] static void SendSpace(uintptr_t Addr,
                                               uint64_t AccessSize,
                                               bool IsWrite, uintptr_t Pc,
                                               uint64_t FailMask) {
  if (!__gpu_is_first_in_lane(FailMask))
    return;

  const Failure F = Recover(Addr);
  Send<Abort>(Addr, AccessSize, IsWrite, Pc, DASAN_KIND_SPACE,
              Addr - (F.ChunkOffset - __dasan::EntryOffset(F.Entry)),
              __dasan::EntrySize(F.Entry), __dasan::IsPoisoned(F.Entry));
}

template <bool Abort>
[[clang::always_inline]] static void SendShared(uintptr_t Addr,
                                                uint64_t AccessSize,
                                                bool IsWrite, uintptr_t Pc,
                                                uint64_t Base, uint64_t Size,
                                                uint64_t FailMask) {
  if (!__gpu_is_first_in_lane(FailMask))
    return;

  Send<Abort>(Addr, AccessSize, IsWrite, Pc, DASAN_KIND_SHARED, Base, Size,
              /*Freed=*/false);
}

#define DASAN_HANDLER(Name, IsWrite)                                           \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE [[gnu::cold, gnu::noreturn]] void   \
  __dasan_report_##Name(void* Addr, uint64_t AccessSize, uint64_t FailMask) {  \
    SendSpace<true>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite,    \
                    reinterpret_cast<uintptr_t>(__builtin_return_address(0)),  \
                    FailMask);                                                 \
    __gpu_exit();                                                              \
  }                                                                            \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE [[gnu::cold]] void                  \
  __dasan_report_##Name##_noabort(void* Addr, uint64_t AccessSize,             \
                                  uint64_t FailMask) {                         \
    SendSpace<false>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite,   \
                     reinterpret_cast<uintptr_t>(__builtin_return_address(0)), \
                     FailMask);                                                \
  }

DASAN_HANDLER(load, false)
DASAN_HANDLER(store, true)

#define DASAN_SHARED_HANDLER(Name, IsWrite)                                    \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE [[gnu::cold, gnu::noreturn]] void   \
  __dasan_report_shared_##Name(void* Addr, uint64_t AccessSize, uint64_t Base, \
                               uint64_t Size, uint64_t FailMask) {             \
    SendShared<true>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite,   \
                     reinterpret_cast<uintptr_t>(__builtin_return_address(0)), \
                     Base, Size, FailMask);                                    \
    __gpu_exit();                                                              \
  }                                                                            \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE [[gnu::cold]] void                  \
  __dasan_report_shared_##Name##_noabort(void* Addr, uint64_t AccessSize,      \
                                         uint64_t Base, uint64_t Size,         \
                                         uint64_t FailMask) {                  \
    SendShared<false>(                                                         \
        reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite,                \
        reinterpret_cast<uintptr_t>(__builtin_return_address(0)), Base, Size,  \
        FailMask);                                                             \
  }

DASAN_SHARED_HANDLER(load, false)
DASAN_SHARED_HANDLER(store, true)
