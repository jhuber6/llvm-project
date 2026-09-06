//===-- asan_offload.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "asan_offload_packet.h"

#include <gpuintrin.h>

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"

using uptr = unsigned long;
using u8 = unsigned char;
using u64 = unsigned long long;

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client __asan_rpc_client asm("__llvm_rpc_client");

namespace __sanitizer {
[[noreturn]] void CheckFailed(const char *, int, const char *, u64, u64) {
  __builtin_verbose_trap("AddressSanitizer", "internal check failed");
}
} // namespace __sanitizer

namespace {

constexpr uptr kScale = 3;
constexpr uptr kOffset = 0x7fff8000;
constexpr uptr kGranule = 8;
constexpr uptr kRedzone = 32;
constexpr uptr kSlab = 2u << 20;
constexpr u8 kHeapRZ = 0xfa;
constexpr u8 kHeapFree = 0xfd;
constexpr u8 kUserPoison = 0xf7;
constexpr u8 kGlobalRZ = 0xf9;

[[gnu::always_inline]] constexpr uptr MemToShadow(uptr Addr) {
  return (Addr >> kScale) + kOffset;
}

void poison(uptr Addr, uptr Size, u8 Mag) {
  if (!Size)
    return;
  uptr End = Addr + Size;
  uptr Aligned = (Addr + kGranule - 1) & ~(kGranule - 1);
  if (Aligned > End)
    return;
  u8 *Beg = reinterpret_cast<u8 *>(MemToShadow(Aligned));
  u8 *Last = reinterpret_cast<u8 *>(MemToShadow(End & ~(kGranule - 1)));
  for (u8 *P = Beg; P < Last; ++P)
    *P = Mag;
  uptr Tail = End & (kGranule - 1);
  if (Tail)
    *Last = Mag;
}

bool poisoned(uptr Addr, uptr Size) {
  if (!Size)
    return false;
  uptr Last = Addr + Size - 1;
  for (uptr A = Addr & ~(kGranule - 1); A <= Last; A += kGranule) {
    u8 S = *reinterpret_cast<u8 *>(MemToShadow(A));
    if (!S)
      continue;
    if (S >= 128)
      return true;
    uptr First = A < Addr ? Addr : A;
    uptr End = A + kGranule - 1 < Last ? A + kGranule : Last + 1;
    if (First + (End - First) - A > S)
      return true;
  }
  return false;
}

bool seen(uptr Pc) {
  static constexpr u64 Bits = 6;
  static constexpr u64 Golden = 0x9E3779B97F4A7C15ull;
  static uptr Table[1u << Bits] = {};
  unsigned Index = (Pc * Golden) >> (sizeof(u64) * 8u - Bits);
  uptr *Last = &Table[Index];
  return __scoped_atomic_exchange_n(Last, Pc, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_DEVICE) == Pc;
}

void report(uptr Pc, uptr Addr, uptr Size, bool IsWrite, bool Fatal) {
  if (seen(Pc))
    return;

  rpc::Client::Port Port =
      __asan_rpc_client.open<ASAN_OFFLOAD_REPORT_OPCODE>();
  Port.send([&](rpc::Buffer *Buf, uint32_t) {
    auto &Rep = *reinterpret_cast<__asan_offload_report *>(Buf);
    Rep.pc = static_cast<uint64_t>(Pc);
    Rep.addr = static_cast<uint64_t>(Addr);
    Rep.size = static_cast<uint64_t>(Size);
    Rep.is_write = IsWrite;
    Rep.fatal = Fatal;
  });
}

void *rpc_allocate(u64 Size) {
  void *Ptr = nullptr;
  rpc::Client::Port Port = __asan_rpc_client.open<LIBC_MALLOC>();
  Port.send_and_recv(
      [=](rpc::Buffer *Buffer, uint32_t) { Buffer->data[0] = Size; },
      [&](rpc::Buffer *Buffer, uint32_t) {
        Ptr = reinterpret_cast<void *>(Buffer->data[0]);
      });
  return Ptr;
}

[[maybe_unused]] void rpc_free(void *Ptr) {
  rpc::Client::Port Port = __asan_rpc_client.open<LIBC_FREE>();
  Port.send([=](rpc::Buffer *Buffer, uint32_t) {
    Buffer->data[0] = reinterpret_cast<u64>(Ptr);
  });
}

struct Slab {
  u64 Bump;
  u64 End;
};

static Slab Cur;

void *bump_alloc(uptr Need) {
  Need = (Need + 15) & ~15u;
  u64 Off = __scoped_atomic_fetch_add(&Cur.Bump, Need, __ATOMIC_RELAXED,
                                      __MEMORY_SCOPE_DEVICE);
  if (Off && Off + Need <= Cur.End)
    return reinterpret_cast<void *>(Off);

  uptr Bytes = Need > kSlab ? ((Need + kSlab - 1) & ~(kSlab - 1)) : kSlab;
  u64 Slab = reinterpret_cast<u64>(rpc_allocate(Bytes));
  if (!Slab)
    return nullptr;
  Cur.End = Slab + Bytes;
  Cur.Bump = Slab + Need;
  return reinterpret_cast<void *>(Slab);
}

void *asan_malloc(uptr Size, uptr Pc) {
  (void)Pc;
  if (!Size)
    Size = 1;
  uptr User = (Size + kGranule - 1) & ~(kGranule - 1);
  uptr Total = kRedzone + User + kRedzone;
  u8 *Raw = static_cast<u8 *>(bump_alloc(Total));
  if (!Raw)
    return nullptr;
  *reinterpret_cast<uptr *>(Raw) = Size;
  poison(reinterpret_cast<uptr>(Raw), kRedzone, kHeapRZ);
  poison(reinterpret_cast<uptr>(Raw + kRedzone + User), kRedzone, kHeapRZ);
  poison(reinterpret_cast<uptr>(Raw + kRedzone), User, 0);
  if (User != Size)
    poison(reinterpret_cast<uptr>(Raw + kRedzone + Size), User - Size, kHeapRZ);
  return Raw + kRedzone;
}

void asan_free(uptr Addr, uptr Pc) {
  (void)Pc;
  if (!Addr)
    return;
  uptr Raw = Addr - kRedzone;
  uptr Size = *reinterpret_cast<uptr *>(Raw);
  uptr User = (Size + kGranule - 1) & ~(kGranule - 1);
  poison(Raw, kRedzone + User + kRedzone, kHeapFree);
}

} // namespace

extern "C" {

void __asan_init() {}
void __asan_version_mismatch_check_v8() {}
void __asan_handle_no_return() {}

#define ASAN_REPORT(type, is_write, size)                                      \
  [[gnu::cold, gnu::noinline]] void __asan_report_##type##size(uptr Addr) {    \
    report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, size,    \
           is_write, true);                                                    \
    __builtin_verbose_trap("AddressSanitizer", "invalid access");              \
  }                                                                            \
  [[gnu::cold, gnu::noinline]] void __asan_report_##type##size##_noabort(      \
      uptr Addr) {                                                             \
    report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, size,    \
           is_write, false);                                                   \
  }

ASAN_REPORT(load, false, 1)
ASAN_REPORT(load, false, 2)
ASAN_REPORT(load, false, 4)
ASAN_REPORT(load, false, 8)
ASAN_REPORT(load, false, 16)
ASAN_REPORT(store, true, 1)
ASAN_REPORT(store, true, 2)
ASAN_REPORT(store, true, 4)
ASAN_REPORT(store, true, 8)
ASAN_REPORT(store, true, 16)

#define ASAN_REPORT_N(type, is_write)                                          \
  [[gnu::cold, gnu::noinline]] void __asan_report_##type##_n(uptr Addr,        \
                                                             uptr Size) {      \
    report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, Size,    \
           is_write, true);                                                    \
    __builtin_verbose_trap("AddressSanitizer", "invalid access");              \
  }                                                                            \
  [[gnu::cold, gnu::noinline]] void __asan_report_##type##_n_noabort(          \
      uptr Addr, uptr Size) {                                                  \
    report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, Size,    \
           is_write, false);                                                   \
  }

ASAN_REPORT_N(load, false)
ASAN_REPORT_N(store, true)

#define ASAN_ACCESS(type, is_write, size)                                      \
  [[gnu::noinline]] void __asan_##type##size(uptr Addr) {                      \
    if (!poisoned(Addr, size))                                                 \
      return;                                                                  \
    report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, size,    \
           is_write, true);                                                    \
    __builtin_verbose_trap("AddressSanitizer", "invalid access");              \
  }                                                                            \
  [[gnu::noinline]] void __asan_##type##size##_noabort(uptr Addr) {            \
    if (poisoned(Addr, size))                                                  \
      report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, size,  \
             is_write, false);                                                 \
  }

ASAN_ACCESS(load, false, 1)
ASAN_ACCESS(load, false, 2)
ASAN_ACCESS(load, false, 4)
ASAN_ACCESS(load, false, 8)
ASAN_ACCESS(load, false, 16)
ASAN_ACCESS(store, true, 1)
ASAN_ACCESS(store, true, 2)
ASAN_ACCESS(store, true, 4)
ASAN_ACCESS(store, true, 8)
ASAN_ACCESS(store, true, 16)

#define ASAN_ACCESS_N(type, is_write)                                          \
  [[gnu::noinline]] void __asan_##type##N(uptr Addr, uptr Size) {              \
    if (!poisoned(Addr, Size))                                                 \
      return;                                                                  \
    report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, Size,    \
           is_write, true);                                                    \
    __builtin_verbose_trap("AddressSanitizer", "invalid access");              \
  }                                                                            \
  [[gnu::noinline]] void __asan_##type##N_noabort(uptr Addr, uptr Size) {      \
    if (poisoned(Addr, Size))                                                  \
      report(reinterpret_cast<uptr>(__builtin_return_address(0)), Addr, Size,  \
             is_write, false);                                                 \
  }

ASAN_ACCESS_N(load, false)
ASAN_ACCESS_N(store, true)

void __asan_poison_region(u64 Addr, u64 Size) {
  poison(static_cast<uptr>(Addr), static_cast<uptr>(Size), kHeapRZ);
}

void __asan_poison_memory_region(void const volatile *Addr, uptr Size) {
  poison(reinterpret_cast<uptr>(Addr), Size, kUserPoison);
}

void __asan_unpoison_memory_region(void const volatile *Addr, uptr Size) {
  poison(reinterpret_cast<uptr>(Addr), Size, 0);
}

int __asan_address_is_poisoned(void const volatile *Addr) {
  return poisoned(reinterpret_cast<uptr>(Addr), 1);
}

uptr __asan_region_is_poisoned(void const volatile *Addr, uptr Size) {
  uptr P = reinterpret_cast<uptr>(Addr);
  if (!poisoned(P, Size))
    return 0;
  for (uptr I = 0; I < Size; ++I)
    if (poisoned(P + I, 1))
      return P + I;
  return 0;
}

u64 __asan_malloc_impl(u64 Size, u64 Pc) {
  return reinterpret_cast<u64>(asan_malloc(static_cast<uptr>(Size),
                                           static_cast<uptr>(Pc)));
}

void __asan_free_impl(u64 Addr, u64 Pc) {
  asan_free(static_cast<uptr>(Addr), static_cast<uptr>(Pc));
}

u64 __asan_aligned_alloc_impl(u64 Align, u64 Size, u64 Pc) {
  if (!Align || (Align & (Align - 1)))
    return 0;
  uptr Need = static_cast<uptr>(Size) + static_cast<uptr>(Align);
  u8 *P = static_cast<u8 *>(asan_malloc(Need, static_cast<uptr>(Pc)));
  if (!P)
    return 0;
  return reinterpret_cast<u64>(
      __builtin_align_up(P, static_cast<uptr>(Align)));
}

struct __asan_global {
  uptr beg;
  uptr size;
  uptr size_with_redzone;
  const char *name;
  const char *module_name;
  uptr has_dynamic_init;
  void *gcc_location;
  uptr odr_indicator;
};

void __asan_register_globals(__asan_global *Globals, uptr N) {
  for (uptr I = 0; I < N; ++I) {
    uptr RZ = Globals[I].size_with_redzone - Globals[I].size;
    if (RZ)
      poison(Globals[I].beg + Globals[I].size, RZ, kGlobalRZ);
  }
}

void __asan_unregister_globals(__asan_global *Globals, uptr N) {
  for (uptr I = 0; I < N; ++I) {
    uptr RZ = Globals[I].size_with_redzone - Globals[I].size;
    if (RZ)
      poison(Globals[I].beg + Globals[I].size, RZ, 0);
  }
}

void __asan_register_elf_globals(uptr *, void *, void *) {}
void __asan_unregister_elf_globals(uptr *, void *, void *) {}
void __asan_register_image_globals(uptr *) {}
void __asan_unregister_image_globals(uptr *) {}
void __asan_before_dynamic_init(const char *) {}
void __asan_after_dynamic_init() {}

static void check_mem(uptr Pc, uptr Addr, uptr Size, bool IsWrite) {
  if (!poisoned(Addr, Size))
    return;
  report(Pc, Addr, Size, IsWrite, true);
  __builtin_verbose_trap("AddressSanitizer", "invalid access");
}

void *__asan_memcpy(void *Dst, const void *Src, uptr Size) {
  uptr Pc = reinterpret_cast<uptr>(__builtin_return_address(0));
  check_mem(Pc, reinterpret_cast<uptr>(Src), Size, false);
  check_mem(Pc, reinterpret_cast<uptr>(Dst), Size, true);
  u8 *D = static_cast<u8 *>(Dst);
  const u8 *S = static_cast<const u8 *>(Src);
  for (uptr I = 0; I < Size; ++I)
    D[I] = S[I];
  return Dst;
}

void *__asan_memmove(void *Dst, const void *Src, uptr Size) {
  uptr Pc = reinterpret_cast<uptr>(__builtin_return_address(0));
  check_mem(Pc, reinterpret_cast<uptr>(Src), Size, false);
  check_mem(Pc, reinterpret_cast<uptr>(Dst), Size, true);
  u8 *D = static_cast<u8 *>(Dst);
  const u8 *S = static_cast<const u8 *>(Src);
  if (D < S) {
    for (uptr I = 0; I < Size; ++I)
      D[I] = S[I];
  } else {
    for (uptr I = Size; I; --I)
      D[I - 1] = S[I - 1];
  }
  return Dst;
}

void *__asan_memset(void *Blk, int C, uptr Size) {
  uptr Pc = reinterpret_cast<uptr>(__builtin_return_address(0));
  check_mem(Pc, reinterpret_cast<uptr>(Blk), Size, true);
  u8 *D = static_cast<u8 *>(Blk);
  for (uptr I = 0; I < Size; ++I)
    D[I] = static_cast<u8>(C);
  return Blk;
}

int __asan_option_detect_stack_use_after_return = 0;

} // extern "C"
