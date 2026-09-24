//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Device half of the offload AddressSanitizer. The shadow lives in host memory
/// and is reached through XNACK, so the encoding matches the host runtime.
/// Allocation and reporting are forwarded to the host over RPC.
///
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>

#include "asan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

using namespace __sanitizer;

#define INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client client asm("__llvm_rpc_client");

static constexpr uptr SHADOW_SCALE = 3;
static constexpr uptr SHADOW_OFFSET = 0x7fff8000;
static constexpr uptr GRANULE = 1ull << SHADOW_SCALE;

static constexpr u8 HEAP_REDZONE_MAGIC = 0xfa;
static constexpr u8 USER_POISON_MAGIC = 0xf7;
static constexpr u8 GLOBAL_REDZONE_MAGIC = 0xf9;
static constexpr u8 ARRAY_COOKIE_MAGIC = 0xac;
static constexpr u8 HEAP_FREE_MAGIC = 0xfd;

static u8* shadow(uptr addr) {
  return reinterpret_cast<u8*>((addr >> SHADOW_SCALE) + SHADOW_OFFSET);
}

static constexpr uptr round_up(uptr v, uptr align) {
  return (v + align - 1) & ~(align - 1);
}

static constexpr uptr round_down(uptr v, uptr align) {
  return v & ~(align - 1);
}

static constexpr s8 min(s8 a, s8 b) { return a < b ? a : b; }
static constexpr s8 max(s8 a, s8 b) { return a > b ? a : b; }

// Mirror of the host's ShadowSegmentEndpoint.
struct Endpoint {
  u8* chunk;
  s8 offset;
  s8 value;

  Endpoint(uptr addr)
      : chunk(shadow(addr)),
        offset(addr & (GRANULE - 1)),
        value(static_cast<s8>(*chunk)) {}
};

// Mirror of the host's __asan_poison_memory_region.
static void poison_region(uptr addr, uptr size, u8 magic) {
  if (!size)
    return;
  Endpoint beg(addr);
  Endpoint end(addr + size);
  if (beg.chunk == end.chunk) {
    if (beg.value > 0 && beg.value <= end.offset)
      *beg.chunk = beg.offset ? min(beg.value, beg.offset) : magic;
    return;
  }
  if (beg.offset) {
    *beg.chunk = beg.value == 0 ? beg.offset : min(beg.value, beg.offset);
    ++beg.chunk;
  }
  for (u8* p = beg.chunk; p < end.chunk; ++p) *p = magic;
  if (end.value > 0 && end.value <= end.offset)
    *end.chunk = magic;
}

// Mirror of the host's __asan_unpoison_memory_region.
static void unpoison_region(uptr addr, uptr size) {
  if (!size)
    return;
  Endpoint beg(addr);
  Endpoint end(addr + size);
  if (beg.chunk == end.chunk) {
    if (beg.value != 0)
      *beg.chunk = max(beg.value, end.offset);
    return;
  }
  for (u8* p = beg.chunk; p < end.chunk; ++p) *p = 0;
  if (end.offset && end.value != 0)
    *end.chunk = max(end.value, end.offset);
}

static bool is_poisoned(uptr addr, uptr size) {
  if (!size)
    return false;
  uptr last = addr + size - 1;
  for (uptr a = round_down(addr, GRANULE); a <= last; a += GRANULE) {
    s8 s = static_cast<s8>(*shadow(a));
    if (!s)
      continue;
    if (s < 0)
      return true;
    uptr end = a + GRANULE - 1 < last ? a + GRANULE : last + 1;
    if (end - a > static_cast<uptr>(s))
      return true;
  }
  return false;
}

// Shallow deduplication check to save the host thread work.
static bool should_report(uptr pc) {
  static u64 seen[64] = {};
  u64 idx = (pc * 0x9E3779B97F4A7C15ull) >> 58;
  return __scoped_atomic_exchange_n(&seen[idx], pc, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_DEVICE) != pc;
}

static __asan_offload_packet make_packet(u8 op, uptr pc, uptr addr, uptr size) {
  __asan_offload_packet pkt = {};
  pkt.op = op;
  pkt.pc = pc;
  pkt.addr = addr;
  pkt.size = size;
  pkt.block[0] = __gpu_block_id(__GPU_X_DIM);
  pkt.block[1] = __gpu_block_id(__GPU_Y_DIM);
  pkt.block[2] = __gpu_block_id(__GPU_Z_DIM);
  pkt.thread[0] = __gpu_thread_id(__GPU_X_DIM);
  pkt.thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  pkt.thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  pkt.lane = __gpu_lane_id();
  return pkt;
}

static u64 send_request(const __asan_offload_packet& pkt) {
  u64 result = 0;
  rpc::Client::Port port = client.open<SANITIZER_OFFLOAD_ASAN>();
  port.send_and_recv(
      [&](rpc::Buffer* buf, u32) { __builtin_memcpy(buf->data, &pkt, 64); },
      [&](rpc::Buffer* buf, u32) { result = buf->data[0]; });
  return result;
}

// Waits for the host to print so the trap that follows cannot race the report.
// Fatal reports are never skipped, a trap without one may kill the process
// before the host prints the report from another wave.
[[gnu::cold, gnu::noinline]] static void report(uptr pc, uptr addr, uptr size,
                                                bool is_write, bool fatal) {
  if (!fatal && !should_report(pc))
    return;
  __asan_offload_packet pkt = make_packet(ASAN_OFFLOAD_REPORT, pc, addr, size);
  pkt.is_write = is_write;
  pkt.fatal = fatal;
  send_request(pkt);
}

[[noreturn]] static void trap() {
  __builtin_verbose_trap("AddressSanitizer", "invalid access");
}

static void check_range(uptr pc, uptr addr, uptr size, bool is_write) {
  if (!is_poisoned(addr, size))
    return;
  report(pc, addr, size, is_write, /*fatal=*/true);
  trap();
}

// The AMDGPU backend inserts calls to the reports and to the LDS helpers below
// after LTO has dropped unused symbols.
#define ASAN_REPORT(type, is_write, size)                                   \
  INTERFACE __attribute__((cold, noinline, used)) void                      \
  __asan_report_##type##size(uptr addr) {                                   \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);          \
    trap();                                                                 \
  }                                                                         \
  INTERFACE __attribute__((cold, noinline, used)) void                      \
  __asan_report_##type##size##_noabort(uptr addr) {                         \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);         \
  }                                                                         \
  INTERFACE __attribute__((noinline)) void __asan_##type##size(uptr addr) { \
    if (is_poisoned(addr, size)) {                                          \
      report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);        \
      trap();                                                               \
    }                                                                       \
  }                                                                         \
  INTERFACE __attribute__((noinline)) void __asan_##type##size##_noabort(   \
      uptr addr) {                                                          \
    if (is_poisoned(addr, size))                                            \
      report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);       \
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

#define ASAN_REPORT_N(type, is_write)                                    \
  INTERFACE __attribute__((cold, noinline, used)) void                   \
  __asan_report_##type##_n(uptr addr, uptr size) {                       \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);       \
    trap();                                                              \
  }                                                                      \
  INTERFACE __attribute__((cold, noinline, used)) void                   \
  __asan_report_##type##_n_noabort(uptr addr, uptr size) {               \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);      \
  }                                                                      \
  INTERFACE __attribute__((noinline)) void __asan_##type##N(uptr addr,   \
                                                            uptr size) { \
    if (is_poisoned(addr, size)) {                                       \
      report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);     \
      trap();                                                            \
    }                                                                    \
  }                                                                      \
  INTERFACE __attribute__((noinline)) void __asan_##type##N_noabort(     \
      uptr addr, uptr size) {                                            \
    if (is_poisoned(addr, size))                                         \
      report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);    \
  }

ASAN_REPORT_N(load, false)
ASAN_REPORT_N(store, true)

INTERFACE __attribute__((used)) u64 __asan_malloc_impl(u64 size, u64 pc) {
  return send_request(make_packet(ASAN_OFFLOAD_MALLOC, pc, 0, size));
}

INTERFACE __attribute__((used)) void __asan_free_impl(u64 addr, u64 pc) {
  if (!addr)
    return;
  if (send_request(make_packet(ASAN_OFFLOAD_FREE, pc, addr, 0)))
    __builtin_verbose_trap("AddressSanitizer", "invalid free");
}

INTERFACE __attribute__((used)) void __asan_poison_region(u64 addr, u64 size) {
  poison_region(addr, size, HEAP_REDZONE_MAGIC);
}

INTERFACE void __asan_poison_memory_region(void const volatile* addr,
                                           uptr size) {
  poison_region(reinterpret_cast<uptr>(addr), size, USER_POISON_MAGIC);
}

INTERFACE void __asan_unpoison_memory_region(void const volatile* addr,
                                             uptr size) {
  unpoison_region(reinterpret_cast<uptr>(addr), size);
}

INTERFACE int __asan_address_is_poisoned(void const volatile* addr) {
  return is_poisoned(reinterpret_cast<uptr>(addr), 1);
}

INTERFACE void* __asan_region_is_poisoned(void* beg, uptr size) {
  uptr addr = reinterpret_cast<uptr>(beg);
  if (!is_poisoned(addr, size))
    return nullptr;
  for (uptr p = addr; p < addr + size; ++p)
    if (is_poisoned(p, 1))
      return reinterpret_cast<void*>(p);
  return nullptr;
}

INTERFACE void* __asan_memcpy(void* dst, const void* src, uptr size) {
  check_range(GET_CALLER_PC(), reinterpret_cast<uptr>(src), size, false);
  check_range(GET_CALLER_PC(), reinterpret_cast<uptr>(dst), size, true);
  return __builtin_memcpy(dst, src, size);
}

INTERFACE void* __asan_memmove(void* dst, const void* src, uptr size) {
  check_range(GET_CALLER_PC(), reinterpret_cast<uptr>(src), size, false);
  check_range(GET_CALLER_PC(), reinterpret_cast<uptr>(dst), size, true);
  return __builtin_memmove(dst, src, size);
}

INTERFACE void* __asan_memset(void* dst, int c, uptr size) {
  check_range(GET_CALLER_PC(), reinterpret_cast<uptr>(dst), size, true);
  return __builtin_memset(dst, c, size);
}

// Must match the layout of the host's __asan_global.
struct __asan_global {
  uptr beg;
  uptr size;
  uptr size_with_redzone;
  const char* name;
  const char* module_name;
  uptr has_dynamic_init;
  void* location;
  uptr odr_indicator;
};

// Device globals are placed by the loader in memory whose shadow may carry
// stale poison, so the body is cleared as well as the redzone poisoned.
INTERFACE void __asan_register_globals(__asan_global* globals, uptr n) {
  for (uptr i = 0; i < n; ++i) {
    const __asan_global& g = globals[i];
    unpoison_region(g.beg, g.size);
    uptr aligned = round_up(g.size, GRANULE);
    if (g.size != aligned)
      *shadow(g.beg + round_down(g.size, GRANULE)) =
          static_cast<u8>(g.size & (GRANULE - 1));
    for (uptr p = g.beg + aligned; p < g.beg + g.size_with_redzone;
         p += GRANULE)
      *shadow(p) = GLOBAL_REDZONE_MAGIC;
  }
}

INTERFACE void __asan_unregister_globals(__asan_global* globals, uptr n) {
  for (uptr i = 0; i < n; ++i)
    for (uptr p = globals[i].beg;
         p < globals[i].beg + globals[i].size_with_redzone; p += GRANULE)
      *shadow(p) = 0;
}

INTERFACE void __asan_register_elf_globals(uptr* flag, void* start,
                                           void* stop) {
  if (*flag)
    return;
  __asan_global* beg = static_cast<__asan_global*>(start);
  __asan_global* end = static_cast<__asan_global*>(stop);
  __asan_register_globals(beg, end - beg);
  *flag = 1;
}

INTERFACE void __asan_unregister_elf_globals(uptr* flag, void* start,
                                             void* stop) {
  if (!*flag)
    return;
  __asan_global* beg = static_cast<__asan_global*>(start);
  __asan_global* end = static_cast<__asan_global*>(stop);
  __asan_unregister_globals(beg, end - beg);
  *flag = 0;
}

INTERFACE void __asan_before_dynamic_init(const char*) {}
INTERFACE void __asan_after_dynamic_init() {}

INTERFACE void __asan_poison_cxx_array_cookie(uptr p) {
  *shadow(p) = ARRAY_COOKIE_MAGIC;
}

INTERFACE uptr __asan_load_cxx_array_cookie(uptr* p) {
  // A freed array has no cookie left to trust. The double free is caught when
  // the storage itself is released.
  if (*shadow(reinterpret_cast<uptr>(p)) == HEAP_FREE_MAGIC)
    return 0;
  return *p;
}

INTERFACE void __asan_init() {}
INTERFACE void __asan_version_mismatch_check_v8() {}
INTERFACE void __asan_handle_no_return() {}
INTERFACE void __sanitizer_ptr_cmp(void*, void*) {}
INTERFACE void __sanitizer_ptr_sub(void*, void*) {}

INTERFACE int __asan_option_detect_stack_use_after_return = 0;
