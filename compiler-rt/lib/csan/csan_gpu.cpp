//===-- csan_gpu.cpp - GPU ConcurrencySanitizer runtime -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Watchpoint-based data race detector for GPU targets, modelled on the Linux
// kernel's KCSAN.
//
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>
#include <stdint.h>

#include "csan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client client asm("__llvm_rpc_client");

#define INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE

static uint64_t __csan_num_data_races = 0;

INTERFACE uint64_t __csan_get_num_data_races() {
  return __atomic_load_n(&__csan_num_data_races, __ATOMIC_RELAXED);
}

// Shallow deduplication check to save the host thread work. Keyed on both the
// PC and the race kind so each distinct kind of race at a PC is reported once.
static bool should_report(void *pc, unsigned kind) {
  static uint64_t seen[64] = {};
  const uint64_t token = (reinterpret_cast<uintptr_t>(pc) >> 4) ^
                         (static_cast<uint64_t>(kind) * 0x9E3779B97F4A7C15ull);
  uint64_t idx = (token * 0x9E3779B97F4A7C15ull) >> 58;
  uint64_t last = __scoped_atomic_exchange_n(
      &seen[idx], token, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return last != token;
}

[[gnu::cold, gnu::noinline]] static void
report(unsigned kind, uintptr_t addr, uint32_t size, int access_type, void *pc,
       void *peer = nullptr, int peer_access = 0, uint32_t peer_size = 0,
       uint8_t peer_lane = 0) {
  pc = pc ? pc : reinterpret_cast<void *>(GET_CALLER_PC());
  if (!should_report(pc, kind))
    return;

  __csan_gpu_race rep = {};
  rep.pc = reinterpret_cast<uintptr_t>(pc);
  rep.peer_pc = reinterpret_cast<uintptr_t>(peer);
  rep.addr = addr;
  rep.size = size;
  rep.access_type = static_cast<unsigned>(access_type);
  rep.kind = kind;
  rep.block[0] = __gpu_block_id(__GPU_X_DIM);
  rep.block[1] = __gpu_block_id(__GPU_Y_DIM);
  rep.block[2] = __gpu_block_id(__GPU_Z_DIM);
  rep.thread[0] = __gpu_thread_id(__GPU_X_DIM);
  rep.thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  rep.thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  rep.lane = __gpu_lane_id();
  rep.peer_lane = peer_lane;
  rep.peer_access_type = static_cast<uint8_t>(peer_access);
  rep.peer_size = static_cast<uint8_t>(peer_size);

  rpc::Client::Port Port = client.open<CSAN_OFFLOAD_REPORT_OPCODE>();
  Port.send([&](rpc::Buffer *buf, uint32_t) {
    __builtin_memcpy(buf->data, &rep, sizeof(rep));
  });
  static_assert(sizeof(__csan_gpu_race) <= sizeof(rpc::Buffer),
                "Report must fit in a single packet");

  __scoped_atomic_fetch_add(&__csan_num_data_races, 1, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
}

#if defined(__AMDGPU__)
// AMDGPU does not have a single set frequency. Different architectures and
// cards can have different values. A frequency of 100MHz is most common so we
// use it, if it is wrong it just means we sleep longer than expected.
static constexpr uint64_t CLOCK_FREQ_HZ = 100000000UL;
#else
static constexpr uint64_t CLOCK_FREQ_HZ = 1000000000UL;
#endif
static constexpr uint64_t TICKS_PER_SEC = 1000000000UL;

// Randomized bounds, in nanoseconds, for the watchpoint stall window.
static constexpr uint64_t SAMPLE_DELAY_MIN_NS = 1000;
static constexpr uint64_t SAMPLE_DELAY_MAX_NS = 10000;

static constexpr uint64_t WP_TABLE_SIZE = /*2 MiB=*/2 * 1024ul * 1024ul;
static constexpr uint64_t WP_TABLE_SLOTS = WP_TABLE_SIZE / sizeof(uint64_t);
static constexpr uint32_t WP_CHANCE = 8;
static constexpr uint32_t WP_MAX_SIZE = 16;

static constexpr uint32_t WP_SIZE_BITS =
    __builtin_popcountg(WP_MAX_SIZE - 1u) + 1;
static constexpr uint32_t WP_ADDR_BITS = 64 - 2 - WP_SIZE_BITS;

//===----------------------------------------------------------------------===//
// Watchpoint encoding:
//   [63]                       is_write
//   [62]                       consumed
//   [61 : WP_ADDR_BITS]        size
//   [WP_ADDR_BITS-1]           is_lds
//   [WP_ADDR_BITS-2 : 0]       address, or (block_index << 20 | lds_offset)
//
// LDS is not a global identity: every workgroup has an LDS[0]. The high bit
// keeps those keys out of the global address range, and the workgroup's
// linearized index sits above a 20-bit offset (1 MiB of LDS). Two workgroups
// then differ by far more than any access size, so the ordinary overlap test
// separates them. Grids whose linearized size overflows skip the table.
//===----------------------------------------------------------------------===//

static constexpr uint64_t WP_INVALID = 0;
static constexpr uint64_t WP_CONSUMED_MASK = 1ull << 62;
static constexpr uint64_t WP_WRITE_MASK = 1ull << 63;
static constexpr uint64_t WP_ADDR_MASK = (1ull << WP_ADDR_BITS) - 1;
static constexpr uint64_t WP_SIZE_MASK = ((1ull << WP_SIZE_BITS) - 1)
                                         << WP_ADDR_BITS;
static constexpr uint64_t WP_LDS_FLAG = 1ull << (WP_ADDR_BITS - 1);
static constexpr uint64_t WP_GLOBAL_MASK = WP_LDS_FLAG - 1;
static constexpr uint32_t LDS_OFFSET_BITS = 20;
static constexpr uint64_t LDS_OFFSET_MASK = (1ull << LDS_OFFSET_BITS) - 1;
static constexpr uint64_t LDS_MAX_BLOCKS =
    1ull << (WP_ADDR_BITS - 1 - LDS_OFFSET_BITS);
static_assert((WP_MAX_SIZE & (WP_MAX_SIZE - 1)) == 0,
              "WP_MAX_SIZE must be a power of two");
static_assert((WP_WRITE_MASK ^ WP_CONSUMED_MASK ^ WP_SIZE_MASK ^
               WP_ADDR_MASK) == ~0ull,
              "watchpoint fields must partition the 64-bit word");
static_assert((WP_LDS_FLAG | ((LDS_MAX_BLOCKS - 1) << LDS_OFFSET_BITS) |
               LDS_OFFSET_MASK) == WP_ADDR_MASK,
              "LDS key fields must fill the address bits");

static constexpr uint64_t encode_watchpoint(uint64_t addr, uint32_t size,
                                            bool is_write) {
  return (is_write ? WP_WRITE_MASK : 0) | ((uint64_t)size << WP_ADDR_BITS) |
         (addr & WP_ADDR_MASK);
}

static constexpr bool decode_watchpoint(uint64_t wp, uint64_t &addr,
                                        uint32_t &size, bool &is_write) {
  if (wp == WP_INVALID || (wp & WP_CONSUMED_MASK))
    return false;
  is_write = (wp & WP_WRITE_MASK) != 0;
  size = (wp & WP_SIZE_MASK) >> WP_ADDR_BITS;
  addr = wp & WP_ADDR_MASK;
  return true;
}

static constexpr bool ranges_overlap(uint64_t a1, uint32_t s1, uint64_t a2,
                                     uint32_t s2) {
  return a1 < a2 + s2 && a2 < a1 + s1;
}

static_assert((WP_TABLE_SLOTS & (WP_TABLE_SLOTS - 1)) == 0,
              "WP_TABLE_SLOTS must be a power of two");

// Fold the workgroup index (high) into the LDS offset so distinct groups do
// not collide in the low bits the table actually uses.
static constexpr uint32_t watchpoint_slot(uint64_t key) {
  key ^= key >> LDS_OFFSET_BITS;
  return (key >> (WP_SIZE_BITS - 1)) & (WP_TABLE_SLOTS - 1);
}

static uint64_t watchpoints[WP_TABLE_SLOTS] = {};

// FIXME: Optimize this properly in the backend for COV5+.
static uint32_t num_blocks(int dim) {
#ifdef __AMDGPU__
  return ((const uint32_t __gpu_constant *)__builtin_amdgcn_implicitarg_ptr())
      [dim];
#else
  return __gpu_num_blocks(dim);
#endif
}

static uint64_t lds_block_index() {
  return (uint64_t)__gpu_block_id(__GPU_X_DIM) +
         (uint64_t)num_blocks(__GPU_X_DIM) *
             ((uint64_t)__gpu_block_id(__GPU_Y_DIM) +
              (uint64_t)num_blocks(__GPU_Y_DIM) *
                  (uint64_t)__gpu_block_id(__GPU_Z_DIM));
}

static bool lds_fits() {
  uint64_t nxy, nxyz;
  if (__builtin_mul_overflow((uint64_t)num_blocks(__GPU_X_DIM),
                             (uint64_t)num_blocks(__GPU_Y_DIM), &nxy) ||
      __builtin_mul_overflow(nxy, (uint64_t)num_blocks(__GPU_Z_DIM), &nxyz))
    return false;
  return nxyz <= LDS_MAX_BLOCKS;
}

static uint64_t rng() {
  uint64_t z = __builtin_readcyclecounter();
  z ^= (__gpu_thread_id(__GPU_X_DIM) + (uint64_t)__gpu_block_id(__GPU_X_DIM) *
                                           __gpu_num_threads(__GPU_X_DIM)) *
       0xD1B54A32D192ED03ull;
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

template <typename> struct is_ptr_local {
  static constexpr bool value = false;
};
template <typename T> struct is_ptr_local<T __gpu_local *> {
  static constexpr bool value = true;
};

template <typename PtrTy> static bool uses_table() {
  if constexpr (is_ptr_local<PtrTy>::value)
    return lds_fits();
  return true;
}

template <typename PtrTy> static uint64_t watch_key(uintptr_t addr) {
  if constexpr (is_ptr_local<PtrTy>::value)
    return WP_LDS_FLAG | (lds_block_index() << LDS_OFFSET_BITS) |
           (addr & LDS_OFFSET_MASK);
  return addr & WP_GLOBAL_MASK;
}

static bool should_watch(uint64_t lane_mask, uint32_t access_type) {
  // If every access is atomic we cannot have a race.
  if (!__gpu_ballot(lane_mask, !(access_type & CSAN_ACCESS_ATOMIC)))
    return false;

  static_assert(WP_CHANCE >= 2 && (WP_CHANCE & (WP_CHANCE - 1)) == 0,
                "WP_CHANCE must be a power of two >= 2");
  constexpr unsigned N = __builtin_ctzg(WP_CHANCE);
  return __gpu_read_first_lane_u32(lane_mask, (rng() >> (64 - N)) == 0);
}

// Probe the hashed slot for a conflicting watchpoint.
template <typename PtrTy>
static uint64_t *find_watchpoint(uintptr_t addr, uint32_t size,
                                 bool expect_write, uint64_t &encoded) {
  const uint64_t key = watch_key<PtrTy>(addr);
  uint64_t *wp = &watchpoints[watchpoint_slot(key)];
  encoded = __scoped_atomic_load_n(wp, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);

  uint64_t wp_addr;
  uint32_t wp_size;
  bool is_write;
  if (!decode_watchpoint(encoded, wp_addr, wp_size, is_write))
    return nullptr;
  if (expect_write && !is_write)
    return nullptr;
  if (ranges_overlap(wp_addr, wp_size, key, size))
    return wp;
  return nullptr;
}

static uint64_t *insert_watchpoint(uint64_t key, uint32_t size, bool is_write) {
  uint64_t *wp = &watchpoints[watchpoint_slot(key)];
  const uint64_t encoded = encode_watchpoint(key, size, is_write);
  uint64_t expected = WP_INVALID;
  if (__scoped_atomic_compare_exchange_n(wp, &expected, encoded, false,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_DEVICE))
    return wp;
  return nullptr;
}

static uint32_t encode_size(uint32_t size) {
  return size < WP_MAX_SIZE ? size : WP_MAX_SIZE;
}

// Replace the live watchpoint with the finder's PC so the setter can name it.
static bool try_consume_watchpoint(uint64_t *wp, uint64_t encoded, void *pc,
                                   bool is_write, uint32_t size) {
  uint64_t consumed = WP_CONSUMED_MASK | (is_write ? WP_WRITE_MASK : 0) |
                      ((uint64_t)encode_size(size) << WP_ADDR_BITS) |
                      (reinterpret_cast<uintptr_t>(pc) & WP_ADDR_MASK);
  return __scoped_atomic_compare_exchange_n(wp, &encoded, consumed, false,
                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                            __MEMORY_SCOPE_DEVICE);
}

static bool consume_watchpoint(uint64_t *wp, void *&peer, int &peer_access,
                               uint32_t &peer_size) {
  uint64_t old = __scoped_atomic_exchange_n(
      wp, WP_CONSUMED_MASK, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  peer = reinterpret_cast<void *>(old & WP_ADDR_MASK);
  peer_access = (old & WP_WRITE_MASK) ? CSAN_ACCESS_WRITE : 0;
  peer_size = (old & WP_SIZE_MASK) >> WP_ADDR_BITS;
  return !(old & WP_CONSUMED_MASK);
}

static void remove_watchpoint(uint64_t *wp) {
  __scoped_atomic_store_n(wp, WP_INVALID, __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_DEVICE);
}

// FNV-1a digest of a byte range so wide accesses can reuse the 64-bit value
// comparison semantics.
template <typename BytePtr, typename WordPtr>
static uint64_t read_range(BytePtr bytes, WordPtr, uint32_t size, int scope) {
  uint64_t sum = 0xcbf29ce484222325ull;
  uint32_t i = 0;

  for (; i < size && ((reinterpret_cast<uintptr_t>(bytes) + i) & 7u); ++i)
    sum = (sum ^ __scoped_atomic_load_n(bytes + i, __ATOMIC_RELAXED, scope)) *
          0x100000001b3ull;
  for (; i + 8 <= size; i += 8)
    sum = (sum ^ __scoped_atomic_load_n(reinterpret_cast<WordPtr>(bytes + i),
                                        __ATOMIC_RELAXED, scope)) *
          0x100000001b3ull;
  for (; i < size; ++i)
    sum = (sum ^ __scoped_atomic_load_n(bytes + i, __ATOMIC_RELAXED, scope)) *
          0x100000001b3ull;
  return sum;
}

// Snapshot the watched location for value-change detection. Larger sizes get
// converted into a single checksum.
static uint64_t read_instrumented_memory(const volatile __gpu_global void *ptr,
                                         uint32_t size) {
  const uintptr_t addr =
      reinterpret_cast<uintptr_t>(const_cast<const __gpu_global void *>(ptr));
  if ((addr & (size - 1)) == 0) {
    switch (size) {
    case 1:
      return __scoped_atomic_load_n((const volatile __gpu_global uint8_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    case 2:
      return __scoped_atomic_load_n((const volatile __gpu_global uint16_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    case 4:
      return __scoped_atomic_load_n((const volatile __gpu_global uint32_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    case 8:
      return __scoped_atomic_load_n((const volatile __gpu_global uint64_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
  }
  return read_range((const volatile __gpu_global uint8_t *)ptr,
                    (const volatile __gpu_global uint64_t *)ptr, size,
                    __MEMORY_SCOPE_DEVICE);
}

static uint64_t read_instrumented_memory(const volatile __gpu_local void *ptr,
                                         uint32_t size) {
  const uintptr_t addr =
      reinterpret_cast<uintptr_t>(const_cast<const __gpu_local void *>(ptr));
  if ((addr & (size - 1)) == 0) {
    switch (size) {
    case 1:
      return __scoped_atomic_load_n((const volatile __gpu_local uint8_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    case 2:
      return __scoped_atomic_load_n((const volatile __gpu_local uint16_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    case 4:
      return __scoped_atomic_load_n((const volatile __gpu_local uint32_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    case 8:
      return __scoped_atomic_load_n((const volatile __gpu_local uint64_t *)ptr,
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    }
  }
  return read_range((const volatile __gpu_local uint8_t *)ptr,
                    (const volatile __gpu_local uint64_t *)ptr, size,
                    __MEMORY_SCOPE_WRKGRP);
}

static bool intra_wave_race(uint64_t lane_mask, uintptr_t addr, int access_type,
                            uint8_t &peer_lane) {
  const bool is_write = (access_type & CSAN_ACCESS_WRITE) != 0;
  const bool is_atomic = (access_type & CSAN_ACCESS_ATOMIC) != 0;
  const uint64_t writers = __gpu_ballot(lane_mask, is_write);
  const uint64_t nonatomic = __gpu_ballot(lane_mask, !is_atomic);
  if (!writers || !nonatomic)
    return false;

  const uint64_t same_addr = __gpu_match_any_u64(lane_mask, addr);
  const bool is_race = __builtin_popcountg(same_addr) >= 2 &&
                       (same_addr & writers) && (same_addr & nonatomic);
  if (!is_race || !__gpu_is_first_in_lane(same_addr))
    return false;
  peer_lane = static_cast<uint8_t>(63u - __builtin_clzg(same_addr));
  return true;
}

static void delay_ns(uint64_t nsecs) {
  const uint64_t tick_rate = TICKS_PER_SEC / CLOCK_FREQ_HZ;
  const uint64_t start = __builtin_readsteadycounter();
  const uint64_t end = start + (nsecs + tick_rate - 1) / tick_rate;
#if defined(__AMDGPU__)
  __builtin_amdgcn_s_sleep(2);
  while (__builtin_readsteadycounter() < end)
    __builtin_amdgcn_s_sleep(15);
#else
  while (__builtin_readsteadycounter() < end)
    __gpu_thread_suspend();
#endif
}

static void sample_delay(uint64_t lane_mask) {
  uint64_t nsecs = SAMPLE_DELAY_MIN_NS;
  if (__gpu_is_first_in_lane(lane_mask))
    nsecs += (rng() >> 32) % (SAMPLE_DELAY_MAX_NS - SAMPLE_DELAY_MIN_NS);
  delay_ns(__gpu_read_first_lane_u64(lane_mask, nsecs));
}

[[gnu::cold, gnu::noinline]] static void
found_watchpoint(uint64_t *wp, uint64_t encoded, bool is_write, uint32_t size,
                 void *pc) {
  try_consume_watchpoint(wp, encoded,
                         pc ? pc : reinterpret_cast<void *>(GET_CALLER_PC()),
                         is_write, size);
}

// The slow path, sets a watchpoint in the table and waits to see if any other
// thread tripped it.
template <typename PtrTy>
static void watch(uint64_t lane_mask, const PtrTy addr, uint32_t size,
                  int access_type, void *pc) {
  const bool is_write = (access_type & CSAN_ACCESS_WRITE) != 0;
  const uintptr_t iaddr = reinterpret_cast<uintptr_t>(addr);

  const uint32_t wp_size = encode_size(size);
  const bool armable = uses_table<PtrTy>() &&
                       !(access_type & CSAN_ACCESS_ATOMIC) &&
                       (is_ptr_local<PtrTy>::value || iaddr != 0);
  uint64_t *wp =
      armable ? insert_watchpoint(watch_key<PtrTy>(iaddr), wp_size, is_write)
              : nullptr;

  const uint64_t old = read_instrumented_memory(addr, size);
  sample_delay(lane_mask);
  const uint64_t now = read_instrumented_memory(addr, size);

  void *peer;
  int peer_access;
  uint32_t peer_size;
  if (wp && !consume_watchpoint(wp, peer, peer_access, peer_size))
    report(CSAN_RACE_DATA, iaddr, size, access_type, pc, peer, peer_access,
           peer_size);
  else if (old != now)
    report(CSAN_RACE_UNKNOWN_ORIGIN, iaddr, size, access_type, pc);

  if (wp)
    remove_watchpoint(wp);
}

template <typename PtrTy>
static void check_access_impl(uint64_t lane_mask, const PtrTy addr,
                              uint32_t size, int access_type) {
  void *pc = reinterpret_cast<void *>(GET_CALLER_PC());
  if (uses_table<PtrTy>()) {
    const bool is_write = (access_type & CSAN_ACCESS_WRITE) != 0;
    uint64_t encoded;
    uint64_t *wp =
        find_watchpoint<PtrTy>((uint64_t)addr, size, !is_write, encoded);
    if (wp)
      found_watchpoint(wp, encoded, is_write, size, pc);
  }

  if (!should_watch(lane_mask, access_type))
    return;

  const uintptr_t iaddr = reinterpret_cast<uintptr_t>(addr);
  uint8_t peer_lane;
  if (intra_wave_race(lane_mask, iaddr, access_type, peer_lane))
    report(CSAN_RACE_INTRA_WAVE, iaddr, size, access_type, pc, nullptr, 0, 0,
           peer_lane);

  watch(lane_mask, addr, size, access_type, pc);
}

static void check_access(const volatile void *addr, uintptr_t size,
                         int access_type) {
  if (__gpu_is_ptr_private(const_cast<void *>(addr)) || !size)
    return;

  if (__gpu_is_ptr_local(const_cast<void *>(addr)))
    return check_access_impl(__gpu_lane_mask(),
                             (const volatile __gpu_local void *)addr, size,
                             access_type);
  check_access_impl(__gpu_lane_mask(), (const volatile __gpu_global void *)addr,
                    size, access_type);
}

//===----------------------------------------------------------------------===//
// Public ABI (emitted by the ConcurrencySanitizer pass)
//===----------------------------------------------------------------------===//

INTERFACE void __csan_init() {}
INTERFACE void __csan_func_entry(void *) {}
INTERFACE void __csan_func_exit() {}
INTERFACE void __csan_ignore_thread_begin() {}
INTERFACE void __csan_ignore_thread_end() {}
INTERFACE void __csan_vptr_update(void *, void *) {}
INTERFACE void __csan_vptr_read(void *) {}

static int access_flags(int flags, bool is_write) {
  return flags | (is_write ? CSAN_ACCESS_WRITE : 0);
}

#define CSAN_PROBE(name, N, is_write)                                          \
  INTERFACE void name(void *addr, int flags) {                                 \
    check_access(addr, N, access_flags(flags, is_write));                      \
  }

#define CSAN_ACCESS(N)                                                         \
  CSAN_PROBE(__csan_read##N, N, false)                                         \
  CSAN_PROBE(__csan_unaligned_read##N, N, false)                               \
  CSAN_PROBE(__csan_write##N, N, true)                                         \
  CSAN_PROBE(__csan_unaligned_write##N, N, true)

CSAN_ACCESS(1)
CSAN_ACCESS(2)
CSAN_ACCESS(4)
CSAN_ACCESS(8)
CSAN_ACCESS(16)

INTERFACE void __csan_read_range(void *addr, uintptr_t size, int flags) {
  check_access(addr, size, access_flags(flags, false));
}

INTERFACE void __csan_write_range(void *addr, uintptr_t size, int flags) {
  check_access(addr, size, access_flags(flags, true));
}

#ifdef __AMDGPU__
// FIXME: Really shouldn't need this.
extern "C" const inline uint32_t __oclc_ABI_version = 500;
[[gnu::alias("__oclc_ABI_version")]] const uint32_t __oclc_ABI_version__;
#endif
