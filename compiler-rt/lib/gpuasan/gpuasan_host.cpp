//===-- gpuasan_host.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host half of the GPU address sanitizer.  It places every device allocation at
// the base of a power-of-two aligned slot inside a reserved virtual address
// region, so instrumented device code can recover the allocation's identity and
// base by arithmetic and its exact extent from one table load.  The geometry
// must match llvm/include/llvm/Transforms/Instrumentation/GPUSanitizer.h.
//
// The allocator is reached through HSA's tool interface.  At hsa_init ROCr
// walks every object already loaded in the process looking for one that defines
// HSA_AMD_TOOL_PRIORITY, and hands each one it finds the API tables through
// OnLoad.  Its own exported entry points are trampolines through those tables,
// so an entry replaced there is reached by every client: HIP, a profiler, and
// libomptarget, which dlopens the runtime and calls through the pointers dlsym
// returned.  That last one is why this is not ELF symbol interposition, which
// would be invisible to it -- and it is also why there is no environment
// variable and no preload.  Linking the library is the whole of the setup;
// being found is a property of defining the symbol the vendor documents.
//
// Measured on gfx1030: a VMEM handle per allocation costs ~32 us, 155x
// hsa_amd_memory_pool_allocate, so physical memory is mapped in large chunks
// and slots are carved from it.  That brings allocation to ~0.03 us.
//
//===----------------------------------------------------------------------===//

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gpuasan_hsa.h"

#include "sanitizer/gpuasan_interface.h"
#include "shared/rpc_server.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define GPUASAN_INTERFACE extern "C" __attribute__((visibility("default")))

// Defined at the bottom, next to the other exported entry points.
GPUASAN_INTERFACE unsigned gpuasanServeOpcode(void *PortPtr, unsigned);

namespace {

//===----------------------------------------------------------------------===//
// Geometry -- keep in sync with GPUSanitizer.h
//===----------------------------------------------------------------------===//

constexpr uint64_t kRegionBase = 1ULL << 45; // 32 TiB
constexpr unsigned kClassShift = 41;         // 2 TiB per class
constexpr unsigned kNumClasses = 25;         // 4 KiB .. 64 GiB
constexpr unsigned kMinSlotLog2 = 12;
constexpr unsigned kTableClassShift = 21;
constexpr uint64_t kTableBase = 3ULL << 45; // 96 TiB
constexpr unsigned kSizeBits = 37;
constexpr unsigned kColorBits = 11;
constexpr unsigned kColorScaleLog2 = 8;
// Set on a freed slot's entry, which otherwise keeps the extent of what used to
// be there. The device check only has to notice that the entry went negative,
// and the report it sends then names the allocation the pointer used to own
// rather than an empty slot.
constexpr uint64_t kPoisonBit = 1ULL << 63;

constexpr uint64_t kRegionSpan = (uint64_t)kNumClasses << kClassShift;
constexpr uint64_t kPage = 4096;
// Best amortization point measured: 32 MiB chunks bring the mapping cost to
// 0.006 us/slot.
constexpr uint64_t kChunkBytes = 32ULL << 20;
// Above this slot size a whole-slot chunk would waste too much physical
// memory, so these get an exact page-count mapping of their own.
constexpr uint64_t kBulkMaxSlot = 64ULL << 10;

// Most slots a class can hand out, which is also the per-class stride in the
// table.  The device bounds the slot index against this before indexing, so
// every address the check accepts has a table entry that exists; that is what
// keeps a wild pointer from faulting inside the check itself rather than being
// reported.  It costs capacity only in the smallest classes, where 2M live
// allocations is already past anything a GPU program does: above class 8 the
// 2 TiB subregion runs out of slots first.
constexpr uint64_t kMaxSlots = 1ULL << kTableClassShift;
// One class's worth of entries, and the granularity everything about the table
// is done at: the whole span is backed by a single shared page of zeroes at
// this granularity, and a class that starts handing out slots swaps its range
// for one it can write.
constexpr uint64_t kTableClassBytes = 8 * kMaxSlots; // 16 MiB
constexpr uint64_t kTableSpan = (uint64_t)kNumClasses * kTableClassBytes;

// The cache maintenance queue never holds more than the one packet it waits on,
// but ROCr enforces a per-agent minimum ring size (64 on gfx1030) and rejects
// anything smaller, so the agent is asked what it will accept.
constexpr uint32_t kInvQueueFallbackPackets = 64;
constexpr uint64_t kInvTimeoutNs = 1000ULL * 1000 * 1000;

constexpr uint64_t kMaxObject = 1ULL << (kMinSlotLog2 + kNumClasses - 1);

//===----------------------------------------------------------------------===//
// Device globals -- keep in sync with GPUSanitizer.h
//===----------------------------------------------------------------------===//
//
// A global keeps the address the loader gave it. The instrumented ones are
// collected into one section, each padded with a redzone and aligned so that a
// 16-byte granule belongs to exactly one object, and the image carries a table
// with one entry per granule that this side fills in at load.
//
// One symbol per module ties it together, found by iterating the frozen
// executable's symbols. The descriptors it points at are what make internal
// globals checkable at all: after linking most device globals have no symbol,
// so the ELF symbol table cannot describe them and cannot name them either.

constexpr char kGlobalsInfoPrefix[] = "__gpuasan_globals_info.";
constexpr unsigned kGlobalGranuleLog2 = 4;
constexpr uint64_t kGlobalGranule = 1ULL << kGlobalGranuleLog2;
constexpr unsigned kGlobalEndShift = 32;
constexpr uint64_t kGlobalPoisonEntry = ~0ULL;
constexpr uint64_t kMaxGlobalNameLen = 1024;

// The image's info block, all of it written by the compiler and read only here.
// The addresses in it arrive through relocations the loader has applied by the
// time we get to it, which is how the layout becomes readable. `base` is the
// section's `__start_` symbol, and is the origin both the table and the checks
// measure from.
enum {
  kGlobalsInfoBase = 0,
  kGlobalsInfoTable = 1,
  kGlobalsInfoGranules = 2,
  kGlobalsInfoNumDescs = 3,
  kGlobalsInfoDescs = 4,
  kGlobalsInfoFields = 5,
};

struct GlobalDesc {
  uint64_t addr;
  uint64_t size; // what the program declared, not the padded storage
  uint64_t name;
  uint64_t name_len;
};

// What one checked global is, for naming it in a report and for answering a
// size query about it.
struct GlobalRecord {
  uint64_t addr;
  uint64_t size;
  std::string name;
};

// A granule entry: [31:0] the owning object's offset from the section base,
// [63:32] the complement of the offset one past its end. The complement is what
// makes a zero entry permissive, so an image whose table was never written
// checks nothing instead of reporting everything.
uint64_t makeGlobalEntry(uint64_t begin, uint64_t end) {
  return (begin & UINT32_MAX) | ((~end & UINT32_MAX) << kGlobalEndShift);
}

// Enough ports that a wave rarely has to spin waiting for one, small enough
// that the buffer stays a single page-ish allocation.
constexpr uint32_t kRPCPorts = 64;

//===----------------------------------------------------------------------===//
// The real entry points
//===----------------------------------------------------------------------===//

// Everything the runtime calls, and for the entries it also replaces, the
// implementation that was in the table before ours went in.  Chaining to the
// previous entry rather than to ROCr's own function is what lets this coexist
// with a profiler that hooked the same slot.
//
// The pairs are (name in the table, slot).  Core and AMD-extension entries are
// listed separately because they live in different sub-tables.
#define GPUASAN_HSA_CORE_API(X)                                                \
  X(hsa_init, GPUASAN_CORE_INIT)                                               \
  X(hsa_shut_down, GPUASAN_CORE_SHUT_DOWN)                                     \
  X(hsa_iterate_agents, GPUASAN_CORE_ITERATE_AGENTS)                           \
  X(hsa_agent_get_info, GPUASAN_CORE_AGENT_GET_INFO)                           \
  X(hsa_memory_copy, GPUASAN_CORE_MEMORY_COPY)                                 \
  X(hsa_executable_freeze, GPUASAN_CORE_EXECUTABLE_FREEZE)                     \
  X(hsa_executable_get_symbol_by_name,                                         \
    GPUASAN_CORE_EXECUTABLE_GET_SYMBOL_BY_NAME)                                \
  X(hsa_executable_symbol_get_info, GPUASAN_CORE_EXECUTABLE_SYMBOL_GET_INFO)   \
  X(hsa_executable_iterate_agent_symbols,                                      \
    GPUASAN_CORE_EXECUTABLE_ITERATE_AGENT_SYMBOLS)                             \
  X(hsa_queue_create, GPUASAN_CORE_QUEUE_CREATE)                               \
  X(hsa_queue_destroy, GPUASAN_CORE_QUEUE_DESTROY)                             \
  X(hsa_queue_add_write_index_screlease,                                       \
    GPUASAN_CORE_QUEUE_ADD_WRITE_INDEX_SCRELEASE)                              \
  X(hsa_signal_create, GPUASAN_CORE_SIGNAL_CREATE)                             \
  X(hsa_signal_destroy, GPUASAN_CORE_SIGNAL_DESTROY)                           \
  X(hsa_signal_store_screlease, GPUASAN_CORE_SIGNAL_STORE_SCRELEASE)           \
  X(hsa_signal_wait_scacquire, GPUASAN_CORE_SIGNAL_WAIT_SCACQUIRE)

#define GPUASAN_HSA_AMD_API(X)                                                 \
  X(hsa_amd_memory_pool_allocate, GPUASAN_AMD_MEMORY_POOL_ALLOCATE)            \
  X(hsa_amd_memory_pool_free, GPUASAN_AMD_MEMORY_POOL_FREE)                    \
  X(hsa_amd_memory_async_copy, GPUASAN_AMD_MEMORY_ASYNC_COPY)                  \
  X(hsa_amd_memory_async_copy_on_engine,                                       \
    GPUASAN_AMD_MEMORY_ASYNC_COPY_ON_ENGINE)                                   \
  X(hsa_amd_memory_fill, GPUASAN_AMD_MEMORY_FILL)                              \
  X(hsa_amd_memory_pool_get_info, GPUASAN_AMD_MEMORY_POOL_GET_INFO)            \
  X(hsa_amd_agent_iterate_memory_pools,                                        \
    GPUASAN_AMD_AGENT_ITERATE_MEMORY_POOLS)                                    \
  X(hsa_amd_agents_allow_access, GPUASAN_AMD_AGENTS_ALLOW_ACCESS)              \
  X(hsa_amd_pointer_info, GPUASAN_AMD_POINTER_INFO)                            \
  X(hsa_amd_pointer_info_set_userdata, GPUASAN_AMD_POINTER_INFO_SET_USERDATA)  \
  X(hsa_amd_vmem_address_reserve_align,                                        \
    GPUASAN_AMD_VMEM_ADDRESS_RESERVE_ALIGN)                                    \
  X(hsa_amd_vmem_address_free, GPUASAN_AMD_VMEM_ADDRESS_FREE)                  \
  X(hsa_amd_vmem_handle_create, GPUASAN_AMD_VMEM_HANDLE_CREATE)                \
  X(hsa_amd_vmem_handle_release, GPUASAN_AMD_VMEM_HANDLE_RELEASE)              \
  X(hsa_amd_vmem_map, GPUASAN_AMD_VMEM_MAP)                                    \
  X(hsa_amd_vmem_unmap, GPUASAN_AMD_VMEM_UNMAP)                                \
  X(hsa_amd_vmem_set_access, GPUASAN_AMD_VMEM_SET_ACCESS)

struct RealApi {
#define X(Name, Slot) decltype(&::Name) Name = nullptr;
  GPUASAN_HSA_CORE_API(X)
  GPUASAN_HSA_AMD_API(X)
#undef X
};

// Filled by OnLoad, which runs before any client can reach a hook.
RealApi RealApiTable;
const RealApi &real() { return RealApiTable; }

int verbose() {
  static int V = [] {
    const char *E = getenv("GPUASAN_VERBOSE");
    return E ? atoi(E) : 0;
  }();
  return V;
}

// Whether a report ends the program, which is what every other address
// sanitizer does and what the device asks about on each report.  An access that
// got as far as a report has already lost the argument about which object it
// owns, and the next thing it does may be to touch a slot that was never
// backed, where the fault is a page fault with nothing left to explain it.
//
// Read once: the answer has to be the same for every report in a run, or a
// report means something different depending on when it arrived.
bool haltOnError() {
  static bool H = [] {
    const char *E = getenv("GPUASAN_HALT_ON_ERROR");
    return E ? atoi(E) != 0 : true;
  }();
  return H;
}

void logf(const char *Fmt, ...) {
  if (!verbose())
    return;
  va_list Ap;
  va_start(Ap, Fmt);
  fprintf(stderr, "[gpuasan] ");
  vfprintf(stderr, Fmt, Ap);
  fprintf(stderr, "\n");
  va_end(Ap);
}

void errf(const char *Fmt, ...) {
  va_list Ap;
  va_start(Ap, Fmt);
  fprintf(stderr, "[gpuasan] ");
  vfprintf(stderr, Fmt, Ap);
  fprintf(stderr, "\n");
  va_end(Ap);
}

//===----------------------------------------------------------------------===//
// Allocator state
//===----------------------------------------------------------------------===//

struct LargeMapping {
  hsa_amd_vmem_alloc_handle_t handle{};
  uint64_t mapped = 0; // bytes currently backed at the slot base
};

// The slots of one class backed by one memory pool.  Backing is per pool and
// not per class because the physical memory behind a slot is the memory the
// program asked for: a chunk mapped from one GPU's pool cannot satisfy a
// request naming another's without quietly moving the allocation to a peer
// device, and it cannot satisfy a fine-grained request at all, because the
// coherence the program is relying on is a property of the pool.
struct Bank {
  std::vector<uint64_t> freelist;
  std::vector<hsa_amd_vmem_alloc_handle_t> chunks;
};

struct SizeClass {
  uint64_t base = 0; // VA of this class's subregion
  // Slot zero is never handed out.  It is the left redzone of the whole class:
  // an underflow from the first allocation would otherwise land below the
  // region, fail the range compare, and go unchecked.
  uint64_t next_slot = 1;
  uint64_t mapped_slots = 0; // slots claimed by a bulk chunk, from any bank
  std::unordered_map<uint64_t, Bank> banks; // pool handle -> its slots
  std::vector<uint64_t> chunk_pool;         // bulk chunk index -> pool handle
  std::unordered_map<uint64_t, uint64_t> large_pool; // large slot -> pool
  std::unordered_map<uint64_t, LargeMapping> large;  // slot index -> mapping
};

struct LiveAlloc {
  uint64_t slot_id; // (class << 21) | index
  unsigned cls;
  uint64_t index;
  uint64_t size;
  uint64_t color;
  bool freed;
  // Which free this was, so a report can say how long ago it happened and
  // whether the slot has been handed out again since.
  uint64_t free_serial;
};

// A slot waiting out its quarantine. It is not reusable yet, and its table
// entry stays poisoned until it is -- and after, until something is allocated
// in it.
struct Quarantined {
  unsigned cls;
  uint64_t index;
  uint64_t
      bytes; // what the allocation asked for, which is what the budget counts
};

struct State {
  std::mutex mu;
  bool active = false;
  bool tried = false;

  // Our own copy of the runtime's reference count, so the last hsa_shut_down
  // can be recognised on the way in.  ROCr has already dropped its own count to
  // zero by the time it tells a tool about the shutdown, and every entry point
  // refuses to work at that point, so waiting for that notification means never
  // getting the chance to give anything back.
  unsigned hsa_refs = 0;

  // Every GPU in the system, not just the first: a placed allocation has to be
  // reachable from whichever one the program actually launches on, and the
  // table has to be readable by all of them because the check is compiled into
  // code that can run anywhere.
  std::vector<hsa_agent_t> gpus;
  hsa_agent_t gpu{}; // gpus[0], which owns the table and the report buffer
  hsa_agent_t cpu{};
  bool have_gpu = false, have_cpu = false;
  hsa_amd_memory_pool_t vram{}; // backs the table and the report buffer
  bool have_vram = false;
  std::unordered_set<uint64_t> coarse_pools;
  // Which agent a coarse pool belongs to.  A pool is only ever backed from its
  // owner, so an allocation ends up in the same physical memory it would have
  // without the tool.
  std::unordered_map<uint64_t, hsa_agent_t> pool_owner;
  hsa_amd_memory_pool_t host_pool{}; // fine-grained system memory for RPC
  bool have_host_pool = false;
  // Fine-grained pools, which back pinned host allocations.  Placing these is
  // what brings hipHostMalloc and its equivalents under the same checks as
  // device memory, and is conditional on the self-test below.
  std::unordered_set<uint64_t> fine_pools;
  bool fine_ok = false;
  bool fine_checked = false;

  void *region = nullptr;
  void *table = nullptr;

  // RPC transport. Owned here only when nothing else in the process already
  // provides one; under OpenMP libomptarget owns it and we just register a
  // handler with its server.
  void *rpc_buffer = nullptr;
  uint32_t rpc_lanes = 64;
  std::thread rpc_thread;
  bool rpc_owned = false;
  bool rpc_checked = false;

  SizeClass classes[kNumClasses];
  // The whole table span is mapped from startup, every class aliasing this one
  // allocation full of zeroes.  A class that starts handing out slots replaces
  // its own range with memory it can write; the rest keep reading zeroes, which
  // is the encoding for "no allocation" and exactly what the check wants to see
  // for a pointer that never came from here.  Reading the table can therefore
  // never fault, which is what lets the check report a wild pointer instead of
  // dying inside itself.
  hsa_amd_vmem_alloc_handle_t zero_handle{};
  bool have_zero = false;
  bool table_live[kNumClasses] = {};
  std::vector<hsa_amd_vmem_alloc_handle_t> aux_handles;

  // Every mapping made into the two reservations, so a shut-down can take them
  // all back down.  ROCr destroys its runtime at the last hsa_shut_down and
  // knows nothing about reservations a tool made, and a region-sized hole left
  // behind in the address space stops its next init from finding memory at all.
  struct Mapping {
    void *va;
    uint64_t bytes;
    hsa_amd_vmem_alloc_handle_t handle;
    bool owned; // false when the handle is an alias released elsewhere
  };
  std::vector<Mapping> maps;

  std::unordered_map<uint64_t, LiveAlloc> live;  // user pointer -> record
  std::unordered_map<uint64_t, void *> userdata; // CLR bookkeeping
  // Kept across frees so a report that drains after the free can still name
  // the allocation.  Overwritten when the slot is handed out again.
  std::unordered_map<uint64_t, LiveAlloc> by_slot;

  // Every checked global in the process, sorted by address, so a report can
  // name the variable and a size query can be answered with what the program
  // declared rather than with the padded storage.
  std::vector<GlobalRecord> globals;
  uint64_t n_globals = 0;
  uint64_t n_globals_unchecked = 0;
  // Info blocks already dealt with, keyed by their loaded address, because an
  // executable can be frozen more than once and its table is ours to fill only
  // the first time.
  std::unordered_set<uint64_t> described;

  // A freed slot is not handed back out until it has been through here, so a
  // stale pointer keeps resolving to poisoned metadata rather than to whatever
  // was allocated next.  Bounded two ways: by slots, and by the bytes those
  // slots held, because a class with only a handful of slots would otherwise be
  // driven to exhaustion by the quarantine itself.
  std::deque<Quarantined> quarantine;
  uint64_t quarantine_bytes = 0;
  size_t quarantine_slots = 4096;
  uint64_t quarantine_budget = 256ULL << 20;
  uint64_t n_retired = 0;
  uint64_t free_serial = 0;

  // Owned queue used for nothing but cache maintenance: see invalidateCaches().
  hsa_queue_t *inv_queue = nullptr;
  hsa_signal_t inv_signal{};
  bool inv_broken = false;
  uint64_t n_inv = 0;

  uint64_t n_alloc = 0, n_free = 0;
  // An allocation the tool let through unchanged is an allocation whose
  // overflows nobody will catch, so each reason is counted separately and
  // reported at exit.  Silence here would look exactly like a clean run.
  uint64_t n_pass_inactive = 0;  // never got set up
  uint64_t n_pass_fine = 0;      // fine-grained or host pool, not a device heap
  uint64_t n_pass_toobig = 0;    // larger than the largest class
  uint64_t n_pass_exhausted = 0; // class out of slots
  uint64_t n_pass_backing = 0;   // could not get physical memory
  uint64_t n_double_free = 0;
  std::atomic<uint64_t> reported{0};
  std::atomic<bool> stop{false};
};

// Deliberately leaked.  libomptarget's MemoryManagerTy frees device buffers
// from its own static destructor, which runs after ours would, so a destroyed
// map here is a use-after-free in the tool itself.
State &st() {
  static State *S = new State();
  return *S;
}

//===----------------------------------------------------------------------===//
// Geometry helpers
//===----------------------------------------------------------------------===//

// The record covering an address, or the one nearest below it, which for an
// overflow off the end of a global is the global that was overrun.  Callers
// hold the lock and decide for themselves how far past a record they will
// believe it.
const GlobalRecord *findGlobal(uint64_t addr) {
  State &S = st();
  auto it = std::upper_bound(
      S.globals.begin(), S.globals.end(), addr,
      [](uint64_t a, const GlobalRecord &r) { return a < r.addr; });
  if (it == S.globals.begin())
    return nullptr;
  return &*std::prev(it);
}

unsigned classOf(uint64_t size) {
  unsigned k = kMinSlotLog2;
  while (k < kMinSlotLog2 + kNumClasses && (1ULL << k) < size)
    ++k;
  return k - kMinSlotLog2;
}

// How much slack to insist on past the end of an allocation, on ASan's curve:
// proportional to the size so a big buffer gets a big redzone, clamped at both
// ends.  A slot is a power of two and the allocation sits at its base, so the
// slack a size already has is the rest of its own slot -- which for a power of
// two size is nothing at all.  Asking for the class that fits size + redzone is
// what turns those cases into a whole slot of guard, at the cost of doubling
// their footprint.  That is the price of catching an off-by-one on an array
// whose length happens to be a power of two, which is not a rare shape.
uint64_t redzoneFor(uint64_t size) {
  uint64_t rz = size / 4;
  if (rz < 32)
    rz = 32;
  if (rz > 2048)
    rz = 2048;
  return rz;
}

// The left redzone, expressed in the units the table entry stores it in.  It
// pushes the allocation forward inside its slot, so an underflow lands on
// padding the program did not ask for instead of on the slot base -- and since
// the check computes the offset from the base, an address below it wraps to a
// huge unsigned value and trips the same past-the-end test.  One code path
// catches both directions.
uint64_t colorFor(uint64_t size, unsigned cls) {
  uint64_t slot = 1ULL << (cls + kMinSlotLog2);
  uint64_t rz = redzoneFor(size);
  uint64_t granule = 1ULL << kColorScaleLog2;
  uint64_t left = (rz + granule - 1) & ~(granule - 1);
  uint64_t max = ((1ULL << kColorBits) - 1) << kColorScaleLog2;
  if (left > max)
    left = max;
  // Only shift by what the slot can spare while still holding the allocation.
  while (left && left + size > slot)
    left -= granule;
  return left;
}

// The class an allocation of this size goes in: the one that fits it with its
// redzone, but never past the largest class, where a caller that would only
// just have fit still gets its exact-size slot rather than being refused.
unsigned classForAlloc(uint64_t size) {
  unsigned c = classOf(size + redzoneFor(size));
  if (c >= kNumClasses)
    c = classOf(size);
  return c;
}

uint64_t slotAddr(unsigned c, uint64_t index) {
  return kRegionBase + ((uint64_t)c << kClassShift) +
         (index << (c + kMinSlotLog2));
}

uint64_t tableIndex(unsigned c, uint64_t index) {
  return ((uint64_t)c << kTableClassShift) | index;
}

uint64_t makeEntry(uint64_t size, uint64_t color_bytes) {
  uint64_t color =
      (color_bytes >> kColorScaleLog2) & ((1ULL << kColorBits) - 1);
  return (size & ((1ULL << kSizeBits) - 1)) | (color << kSizeBits);
}

// What a slot's entry becomes when the allocation in it is freed.  The extent
// stays, so the check still recovers the object's base and size and the report
// can describe what the pointer used to point at; the top bit is what makes the
// check fail.  Zeroing the entry instead would report the same access as a
// stray pointer into an empty slot, which is a strictly worse diagnosis of the
// same bug.
uint64_t makeFreedEntry(uint64_t size, uint64_t color_bytes) {
  return makeEntry(size, color_bytes) | kPoisonBit;
}

//===----------------------------------------------------------------------===//
// VMEM helpers
//===----------------------------------------------------------------------===//

// Everything in the process: every GPU because the instrumented check is
// compiled once and can run on any of them, and the host because it stores into
// table pages directly, which faults if only the GPU is granted.
std::vector<hsa_amd_memory_access_desc_t> allAgentsDesc() {
  State &S = st();
  std::vector<hsa_amd_memory_access_desc_t> d;
  d.reserve(S.gpus.size() + 1);
  for (hsa_agent_t A : S.gpus)
    d.push_back({HSA_ACCESS_PERMISSION_RW, A});
  if (S.have_cpu)
    d.push_back({HSA_ACCESS_PERMISSION_RW, S.cpu});
  return d;
}

bool grantAll(void *va, uint64_t bytes) {
  std::vector<hsa_amd_memory_access_desc_t> d = allAgentsDesc();
  return real().hsa_amd_vmem_set_access(va, bytes, d.data(), d.size()) ==
         HSA_STATUS_SUCCESS;
}

bool mapPhysical(void *va, uint64_t bytes,
                 hsa_amd_vmem_alloc_handle_t *out_handle,
                 const hsa_amd_memory_pool_t *from = nullptr) {
  State &S = st();
  hsa_amd_vmem_alloc_handle_t h{};
  // Back a placed allocation from the same pool the caller asked for, so its
  // heap properties are the ones the program expects.
  hsa_amd_memory_pool_t pool = from ? *from : S.vram;
  hsa_status_t s =
      real().hsa_amd_vmem_handle_create(pool, bytes, MEMORY_TYPE_PINNED, 0, &h);
  if (s != HSA_STATUS_SUCCESS) {
    logf("vmem_handle_create(pool 0x%" PRIx64 ", %" PRIu64 " B) failed: %d",
         pool.handle, bytes, (int)s);
    return false;
  }
  if ((s = real().hsa_amd_vmem_map(va, bytes, 0, h, 0)) != HSA_STATUS_SUCCESS) {
    logf("vmem_map(%p, %" PRIu64 " B) failed: %d", va, bytes, (int)s);
    real().hsa_amd_vmem_handle_release(h);
    return false;
  }
  std::vector<hsa_amd_memory_access_desc_t> desc = allAgentsDesc();
  if ((s = real().hsa_amd_vmem_set_access(va, bytes, desc.data(),
                                          desc.size())) != HSA_STATUS_SUCCESS) {
    logf("vmem_set_access(%p, %" PRIu64 " B, %zu agents) failed: %d", va, bytes,
         desc.size(), (int)s);
    real().hsa_amd_vmem_unmap(va, bytes);
    real().hsa_amd_vmem_handle_release(h);
    return false;
  }
  S.maps.push_back({va, bytes, h, /*owned=*/true});
  if (out_handle)
    *out_handle = h;
  return true;
}

// Map an existing handle at a second address, which is how every class's table
// starts out aliasing the one page of zeroes.
bool mapAlias(void *va, uint64_t bytes, hsa_amd_vmem_alloc_handle_t h) {
  State &S = st();
  if (real().hsa_amd_vmem_map(va, bytes, 0, h, 0) != HSA_STATUS_SUCCESS)
    return false;
  if (!grantAll(va, bytes)) {
    real().hsa_amd_vmem_unmap(va, bytes);
    return false;
  }
  S.maps.push_back({va, bytes, h, /*owned=*/false});
  return true;
}

// Drop a mapping and the record of it, leaving what remains an accurate
// description of the address space.
void unmapRecorded(void *va, uint64_t bytes) {
  State &S = st();
  real().hsa_amd_vmem_unmap(va, bytes);
  for (size_t i = S.maps.size(); i--;)
    if (S.maps[i].va == va && S.maps[i].bytes == bytes) {
      S.maps.erase(S.maps.begin() + i);
      break;
    }
}

// Give a class table memory it can be written into, replacing the shared page
// of zeroes it was reading until now.  Deliberately not fault-driven: demand
// paging in HBM is the XNACK dependency this design exists to remove.
//
// The unmap-then-map window is not a hazard.  Only a class that has never
// handed out a slot is still on the zero mapping, so nothing on the device can
// be holding a live pointer whose entry lives in the range being swapped.
bool ensureClassTable(unsigned c) {
  State &S = st();
  if (S.table_live[c])
    return true;
  void *va = (char *)S.table + (uint64_t)c * kTableClassBytes;
  if (S.have_zero)
    unmapRecorded(va, kTableClassBytes);
  hsa_amd_vmem_alloc_handle_t h{};
  if (!mapPhysical(va, kTableClassBytes, &h)) {
    errf("failed to map the table for class %u; remapping zeroes", c);
    // Put the zero page back rather than leaving a hole: a load from a hole is
    // a fault the program cannot be blamed for.
    if (S.have_zero)
      mapAlias(va, kTableClassBytes, S.zero_handle);
    return false;
  }
  memset(va, 0, kTableClassBytes);
  S.table_live[c] = true;
  S.aux_handles.push_back(h);
  logf("table for class %u is live", c);
  return true;
}

void writeEntry(unsigned c, uint64_t index, uint64_t value) {
  State &S = st();
  if (!ensureClassTable(c))
    return;
  uint64_t ti = tableIndex(c, index);
  ((volatile uint64_t *)S.table)[ti] = value;
  // The store above can be sitting in a write-combining buffer.  The cache
  // maintenance packet that follows orders the device against the table, but
  // only for writes that have actually left this core.
  __atomic_thread_fence(__ATOMIC_RELEASE);
  if (verbose() > 1)
    logf("table[%" PRIu64 "] = 0x%" PRIx64 " (readback 0x%" PRIx64 ")", ti,
         value, ((volatile uint64_t *)S.table)[ti]);
}

//===----------------------------------------------------------------------===//
// Cache maintenance
//===----------------------------------------------------------------------===//
//
// The table store above goes to device memory across the BAR, which does not
// snoop device L2, and a kernel dispatch only takes an agent-scope acquire,
// which invalidates the vector and scalar caches but not L2.  A kernel can
// therefore read an entry a previous kernel left in L2 and miss the free that
// has happened since -- the exact window the tool exists to close.
//
// The architected way to ask for that invalidation is a barrier-AND packet with
// a system-scope acquire, which is what ROCclr itself submits at
// synchronization points.  L2 is per-device rather than per-queue, so a queue
// of our own reaches the caches the user's kernels read through, and waiting
// for it means the mutation lands before we return to a caller who cannot have
// dispatched anything yet.
//
// This is what lets table loads on the device go through the ordinary caches.
// The cost is paid once per allocation and free rather than once per access.
bool ensureInvalidationQueue() {
  State &S = st();
  if (S.inv_queue)
    return true;
  if (S.inv_broken || !S.have_gpu || !real().hsa_queue_create ||
      !real().hsa_signal_create)
    return false;

  // The smallest legal ring: only ever one packet in flight, and it is waited
  // on before the next is written.
  uint32_t packets = 0;
  if (real().hsa_agent_get_info(S.gpu, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                                &packets) != HSA_STATUS_SUCCESS ||
      packets == 0)
    packets = kInvQueueFallbackPackets;

  hsa_status_t s =
      real().hsa_queue_create(S.gpu, packets, HSA_QUEUE_TYPE_SINGLE, nullptr,
                              nullptr, 0, 0, &S.inv_queue);
  if (s != HSA_STATUS_SUCCESS || !S.inv_queue) {
    errf(
        "could not create the cache maintenance queue (%u packets, status %d); "
        "device caches will not be invalidated and use-after-free may be "
        "missed",
        packets, (int)s);
    S.inv_queue = nullptr;
    S.inv_broken = true;
    return false;
  }
  if (real().hsa_signal_create(1, 0, nullptr, &S.inv_signal) !=
      HSA_STATUS_SUCCESS) {
    errf("could not create the cache maintenance signal");
    real().hsa_queue_destroy(S.inv_queue);
    S.inv_queue = nullptr;
    S.inv_broken = true;
    return false;
  }
  logf("cache maintenance queue ready (id %" PRIu64 ")", S.inv_queue->id);
  return true;
}

// Must be called with the state lock held and after every table mutation.
void invalidateCaches() {
  State &S = st();
  if (!ensureInvalidationQueue())
    return;

  // Acquire only.  The device never writes the table, so there is nothing dirty
  // to write back; asking for a system-scope release as well would add an L2
  // flush that costs roughly twice what the invalidation does.
  constexpr uint16_t kHeader =
      (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
      (1u << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  real().hsa_signal_store_screlease(S.inv_signal, 1);

  uint64_t index = real().hsa_queue_add_write_index_screlease(S.inv_queue, 1);
  auto *ring = (hsa_barrier_and_packet_t *)S.inv_queue->base_address;
  hsa_barrier_and_packet_t &pkt = ring[index % S.inv_queue->size];

  // Everything but the header first: the header is what makes the slot live, so
  // it is published last with a release store.
  pkt.reserved0 = 0;
  pkt.reserved1 = 0;
  pkt.reserved2 = 0;
  for (auto &dep : pkt.dep_signal)
    dep.handle = 0;
  pkt.completion_signal = S.inv_signal;
  __atomic_store_n(&pkt.header, kHeader, __ATOMIC_RELEASE);

  real().hsa_signal_store_screlease(S.inv_queue->doorbell_signal, index);

  // A cache operation that never lands would leave the checks reading stale
  // entries, so this waits rather than letting it drift.
  // Busy-wait: the packet is a few microseconds of work, and a blocked wait
  // pays interrupt latency that dominates it.
  if (real().hsa_signal_wait_scacquire(S.inv_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                       kInvTimeoutNs,
                                       HSA_WAIT_STATE_ACTIVE) != 0) {
    errf("cache maintenance packet did not complete after %" PRIu64
         " ns; device caches may hold stale table entries and use-after-free "
         "may be missed",
         kInvTimeoutNs);
    return;
  }
  ++S.n_inv;
}

//===----------------------------------------------------------------------===//
// Slot allocation
//===----------------------------------------------------------------------===//

// Hand a slot back to the bank that mapped it, which is the pool its backing
// came from: giving it to a caller naming another pool would quietly move the
// allocation to another device or drop the coherence the program asked for.
void returnToBank(unsigned c, uint64_t index) {
  State &S = st();
  SizeClass &SC = S.classes[c];
  uint64_t slot_bytes = 1ULL << (c + kMinSlotLog2);
  if (slot_bytes <= kBulkMaxSlot) {
    uint64_t chunk_slots = kChunkBytes / slot_bytes;
    if (chunk_slots == 0)
      chunk_slots = 1;
    uint64_t chunk = index / chunk_slots;
    if (chunk < SC.chunk_pool.size())
      SC.banks[SC.chunk_pool[chunk]].freelist.push_back(index);
    return;
  }
  auto it = SC.large_pool.find(index);
  if (it != SC.large_pool.end())
    SC.banks[it->second].freelist.push_back(index);
}

// Cut the quarantine short for one class, oldest first.
//
// A class's address space is finite and the largest ones have only a handful of
// slots, so a quarantine sized for small allocations can exhaust them on its
// own.  Yielding a slot early weakens use-after-free detection for that one
// allocation; refusing would send the next allocation to the stock allocator,
// where nothing about it is checked at all.  The first is a smaller loss.
bool retireOldest(unsigned c) {
  State &S = st();
  for (auto it = S.quarantine.begin(); it != S.quarantine.end(); ++it) {
    if (it->cls != c)
      continue;
    S.quarantine_bytes -= it->bytes;
    returnToBank(it->cls, it->index);
    S.quarantine.erase(it);
    ++S.n_retired;
    return true;
  }
  return false;
}

// Returns the slot index within the class, or UINT64_MAX.  Backing comes from
// `pool` when given, which is the pool the caller asked to allocate from and so
// the agent the memory has to be local to.
uint64_t acquireSlot(unsigned c, uint64_t size,
                     const hsa_amd_memory_pool_t *pool = nullptr) {
  State &S = st();
  SizeClass &SC = S.classes[c];
  uint64_t slot_bytes = 1ULL << (c + kMinSlotLog2);
  // The allocation starts a left redzone above the slot base, so the pages that
  // have to be backed are the ones it actually spans.
  uint64_t span = colorFor(size, c) + size;
  uint64_t need = (span + kPage - 1) & ~(kPage - 1);

  // Only ever hand out a slot the device can look up.  The check bounds the
  // index against the same limit before touching the table, so an index past it
  // would be one the check refuses to resolve -- it would read the next class's
  // entries instead.  The address space runs out first for every class above
  // the eighth; below that this caps a class at 2M live allocations.
  uint64_t max_slots = 1ULL << (kClassShift - (c + kMinSlotLog2));
  if (max_slots > kMaxSlots)
    max_slots = kMaxSlots;

  uint64_t pool_key = pool ? pool->handle : S.vram.handle;
  Bank &B = SC.banks[pool_key];

  if (slot_bytes <= kBulkMaxSlot) {
    // Bulk regime: whole chunks of many slots are mapped at once and every
    // slot in a mapped chunk is fully backed.  A chunk is claimed by the bank
    // that mapped it, so the slots inside it are only ever handed back to
    // callers naming the same pool.
    uint64_t chunk_slots = kChunkBytes / slot_bytes;
    if (chunk_slots == 0)
      chunk_slots = 1;
    while (B.freelist.empty()) {
      if (SC.mapped_slots + chunk_slots > max_slots) {
        // Out of address space in this class.  The only slots left are the ones
        // serving their quarantine, so shorten it rather than let an allocation
        // through unchecked.
        if (retireOldest(c))
          continue;
        ++S.n_pass_exhausted;
        return UINT64_MAX; // class exhausted: fall back to the stock path
      }
      void *at = (void *)(SC.base + SC.mapped_slots * slot_bytes);
      hsa_amd_vmem_alloc_handle_t h{};
      if (!mapPhysical(at, chunk_slots * slot_bytes, &h, pool)) {
        errf("chunk map failed for class %u", c);
        ++S.n_pass_backing;
        return UINT64_MAX;
      }
      B.chunks.push_back(h);
      SC.chunk_pool.push_back(pool_key);
      // Descending, so that popping the back hands out the lowest slot first
      // and a fresh class stays dense at its base.
      for (uint64_t i = chunk_slots; i-- > 0;) {
        uint64_t s = SC.mapped_slots + i;
        if (s) // slot zero is the class-wide left redzone
          B.freelist.push_back(s);
      }
      SC.mapped_slots += chunk_slots;
    }
    uint64_t index = B.freelist.back();
    B.freelist.pop_back();
    return index;
  }

  // Large regime: map only the pages the allocation needs, so the power-of-two
  // slot costs address space rather than HBM.  The mapping is kept across a
  // free so a recycled slot pays nothing, which is also why a recycled slot
  // has to come from the bank that mapped it.
  uint64_t index;
  while (true) {
    if (!B.freelist.empty()) {
      index = B.freelist.back();
      B.freelist.pop_back();
      break;
    }
    if (SC.next_slot < max_slots) {
      index = SC.next_slot++;
      break;
    }
    if (!retireOldest(c)) {
      ++S.n_pass_exhausted;
      return UINT64_MAX;
    }
  }

  LargeMapping &LM = SC.large[index];
  if (LM.mapped < need) {
    void *at = (void *)(slotAddr(c, index) + LM.mapped);
    uint64_t extra = need - LM.mapped;
    hsa_amd_vmem_alloc_handle_t h{};
    if (!mapPhysical(at, extra, &h, pool)) {
      errf("large map failed for class %u index %" PRIu64, c, index);
      ++S.n_pass_backing;
      B.freelist.push_back(index);
      return UINT64_MAX;
    }
    S.aux_handles.push_back(h);
    LM.mapped = need;
  }
  SC.large_pool[index] = pool_key;
  return index;
}

// The VA range of the mapping a slot lives in.  set_access is keyed on the
// address a mapping starts at, and in the bulk regime a slot is one of many
// inside a chunk, so the slot's own base is not a key ROCr knows.  Returns
// false when the slot has no backing yet, which for a live allocation cannot
// happen.
bool mappingFor(unsigned c, uint64_t index, uint64_t &base, uint64_t &bytes) {
  State &S = st();
  SizeClass &SC = S.classes[c];
  uint64_t slot_bytes = 1ULL << (c + kMinSlotLog2);
  if (slot_bytes > kBulkMaxSlot) {
    auto it = SC.large.find(index);
    if (it == SC.large.end() || !it->second.mapped)
      return false;
    base = slotAddr(c, index);
    bytes = it->second.mapped;
    return true;
  }
  uint64_t chunk_slots = kChunkBytes / slot_bytes;
  if (chunk_slots == 0)
    chunk_slots = 1;
  if (index >= SC.mapped_slots)
    return false;
  bytes = chunk_slots * slot_bytes;
  base = SC.base + (index / chunk_slots) * bytes;
  return true;
}

// Put a freed slot into quarantine rather than back on a free list.
//
// Detection of a use after free lasts exactly as long as the pointer still
// resolves to poisoned metadata, and that ends the moment something else is
// allocated in the slot: the new allocation's entry makes the stale pointer
// look valid again, and reports on it stop.  Delaying reuse is what turns that
// from a property of allocation order into a budget.  Nothing here touches the
// table, so a slot that has left quarantine but not been reallocated stays
// poisoned and still reports.
void releaseSlot(unsigned c, uint64_t index, uint64_t bytes) {
  State &S = st();
  S.quarantine.push_back({c, index, bytes});
  S.quarantine_bytes += bytes;
  while (!S.quarantine.empty() && (S.quarantine.size() > S.quarantine_slots ||
                                   S.quarantine_bytes > S.quarantine_budget)) {
    const Quarantined &Q = S.quarantine.front();
    S.quarantine_bytes -= Q.bytes;
    returnToBank(Q.cls, Q.index);
    S.quarantine.pop_front();
    ++S.n_retired;
  }
}

// CLR asks about interior pointers, not just bases, so resolve any address in
// the region back to its allocation through the same arithmetic the device
// uses.
const LiveAlloc *findByAddr(uint64_t addr) {
  State &S = st();
  if (addr < kRegionBase || addr >= kRegionBase + kRegionSpan)
    return nullptr;
  uint64_t d = addr - kRegionBase;
  unsigned c = (unsigned)(d >> kClassShift);
  if (c >= kNumClasses)
    return nullptr;
  uint64_t index = (d & ((1ULL << kClassShift) - 1)) >> (c + kMinSlotLog2);
  auto it = S.by_slot.find(tableIndex(c, index));
  if (it == S.by_slot.end() || it->second.freed)
    return nullptr;
  return &it->second;
}

//===----------------------------------------------------------------------===//
// Report handling
//===----------------------------------------------------------------------===//

void printReport(const gpuasan_report_t &r) {
  State &S = st();

  // An address outside the slot region never came from the placing allocator,
  // so there is no slot, class or allocation history to name. The pass
  // resolved the object itself and sent its exact base, extent and address
  // space.
  if (r.addr - kRegionBase >= kRegionSpan) {
    const char *kind = "unknown";
    const char *what = "object";
    uint64_t base = r.base, size = r.alloc_size;
    std::string name;
    if (r.addr_space == 3) {
      kind = "shared";
      what = "LDS object";
    } else if (r.addr_space == 1) {
      kind = "global";
      what = "global";
      // The granule the device read describes the object that owns it, and for
      // a redzone it describes nothing at all.  Either way this side has the
      // descriptors and can name the variable and give its real extent, so the
      // device's arithmetic is only a fallback.
      std::lock_guard<std::mutex> g(S.mu);
      if (const GlobalRecord *rec = findGlobal(r.addr)) {
        base = rec->addr;
        size = rec->size;
        name = rec->name;
      }
    }

    // Unlike the heap path below, which has to hunt for the neighbouring slot,
    // the object here is known exactly, so the sign of the offset settles it.
    int64_t off = (int64_t)(r.addr - base);
    fprintf(stderr, "\n=============================================="
                    "===================\n");
    fprintf(stderr,
            "ERROR: GPUAddressSanitizer: %s-buffer-%s on address 0x%" PRIx64
            "\n",
            kind, off < 0 ? "underflow" : "overflow", r.addr);
    if (off < 0)
      fprintf(
          stderr,
          "%s of size %u at %" PRId64 " bytes before the start of a %" PRIu64
          "-byte %s at 0x%" PRIx64 "\n",
          r.is_write ? "WRITE" : "READ", r.access_size, -off, size, what, base);
    else
      fprintf(stderr,
              "%s of size %u at offset %" PRId64 " of a %" PRIu64
              "-byte %s at 0x%" PRIx64 "\n",
              r.is_write ? "WRITE" : "READ", r.access_size, off, size, what,
              base);
    if (!name.empty())
      fprintf(stderr, "  of global variable '%s'\n", name.c_str());
    fprintf(stderr,
            "  in block (%u,%u,%u) thread (%u,%u,%u) lane %u, pc 0x%" PRIx64
            "\n",
            r.block[0], r.block[1], r.block[2], r.thread[0], r.thread[1],
            r.thread[2], r.lane, r.pc);
    fprintf(stderr, "=============================================="
                    "===================\n");
    S.reported.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  uint64_t d = r.addr - kRegionBase;
  unsigned c = (unsigned)(d >> kClassShift);
  uint64_t slot_bytes = 1ULL << (c + kMinSlotLog2);
  uint64_t index = (d & ((1ULL << kClassShift) - 1)) >> (c + kMinSlotLog2);
  uint64_t off = r.addr - r.base;

  // An allocation sits a redzone above its slot base, so a small underflow
  // stays inside its own slot and is caught against its own entry: the base the
  // device sent is the right one, and the comparison settles the direction
  // outright.
  bool below = r.addr < r.base;

  // A larger underflow clears the redzone and lands in the slot below, where it
  // is caught against whatever that slot holds -- so the arithmetic names the
  // wrong allocation.  If the neighbour above is live and the access sits near
  // its base, report it as that allocation's underflow instead.  Not for a use
  // after free, which the metadata states outright and which needs no guessing.
  const LiveAlloc *neighbour = nullptr;
  uint64_t next_base = (r.addr & ~(slot_bytes - 1)) + slot_bytes;
  {
    std::lock_guard<std::mutex> g(S.mu);
    auto nit = S.by_slot.find(tableIndex(c, index + 1));
    if (!below && !r.freed && nit != S.by_slot.end() && !nit->second.freed &&
        off >= r.alloc_size && next_base - r.addr <= slot_bytes / 4)
      neighbour = &nit->second;

    // The poison bit came back with the extent the allocation had, so the
    // report describes the object the pointer used to own rather than an empty
    // slot. Whether the slot has since been handed out again is this side's to
    // say.
    auto sit = S.by_slot.find(tableIndex(c, index));
    const LiveAlloc *owner = sit == S.by_slot.end() ? nullptr : &sit->second;
    const char *what = "allocation";
    const char *kind = r.freed                ? "use-after-free"
                       : (below || neighbour) ? "heap-buffer-underflow"
                                              : "heap-buffer-overflow";

    fprintf(stderr, "\n=============================================="
                    "===================\n");
    fprintf(stderr, "ERROR: GPUAddressSanitizer: %s on address 0x%" PRIx64 "\n",
            kind, r.addr);
    if (r.freed)
      fprintf(stderr,
              "%s of size %u at offset %" PRIu64 " of a %" PRIu64
              "-byte region [0x%" PRIx64 ", 0x%" PRIx64 ") that was freed\n",
              r.is_write ? "WRITE" : "READ", r.access_size, off, r.alloc_size,
              r.base, r.base + r.alloc_size);
    else if (below)
      fprintf(stderr,
              "%s of size %u at %" PRIu64
              " bytes before the start of a %" PRIu64 "-byte %s\n",
              r.is_write ? "WRITE" : "READ", r.access_size, r.base - r.addr,
              r.alloc_size, what);
    else if (neighbour)
      fprintf(stderr,
              "%s of size %u at %" PRIu64
              " bytes before the start of a %" PRIu64 "-byte %s\n",
              r.is_write ? "WRITE" : "READ", r.access_size, next_base - r.addr,
              neighbour->size, what);
    else
      fprintf(stderr,
              "%s of size %u at offset %" PRIu64 " of a %" PRIu64 "-byte %s\n",
              r.is_write ? "WRITE" : "READ", r.access_size, off, r.alloc_size,
              what);

    fprintf(stderr,
            "  %s base 0x%" PRIx64 ", class %u (slot %" PRIu64
            " B), slot index %" PRIu64 "\n",
            what, neighbour ? next_base : r.base, c, slot_bytes,
            neighbour ? index + 1 : index);
    fprintf(stderr,
            "  in block (%u,%u,%u) thread (%u,%u,%u) lane %u, pc 0x%" PRIx64
            "\n",
            r.block[0], r.block[1], r.block[2], r.thread[0], r.thread[1],
            r.thread[2], r.lane, r.pc);
    // How much of the quarantine was left when the access happened is what says
    // whether the report is exact or merely the last thing this slot held.
    if (r.freed) {
      if (owner && owner->freed)
        fprintf(stderr,
                "  freed by the host at free #%" PRIu64 ", %" PRIu64
                " frees ago; the slot is still in quarantine\n",
                owner->free_serial, S.free_serial - owner->free_serial);
      else
        fprintf(stderr, "  freed by the host, and the slot has since been "
                        "handed out again: the extent above is the dead "
                        "allocation's, not the live one's\n");
    }
    // Only ambiguous when the slot below holds something the access could
    // instead have run off the end of.
    if (neighbour && r.alloc_size)
      fprintf(stderr,
              "  NOTE: could also be a %" PRIu64
              "-byte overflow of the %" PRIu64 "-byte allocation at 0x%" PRIx64
              "; the two are indistinguishable without a colour\n",
              off - r.alloc_size, r.alloc_size, r.base);
    fprintf(stderr, "=============================================="
                    "===================\n");
  }
  S.reported.fetch_add(1, std::memory_order_relaxed);
}

// A transfer naming a checked global.  Its extent is known
// exactly, which makes a hipMemcpyToSymbol that writes past the variable as
// reportable as a kernel doing it.  Returns whether the address was one of
// ours, so that an ordinary host pointer is left alone rather than measured
// against whichever global happens to sit below it.
bool checkGlobalTransfer(uint64_t addr, uint64_t bytes, bool is_write,
                         const char *api) {
  State &S = st();
  uint64_t base = 0, size = 0, storage = 0;
  std::string name;
  {
    std::lock_guard<std::mutex> g(S.mu);
    const GlobalRecord *rec = findGlobal(addr);
    if (!rec)
      return false;
    base = rec->addr;
    size = rec->size;
    name = rec->name;
    // The padded storage, which is the most of the address space the variable
    // can account for.  Past it the address belongs to whatever came next.
    storage =
        ((size + kGlobalGranule - 1) & ~(kGlobalGranule - 1)) + kGlobalGranule;
  }
  if (addr - base >= storage)
    return false;
  if (addr + bytes <= base + size) {
    logf("%s: %s %" PRIu64 " B at 0x%" PRIx64 " in bounds of global %s", api,
         is_write ? "write" : "read", bytes, addr, name.c_str());
    return true;
  }

  fprintf(stderr, "\n=============================================="
                  "===================\n");
  fprintf(stderr,
          "ERROR: GPUAddressSanitizer: global-buffer-overflow on address "
          "0x%" PRIx64 "\n",
          addr);
  fprintf(stderr,
          "%s of size %" PRIu64 " at offset %" PRIu64 " of a %" PRIu64
          "-byte global runs %" PRIu64 " bytes past its end\n",
          is_write ? "WRITE" : "READ", bytes, addr - base, size,
          addr + bytes - (base + size));
  if (!name.empty())
    fprintf(stderr, "  of global variable '%s'\n", name.c_str());
  fprintf(stderr, "  global base 0x%" PRIx64 "\n", base);
  fprintf(stderr, "  in %s called from the host\n", api);
  fprintf(stderr, "=============================================="
                  "===================\n");
  S.reported.fetch_add(1, std::memory_order_relaxed);
  return true;
}

// A copy or a fill issued from the host moves memory with no instrumented code
// involved, so its arguments are the only place it can be caught.  Only a
// placed pointer or a described global can be judged: anything else has no
// recorded extent, and is as likely to be the runtime's own staging traffic as
// the program's data.
//
// The transfer is issued either way.  The tool recovers rather than aborts, and
// a copy the program expected to happen silently not happening would be a worse
// bug than the one being reported.
void checkHostTransfer(const void *ptr, uint64_t bytes, bool is_write,
                       const char *api) {
  State &S = st();
  uint64_t addr = (uint64_t)ptr;
  if (!bytes || bytes > kRegionSpan)
    return;
  if (addr - kRegionBase >= kRegionSpan) {
    checkGlobalTransfer(addr, bytes, is_write, api);
    return;
  }

  uint64_t d = addr - kRegionBase;
  unsigned c = (unsigned)(d >> kClassShift);
  if (c >= kNumClasses)
    return;
  uint64_t slot_bytes = 1ULL << (c + kMinSlotLog2);
  uint64_t index = (d & ((1ULL << kClassShift) - 1)) >> (c + kMinSlotLog2);

  uint64_t base = 0, size = 0;
  bool freed = false, known = false;
  {
    std::lock_guard<std::mutex> g(S.mu);
    auto it = S.by_slot.find(tableIndex(c, index));
    if ((known = it != S.by_slot.end())) {
      base = slotAddr(c, index) + it->second.color;
      size = it->second.size;
      freed = it->second.freed;
    }
  }
  // A slot that was never handed out is not evidence of anything: the address
  // is inside the region but names no allocation this tool made.
  if (!known || (!freed && addr >= base && addr + bytes <= base + size)) {
    logf("%s: %s %" PRIu64 " B at 0x%" PRIx64 " %s", api,
         is_write ? "write" : "read", bytes, addr,
         known ? "in bounds" : "in no known slot");
    return;
  }

  bool below = addr < base;
  const char *what = "allocation";
  const char *kind = freed   ? "use-after-free"
                     : below ? "heap-buffer-underflow"
                             : "heap-buffer-overflow";

  fprintf(stderr, "\n=============================================="
                  "===================\n");
  fprintf(stderr, "ERROR: GPUAddressSanitizer: %s on address 0x%" PRIx64 "\n",
          kind, addr);
  if (freed)
    fprintf(stderr,
            "%s of size %" PRIu64 " at offset %" PRIu64 " of a %" PRIu64
            "-byte region that was already freed\n",
            is_write ? "WRITE" : "READ", bytes, addr - base, size);
  else if (below)
    fprintf(stderr,
            "%s of size %" PRIu64 " starting %" PRIu64
            " bytes before a %" PRIu64 "-byte %s\n",
            is_write ? "WRITE" : "READ", bytes, base - addr, size, what);
  else
    fprintf(stderr,
            "%s of size %" PRIu64 " at offset %" PRIu64 " of a %" PRIu64
            "-byte %s runs %" PRIu64 " bytes past its end\n",
            is_write ? "WRITE" : "READ", bytes, addr - base, size, what,
            addr + bytes - (base + size));
  fprintf(stderr,
          "  %s base 0x%" PRIx64 ", class %u (slot %" PRIu64
          " B), slot index %" PRIu64 "\n",
          what, base, c, slot_bytes, index);
  // No block, thread or lane to give: the host did this, not a kernel.
  fprintf(stderr, "  in %s called from the host\n", api);
  fprintf(stderr, "=============================================="
                  "===================\n");
  S.reported.fetch_add(1, std::memory_order_relaxed);
}

// Drains one packet's worth of reports, one per active lane, and answers each
// one.  The reply is the device's instruction to stop or to carry on, and
// sending it last is what guarantees the report is on the user's terminal
// before a trap can take the queue, and the process, down.
void handleReport(rpc::Server::Port &Port) {
  Port.recv_and_send([](rpc::Buffer *Buffer, uint32_t) {
    gpuasan_report_t r;
    memcpy(&r, Buffer->data, sizeof(r));
    printReport(r);
    bool halt = haltOnError() && !r.recover;
    // Said once, and only when it is about to happen: a user who wanted the
    // rest of the run has no way to guess that there is a knob for it.
    static bool told = false;
    if (halt && !told) {
      told = true;
      fprintf(stderr, "[gpuasan] stopping the program; set "
                      "GPUASAN_HALT_ON_ERROR=0 to report and keep going\n");
    }
    fflush(stderr);
    Buffer->data[0] = halt;
  });
}

void rpcServerLoop() {
  State &S = st();
  rpc::Server Server(kRPCPorts, S.rpc_buffer);
  while (!S.stop.load(std::memory_order_relaxed)) {
    auto Port = Server.try_open(S.rpc_lanes);
    if (!Port) {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      continue;
    }
    if (Port->get_opcode() == GPUASAN_REPORT)
      handleReport(*Port);
    else
      errf("unexpected RPC opcode 0x%x on the sanitizer's server",
           Port->get_opcode());
    // The port is released by the optional's destructor.
  }
}

//===----------------------------------------------------------------------===//
// Topology discovery and setup
//===----------------------------------------------------------------------===//

hsa_status_t poolCb(hsa_amd_memory_pool_t pool, void *data) {
  State &S = st();
  hsa_agent_t owner = *(hsa_agent_t *)data;
  hsa_amd_segment_t seg;
  if (real().hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) != HSA_STATUS_SUCCESS ||
      seg != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;
  bool ok = false;
  real().hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &ok);
  if (!ok)
    return HSA_STATUS_SUCCESS;
  uint32_t flags = 0;
  real().hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (!(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
    // An agent usually exposes more than one coarse global pool, and with
    // several GPUs the same program will allocate from more than one of them.
    // Every one is recorded along with the agent it belongs to, so an
    // allocation can be backed from the pool the caller actually named rather
    // than from whichever one happened to be found first.
    S.coarse_pools.insert(pool.handle);
    S.pool_owner[pool.handle] = owner;
    if (!S.have_vram) {
      S.vram = pool;
      S.have_vram = true;
    }
  }
  return HSA_STATUS_SUCCESS;
}

// The RPC buffer must be visible to both sides without an explicit copy, so it
// comes from fine-grained system memory rather than VRAM.
hsa_status_t hostPoolCb(hsa_amd_memory_pool_t pool, void *data) {
  State &S = st();
  hsa_agent_t owner = *(hsa_agent_t *)data;
  hsa_amd_segment_t seg;
  if (real().hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) != HSA_STATUS_SUCCESS ||
      seg != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;
  bool ok = false;
  real().hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &ok);
  uint32_t flags = 0;
  real().hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (!ok || !(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED))
    return HSA_STATUS_SUCCESS;
  // Every one is a candidate for placement; the first also backs the RPC
  // buffer, which is allocated through the real entry point and so never ends
  // up placed itself.
  S.fine_pools.insert(pool.handle);
  S.pool_owner[pool.handle] = owner;
  if (!S.have_host_pool) {
    S.host_pool = pool;
    S.have_host_pool = true;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t agentCb(hsa_agent_t agent, void *) {
  State &S = st();
  hsa_device_type_t type;
  if (real().hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (type == HSA_DEVICE_TYPE_GPU) {
    // Every GPU, not just the first.  The instrumented check is compiled once
    // and runs on whichever device the program launches on, so all of them have
    // to be able to read the table, and an allocation for any of them has to be
    // placed rather than quietly left unchecked.
    S.gpus.push_back(agent);
    if (!S.have_gpu) {
      S.gpu = agent;
      S.have_gpu = true;
    }
    real().hsa_amd_agent_iterate_memory_pools(agent, poolCb, &agent);
  } else if (type == HSA_DEVICE_TYPE_CPU && !S.have_cpu) {
    S.cpu = agent;
    S.have_cpu = true;
    real().hsa_amd_agent_iterate_memory_pools(agent, hostPoolCb, &agent);
  }
  return HSA_STATUS_SUCCESS;
}

// Deferred to the first allocation rather than run from a constructor: the
// VMEM entry points are not usable while hsa_init is still on the stack, and a
// reservation attempted there fails even though the same call succeeds a
// moment later.
bool setup() {
  State &S = st();
  if (S.tried)
    return S.active;
  S.tried = true;

  if (!real().hsa_iterate_agents) {
    errf("HSA runtime entry points not resolvable; disabled");
    return false;
  }

  real().hsa_iterate_agents(agentCb, nullptr);
  if (!S.have_gpu || !S.have_vram) {
    errf("no GPU agent or coarse device pool; disabled");
    return false;
  }

  // VMemoryAddressReserve retries with FixedAddress=0 on failure and returns
  // whatever it got.  The pass hardcodes the base, so anything other than the
  // exact address must disable the tool rather than silently mis-instrument.
  void *va = nullptr;
  if (real().hsa_amd_vmem_address_reserve_align(&va, kRegionSpan, kRegionBase,
                                                1ULL << kClassShift,
                                                0) != HSA_STATUS_SUCCESS) {
    errf("could not reserve %.1f TiB region; disabled",
         kRegionSpan / (1024.0 * 1024.0 * 1024.0 * 1024.0));
    return false;
  }
  if ((uint64_t)va != kRegionBase) {
    errf("region landed at 0x%" PRIx64 " not 0x%" PRIx64 "; disabled",
         (uint64_t)va, kRegionBase);
    real().hsa_amd_vmem_address_free(va, kRegionSpan);
    return false;
  }
  S.region = va;

  void *tva = nullptr;
  if (real().hsa_amd_vmem_address_reserve_align(
          &tva, kTableSpan, kTableBase, 1ULL << 21, 0) != HSA_STATUS_SUCCESS ||
      (uint64_t)tva != kTableBase) {
    errf("could not reserve metadata table at 0x%" PRIx64 "; disabled",
         kTableBase);
    if (tva)
      real().hsa_amd_vmem_address_free(tva, kTableSpan);
    real().hsa_amd_vmem_address_free(S.region, kRegionSpan);
    S.region = nullptr;
    return false;
  }
  S.table = tva;

  for (unsigned c = 0; c < kNumClasses; ++c)
    S.classes[c].base = kRegionBase + ((uint64_t)c << kClassShift);

  // Back the whole table before anything can read it, every class aliasing one
  // allocation full of zeroes.  This is what makes a table load unable to
  // fault: the device bounds the slot index into the span, and every address in
  // the span now resolves to memory that reads as "no allocation".  A check on
  // a wild pointer therefore produces a report instead of a memory violation
  // inside the check itself, which is the difference between a diagnosis and a
  // dead queue.  A class that starts handing out slots swaps its own range for
  // memory it can write.
  if (mapPhysical(tva, kTableClassBytes, &S.zero_handle)) {
    memset(tva, 0, kTableClassBytes);
    S.have_zero = true;
    for (unsigned c = 1; c < kNumClasses; ++c) {
      void *at = (char *)tva + (uint64_t)c * kTableClassBytes;
      if (!mapAlias(at, kTableClassBytes, S.zero_handle)) {
        errf("could not back the table for class %u; a stray pointer into it "
             "will fault instead of being reported",
             c);
        break;
      }
    }
  } else {
    errf("could not back the metadata table with zeroes; a stray pointer will "
         "fault instead of being reported");
  }
  // No class writes through the zero mapping, class zero included: it is shared
  // by all of them, so a store there would appear in every other class's table.
  // Each one swaps for private memory when it first hands out a slot.

  // Both quarantine limits are honoured at once: the slot count keeps a stream
  // of tiny allocations from holding a class hostage, the byte budget keeps a
  // handful of huge ones from holding the device's memory.
  if (const char *q = getenv("GPUASAN_QUARANTINE"))
    S.quarantine_slots = (size_t)strtoull(q, nullptr, 0);
  if (const char *q = getenv("GPUASAN_QUARANTINE_BYTES"))
    S.quarantine_budget = strtoull(q, nullptr, 0);

  S.active = true;
  logf("region 0x%" PRIx64 " (%.0f TiB), table 0x%" PRIx64, kRegionBase,
       kRegionSpan / 1099511627776.0, kTableBase);
  return true;
}

// Give back everything setup() took, in reverse.  Called when the runtime is
// shutting down, which destroys the agents, pools and reservations this all
// refers to; anything kept past that point would name memory nobody owns.
// Leaving the reservations in particular is not survivable for the program,
// whose next hsa_init finds the address space it wanted already occupied.
//
// The counters are deliberately not reset: they describe the process, not the
// runtime, and the summary is printed once at exit however many times the
// program cycled HSA underneath.
void teardown() {
  State &S = st();

  unsigned failed_unmap = 0, failed_release = 0;
  for (size_t i = S.maps.size(); i--;) {
    if (real().hsa_amd_vmem_unmap(S.maps[i].va, S.maps[i].bytes) !=
        HSA_STATUS_SUCCESS)
      ++failed_unmap;
    if (S.maps[i].owned && real().hsa_amd_vmem_handle_release(
                               S.maps[i].handle) != HSA_STATUS_SUCCESS)
      ++failed_release;
  }
  if (failed_unmap || failed_release)
    errf("%u of %zu mappings would not unmap and %u handles would not release",
         failed_unmap, S.maps.size(), failed_release);
  S.maps.clear();
  S.have_zero = false;

  if (S.table) {
    if (real().hsa_amd_vmem_address_free(S.table, kTableSpan) !=
        HSA_STATUS_SUCCESS)
      errf("could not give back the metadata table reservation");
    S.table = nullptr;
  }
  if (S.region) {
    if (real().hsa_amd_vmem_address_free(S.region, kRegionSpan) !=
        HSA_STATUS_SUCCESS)
      errf("could not give back the region reservation");
    S.region = nullptr;
  }

  if (S.inv_queue) {
    real().hsa_queue_destroy(S.inv_queue);
    S.inv_queue = nullptr;
  }
  if (S.inv_signal.handle) {
    real().hsa_signal_destroy(S.inv_signal);
    S.inv_signal = {};
  }
  S.inv_broken = false;

  for (SizeClass &SC : S.classes)
    SC = SizeClass();
  memset(S.table_live, 0, sizeof(S.table_live));
  S.aux_handles.clear();
  S.live.clear();
  S.by_slot.clear();
  S.userdata.clear();
  S.globals.clear();
  S.described.clear();
  S.quarantine.clear();
  S.quarantine_bytes = 0;

  // Agent and pool handles belong to the runtime that is going away, so the
  // next setup() rediscovers them rather than reusing these.
  S.gpus.clear();
  S.coarse_pools.clear();
  S.fine_pools.clear();
  S.pool_owner.clear();
  S.fine_ok = S.fine_checked = false;
  S.have_gpu = S.have_cpu = S.have_vram = S.have_host_pool = false;

  S.active = false;
  S.tried = false;
}

//===----------------------------------------------------------------------===//
// RPC transport
//===----------------------------------------------------------------------===//

// Hand our opcode to a server somebody else already runs, and say whether one
// was there.  Registration only takes effect once that runtime has devices, so
// this cannot be done from a constructor: it is asked at the first image load,
// which is late enough for offload to have brought its plugins up and still
// before any kernel can send a report.
bool registerWithForeignServer() {
  using RegisterFn = void (*)(unsigned (*)(void *, unsigned));
  auto Register =
      (RegisterFn)dlsym(RTLD_DEFAULT, "__tgt_register_rpc_callback");
  if (!Register)
    return false;
  Register(gpuasanServeOpcode);
  logf("reporting through the offload runtime's RPC server");
  return true;
}

// Stand up the report channel for a freshly frozen executable, which is the
// first moment the client symbol can be written.
//
// Whether the process already has an RPC server decides who owns the client.
// Under OpenMP libomptarget allocates the buffer and writes the symbol itself,
// so touching it here would leave the device talking to a server that does not
// know our opcode; a handler is registered with that server instead.  Under HIP
// nothing provides RPC, so we own the whole transport.
void attachRPC(hsa_executable_t exec) {
  State &S = st();
  if (!S.active || !S.have_gpu)
    return;

  hsa_executable_symbol_t sym;
  if (real().hsa_executable_get_symbol_by_name(
          exec, "__llvm_rpc_client", &S.gpu, &sym) != HSA_STATUS_SUCCESS)
    return;

  if (!S.rpc_checked) {
    S.rpc_checked = true;
    S.rpc_owned = !registerWithForeignServer();
  }
  if (!S.rpc_owned)
    return;

  uint64_t addr = 0;
  if (real().hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr) !=
          HSA_STATUS_SUCCESS ||
      !addr)
    return;

  if (!S.rpc_buffer) {
    if (!S.have_host_pool) {
      errf("no fine-grained host pool; reporting disabled");
      return;
    }
    real().hsa_agent_get_info(S.gpu, HSA_AGENT_INFO_WAVEFRONT_SIZE,
                              &S.rpc_lanes);
    uint64_t bytes = rpc::Server::allocation_size(S.rpc_lanes, kRPCPorts);
    void *buf = nullptr;
    if (real().hsa_amd_memory_pool_allocate(S.host_pool, bytes, 0, &buf) !=
            HSA_STATUS_SUCCESS ||
        !buf) {
      errf("could not allocate the RPC buffer; reporting disabled");
      return;
    }
    memset(buf, 0, bytes);
    hsa_agent_t agents[1] = {S.gpu};
    real().hsa_amd_agents_allow_access(1, agents, nullptr, buf);
    S.rpc_buffer = buf;
    // Kept joinable: the buffer it polls belongs to the HSA runtime, so the
    // thread has to be gone before the runtime goes away.
    S.rpc_thread = std::thread(rpcServerLoop);
    logf("serving reports on %u ports, %u lanes", kRPCPorts, S.rpc_lanes);
  }

  rpc::Client client(kRPCPorts, S.rpc_buffer);
  real().hsa_memory_copy((void *)addr, &client, sizeof(client));
}

//===----------------------------------------------------------------------===//
// Device globals
//===----------------------------------------------------------------------===//

struct SymInfo {
  hsa_executable_symbol_t sym;
  uint64_t addr;
  std::string name;
};

hsa_status_t collectVariable(hsa_executable_t, hsa_agent_t,
                             hsa_executable_symbol_t sym, void *data) {
  auto *out = (std::vector<SymInfo> *)data;
  uint32_t kind = 0;
  if (real().hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS ||
      kind != HSA_SYMBOL_KIND_VARIABLE)
    return HSA_STATUS_SUCCESS;

  SymInfo si{sym, 0, {}};
  if (real().hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &si.addr) !=
          HSA_STATUS_SUCCESS ||
      !si.addr)
    return HSA_STATUS_SUCCESS;

  uint32_t len = 0;
  if (real().hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len) ==
          HSA_STATUS_SUCCESS &&
      len && len < 8192) {
    si.name.resize(len);
    if (real().hsa_executable_symbol_get_info(
            sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, si.name.data()) !=
        HSA_STATUS_SUCCESS)
      si.name.clear();
  }
  out->push_back(std::move(si));
  return HSA_STATUS_SUCCESS;
}

// One module's contribution to the globals section: its info block, the
// descriptors it points at, and the window of the section its table covers.
struct GlobalsBlock {
  uint64_t addr = 0; // the info block itself, which is how we know it again
  std::string name;
  uint64_t table = 0;
  uint64_t base = 0, end = 0;
  std::vector<GlobalDesc> descs;
};

// Read one info block and everything it points at.
//
// Everything the compiler could work out is already in the image: which
// variables are in the section, how large each one is, where the table lives,
// and where the section starts.  What it could not know is where the loader put
// any of it, and it does not have to -- those addresses are relocations,
// applied by the time we read them, so the layout is readable rather than
// inferred.
bool readGlobalsBlock(const SymInfo &si, GlobalsBlock &B) {
  uint64_t info[kGlobalsInfoFields] = {};
  if (real().hsa_memory_copy(info, (void *)si.addr, sizeof(info)) !=
      HSA_STATUS_SUCCESS)
    return false;

  uint64_t base = info[kGlobalsInfoBase];
  uint64_t granules = info[kGlobalsInfoGranules];
  uint64_t ndescs = info[kGlobalsInfoNumDescs];
  // The origin has to be granule-aligned, since it is what every offset in the
  // table is measured from; the section's alignment guarantees it.
  if (!base || (base & (kGlobalGranule - 1)) || !granules || !ndescs ||
      !info[kGlobalsInfoTable] || !info[kGlobalsInfoDescs])
    return false;

  B.descs.resize(ndescs);
  if (real().hsa_memory_copy(B.descs.data(), (void *)info[kGlobalsInfoDescs],
                             ndescs * sizeof(GlobalDesc)) != HSA_STATUS_SUCCESS)
    return false;

  B.addr = si.addr;
  B.name = si.name;
  B.table = info[kGlobalsInfoTable];
  B.base = base;
  B.end = base + (granules << kGlobalGranuleLog2);
  return true;
}

// Write one block's table, describing every object in the image that falls in
// the window it covers.
//
// The window starts at the section's `__start_`, which is what the checks
// measure from, and runs as far as the table can name.  An image is usually one
// module, but when it is several they share the section and each has a table of
// its own starting at the same place, so a table is filled from every
// descriptor in the image rather than only from its own module's: a check reads
// the table in its own module, and has to get a straight answer about a global
// that came from another one.
bool writeGlobalsTable(const GlobalsBlock &B,
                       const std::vector<const GlobalDesc *> &All) {
  // Poison first, then carve the objects out of it.  What is left poisoned is
  // every redzone and every byte of alignment padding the linker inserted,
  // which is where an overflow off the end of one global lands instead of in
  // the neighbour.
  std::vector<uint64_t> table((B.end - B.base) >> kGlobalGranuleLog2,
                              kGlobalPoisonEntry);
  for (const GlobalDesc *d : All) {
    if (d->addr < B.base || d->addr + d->size > B.end)
      continue;
    uint64_t begin = d->addr - B.base;
    uint64_t stop = begin + d->size;
    uint64_t entry = makeGlobalEntry(begin, stop);
    for (uint64_t g = begin >> kGlobalGranuleLog2;
         (g << kGlobalGranuleLog2) < stop; ++g)
      table[g] = entry;
  }
  return real().hsa_memory_copy((void *)B.table, table.data(),
                                table.size() * sizeof(uint64_t)) ==
         HSA_STATUS_SUCCESS;
}

// Remember the objects a table now describes, so a report can name the variable
// and a size query can be answered with what the program declared rather than
// with the padded storage.
void recordGlobals(const GlobalsBlock &B) {
  State &S = st();
  for (const GlobalDesc &d : B.descs) {
    if (!d.addr || !d.size)
      continue;
    if (d.addr < B.base || d.addr + d.size > B.end) {
      // Past what the table can name: the linker spread the section further
      // than the compiler's bound allowed for.  Safe, because a granule with no
      // entry reads as zero and every check waves that through, but it is lost
      // coverage.
      ++S.n_globals_unchecked;
      continue;
    }
    GlobalRecord rec{d.addr, d.size, {}};
    if (d.name && d.name_len && d.name_len <= kMaxGlobalNameLen) {
      rec.name.resize(d.name_len);
      if (real().hsa_memory_copy(rec.name.data(), (void *)d.name, d.name_len) !=
          HSA_STATUS_SUCCESS)
        rec.name.clear();
    }
    S.globals.push_back(std::move(rec));
    ++S.n_globals;
  }
}

void describeExecutableGlobals(hsa_executable_t exec) {
  State &S = st();
  if (!S.active || !S.have_gpu || !real().hsa_executable_iterate_agent_symbols)
    return;

  std::vector<SymInfo> syms;
  if (real().hsa_executable_iterate_agent_symbols(exec, S.gpu, collectVariable,
                                                  &syms) != HSA_STATUS_SUCCESS)
    return;

  // One info block per module, and an image is usually one module but need not
  // be, so this iterates rather than looking a single name up.
  constexpr size_t kPrefixLen = sizeof(kGlobalsInfoPrefix) - 1;
  std::vector<GlobalsBlock> blocks;
  for (const SymInfo &si : syms) {
    if (si.name.compare(0, kPrefixLen, kGlobalsInfoPrefix) != 0)
      continue;
    if (!S.described.insert(si.addr).second)
      continue;
    GlobalsBlock B;
    if (readGlobalsBlock(si, B))
      blocks.push_back(std::move(B));
  }
  if (blocks.empty())
    return;

  std::vector<const GlobalDesc *> all;
  for (const GlobalsBlock &B : blocks)
    for (const GlobalDesc &d : B.descs)
      if (d.addr && d.size)
        all.push_back(&d);

  for (const GlobalsBlock &B : blocks) {
    if (!writeGlobalsTable(B, all)) {
      errf("could not write the globals table for %s; its globals stay "
           "unchecked",
           B.name.c_str());
      S.n_globals_unchecked += B.descs.size();
      continue;
    }
    recordGlobals(B);
    logf("%zu globals at 0x%" PRIx64 " + %" PRIu64 " B", B.descs.size(), B.base,
         B.end - B.base);
  }
  invalidateCaches();

  // Sorted, so a report can find the object an address belongs to and a size
  // query can be answered without a scan.
  std::sort(S.globals.begin(), S.globals.end(),
            [](const GlobalRecord &a, const GlobalRecord &b) {
              return a.addr < b.addr;
            });
}

// Reports are picked up by a polling thread, so a fault found in the last
// kernel can still be sitting in the buffer when the process starts tearing
// down. Killing the server there loses exactly the report the run existed to
// produce, so wait for the count to stop moving first.
//
// FIXME: A timeout is a stopgap.  The right drain point is kernel completion,
// which needs a dispatch hook.
void drainReports() {
  State &S = st();
  if (!S.rpc_owned || !S.rpc_buffer)
    return;
  uint64_t last = S.reported.load(std::memory_order_relaxed);
  for (int idle = 0, waited = 0; idle < 4 && waited < 50; ++waited) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    uint64_t now = S.reported.load(std::memory_order_relaxed);
    idle = now == last ? idle + 1 : 0;
    last = now;
  }
}

// Stop serving reports and give the buffer back.  Has to run while the runtime
// is still up: the thread polls memory that came from an HSA pool, so the
// thread goes first and the pool call second.
void stopRPC() {
  State &S = st();
  S.stop.store(true);
  if (S.rpc_thread.joinable())
    S.rpc_thread.join();

  std::lock_guard<std::mutex> g(S.mu);
  if (S.rpc_buffer && S.rpc_owned)
    real().hsa_amd_memory_pool_free(S.rpc_buffer);
  S.rpc_buffer = nullptr;
  S.rpc_checked = false;
  S.rpc_owned = false;
  S.stop.store(false);
}

void finalReport() {
  State &S = st();
  static bool done = false;
  if (done)
    return;
  done = true;
  uint64_t reported = S.reported.load(std::memory_order_relaxed);
  uint64_t passed = S.n_pass_inactive + S.n_pass_fine + S.n_pass_toobig +
                    S.n_pass_exhausted + S.n_pass_backing;
  // Coverage that was wanted and not achieved, as opposed to memory that was
  // never in scope: every process allocates kernarg, signal and staging buffers
  // out of fine-grained pools, so counting those as lost coverage would put a
  // warning on every run and make the one that matters invisible.
  uint64_t lost = S.n_pass_inactive + S.n_pass_toobig + S.n_pass_exhausted +
                  S.n_pass_backing;
  if (!verbose() && reported == 0 && lost == 0)
    return;
  fprintf(stderr,
          "[gpuasan] %" PRIu64 " placed, %" PRIu64 " freed, %" PRIu64
          " passed through, %" PRIu64 " globals described, %" PRIu64
          " cache flushes, %" PRIu64 " errors reported\n",
          S.n_alloc, S.n_free, passed, S.n_globals, S.n_inv, reported);
  // What the quarantine is still holding is the answer to "why did that
  // use-after-free not fire", so it is worth saying out loud.
  if (verbose())
    fprintf(stderr,
            "[gpuasan] quarantine holds %zu slots (%" PRIu64 " B); %" PRIu64
            " retired early\n",
            S.quarantine.size(), S.quarantine_bytes, S.n_retired);
  // A global the linker laid out past the reach of its module's table is a
  // global nothing checks, which is the same silent hole as an unplaced
  // allocation.  It takes either a table too small for the section or an image
  // built from several instrumented objects, whose tables all start at the
  // section's beginning and so cover only as far as the shortest of them
  // reaches.
  if (S.n_globals_unchecked)
    fprintf(stderr,
            "[gpuasan] WARNING: %" PRIu64
            " global variables are unchecked; raise -mllvm "
            "-gpuasan-globals-budget past the size of the image's data, or "
            "build the device side as one module\n",
            S.n_globals_unchecked);
  // An allocation that should have been placed and was not is an allocation
  // nobody is checking, so say so and say why.  A run that quietly checked
  // nothing is indistinguishable from a clean one, which is the worst thing a
  // tool like this can do.
  if (lost)
    fprintf(stderr,
            "[gpuasan] WARNING: %" PRIu64
            " device allocations are unchecked (%" PRIu64
            " before setup, %" PRIu64 " too large, %" PRIu64
            " out of slots, %" PRIu64 " out of memory)\n",
            lost, S.n_pass_inactive, S.n_pass_toobig, S.n_pass_exhausted,
            S.n_pass_backing);
  if (verbose() && S.n_pass_fine)
    fprintf(stderr,
            "[gpuasan] %" PRIu64 " allocations were not device memory\n",
            S.n_pass_fine);
  if (S.n_double_free)
    fprintf(stderr, "[gpuasan] %" PRIu64 " double frees\n", S.n_double_free);
}

//===----------------------------------------------------------------------===//
// Hooks
//
// Each of these replaces one entry in the API tables and chains to whatever was
// in that slot before, which is ROCr's own implementation unless another tool
// got there first.  They are not exported: being reached is a property of being
// in the table, and a client that resolved the entry point through dlsym gets
// ROCr's trampoline, which reads the table on every call.
//===----------------------------------------------------------------------===//

// Freeze is the only moment that works for either job below: it is the first
// point at which the image's symbols and its relocated data exist, and it is
// still before the program can dispatch anything or ask where a global lives.
hsa_status_t hookExecutableFreeze(hsa_executable_t exec, const char *options) {
  hsa_status_t s = real().hsa_executable_freeze(exec, options);
  if (s == HSA_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> g(st().mu);
    setup();
    attachRPC(exec);
    describeExecutableGlobals(exec);
  }
  return s;
}

// A checked global keeps its address, so nothing has to be corrected there. Its
// ELF size is also right, because the padded storage is private and the name is
// an alias whose size AsmPrinter takes from the declared type -- but only when
// the compiler could arrange that, so the answer is still checked against what
// the descriptors said the program asked for.  Every path that reaches a device
// global by name, hipGetSymbolAddress and hipMemcpyToSymbol and the OpenMP
// equivalents alike, bottoms out in this one query.
hsa_status_t hookExecutableSymbolGetInfo(hsa_executable_symbol_t symbol,
                                         hsa_executable_symbol_info_t attribute,
                                         void *value) {
  hsa_status_t s =
      real().hsa_executable_symbol_get_info(symbol, attribute, value);
  if (s != HSA_STATUS_SUCCESS ||
      attribute != HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE)
    return s;

  uint64_t addr = 0;
  if (real().hsa_executable_symbol_get_info(
          symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr) !=
      HSA_STATUS_SUCCESS)
    return s;

  State &S = st();
  std::lock_guard<std::mutex> g(S.mu);
  if (const GlobalRecord *rec = findGlobal(addr))
    if (rec->addr == addr)
      *(uint32_t *)value = (uint32_t)rec->size;
  return s;
}

// Whether a slot in the region can stand in for host memory.  Device memory is
// only ever read through a pointer the program hands to a kernel, but pinned
// memory is also loaded and stored by the host directly, and a mapping that the
// CPU cannot reach would fault in the program rather than in a kernel.  Proven
// once, on a real slot backed by the real pool, because being wrong here does
// not cost a missed report -- it breaks a program that works today.
//
// This settles reachability, not coherence: that the device's view agrees with
// the host's is what the clean test exercises end to end.
bool fineGrainedUsable() {
  State &S = st();
  if (S.fine_checked)
    return S.fine_ok;
  S.fine_checked = true;
  if (!S.have_host_pool)
    return false;

  // Backing a mapping from a system pool is a capability ROCr only gained in
  // mid-2026; an older runtime refuses the pool outright.  Ask for a single
  // page first, so that costs one rejected call rather than a failed chunk.
  hsa_amd_vmem_alloc_handle_t probe{};
  hsa_status_t s = real().hsa_amd_vmem_handle_create(
      S.host_pool, kPage, MEMORY_TYPE_PINNED, 0, &probe);
  if (s != HSA_STATUS_SUCCESS) {
    logf("this runtime will not back a mapping from a host pool (status %d); "
         "pinned allocations stay unplaced",
         (int)s);
    return false;
  }
  real().hsa_amd_vmem_handle_release(probe);

  constexpr uint64_t kProbe = 256;
  unsigned c = classForAlloc(kProbe);
  uint64_t index = acquireSlot(c, kProbe, &S.host_pool);
  if (index == UINT64_MAX) {
    errf("host pool will not back a placed slot; pinned memory stays unplaced");
    return false;
  }

  auto *p = (volatile uint64_t *)(slotAddr(c, index) + colorFor(kProbe, c));
  constexpr unsigned n = kProbe / sizeof(uint64_t);
  for (unsigned i = 0; i < n; ++i)
    p[i] = 0x5a5a5a5a00000000ULL | i;
  bool ok = true;
  for (unsigned i = 0; i < n; ++i)
    ok &= p[i] == (0x5a5a5a5a00000000ULL | i);

  releaseSlot(c, index, kProbe);
  if (!ok)
    errf("placed host memory did not read back; pinned memory stays unplaced");
  else
    logf("host pool is placeable; pinned allocations will be checked");
  S.fine_ok = ok;
  return ok;
}

hsa_status_t hookMemoryPoolAllocate(hsa_amd_memory_pool_t pool, size_t size,
                                    uint32_t flags, void **ptr) {
  State &S = st();
  std::unique_lock<std::mutex> g(S.mu);
  setup();

  // Only user data allocations are placed.  The runtime's own executable and
  // queue allocations carry flags and must keep whatever address ROCr wants to
  // give them.  Any coarse pool on any GPU qualifies: with several devices the
  // program will use more than one, and limiting this to the first one seen
  // would leave the rest unchecked.  A fine-grained pool qualifies too, once
  // the region has been shown able to stand in for host memory, which is what
  // brings pinned allocations under the same checks.
  bool coarse = S.coarse_pools.count(pool.handle) != 0;
  bool fine = !coarse && S.fine_pools.count(pool.handle) != 0 && S.active &&
              flags == 0 && fineGrainedUsable();
  bool eligible = S.active && flags == 0 && size > 0 && size <= kMaxObject &&
                  (coarse || fine);
  if (!eligible) {
    // Exhaustive on purpose: an allocation that reaches the stock allocator
    // without being counted is coverage lost with nothing to show for it.
    if (!S.active)
      ++S.n_pass_inactive;
    else if ((coarse || fine) && size > kMaxObject)
      ++S.n_pass_toobig;
    else
      // A pool that is neither placeable kind, or one of the runtime's own
      // flagged requests: never memory a pointer check was meant to cover.
      ++S.n_pass_fine;
    g.unlock();
    return real().hsa_amd_memory_pool_allocate(pool, size, flags, ptr);
  }

  // The class that fits the allocation plus a redzone, so there is always guard
  // space past the end even when the size is itself a power of two.
  unsigned c = classForAlloc(size);
  uint64_t index = acquireSlot(c, size, &pool);
  // The class is exhausted or out of memory: hand it to the real allocator,
  // which returns an address outside the region that every check then ignores.
  // The counters were bumped inside acquireSlot, which knows which it was.
  if (index == UINT64_MAX) {
    g.unlock();
    return real().hsa_amd_memory_pool_allocate(pool, size, flags, ptr);
  }

  uint64_t addr = slotAddr(c, index);
  // Push the allocation up inside its slot so an underflow has padding to land
  // in, which is what makes a read just below the base reportable rather than a
  // silent hit on the previous allocation.
  uint64_t color = colorFor(size, c);
  writeEntry(c, index, makeEntry(size, color));
  // A neighbouring slot's entry shares this cache line, so a kernel that has
  // already checked against one of them may be holding this line in L2.
  invalidateCaches();

  *ptr = (void *)(addr + color);
  LiveAlloc rec{tableIndex(c, index), c, index, size, color, false, 0};
  S.live[(uint64_t)*ptr] = rec;
  S.by_slot[rec.slot_id] = rec;
  ++S.n_alloc;
  logf("alloc %zu B -> 0x%" PRIx64 " class %u slot %" PRIu64 " color %" PRIu64,
       size, (uint64_t)*ptr, c, index, color);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hookMemoryAsyncCopy(void *dst, hsa_agent_t dst_agent,
                                 const void *src, hsa_agent_t src_agent,
                                 size_t size, uint32_t num_dep_signals,
                                 const hsa_signal_t *dep_signals,
                                 hsa_signal_t completion_signal) {
  checkHostTransfer(dst, size, true, "hsa_amd_memory_async_copy");
  checkHostTransfer(src, size, false, "hsa_amd_memory_async_copy");
  return real().hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent, size,
                                          num_dep_signals, dep_signals,
                                          completion_signal);
}

hsa_status_t hookMemoryAsyncCopyOnEngine(void *dst, hsa_agent_t dst_agent,
                                         const void *src, hsa_agent_t src_agent,
                                         size_t size, uint32_t num_dep_signals,
                                         const hsa_signal_t *dep_signals,
                                         hsa_signal_t completion_signal,
                                         uint32_t engine_id,
                                         bool force_copy_on_sdma) {
  checkHostTransfer(dst, size, true, "hsa_amd_memory_async_copy_on_engine");
  checkHostTransfer(src, size, false, "hsa_amd_memory_async_copy_on_engine");
  return real().hsa_amd_memory_async_copy_on_engine(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals,
      completion_signal, engine_id, force_copy_on_sdma);
}

hsa_status_t hookMemoryCopy(void *dst, const void *src, size_t size) {
  checkHostTransfer(dst, size, true, "hsa_memory_copy");
  checkHostTransfer(src, size, false, "hsa_memory_copy");
  return real().hsa_memory_copy(dst, src, size);
}

hsa_status_t hookMemoryFill(void *ptr, uint32_t value, size_t count) {
  checkHostTransfer(ptr, count * sizeof(uint32_t), true, "hsa_amd_memory_fill");
  return real().hsa_amd_memory_fill(ptr, value, count);
}

hsa_status_t hookMemoryPoolFree(void *ptr) {
  State &S = st();
  std::unique_lock<std::mutex> g(S.mu);
  auto it = S.live.find((uint64_t)ptr);
  if (it == S.live.end()) {
    // An address inside the region that is not a live base came from here and
    // has already been freed, or is an interior pointer.  Either way ROCr never
    // allocated it and would fail or corrupt its own bookkeeping, so this is
    // reported and stops here.
    if ((uint64_t)ptr - kRegionBase < kRegionSpan) {
      ++S.n_double_free;
      const LiveAlloc *prev = nullptr;
      uint64_t d = (uint64_t)ptr - kRegionBase;
      unsigned c = (unsigned)(d >> kClassShift);
      if (c < kNumClasses) {
        uint64_t idx = (d & ((1ULL << kClassShift) - 1)) >> (c + kMinSlotLog2);
        auto sit = S.by_slot.find(tableIndex(c, idx));
        if (sit != S.by_slot.end())
          prev = &sit->second;
      }
      fprintf(stderr, "\n=============================================="
                      "===================\n");
      fprintf(stderr,
              "ERROR: GPUAddressSanitizer: attempting double free on address "
              "0x%" PRIx64 "\n",
              (uint64_t)ptr);
      if (prev)
        fprintf(stderr,
                "  address is %" PRIu64 " bytes into a %" PRIu64
                "-byte allocation that was already freed\n",
                (uint64_t)ptr -
                    (slotAddr(prev->cls, prev->index) + prev->color),
                prev->size);
      fprintf(stderr, "=============================================="
                      "===================\n");
      S.reported.fetch_add(1, std::memory_order_relaxed);
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    g.unlock();
    return real().hsa_amd_memory_pool_free(ptr);
  }
  LiveAlloc la = it->second;
  S.live.erase(it);
  LiveAlloc &rec = S.by_slot[la.slot_id];
  rec.freed = true;
  rec.free_serial = ++S.free_serial;
  // Poisoning the entry rather than clearing it is what makes the check fire
  // while still describing what used to be here.  The physical pages stay
  // mapped, because unmapping costs ~15 us per call and the check no longer
  // needs the memory to be unreachable to catch the access.
  writeEntry(la.cls, la.index, makeFreedEntry(la.size, la.color));
  // Without this the poison can sit behind a stale L2 line and the next kernel
  // keeps passing checks against memory it no longer owns.
  invalidateCaches();
  releaseSlot(la.cls, la.index, la.size);
  ++S.n_free;
  logf("free 0x%" PRIx64 " class %u slot %" PRIu64, (uint64_t)ptr, la.cls,
       la.index);
  return HSA_STATUS_SUCCESS;
}

// A slot has to become reachable by the agents the program names, exactly as a
// stock allocation would.  This cannot be forwarded: that entry point works on
// pointers ROCr allocated, and ours came from VMEM, where the equivalent is
// set_access on the mapping.  Nor can it be answered with success and nothing
// else -- that is a lie the program finds out about as a page fault the first
// time the peer touches the buffer.
hsa_status_t hookAgentsAllowAccess(uint32_t num_agents,
                                   const hsa_agent_t *agents,
                                   const uint32_t *flags, const void *ptr) {
  State &S = st();
  std::unique_lock<std::mutex> g(S.mu);
  auto it = S.live.find((uint64_t)ptr);
  if (it == S.live.end()) {
    g.unlock();
    return real().hsa_amd_agents_allow_access(num_agents, agents, flags, ptr);
  }

  // Whatever the mapping already had, plus what was asked for.  Access is set
  // for the whole mapping the slot lives in, because that is the granularity
  // set_access works at; the extra reach covers neighbouring slots of the same
  // class, which are either unallocated or already reachable by every agent.
  const LiveAlloc &la = it->second;
  uint64_t base = 0, bytes = 0;
  if (!mappingFor(la.cls, la.index, base, bytes))
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  std::vector<hsa_amd_memory_access_desc_t> desc = allAgentsDesc();
  for (uint32_t i = 0; i < num_agents; ++i) {
    bool have = false;
    for (const hsa_amd_memory_access_desc_t &d : desc)
      have |= d.agent_handle.handle == agents[i].handle;
    if (!have)
      desc.push_back({HSA_ACCESS_PERMISSION_RW, agents[i]});
  }
  hsa_status_t s = real().hsa_amd_vmem_set_access((void *)base, bytes,
                                                  desc.data(), desc.size());
  if (s != HSA_STATUS_SUCCESS)
    errf("could not grant %u agents access to 0x%" PRIx64 " (status %d)",
         num_agents, (uint64_t)ptr, (int)s);
  return s;
}

// Two things have to be corrected here, and CLR is sensitive to both.  A
// VMEM-backed pointer reports HSA_EXT_POINTER_TYPE_HSA_VMEM rather than
// HSA_EXT_POINTER_TYPE_HSA, and rocdevice.cpp branches on that when deciding
// coherency; and sizeInBytes reports the *mapped* size, which for us is the
// slot or chunk rather than what the program asked for, which would make
// hipMemGetAddressRange lie.  The underlying call is still made first so that
// the accessible-agent list and the remaining fields stay whatever ROCr thinks.
hsa_status_t hookPointerInfo(const void *ptr, hsa_amd_pointer_info_t *info,
                             void *(*alloc)(size_t),
                             uint32_t *num_agents_accessible,
                             hsa_agent_t **accessible) {
  State &S = st();
  LiveAlloc rec;
  bool ours = false;
  {
    std::lock_guard<std::mutex> g(S.mu);
    if (const LiveAlloc *r = findByAddr((uint64_t)ptr)) {
      rec = *r;
      ours = true;
    }
  }
  if (!ours)
    return real().hsa_amd_pointer_info(ptr, info, alloc, num_agents_accessible,
                                       accessible);

  hsa_status_t s = real().hsa_amd_pointer_info(
      ptr, info, alloc, num_agents_accessible, accessible);
  if (!info)
    return s;

  uint64_t base = slotAddr(rec.cls, rec.index) + rec.color;
  if (s != HSA_STATUS_SUCCESS) {
    // Synthesize the whole thing if ROCr refused to describe it.
    memset(info, 0, sizeof(*info));
    info->size = sizeof(*info);
    info->agentOwner = S.gpu;
    if (num_agents_accessible)
      *num_agents_accessible = 0;
  }
  info->type = HSA_EXT_POINTER_TYPE_HSA;
  info->agentBaseAddress = (void *)base;
  info->hostBaseAddress = (void *)base;
  info->sizeInBytes = rec.size;
  {
    std::lock_guard<std::mutex> g(S.mu);
    auto ud = S.userdata.find((uint64_t)info->agentBaseAddress);
    if (ud != S.userdata.end())
      info->userData = ud->second;
  }
  return HSA_STATUS_SUCCESS;
}

// CLR attaches its own bookkeeping to allocations through this.  ROCr keys it
// off its allocation records, which do not describe a slot carved out of a
// shared chunk, so keep it ourselves for placed pointers.
hsa_status_t hookPointerInfoSetUserdata(const void *ptr, void *userdata) {
  State &S = st();
  {
    std::lock_guard<std::mutex> g(S.mu);
    if (findByAddr((uint64_t)ptr)) {
      S.userdata[(uint64_t)ptr] = userdata;
      return HSA_STATUS_SUCCESS;
    }
  }
  return real().hsa_amd_pointer_info_set_userdata(ptr, userdata);
}

// The last point at which a report can still be collected: after this the
// queues and the RPC buffer belong to nobody.  A program that ends without
// synchronizing leaves reports in the buffer, and this is where they are picked
// up.
hsa_status_t hookInit(void) {
  hsa_status_t s = real().hsa_init();
  if (s == HSA_STATUS_SUCCESS) {
    State &S = st();
    std::lock_guard<std::mutex> g(S.mu);
    ++S.hsa_refs;
  }
  return s;
}

hsa_status_t hookShutDown(void) {
  State &S = st();
  bool last;
  {
    std::lock_guard<std::mutex> g(S.mu);
    last = S.hsa_refs <= 1;
    if (S.hsa_refs)
      --S.hsa_refs;
  }

  // Give everything back before the reference that keeps the runtime alive goes
  // away.  Past this call the agents, pools and reservations below stop being
  // things anyone owns, and a region left reserved is not merely leaked: the
  // program's next hsa_init finds the address space taken and fails.
  if (last) {
    drainReports();
    stopRPC();
    std::lock_guard<std::mutex> g(S.mu);
    teardown();
  }
  return real().hsa_shut_down();
}

} // namespace

//===----------------------------------------------------------------------===//
// Tool interface
//
// ROCr scans the loaded objects at hsa_init for one that exports
// HSA_AMD_TOOL_PRIORITY and calls OnLoad on each, lowest priority first, giving
// each one the API tables to modify.  This is the whole activation mechanism:
// nothing here runs unless the process initialises the HSA runtime.  Being
// loaded is enough to be found; HSA_TOOLS_LIB names one that is not.
//===----------------------------------------------------------------------===//

// Ordering against other tools.  ROCr sorts the tools it found by this and
// calls OnLoad in that order, each one replacing the entries it finds, so a
// lower value is loaded earlier and ends up nearer the runtime, underneath
// everything loaded after it.  Underneath is where this belongs: a profiler
// above it sees the size and address the program asked for rather than the slot
// the allocation was placed in.  The other tools use 25 (rocprofiler), 50
// (roctracer) and 1025/1050 (their front-ends), and a tool named in
// HSA_TOOLS_LIB is ordered ahead of every tool that was merely found, whatever
// its priority.  Const because ROCr reads it as soon as it opens the library,
// so it has to be initialised by the loader and not by a constructor.
GPUASAN_INTERFACE const uint32_t HSA_AMD_TOOL_PRIORITY = 10;

GPUASAN_INTERFACE bool OnLoad(void *root, uint64_t runtime_version,
                              uint64_t failed_tool_count,
                              const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;

  if (getenv("GPUASAN_DISABLE")) {
    logf("disabled by GPUASAN_DISABLE");
    return true;
  }

  auto *T = reinterpret_cast<hsa_api_table_t *>(root);
  if (!T || T->version.major_id != GPUASAN_HSA_API_TABLE_MAJOR_VERSION) {
    errf("HSA API table is version %u, expected %u; disabled",
         T ? T->version.major_id : 0, GPUASAN_HSA_API_TABLE_MAJOR_VERSION);
    return true;
  }
  void *core = T->core;
  void *amd = T->amd_ext;
  if (!core || !amd) {
    errf("HSA API table has no core or AMD extension table; disabled");
    return true;
  }
  auto *core_version = reinterpret_cast<hsa_api_table_version_t *>(core);
  auto *amd_version = reinterpret_cast<hsa_api_table_version_t *>(amd);
  if (core_version->major_id != GPUASAN_HSA_CORE_API_TABLE_MAJOR_VERSION ||
      amd_version->major_id != GPUASAN_HSA_AMD_EXT_TABLE_MAJOR_VERSION) {
    errf("HSA sub-table versions are %u/%u, expected %u/%u; disabled",
         core_version->major_id, amd_version->major_id,
         GPUASAN_HSA_CORE_API_TABLE_MAJOR_VERSION,
         GPUASAN_HSA_AMD_EXT_TABLE_MAJOR_VERSION);
    return true;
  }

  // Every entry point the tool calls, taken from the table rather than from the
  // dynamic symbol table.  For a slot that is also hooked below this is the
  // implementation that was there first, which is ROCr's own unless a tool of
  // lower priority already replaced it -- calling it is what keeps that tool
  // working.
#define BIND(Table, Name, Slot)                                                \
  if (!hsaTableHasSlot(Table, Slot)) {                                         \
    errf("HSA runtime has no slot for %s; disabled", #Name);                   \
    return true;                                                               \
  }                                                                            \
  RealApiTable.Name = (decltype(RealApiTable.Name))hsaTableSlots(Table)[Slot]; \
  if (!RealApiTable.Name) {                                                    \
    errf("HSA runtime has a null entry for %s; disabled", #Name);              \
    return true;                                                               \
  }
#define BIND_CORE(Name, Slot) BIND(core, Name, Slot)
#define BIND_AMD(Name, Slot) BIND(amd, Name, Slot)
  GPUASAN_HSA_CORE_API(BIND_CORE)
  GPUASAN_HSA_AMD_API(BIND_AMD)
#undef BIND_AMD
#undef BIND_CORE
#undef BIND

  // Nothing above this point has changed the runtime's behaviour, so a bail-out
  // leaves the process exactly as it was.  From here it is hooked.
  // The init that brought the tool in is this one, and it never reaches the
  // hook below because the hook is only installed now.
  {
    State &S = st();
    std::lock_guard<std::mutex> g(S.mu);
    S.hsa_refs = 1;
  }

#define HOOK(Table, Slot, Fn) hsaTableSlots(Table)[Slot] = (void *)&Fn;
  HOOK(core, GPUASAN_CORE_INIT, hookInit)
  HOOK(core, GPUASAN_CORE_SHUT_DOWN, hookShutDown)
  HOOK(core, GPUASAN_CORE_EXECUTABLE_FREEZE, hookExecutableFreeze)
  HOOK(core, GPUASAN_CORE_EXECUTABLE_SYMBOL_GET_INFO,
       hookExecutableSymbolGetInfo)
  HOOK(amd, GPUASAN_AMD_MEMORY_POOL_ALLOCATE, hookMemoryPoolAllocate)
  HOOK(amd, GPUASAN_AMD_MEMORY_POOL_FREE, hookMemoryPoolFree)
  HOOK(core, GPUASAN_CORE_MEMORY_COPY, hookMemoryCopy)
  HOOK(amd, GPUASAN_AMD_MEMORY_ASYNC_COPY, hookMemoryAsyncCopy)
  HOOK(amd, GPUASAN_AMD_MEMORY_ASYNC_COPY_ON_ENGINE,
       hookMemoryAsyncCopyOnEngine)
  HOOK(amd, GPUASAN_AMD_MEMORY_FILL, hookMemoryFill)
  HOOK(amd, GPUASAN_AMD_AGENTS_ALLOW_ACCESS, hookAgentsAllowAccess)
  HOOK(amd, GPUASAN_AMD_POINTER_INFO, hookPointerInfo)
  HOOK(amd, GPUASAN_AMD_POINTER_INFO_SET_USERDATA, hookPointerInfoSetUserdata)
#undef HOOK

  logf("installed into the HSA API tables (runtime version %" PRIu64 ")",
       runtime_version);
  return true;
}

// ROCr calls this at hsa_shut_down, before it restores the tables.  The reports
// still in the buffer are collected in the shut-down hook rather than here,
// because by this point the tables are being taken apart and a copy out of
// device memory is no longer something to attempt.
//
// Too late to give anything back: ROCr drops the last reference before it tells
// a tool about the shutdown, and from then on every entry point returns
// NOT_INITIALIZED.  The release happens in the shut-down hook instead, and all
// that is left here is to make sure nothing hands out an address in a region
// that is going away -- which matters when the program shut the runtime down
// by some path that did not come through the hook.
GPUASAN_INTERFACE void OnUnload() {
  State &S = st();
  std::lock_guard<std::mutex> g(S.mu);
  S.active = false;
  S.hsa_refs = 0;
}

//===----------------------------------------------------------------------===//
// Reporting through a server somebody else owns
//===----------------------------------------------------------------------===//

// Matches libomptarget's registration signature.  Only reachable when another
// component already owns the RPC transport for the device.
GPUASAN_INTERFACE unsigned gpuasanServeOpcode(void *PortPtr, unsigned) {
  auto *Port = reinterpret_cast<rpc::Server::Port *>(PortPtr);
  if (Port->get_opcode() != GPUASAN_REPORT)
    return rpc::RPC_UNHANDLED_OPCODE;
  handleReport(*Port);
  return rpc::RPC_SUCCESS;
}

__attribute__((destructor)) static void atExit() {
  drainReports();
  st().stop.store(true);
  finalReport();
}
