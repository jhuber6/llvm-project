//===-- gpuasan_hsa.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Just enough of the HSA interface to place device allocations, declared here
// rather than included from ROCm.  A sanitizer runtime cannot require a vendor
// SDK to be present at build time, and it must not link against one either:
// every entry point below is resolved at runtime, so a process that never
// loads the HSA runtime never notices this code exists.  Offload's
// dynamic_hsa headers take the same approach for the same reason.
//
// Only what the runtime calls is declared.  The layouts here are ABI, so they
// are asserted against the values measured from ROCm's own headers; a
// mismatch is a build failure rather than a corrupted struct at runtime.
//
// The API tables at the bottom are how the tool reaches the allocator: ROCr's
// own exported entry points are trampolines through them, so replacing an
// entry there catches every caller, including one that resolved the symbol
// with dlsym.
//
//===----------------------------------------------------------------------===//

#ifndef GPUASAN_HSA_H
#define GPUASAN_HSA_H

#include <stddef.h>
#include <stdint.h>

extern "C" {

//===----------------------------------------------------------------------===//
// Handles.  Every HSA object is an opaque 64-bit handle in a struct.
//===----------------------------------------------------------------------===//

typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;
typedef struct hsa_executable_s {
  uint64_t handle;
} hsa_executable_t;
typedef struct hsa_executable_symbol_s {
  uint64_t handle;
} hsa_executable_symbol_t;
typedef struct hsa_amd_memory_pool_s {
  uint64_t handle;
} hsa_amd_memory_pool_t;
typedef struct hsa_amd_vmem_alloc_handle_s {
  uint64_t handle;
} hsa_amd_vmem_alloc_handle_t;
typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;

typedef int64_t hsa_signal_value_t;
typedef uint32_t hsa_queue_type32_t;

//===----------------------------------------------------------------------===//
// Enumerations.  Only the members the runtime names are given.
//===----------------------------------------------------------------------===//

typedef enum {
  HSA_STATUS_SUCCESS = 0,
  HSA_STATUS_ERROR = 0x1000,
  HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003,
} hsa_status_t;

typedef enum {
  HSA_DEVICE_TYPE_CPU = 0,
  HSA_DEVICE_TYPE_GPU = 1,
} hsa_device_type_t;

typedef enum {
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13,
  HSA_AGENT_INFO_DEVICE = 17,
} hsa_agent_info_t;

typedef enum {
  HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
} hsa_executable_symbol_info_t;

typedef enum { HSA_SYMBOL_KIND_VARIABLE = 0 } hsa_symbol_kind_t;

typedef enum { HSA_AMD_SEGMENT_GLOBAL = 0 } hsa_amd_segment_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,
} hsa_amd_memory_pool_info_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2,
} hsa_amd_memory_pool_global_flag_t;

typedef enum {
  HSA_EXT_POINTER_TYPE_HSA = 1,
  HSA_EXT_POINTER_TYPE_HSA_VMEM = 6,
} hsa_amd_pointer_type_t;

typedef enum { HSA_ACCESS_PERMISSION_RW = 3 } hsa_access_permission_t;

typedef enum {
  MEMORY_TYPE_NONE = 0,
  MEMORY_TYPE_PINNED = 1,
} hsa_amd_memory_type_t;

typedef enum { HSA_QUEUE_TYPE_SINGLE = 1 } hsa_queue_type_t;

typedef enum { HSA_SIGNAL_CONDITION_EQ = 0 } hsa_signal_condition_t;

typedef enum {
  HSA_WAIT_STATE_BLOCKED = 0,
  HSA_WAIT_STATE_ACTIVE = 1,
} hsa_wait_state_t;

// Packet header field offsets and the two values the runtime builds a header
// from.  A barrier-AND whose acquire scope is system is the architected way to
// ask the command processor for the cache maintenance that makes a host store
// to device memory visible to a later dispatch.
typedef enum {
  HSA_PACKET_HEADER_TYPE = 0,
  HSA_PACKET_HEADER_BARRIER = 8,
  HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
  HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11,
} hsa_packet_header_t;

typedef enum { HSA_PACKET_TYPE_BARRIER_AND = 3 } hsa_packet_type_t;

typedef enum {
  HSA_FENCE_SCOPE_NONE = 0,
  HSA_FENCE_SCOPE_AGENT = 1,
  HSA_FENCE_SCOPE_SYSTEM = 2,
} hsa_fence_scope_t;

//===----------------------------------------------------------------------===//
// Structures passed across the boundary, so the layout is load bearing.
//===----------------------------------------------------------------------===//

typedef struct hsa_amd_pointer_info_s {
  uint32_t size;
  hsa_amd_pointer_type_t type;
  void *agentBaseAddress;
  void *hostBaseAddress;
  size_t sizeInBytes;
  void *userData;
  hsa_agent_t agentOwner;
  uint32_t global_flags;
  bool registered;
} hsa_amd_pointer_info_t;

typedef struct hsa_amd_memory_access_desc_s {
  hsa_access_permission_t permissions;
  hsa_agent_t agent_handle;
} hsa_amd_memory_access_desc_t;

// Only the fields the runtime reads: the ring's base address, the doorbell to
// ring, and the ring size in packets.  This is the 64-bit layout; the runtime
// only ever builds for that.
typedef struct hsa_queue_s {
  hsa_queue_type32_t type;
  uint32_t features;
  void *base_address;
  hsa_signal_t doorbell_signal;
  uint32_t size;
  uint32_t reserved1;
  uint64_t id;
} hsa_queue_t;

typedef struct hsa_barrier_and_packet_s {
  uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;
  hsa_signal_t dep_signal[5];
  uint64_t reserved2;
  hsa_signal_t completion_signal;
} hsa_barrier_and_packet_t;

// Measured against ROCm 6's headers.  These are the only structures we hand to
// the real runtime by pointer, so a silent layout change here would corrupt
// whatever follows the field we write.
static_assert(sizeof(hsa_amd_pointer_info_t) == 56, "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, type) == 4, "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, agentBaseAddress) == 8,
              "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, sizeInBytes) == 24,
              "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, userData) == 32, "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, global_flags) == 48,
              "layout drift");
static_assert(sizeof(hsa_amd_memory_access_desc_t) == 16, "layout drift");
static_assert(sizeof(hsa_queue_t) == 40, "layout drift");
static_assert(offsetof(hsa_queue_t, base_address) == 8, "layout drift");
static_assert(offsetof(hsa_queue_t, doorbell_signal) == 16, "layout drift");
static_assert(offsetof(hsa_queue_t, size) == 24, "layout drift");
// An AQL packet slot is 64 bytes; the ring is indexed in those units.
static_assert(sizeof(hsa_barrier_and_packet_t) == 64, "layout drift");
static_assert(offsetof(hsa_barrier_and_packet_t, completion_signal) == 56,
              "layout drift");

//===----------------------------------------------------------------------===//
// Entry points.  Declared, never linked: see the file comment.
//===----------------------------------------------------------------------===//

hsa_status_t hsa_init(void);
hsa_status_t hsa_shut_down(void);
hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void *),
                                void *data);
hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                void *value);
hsa_status_t hsa_memory_copy(void *dst, const void *src, size_t size);
hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                   const char *options);
hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable,
                                               const char *symbol_name,
                                               const hsa_agent_t *agent,
                                               hsa_executable_symbol_t *symbol);
hsa_status_t
hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                               hsa_executable_symbol_info_t attribute,
                               void *value);
hsa_status_t hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent,
                             hsa_executable_symbol_t symbol, void *data),
    void *data);

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool,
                                          size_t size, uint32_t flags,
                                          void **ptr);
hsa_status_t hsa_amd_memory_pool_free(void *ptr);
hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute,
                                          void *value);
hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *),
    void *data);
hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents,
                                         const hsa_agent_t *agents,
                                         const uint32_t *flags,
                                         const void *ptr);
// The transfer entry points.  A copy or a fill issued from the host reads and
// writes device memory without any instrumented code being involved, so the
// only place to check one is here, against the extents the allocator recorded.
hsa_status_t hsa_amd_memory_async_copy(void *dst, hsa_agent_t dst_agent,
                                       const void *src, hsa_agent_t src_agent,
                                       size_t size, uint32_t num_dep_signals,
                                       const hsa_signal_t *dep_signals,
                                       hsa_signal_t completion_signal);
// `engine_id` is an enum in ROCr's header, spelled as its underlying type here
// so that this file does not have to track the engine list.
hsa_status_t hsa_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal, uint32_t engine_id,
    bool force_copy_on_sdma);
/// `count` is a number of 32-bit words, not bytes.
hsa_status_t hsa_amd_memory_fill(void *ptr, uint32_t value, size_t count);

hsa_status_t hsa_amd_pointer_info(const void *ptr, hsa_amd_pointer_info_t *info,
                                  void *(*alloc)(size_t),
                                  uint32_t *num_agents_accessible,
                                  hsa_agent_t **accessible);
hsa_status_t hsa_amd_pointer_info_set_userdata(const void *ptr, void *userdata);

hsa_status_t hsa_amd_vmem_address_reserve_align(void **va, size_t size,
                                                uint64_t address,
                                                uint64_t alignment,
                                                uint64_t flags);
hsa_status_t hsa_amd_vmem_address_free(void *va, size_t size);
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
                                        hsa_amd_memory_type_t type,
                                        uint64_t flags,
                                        hsa_amd_vmem_alloc_handle_t *handle);
hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t handle);
hsa_status_t hsa_amd_vmem_map(void *va, size_t size, size_t in_offset,
                              hsa_amd_vmem_alloc_handle_t handle,
                              uint64_t flags);
hsa_status_t hsa_amd_vmem_unmap(void *va, size_t size);
hsa_status_t hsa_amd_vmem_set_access(void *va, size_t size,
                                     const hsa_amd_memory_access_desc_t *desc,
                                     size_t desc_cnt);

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size,
                              hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t status,
                                               hsa_queue_t *source, void *data),
                              void *data, uint32_t private_segment_size,
                              uint32_t group_segment_size, hsa_queue_t **queue);
hsa_status_t hsa_queue_destroy(hsa_queue_t *queue);
uint64_t hsa_queue_add_write_index_screlease(const hsa_queue_t *queue,
                                             uint64_t value);
hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value,
                               uint32_t num_consumers,
                               const hsa_agent_t *consumers,
                               hsa_signal_t *signal);
hsa_status_t hsa_signal_destroy(hsa_signal_t signal);
void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal,
                                             hsa_signal_condition_t condition,
                                             hsa_signal_value_t compare_value,
                                             uint64_t timeout_hint,
                                             hsa_wait_state_t wait_state_hint);

//===----------------------------------------------------------------------===//
// The API tables.
//
// ROCr hands a tool the root table from OnLoad, and every one of its exported
// entry points dispatches through it, so an entry replaced here is reached by
// every client in the process.  Each sub-table begins with a version and
// continues as function pointers; `minor_id` is the size of the table, which
// is what lets a tool built against one version check whether the slot it
// wants exists in another.
//
// The sub-tables are append-only by ROCr's own ABI rule, so a slot index is
// stable.  They are given as indices rather than as a transcribed struct
// because nothing here needs the types of the entries it does not touch, and a
// short list of what we use is easier to audit than three hundred declarations
// of what we do not.
//===----------------------------------------------------------------------===//

typedef struct hsa_api_table_version_s {
  uint32_t major_id;
  uint32_t minor_id; // sizeof the table this heads
  uint32_t step_id;
  uint32_t reserved;
} hsa_api_table_version_t;

typedef struct hsa_api_table_s {
  hsa_api_table_version_t version;
  void *core;    // CoreApiTable *
  void *amd_ext; // AmdExtTable *
  void *finalizer_ext;
  void *image_ext;
  void *tools;
  void *pc_sampling_ext;
} hsa_api_table_t;

static_assert(sizeof(hsa_api_table_version_t) == 16, "layout drift");
static_assert(offsetof(hsa_api_table_t, core) == 16, "layout drift");
static_assert(offsetof(hsa_api_table_t, amd_ext) == 24, "layout drift");

// The major version of each table this was built against.  ROCr bumps these
// only for a change that moves or reinterprets existing slots, which is the one
// thing a tool cannot survive, so a mismatch has to mean disabling the tool
// rather than patching what it thinks it recognises.
enum {
  GPUASAN_HSA_API_TABLE_MAJOR_VERSION = 0x03,
  GPUASAN_HSA_CORE_API_TABLE_MAJOR_VERSION = 0x02,
  GPUASAN_HSA_AMD_EXT_TABLE_MAJOR_VERSION = 0x02,
};

/// Slot indices into CoreApiTable, counted from ROCm's hsa_api_trace.h.
enum {
  GPUASAN_CORE_INIT = 0,
  GPUASAN_CORE_SHUT_DOWN = 1,
  GPUASAN_CORE_ITERATE_AGENTS = 5,
  GPUASAN_CORE_AGENT_GET_INFO = 6,
  GPUASAN_CORE_QUEUE_CREATE = 7,
  GPUASAN_CORE_QUEUE_DESTROY = 9,
  GPUASAN_CORE_QUEUE_ADD_WRITE_INDEX_SCRELEASE = 24,
  GPUASAN_CORE_MEMORY_COPY = 35,
  GPUASAN_CORE_SIGNAL_CREATE = 37,
  GPUASAN_CORE_SIGNAL_DESTROY = 38,
  GPUASAN_CORE_SIGNAL_STORE_SCRELEASE = 42,
  GPUASAN_CORE_SIGNAL_WAIT_SCACQUIRE = 44,
  GPUASAN_CORE_EXECUTABLE_FREEZE = 86,
  GPUASAN_CORE_EXECUTABLE_SYMBOL_GET_INFO = 93,
  GPUASAN_CORE_EXECUTABLE_GET_SYMBOL_BY_NAME = 122,
  GPUASAN_CORE_EXECUTABLE_ITERATE_AGENT_SYMBOLS = 123,
};

/// Slot indices into AmdExtTable, counted the same way.
enum {
  GPUASAN_AMD_MEMORY_POOL_GET_INFO = 11,
  GPUASAN_AMD_AGENT_ITERATE_MEMORY_POOLS = 12,
  GPUASAN_AMD_MEMORY_POOL_ALLOCATE = 13,
  GPUASAN_AMD_MEMORY_POOL_FREE = 14,
  GPUASAN_AMD_MEMORY_ASYNC_COPY = 15,
  GPUASAN_AMD_MEMORY_ASYNC_COPY_ON_ENGINE = 16,
  GPUASAN_AMD_AGENTS_ALLOW_ACCESS = 19,
  GPUASAN_AMD_MEMORY_FILL = 24,
  GPUASAN_AMD_POINTER_INFO = 28,
  GPUASAN_AMD_POINTER_INFO_SET_USERDATA = 29,
  GPUASAN_AMD_VMEM_ADDRESS_FREE = 56,
  GPUASAN_AMD_VMEM_HANDLE_CREATE = 57,
  GPUASAN_AMD_VMEM_HANDLE_RELEASE = 58,
  GPUASAN_AMD_VMEM_MAP = 59,
  GPUASAN_AMD_VMEM_UNMAP = 60,
  GPUASAN_AMD_VMEM_SET_ACCESS = 61,
  GPUASAN_AMD_VMEM_ADDRESS_RESERVE_ALIGN = 69,
};

} // extern "C"

/// The entries of a sub-table, which follow its version header.
inline void **hsaTableSlots(void *Table) {
  return reinterpret_cast<void **>(reinterpret_cast<char *>(Table) +
                                   sizeof(hsa_api_table_version_t));
}

/// Whether a table is long enough to contain a slot.  A runtime older than
/// this build simply will not have the tail of the table we were compiled
/// against, and reading past its end would be a wild store into ROCr's state.
inline bool hsaTableHasSlot(void *Table, unsigned Slot) {
  auto *V = reinterpret_cast<hsa_api_table_version_t *>(Table);
  return V->minor_id >=
         sizeof(hsa_api_table_version_t) + (Slot + 1) * sizeof(void *);
}

#endif // GPUASAN_HSA_H
