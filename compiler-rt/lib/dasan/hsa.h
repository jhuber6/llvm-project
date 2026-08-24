//===-- dasan_hsa.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal HSA declarations without the dependency.
//
//===----------------------------------------------------------------------===//

#ifndef HSA_H
#define HSA_H

#include <stddef.h>
#include <stdint.h>

extern "C" {

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
typedef struct hsa_region_s {
  uint64_t handle;
} hsa_region_t;
typedef struct hsa_amd_vmem_alloc_handle_s {
  uint64_t handle;
} hsa_amd_vmem_alloc_handle_t;
typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;
typedef struct hsa_loaded_code_object_s {
  uint64_t handle;
} hsa_loaded_code_object_t;

typedef int64_t hsa_signal_value_t;
typedef uint32_t hsa_queue_type32_t;

typedef enum {
  HSA_STATUS_SUCCESS = 0,
  HSA_STATUS_ERROR = 0x1001,
} hsa_status_t;

typedef enum {
  HSA_DEVICE_TYPE_CPU = 0,
  HSA_DEVICE_TYPE_GPU = 1,
} hsa_device_type_t;

typedef enum {
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  HSA_AGENT_INFO_DEVICE = 17,
} hsa_agent_info_t;

typedef enum { HSA_AMD_SEGMENT_GLOBAL = 0 } hsa_amd_segment_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18,
} hsa_amd_memory_pool_info_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8,
} hsa_amd_memory_pool_global_flag_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0,
  HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = 1,
  HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = 2,
} hsa_amd_memory_pool_access_t;

typedef enum {
  HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0,
} hsa_amd_agent_memory_pool_info_t;

typedef enum { HSA_EXT_POINTER_TYPE_HSA = 1 } hsa_amd_pointer_type_t;

typedef enum {
  HSA_ACCESS_PERMISSION_RO = 1,
  HSA_ACCESS_PERMISSION_RW = 3,
} hsa_access_permission_t;

typedef enum { MEMORY_TYPE_NONE = 0 } hsa_amd_memory_type_t;

typedef enum { HSA_QUEUE_TYPE_SINGLE = 1 } hsa_queue_type_t;

typedef enum {
  HSA_SIGNAL_CONDITION_EQ = 0,
  HSA_SIGNAL_CONDITION_NE = 1,
} hsa_signal_condition_t;

typedef enum {
  HSA_WAIT_STATE_BLOCKED = 0,
  HSA_WAIT_STATE_ACTIVE = 1,
} hsa_wait_state_t;

typedef enum {
  HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 0xA002,
  HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 0xA00A,
  HSA_AMD_AGENT_INFO_HDP_FLUSH = 0xA00E,
} hsa_amd_agent_info_t;

typedef struct hsa_amd_hdp_flush_s {
  uint32_t* HDP_MEM_FLUSH_CNTL;
  uint32_t* HDP_REG_FLUSH_CNTL;
} hsa_amd_hdp_flush_t;

typedef enum {
  HSA_PACKET_HEADER_TYPE = 0,
  HSA_PACKET_HEADER_BARRIER = 8,
  HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
  HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11,
} hsa_packet_header_t;

typedef enum { HSA_PACKET_TYPE_BARRIER_AND = 3 } hsa_packet_type_t;

typedef enum {
  HSA_FENCE_SCOPE_NONE = 0,
  HSA_FENCE_SCOPE_SYSTEM = 2,
} hsa_fence_scope_t;

typedef enum {
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22,
} hsa_executable_symbol_info_t;

typedef enum { HSA_EXTENSION_AMD_LOADER = 0x201 } hsa_extension_t;

typedef enum {
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND = 2,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT = 3,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE = 4,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE =
      5,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE =
      6,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE = 9,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE = 10,
} hsa_ven_amd_loader_loaded_code_object_info_t;

typedef enum {
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_PROGRAM = 1,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT = 2,
} hsa_ven_amd_loader_loaded_code_object_kind_t;

typedef enum {
  HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY = 2,
} hsa_ven_amd_loader_code_object_storage_type_t;

typedef struct hsa_amd_pointer_info_s {
  uint32_t size;
  hsa_amd_pointer_type_t type;
  void* agentBaseAddress;
  void* hostBaseAddress;
  size_t sizeInBytes;
  void* userData;
  hsa_agent_t agentOwner;
  uint32_t global_flags;
  bool registered;
} hsa_amd_pointer_info_t;

typedef struct hsa_amd_memory_access_desc_s {
  hsa_access_permission_t permissions;
  hsa_agent_t agent_handle;
} hsa_amd_memory_access_desc_t;

typedef struct hsa_queue_s {
  hsa_queue_type32_t type;
  uint32_t features;
  void* base_address;
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

typedef struct hsa_ven_amd_loader_1_03_pfn_s {
  hsa_status_t (*hsa_ven_amd_loader_query_host_address)(const void*,
                                                        const void**);
  hsa_status_t (*hsa_ven_amd_loader_query_segment_descriptors)(void*, size_t*);
  hsa_status_t (*hsa_ven_amd_loader_query_executable)(const void*,
                                                      hsa_executable_t*);
  hsa_status_t (*hsa_ven_amd_loader_executable_iterate_loaded_code_objects)(
      hsa_executable_t,
      hsa_status_t (*)(hsa_executable_t, hsa_loaded_code_object_t, void*),
      void*);
  hsa_status_t (*hsa_ven_amd_loader_loaded_code_object_get_info)(
      hsa_loaded_code_object_t, hsa_ven_amd_loader_loaded_code_object_info_t,
      void*);
  hsa_status_t (*unused_create_from_file_with_offset_size)(int, size_t, size_t,
                                                           void*);
  hsa_status_t (*unused_iterate_executables)(hsa_status_t (*)(hsa_executable_t,
                                                              void*),
                                             void*);
} hsa_ven_amd_loader_1_03_pfn_t;

static_assert(sizeof(hsa_ven_amd_loader_1_03_pfn_t) == 7 * sizeof(void*),
              "layout drift");

static_assert(sizeof(hsa_amd_pointer_info_t) == 56, "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, agentBaseAddress) == 8,
              "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, sizeInBytes) == 24,
              "layout drift");
static_assert(offsetof(hsa_amd_pointer_info_t, global_flags) == 48,
              "layout drift");
static_assert(sizeof(hsa_amd_memory_access_desc_t) == 16, "layout drift");
static_assert(sizeof(hsa_queue_t) == 40, "layout drift");
static_assert(offsetof(hsa_queue_t, doorbell_signal) == 16, "layout drift");
static_assert(sizeof(hsa_barrier_and_packet_t) == 64, "layout drift");
static_assert(offsetof(hsa_barrier_and_packet_t, completion_signal) == 56,
              "layout drift");

hsa_status_t hsa_init(void);
hsa_status_t hsa_shut_down(void);
hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void*),
                                void* data);
hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                void* value);
hsa_status_t hsa_executable_destroy(hsa_executable_t executable);
hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                   const char* options);
hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable,
                                               const char* symbol_name,
                                               const hsa_agent_t* agent,
                                               hsa_executable_symbol_t* symbol);
hsa_status_t hsa_executable_symbol_get_info(
    hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute,
    void* value);

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool,
                                          size_t size, uint32_t flags,
                                          void** ptr);
hsa_status_t hsa_amd_memory_pool_free(void* ptr);
hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute,
                                          void* value);
hsa_status_t hsa_amd_agent_memory_pool_get_info(
    hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool,
    hsa_amd_agent_memory_pool_info_t attribute, void* value);
hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void*),
    void* data);
hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents,
                                         const hsa_agent_t* agents,
                                         const uint32_t* flags,
                                         const void* ptr);
hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info,
                                  void* (*alloc)(size_t),
                                  uint32_t* num_agents_accessible,
                                  hsa_agent_t** accessible);

hsa_status_t hsa_amd_vmem_address_reserve_align(void** va, size_t size,
                                                uint64_t address,
                                                uint64_t alignment,
                                                uint64_t flags);
hsa_status_t hsa_amd_vmem_address_free(void* va, size_t size);
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
                                        hsa_amd_memory_type_t type,
                                        uint64_t flags,
                                        hsa_amd_vmem_alloc_handle_t* handle);
hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t handle);
hsa_status_t hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                              hsa_amd_vmem_alloc_handle_t handle,
                              uint64_t flags);
hsa_status_t hsa_amd_vmem_unmap(void* va, size_t size);

hsa_status_t hsa_amd_portable_export_dmabuf(const void* ptr, size_t size,
                                            int* dmabuf, uint64_t* offset);
hsa_status_t hsa_amd_portable_close_dmabuf(int dmabuf);
hsa_status_t hsa_amd_vmem_import_shareable_handle(
    int dmabuf_fd, hsa_amd_vmem_alloc_handle_t* handle);

hsa_status_t hsa_system_get_major_extension_table(uint16_t extension,
                                                  uint16_t version_major,
                                                  size_t table_length,
                                                  void* table);
hsa_status_t hsa_amd_vmem_set_access(void* va, size_t size,
                                     const hsa_amd_memory_access_desc_t* desc,
                                     size_t desc_cnt);

hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size);
hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr);
hsa_status_t hsa_memory_free(void* ptr);

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size,
                              hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t status,
                                               hsa_queue_t* source, void* data),
                              void* data, uint32_t private_segment_size,
                              uint32_t group_segment_size, hsa_queue_t** queue);
hsa_status_t hsa_queue_destroy(hsa_queue_t* queue);
uint64_t hsa_queue_add_write_index_screlease(const hsa_queue_t* queue,
                                             uint64_t value);
hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value,
                               uint32_t num_consumers,
                               const hsa_agent_t* consumers,
                               hsa_signal_t* signal);
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value,
                                   uint32_t num_consumers,
                                   const hsa_agent_t* consumers,
                                   uint64_t attributes, hsa_signal_t* signal);
hsa_status_t hsa_signal_destroy(hsa_signal_t signal);
void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal,
                                             hsa_signal_condition_t condition,
                                             hsa_signal_value_t compare_value,
                                             uint64_t timeout_hint,
                                             hsa_wait_state_t wait_state_hint);

}  // extern "C"

#endif  // HSA_H
