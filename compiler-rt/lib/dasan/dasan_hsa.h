//===-- dasan_hsa.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Persistent HSA binding for the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_HSA_H
#define DASAN_HSA_H

#include "dasan_platform.h"
#include "hsa.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"

#define DASAN_HSA_LIBRARY "libhsa-runtime64"

namespace __dasan {

struct AliasedImage;

struct Device {
  hsa_agent_t Agent;
  DeviceId Pool;
  u32 Lanes;
  u32 Waves;
};

#define DASAN_HSA_FUNCTIONS(X)            \
  X(hsa_iterate_agents)                   \
  X(hsa_agent_get_info)                   \
  X(hsa_executable_get_symbol_by_name)    \
  X(hsa_executable_symbol_get_info)       \
  X(hsa_memory_copy)                      \
  X(hsa_amd_memory_pool_allocate)         \
  X(hsa_amd_memory_pool_free)             \
  X(hsa_amd_agents_allow_access)          \
  X(hsa_amd_memory_pool_get_info)         \
  X(hsa_amd_agent_memory_pool_get_info)   \
  X(hsa_amd_agent_iterate_memory_pools)   \
  X(hsa_amd_vmem_address_reserve_align)   \
  X(hsa_amd_vmem_address_free)            \
  X(hsa_amd_vmem_handle_create)           \
  X(hsa_amd_vmem_handle_release)          \
  X(hsa_amd_vmem_map)                     \
  X(hsa_amd_vmem_unmap)                   \
  X(hsa_amd_vmem_set_access)              \
  X(hsa_amd_portable_export_dmabuf)       \
  X(hsa_amd_portable_close_dmabuf)        \
  X(hsa_amd_vmem_import_shareable_handle) \
  X(hsa_system_get_major_extension_table) \
  X(hsa_queue_create)                     \
  X(hsa_queue_destroy)                    \
  X(hsa_queue_add_write_index_screlease)  \
  X(hsa_amd_signal_create)                \
  X(hsa_signal_destroy)                   \
  X(hsa_signal_store_screlease)           \
  X(hsa_signal_wait_scacquire)

class Hsa {
 public:
  bool Ready() const { return atomic_load(&Bound, memory_order_acquire) != 0; }
  bool Init();
  void Shutdown();

  // Caller holds DasanMutex. DropRef clears Bound on the last count so
  // interceptors pass through while Shutdown joins the report thread.
  bool AddRef();
  bool DropRef();

  InternalMmapVectorNoCtor<Device> Gpus;

  bool AllocHost(uptr Bytes, void** Out);
  void FreeHost(void* P);
  bool Copy(void* Dst, const void* Src, uptr N);
  bool SymbolAddr(hsa_executable_t Exec, const char* Name, hsa_agent_t Agent,
                  u64* Addr);

  bool InitDoorbell(hsa_signal_t* Sig, u64** Value, u64** Mailbox,
                    u32* EventId);
  void DestroySignal(hsa_signal_t Sig);
  void WaitSignalNE(hsa_signal_t Sig, u64 Cmp);
  void StoreSignal(hsa_signal_t Sig, u64 Val);

  DeviceId DeviceForPool(hsa_amd_memory_pool_t Pool) const;
  bool AgentForDevice(DeviceId Id, hsa_agent_t* Out) const;
  u32 PoolFlags(DeviceId Id) const;
  bool AllowAccess(uptr Addr, const hsa_agent_t* Agents, u32 N);
  bool AccessibleAgents(uptr Addr, void* (*Alloc)(size_t), u32* Num,
                        hsa_agent_t** Out);

  void RecordExecutable(hsa_executable_t Exec);
  void ForgetExecutable(hsa_executable_t Exec);

 private:
  friend uptr VaGranule();
  friend bool VaReserve(uptr Addr, uptr Size);
  friend void VaRelease(uptr Addr, uptr Size);
  friend bool VaCreate(DeviceId Device, uptr Size, u32 Flags, u64* Handle);
  friend void VaDestroy(u64 Handle);
  friend bool VaMap(uptr Addr, uptr Size, u64 Handle, bool ReadOnly,
                    DeviceId Device);
  friend void VaUnmap(uptr Addr, uptr Size);
  friend bool VaAllow(uptr Addr, uptr Size, DeviceId Device, bool ReadOnly);
  friend bool VaProbe(uptr Addr);
  friend bool VaWrite(uptr Dst, const void* Src, uptr N);
  friend bool VaFill(uptr Dst, uptr N);
  friend void VaPublish(uptr Touched);

  struct Api {
#define DASAN_DECLARE(Name) decltype(&::Name) Name;
    DASAN_HSA_FUNCTIONS(DASAN_DECLARE)
#undef DASAN_DECLARE
  } Api;

  struct GpuPool {
    hsa_agent_t Agent;
    hsa_amd_memory_pool_t Pool;
    u32 Flags;
    bool CpuAccess;
  };
  struct MappedRange {
    uptr Addr;
    uptr Size;
    DeviceId Device;
    bool CpuRw;
    InternalMmapVectorNoCtor<hsa_agent_t> Agents;

    bool Covers(uptr P) const { return P >= Addr && P < Addr + Size; }
    bool Overlaps(uptr Beg, uptr End) const {
      return Addr < End && Beg < Addr + Size;
    }
    void Take(MappedRange& O) {
      Addr = O.Addr;
      Size = O.Size;
      Device = O.Device;
      CpuRw = O.CpuRw;
      Agents.swap(O.Agents);
    }
  };
  struct Invalidator {
    hsa_agent_t Agent;
    hsa_queue_t* Queue;
    hsa_signal_t Signal;
    uint32_t* HdpFlush;
  };

  InternalMmapVectorNoCtor<hsa_agent_t> Agents;
  InternalMmapVectorNoCtor<hsa_agent_t> Cpus;
  InternalMmapVectorNoCtor<GpuPool> Pools;
  InternalMmapVectorNoCtor<MappedRange> Mapped;
  InternalMmapVectorNoCtor<Invalidator> Invalidators;
  InternalMmapVectorNoCtor<hsa_executable_t> Execs;

  hsa_amd_memory_pool_t HostPool;
  uptr Granule;
  bool CpuDirect;
  uptr Refs;
  atomic_uint8_t Bound;

  hsa_ven_amd_loader_1_03_pfn_t Loader;
  bool LoaderReady;

  bool Resolve();
  bool Discover();
  void NotePool(hsa_device_type_t Type, hsa_agent_t Agent,
                hsa_amd_memory_pool_t Pool);
  void ProbeCpuAccess();
  bool HostCanAccess(DeviceId Id) const;
  bool CpuCanAccess(hsa_agent_t Cpu, DeviceId Id) const;
  bool CpuMapped(uptr Addr) const;
  void NoteMappedAgent(MappedRange& M, hsa_agent_t Agent);
  void DropMapped(uptr I);

  bool Reserve(uptr Addr, uptr Size);
  void Release(uptr Addr, uptr Size);
  bool Create(DeviceId Id, uptr Size, u32 Flags, u64* Handle);
  void Destroy(u64 Handle);
  bool Grant(uptr Addr, uptr Size, bool ReadOnly, DeviceId Id, bool WithCpu);
  bool Allow(uptr Addr, uptr Size, DeviceId Id, bool ReadOnly);
  bool AllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N,
                   bool ReadOnly);
  bool Map(uptr Addr, uptr Size, u64 Handle, bool ReadOnly, DeviceId Id,
           uptr Off = 0);
  void Unmap(uptr Addr, uptr Size);
  bool Probe(uptr Addr);
  bool Write(uptr Dst, const void* Src, uptr N);
  bool Fill(uptr Dst, uptr N);
  void Publish(uptr Touched);
  bool Export(uptr Addr, uptr Size, u64* Handle, uptr* Off);

  bool EnsureInvalidator(hsa_agent_t Agent, Invalidator** Out);
  void FlushHdp();
  void PostInvalidate(hsa_agent_t Agent);
  void AwaitInvalidate(hsa_agent_t Agent);
  void DestroyInvalidators();

  bool EnsureLoader();
  bool CodeObjectInfo(hsa_loaded_code_object_t Obj,
                      hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                      u64* Out);
  void AliasExecutable(hsa_executable_t Exec);
  bool StillLoaded(uptr LoadBase);

  DeviceId DeviceForAgent(hsa_agent_t Agent) const;
  const GpuPool* PoolFor(DeviceId Id) const;
  void DropMapping(AliasedImage& Img);
  bool IsGpu(hsa_agent_t Agent) const;
  void FillGpus();

  template <typename Cb>
  void ForEachAgentObject(hsa_executable_t Exec, Cb F);
};

Hsa& GetHsa();

}  // namespace __dasan

#endif  // DASAN_HSA_H
