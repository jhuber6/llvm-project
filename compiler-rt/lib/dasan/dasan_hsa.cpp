//===-- dasan_hsa.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HSA binding. Topology, vmem, and code-object aliases live on Hsa.
//
//===----------------------------------------------------------------------===//

#include "dasan_hsa.h"

#include <dlfcn.h>

#include "dasan.h"
#include "dasan_allocator.h"
#include "dasan_flags.h"
#include "dasan_image.h"
#include "dasan_platform.h"
#include "dasan_rpc.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

using namespace __sanitizer;

namespace __dasan {
namespace {

// HSA iterate adapters, same shape as offload's hsa_utils::iterate.
template <typename ElemTy, typename IterFuncTy, typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void* Data) -> hsa_status_t {
    return (*static_cast<CallbackTy*>(Data))(Elem);
  };
  return Func(L, &Cb);
}

template <typename ElemTy, typename IterFuncTy, typename ArgTy,
          typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, ArgTy Arg, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void* Data) -> hsa_status_t {
    return (*static_cast<CallbackTy*>(Data))(Elem);
  };
  return Func(Arg, L, &Cb);
}

template <typename Elem1Ty, typename Elem2Ty, typename IterFuncTy,
          typename ArgTy, typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, ArgTy Arg, CallbackTy Cb) {
  auto L = [](Elem1Ty A, Elem2Ty B, void* Data) -> hsa_status_t {
    return (*static_cast<CallbackTy*>(Data))(A, B);
  };
  return Func(Arg, L, &Cb);
}

constexpr uptr kNpos = ~(uptr)0;

template <typename Vec, typename Pred>
uptr FindIndex(const Vec& V, Pred P) {
  for (uptr I = 0; I < V.size(); ++I)
    if (P(V[I]))
      return I;
  return kNpos;
}

template <typename T, typename Pred>
T* FindIf(InternalMmapVectorNoCtor<T>& V, Pred P) {
  const uptr I = FindIndex(V, P);
  return I == kNpos ? nullptr : &V[I];
}

template <typename T, typename Pred>
const T* FindIf(const InternalMmapVectorNoCtor<T>& V, Pred P) {
  const uptr I = FindIndex(V, P);
  return I == kNpos ? nullptr : &V[I];
}

template <typename Vec, typename Pred>
bool Contains(const Vec& V, Pred P) {
  return FindIndex(V, P) != kNpos;
}

template <typename T>
bool HasHandle(const InternalMmapVectorNoCtor<T>& V, u64 Handle) {
  return Contains(V, [&](const T& E) { return E.handle == Handle; });
}

template <typename T>
void NoteHandle(InternalMmapVectorNoCtor<T>& V, T X) {
  if (!X.handle || HasHandle(V, X.handle))
    return;
  V.push_back(X);
}

template <typename Vec>
void EraseSwap(Vec& V, uptr I) {
  if (I + 1 != V.size())
    V[I] = V.back();
  V.pop_back();
}

constexpr auto kKind = HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND;
constexpr auto kAgent = HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT;
constexpr auto kLoadBase = HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE;
constexpr auto kLoadSize = HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE;
constexpr auto kStorageType =
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE;
constexpr auto kStorageBase =
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE;
constexpr auto kStorageSize =
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE;

const u64 kInvalidateTimeoutNs = 1000ULL * 1000 * 1000;

}  // namespace

template <typename Cb>
void Hsa::ForEachAgentObject(hsa_executable_t Exec, Cb F) {
  if (!EnsureLoader())
    return;
  Iterate<hsa_executable_t, hsa_loaded_code_object_t>(
      Loader.hsa_ven_amd_loader_executable_iterate_loaded_code_objects, Exec,
      [&](hsa_executable_t, hsa_loaded_code_object_t Obj) {
        u32 Kind = 0;
        if (Loader.hsa_ven_amd_loader_loaded_code_object_get_info(
                Obj, kKind, &Kind) != HSA_STATUS_SUCCESS ||
            Kind != HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT)
          return HSA_STATUS_SUCCESS;
        hsa_agent_t Agent{};
        if (Loader.hsa_ven_amd_loader_loaded_code_object_get_info(
                Obj, kAgent, &Agent) != HSA_STATUS_SUCCESS ||
            !Agent.handle)
          return HSA_STATUS_SUCCESS;
        F(Obj, Agent);
        return HSA_STATUS_SUCCESS;
      });
}

bool Hsa::Resolve() {
#define DASAN_RESOLVE(Name)                                                \
  Api.Name = reinterpret_cast<decltype(&::Name)>(dlsym(RTLD_NEXT, #Name)); \
  if (!Api.Name)                                                           \
    return false;
  DASAN_HSA_FUNCTIONS(DASAN_RESOLVE)
#undef DASAN_RESOLVE
  return true;
}

void Hsa::NotePool(hsa_device_type_t Type, hsa_agent_t Agent,
                   hsa_amd_memory_pool_t Pool) {
  hsa_amd_segment_t Segment;
  bool Allowed = false;
  u32 Flags = 0;
  if (Api.hsa_amd_memory_pool_get_info(Pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                       &Segment) != HSA_STATUS_SUCCESS ||
      Segment != HSA_AMD_SEGMENT_GLOBAL ||
      Api.hsa_amd_memory_pool_get_info(
          Pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &Allowed) !=
          HSA_STATUS_SUCCESS ||
      !Allowed ||
      Api.hsa_amd_memory_pool_get_info(Pool,
                                       HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                       &Flags) != HSA_STATUS_SUCCESS)
    return;

  if (Type == HSA_DEVICE_TYPE_CPU) {
    if ((Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        !HostPool.handle)
      HostPool = Pool;
    return;
  }

  // GPU coarse, fine, or extended-scope fine. Fine-grain vmem into a reserved
  // VA is new; Create/Map will fail on older runtimes and the interceptor
  // passes the allocate through. Host pools stay unplaced: a code object
  // allocated there must keep its loader address or the kernel sees one fat
  // heap object instead of per-global entries.
  const u32 Placeable =
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED |
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED;
  if (!(Flags & Placeable))
    return;

  uptr G = 0;
  if (Api.hsa_amd_memory_pool_get_info(
          Pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &G) ==
          HSA_STATUS_SUCCESS &&
      G > Granule)
    Granule = G;
  if (Api.hsa_amd_memory_pool_get_info(
          Pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &G) ==
          HSA_STATUS_SUCCESS &&
      G > Granule)
    Granule = G;
  Pools.push_back(GpuPool{Agent, Pool, Flags, false});
}

bool Hsa::Discover() {
  Agents.clear();
  Gpus.clear();
  Cpus.clear();
  Pools.clear();
  Granule = 0;
  HostPool = {};
  CpuDirect = false;

  Iterate<hsa_agent_t>(Api.hsa_iterate_agents, [&](hsa_agent_t Agent) {
    hsa_device_type_t Type;
    if (Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;
    Agents.push_back(Agent);
    if (Type == HSA_DEVICE_TYPE_GPU) {
      Device G = {};
      G.Agent = Agent;
      Gpus.push_back(G);
    } else if (Type == HSA_DEVICE_TYPE_CPU) {
      Cpus.push_back(Agent);
    }
    Iterate<hsa_amd_memory_pool_t>(Api.hsa_amd_agent_iterate_memory_pools,
                                   Agent, [&](hsa_amd_memory_pool_t Pool) {
                                     NotePool(Type, Agent, Pool);
                                     return HSA_STATUS_SUCCESS;
                                   });
    return HSA_STATUS_SUCCESS;
  });

  ProbeCpuAccess();
  FillGpus();
  return !Gpus.empty() && !Pools.empty() && Granule != 0;
}

void Hsa::FillGpus() {
  for (Device& G : Gpus) {
    G.Pool = DeviceForAgent(G.Agent);
    u32 Lanes = 0;
    if (Api.hsa_agent_get_info(G.Agent, HSA_AGENT_INFO_WAVEFRONT_SIZE,
                               &Lanes) != HSA_STATUS_SUCCESS ||
        !Lanes)
      Lanes = 64;
    G.Lanes = Lanes;
    u32 CUs = 0, Waves = 0;
    if (Api.hsa_agent_get_info(G.Agent,
                               static_cast<hsa_agent_info_t>(
                                   HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                               &CUs) != HSA_STATUS_SUCCESS ||
        Api.hsa_agent_get_info(
            G.Agent,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
            &Waves) != HSA_STATUS_SUCCESS)
      G.Waves = 0;
    else
      G.Waves = CUs * Waves;
  }
}

bool Hsa::IsGpu(hsa_agent_t Agent) const {
  return Contains(
      Gpus, [&](const Device& G) { return G.Agent.handle == Agent.handle; });
}

void Hsa::ProbeCpuAccess() {
  CpuDirect = !Pools.empty();
  for (GpuPool& P : Pools) {
    P.CpuAccess = false;
    for (hsa_agent_t Cpu : Cpus) {
      if (CpuCanAccess(Cpu, P.Pool.handle)) {
        P.CpuAccess = true;
        break;
      }
    }
    CpuDirect = CpuDirect && P.CpuAccess;
  }
}

bool Hsa::HostCanAccess(DeviceId Device) const {
  if (!Device)
    return HostPool.handle != 0;
  const GpuPool* P = PoolFor(Device);
  return P && P->CpuAccess;
}

bool Hsa::CpuCanAccess(hsa_agent_t Cpu, DeviceId Device) const {
  if (!Device)
    return true;
  hsa_amd_memory_pool_t Pool{Device};
  hsa_amd_memory_pool_access_t Access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  return Api.hsa_amd_agent_memory_pool_get_info(
             Cpu, Pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &Access) ==
             HSA_STATUS_SUCCESS &&
         Access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
}

void Hsa::NoteMappedAgent(MappedRange& M, hsa_agent_t Agent) {
  NoteHandle(M.Agents, Agent);
}

void Hsa::DropMapped(uptr I) {
  Mapped[I].Agents.Destroy();
  if (I + 1 != Mapped.size())
    Mapped[I].Take(Mapped.back());
  Mapped.pop_back();
}

bool Hsa::CpuMapped(uptr Addr) const {
  const MappedRange* M =
      FindIf(Mapped, [&](const MappedRange& R) { return R.Covers(Addr); });
  return M && M->CpuRw;
}

bool Hsa::Reserve(uptr Addr, uptr Size) {
  void* Va = nullptr;
  if (Api.hsa_amd_vmem_address_reserve_align(&Va, Size, Addr, Addr & -Addr,
                                             0) != HSA_STATUS_SUCCESS)
    return false;
  if (reinterpret_cast<uptr>(Va) != Addr) {
    Api.hsa_amd_vmem_address_free(Va, Size);
    return false;
  }
  return true;
}

void Hsa::Release(uptr Addr, uptr Size) {
  Api.hsa_amd_vmem_address_free(reinterpret_cast<void*>(Addr), Size);
}

bool Hsa::Create(DeviceId Device, uptr Size, u32 Flags, u64* Handle) {
  hsa_amd_memory_pool_t Pool =
      Device ? hsa_amd_memory_pool_t{Device} : HostPool;
  if (!Pool.handle)
    return false;
  hsa_amd_vmem_alloc_handle_t H;
  if (Api.hsa_amd_vmem_handle_create(Pool, Size, MEMORY_TYPE_NONE, Flags, &H) !=
      HSA_STATUS_SUCCESS)
    return false;
  *Handle = H.handle;
  return true;
}

void Hsa::Destroy(u64 Handle) {
  Api.hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t{Handle});
}

bool Hsa::Grant(uptr Addr, uptr Size, bool ReadOnly, DeviceId Device,
                bool WithCpu) {
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  hsa_agent_t Owner = {};
  const bool HaveOwner = Device && AgentForDevice(Device, &Owner);
  if (WithCpu) {
    for (hsa_agent_t Cpu : Cpus) {
      if (HaveOwner && Cpu.handle == Owner.handle)
        continue;
      if (!CpuCanAccess(Cpu, Device))
        continue;
      hsa_amd_memory_access_desc_t D;
      D.permissions = HSA_ACCESS_PERMISSION_RW;
      D.agent_handle = Cpu;
      Descs.push_back(D);
    }
  }
  if (HaveOwner) {
    hsa_amd_memory_access_desc_t D;
    D.permissions = (ReadOnly && WithCpu) ? HSA_ACCESS_PERMISSION_RO
                                          : HSA_ACCESS_PERMISSION_RW;
    D.agent_handle = Owner;
    Descs.push_back(D);
  }
  if (Descs.empty())
    return false;
  void* Va = reinterpret_cast<void*>(Addr);
  if (Api.hsa_amd_vmem_set_access(Va, Size, Descs.data(), Descs.size()) ==
      HSA_STATUS_SUCCESS)
    return true;
  if (!ReadOnly || !Device || !WithCpu)
    return false;
  Descs.back().permissions = HSA_ACCESS_PERMISSION_RW;
  return Api.hsa_amd_vmem_set_access(Va, Size, Descs.data(), Descs.size()) ==
         HSA_STATUS_SUCCESS;
}

bool Hsa::AllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N,
                      bool ReadOnly) {
  if (!N)
    return true;
  const uptr End = Addr + Size;
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  for (u32 I = 0; I < N; ++I) {
    bool Ro = ReadOnly && IsGpu(Agents[I]);
    if (Ro) {
      for (const MappedRange& R : Mapped) {
        if (!R.Overlaps(Addr, End))
          continue;
        hsa_agent_t Owner = {};
        if (R.Device && AgentForDevice(R.Device, &Owner) &&
            Owner.handle == Agents[I].handle && !R.CpuRw)
          Ro = false;
      }
    }
    hsa_amd_memory_access_desc_t D;
    D.permissions = Ro ? HSA_ACCESS_PERMISSION_RO : HSA_ACCESS_PERMISSION_RW;
    D.agent_handle = Agents[I];
    Descs.push_back(D);
  }
  bool Hit = false;
  for (MappedRange& M : Mapped) {
    if (!M.Overlaps(Addr, End))
      continue;
    Hit = true;
    if (Api.hsa_amd_vmem_set_access(reinterpret_cast<void*>(M.Addr), M.Size,
                                    Descs.data(),
                                    Descs.size()) != HSA_STATUS_SUCCESS)
      return false;
    for (u32 A = 0; A < N; ++A) NoteMappedAgent(M, Agents[A]);
  }
  return Hit;
}

bool Hsa::Allow(uptr Addr, uptr Size, DeviceId Device, bool ReadOnly) {
  hsa_agent_t Agent = {};
  if (!AgentForDevice(Device, &Agent))
    return false;
  return AllowAgents(Addr, Size, &Agent, 1, ReadOnly);
}

bool Hsa::Map(uptr Addr, uptr Size, u64 Handle, bool ReadOnly, DeviceId Device,
              uptr Off) {
  hsa_amd_vmem_alloc_handle_t H{Handle};
  if (Api.hsa_amd_vmem_map(reinterpret_cast<void*>(Addr), Size, Off, H, 0) !=
      HSA_STATUS_SUCCESS)
    return false;
  const bool WithCpu = HostCanAccess(Device);
  if (!Grant(Addr, Size, ReadOnly, Device, WithCpu)) {
    Api.hsa_amd_vmem_unmap(reinterpret_cast<void*>(Addr), Size);
    return false;
  }
  Mapped.resize(Mapped.size() + 1);
  MappedRange& R = Mapped.back();
  R.Addr = Addr;
  R.Size = Size;
  R.Device = Device;
  R.CpuRw = WithCpu;
  hsa_agent_t Gpu = {};
  if (Device && AgentForDevice(Device, &Gpu))
    NoteMappedAgent(R, Gpu);
  if (WithCpu) {
    for (hsa_agent_t Cpu : Cpus)
      if (CpuCanAccess(Cpu, Device))
        NoteMappedAgent(R, Cpu);
  }
  return true;
}

void Hsa::Unmap(uptr Addr, uptr Size) {
  Api.hsa_amd_vmem_unmap(reinterpret_cast<void*>(Addr), Size);
  const uptr I = FindIndex(Mapped, [&](const MappedRange& M) {
    return M.Addr == Addr && M.Size == Size;
  });
  if (I != kNpos)
    DropMapped(I);
}

bool Hsa::Export(uptr Addr, uptr Size, u64* Handle, uptr* Off) {
  int Fd = -1;
  uint64_t Offset = 0;
  if (Api.hsa_amd_portable_export_dmabuf(reinterpret_cast<void*>(Addr), Size,
                                         &Fd, &Offset) != HSA_STATUS_SUCCESS) {
    VReport(1, "%s: could not export %zu bytes at 0x%zx for aliasing\n",
            SanitizerToolName, Size, Addr);
    return false;
  }
  hsa_amd_vmem_alloc_handle_t H;
  hsa_status_t St = Api.hsa_amd_vmem_import_shareable_handle(Fd, &H);
  Api.hsa_amd_portable_close_dmabuf(Fd);
  if (St != HSA_STATUS_SUCCESS) {
    VReport(1, "%s: could not import a share of 0x%zx\n", SanitizerToolName,
            Addr);
    return false;
  }
  *Handle = H.handle;
  *Off = Offset;
  return true;
}

bool Hsa::Probe(uptr Addr) {
  const u64 Magic = 0xda5a4e0000da5a4eULL;
  u64 Got = 0;
  void* Dst = reinterpret_cast<void*>(Addr);
  if (CpuMapped(Addr)) {
    *reinterpret_cast<volatile u64*>(Addr) = Magic;
    Got = *reinterpret_cast<volatile u64*>(Addr);
    return Got == Magic;
  }
  if (Api.hsa_memory_copy(Dst, &Magic, sizeof(Magic)) != HSA_STATUS_SUCCESS ||
      Api.hsa_memory_copy(&Got, Dst, sizeof(Got)) != HSA_STATUS_SUCCESS)
    return false;
  return Got == Magic;
}

bool Hsa::Write(uptr Dst, const void* Src, uptr N) {
  if (CpuMapped(Dst)) {
    internal_memcpy(reinterpret_cast<void*>(Dst), Src, N);
    return true;
  }
  return Api.hsa_memory_copy(reinterpret_cast<void*>(Dst), Src, N) ==
         HSA_STATUS_SUCCESS;
}

bool Hsa::Fill(uptr Dst, uptr N) {
  if (CpuMapped(Dst)) {
    internal_memset(reinterpret_cast<void*>(Dst), 0, N);
    return true;
  }
  char Z[4096];
  internal_memset(Z, 0, sizeof(Z));
  while (N) {
    uptr Step = N < sizeof(Z) ? N : sizeof(Z);
    if (Api.hsa_memory_copy(reinterpret_cast<void*>(Dst), Z, Step) !=
        HSA_STATUS_SUCCESS)
      return false;
    Dst += Step;
    N -= Step;
  }
  return true;
}

void Hsa::Publish(uptr Touched) {
  // Direct stores need the BAR drain. Copies are already in VRAM; the
  // barrier still drops a stale GL2 line so the next kernel's invariant
  // load cannot keep the old word.
  if (Touched && CpuMapped(Touched)) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    (void)*reinterpret_cast<volatile u64*>(Touched);
    FlushHdp();
  }
  for (const Device& G : Gpus) PostInvalidate(G.Agent);
  for (const Device& G : Gpus) AwaitInvalidate(G.Agent);
}

bool Hsa::EnsureInvalidator(hsa_agent_t Agent, Invalidator** Out) {
  if (Invalidator* Inv = FindIf(Invalidators, [&](const Invalidator& I) {
        return I.Agent.handle == Agent.handle;
      })) {
    *Out = Inv;
    return Inv->Queue != nullptr;
  }

  Invalidators.push_back(Invalidator{Agent, nullptr, {0}, nullptr});
  Invalidator& Slot = Invalidators.back();
  *Out = &Slot;

  hsa_amd_hdp_flush_t Hdp = {};
  if (Api.hsa_agent_get_info(
          Agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH),
          &Hdp) == HSA_STATUS_SUCCESS)
    Slot.HdpFlush = Hdp.HDP_MEM_FLUSH_CNTL;

  u32 MaxSize = 0;
  if (Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &MaxSize) !=
      HSA_STATUS_SUCCESS)
    return false;
  u32 Size = MaxSize < 64 ? MaxSize : 64;
  if (Api.hsa_queue_create(Agent, Size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                           0, 0, &Slot.Queue) != HSA_STATUS_SUCCESS) {
    Slot.Queue = nullptr;
    return false;
  }
  if (Api.hsa_amd_signal_create(0, 0, nullptr, 0, &Slot.Signal) !=
      HSA_STATUS_SUCCESS) {
    Api.hsa_queue_destroy(Slot.Queue);
    Slot.Queue = nullptr;
    return false;
  }
  return true;
}

void Hsa::FlushHdp() {
  for (const Device& G : Gpus) {
    Invalidator* Inv;
    if (!EnsureInvalidator(G.Agent, &Inv) || !Inv->HdpFlush)
      continue;
    __atomic_store_n(Inv->HdpFlush, 1u, __ATOMIC_RELEASE);
  }
}

void Hsa::PostInvalidate(hsa_agent_t Agent) {
  Invalidator* Inv;
  if (!EnsureInvalidator(Agent, &Inv))
    return;

  const u16 Header =
      (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
      (1u << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  Api.hsa_signal_store_screlease(Inv->Signal, 1);
  u64 Index = Api.hsa_queue_add_write_index_screlease(Inv->Queue, 1);
  auto* Ring =
      reinterpret_cast<hsa_barrier_and_packet_t*>(Inv->Queue->base_address);
  hsa_barrier_and_packet_t& Packet = Ring[Index % Inv->Queue->size];

  Packet.reserved0 = 0;
  Packet.reserved1 = 0;
  Packet.reserved2 = 0;
  for (auto& Dep : Packet.dep_signal) Dep.handle = 0;
  Packet.completion_signal = Inv->Signal;
  __atomic_store_n(&Packet.header, Header, __ATOMIC_RELEASE);
  Api.hsa_signal_store_screlease(Inv->Queue->doorbell_signal, Index);
}

void Hsa::AwaitInvalidate(hsa_agent_t Agent) {
  Invalidator* Inv = FindIf(Invalidators, [&](const Invalidator& I) {
    return I.Agent.handle == Agent.handle && I.Queue;
  });
  if (!Inv)
    return;
  if (Api.hsa_signal_wait_scacquire(Inv->Signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                    kInvalidateTimeoutNs,
                                    HSA_WAIT_STATE_BLOCKED) != 0)
    VReport(1,
            "%s: cache maintenance did not complete; a newly published "
            "allocation may look unallocated\n",
            SanitizerToolName);
}

void Hsa::DestroyInvalidators() {
  for (Invalidator& Inv : Invalidators) {
    if (!Inv.Queue)
      continue;
    Api.hsa_signal_destroy(Inv.Signal);
    Api.hsa_queue_destroy(Inv.Queue);
  }
  Invalidators.clear();
}

bool Hsa::EnsureLoader() {
  if (LoaderReady)
    return Loader.hsa_ven_amd_loader_executable_iterate_loaded_code_objects !=
           nullptr;
  LoaderReady = true;
  bool Ok = Api.hsa_system_get_major_extension_table(
                HSA_EXTENSION_AMD_LOADER, 1, sizeof(Loader), &Loader) ==
                HSA_STATUS_SUCCESS &&
            Loader.hsa_ven_amd_loader_executable_iterate_loaded_code_objects &&
            Loader.hsa_ven_amd_loader_loaded_code_object_get_info;
  if (!Ok) {
    Loader = {};
    VReport(1, "%s: no loader extension; globals stay unchecked\n",
            SanitizerToolName);
  }
  return Ok;
}

bool Hsa::CodeObjectInfo(hsa_loaded_code_object_t Obj,
                         hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                         u64* Out) {
  *Out = 0;
  return Loader.hsa_ven_amd_loader_loaded_code_object_get_info(
             Obj, Attr, Out) == HSA_STATUS_SUCCESS;
}

void Hsa::DropMapping(AliasedImage& Img) {
  if (!Img.Alias)
    return;
  Unmap(Img.Alias, Img.LoadSize);
  Destroy(Img.Handle);
  GetAllocator().Deallocate(Img.Alias, Img.LoadSize, 0);
}

void Hsa::AliasExecutable(hsa_executable_t Exec) {
  if (!flags()->alias_globals || Pools.empty() || !EnsureLoader())
    return;
  ForEachAgentObject(
      Exec, [&](hsa_loaded_code_object_t Obj, hsa_agent_t Agent) {
        DeviceId Device = DeviceForAgent(Agent);
        if (!Device)
          return;

        u64 LoadBase = 0, LoadSize = 0;
        if (!CodeObjectInfo(Obj, kLoadBase, &LoadBase) ||
            !CodeObjectInfo(Obj, kLoadSize, &LoadSize) || !LoadBase ||
            !LoadSize || AlreadyAliased(LoadBase))
          return;

        uptr Alias = GetAllocator().Allocate(Device, (uptr)LoadSize, uptr(0));
        if (!Alias) {
          VReport(1, "%s: no room in class zero for a %zu byte code object\n",
                  SanitizerToolName, (uptr)LoadSize);
          return;
        }

        u64 H = 0;
        uptr Off = 0;
        if (!Export(LoadBase, (uptr)LoadSize, &H, &Off)) {
          GetAllocator().Deallocate(Alias, (uptr)LoadSize, 0);
          return;
        }
        if (!Map(Alias, (uptr)LoadSize, H, /*ReadOnly=*/false, Device, Off)) {
          Destroy(H);
          GetAllocator().Deallocate(Alias, (uptr)LoadSize, 0);
          VReport(1, "%s: could not map the alias for 0x%zx (+%zu) at 0x%zx\n",
                  SanitizerToolName, (uptr)LoadBase, (uptr)LoadSize, Alias);
          return;
        }

        u64 StorageType = 0, StorageBase = 0, StorageSize = 0;
        const void* Storage = nullptr;
        if (CodeObjectInfo(Obj, kStorageType, &StorageType) &&
            StorageType == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY &&
            CodeObjectInfo(Obj, kStorageBase, &StorageBase) &&
            CodeObjectInfo(Obj, kStorageSize, &StorageSize) && StorageBase &&
            StorageSize)
          Storage = reinterpret_cast<const void*>(StorageBase);

        ImageStats Stats = {};
        TrackImage((uptr)LoadBase, (uptr)LoadSize, Alias, H, Storage,
                   (uptr)StorageSize, &Stats);
        VReport(1,
                "%s: aliased 0x%zx (+%zu) at 0x%zx; %zu globals described, %zu "
                "chunks shared\n",
                SanitizerToolName, (uptr)LoadBase, (uptr)LoadSize, Alias,
                Stats.Described, Stats.Shared);
      });
}

bool Hsa::StillLoaded(uptr LoadBase) {
  if (!EnsureLoader())
    return false;
  for (hsa_executable_t Exec : Execs) {
    u64 Wanted = LoadBase;
    Iterate<hsa_executable_t, hsa_loaded_code_object_t>(
        Loader.hsa_ven_amd_loader_executable_iterate_loaded_code_objects, Exec,
        [&](hsa_executable_t, hsa_loaded_code_object_t Obj) {
          u64 Base = 0;
          if (CodeObjectInfo(Obj, kLoadBase, &Base) && Base == Wanted)
            Wanted = 0;
          return HSA_STATUS_SUCCESS;
        });
    if (Wanted == 0)
      return true;
  }
  return false;
}

bool Hsa::AddRef() { return Refs++ == 0; }

bool Hsa::DropRef() {
  if (!Refs)
    return false;
  if (--Refs)
    return false;
  atomic_store(&Bound, 0, memory_order_release);
  return true;
}

bool Hsa::Init() {
  if (atomic_load(&Bound, memory_order_acquire))
    return true;
  if (!Resolve()) {
    VReport(1,
            "%s: the HSA runtime does not offer the virtual memory "
            "interface; allocations stay unplaced\n",
            SanitizerToolName);
    return false;
  }
  if (!Discover()) {
    VReport(1,
            "%s: no device with placeable memory; allocations stay "
            "unplaced\n",
            SanitizerToolName);
    return false;
  }

  GetAllocator().Init();
  VReport(1, "%s: %zu placeable pools, granule %zu%s\n", SanitizerToolName,
          Pools.size(), Granule, CpuDirect ? "" : ", host copies metadata");
  atomic_store(&Bound, 1, memory_order_release);
  return true;
}

void Hsa::Shutdown() {
  atomic_store(&Bound, 0, memory_order_release);
  StopRpc();
  Lock L(&DasanMutex);
  // hsa_init can AddRef during StopRpc; leave that instance standing.
  if (Refs)
    return;
  for (uptr I = 0; I < NumAliasedImages(); ++I) DropMapping(AliasedImageAt(I));
  Execs.clear();
  ForgetImages();
  ForgetGlobals();
  LoaderReady = false;
  Loader = {};
  GetAllocator().Shutdown();
  DestroyInvalidators();
  Agents.clear();
  Gpus.clear();
  Cpus.clear();
  Pools.clear();
  for (MappedRange& M : Mapped) M.Agents.Destroy();
  Mapped.clear();
  HostPool = {};
  Granule = 0;
  CpuDirect = false;
}

DeviceId Hsa::DeviceForPool(hsa_amd_memory_pool_t Pool) const {
  return PoolFor(Pool.handle) ? Pool.handle : 0;
}

DeviceId Hsa::DeviceForAgent(hsa_agent_t Agent) const {
  if (!IsGpu(Agent))
    return 0;
  DeviceId Fallback = 0;
  for (const GpuPool& P : Pools) {
    if (P.Agent.handle != Agent.handle)
      continue;
    if (P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
      return P.Pool.handle;
    if (!Fallback)
      Fallback = P.Pool.handle;
  }
  return Fallback;
}

u32 Hsa::PoolFlags(DeviceId Device) const {
  const GpuPool* P = PoolFor(Device);
  return P ? P->Flags : 0;
}

const Hsa::GpuPool* Hsa::PoolFor(DeviceId Device) const {
  return FindIf(Pools,
                [&](const GpuPool& P) { return P.Pool.handle == Device; });
}

bool Hsa::AgentForDevice(DeviceId Device, hsa_agent_t* Out) const {
  const GpuPool* P = PoolFor(Device);
  if (!P)
    return false;
  *Out = P->Agent;
  return true;
}

void Hsa::ForgetExecutable(hsa_executable_t Exec) {
  const uptr I = FindIndex(
      Execs, [&](hsa_executable_t E) { return E.handle == Exec.handle; });
  if (I != kNpos)
    EraseSwap(Execs, I);
  for (uptr I = NumAliasedImages(); I > 0; --I) {
    AliasedImage& Img = AliasedImageAt(I - 1);
    if (!Img.Alias || StillLoaded(Img.LoadBase))
      continue;
    const uptr LoadBase = Img.LoadBase;
    const uptr LoadSize = Img.LoadSize;
    const uptr Alias = Img.Alias;
    DropMapping(Img);
    ForgetGlobals(Alias, LoadSize);
    DropAliasedImage(I - 1);
    VReport(1, "%s: released 0x%zx (+%zu) at 0x%zx\n", SanitizerToolName,
            LoadBase, LoadSize, Alias);
  }
}

void Hsa::RecordExecutable(hsa_executable_t Exec) {
  NoteHandle(Execs, Exec);
  AliasExecutable(Exec);
}

bool Hsa::AccessibleAgents(uptr Addr, void* (*Alloc)(size_t), u32* Num,
                           hsa_agent_t** Out) {
  const MappedRange* M =
      FindIf(Mapped, [&](const MappedRange& R) { return R.Covers(Addr); });
  if (!M)
    return false;
  const uptr N = M->Agents.size();
  if (Num)
    *Num = static_cast<u32>(N);
  if (!Out)
    return true;
  *Out = nullptr;
  if (!N || !Alloc)
    return true;
  auto* Agents = reinterpret_cast<hsa_agent_t*>(Alloc(N * sizeof(hsa_agent_t)));
  if (!Agents)
    return false;
  internal_memcpy(Agents, M->Agents.data(), N * sizeof(hsa_agent_t));
  *Out = Agents;
  return true;
}

// OpenMP registers images from a ctor that can run before C++ dynamic
// initializers. Hsa must be trivially constructible so Init is not wiped.
static_assert(__is_trivially_constructible(Hsa),
              "Hsa would re-zero Bound after OpenMP's Init");
Hsa TheHsa;
Hsa& GetHsa() { return TheHsa; }

uptr VaGranule() { return GetHsa().Granule; }
bool VaReserve(uptr Addr, uptr Size) { return GetHsa().Reserve(Addr, Size); }
void VaRelease(uptr Addr, uptr Size) { GetHsa().Release(Addr, Size); }
bool VaCreate(DeviceId Device, uptr Size, u32 Flags, u64* Handle) {
  return GetHsa().Create(Device, Size, Flags, Handle);
}
void VaDestroy(u64 Handle) { GetHsa().Destroy(Handle); }
bool VaMap(uptr Addr, uptr Size, u64 Handle, bool ReadOnly, DeviceId Device) {
  return GetHsa().Map(Addr, Size, Handle, ReadOnly, Device);
}
void VaUnmap(uptr Addr, uptr Size) { GetHsa().Unmap(Addr, Size); }
bool VaAllow(uptr Addr, uptr Size, DeviceId Device, bool ReadOnly) {
  return GetHsa().Allow(Addr, Size, Device, ReadOnly);
}
bool VaProbe(uptr Addr) { return GetHsa().Probe(Addr); }
bool VaWrite(uptr Dst, const void* Src, uptr N) {
  return GetHsa().Write(Dst, Src, N);
}
bool VaFill(uptr Dst, uptr N) { return GetHsa().Fill(Dst, N); }
void VaPublish(uptr Touched) { GetHsa().Publish(Touched); }

bool Hsa::AllocHost(uptr Bytes, void** Out) {
  void* P = nullptr;
  if (!HostPool.handle ||
      Api.hsa_amd_memory_pool_allocate(HostPool, Bytes, 0, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  Api.hsa_amd_agents_allow_access(Agents.size(), Agents.data(), nullptr, P);
  *Out = P;
  return true;
}

void Hsa::FreeHost(void* P) {
  if (P)
    Api.hsa_amd_memory_pool_free(P);
}

bool Hsa::Copy(void* Dst, const void* Src, uptr N) {
  return Api.hsa_memory_copy(Dst, Src, N) == HSA_STATUS_SUCCESS;
}

bool Hsa::SymbolAddr(hsa_executable_t Exec, const char* Name, hsa_agent_t Agent,
                     u64* Addr) {
  hsa_executable_symbol_t Symbol;
  if (Api.hsa_executable_get_symbol_by_name(Exec, Name, &Agent, &Symbol) !=
      HSA_STATUS_SUCCESS)
    return false;
  *Addr = 0;
  return Api.hsa_executable_symbol_get_info(
             Symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, Addr) ==
             HSA_STATUS_SUCCESS &&
         *Addr;
}

bool Hsa::InitDoorbell(hsa_signal_t* Sig, u64** Value, u64** Mailbox,
                       u32* EventId) {
  if (Api.hsa_amd_signal_create(0, 0, nullptr, 0, Sig) != HSA_STATUS_SUCCESS)
    return false;
  // Same ROCr layout the offload plugin peels; not a public interface.
  struct AMDSignal {
    int64_t Kind;
    int64_t Value;
    uint64_t EventMailboxPtr;
    uint32_t EventId;
  };
  auto* S = reinterpret_cast<AMDSignal*>(Sig->handle);
  *Value = reinterpret_cast<u64*>(&S->Value);
  *Mailbox = reinterpret_cast<u64*>(S->EventMailboxPtr);
  *EventId = S->EventId;
  return true;
}

void Hsa::DestroySignal(hsa_signal_t Sig) {
  if (Sig.handle)
    Api.hsa_signal_destroy(Sig);
}

void Hsa::WaitSignalNE(hsa_signal_t Sig, u64 Cmp) {
  Api.hsa_signal_wait_scacquire(Sig, HSA_SIGNAL_CONDITION_NE, Cmp, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED);
}

void Hsa::StoreSignal(hsa_signal_t Sig, u64 Val) {
  Api.hsa_signal_store_screlease(Sig, Val);
}

bool Hsa::AllowAccess(uptr Addr, const hsa_agent_t* Agents, u32 N) {
  Allocator& A = GetAllocator();
  if (!A.IsPlaced(Addr))
    return false;
  AllocInfo Info;
  uptr UserBeg = Addr;
  uptr UserSize = 1;
  if (A.Describe(Addr, &Info)) {
    UserBeg = Info.ChunkBeg;
    UserSize = Info.ChunkSize;
  }
  if (!AllowAgents(UserBeg, UserSize, Agents, N, /*ReadOnly=*/false))
    return false;
  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  return AllowAgents(GetMetadata(Addr, Idx), kMetadataSize, Agents, N,
                     /*ReadOnly=*/true);
}

}  // namespace __dasan
