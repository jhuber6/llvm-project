//===-- dasan_platform.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Which pools are placed into, how each range is mapped, and who may touch it.
// The HSA calls themselves go through the common offload layer.
//
//===----------------------------------------------------------------------===//

#include "dasan_platform.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_offload.h"

using namespace __sanitizer;

namespace __dasan {
namespace {

constexpr uptr kNpos = ~(uptr)0;

template <typename Vec, typename Pred>
uptr FindIndex(const Vec& V, Pred P) {
  for (uptr I = 0; I < V.size(); ++I)
    if (P(V[I]))
      return I;
  return kNpos;
}

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
  void Note(hsa_agent_t Agent) {
    for (hsa_agent_t A : Agents)
      if (A.handle == Agent.handle)
        return;
    Agents.push_back(Agent);
  }
  void Take(MappedRange& O) {
    Addr = O.Addr;
    Size = O.Size;
    Device = O.Device;
    CpuRw = O.CpuRw;
    Agents.swap(O.Agents);
  }
};

// OpenMP registers images from a ctor that can run before C++ dynamic
// initializers, this must be trivially constructible so Init is not wiped.
struct State {
  InternalMmapVectorNoCtor<GpuPool> Pools;
  InternalMmapVectorNoCtor<DeviceId> Devices;
  InternalMmapVectorNoCtor<MappedRange> Mapped;
  hsa_amd_memory_pool_t HostPool;
  uptr Granule;
  bool CpuDirect;
  bool Bound;
};
static_assert(__is_trivially_constructible(State),
              "State would re-zero after OpenMP's Init");
State S;

Offload& O() { return Offload::Get(); }

const GpuPool* PoolFor(DeviceId Device) {
  const uptr I = FindIndex(
      S.Pools, [&](const GpuPool& P) { return P.Pool.handle == Device; });
  return I == kNpos ? nullptr : &S.Pools[I];
}

bool HostCanAccess(DeviceId Device) {
  if (!Device)
    return S.HostPool.handle != 0;
  const GpuPool* P = PoolFor(Device);
  return P && P->CpuAccess;
}

bool CpuCanAccess(hsa_agent_t Cpu, DeviceId Device) {
  return !Device || O().CanAccess(Cpu, hsa_amd_memory_pool_t{Device});
}

bool CpuMapped(uptr Addr) {
  const uptr I =
      FindIndex(S.Mapped, [&](const MappedRange& R) { return R.Covers(Addr); });
  return I != kNpos && S.Mapped[I].CpuRw;
}

void DropMapped(uptr I) {
  S.Mapped[I].Agents.Destroy();
  if (I + 1 != S.Mapped.size())
    S.Mapped[I].Take(S.Mapped.back());
  S.Mapped.pop_back();
}

// Metadata is read-only to the device unless the host cannot reach it
// directly, in which case the copy engine needs the owner to hold it RW.
bool Grant(uptr Addr, uptr Size, bool ReadOnly, DeviceId Device, bool WithCpu) {
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  hsa_agent_t Owner = {};
  const bool HaveOwner = Device && AgentForDevice(Device, &Owner);
  if (WithCpu) {
    for (hsa_agent_t Cpu : O().Cpus()) {
      if (!CpuCanAccess(Cpu, Device))
        continue;
      Descs.push_back({HSA_ACCESS_PERMISSION_RW, Cpu});
    }
  }
  if (HaveOwner)
    Descs.push_back({(ReadOnly && WithCpu) ? HSA_ACCESS_PERMISSION_RO
                                           : HSA_ACCESS_PERMISSION_RW,
                     Owner});
  if (Descs.empty())
    return false;
  if (O().SetAccess(Addr, Size, Descs.data(), Descs.size()))
    return true;
  if (!ReadOnly || !Device || !WithCpu)
    return false;
  Descs.back().permissions = HSA_ACCESS_PERMISSION_RW;
  return O().SetAccess(Addr, Size, Descs.data(), Descs.size());
}

}  // namespace

bool PlatformInit() {
  if (S.Bound)
    return true;
  Offload& Off = O();
  if (!Off.Ready())
    return false;
  if (!Off.HasVirtualMemory()) {
    VReport(1,
            "%s: the HSA runtime does not offer the virtual memory "
            "interface; allocations stay unplaced\n",
            SanitizerToolName);
    return false;
  }

  S.Pools.clear();
  S.Devices.clear();
  S.HostPool = {};
  S.Granule = 0;
  // Host pools stay unplaced: a code object allocated there must keep its
  // loader address or the kernel sees one fat heap object instead of
  // per-global entries.
  u32 Placeable = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  if (Off.FineGrainVmem())
    Placeable |= HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                 HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED;
  for (const Offload::MemoryPool& P : Off.Pools()) {
    if (!Off.IsGpu(P.Agent)) {
      if ((P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
          !S.HostPool.handle)
        S.HostPool = P.Pool;
      continue;
    }
    if (!(P.Flags & Placeable))
      continue;
    S.Granule = Max(S.Granule, P.Granule);
    S.Pools.push_back({P.Agent, P.Pool, P.Flags, P.HostAccess});
  }

  S.CpuDirect = !S.Pools.empty();
  for (const GpuPool& P : S.Pools) S.CpuDirect &= P.CpuAccess;

  // Pools are listed per agent, so first-seen agent order is GPU order.
  for (const GpuPool& P : S.Pools) {
    DeviceId D = DeviceForAgent(P.Agent);
    if (FindIndex(S.Devices, [&](DeviceId E) { return E == D; }) == kNpos)
      S.Devices.push_back(D);
  }

  if (S.Pools.empty() || !S.Granule) {
    VReport(1,
            "%s: no device with placeable memory; allocations stay "
            "unplaced\n",
            SanitizerToolName);
    return false;
  }
  VReport(1, "%s: %zu placeable pools, granule %zu%s%s\n", SanitizerToolName,
          S.Pools.size(), S.Granule,
          S.CpuDirect ? "" : ", host copies metadata",
          Off.FineGrainVmem() ? ", fine-grain vmem" : "");
  S.Bound = true;
  return true;
}

void PlatformShutdown() {
  S.Bound = false;
  S.Pools.clear();
  S.Devices.clear();
  for (MappedRange& M : S.Mapped) M.Agents.Destroy();
  S.Mapped.clear();
  S.HostPool = {};
  S.Granule = 0;
  S.CpuDirect = false;
}

bool PlatformReady() { return S.Bound; }

const InternalMmapVectorNoCtor<DeviceId>& PlatformDevices() {
  return S.Devices;
}

DeviceId DeviceForPool(hsa_amd_memory_pool_t Pool) {
  return PoolFor(Pool.handle) ? Pool.handle : 0;
}

DeviceId DeviceForAgent(hsa_agent_t Agent) {
  DeviceId Fallback = 0;
  for (const GpuPool& P : S.Pools) {
    if (P.Agent.handle != Agent.handle)
      continue;
    if (P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
      return P.Pool.handle;
    if (!Fallback)
      Fallback = P.Pool.handle;
  }
  return Fallback;
}

bool AgentForDevice(DeviceId Device, hsa_agent_t* Out) {
  const GpuPool* P = PoolFor(Device);
  if (!P)
    return false;
  *Out = P->Agent;
  return true;
}

u32 PoolFlags(DeviceId Device) {
  const GpuPool* P = PoolFor(Device);
  return P ? P->Flags : 0;
}

uptr VaGranule() { return S.Granule; }

bool VaReserve(uptr Addr, uptr Size) { return O().ReserveAddress(Addr, Size); }

void VaRelease(uptr Addr, uptr Size) { O().FreeAddress(Addr, Size); }

bool VaCreate(DeviceId Device, uptr Size, u32 Flags, u64* Handle) {
  hsa_amd_memory_pool_t Pool =
      Device ? hsa_amd_memory_pool_t{Device} : S.HostPool;
  return Pool.handle && O().CreateHandle(Pool, Size, Flags, Handle);
}

bool VaShare(uptr Addr, uptr Size, u64* Handle, uptr* Offset) {
  if (O().ExportHandle(Addr, Size, Handle, Offset))
    return true;
  VReport(1, "%s: could not share %zu bytes at 0x%zx for aliasing\n",
          SanitizerToolName, Size, Addr);
  return false;
}

void VaDestroy(u64 Handle) { O().ReleaseHandle(Handle); }

bool VaMap(uptr Addr, uptr Size, u64 Handle, bool ReadOnly, DeviceId Device,
           uptr Offset) {
  if (!O().Map(Addr, Size, Offset, Handle))
    return false;
  const bool WithCpu = HostCanAccess(Device);
  if (!Grant(Addr, Size, ReadOnly, Device, WithCpu)) {
    O().Unmap(Addr, Size);
    return false;
  }
  S.Mapped.resize(S.Mapped.size() + 1);
  MappedRange& R = S.Mapped.back();
  R.Addr = Addr;
  R.Size = Size;
  R.Device = Device;
  R.CpuRw = WithCpu;
  hsa_agent_t Gpu = {};
  if (Device && AgentForDevice(Device, &Gpu))
    R.Note(Gpu);
  if (WithCpu)
    for (hsa_agent_t Cpu : O().Cpus())
      if (CpuCanAccess(Cpu, Device))
        R.Note(Cpu);
  return true;
}

void VaUnmap(uptr Addr, uptr Size) {
  O().Unmap(Addr, Size);
  const uptr I = FindIndex(S.Mapped, [&](const MappedRange& M) {
    return M.Addr == Addr && M.Size == Size;
  });
  if (I != kNpos)
    DropMapped(I);
}

bool VaAllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N,
                   bool ReadOnly) {
  if (!N)
    return true;
  const uptr End = Addr + Size;
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  for (u32 I = 0; I < N; ++I) {
    bool Ro = ReadOnly && O().IsGpu(Agents[I]);
    if (Ro) {
      for (const MappedRange& R : S.Mapped) {
        if (!R.Overlaps(Addr, End))
          continue;
        hsa_agent_t Owner = {};
        if (R.Device && AgentForDevice(R.Device, &Owner) &&
            Owner.handle == Agents[I].handle && !R.CpuRw)
          Ro = false;
      }
    }
    Descs.push_back(
        {Ro ? HSA_ACCESS_PERMISSION_RO : HSA_ACCESS_PERMISSION_RW, Agents[I]});
  }
  bool Hit = false;
  for (MappedRange& M : S.Mapped) {
    if (!M.Overlaps(Addr, End))
      continue;
    Hit = true;
    if (!O().SetAccess(M.Addr, M.Size, Descs.data(), Descs.size()))
      return false;
    for (u32 A = 0; A < N; ++A) M.Note(Agents[A]);
  }
  return Hit;
}

bool VaAllow(uptr Addr, uptr Size, DeviceId Device, bool ReadOnly) {
  hsa_agent_t Agent = {};
  if (!AgentForDevice(Device, &Agent))
    return false;
  return VaAllowAgents(Addr, Size, &Agent, 1, ReadOnly);
}

bool VaAccessibleAgents(uptr Addr, void* (*Alloc)(size_t), u32* Num,
                        hsa_agent_t** Out) {
  const uptr I =
      FindIndex(S.Mapped, [&](const MappedRange& R) { return R.Covers(Addr); });
  if (I == kNpos)
    return false;
  const MappedRange& M = S.Mapped[I];
  const uptr N = M.Agents.size();
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
  internal_memcpy(Agents, M.Agents.data(), N * sizeof(hsa_agent_t));
  *Out = Agents;
  return true;
}

bool VaProbe(uptr Addr) {
  const u64 Magic = 0xda5a4e0000da5a4eULL;
  u64 Got = 0;
  if (CpuMapped(Addr)) {
    *reinterpret_cast<volatile u64*>(Addr) = Magic;
    Got = *reinterpret_cast<volatile u64*>(Addr);
    return Got == Magic;
  }
  void* Dst = reinterpret_cast<void*>(Addr);
  return O().Copy(Dst, &Magic, sizeof(Magic)) &&
         O().Copy(&Got, Dst, sizeof(Got)) && Got == Magic;
}

bool VaWrite(uptr Dst, const void* Src, uptr N) {
  if (CpuMapped(Dst)) {
    internal_memcpy(reinterpret_cast<void*>(Dst), Src, N);
    return true;
  }
  return O().Copy(reinterpret_cast<void*>(Dst), Src, N);
}

bool VaFill(uptr Dst, uptr N) {
  if (CpuMapped(Dst)) {
    internal_memset(reinterpret_cast<void*>(Dst), 0, N);
    return true;
  }
  char Z[4096];
  internal_memset(Z, 0, sizeof(Z));
  while (N) {
    uptr Step = Min(N, (uptr)sizeof(Z));
    if (!O().Copy(reinterpret_cast<void*>(Dst), Z, Step))
      return false;
    Dst += Step;
    N -= Step;
  }
  return true;
}

// Copies are already in VRAM; the barrier still drops a stale cache line so
// the next kernel's invariant load cannot keep the old word.
void VaPublish(uptr Touched) {
  if (Touched && CpuMapped(Touched))
    O().FlushHostWrites(reinterpret_cast<const volatile u64*>(Touched));
  O().InvalidateDeviceCaches();
}

}  // namespace __dasan
