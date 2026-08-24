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
  PoolId Pool;
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
    Pool = O.Pool;
    CpuRw = O.CpuRw;
    Agents.swap(O.Agents);
  }
};

// OpenMP registers images from a ctor that can run before C++ dynamic
// initializers, this must be trivially constructible so Init is not wiped.
struct State {
  InternalMmapVectorNoCtor<GpuPool> Pools;
  InternalMmapVectorNoCtor<PoolId> Placeable;
  InternalMmapVectorNoCtor<MappedRange> Mapped;
  hsa_amd_memory_pool_t HostPool;
  uptr Granule;
  uptr MetaGranule;
  bool CpuDirect;
  bool Bound;
};
static_assert(__is_trivially_constructible(State),
              "State would re-zero after OpenMP's Init");
State S;

Offload& O() { return Offload::Get(); }

const GpuPool* FindPool(PoolId Pool) {
  const uptr I = FindIndex(
      S.Pools, [&](const GpuPool& P) { return P.Pool.handle == Pool; });
  return I == kNpos ? nullptr : &S.Pools[I];
}

bool HostCanAccess(PoolId Pool) {
  if (!Pool)
    return S.HostPool.handle != 0;
  const GpuPool* P = FindPool(Pool);
  return P && P->CpuAccess;
}

bool CpuCanAccess(hsa_agent_t Cpu, PoolId Pool) {
  return !Pool || O().CanAccess(Cpu, hsa_amd_memory_pool_t{Pool});
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

// The host stays writable when it can reach the pages. The owning GPU gets
// read-write, or read-write-execute for a code object.
bool Grant(uptr Addr, uptr Size, PoolId Pool, bool WithCpu, bool Executable) {
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  hsa_agent_t Owner = {};
  const bool HaveOwner = Pool && AgentForPool(Pool, &Owner);
  if (WithCpu) {
    for (hsa_agent_t Cpu : O().Cpus()) {
      if (!CpuCanAccess(Cpu, Pool))
        continue;
      Descs.push_back({HSA_ACCESS_PERMISSION_RW, Cpu});
    }
  }
  if (HaveOwner)
    Descs.push_back(
        {Executable ? HSA_ACCESS_PERMISSION_RWX : HSA_ACCESS_PERMISSION_RW,
         Owner});
  if (Descs.empty())
    return false;
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
  S.Placeable.clear();
  S.HostPool = {};
  S.Granule = S.MetaGranule = 0;
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
    S.MetaGranule =
        Max(S.MetaGranule, P.AllocGranule ? P.AllocGranule : P.Granule);
    S.Pools.push_back({P.Agent, P.Pool, P.Flags, P.HostAccess});
  }

  S.CpuDirect = !S.Pools.empty();
  for (const GpuPool& P : S.Pools) S.CpuDirect &= P.CpuAccess;

  // Pools are listed per agent, so first-seen agent order is GPU order.
  for (const GpuPool& P : S.Pools) {
    PoolId Id = PoolForAgent(P.Agent);
    if (FindIndex(S.Placeable, [&](PoolId E) { return E == Id; }) == kNpos)
      S.Placeable.push_back(Id);
  }

  if (S.Pools.empty() || !S.Granule || !S.MetaGranule) {
    VReport(1,
            "%s: no device with placeable memory; allocations stay "
            "unplaced\n",
            SanitizerToolName);
    return false;
  }
  VReport(1, "%s: %zu placeable pools, granule %zu, metadata %zu%s%s\n",
          SanitizerToolName, S.Pools.size(), S.Granule, S.MetaGranule,
          S.CpuDirect ? "" : ", host copies metadata",
          Off.FineGrainVmem() ? ", fine-grain vmem" : "");
  S.Bound = true;
  return true;
}

void PlatformShutdown() {
  S.Bound = false;
  S.Pools.clear();
  S.Placeable.clear();
  for (MappedRange& M : S.Mapped) M.Agents.Destroy();
  S.Mapped.clear();
  S.HostPool = {};
  S.Granule = S.MetaGranule = 0;
  S.CpuDirect = false;
}

bool PlatformReady() { return S.Bound; }

const InternalMmapVectorNoCtor<PoolId>& PlacedPools() { return S.Placeable; }

PoolId PlaceablePool(hsa_amd_memory_pool_t Pool) {
  return FindPool(Pool.handle) ? Pool.handle : 0;
}

PoolId PoolForAgent(hsa_agent_t Agent) {
  PoolId Fallback = 0;
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

bool AgentForPool(PoolId Pool, hsa_agent_t* Out) {
  const GpuPool* P = FindPool(Pool);
  if (!P)
    return false;
  *Out = P->Agent;
  return true;
}

u32 PoolFlags(PoolId Pool) {
  const GpuPool* P = FindPool(Pool);
  return P ? P->Flags : 0;
}

uptr VaGranule() { return S.Granule; }

uptr VaMetaGranule() { return S.MetaGranule; }

bool VaReserve(uptr Addr, uptr Size) { return O().ReserveAddress(Addr, Size); }

void VaRelease(uptr Addr, uptr Size) { O().FreeAddress(Addr, Size); }

bool VaCreate(PoolId Pool, uptr Size, u32 Flags, u64* Handle) {
  hsa_amd_memory_pool_t Memory =
      Pool ? hsa_amd_memory_pool_t{Pool} : S.HostPool;
  return Memory.handle && O().CreateHandle(Memory, Size, Flags, Handle);
}

bool VaShare(uptr Addr, uptr Size, u64* Handle, uptr* Offset) {
  if (O().ExportHandle(Addr, Size, Handle, Offset))
    return true;
  VReport(1, "%s: could not share %zu bytes at 0x%zx for aliasing\n",
          SanitizerToolName, Size, Addr);
  return false;
}

void VaDestroy(u64 Handle) { O().ReleaseHandle(Handle); }

bool VaMap(uptr Addr, uptr Size, u64 Handle, PoolId Pool, uptr Offset,
           bool Executable) {
  if (!O().Map(Addr, Size, Offset, Handle))
    return false;
  const bool WithCpu = HostCanAccess(Pool);
  if (!Grant(Addr, Size, Pool, WithCpu, Executable)) {
    O().Unmap(Addr, Size);
    return false;
  }
  S.Mapped.resize(S.Mapped.size() + 1);
  MappedRange& R = S.Mapped.back();
  R.Addr = Addr;
  R.Size = Size;
  R.Pool = Pool;
  R.CpuRw = WithCpu;
  hsa_agent_t Gpu = {};
  if (Pool && AgentForPool(Pool, &Gpu))
    R.Note(Gpu);
  if (WithCpu)
    for (hsa_agent_t Cpu : O().Cpus())
      if (CpuCanAccess(Cpu, Pool))
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

bool VaAllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N) {
  if (!N)
    return true;
  const uptr End = Addr + Size;
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  for (u32 I = 0; I < N; ++I)
    Descs.push_back({HSA_ACCESS_PERMISSION_RW, Agents[I]});
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

bool VaAllow(uptr Addr, uptr Size, PoolId Pool) {
  hsa_agent_t Agent = {};
  if (!AgentForPool(Pool, &Agent))
    return false;
  return VaAllowAgents(Addr, Size, &Agent, 1);
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

bool VaPublish(uptr Addr, const void* Src, uptr N) {
  if (!N)
    return true;
  uptr Done = 0;
  while (Done < N) {
    uptr Pos = Addr + Done;
    const uptr I = FindIndex(
        S.Mapped, [&](const MappedRange& R) { return R.Covers(Pos); });
    if (I == kNpos) {
      Report("%s: device write has no mapping at 0x%zx\n", SanitizerToolName,
             Pos);
      return false;
    }
    const MappedRange& M = S.Mapped[I];
    uptr Step = Min(N - Done, M.Addr + M.Size - Pos);
    const void* Part = Src ? static_cast<const u8*>(Src) + Done : nullptr;
    if (!O().PublishDeviceBytes(Pos, Part, Step, M.Agents.data(),
                                M.Agents.size()))
      return false;
    // The host check loads this mapping. The confirmed L2 store does not
    // update the copy that load reads.
    if (M.CpuRw) {
      void* Dst = reinterpret_cast<void*>(Pos);
      if (Part)
        internal_memcpy(Dst, Part, Step);
      else
        internal_memset(Dst, 0, Step);
    }
    Done += Step;
  }
  return true;
}

}  // namespace __dasan
