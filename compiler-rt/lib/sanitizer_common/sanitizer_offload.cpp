//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_offload.h"

#include <dlfcn.h>

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_offload_rpc.h"

namespace __sanitizer {
namespace {

// Generic helpers to iterate HSA agents and pools.
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

void CheckHsa(hsa_status_t S) {
  if (S == HSA_STATUS_SUCCESS)
    return;
  Report("ERROR: %s: HSA query failed\n", SanitizerToolName);
  Die();
}

// PM4 type-3 header. Count is the body size minus one. Bit 1 is the compute
// shader type.
u32 Pm4Header(u32 Opcode, u32 Dwords) {
  return (1u << 1) | (Opcode << 8) | ((Dwords - 2) << 16) | (3u << 30);
}

// One WRITE_DATA of a contiguous run. dst_sel = TC_L2, wr_confirm, address
// increments per dword, cache policy LRU. The control word is the same from
// gfx9 onward; a qword is two data dwords, a longer run is more of them.
constexpr uptr kPm4IbBytes = 64 * 1024;
constexpr u32 kIbDwords = kPm4IbBytes / sizeof(u32);
constexpr uptr kWriteChunkBytes = (kIbDwords - 4) * sizeof(u32);

u32 EmitWriteData(u32* Out, uptr Addr, const void* Src, uptr Bytes) {
  u32 DataDwords = Bytes / sizeof(u32);
  u32 PacketDwords = DataDwords + 4;
  Out[0] = Pm4Header(0x37, PacketDwords);
  Out[1] = (2u << 8) | (1u << 20);
  Out[2] = static_cast<u32>(Addr);
  Out[3] = static_cast<u32>(Addr >> 32);
  if (Src)
    internal_memcpy(Out + 4, Src, Bytes);
  else
    internal_memset(Out + 4, 0, Bytes);
  return PacketDwords;
}

// Vendor AQL packet that jumps to a PM4 indirect buffer. dw_cnt_remain counts
// the dwords after that field; the firmware expects 0xA.
struct AmdAqlPm4Ib {
  u16 Header;
  u16 VenHdr;
  u32 IbJump[4];
  u32 DwCntRemain;
  u32 Reserved[8];
  hsa_signal_t Completion;
};
static_assert(sizeof(AmdAqlPm4Ib) == 64, "vendor PM4 packet");
static_assert(offsetof(AmdAqlPm4Ib, Completion) == 56, "vendor PM4 packet");

}  // namespace

Offload Offload::Ctx;

Offload& Offload::Get() { return Ctx; }

template <typename Cb>
void Offload::ForEachAgentObject(hsa_executable_t Exec, Cb F) {
  Iterate<hsa_executable_t, hsa_loaded_code_object_t>(
      Loader.IterateLoadedCodeObjects, Exec,
      [&](hsa_executable_t, hsa_loaded_code_object_t Obj) {
        u32 Kind = 0;
        if (Loader.GetCodeObjectInfo(
                Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND, &Kind) !=
                HSA_STATUS_SUCCESS ||
            Kind != HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT)
          return HSA_STATUS_SUCCESS;
        hsa_agent_t Agent{};
        if (Loader.GetCodeObjectInfo(
                Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                &Agent) != HSA_STATUS_SUCCESS ||
            !Agent.handle)
          return HSA_STATUS_SUCCESS;
        F(Obj, Agent);
        return HSA_STATUS_SUCCESS;
      });
}

// Fetch all the real HSA library calls we need.
bool Offload::Resolve() {
#define SANITIZER_HSA_RESOLVE(Name)                                        \
  Api.Name = reinterpret_cast<decltype(&::Name)>(dlsym(RTLD_NEXT, #Name)); \
  if (!Api.Name)                                                           \
    return false;
  SANITIZER_HSA_FUNCTIONS(SANITIZER_HSA_RESOLVE)
#undef SANITIZER_HSA_RESOLVE
#define SANITIZER_HSA_RESOLVE_OPTIONAL(Name) \
  Api.Name = reinterpret_cast<decltype(&::Name)>(dlsym(RTLD_NEXT, #Name));
  SANITIZER_HSA_VMEM_FUNCTIONS(SANITIZER_HSA_RESOLVE_OPTIONAL)
#undef SANITIZER_HSA_RESOLVE_OPTIONAL
  return true;
}

// Iterate the topology to discover agents. If this fails the interceptors are
// disabled.
bool Offload::Discover() {
  DeviceList.clear();
  CpuList.clear();
  PoolList.clear();
  HostDevice = {};

  auto Info = [&](hsa_agent_t Agent, u32 Attr, void* Out) {
    return Api.hsa_agent_get_info(Agent, static_cast<hsa_agent_info_t>(Attr),
                                  Out) == HSA_STATUS_SUCCESS;
  };
  auto PoolInfo = [&](hsa_amd_memory_pool_t Mem, u32 Attr, void* Out) {
    return Api.hsa_amd_memory_pool_get_info(
               Mem, static_cast<hsa_amd_memory_pool_info_t>(Attr), Out) ==
           HSA_STATUS_SUCCESS;
  };
  // Record every allocatable global pool, returns the fine-grained one.
  auto Pool = [&](hsa_agent_t Agent) {
    hsa_amd_memory_pool_t Fine{};
    Iterate<hsa_amd_memory_pool_t>(
        Api.hsa_amd_agent_iterate_memory_pools, Agent,
        [&](hsa_amd_memory_pool_t Mem) {
          hsa_amd_segment_t Seg;
          u32 Flags = 0;
          if (!PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &Seg) ||
              Seg != HSA_AMD_SEGMENT_GLOBAL)
            return HSA_STATUS_SUCCESS;
          if (!PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags))
            return HSA_STATUS_SUCCESS;
          if (Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
            Fine = Mem;

          bool Allowed = false;
          if (!PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                        &Allowed) ||
              !Allowed)
            return HSA_STATUS_SUCCESS;
          MemoryPool P = {Agent, Mem, Flags, 0, 0, false};
          uptr Granule = 0;
          if (PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                       &Granule))
            P.AllocGranule = P.Granule = Granule;
          if (PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                       &Granule) &&
              Granule > P.Granule)
            P.Granule = Granule;
          PoolList.push_back(P);
          return HSA_STATUS_SUCCESS;
        });
    return Fine;
  };

  CheckHsa(Iterate<hsa_agent_t>(Api.hsa_iterate_agents, [&](hsa_agent_t Agent) {
    hsa_device_type_t Type;
    if (hsa_status_t S =
            Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type))
      return S;
    Device D = {};
    D.Agent = Agent;
    D.Pool = Pool(Agent);
    if (Type == HSA_DEVICE_TYPE_CPU) {
      CpuList.push_back(Agent);
      if (!HostDevice.Agent.handle)
        HostDevice = D;
    } else if (Type == HSA_DEVICE_TYPE_GPU) {
      u32 CUs = 0, WavesPerCU = 0;
      if (!Info(Agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &D.Lanes) || !D.Lanes ||
          !Info(Agent, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &CUs) ||
          !Info(Agent, HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, &WavesPerCU) ||
          !CUs || !WavesPerCU)
        CheckHsa(HSA_STATUS_ERROR);
      // The RPC interface is deliberately sized to the hardware parallel
      // limits of the device to make deadlock impossible.
      D.MaxWaves = CUs * WavesPerCU;
      DeviceList.push_back(D);
    }
    return HSA_STATUS_SUCCESS;
  }));

  for (MemoryPool& P : PoolList)
    for (hsa_agent_t Cpu : CpuList) P.HostAccess |= CanAccess(Cpu, P.Pool);
  return HostDevice.Agent.handle && !DeviceList.empty() &&
         HostDevice.Pool.handle;
}

// Initialize the HSA loader extension used to manage host addresses.
bool Offload::BindLoader() {
  Loader = {};
  // Older runtimes reject a table larger than the one they publish.
  const uptr Prefix = offsetof(LoaderApi, GetLoadSize);
  CheckHsa(Api.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1,
                                                    Prefix, &Loader));
  // ROCr's hsa_system_major_extension_supported reports minor 0 for every
  // loader revision, including one that publishes the 1.04 slots. A zeroed
  // table stays null past the prefix an older runtime copies.
  LoaderApi104 Table = {};
  if (Api.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1,
                                               sizeof(Table),
                                               &Table) == HSA_STATUS_SUCCESS &&
      Table.GetLoadSize && Table.LoadAtAddress) {
    Loader.GetLoadSize = Table.GetLoadSize;
    Loader.LoadAtAddress = Table.LoadAtAddress;
    LoaderMinor = 4;
  }
  return Loader.QueryHostAddress && Loader.IterateLoadedCodeObjects &&
         Loader.GetCodeObjectInfo;
}

bool Offload::ExecutableInfo(hsa_loaded_code_object_t Obj,
                             hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                             u64* Out) {
  *Out = 0;
  return Loader.GetCodeObjectInfo(Obj, Attr, Out) == HSA_STATUS_SUCCESS;
}

bool Offload::DescribeCodeObject(hsa_loaded_code_object_t Obj,
                                 hsa_agent_t Agent, CodeObject* Out) {
  *Out = {};
  Out->Agent = Agent;
  u64 LoadBase = 0, LoadSize = 0;
  if (!ExecutableInfo(Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                      &LoadBase) ||
      !ExecutableInfo(Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                      &LoadSize) ||
      !LoadBase || !LoadSize)
    return false;
  Out->LoadBase = (uptr)LoadBase;
  Out->LoadSize = (uptr)LoadSize;

  u64 StorageType = 0, StorageBase = 0, StorageSize = 0;
  if (ExecutableInfo(
          Obj,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &StorageType) &&
      StorageType == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY &&
      ExecutableInfo(
          Obj,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &StorageBase) &&
      ExecutableInfo(
          Obj,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &StorageSize) &&
      StorageBase && StorageSize) {
    Out->Storage = reinterpret_cast<const void*>(StorageBase);
    Out->StorageSize = (uptr)StorageSize;
  }
  return true;
}

void Offload::VisitCodeObjects(hsa_executable_t Exec,
                               void (*Fn)(const CodeObject&, void*),
                               void* Arg) {
  InternalMmapVector<CodeObject> Objects;
  {
    Lock L(&OffloadMtx);
    if (!Ready())
      return;
    ForEachAgentObject(Exec,
                       [&](hsa_loaded_code_object_t Obj, hsa_agent_t Agent) {
                         CodeObject Desc;
                         if (DescribeCodeObject(Obj, Agent, &Desc))
                           Objects.push_back(Desc);
                       });
  }
  for (const CodeObject& Obj : Objects) Fn(Obj, Arg);
}

bool Offload::Init() {
  Lock Life(&LifetimeMtx);
  Lock L(&OffloadMtx);
  if (++Refs > 1)
    return Ready();
  if (!Resolve()) {
    Report("ERROR: %s: cannot resolve HSA\n", SanitizerToolName);
    Die();
  }
  if (!Discover() || !BindLoader()) {
    --Refs;
    return false;
  }
  u16 Minor = 0;
  if (Api.hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR, &Minor) ==
      HSA_STATUS_SUCCESS)
    AmdExtMinor = Minor;
  VReport(1,
          "%s: device reporting on %zu GPU(s), AMD extension 1.%u, "
          "loader 1.%u\n",
          SanitizerToolName, DeviceList.size(), AmdExtMinor, LoaderMinor);
  CreateCommands();
  atomic_store(&Active, 1, memory_order_release);
  return true;
}

bool Offload::Release() {
  if (!Refs || --Refs)
    return false;
  atomic_store(&Active, 0, memory_order_release);
  return true;
}

void Offload::Teardown() {
  UntrackImages();
  Loader = {};
  for (Device& D : DeviceList) {
    if (D.Queue)
      Api.hsa_queue_destroy(D.Queue);
    if (D.Signal.handle)
      DestroySignal(D.Signal);
  }
  if (Ib)
    Free(Ib);
  Ib = nullptr;
  DeviceList.clear();
  CpuList.clear();
  PoolList.clear();
  HostDevice = {};
  AmdExtMinor = 0;
  LoaderMinor = 0;
}

void Offload::Shutdown() {
  Lock Life(&LifetimeMtx);
  bool Last;
  {
    Lock L(&OffloadMtx);
    Last = Release();
  }
  if (!Last)
    return;
  OffloadRpc::Stop(*this);
  Lock L(&OffloadMtx);
  Teardown();
}

bool Offload::Ready() const {
  return atomic_load(&Active, memory_order_acquire) != 0;
}

void Offload::RegisterHandler(Handler Fn) { OffloadRpc::RegisterHandler(Fn); }

// Record every executable we come across for symbolization and address lookup.
void Offload::TrackExecutable(hsa_executable_t Exec) {
  {
    Lock L(&OffloadMtx);
    if (!Ready())
      return;
    ForEachAgentObject(Exec,
                       [&](hsa_loaded_code_object_t Obj, hsa_agent_t Agent) {
                         CodeObject Desc;
                         if (DescribeCodeObject(Obj, Agent, &Desc))
                           TrackImage(Desc.LoadBase, Desc.LoadSize,
                                      Desc.Storage, Desc.StorageSize);
                       });
  }
  OffloadRpc::Start(*this, Exec);
}

void Offload::UntrackExecutable(hsa_executable_t Exec) {
  OffloadRpc::Flush();
  Lock L(&OffloadMtx);
  ForEachAgentObject(Exec, [&](hsa_loaded_code_object_t Obj, hsa_agent_t) {
    u64 LoadBase = 0;
    if (ExecutableInfo(Obj,
                       HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                       &LoadBase) &&
        LoadBase)
      UntrackImage((uptr)LoadBase);
  });
}

// Allocate coherent 'fine-grained' memory for host and device communication.
bool Offload::Alloc(const Device& D, uptr Bytes, void** Out, u64 Flags) {
  void* P = nullptr;
  if (!D.Pool.handle ||
      Api.hsa_amd_memory_pool_allocate(D.Pool, Bytes, Flags, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  InternalMmapVector<hsa_agent_t> All;
  All.push_back(HostDevice.Agent);
  for (uptr I = 0; I < DeviceList.size(); ++I)
    All.push_back(DeviceList[I].Agent);
  if (Api.hsa_amd_agents_allow_access(All.size(), All.data(), nullptr, P) !=
      HSA_STATUS_SUCCESS) {
    Api.hsa_amd_memory_pool_free(P);
    return false;
  }
  *Out = P;
  return true;
}

bool Offload::GetMemoryPool(hsa_agent_t Agent, hsa_amd_memory_pool_t* Pool) {
  Lock L(&OffloadMtx);
  if (!Ready() || !Pool)
    return false;
  *Pool = {};
  hsa_status_t Status = Iterate<hsa_amd_memory_pool_t>(
      Api.hsa_amd_agent_iterate_memory_pools, Agent,
      [&](hsa_amd_memory_pool_t Mem) {
        hsa_amd_segment_t Segment;
        u32 Flags = 0;
        bool Allowed = false;
        if (Api.hsa_amd_memory_pool_get_info(Mem,
                                             HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                             &Segment) != HSA_STATUS_SUCCESS ||
            Segment != HSA_AMD_SEGMENT_GLOBAL ||
            Api.hsa_amd_memory_pool_get_info(
                Mem, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags) !=
                HSA_STATUS_SUCCESS ||
            !(Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) ||
            Api.hsa_amd_memory_pool_get_info(
                Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                &Allowed) != HSA_STATUS_SUCCESS ||
            !Allowed)
          return HSA_STATUS_SUCCESS;
        *Pool = Mem;
        return HSA_STATUS_SUCCESS;
      });
  return Status == HSA_STATUS_SUCCESS && Pool->handle;
}

bool Offload::Allocate(hsa_amd_memory_pool_t Pool, uptr Bytes, void** Out) {
  Lock L(&OffloadMtx);
  if (!Ready() || !Pool.handle || !Bytes || !Out)
    return false;
  void* P = nullptr;
  if (Api.hsa_amd_memory_pool_allocate(Pool, Bytes, 0, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  *Out = P;
  return true;
}

void Offload::Free(void* P) { Api.hsa_amd_memory_pool_free(P); }

bool Offload::Copy(void* Dst, const void* Src, uptr N) {
  return Api.hsa_memory_copy(Dst, Src, N) == HSA_STATUS_SUCCESS;
}

// The runtime knows the original host address of a device pointer that is
// located inside one of the loaded segments, accesses read-only data we need.
const void* Offload::HostPointer(uptr DeviceAddr) {
  Lock L(&OffloadMtx);
  if (!Ready())
    return nullptr;
  if (!DeviceAddr)
    return nullptr;
  const void* HostAddr = nullptr;
  if (Loader.QueryHostAddress(reinterpret_cast<const void*>(DeviceAddr),
                              &HostAddr) != HSA_STATUS_SUCCESS)
    return nullptr;
  return HostAddr;
}

bool Offload::Lookup(hsa_executable_t Exec, const char* Name, hsa_agent_t Agent,
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

bool Offload::CreateSignal(hsa_signal_t* Out) {
  return Api.hsa_amd_signal_create(0, 0, nullptr, 0, Out) == HSA_STATUS_SUCCESS;
}

void Offload::DestroySignal(hsa_signal_t Sig) { Api.hsa_signal_destroy(Sig); }

void Offload::WaitSignal(hsa_signal_t Sig) {
  Api.hsa_signal_wait_scacquire(Sig, HSA_SIGNAL_CONDITION_NE, 0, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED);
}

void Offload::StoreSignal(hsa_signal_t Sig) {
  Api.hsa_signal_store_screlease(Sig, 1);
}

bool Offload::IsGpu(hsa_agent_t Agent) const {
  for (const Device& D : DeviceList)
    if (D.Agent.handle == Agent.handle)
      return true;
  return false;
}

bool Offload::CanAccess(hsa_agent_t Agent, hsa_amd_memory_pool_t Pool) {
  hsa_amd_memory_pool_access_t Access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  return Api.hsa_amd_agent_memory_pool_get_info(
             Agent, Pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &Access) ==
             HSA_STATUS_SUCCESS &&
         Access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
}

bool Offload::FineGrainVmem() const { return AmdExtMinor >= 33; }

bool Offload::CanLoadAtAddress() const {
  return Loader.GetLoadSize && Loader.LoadAtAddress;
}

bool Offload::DmaBufAliasExecutable() const { return AmdExtMinor < 16; }

bool Offload::CodeObjectLoadSize(hsa_code_object_reader_t Reader,
                                 uptr* Size) const {
  size_t Bytes = 0;
  if (!Loader.GetLoadSize ||
      Loader.GetLoadSize(Reader, &Bytes) != HSA_STATUS_SUCCESS || !Bytes)
    return false;
  *Size = (uptr)Bytes;
  return true;
}

bool Offload::LoadCodeObjectAt(hsa_executable_t Exec, hsa_agent_t Agent,
                               hsa_code_object_reader_t Reader,
                               const char* Options, void* Address, uptr Size,
                               hsa_loaded_code_object_t* Loaded) const {
  return Loader.LoadAtAddress &&
         Loader.LoadAtAddress(Exec, Agent, Reader, Options, Address, Size,
                              Loaded) == HSA_STATUS_SUCCESS;
}

bool Offload::HasVirtualMemory() const {
#define SANITIZER_HSA_PRESENT(Name) &&Api.Name
  return true SANITIZER_HSA_VMEM_FUNCTIONS(SANITIZER_HSA_PRESENT);
#undef SANITIZER_HSA_PRESENT
}

// Reserve exactly [Addr, Addr + Size) or nothing.
bool Offload::ReserveAddress(uptr Addr, uptr Size) {
  // NO_REGISTER adds the alignment to the mmap, so that call gets a page.
  // The registered fallback wants the address's own alignment.
  auto Exact = [&](u64 Flags, uptr Align) {
    void* Va = nullptr;
    if (!Align || Api.hsa_amd_vmem_address_reserve_align(
                      &Va, Size, Addr, Align, Flags) != HSA_STATUS_SUCCESS)
      return false;
    if (reinterpret_cast<uptr>(Va) == Addr)
      return true;
    Api.hsa_amd_vmem_address_free(Va, Size);
    return false;
  };
  uptr Page = GetPageSizeCached();
  return Exact(HSA_AMD_VMEM_ADDRESS_NO_REGISTER, Page) ||
         Exact(0, Addr & -Addr);
}

void Offload::FreeAddress(uptr Addr, uptr Size) {
  Api.hsa_amd_vmem_address_free(reinterpret_cast<void*>(Addr), Size);
}

bool Offload::CreateHandle(hsa_amd_memory_pool_t Pool, uptr Size, u64 Flags,
                           u64* Handle) {
  hsa_amd_vmem_alloc_handle_t H;
  if (Api.hsa_amd_vmem_handle_create(Pool, Size, MEMORY_TYPE_NONE, Flags, &H) !=
      HSA_STATUS_SUCCESS)
    return false;
  *Handle = H.handle;
  return true;
}

void Offload::ReleaseHandle(u64 Handle) {
  Api.hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t{Handle});
}

// A second handle to the physical memory behind an existing allocation.
bool Offload::ExportHandle(uptr Addr, uptr Size, u64* Handle, uptr* Offset) {
  int Fd = -1;
  uint64_t Off = 0;
  if (Api.hsa_amd_portable_export_dmabuf(reinterpret_cast<void*>(Addr), Size,
                                         &Fd, &Off) != HSA_STATUS_SUCCESS)
    return false;
  hsa_amd_vmem_alloc_handle_t H;
  hsa_status_t Status = Api.hsa_amd_vmem_import_shareable_handle(Fd, &H);
  Api.hsa_amd_portable_close_dmabuf(Fd);
  if (Status != HSA_STATUS_SUCCESS)
    return false;
  *Handle = H.handle;
  *Offset = (uptr)Off;
  return true;
}

bool Offload::Map(uptr Addr, uptr Size, uptr Offset, u64 Handle) {
  return Api.hsa_amd_vmem_map(reinterpret_cast<void*>(Addr), Size, Offset,
                              hsa_amd_vmem_alloc_handle_t{Handle},
                              0) == HSA_STATUS_SUCCESS;
}

void Offload::Unmap(uptr Addr, uptr Size) {
  Api.hsa_amd_vmem_unmap(reinterpret_cast<void*>(Addr), Size);
}

bool Offload::SetAccess(uptr Addr, uptr Size,
                        const hsa_amd_memory_access_desc_t* Desc, uptr Count) {
  return Api.hsa_amd_vmem_set_access(reinterpret_cast<void*>(Addr), Size, Desc,
                                     Count) == HSA_STATUS_SUCCESS;
}

// One queue per GPU, at the agent's minimum depth, and one executable
// fine-grained buffer. Both stay until shutdown. A store fills the buffer
// and rings the queue.
void Offload::CreateCommands() {
  if (!Alloc(Host(), kPm4IbBytes, &Ib, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG)) {
    Report("%s: device write buffer unavailable\n", SanitizerToolName);
    return;
  }
  for (Device& D : DeviceList) {
    u32 Packets = 0;
    if (Api.hsa_agent_get_info(D.Agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                               &Packets) != HSA_STATUS_SUCCESS ||
        !Packets ||
        Api.hsa_queue_create(D.Agent, Packets, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &D.Queue) != HSA_STATUS_SUCCESS ||
        !CreateSignal(&D.Signal)) {
      if (D.Queue) {
        Api.hsa_queue_destroy(D.Queue);
        D.Queue = nullptr;
      }
      Report("%s: device write queue unavailable\n", SanitizerToolName);
    }
  }
}

void Offload::SubmitIb(Device& D, u32 Dwords) {
  AmdAqlPm4Ib Pkt = {};
  Pkt.VenHdr = 1;
  uptr Base = reinterpret_cast<uptr>(Ib);
  // The AQL firmware rejects the compute bit on this jump.
  Pkt.IbJump[0] = Pm4Header(0x3F, 4) & ~(1u << 1);
  Pkt.IbJump[1] = static_cast<u32>(Base);
  Pkt.IbJump[2] = static_cast<u32>(Base >> 32);
  Pkt.IbJump[3] = (Dwords & 0xFFFFFu) | (1u << 23);
  Pkt.DwCntRemain = 0xA;
  Pkt.Completion = D.Signal;

  Api.hsa_signal_store_screlease(D.Signal, 1);
  u64 Index = Api.hsa_queue_add_write_index_screlease(D.Queue, 1);
  auto* Slot = static_cast<u8*>(D.Queue->base_address) +
               (Index % D.Queue->size) * sizeof(Pkt);
  internal_memcpy(Slot + sizeof(u32), reinterpret_cast<u8*>(&Pkt) + sizeof(u32),
                  sizeof(Pkt) - sizeof(u32));
  __atomic_store_n(reinterpret_cast<u32*>(Slot),
                   u32(Pkt.VenHdr) << 16 | Pkt.Header, __ATOMIC_RELEASE);
  Api.hsa_signal_store_screlease(D.Queue->doorbell_signal, Index);
}

void Offload::Await(Device& D) {
  // The runtime may wake before the value meets the condition.
  while (Api.hsa_signal_wait_scacquire(D.Signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                       UINT64_MAX,
                                       HSA_WAIT_STATE_BLOCKED) != 0);
}

bool Offload::PublishDeviceBytes(uptr Addr, const void* Src, uptr Bytes,
                                 const hsa_agent_t* Agents, u32 AgentCount) {
  Lock L(&OffloadMtx);
  if (!Bytes)
    return true;
  if (!Ready() || !Ib || (Addr % sizeof(u64)) || (Bytes % sizeof(u64))) {
    Report("%s: device write rejected\n", SanitizerToolName);
    return false;
  }

  InternalMmapVector<Device*> Targets;
  for (u32 I = 0; Agents && I < AgentCount; ++I)
    for (Device& D : DeviceList)
      if (D.Agent.handle == Agents[I].handle)
        Targets.push_back(&D);
  if (Targets.empty())
    return true;
  for (Device* D : Targets) {
    if (D->Queue)
      continue;
    Report("%s: device write queue unavailable\n", SanitizerToolName);
    return false;
  }

  for (uptr Off = 0; Off < Bytes;) {
    uptr Step = Min(Bytes - Off, kWriteChunkBytes);
    const void* Part = Src ? static_cast<const u8*>(Src) + Off : nullptr;
    u32 N = EmitWriteData(static_cast<u32*>(Ib), Addr + Off, Part, Step);
    CHECK_LE(N, kIbDwords);
    for (Device* D : Targets) SubmitIb(*D, N);
    for (Device* D : Targets) Await(*D);
    Off += Step;
  }
  return true;
}

}  // namespace __sanitizer
