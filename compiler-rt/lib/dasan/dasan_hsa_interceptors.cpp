//===-- dasan_hsa_interceptors.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HSA interceptors for image freeze/destroy and memory allocation or access.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include "dasan.h"
#include "dasan_allocator.h"
#include "dasan_image.h"
#include "dasan_platform.h"
#include "dasan_report.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_platform.h"

#if !SANITIZER_LINUX
#  error "the device address sanitizer is supported on Linux only"
#endif

using namespace __sanitizer;
using namespace __dasan;

// The runtime exports these to every program, act as if HSA is absent when it
// is not loaded.
#define DASAN_HSA_ENTER(name)                                               \
  Initialize();                                                             \
  if (UNLIKELY(!REAL(name))) {                                              \
    INTERCEPT_FUNCTION(name);                                               \
    if (UNLIKELY(!REAL(name))) {                                            \
      VReport(1, "%s: cannot find %s in this process\n", SanitizerToolName, \
              #name);                                                       \
      return HSA_STATUS_ERROR;                                              \
    }                                                                       \
  }

#define DASAN_HSA_FORWARD(name, ...)     \
  DASAN_HSA_ENTER(name);                 \
  if (UNLIKELY(!Offload::Get().Ready())) \
    return REAL(name)(__VA_ARGS__);

static bool FromHsa(void* P) {
  Dl_info Info = {};
  if (!dladdr(P, &Info) || !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, SANITIZER_HSA_LIBRARY);
}

// Callers bind to whichever 'hsa_init' comes first, if HSA was loaded before
// the runtime the interceptors are bypassed.
static void CheckInterposed() {
  void* Sym = dlsym(RTLD_DEFAULT, "hsa_init");
  if (!Sym || !FromHsa(Sym))
    return;
  Report(
      "WARNING: %s: the runtime is loaded too late to intercept HSA, device "
      "memory will not be checked. Link the runtime first or use "
      "LD_PRELOAD.\n",
      SanitizerToolName);
}

// Dasan's state follows the offload layer's lifetime.
static void Bind() {
  Lock L(&DasanMutex);
  if (!Offload::Get().Ready() || PlatformReady() || !PlatformInit())
    return;
  GetAllocator().Init();
}

static void Unbind() {
  Lock L(&DasanMutex);
  if (!PlatformReady() || Offload::Get().Ready())
    return;
  UnaliasAll();
  GetAllocator().Shutdown();
  PlatformShutdown();
}

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  DASAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  Offload::Get().Init();
  Bind();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  DASAN_HSA_ENTER(hsa_shut_down);

  // Drains outstanding reports while the allocator can still describe them.
  Offload::Get().Shutdown();
  Unbind();
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char* Options) {
  DASAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  {
    Lock L(&DasanMutex);
    AliasExecutable(Executable);
  }
  // Takes the RPC server lock, which nests outside DasanMutex.
  Offload::Get().TrackExecutable(Executable);
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  DASAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  Offload::Get().UntrackExecutable(Executable);
  {
    Lock L(&DasanMutex);
    UnaliasExecutable(Executable);
  }
  return REAL(hsa_executable_destroy)(Executable);
}

// Kernels must run from the alias so their PC-relative globals are checked.
INTERCEPTOR(hsa_status_t, hsa_executable_symbol_get_info,
            hsa_executable_symbol_t Symbol,
            hsa_executable_symbol_info_t Attribute, void* Value) {
  DASAN_HSA_FORWARD(hsa_executable_symbol_get_info, Symbol, Attribute, Value);

  hsa_status_t Status =
      REAL(hsa_executable_symbol_get_info)(Symbol, Attribute, Value);
  if (Status != HSA_STATUS_SUCCESS || !Value ||
      (Attribute != HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS &&
       Attribute != HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT))
    return Status;

  Lock L(&DasanMutex);
  uptr* Address = reinterpret_cast<uptr*>(Value);
  uptr Alias = AliasOf(*Address);
  if (Alias != *Address)
    VReport(2, "%s: the symbol at 0x%zx answers at 0x%zx\n", SanitizerToolName,
            *Address, Alias);
  *Address = Alias;
  return Status;
}

static bool PlaceAlloc(hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags,
                       void** Ptr) {
  if (!Size)
    return false;
  Lock L(&DasanMutex);
  if (!PlatformReady())
    return false;
  DeviceId Device = DeviceForPool(Pool);
  if (!Device)
    return false;
  uptr Placed = GetAllocator().Allocate(Device, Size, Flags);
  if (!Placed)
    return false;
  *Ptr = reinterpret_cast<void*>(Placed);
  return true;
}

static bool PlaceFree(void* Ptr) {
  Lock L(&DasanMutex);
  return GetAllocator().Deallocate(reinterpret_cast<uptr>(Ptr));
}

static bool PlaceAllow(uptr Addr, const hsa_agent_t* Agents, u32 N,
                       hsa_status_t* Status) {
  Lock L(&DasanMutex);
  if (!GetAllocator().IsPlaced(Addr))
    return false;
  *Status = GetAllocator().AllowAccess(Addr, Agents, N) ? HSA_STATUS_SUCCESS
                                                        : HSA_STATUS_ERROR;
  return true;
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags,
            void** Ptr) {
  DASAN_HSA_FORWARD(hsa_amd_memory_pool_allocate, Pool, Size, Flags, Ptr);

  if (PlaceAlloc(Pool, Size, Flags, Ptr))
    return HSA_STATUS_SUCCESS;
  // Unplaced pointers sit outside the checked window, including fine-grain
  // pools whose vmem path the runtime does not yet support.
  return REAL(hsa_amd_memory_pool_allocate)(Pool, Size, Flags, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void* Ptr) {
  DASAN_HSA_FORWARD(hsa_amd_memory_pool_free, Ptr);

  if (PlaceFree(Ptr))
    return HSA_STATUS_SUCCESS;
  return REAL(hsa_amd_memory_pool_free)(Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_memory_allocate, hsa_region_t Region, size_t Size,
            void** Ptr) {
  DASAN_HSA_FORWARD(hsa_memory_allocate, Region, Size, Ptr);

  // ROCr's region handle is the pool handle.
  if (PlaceAlloc({Region.handle}, Size, 0, Ptr))
    return HSA_STATUS_SUCCESS;
  return REAL(hsa_memory_allocate)(Region, Size, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_memory_free, void* Ptr) {
  DASAN_HSA_FORWARD(hsa_memory_free, Ptr);

  if (PlaceFree(Ptr))
    return HSA_STATUS_SUCCESS;
  return REAL(hsa_memory_free)(Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_agents_allow_access, uint32_t NumAgents,
            const hsa_agent_t* Agents, const uint32_t* Flags, const void* Ptr) {
  DASAN_HSA_FORWARD(hsa_amd_agents_allow_access, NumAgents, Agents, Flags, Ptr);

  hsa_status_t Status;
  if (PlaceAllow(reinterpret_cast<uptr>(Ptr), Agents, NumAgents, &Status))
    return Status;
  return REAL(hsa_amd_agents_allow_access)(NumAgents, Agents, Flags, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_set_access, void* Va, size_t Size,
            const hsa_amd_memory_access_desc_t* Desc, size_t DescCnt) {
  DASAN_HSA_FORWARD(hsa_amd_vmem_set_access, Va, Size, Desc, DescCnt);

  hsa_status_t Status;
  if (!Desc || !DescCnt) {
    if (PlaceAllow(reinterpret_cast<uptr>(Va), nullptr, 0, &Status))
      return HSA_STATUS_ERROR;
    return REAL(hsa_amd_vmem_set_access)(Va, Size, Desc, DescCnt);
  }

  InternalMmapVector<hsa_agent_t> Agents;
  for (size_t I = 0; I < DescCnt; ++I) Agents.push_back(Desc[I].agent_handle);
  if (PlaceAllow(reinterpret_cast<uptr>(Va), Agents.data(),
                 static_cast<u32>(Agents.size()), &Status))
    return Status;
  return REAL(hsa_amd_vmem_set_access)(Va, Size, Desc, DescCnt);
}

INTERCEPTOR(hsa_status_t, hsa_amd_pointer_info, const void* Ptr,
            hsa_amd_pointer_info_t* Info, void* (*Alloc)(size_t),
            uint32_t* NumAgents, hsa_agent_t** Accessible) {
  DASAN_HSA_FORWARD(hsa_amd_pointer_info, Ptr, Info, Alloc, NumAgents,
                    Accessible);

  {
    Lock L(&DasanMutex);
    AllocInfo A;
    if (Info && GetAllocator().Describe(reinterpret_cast<uptr>(Ptr), &A) &&
        !A.Freed) {
      Info->type = HSA_EXT_POINTER_TYPE_HSA;
      Info->agentBaseAddress = reinterpret_cast<void*>(A.Beg);
      Info->hostBaseAddress = reinterpret_cast<void*>(A.Beg);
      Info->sizeInBytes = A.Size;
      Info->global_flags = PoolFlags(A.Device);
      Info->registered = false;
      hsa_agent_t Owner = {};
      const bool HaveOwner = AgentForDevice(A.Device, &Owner);
      Info->agentOwner = Owner;
      u32 N = HaveOwner ? 1 : 0;
      hsa_agent_t* Agents = nullptr;
      if (!VaAccessibleAgents(reinterpret_cast<uptr>(Ptr), Alloc, &N,
                              &Agents) &&
          HaveOwner && Alloc) {
        Agents = reinterpret_cast<hsa_agent_t*>(Alloc(sizeof(Owner)));
        if (Agents)
          Agents[0] = Owner;
        N = Agents ? 1 : 0;
      }
      if (NumAgents)
        *NumAgents = N;
      if (Accessible)
        *Accessible = Agents;
      return HSA_STATUS_SUCCESS;
    }
  }
  return REAL(hsa_amd_pointer_info)(Ptr, Info, Alloc, NumAgents, Accessible);
}

__attribute__((constructor(0))) static void DasanDynInit() {
  __dasan_init();
  CheckInterposed();
}

__attribute__((destructor)) static void DasanAtExit() {
  Lock L(&DasanMutex);
  PrintStats();
}
