//===-- asan_offload_hsa_interceptors.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interception of the HSA allocation and copy entry points. Memory the runtime
// hands to the device gets the same redzones and shadow encoding as host
// memory, so both halves of the sanitizer agree on every allocation.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <stddef.h>

#include "asan_interceptors_memintrinsics.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_suppressions.h"
#include "asan_offload.h"
#include "asan_offload_hsa.h"
#include "asan_offload_rpc.h"
#include "asan_offload_symbolize.h"
#include "asan_poisoning.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

#if !SANITIZER_LINUX
#error "Offload ASan reporting is supported on Linux only"
#endif

#if SANITIZER_GLIBC
#pragma weak dlvsym
#endif

using namespace __sanitizer;
using namespace __asan;

namespace __asan {

Mutex AsanOffloadMutex;
Mutex HsaLifecycleMutex;

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  SanitizerToolName = "AddressSanitizer";
  AsanInitFromRtl();
  Atexit(ForgetDeviceImages);
  AddDieCallback(ForgetDeviceImages);
  atomic_store(&Initialized, 1, memory_order_release);
}

} // namespace __asan

#define ASAN_HSA_ENTER(name)                                                   \
  Initialize();                                                                \
  if (UNLIKELY(!REAL(name))) {                                                 \
    INTERCEPT_FUNCTION(name);                                                  \
    if (UNLIKELY(!REAL(name))) {                                               \
      Report("ERROR: %s: cannot find %s in this process\n", SanitizerToolName, \
             #name);                                                           \
      Die();                                                                   \
    }                                                                          \
  }

#define ASAN_HSA_FORWARD(name, ...)                                            \
  ASAN_HSA_ENTER(name);                                                        \
  if (UNLIKELY(!GetHsa().Ready()))                                             \
    return REAL(name)(__VA_ARGS__);

#define ASAN_HSA_WRAPS(X)                                                      \
  X(hsa_init)                                                                  \
  X(hsa_shut_down)                                                             \
  X(hsa_executable_freeze)                                                     \
  X(hsa_executable_destroy)                                                    \
  X(hsa_amd_memory_pool_allocate)                                              \
  X(hsa_amd_memory_pool_free)                                                  \
  X(hsa_memory_free)                                                           \
  X(hsa_amd_agents_allow_access)                                               \
  X(hsa_amd_pointer_info)                                                      \
  X(hsa_memory_copy)                                                           \
  X(hsa_amd_memory_async_copy)                                                 \
  X(hsa_amd_memory_async_copy_on_engine)

// A page keeps the alignment the memory pool promised its callers; the runtime
// and its clients rely on allocations being page aligned.
static constexpr uptr kOffloadRedzone = 4096;

struct DeviceAlloc {
  uptr Raw;
  uptr Total;
  uptr User;
  uptr Size;
  u32 AllocStack;
  u32 FreeStack;
  RegionOrigin Origin;
  // False when the block sits outside the shadow's reach, in which case it is
  // tracked for pointer translation but carries no redzones.
  bool Shadowed;
};

// Device memory normally falls in the same range as any other mapping, but a
// block outside it has no shadow to poison and must be left alone.
static bool CanShadow(uptr Beg, uptr Size) {
  return AddrIsInMem(Beg) && AddrIsInMem(Beg + Size - 1);
}

static InternalMmapVectorNoCtor<DeviceAlloc> Allocs;

// Freed allocations are remembered so a use-after-free can name the region and
// the code that released it. Only metadata is held; the memory itself goes back
// to the runtime immediately.
static constexpr uptr kFreedHistory = 256;
static InternalMmapVectorNoCtor<DeviceAlloc> Freed;
static uptr FreedNext;

static DeviceAlloc *FindAlloc(uptr Ptr) {
  for (uptr I = 0; I < Allocs.size(); ++I) {
    DeviceAlloc &A = Allocs[I];
    if (Ptr >= A.Raw && Ptr < A.Raw + A.Total)
      return &A;
  }
  return nullptr;
}

static void RecordAlloc(const DeviceAlloc &A) { Allocs.push_back(A); }

static void ForgetAlloc(DeviceAlloc *A) {
  uptr I = A - Allocs.data();
  if (I + 1 != Allocs.size())
    Allocs[I] = Allocs.back();
  Allocs.pop_back();
}

static void RememberFreed(const DeviceAlloc &A) {
  if (Freed.size() < kFreedHistory) {
    Freed.push_back(A);
    return;
  }
  Freed[FreedNext] = A;
  FreedNext = (FreedNext + 1) % kFreedHistory;
}

void __asan::RecordDeviceHeap(uptr Base, uptr Size) {
  DeviceAlloc A = {};
  A.Raw = Base;
  A.Total = Size;
  A.Origin = kRegionDevice;
  Lock L(&AsanOffloadMutex);
  RecordAlloc(A);
}

// Fills in the user extent of a device allocation from the header the device
// allocator keeps in the left redzone. Only valid while the block is still
// mapped, so the result is cached before the memory goes back to the runtime.
static void ResolveDeviceBlock(DeviceAlloc &A) {
  A.User = A.Raw + ASAN_OFFLOAD_BLOCK_REDZONE;
  auto *B = reinterpret_cast<__asan_offload_block *>(A.Raw);
  A.Size = B->magic == ASAN_OFFLOAD_BLOCK_MAGIC ? B->size : 0;
}

void __asan::ForgetDeviceHeap(uptr Base) {
  Lock L(&AsanOffloadMutex);
  if (DeviceAlloc *A = FindAlloc(Base)) {
    DeviceAlloc Gone = *A;
    ResolveDeviceBlock(Gone);
    ForgetAlloc(A);
    RememberFreed(Gone);
  }
}

bool __asan::FindOffloadRegion(uptr Addr, OffloadRegion *Out) {
  Lock L(&AsanOffloadMutex);
  DeviceAlloc Hit;
  if (DeviceAlloc *A = FindAlloc(Addr)) {
    Hit = *A;
    if (Hit.Origin == kRegionDevice)
      ResolveDeviceBlock(Hit);
  } else {
    bool Found = false;
    for (uptr I = 0; I < Freed.size(); ++I) {
      if (Addr >= Freed[I].Raw && Addr < Freed[I].Raw + Freed[I].Total) {
        Hit = Freed[I];
        Found = true;
        break;
      }
    }
    if (!Found)
      return false;
  }

  Out->Beg = Hit.User;
  Out->Size = Hit.Size;
  Out->AllocStack = Hit.AllocStack;
  Out->FreeStack = Hit.FreeStack;
  Out->Origin = Hit.Origin;
  return true;
}

static void *WrapperFor(const char *Name) {
#define ASAN_HSA_WRAP(Fn)                                                      \
  if (!internal_strcmp(Name, #Fn))                                             \
    return reinterpret_cast<void *>(Fn);
  ASAN_HSA_WRAPS(ASAN_HSA_WRAP)
#undef ASAN_HSA_WRAP
  return nullptr;
}

static bool FromHsa(void *P) {
  Dl_info Info = {};
  if (!dladdr(P, &Info) || !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, ASAN_HSA_LIBRARY);
}

static void BindRealDlsym();

// Deliberately does not call Initialize(): ASan's own start up resolves its
// interceptors through 'dlsym', so doing so would re-enter initialization and
// hang before the process ever reaches main.
INTERCEPTOR(void *, dlsym, void *Handle, const char *Name) {
  BindRealDlsym();
  if (UNLIKELY(!REAL(dlsym)))
    return nullptr;

  if (Handle == RTLD_NEXT) [[clang::musttail]]
    return REAL(dlsym)(Handle, Name);

  void *Sym = REAL(dlsym)(Handle, Name);
  if (!Sym || !Name)
    return Sym;

  void *Wrapper = WrapperFor(Name);
  if (!Wrapper || !FromHsa(Sym))
    return Sym;
  return Wrapper;
}

static void BindRealDlsym() {
  if (LIKELY(REAL(dlsym)))
    return;
#if SANITIZER_GLIBC
  // The unversioned symbol resolves back to this interceptor, so the real one
  // has to be found through its version.
  static const char *kVers[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5",
                                "GLIBC_2.0"};
  if (dlvsym) {
    for (const char *Ver : kVers) {
      if (void *P = dlvsym(RTLD_NEXT, "dlsym", Ver)) {
        REAL(dlsym) = reinterpret_cast<decltype(REAL(dlsym))>(P);
        return;
      }
    }
  }
#endif
  // Without a versioned lookup there is no way to reach the real 'dlsym'.
  // Interception of HSA through 'dlsym' is lost, but direct linking still
  // works, so this is not fatal.
  VReport(1, "%s: cannot bind dlsym, only linked HSA calls are intercepted\n",
          SanitizerToolName);
}

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  ASAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  Lock L(&HsaLifecycleMutex);
  if (GetHsa().AddRef())
    GetHsa().Init();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  ASAN_HSA_ENTER(hsa_shut_down);

  {
    Lock L(&HsaLifecycleMutex);
    if (GetHsa().DropRef())
      GetHsa().Shutdown();
  }
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char *Options) {
  ASAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status == HSA_STATUS_SUCCESS) {
    {
      Lock L(&AsanOffloadMutex);
      if (GetHsa().Ready())
        GetHsa().RecordExecutable(Executable);
    }
    StartRpc(Executable);
  }
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  ASAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  FlushRpc();
  {
    Lock L(&AsanOffloadMutex);
    GetHsa().ForgetExecutable(Executable);
  }
  return REAL(hsa_executable_destroy)(Executable);
}

// Lays out redzones around a freshly allocated block exactly as the host
// allocator does, so the shadow byte legend and bug naming carry over.
static void PoisonBlock(uptr Raw, uptr Total, uptr User, uptr Size) {
  uptr UserEnd = User + Size;
  uptr EndAlignedDown = RoundDownTo(UserEnd, ASAN_SHADOW_GRANULARITY);
  PoisonShadow(Raw, User - Raw, kAsanHeapLeftRedzoneMagic);
  PoisonShadow(User, EndAlignedDown - User, 0);
  FastPoisonShadowPartialRightRedzone(EndAlignedDown,
                                      UserEnd - EndAlignedDown,
                                      Raw + Total - EndAlignedDown,
                                      kAsanHeapLeftRedzoneMagic);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags,
            void **Ptr) {
  ASAN_HSA_FORWARD(hsa_amd_memory_pool_allocate, Pool, Size, Flags, Ptr);
  bool Redzoned;
  {
    Lock L(&AsanOffloadMutex);
    Redzoned = GetHsa().PoolTakesRedzones(Pool);
  }
  if (!Ptr || !Size || !Redzoned)
    return REAL(hsa_amd_memory_pool_allocate)(Pool, Size, Flags, Ptr);
  if (Size > ~(size_t)0 - 2 * kOffloadRedzone)
    return HSA_STATUS_ERROR;

  uptr User = RoundUpTo(Size, ASAN_SHADOW_GRANULARITY);
  uptr Total = kOffloadRedzone + User + kOffloadRedzone;
  void *Raw = nullptr;
  hsa_status_t Status =
      REAL(hsa_amd_memory_pool_allocate)(Pool, Total, Flags, &Raw);
  if (Status != HSA_STATUS_SUCCESS || !Raw)
    return Status;

  uptr RawAddr = reinterpret_cast<uptr>(Raw);
  uptr UserPtr = RawAddr + kOffloadRedzone;
  bool Shadowed = CanShadow(RawAddr, Total);
  if (Shadowed)
    PoisonBlock(RawAddr, Total, UserPtr, Size);
  else
    VReport(1, "%s: allocation at 0x%zx has no shadow, left unpoisoned\n",
            SanitizerToolName, RawAddr);

  GET_STACK_TRACE_MALLOC;
  DeviceAlloc A = {};
  A.Raw = RawAddr;
  A.Total = Total;
  A.User = UserPtr;
  A.Size = Size;
  A.AllocStack = StackDepotPut(stack);
  A.Origin = kRegionHost;
  A.Shadowed = Shadowed;
  {
    Lock L(&AsanOffloadMutex);
    RecordAlloc(A);
  }
  *Ptr = reinterpret_cast<void *>(UserPtr);
  return Status;
}

// Shared by the two runtime entry points that release pool memory.
static bool ReleaseBlock(void *Ptr, void **Raw, BufferedStackTrace *Stack) {
  DeviceAlloc A;
  {
    Lock L(&AsanOffloadMutex);
    DeviceAlloc *Hit = FindAlloc(reinterpret_cast<uptr>(Ptr));
    if (!Hit || Hit->Origin != kRegionHost)
      return false;
    A = *Hit;
    A.FreeStack = StackDepotPut(*Stack);
    ForgetAlloc(Hit);
    RememberFreed(A);
  }
  if (A.Shadowed)
    PoisonShadow(A.Raw, A.Total, kAsanHeapFreeMagic);
  *Raw = reinterpret_cast<void *>(A.Raw);
  return true;
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void *Ptr) {
  ASAN_HSA_FORWARD(hsa_amd_memory_pool_free, Ptr);
  if (!Ptr)
    return REAL(hsa_amd_memory_pool_free)(Ptr);

  GET_STACK_TRACE_FREE;
  void *Raw = nullptr;
  if (!ReleaseBlock(Ptr, &Raw, &stack))
    return REAL(hsa_amd_memory_pool_free)(Ptr);
  return REAL(hsa_amd_memory_pool_free)(Raw);
}

// The legacy allocation interface frees pool memory too, so a shifted pointer
// reaching it has to be translated back or the runtime rejects it.
INTERCEPTOR(hsa_status_t, hsa_memory_free, void *Ptr) {
  ASAN_HSA_FORWARD(hsa_memory_free, Ptr);
  if (!Ptr)
    return REAL(hsa_memory_free)(Ptr);

  GET_STACK_TRACE_FREE;
  void *Raw = nullptr;
  if (!ReleaseBlock(Ptr, &Raw, &stack))
    return REAL(hsa_memory_free)(Ptr);
  return REAL(hsa_memory_free)(Raw);
}

INTERCEPTOR(hsa_status_t, hsa_amd_agents_allow_access, uint32_t NumAgents,
            const hsa_agent_t *Agents, const uint32_t *Flags, const void *Ptr) {
  ASAN_HSA_FORWARD(hsa_amd_agents_allow_access, NumAgents, Agents, Flags, Ptr);
  {
    Lock L(&AsanOffloadMutex);
    if (DeviceAlloc *A = FindAlloc(reinterpret_cast<uptr>(Ptr)))
      Ptr = reinterpret_cast<const void *>(A->Raw);
  }
  return REAL(hsa_amd_agents_allow_access)(NumAgents, Agents, Flags, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_pointer_info, const void *Ptr,
            hsa_amd_pointer_info_t *Info, void *(*Alloc)(size_t),
            uint32_t *NumAccessible, hsa_agent_t **Accessible) {
  ASAN_HSA_FORWARD(hsa_amd_pointer_info, Ptr, Info, Alloc, NumAccessible,
                   Accessible);
  DeviceAlloc Hit;
  bool Found = false;
  {
    Lock L(&AsanOffloadMutex);
    if (DeviceAlloc *A = FindAlloc(reinterpret_cast<uptr>(Ptr))) {
      Hit = *A;
      Found = true;
    }
  }
  const void *Query = Found ? reinterpret_cast<const void *>(Hit.Raw) : Ptr;
  hsa_status_t Status =
      REAL(hsa_amd_pointer_info)(Query, Info, Alloc, NumAccessible, Accessible);
  if (Status != HSA_STATUS_SUCCESS || !Found || !Info)
    return Status;
  // Report the user extent rather than the block the redzones live in.
  if (Info->size >= offsetof(hsa_amd_pointer_info_t, sizeInBytes) +
                        sizeof(Hit.Size)) {
    Info->agentBaseAddress = reinterpret_cast<void *>(Hit.User);
    Info->hostBaseAddress = reinterpret_cast<void *>(Hit.User);
    Info->sizeInBytes = Hit.Size;
  }
  return Status;
}

// Validates the two ends of a runtime copy. A device pointer can fall outside
// the shadow's reach, where every byte reads as poisoned, so such a range is
// left unchecked rather than reported as a wild access.
static void CheckCopy(const char *Name, void *Dst, const void *Src, uptr Size) {
  if (!Size)
    return;
  AsanInterceptorContext Ctx = {Name};
  CHECK_RANGES_OVERLAP(Name, Dst, Size, Src, Size);
  if (CanShadow(reinterpret_cast<uptr>(Src), Size))
    ASAN_READ_RANGE(&Ctx, Src, Size);
  if (CanShadow(reinterpret_cast<uptr>(Dst), Size))
    ASAN_WRITE_RANGE(&Ctx, Dst, Size);
}

INTERCEPTOR(hsa_status_t, hsa_memory_copy, void *Dst, const void *Src,
            size_t Size) {
  ASAN_HSA_ENTER(hsa_memory_copy);
  CheckCopy("hsa_memory_copy", Dst, Src, Size);
  return REAL(hsa_memory_copy)(Dst, Src, Size);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy, void *Dst,
            hsa_agent_t DstAgent, const void *Src, hsa_agent_t SrcAgent,
            size_t Size, uint32_t NumDep, const hsa_signal_t *Dep,
            hsa_signal_t Done) {
  ASAN_HSA_ENTER(hsa_amd_memory_async_copy);
  CheckCopy("hsa_amd_memory_async_copy", Dst, Src, Size);
  return REAL(hsa_amd_memory_async_copy)(Dst, DstAgent, Src, SrcAgent, Size,
                                         NumDep, Dep, Done);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy_on_engine, void *Dst,
            hsa_agent_t DstAgent, const void *Src, hsa_agent_t SrcAgent,
            size_t Size, uint32_t NumDep, const hsa_signal_t *Dep,
            hsa_signal_t Done, hsa_amd_sdma_engine_id_t Engine, bool ForceSdma) {
  ASAN_HSA_ENTER(hsa_amd_memory_async_copy_on_engine);
  CheckCopy("hsa_amd_memory_async_copy_on_engine", Dst, Src, Size);
  return REAL(hsa_amd_memory_async_copy_on_engine)(
      Dst, DstAgent, Src, SrcAgent, Size, NumDep, Dep, Done, Engine, ForceSdma);
}

extern "C" void __asan_offload_init() { __asan::Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static void (
    *asan_offload_preinit)(void) = __asan_offload_init;
#endif

__attribute__((constructor(0))) static void AsanOffloadDynInit() {
  __asan_offload_init();
}
