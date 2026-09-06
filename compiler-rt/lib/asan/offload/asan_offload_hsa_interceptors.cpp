//===-- asan_offload_hsa_interceptors.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  X(hsa_amd_agents_allow_access)                                               \
  X(hsa_amd_pointer_info)                                                      \
  X(hsa_memory_copy)                                                           \
  X(hsa_amd_memory_async_copy)

static constexpr uptr kOffloadRedzone = 32;

struct DeviceAlloc {
  uptr Raw;
  uptr User;
  uptr Size;
  uptr Pc;
};

static InternalMmapVectorNoCtor<DeviceAlloc> Allocs;

static DeviceAlloc *FindAlloc(uptr Ptr) {
  for (uptr I = 0; I < Allocs.size(); ++I) {
    DeviceAlloc &A = Allocs[I];
    uptr End = A.User + RoundUpTo(A.Size, 8) + kOffloadRedzone;
    if (Ptr >= A.Raw && Ptr < End)
      return &A;
  }
  return nullptr;
}

static void RecordAlloc(uptr Raw, uptr User, uptr Size, uptr Pc) {
  DeviceAlloc A = {Raw, User, Size, Pc};
  Allocs.push_back(A);
}

static void ForgetAlloc(DeviceAlloc *A) {
  uptr I = A - Allocs.data();
  if (I + 1 != Allocs.size())
    Allocs[I] = Allocs.back();
  Allocs.pop_back();
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

INTERCEPTOR(void *, dlsym, void *Handle, const char *Name) {
  Initialize();
  BindRealDlsym();

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
  Report("ERROR: %s: cannot bind dlsym\n", SanitizerToolName);
  Die();
}

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  ASAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  Lock L(&AsanOffloadMutex);
  if (GetHsa().AddRef())
    GetHsa().Init();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  ASAN_HSA_ENTER(hsa_shut_down);

  bool Last;
  {
    Lock L(&AsanOffloadMutex);
    Last = GetHsa().DropRef();
  }
  if (Last)
    GetHsa().Shutdown();
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

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags,
            void **Ptr) {
  ASAN_HSA_ENTER(hsa_amd_memory_pool_allocate);
  AsanInitFromRtl();
  if (!Ptr)
    return REAL(hsa_amd_memory_pool_allocate)(Pool, Size, Flags, Ptr);
  if (Size > ~(size_t)0 - 2 * kOffloadRedzone)
    return HSA_STATUS_ERROR;

  uptr User = RoundUpTo(Size, 8);
  uptr Total = kOffloadRedzone + User + kOffloadRedzone;
  void *Raw = nullptr;
  hsa_status_t Status =
      REAL(hsa_amd_memory_pool_allocate)(Pool, Total, Flags, &Raw);
  if (Status != HSA_STATUS_SUCCESS || !Raw)
    return Status;

  uptr UserPtr = reinterpret_cast<uptr>(Raw) + kOffloadRedzone;
  PoisonShadow(reinterpret_cast<uptr>(Raw), kOffloadRedzone,
               kAsanHeapLeftRedzoneMagic);
  PoisonShadow(UserPtr, User, 0);
  PoisonShadow(UserPtr + User, kOffloadRedzone, kAsanHeapLeftRedzoneMagic);
  if (User != Size)
    PoisonShadow(UserPtr + Size, User - Size, kAsanHeapLeftRedzoneMagic);
  {
    Lock L(&AsanOffloadMutex);
    RecordAlloc(reinterpret_cast<uptr>(Raw), UserPtr, Size,
                GET_CALLER_PC());
  }
  *Ptr = reinterpret_cast<void *>(UserPtr);
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void *Ptr) {
  ASAN_HSA_ENTER(hsa_amd_memory_pool_free);
  AsanInitFromRtl();
  if (!Ptr)
    return REAL(hsa_amd_memory_pool_free)(Ptr);

  DeviceAlloc A;
  bool Found = false;
  {
    Lock L(&AsanOffloadMutex);
    if (DeviceAlloc *Hit = FindAlloc(reinterpret_cast<uptr>(Ptr))) {
      A = *Hit;
      ForgetAlloc(Hit);
      Found = true;
    }
  }
  if (!Found)
    return REAL(hsa_amd_memory_pool_free)(Ptr);

  uptr User = RoundUpTo(A.Size, 8);
  PoisonShadow(A.Raw, kOffloadRedzone + User + kOffloadRedzone,
               kAsanHeapFreeMagic);
  return REAL(hsa_amd_memory_pool_free)(reinterpret_cast<void *>(A.Raw));
}

INTERCEPTOR(hsa_status_t, hsa_amd_agents_allow_access, uint32_t NumAgents,
            const hsa_agent_t *Agents, const uint32_t *Flags, const void *Ptr) {
  ASAN_HSA_ENTER(hsa_amd_agents_allow_access);
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
  ASAN_HSA_ENTER(hsa_amd_pointer_info);
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
  if (Info->size >= offsetof(hsa_amd_pointer_info_t, sizeInBytes) +
                        sizeof(Hit.Size)) {
    Info->agentBaseAddress = reinterpret_cast<void *>(Hit.User);
    Info->hostBaseAddress = reinterpret_cast<void *>(Hit.User);
    Info->sizeInBytes = Hit.Size;
  }
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_memory_copy, void *Dst, const void *Src,
            size_t Size) {
  ASAN_HSA_ENTER(hsa_memory_copy);
  AsanInitFromRtl();
  AsanInterceptorContext Ctx = {"hsa_memory_copy"};
  ASAN_READ_RANGE(&Ctx, Src, Size);
  ASAN_WRITE_RANGE(&Ctx, Dst, Size);
  return REAL(hsa_memory_copy)(Dst, Src, Size);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy, void *Dst,
            hsa_agent_t DstAgent, const void *Src, hsa_agent_t SrcAgent,
            size_t Size, uint32_t NumDep, const hsa_signal_t *Dep,
            hsa_signal_t Done) {
  ASAN_HSA_ENTER(hsa_amd_memory_async_copy);
  AsanInitFromRtl();
  AsanInterceptorContext Ctx = {"hsa_amd_memory_async_copy"};
  ASAN_READ_RANGE(&Ctx, Src, Size);
  ASAN_WRITE_RANGE(&Ctx, Dst, Size);
  return REAL(hsa_amd_memory_async_copy)(Dst, DstAgent, Src, SrcAgent, Size,
                                         NumDep, Dep, Done);
}

extern "C" void __asan_offload_init() { __asan::Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static void (
    *asan_offload_preinit)(void) = __asan_offload_init;
#endif

__attribute__((constructor(0))) static void AsanOffloadDynInit() {
  __asan_offload_init();
}
