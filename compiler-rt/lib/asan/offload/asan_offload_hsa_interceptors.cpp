//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// HSA interceptors for host-side AddressSanitizer offload support. Memory
/// handed out through HSA gets the same shadow encoding as host heap memory so
/// the host and the device agree on every allocation.
///
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include "asan_flags.h"
#include "asan_interceptors_memintrinsics.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_offload.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_suppressions.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

#if !SANITIZER_LINUX
#  error "Offload ASan reporting is supported on Linux only"
#endif

#if SANITIZER_GLIBC
#  pragma weak dlvsym
#endif

using namespace __sanitizer;
using namespace __asan;

static void* HsaSymbol(const char* Name);

namespace {

Mutex ChunkMutex;
// Live blocks, sorted by base address.
InternalMmapVectorNoCtor<OffloadChunk> Chunks;
// Freed blocks in the order they were freed, starting at QuarantineHead. They
// are held back from HSA up to quarantine_size_mb so a use-after-free finds
// them poisoned, and unpoisoned before release as HSA may reuse the range for
// memory we never see.
InternalMmapVectorNoCtor<OffloadChunk> Quarantine;
uptr QuarantineHead;
uptr QuarantineSize;
uptr HsaRefs;

// Returns the index of the first chunk with a base above Addr.
uptr UpperBound(uptr Addr) {
  uptr Lo = 0, Hi = Chunks.size();
  while (Lo < Hi) {
    uptr Mid = Lo + (Hi - Lo) / 2;
    if (Chunks[Mid].Beg <= Addr)
      Lo = Mid + 1;
    else
      Hi = Mid;
  }
  return Lo;
}

OffloadChunk* FindLive(uptr Addr) {
  uptr I = UpperBound(Addr);
  if (!I)
    return nullptr;
  OffloadChunk& C = Chunks[I - 1];
  return Addr < C.Beg + C.Total ? &C : nullptr;
}

void EraseLive(OffloadChunk* C) {
  uptr I = C - Chunks.data();
  for (; I + 1 < Chunks.size(); ++I) Chunks[I] = Chunks[I + 1];
  Chunks.pop_back();
}

bool CanShadow(uptr Beg, uptr Size) {
  return Size && AddrIsInMem(Beg) && AddrIsInMem(Beg + Size - 1);
}

void Release(const OffloadChunk& C) {
  if (C.Kind == kOffloadChunkDevice) {
    Offload::Get().Deallocate(reinterpret_cast<void*>(C.Beg));
    return;
  }
  if (C.Kind == kOffloadChunkReserve) {
    if (auto Free = reinterpret_cast<decltype(&hsa_amd_vmem_address_free)>(
            HsaSymbol("hsa_amd_vmem_address_free")))
      Free(reinterpret_cast<void*>(C.Beg), C.Total);
    return;
  }
  if (auto Free = reinterpret_cast<decltype(&hsa_amd_memory_pool_free)>(
          HsaSymbol("hsa_amd_memory_pool_free")))
    Free(reinterpret_cast<void*>(C.Beg));
}

// Pops blocks until the quarantine fits in Limit bytes. The caller releases
// them to HSA once the lock is dropped.
void Evict(uptr Limit, InternalMmapVector<OffloadChunk>* Out) {
  while (QuarantineSize > Limit && QuarantineHead < Quarantine.size()) {
    const OffloadChunk& C = Quarantine[QuarantineHead++];
    QuarantineSize -= C.Total;
    PoisonShadow(C.Beg, C.Total, 0);
    Out->push_back(C);
  }
  if (QuarantineHead == Quarantine.size()) {
    Quarantine.clear();
    QuarantineHead = 0;
  } else if (QuarantineHead > Quarantine.size() / 2) {
    uptr N = Quarantine.size() - QuarantineHead;
    for (uptr I = 0; I < N; ++I) Quarantine[I] = Quarantine[QuarantineHead + I];
    Quarantine.resize(N);
    QuarantineHead = 0;
  }
}

void ReleaseAll(const InternalMmapVector<OffloadChunk>& Gone) {
  for (uptr I = 0; I < Gone.size(); ++I) Release(Gone[I]);
}

void Retain() {
  Lock L(&ChunkMutex);
  ++HsaRefs;
}

// Blocks outlive the runtime that owns them, drop their shadow before the
// range can be reused by something else.
void Shutdown() {
  InternalMmapVector<OffloadChunk> Gone;
  {
    Lock L(&ChunkMutex);
    if (!HsaRefs || --HsaRefs)
      return;
    Evict(0, &Gone);
    for (uptr I = 0; I < Chunks.size(); ++I)
      PoisonShadow(Chunks[I].Beg, Chunks[I].Total, 0);
    Chunks.clear();
  }
  ReleaseAll(Gone);
}

}  // namespace

namespace __asan {

uptr OffloadChunkTotal(uptr Size) {
  uptr Page = GetPageSizeCached();
  if (Size > ~uptr(0) - 2 * Page)
    return 0;
  return RoundUpTo(Size + Page, Page);
}

void PoisonOffloadChunk(uptr Beg, uptr Size, uptr Total) {
  uptr End = Beg + Size;
  uptr EndAligned = RoundDownTo(End, ASAN_SHADOW_GRANULARITY);
  PoisonShadow(Beg, EndAligned - Beg, 0);
  FastPoisonShadowPartialRightRedzone(EndAligned, End - EndAligned,
                                      Beg + Total - EndAligned,
                                      kAsanHeapLeftRedzoneMagic);
}

bool TrackOffloadChunk(const OffloadChunk& C) {
  if (!CanShadow(C.Beg, C.Total))
    return false;
  PoisonOffloadChunk(C.Beg, C.Size, C.Total);
  Lock L(&ChunkMutex);
  uptr I = UpperBound(C.Beg);
  Chunks.push_back(C);
  for (uptr J = Chunks.size() - 1; J > I; --J) Chunks[J] = Chunks[J - 1];
  Chunks[I] = C;
  return true;
}

OffloadFreeResult FreeOffloadChunk(uptr Beg, OffloadChunkKind Kind, u64 Pc,
                                   u32 Stack, OffloadChunk* Out) {
  InternalMmapVector<OffloadChunk> Gone;
  {
    Lock L(&ChunkMutex);
    OffloadChunk* C = FindLive(Beg);
    if (!C || C->Beg != Beg) {
      for (uptr I = QuarantineHead; I < Quarantine.size(); ++I)
        if (Quarantine[I].Beg == Beg)
          return OffloadFreeResult::kDoubleFree;
      return OffloadFreeResult::kUntracked;
    }
    if (Kind == kOffloadChunkDevice && C->Kind != kOffloadChunkDevice)
      return OffloadFreeResult::kWrongKind;

    OffloadChunk Freed = *C;
    Freed.FreePc = Pc;
    Freed.FreeStack = Stack;
    EraseLive(C);
    PoisonShadow(Freed.Beg, Freed.Total, kAsanHeapFreeMagic);
    if (Out)
      *Out = Freed;
    Quarantine.push_back(Freed);
    QuarantineSize += Freed.Total;
    Evict((uptr)flags()->quarantine_size_mb << 20, &Gone);
  }
  ReleaseAll(Gone);
  return OffloadFreeResult::kFreed;
}

bool FindOffloadChunk(uptr Addr, OffloadChunk* Out, bool* Freed) {
  Lock L(&ChunkMutex);
  if (OffloadChunk* C = FindLive(Addr)) {
    *Out = *C;
    *Freed = false;
    return true;
  }
  for (uptr I = Quarantine.size(); I > QuarantineHead; --I) {
    const OffloadChunk& C = Quarantine[I - 1];
    if (Addr >= C.Beg && Addr < C.Beg + C.Total) {
      *Out = C;
      *Freed = true;
      return true;
    }
  }
  return false;
}

bool OffloadChunkTotalFor(uptr Beg, uptr Size, uptr* Total) {
  Lock L(&ChunkMutex);
  OffloadChunk* C = FindLive(Beg);
  if (!C || C->Beg != Beg || C->Size != Size)
    return false;
  *Total = C->Total;
  return true;
}

}  // namespace __asan

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;
static StaticSpinMutex HsaMutex;
static void* HsaHandle;

static void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  Offload::Get().RegisterHandler(HandleOffloadRequest);
  Atexit([] { Offload::Get().UntrackImages(); });
  AddDieCallback([] { Offload::Get().UntrackImages(); });
  atomic_store(&Initialized, 1, memory_order_release);
}

static void BindRealDlsym();

#define ASAN_HSA_ENTER(name)                                                   \
  Initialize();                                                                \
  if (UNLIKELY(!REAL(name) || REAL(name) == name)) {                           \
    REAL(name) = reinterpret_cast<decltype(REAL(name))>(HsaSymbol(#name));     \
    if (UNLIKELY(!REAL(name) || REAL(name) == name)) {                         \
      Report("ERROR: %s: cannot find %s in this process\n", SanitizerToolName, \
             #name);                                                           \
      Die();                                                                   \
    }                                                                          \
  }

#define ASAN_HSA_FORWARD(name, ...)      \
  ASAN_HSA_ENTER(name);                  \
  if (UNLIKELY(!Offload::Get().Ready())) \
    return REAL(name)(__VA_ARGS__);

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  ASAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  if (Offload::Get().Init())
    Retain();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  ASAN_HSA_ENTER(hsa_shut_down);

  if (Offload::Get().Ready())
    Shutdown();
  Offload::Get().Shutdown();
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char* Options) {
  ASAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status == HSA_STATUS_SUCCESS)
    Offload::Get().TrackExecutable(Executable);
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  ASAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  Offload::Get().UntrackExecutable(Executable);
  return REAL(hsa_executable_destroy)(Executable);
}

static bool TrackHostChunk(void* Raw, uptr Size, uptr Total,
                           OffloadChunkKind Kind, u32 Stack) {
  OffloadChunk C = {};
  C.Beg = reinterpret_cast<uptr>(Raw);
  C.Size = Size;
  C.Total = Total;
  C.AllocStack = Stack;
  C.Kind = Kind;
  if (TrackOffloadChunk(C))
    return true;
  VReport(1, "%s: HSA allocation at %p has no shadow\n", SanitizerToolName,
          Raw);
  return false;
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags,
            void** Ptr) {
  ASAN_HSA_FORWARD(hsa_amd_memory_pool_allocate, Pool, Size, Flags, Ptr);
  uptr Total = OffloadChunkTotal(Size);
  if (!Ptr || !Size || !Total)
    return REAL(hsa_amd_memory_pool_allocate)(Pool, Size, Flags, Ptr);

  void* Raw = nullptr;
  hsa_status_t Status =
      REAL(hsa_amd_memory_pool_allocate)(Pool, Total, Flags, &Raw);
  if (Status != HSA_STATUS_SUCCESS || !Raw)
    return Status;
  GET_STACK_TRACE_MALLOC;
  TrackHostChunk(Raw, Size, Total, kOffloadChunkPool, StackDepotPut(stack));
  *Ptr = Raw;
  return Status;
}

// Returns false if Ptr is not a block of this kind, which HSA should free.
static bool FreeHostChunk(void* Ptr, OffloadChunkKind Kind, u32 Stack,
                          hsa_status_t* Status) {
  switch (
      FreeOffloadChunk(reinterpret_cast<uptr>(Ptr), Kind, 0, Stack, nullptr)) {
    case OffloadFreeResult::kFreed:
      *Status = HSA_STATUS_SUCCESS;
      return true;
    case OffloadFreeResult::kDoubleFree:
      ReportOffloadFreeError(OffloadFreeResult::kDoubleFree,
                             reinterpret_cast<uptr>(Ptr), Stack, nullptr);
      *Status = HSA_STATUS_ERROR;
      return true;
    default:
      return false;
  }
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void* Ptr) {
  ASAN_HSA_FORWARD(hsa_amd_memory_pool_free, Ptr);
  hsa_status_t Status;
  if (Ptr) {
    GET_STACK_TRACE_FREE;
    if (FreeHostChunk(Ptr, kOffloadChunkPool, StackDepotPut(stack), &Status))
      return Status;
  }
  return REAL(hsa_amd_memory_pool_free)(Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_memory_free, void* Ptr) {
  ASAN_HSA_FORWARD(hsa_memory_free, Ptr);
  hsa_status_t Status;
  if (Ptr) {
    GET_STACK_TRACE_FREE;
    if (FreeHostChunk(Ptr, kOffloadChunkPool, StackDepotPut(stack), &Status))
      return Status;
  }
  return REAL(hsa_memory_free)(Ptr);
}

// Unregistered ranges at no fixed address are system memory for HMM, such as
// managed memory. Other reservations back the virtual memory API, whose users
// map exact sizes into them, and are left alone.
static bool IsHmmReserve(uint64_t Address, uint64_t Flags) {
  return !Address && (Flags & HSA_AMD_VMEM_ADDRESS_NO_REGISTER);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_free, void* Va, size_t Size) {
  ASAN_HSA_FORWARD(hsa_amd_vmem_address_free, Va, Size);
  hsa_status_t Status;
  if (Va) {
    GET_STACK_TRACE_FREE;
    if (FreeHostChunk(Va, kOffloadChunkReserve, StackDepotPut(stack), &Status))
      return Status;
  }
  return REAL(hsa_amd_vmem_address_free)(Va, Size);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_reserve_align, void** Va,
            size_t Size, uint64_t Address, uint64_t Alignment, uint64_t Flags) {
  ASAN_HSA_FORWARD(hsa_amd_vmem_address_reserve_align, Va, Size, Address,
                   Alignment, Flags);
  uptr Total = OffloadChunkTotal(Size);
  if (!Va || !Size || !Total || !IsHmmReserve(Address, Flags))
    return REAL(hsa_amd_vmem_address_reserve_align)(Va, Size, Address,
                                                    Alignment, Flags);
  hsa_status_t Status = REAL(hsa_amd_vmem_address_reserve_align)(
      Va, Total, Address, Alignment, Flags);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  GET_STACK_TRACE_MALLOC;
  if (TrackHostChunk(*Va, Size, Total, kOffloadChunkReserve,
                     StackDepotPut(stack)))
    return Status;
  // The caller frees an untracked range with its own size.
  ASAN_HSA_ENTER(hsa_amd_vmem_address_free);
  REAL(hsa_amd_vmem_address_free)(*Va, Total);
  return REAL(hsa_amd_vmem_address_reserve_align)(Va, Size, Address, Alignment,
                                                  Flags);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_reserve, void** Va, size_t Size,
            uint64_t Address, uint64_t Flags) {
  ASAN_HSA_FORWARD(hsa_amd_vmem_address_reserve, Va, Size, Address, Flags);
  uptr Total = OffloadChunkTotal(Size);
  if (!Va || !Size || !Total || !IsHmmReserve(Address, Flags))
    return REAL(hsa_amd_vmem_address_reserve)(Va, Size, Address, Flags);
  hsa_status_t Status =
      REAL(hsa_amd_vmem_address_reserve)(Va, Total, Address, Flags);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  GET_STACK_TRACE_MALLOC;
  if (TrackHostChunk(*Va, Size, Total, kOffloadChunkReserve,
                     StackDepotPut(stack)))
    return Status;
  ASAN_HSA_ENTER(hsa_amd_vmem_address_free);
  REAL(hsa_amd_vmem_address_free)(*Va, Total);
  return REAL(hsa_amd_vmem_address_reserve)(Va, Size, Address, Flags);
}

// The runtime insists the length matches the whole block it handed out.
INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_create, void* Ptr, size_t Len,
            hsa_amd_ipc_memory_t* Handle) {
  ASAN_HSA_FORWARD(hsa_amd_ipc_memory_create, Ptr, Len, Handle);
  uptr Total;
  if (OffloadChunkTotalFor(reinterpret_cast<uptr>(Ptr), Len, &Total))
    Len = Total;
  return REAL(hsa_amd_ipc_memory_create)(Ptr, Len, Handle);
}

// Both ends of a copy must be addressable. A range the shadow cannot cover
// would read as poisoned, so it is left unchecked.
static void CheckCopy(const char* Name, void* Dst, const void* Src, uptr Size) {
  if (!Size || !AsanInited())
    return;
  AsanInterceptorContext Ctx = {Name};
  if (CanShadow(reinterpret_cast<uptr>(Src), Size))
    ASAN_READ_RANGE(&Ctx, Src, Size);
  if (CanShadow(reinterpret_cast<uptr>(Dst), Size))
    ASAN_WRITE_RANGE(&Ctx, Dst, Size);
}

INTERCEPTOR(hsa_status_t, hsa_memory_copy, void* Dst, const void* Src,
            size_t Size) {
  ASAN_HSA_FORWARD(hsa_memory_copy, Dst, Src, Size);
  CheckCopy("hsa_memory_copy", Dst, Src, Size);
  return REAL(hsa_memory_copy)(Dst, Src, Size);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy, void* Dst,
            hsa_agent_t DstAgent, const void* Src, hsa_agent_t SrcAgent,
            size_t Size, uint32_t NumDep, const hsa_signal_t* Dep,
            hsa_signal_t Done) {
  ASAN_HSA_FORWARD(hsa_amd_memory_async_copy, Dst, DstAgent, Src, SrcAgent,
                   Size, NumDep, Dep, Done);
  CheckCopy("hsa_amd_memory_async_copy", Dst, Src, Size);
  return REAL(hsa_amd_memory_async_copy)(Dst, DstAgent, Src, SrcAgent, Size,
                                         NumDep, Dep, Done);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy_on_engine, void* Dst,
            hsa_agent_t DstAgent, const void* Src, hsa_agent_t SrcAgent,
            size_t Size, uint32_t NumDep, const hsa_signal_t* Dep,
            hsa_signal_t Done, hsa_amd_sdma_engine_id_t Engine,
            bool ForceSdma) {
  ASAN_HSA_FORWARD(hsa_amd_memory_async_copy_on_engine, Dst, DstAgent, Src,
                   SrcAgent, Size, NumDep, Dep, Done, Engine, ForceSdma);
  CheckCopy("hsa_amd_memory_async_copy_on_engine", Dst, Src, Size);
  return REAL(hsa_amd_memory_async_copy_on_engine)(
      Dst, DstAgent, Src, SrcAgent, Size, NumDep, Dep, Done, Engine, ForceSdma);
}

// PPC cannot transparently tail-call an indirect dlsym target for RTLD_NEXT.
#if !SANITIZER_PPC
#  define ASAN_HSA_WRAPS(X)               \
    X(hsa_init)                           \
    X(hsa_shut_down)                      \
    X(hsa_executable_freeze)              \
    X(hsa_executable_destroy)             \
    X(hsa_amd_memory_pool_allocate)       \
    X(hsa_amd_memory_pool_free)           \
    X(hsa_memory_free)                    \
    X(hsa_amd_ipc_memory_create)          \
    X(hsa_amd_vmem_address_reserve)       \
    X(hsa_amd_vmem_address_reserve_align) \
    X(hsa_amd_vmem_address_free)          \
    X(hsa_memory_copy)                    \
    X(hsa_amd_memory_async_copy)          \
    X(hsa_amd_memory_async_copy_on_engine)

static void* WrapperFor(const char* Name) {
#  define ASAN_HSA_WRAP(Fn)          \
    if (!internal_strcmp(Name, #Fn)) \
      return reinterpret_cast<void*>(Fn);
  ASAN_HSA_WRAPS(ASAN_HSA_WRAP)
#  undef ASAN_HSA_WRAP
  return nullptr;
}

static bool FromHsa(void* P) {
  Dl_info Info = {};
  if (!dladdr(P, &Info) || !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, SANITIZER_HSA_LIBRARY);
}

// OpenMP and sometimes HIP access HSA through 'dlsym' so we need to intercept
// it here if we want to reliably override its definitions.
INTERCEPTOR(void*, dlsym, void* Handle, const char* Name) {
  BindRealDlsym();

  // This interceptor interferes with the order of 'RTLD_NEXT'. Force a tail
  // call to bypass this process in the stack.
  if (Handle == RTLD_NEXT) [[clang::musttail]]
    return REAL(dlsym)(Handle, Name);

  void* Sym = REAL(dlsym)(Handle, Name);
  if (!Sym || !Name)
    return Sym;

  void* Wrapper = WrapperFor(Name);
  if (!Wrapper || !FromHsa(Sym))
    return Sym;
  return Wrapper;
}
#else
DEFINE_REAL(void*, dlsym, void*, const char*)
#endif

static void BindRealDlsym() {
  if (LIKELY(REAL(dlsym)))
    return;
#if SANITIZER_GLIBC
  static const char* kVers[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5",
                                "GLIBC_2.0"};
  if (dlvsym) {
    for (const char* Ver : kVers) {
      if (void* P = dlvsym(RTLD_NEXT, "dlsym", Ver)) {
        REAL(dlsym) = reinterpret_cast<decltype(REAL(dlsym))>(P);
        return;
      }
    }
  }
#endif
  Report("ERROR: %s: cannot bind dlsym\n", SanitizerToolName);
  Die();
}

static void* HsaSymbol(const char* Name) {
  BindRealDlsym();
  if (!HsaHandle) {
    SpinMutexLock L(&HsaMutex);
    if (HsaHandle)
      return REAL(dlsym)(HsaHandle, Name);
    constexpr const char* Names[] = {"libhsa-runtime64.so.1",
                                     "libhsa-runtime64.so"};
    for (const char* Name : Names)
      if (void* H = dlopen(Name, RTLD_LAZY | RTLD_NOLOAD))
        HsaHandle = H;
    for (const char* Name : Names)
      if (!HsaHandle)
        HsaHandle = dlopen(Name, RTLD_LAZY | RTLD_LOCAL);
  }
  return HsaHandle ? REAL(dlsym)(HsaHandle, Name) : nullptr;
}

extern "C" void __asan_offload_init() { Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static void (
    *asan_offload_preinit)(void) = __asan_offload_init;
#endif

__attribute__((constructor(0))) static void AsanOffloadDynInit() {
  __asan_offload_init();
}
