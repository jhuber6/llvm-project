//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Host-side handling of device AddressSanitizer requests: reports, and the
/// heap behind device 'malloc'.
///
//===----------------------------------------------------------------------===//

#include "asan_descriptions.h"
#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_offload.h"
#include "asan_report.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "shared/rpc.h"

using namespace __sanitizer;

namespace __asan {
namespace {

Mutex ReportMutex;
InternalMmapVectorNoCtor<u64> ReportedPcs;

// Recoverable reports are printed once per faulting instruction.
bool AlreadyReported(u64 Pc) {
  Lock L(&ReportMutex);
  for (uptr I = 0; I < ReportedPcs.size(); ++I)
    if (ReportedPcs[I] == Pc)
      return true;
  ReportedPcs.push_back(Pc);
  return false;
}

// The shadow byte covering the faulting address names the bug, mirroring
// ErrorGeneric so device reports read like host ones.
const char* BugFromShadow(uptr Addr, uptr Size) {
  if (!AddrIsInMem(Addr))
    return "wild-addr";
  u8* S = reinterpret_cast<u8*>(MemToShadow(Addr));
  if (*S == 0 && Size > ASAN_SHADOW_GRANULARITY)
    ++S;
  if (*S > 0 && *S < 128 && S[1] >= 128)
    ++S;
  switch (*S) {
    case kAsanHeapLeftRedzoneMagic:
    case kAsanArrayCookieMagic:
      return "heap-buffer-overflow";
    case kAsanHeapFreeMagic:
      return "heap-use-after-free";
    case kAsanGlobalRedzoneMagic:
      return "global-buffer-overflow";
    case kAsanUserPoisonedMemoryMagic:
      return "use-after-poison";
    default:
      return "unknown-crash";
  }
}

// The device symbol's size includes the redzone, the shadow keeps the original.
uptr GlobalSize(uptr Beg, uptr SymbolSize) {
  if (!SymbolSize || !AddrIsInMem(Beg) || !AddrIsInMem(Beg + SymbolSize - 1))
    return SymbolSize;
  uptr Size = 0;
  for (u8* S = reinterpret_cast<u8*>(MemToShadow(Beg)); Size < SymbolSize;
       ++S) {
    if (*S == 0) {
      Size += ASAN_SHADOW_GRANULARITY;
      continue;
    }
    if (*S < ASAN_SHADOW_GRANULARITY)
      Size += *S;
    break;
  }
  return Min(Size, SymbolSize);
}

void PrintFrames(SymbolizedStack* Frames, u64 Pc) {
  if (!Frames) {
    Printf("    #0 %p\n", reinterpret_cast<void*>(Pc));
    return;
  }
  const SymbolizedStack* F = SkipInternalFrames(Frames);
  if (!F)
    F = Frames;
  for (int N = 0; F; F = F->next, ++N) {
    InternalScopedString Res;
    StackTracePrinter::GetOrInit()->RenderFrame(
        &Res, common_flags()->stack_trace_format, N, F->info.address, &F->info,
        common_flags()->symbolize_vs_style, common_flags()->strip_path_prefix);
    Printf("%s\n", Res.data());
  }
}

void PrintDevicePc(u64 Pc) {
  SymbolizedStack* Frames = Offload::Get().Symbolize(Pc);
  PrintFrames(Frames, Pc);
  if (Frames)
    Frames->ClearAll();
  Printf("\n");
}

void PrintHistory(const char* What, u32 Stack, u64 Pc) {
  Decorator D;
  Printf("%s", D.Allocation());
  if (Stack) {
    Printf("%s by host thread here:\n", What);
    Printf("%s", D.Default());
    StackDepotGet(Stack).Print();
    Printf("\n");
  } else {
    Printf("%s by device here:\n", What);
    Printf("%s", D.Default());
    PrintDevicePc(Pc);
  }
}

void DescribeChunk(uptr Addr, const OffloadChunk& C, bool Freed) {
  // AMDGPUSwLowerLDS allocates each work-group's LDS at kernel entry, where
  // the return address is zero.
  const bool Lds = C.Kind == kOffloadChunkDevice && !C.AllocPc;
  const char* What = Lds ? "LDS block of a work-group" : "region";
  Decorator D;
  uptr End = C.Beg + C.Size;
  Printf("%s", D.Location());
  if (Addr < End)
    Printf("%p is located %zu bytes inside of %zu-byte %s [%p,%p)\n",
           reinterpret_cast<void*>(Addr), Addr - C.Beg, C.Size, What,
           reinterpret_cast<void*>(C.Beg), reinterpret_cast<void*>(End));
  else
    Printf("%p is located %zu bytes after %zu-byte %s [%p,%p)\n",
           reinterpret_cast<void*>(Addr), Addr - End, C.Size, What,
           reinterpret_cast<void*>(C.Beg), reinterpret_cast<void*>(End));
  Printf("%s", D.Default());
  if (Lds) {
    if (Freed)
      Printf("released when its kernel exited\n");
    Printf("\n");
    return;
  }
  if (Freed) {
    PrintHistory("freed", C.FreeStack, C.FreePc);
    PrintHistory("previously allocated", C.AllocStack, C.AllocPc);
  } else {
    PrintHistory("allocated", C.AllocStack, C.AllocPc);
  }
}

void DescribeAddress(uptr Addr, uptr Size, const char* Bug) {
  OffloadChunk C;
  bool Freed;
  if (FindOffloadChunk(Addr, &C, &Freed)) {
    DescribeChunk(Addr, C, Freed);
    return;
  }
  {
    ThreadRegistryLock L(&asanThreadRegistry());
    if (DescribeAddressIfHeap(Addr, Size))
      return;
  }

  DataInfo Info;
  if (!Offload::Get().SymbolizeData(Addr, &Info)) {
    DescribeAddressIfGlobal(Addr, Size, Bug);
    return;
  }
  Info.size = GlobalSize(Info.start, Info.size);
  Decorator D;
  Printf("%s", D.Location());
  if (Addr < Info.start + Info.size)
    Printf(
        "%p is located %zu bytes inside of global variable '%s' of size "
        "%zu\n",
        reinterpret_cast<void*>(Addr), Addr - Info.start, Info.name, Info.size);
  else
    Printf("%p is located %zu bytes after global variable '%s' of size %zu\n",
           reinterpret_cast<void*>(Addr), Addr - Info.start - Info.size,
           Info.name, Info.size);
  Printf("%s", D.Default());
  Info.Clear();
}

void PrintDeviceThread(const __asan_offload_packet& P) {
  Printf("thread (%u,%u,%u) block (%u,%u,%u) lane %u", P.thread[0], P.thread[1],
         P.thread[2], P.block[0], P.block[1], P.block[2], P.lane);
  u32 Device;
  if (Offload::Get().DeviceForPC(P.pc, &Device))
    Printf(" on GPU %u", Device);
}

void Summarize(const char* Bug, u64 Pc) {
  SymbolizedStack* Frames = Offload::Get().Symbolize(Pc);
  if (Frames) {
    const SymbolizedStack* User = SkipInternalFrames(Frames);
    ReportErrorSummary(Bug, (User ? User : Frames)->info);
    Frames->ClearAll();
  } else {
    ReportErrorSummary(Bug);
  }
}

void PrintLanes(const __asan_offload_packet* Wave, u64 Lanes) {
  Printf("Lanes and accessed addresses:\n");
  for (u32 Id = 0, N = 0; Id < 64; ++Id) {
    if (!(Lanes & (1ull << Id)))
      continue;
    Printf("%s%02u : %p", N % 4 ? "  " : "    ", Id,
           reinterpret_cast<void*>(Wave[Id].addr));
    if (++N % 4 == 0 || !(Lanes >> Id >> 1))
      Printf("\n");
  }
  Printf("\n");
}

// Reports every faulting lane of the wave, described from the first.
void ReportAccess(const __asan_offload_packet* Wave, u64 Lanes) {
  const __asan_offload_packet& P = Wave[__builtin_ctzll(Lanes)];
  if (!P.fatal && AlreadyReported(P.pc))
    return;

  uptr Addr = static_cast<uptr>(P.addr);
  const char* Bug = BugFromShadow(Addr, P.size);
  Decorator D;
  Printf(
      "================================================================="
      "\n");
  Printf("%s", D.Error());
  Report("ERROR: AddressSanitizer: %s on address %p at pc %p\n", Bug,
         reinterpret_cast<void*>(Addr), reinterpret_cast<void*>(P.pc));
  Printf("%s", D.Default());
  Printf("%s", D.Access());
  Printf("%s of size %zu at %p by device ", P.is_write ? "WRITE" : "READ",
         static_cast<uptr>(P.size), reinterpret_cast<void*>(Addr));
  PrintDeviceThread(P);
  Printf(":\n");
  Printf("%s", D.Default());
  PrintDevicePc(P.pc);
  if (Lanes & (Lanes - 1))
    PrintLanes(Wave, Lanes);
  DescribeAddress(Addr, P.size, Bug);
  Summarize(Bug, P.pc);
  PrintShadowMemoryForAddress(Addr);
  Printf(
      "================================================================="
      "\n");
  if (P.fatal || flags()->halt_on_error)
    Die();
}

u64 ServeMalloc(const __asan_offload_packet& P) {
  uptr Size = P.size ? static_cast<uptr>(P.size) : 1;
  uptr Total = OffloadChunkTotal(Size);
  void* Ptr = nullptr;
  if (!Total || !Offload::Get().AllocateShared(Total, &Ptr) || !Ptr)
    return 0;
  OffloadChunk C = {};
  C.Beg = reinterpret_cast<uptr>(Ptr);
  C.Size = Size;
  C.Total = Total;
  C.AllocPc = P.pc;
  C.Kind = kOffloadChunkDevice;
  if (!TrackOffloadChunk(C)) {
    Offload::Get().Deallocate(Ptr);
    return 0;
  }
  return C.Beg;
}

u64 ServeFree(const __asan_offload_packet& P) {
  OffloadFreeResult R = FreeOffloadChunk(static_cast<uptr>(P.addr),
                                         kOffloadChunkDevice, P.pc, 0, nullptr);
  if (R == OffloadFreeResult::kFreed)
    return 0;
  ReportOffloadFreeError(R, static_cast<uptr>(P.addr), 0, &P);
  return 1;
}

u64 Serve(const __asan_offload_packet& P) {
  switch (P.op) {
    case ASAN_OFFLOAD_MALLOC:
      return ServeMalloc(P);
    case ASAN_OFFLOAD_FREE:
      return ServeFree(P);
    default:
      VReport(1, "%s: unknown device request %u\n", SanitizerToolName, P.op);
      return 0;
  }
}

}  // namespace

void ReportOffloadFreeError(OffloadFreeResult R, uptr Addr, u32 Stack,
                            const __asan_offload_packet* Pkt) {
  const bool Double = R == OffloadFreeResult::kDoubleFree;
  const char* Bug = Double ? "double-free" : "bad-free";
  Decorator D;
  Printf(
      "================================================================="
      "\n");
  Printf("%s", D.Error());
  if (Double)
    Report("ERROR: AddressSanitizer: attempting double-free on %p",
           reinterpret_cast<void*>(Addr));
  else
    Report(
        "ERROR: AddressSanitizer: attempting free on address which was not "
        "malloc()-ed: %p",
        reinterpret_cast<void*>(Addr));
  Printf("%s", D.Default());
  if (Pkt) {
    Printf(" by device ");
    PrintDeviceThread(*Pkt);
    Printf(":\n");
    PrintDevicePc(Pkt->pc);
  } else {
    Printf(" by host thread:\n");
    StackDepotGet(Stack).Print();
    Printf("\n");
  }
  OffloadChunk C;
  bool Freed;
  if (FindOffloadChunk(Addr, &C, &Freed))
    DescribeChunk(Addr, C, Freed);
  if (Pkt) {
    Summarize(Bug, Pkt->pc);
  } else {
    StackTrace Trace = StackDepotGet(Stack);
    ReportErrorSummary(Bug, &Trace);
  }
  Printf(
      "================================================================="
      "\n");
  if (flags()->halt_on_error)
    Die();
}

u32 HandleOffloadRequest(void* PortPtr, u32) {
  auto& Port = *reinterpret_cast<rpc::Server::Port*>(PortPtr);
  if (Port.get_opcode() != SANITIZER_OFFLOAD_ASAN)
    return rpc::RPC_UNHANDLED_OPCODE;

  __asan_offload_packet Wave[64];
  u64 Reports = 0;
  Port.recv([&](rpc::Buffer* Buffer, u32 Id) {
    internal_memcpy(&Wave[Id], Buffer->data, sizeof(Wave[Id]));
    if (Wave[Id].op == ASAN_OFFLOAD_REPORT)
      Reports |= 1ull << Id;
  });
  if (Reports)
    ReportAccess(Wave, Reports);
  Port.send([&](rpc::Buffer* Buffer, u32 Id) {
    Buffer->data[0] = Reports & (1ull << Id) ? 0 : Serve(Wave[Id]);
  });
  return rpc::RPC_SUCCESS;
}

}  // namespace __asan
