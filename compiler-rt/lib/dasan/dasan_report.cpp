//===-- dasan_report.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Format a device report for the host.
//
//===----------------------------------------------------------------------===//

#include "dasan_report.h"

#include "dasan.h"
#include "dasan_allocator.h"
#include "dasan_flags.h"
#include "dasan_image.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "shared/rpc.h"

using namespace __sanitizer;

namespace __dasan {

namespace {

class Decorator : public SanitizerCommonDecorator {
 public:
  const char* Access() { return Blue(); }
  const char* Location() { return Green(); }
  const char* Allocation() { return Magenta(); }
};
}  // namespace

static void PrintHeader(const char* Bug, uptr Addr, uptr Pc) {
  Decorator D;
  Printf("=================================================================\n");
  Printf("%s", D.Error());
  if (Pc)
    Report("ERROR: %s: %s on address %p at pc %p\n", SanitizerToolName, Bug,
           (void*)Addr, (void*)Pc);
  else
    Report("ERROR: %s: %s on address %p\n", SanitizerToolName, Bug,
           (void*)Addr);
  Printf("%s", D.Default());
}

static void PrintOffset(uptr Addr, uptr Beg, uptr Size) {
  if (Addr < Beg)
    Printf("%p is located %zu bytes before", (void*)Addr, Beg - Addr);
  else if (Addr >= Beg + Size)
    Printf("%p is located %zu bytes after", (void*)Addr, Addr - (Beg + Size));
  else
    Printf("%p is located %zu bytes inside of", (void*)Addr, Addr - Beg);
}

static void PrintLocation(uptr Addr, uptr Beg, uptr Size) {
  Decorator D;
  Printf("%s", D.Location());
  PrintOffset(Addr, Beg, Size);
  Printf(" %zu-byte region [%p,%p)\n", Size, (void*)Beg, (void*)(Beg + Size));
  Printf("%s", D.Default());
}

// Kernels run from the alias, the snapshot is keyed by the loader's address.
static SymbolizedStack* SymbolizeDevicePc(uptr Pc) {
  SymbolizedStack* Frames = Offload::Get().Symbolize(LoadAddressOf(Pc));
  for (SymbolizedStack* F = Frames; F; F = F->next) F->info.address = Pc;
  return Frames;
}

static void PrintDeviceStack(SymbolizedStack* Frames, uptr Pc) {
  if (!Pc)
    return;
  if (!Frames) {
    Printf("    #0 0x%zx\n\n", Pc);
    return;
  }

  InternalScopedString Out;
  uptr N = 0;
  for (SymbolizedStack* F = Frames; F; F = F->next) {
    uptr Was = Out.length();
    StackTracePrinter::GetOrInit()->RenderFrame(
        &Out, common_flags()->stack_trace_format, N++, F->info.address,
        &F->info, common_flags()->symbolize_vs_style,
        common_flags()->strip_path_prefix);
    if (Out.length() != Was)
      Out.Append("\n");
  }
  Out.Append("\n");
  Printf("%s", Out.data());
}

// The trace starts in the HSA interceptor. The next module is whoever called
// it (HIP, the OpenMP plugin). That whole module is the nest under the
// program's call. A direct call from the same module keeps every frame but
// the interceptor.
static uptr CallerNestEnd(const char** Mods, uptr N) {
  if (!N)
    return 0;
  uptr Begin = 1;
  if (Begin < N && Mods[0] && Mods[Begin] && Mods[Begin] != Mods[0]) {
    const char* Wrap = Mods[Begin];
    while (Begin < N && Mods[Begin] == Wrap) ++Begin;
  }
  return Begin;
}

static void PrintUserStack(const StackTrace& Stack) {
  Symbolizer* Sym = Symbolizer::GetOrInit();
  const char* Mods[kStackTraceMax];
  uptr Pcs[kStackTraceMax];
  uptr N = 0;
  for (; N < Stack.size && Stack.trace[N] && N < kStackTraceMax; ++N) {
    Pcs[N] = StackTrace::GetPreviousInstructionPc(Stack.trace[N]);
    Mods[N] = Sym->GetModuleNameForPc(Pcs[N]);
  }
  const uptr Begin = CallerNestEnd(Mods, N);

  InternalScopedString Out;
  uptr Shown = 0;
  StackTracePrinter* Printer = StackTracePrinter::GetOrInit();
  for (uptr I = Begin; I < N; ++I) {
    SymbolizedStack* Frames = Sym->SymbolizePC(Pcs[I]);
    for (SymbolizedStack* F = Frames; F; F = F->next) {
      uptr Was = Out.length();
      Printer->RenderFrame(&Out, common_flags()->stack_trace_format, Shown,
                           Pcs[I], &F->info, common_flags()->symbolize_vs_style,
                           common_flags()->strip_path_prefix);
      if (Out.length() != Was) {
        Out.Append("\n");
        ++Shown;
      }
    }
    if (Frames)
      Frames->ClearAll();
  }
  if (!Shown) {
    Stack.Print();
    return;
  }
  Out.Append("\n");
  Printf("%s", Out.data());
}

static void PrintSite(const char* Verb, u32 Tid, u32 Id) {
  if (!Id)
    return;
  Decorator D;
  Printf("%s%s thread T%u here:%s\n", D.Allocation(), Verb, Tid, D.Default());
  PrintUserStack(StackDepotGet(Id));
}

static void PrintGlobalLocation(uptr Addr, const GlobalInfo& G) {
  Decorator D;
  Printf("%s", D.Location());
  PrintOffset(Addr, G.Beg, G.Size);
  Printf(" global variable '%s' (%p) of size %zu\n",
         G.Name ? G.Name : "<unknown>", (void*)G.Beg, G.Size);
  Printf("%s", D.Default());
}

static bool PrintReport(const __dasan_offload_report& R) {
  const bool IsShared = R.Kind == DASAN_KIND_SHARED;

  GlobalInfo G;
  const bool IsGlobal = !IsShared && LookupGlobal(R.Base, &G);
  AllocInfo A;
  const bool Known =
      !IsShared && !IsGlobal && GetAllocator().Describe(R.Base, &A);
  // The host may free the region before the report is served, so this is the
  // check's view: a freed entry keeps its offset but reads as zero-sized.
  const bool Freed = Known && R.AllocSize == 0;
  uptr Beg = IsGlobal ? G.Beg : Known ? A.Beg : R.Base;
  uptr Size = IsGlobal ? G.Size : Known ? A.Size : R.AllocSize;

  const char* Bug;
  if (Freed)
    Bug = "heap-use-after-free";
  else if (IsShared)
    Bug = R.Addr < Beg ? "shared-buffer-underflow" : "shared-buffer-overflow";
  else if (IsGlobal)
    Bug = R.Addr < Beg ? "global-buffer-underflow" : "global-buffer-overflow";
  else if (!Known && R.AllocSize == 0)
    Bug = "unknown-device-address";
  else if (R.Addr < Beg)
    Bug = "heap-buffer-underflow";
  else
    Bug = "heap-buffer-overflow";

  PrintHeader(Bug, R.Addr, R.Pc);

  Decorator D;
  Printf("%s%s of size %u at %p thread (%u,%u,%u) of block (%u,%u,%u)%s\n",
         D.Access(), R.IsWrite ? "WRITE" : "READ", R.AccessSize, (void*)R.Addr,
         R.Thread[0], R.Thread[1], R.Thread[2], R.Block[0], R.Block[1],
         R.Block[2], D.Default());

  SymbolizedStack* Frames = SymbolizeDevicePc(R.Pc);
  PrintDeviceStack(Frames, R.Pc);

  if (IsGlobal)
    PrintGlobalLocation(R.Addr, G);
  else if (Size)
    PrintLocation(R.Addr, Beg, Size);
  if (IsShared)
    Printf(
        "the region is group memory, which every workgroup has its own copy "
        "of at these addresses\n");
  if (!IsShared && !IsGlobal && Known) {
    if (Freed)
      PrintSite("freed by", A.FreeTid, A.FreeStack);
    PrintSite(Freed ? "previously allocated by" : "allocated by", A.AllocTid,
              A.AllocStack);
  }

  if (Frames) {
    ReportErrorSummary(Bug, Frames->info);
    Frames->ClearAll();
  } else {
    InternalScopedString S;
    S.AppendF("%s (%p)", Bug, (void*)(uptr)R.Addr);
    ReportErrorSummary(S.data());
  }

  const int Want = flags()->halt_on_error;
  const bool Halt = Want < 0 ? !R.Recover : Want != 0;

  static bool Told = false;
  if (Halt && !R.Recover && !Told) {
    Told = true;
    Report(
        "HINT: build with -fsanitize-recover=daddress to report and keep "
        "going\n");
  }
  return Halt;
}

bool handleHostReport(uptr Addr, uptr AccessSize, uptr Base, uptr AllocSize,
                      bool IsWrite, bool Recover, uptr Pc, uptr Bp) {
  BufferedStackTrace Stack;
  Stack.Unwind(Pc, Bp, nullptr, common_flags()->fast_unwind_on_fatal);
  const uptr FaultPc =
      Stack.size ? StackTrace::GetPreviousInstructionPc(Stack.trace[0]) : 0;

  Lock L(&DasanMutex);

  GlobalInfo G;
  const bool IsGlobal = LookupGlobal(Base, &G);
  AllocInfo A;
  const bool Known = !IsGlobal && GetAllocator().Describe(Base, &A);
  // A freed entry keeps its offset but reads as zero-sized.
  const bool Freed = Known && AllocSize == 0;
  uptr Beg = IsGlobal ? G.Beg : Known ? A.Beg : Base;
  uptr Size = IsGlobal ? G.Size : Known ? A.Size : AllocSize;

  const char* Bug;
  if (Freed)
    Bug = "heap-use-after-free";
  else if (IsGlobal)
    Bug = Addr < Beg ? "global-buffer-underflow" : "global-buffer-overflow";
  else if (!Known && AllocSize == 0)
    Bug = "unknown-device-address";
  else if (Addr < Beg)
    Bug = "heap-buffer-underflow";
  else
    Bug = "heap-buffer-overflow";

  PrintHeader(Bug, Addr, FaultPc);

  Decorator D;
  Printf("%s%s of size %zu at %p%s\n", D.Access(), IsWrite ? "WRITE" : "READ",
         AccessSize, (void*)Addr, D.Default());
  Stack.Print();

  if (IsGlobal)
    PrintGlobalLocation(Addr, G);
  else if (Size)
    PrintLocation(Addr, Beg, Size);
  if (!IsGlobal && Known) {
    if (Freed)
      PrintSite("freed by", A.FreeTid, A.FreeStack);
    PrintSite(Freed ? "previously allocated by" : "allocated by", A.AllocTid,
              A.AllocStack);
  }

  SymbolizedStack* Frames = nullptr;
  if (Stack.size)
    Frames = Symbolizer::GetOrInit()->SymbolizePC(FaultPc);
  if (Frames) {
    ReportErrorSummary(Bug, Frames->info);
    Frames->ClearAll();
  } else {
    InternalScopedString S;
    S.AppendF("%s (%p)", Bug, (void*)Addr);
    ReportErrorSummary(S.data());
  }

  const int Want = flags()->halt_on_error;
  const bool Halt = Want < 0 ? !Recover : Want != 0;

  static bool Told = false;
  if (Halt && !Recover && !Told) {
    Told = true;
    Report(
        "HINT: build with -fsanitize-recover=daddress to report and keep "
        "going\n");
  }
  return Halt;
}

u32 HandleDeviceReport(void* PortPtr, u32) {
  auto& Port = *reinterpret_cast<rpc::Server::Port*>(PortPtr);
  if (Port.get_opcode() != SANITIZER_OFFLOAD_DASAN)
    return rpc::RPC_UNHANDLED_OPCODE;

  bool Halt = false;
  Port.recv([&](rpc::Buffer* Buffer, uint32_t) {
    __dasan_offload_report R;
    internal_memcpy(&R, Buffer->data, sizeof(R));
    Lock L(&DasanMutex);
    Halt = PrintReport(R);
  });
  if (Halt)
    Die();
  return rpc::RPC_SUCCESS;
}

void ReportInvalidFree(uptr Addr, u32 Stack, u32 Tid) {
  AllocInfo A;
  bool Known = GetAllocator().Describe(Addr, &A);
  const char* Bug = Known && A.Freed ? "double-free" : "bad-free";
  PrintHeader(Bug, Addr, /*Pc=*/0);
  if (Stack)
    PrintUserStack(StackDepotGet(Stack));
  if (Known)
    PrintLocation(Addr, A.Beg, A.Size);
  if (Known && A.Freed)
    PrintSite("freed by", A.FreeTid, A.FreeStack);
  if (Known)
    PrintSite(A.Freed ? "previously allocated by" : "allocated by", A.AllocTid,
              A.AllocStack);
  InternalScopedString S;
  S.AppendF("%s (%p)", Bug, (void*)Addr);
  ReportErrorSummary(S.data());
}

void PrintStats() {
  if (!flags()->print_stats)
    return;
  Allocator& A = GetAllocator();
  Report(
      "stats: %zu placed (%zu above their own class), %zu freed; passed "
      "through: %zu too large, %zu no room, %zu not instrumented\n",
      A.NumAllocated, A.NumFellForward, A.NumFreed, A.NumPassedTooBig,
      A.NumPassedNoRoom, A.NumPassedInactive);
  Report("stats: %zu code objects aliased over %zu chunks\n", A.NumImages,
         A.NumImageChunks);
}

}  // namespace __dasan
