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
#include "dasan_symbolize.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_report_decorator.h"

using namespace __sanitizer;

namespace __dasan {

namespace {

class Decorator : public SanitizerCommonDecorator {
 public:
  const char* Access() { return Blue(); }
  const char* Location() { return Green(); }
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

static void PrintGlobalLocation(uptr Addr, const GlobalInfo& G) {
  Decorator D;
  Printf("%s", D.Location());
  PrintOffset(Addr, G.Beg, G.Size);
  Printf(" global variable '%s' (%p) of size %zu\n",
         G.Name ? G.Name : "<unknown>", (void*)G.Beg, G.Size);
  Printf("%s", D.Default());
}

bool PrintReport(const dasan_report_t& R) {
  const bool IsShared = R.Kind == DASAN_KIND_SHARED;

  AllocInfo A;
  bool Known = !IsShared && GetAllocator().Describe(R.Addr, &A);
  uptr Beg = Known ? A.Beg : R.Base;
  uptr Size = Known ? A.Size : R.AllocSize;

  GlobalInfo G;
  bool IsGlobal = !IsShared && !Known && !R.Freed && LookupGlobal(R.Addr, &G);
  if (IsGlobal) {
    Beg = G.Beg;
    Size = G.Size;
  }

  const char* Bug;
  if (R.Freed)
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
  if (R.Freed)
    Printf("the region was freed by the host before this access\n");

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

void ReportInvalidFree(uptr Addr) {
  AllocInfo A;
  bool Known = GetAllocator().Describe(Addr, &A);
  const char* Bug = Known && A.Freed ? "double-free" : "bad-free";
  PrintHeader(Bug, Addr, /*Pc=*/0);
  if (Known)
    PrintLocation(Addr, A.Beg, A.Size);
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
