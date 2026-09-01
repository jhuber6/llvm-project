//===-- csan_offload_report.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Print a device race report after symbolizing PCs against loaded images.
//
//===----------------------------------------------------------------------===//

#include "csan_offload.h"

#include "csan_offload_symbolize.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __csan {
namespace {

class Decorator : public SanitizerCommonDecorator {
public:
  const char *Access() { return Blue(); }
  const char *Location() { return Green(); }
};

const char *KindName(u32 Kind) {
  switch (Kind) {
  case CSAN_RACE_UNKNOWN_ORIGIN:
    return "data race of unknown origin";
  case CSAN_RACE_INTRA_WAVE:
    return "intra-wave data race";
  default:
    return "data race";
  }
}

const char *MopDesc(bool First, u32 Type) {
  const bool Write = Type & CSAN_ACCESS_WRITE;
  return First ? (Write ? "Write" : "Read")
               : (Write ? "Previous write" : "Previous read");
}

void PrintFrames(SymbolizedStack *Frames, u64 PC) {
  if (!Frames) {
    Printf("    #0 (%p)\n", (void *)(uptr)PC);
    return;
  }
  const SymbolizedStack *F = SkipInternalFrames(Frames);
  if (!F)
    F = Frames;
  int N = 0;
  for (; F; F = F->next, ++N) {
    InternalScopedString Res;
    StackTracePrinter::GetOrInit()->RenderFrame(
        &Res, common_flags()->stack_trace_format, N, F->info.address, &F->info,
        common_flags()->symbolize_vs_style, common_flags()->strip_path_prefix);
    Printf("%s\n", Res.data());
  }
}

} // namespace

void PrintOffloadReport(const __csan_gpu_race &R) {
  Decorator D;
  Printf("==================\n");
  Printf("%s", D.Warning());
  Printf("WARNING: ConcurrencySanitizer: %s\n", KindName(R.kind));
  Printf("%s", D.Default());

  Printf("%s", D.Access());
  Printf("  %s of size %u at 0x%zx in block (%u,%u,%u) thread (%u,%u,%u) "
         "lane %u:\n",
         MopDesc(true, R.access_type), R.size, (uptr)R.addr, R.block[0],
         R.block[1], R.block[2], R.thread[0], R.thread[1], R.thread[2], R.lane);
  Printf("%s", D.Default());
  SymbolizedStack *This = SymbolizeOffloadPc((uptr)R.pc);
  PrintFrames(This, R.pc);

  Printf("%s", D.Access());
  if (R.peer_pc) {
    Printf("  %s of size %u at 0x%zx:\n", MopDesc(false, R.peer_access_type),
           R.peer_size, (uptr)R.addr);
    Printf("%s", D.Default());
    SymbolizedStack *Peer = SymbolizeOffloadPc((uptr)R.peer_pc);
    PrintFrames(Peer, R.peer_pc);
    if (Peer)
      Peer->ClearAll();
  } else if (R.kind == CSAN_RACE_INTRA_WAVE) {
    Printf("  Previous access by lane %u in the same wave\n", R.peer_lane);
    Printf("%s", D.Default());
  } else {
    Printf("  Previous access of unknown origin\n");
    Printf("%s", D.Default());
  }

  DataInfo Loc;
  if (SymbolizeOffloadData((uptr)R.addr, &Loc)) {
    Printf("%s", D.Location());
    if (Loc.size)
      Printf("  Location is global '%s' of size %zu at 0x%zx\n", Loc.name,
             Loc.size, (uptr)R.addr);
    else
      Printf("  Location is global '%s' at 0x%zx\n", Loc.name, (uptr)R.addr);
    Printf("%s", D.Default());
    Loc.Clear();
  }

  if (This) {
    const SymbolizedStack *User = SkipInternalFrames(This);
    ReportErrorSummary(KindName(R.kind), (User ? User : This)->info);
    This->ClearAll();
  }

  Printf("==================\n");
}

} // namespace __csan
