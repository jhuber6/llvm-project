//===-- asan_offload_report.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Print a device ASan report after symbolizing the faulting PC against the
// loaded device images.
//
//===----------------------------------------------------------------------===//

#include "asan_offload.h"

#include "asan_descriptions.h"
#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_offload_symbolize.h"
#include "asan_report.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __asan {
namespace {

// Mirror of ErrorGeneric's classification: the shadow byte covering the
// faulting address names the bug, so device reports read exactly like host ones.
const char *NameFromShadow(uptr Addr, uptr Size) {
  if (!AddrIsInMem(Addr))
    return "wild-pointer-access";

  u8 *S = reinterpret_cast<u8 *>(MemToShadow(Addr));
  // A wide access may only be caught by the second granule it touches.
  if (*S == 0 && Size > ASAN_SHADOW_GRANULARITY)
    ++S;
  // A partial granule says how far the access got; the byte after it says why
  // the rest is off limits.
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
  case kAsanStackUseAfterScopeMagic:
    return "stack-use-after-scope";
  case kAsanAllocaLeftMagic:
  case kAsanAllocaRightMagic:
    return "dynamic-stack-buffer-overflow";
  default:
    return "unknown-crash";
  }
}

const char *KindName(const __asan_offload_report &R) {
  switch (R.kind) {
  case ASAN_REPORT_INVALID_FREE:
    return "attempting free on address which was not malloc()-ed";
  case ASAN_REPORT_DOUBLE_FREE:
    return "attempting double-free";
  default:
    return NameFromShadow(static_cast<uptr>(R.addr), R.size);
  }
}

void PrintFrames(SymbolizedStack *Frames, u64 PC) {
  if (!Frames) {
    Printf("    #0 0x%zx\n", (uptr)PC);
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

void PrintStackById(u32 Id) {
  if (!Id)
    return;
  StackDepotGet(Id).Print();
}

// Locates the faulting address relative to the allocation it belongs to, in the
// same words the host allocator uses.
void PrintRegion(const __asan_offload_report &R) {
  uptr Addr = static_cast<uptr>(R.addr);
  OffloadRegion Region;
  Decorator D;

  if (FindOffloadRegion(Addr, &Region) && Region.Size) {
    uptr Beg = Region.Beg;
    uptr End = Beg + Region.Size;
    const char *Where = Region.Origin == kRegionDevice ? "device" : "host";
    Printf("%s", D.Location());
    if (Addr < Beg)
      Printf("0x%zx is located %zu bytes before a %zu-byte %s region "
             "[0x%zx,0x%zx)\n",
             Addr, Beg - Addr, Region.Size, Where, Beg, End);
    else if (Addr >= End)
      Printf("0x%zx is located %zu bytes after a %zu-byte %s region "
             "[0x%zx,0x%zx)\n",
             Addr, Addr - End, Region.Size, Where, Beg, End);
    else
      Printf("0x%zx is located %zu bytes inside a %zu-byte %s region "
             "[0x%zx,0x%zx)\n",
             Addr, Addr - Beg, Region.Size, Where, Beg, End);
    Printf("%s", D.Default());

    Printf("%s", D.Allocation());
    if (Region.FreeStack) {
      Printf("freed by host thread here:\n");
      Printf("%s", D.Default());
      PrintStackById(Region.FreeStack);
      Printf("%s", D.Allocation());
      Printf("previously allocated by host thread here:\n");
      Printf("%s", D.Default());
      PrintStackById(Region.AllocStack);
    } else if (Region.AllocStack) {
      Printf("allocated by host thread here:\n");
      Printf("%s", D.Default());
      PrintStackById(Region.AllocStack);
    } else {
      Printf("allocated on the device\n");
      Printf("%s", D.Default());
    }
    return;
  }

  DeviceGlobalInfo Global;
  if (FindOffloadGlobal(Addr, &Global)) {
    const char *Name = Global.Name[0] ? Global.Name : "<unknown>";
    Printf("%s", D.Location());
    if (Addr < Global.Beg + Global.Size)
      Printf("0x%zx is located %zu bytes inside device global '%s' of size "
             "%zu\n",
             Addr, Addr - Global.Beg, Name, Global.Size);
    else
      Printf("0x%zx is located %zu bytes after device global '%s' of size "
             "%zu\n",
             Addr, Addr - Global.Beg - Global.Size, Name, Global.Size);
    Printf("%s", D.Default());
  }
}

} // namespace

void PrintOffloadReport(const __asan_offload_report &R) {
  Decorator D;
  const char *Name = KindName(R);
  uptr Addr = static_cast<uptr>(R.addr);

  Printf(
      "=================================================================\n");
  Printf("%s", D.Error());
  if (R.kind == ASAN_REPORT_ACCESS)
    Report("ERROR: AddressSanitizer: %s on address 0x%zx\n", Name, Addr);
  else
    Report("ERROR: AddressSanitizer: %s: 0x%zx\n", Name, Addr);
  Printf("%s", D.Default());

  Printf("%s", D.Access());
  if (R.kind == ASAN_REPORT_ACCESS)
    Printf("%s of size %u at 0x%zx in block (%u,%u,%u) thread (%u,%u,%u) "
           "lane %u:\n",
           R.access_type & ASAN_ACCESS_WRITE ? "WRITE" : "READ", R.size, Addr,
           R.block[0], R.block[1], R.block[2], R.thread[0], R.thread[1],
           R.thread[2], R.lane);
  else
    Printf("FREE of 0x%zx in block (%u,%u,%u) thread (%u,%u,%u) lane %u:\n",
           Addr, R.block[0], R.block[1], R.block[2], R.thread[0], R.thread[1],
           R.thread[2], R.lane);
  Printf("%s", D.Default());

  SymbolizedStack *Frames = SymbolizeOffloadPc(static_cast<uptr>(R.pc));
  PrintFrames(Frames, R.pc);

  PrintRegion(R);

  if (Frames) {
    const SymbolizedStack *User = SkipInternalFrames(Frames);
    ReportErrorSummary(Name, (User ? User : Frames)->info);
    Frames->ClearAll();
  } else {
    ReportErrorSummary(Name);
  }

  PrintShadowMemoryForAddress(Addr);
  Printf(
      "=================================================================\n");

  if (R.fatal && flags()->halt_on_error)
    Die();
}

} // namespace __asan
