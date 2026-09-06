//===-- asan_offload_report.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "asan_offload.h"

#include "asan_report.h"
#include "asan_offload_symbolize.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __asan {

void PrintOffloadReport(const __asan_offload_report &R) {
  if (SymbolizedStack *Frames = SymbolizeOffloadPc(static_cast<uptr>(R.pc))) {
    for (SymbolizedStack *F = Frames; F; F = F->next) {
      Printf("    #0 0x%zx", (uptr)R.pc);
      if (F->info.function)
        Printf(" in %s", F->info.function);
      if (F->info.file)
        Printf(" %s:%d", F->info.file, F->info.line);
      Printf("\n");
    }
  }
  ReportGenericError(static_cast<uptr>(R.pc), 0, 0, static_cast<uptr>(R.addr),
                     R.is_write, static_cast<uptr>(R.size), 0, R.fatal);
}

} // namespace __asan
