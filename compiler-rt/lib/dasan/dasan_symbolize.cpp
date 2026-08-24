//===-- dasan_symbolize.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dasan_symbolize.h"

#include "dasan_image.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __dasan {

static bool Locate(uptr Pc, AliasedImage** Out, uptr* Offset) {
  for (uptr I = 0; I < NumAliasedImages(); ++I) {
    AliasedImage& Img = AliasedImageAt(I);
    if (Pc >= Img.Alias && Pc < Img.Alias + Img.LoadSize) {
      *Out = &Img;
      *Offset = Pc - Img.Alias;
      return true;
    }
    if (Pc >= Img.LoadBase && Pc < Img.LoadBase + Img.LoadSize) {
      *Out = &Img;
      *Offset = Pc - Img.LoadBase;
      return true;
    }
  }
  return false;
}

static const char* PathFor(AliasedImage& Img) {
  if (Img.Path[0])
    return Img.Path;
  if (!Img.Bytes)
    return nullptr;

  const char* Tmp = GetEnv("TMPDIR");
  char Binary[256];
  const char* Name = "dasan";
  if (ReadBinaryNameCached(Binary, sizeof(Binary)))
    Name = StripModuleName(Binary);
  internal_snprintf(Img.Path, sizeof(Img.Path), "%s/%s.%d.%zx.elf",
                    Tmp ? Tmp : "/tmp", Name, (int)internal_getpid(),
                    Img.Alias);

  fd_t Fd = OpenFile(Img.Path, WrOnly);
  bool Ok = Fd != kInvalidFd && WriteToFile(Fd, Img.Bytes, Img.BytesSize);
  if (Fd != kInvalidFd)
    CloseFile(Fd);
  if (!Ok) {
    VReport(1, "%s: could not write %s; reports will not be symbolized\n",
            SanitizerToolName, Img.Path);
    internal_unlink(Img.Path);
    Img.Path[0] = '\0';
    return nullptr;
  }
  return Img.Path;
}

SymbolizedStack* SymbolizeDevicePc(uptr Pc) {
  AliasedImage* Img = nullptr;
  uptr Offset = 0;
  if (!Pc || !Locate(Pc, &Img, &Offset))
    return nullptr;
  const char* Path = PathFor(*Img);
  if (!Path)
    return nullptr;

  SymbolizedStack* Frames = Symbolizer::GetOrInit()->SymbolizeModuleOffset(
      Path, Offset ? Offset - 1 : Offset);

  for (SymbolizedStack* F = Frames; F; F = F->next) F->info.address = Pc;
  return Frames;
}

void PrintDeviceStack(SymbolizedStack* Frames, uptr Pc) {
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

}  // namespace __dasan
