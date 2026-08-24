//===-- dasan_image.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code objects in the tracked window, and the globals named from their ELF
// symbol tables. Callers hold DasanMutex.
//
// A loader that can load at an address maps the image straight into class 0.
// Freeze then publishes one entry per global. An older loader keeps the
// image; when a dma-buf alias of it is still executable, freeze maps that
// alias into class 0 and symbol queries are answered from there. Past the
// extension that dropped execute on the alias, the globals stay unchecked.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_IMAGE_H
#define DASAN_IMAGE_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_offload_hsa.h"

namespace __dasan {

using namespace __sanitizer;

struct GlobalInfo {
  uptr Beg;
  uptr Size;
  const char* Name;
};

// The image is already loaded at Addr. MapSize is the mapping.
void NotePlacedCodeObject(uptr Addr, uptr LoadSize, uptr MapSize, u64 Handle);

// Executable class-0 mapping large enough for LoadSize. Empty on failure.
bool ReserveExecutable(hsa_agent_t Agent, uptr LoadSize, uptr* Addr, uptr* Span,
                       u64* Handle);
void ReleaseExecutable(uptr Addr, uptr Span, u64 Handle);

void TrackExecutable(hsa_executable_t Exec);
void UntrackExecutable(hsa_executable_t Exec);
void UntrackAll();

// Address in the tracked window, or Addr when this runtime did not place it.
uptr TrackedAddress(uptr Addr);
// Loader address of a pointer in the window, or Addr when it already is one.
// The device snapshot is keyed by the loader's address.
uptr LoaderAddress(uptr Addr);

bool LookupGlobal(uptr Beg, GlobalInfo* Out);

}  // namespace __dasan

#endif  // DASAN_IMAGE_H
