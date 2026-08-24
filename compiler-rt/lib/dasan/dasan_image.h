//===-- dasan_image.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Aliased code objects and the globals described from their symbol tables.
// Callers hold DasanMutex.
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

void AliasExecutable(hsa_executable_t Exec);
void UnaliasExecutable(hsa_executable_t Exec);
void UnaliasAll();

// The alias of a loader address, or the address itself.
uptr AliasOf(uptr Addr);
// The loader address of an alias, or the address itself.
uptr LoadAddressOf(uptr Addr);

// The described global that starts at Beg.
bool LookupGlobal(uptr Beg, GlobalInfo* Out);

}  // namespace __dasan

#endif  // DASAN_IMAGE_H
