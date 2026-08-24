//===-- dasan_image.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Aliased code objects and the globals described from their symbol tables.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_IMAGE_H
#define DASAN_IMAGE_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __dasan {

using namespace __sanitizer;

struct AliasedImage {
  uptr LoadBase;
  uptr LoadSize;
  uptr Alias;
  u64 Handle;
  void* Bytes;
  uptr BytesSize;
  char Path[128];
};

struct ImageStats {
  uptr Described;
  uptr Shared;
};

struct GlobalInfo {
  uptr Beg;
  uptr Size;
  uptr Reach;
  const char* Name;
};

bool AlreadyAliased(uptr LoadBase);
uptr AliasOf(uptr Addr);

void TrackImage(uptr LoadBase, uptr LoadSize, uptr Alias, u64 Handle,
                const void* Storage, uptr StorageSize, ImageStats* Stats);
void DropAliasedImage(uptr I);

uptr NumAliasedImages();
AliasedImage& AliasedImageAt(uptr I);

bool LookupGlobal(uptr Addr, GlobalInfo* Out);
void ForgetGlobals();
void ForgetGlobals(uptr Beg, uptr Size);

void ForgetImages();

}  // namespace __dasan

#endif  // DASAN_IMAGE_H
