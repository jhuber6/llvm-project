//===-- asan_offload_globals.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_GLOBALS_H
#define ASAN_OFFLOAD_GLOBALS_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __asan {

struct DeviceGlobalInfo {
  uptr Beg;
  uptr Size;
  char Name[256];
};

// Poison, or unpoison, the redzones of the globals described by the
// 'asan_globals' section of a device image loaded at 'LoadBase'.
void PoisonDeviceGlobals(uptr LoadBase, const void *Bytes, uptr Size,
                         bool Poison);

// Describe the global whose redzone covers 'Addr'. The descriptors carry the
// size the user declared, which the ELF symbol has already been padded past.
bool FindDeviceGlobal(uptr LoadBase, const void *Bytes, uptr Size, uptr Addr,
                      DeviceGlobalInfo *Out);

} // namespace __asan

#endif // ASAN_OFFLOAD_GLOBALS_H
