//===-- dasan_platform.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device VA: reserve a window, create backing, map it. One HSA implementation.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_PLATFORM_H
#define DASAN_PLATFORM_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __dasan {

using namespace __sanitizer;

// HSA coarse pool handle. Allocator uses it as a key and as Create's argument.
using DeviceId = u64;

uptr VaGranule();

bool VaReserve(uptr Addr, uptr Size);
void VaRelease(uptr Addr, uptr Size);

bool VaCreate(DeviceId Device, uptr Size, u32 Flags, u64* Handle);
void VaDestroy(u64 Handle);

bool VaMap(uptr Addr, uptr Size, u64 Handle, bool ReadOnly, DeviceId Device);
void VaUnmap(uptr Addr, uptr Size);
bool VaAllow(uptr Addr, uptr Size, DeviceId Device, bool ReadOnly);

bool VaProbe(uptr Addr);
bool VaWrite(uptr Dst, const void* Src, uptr N);
bool VaFill(uptr Dst, uptr N);
void VaPublish(uptr Touched);

}  // namespace __dasan

#endif  // DASAN_PLATFORM_H
