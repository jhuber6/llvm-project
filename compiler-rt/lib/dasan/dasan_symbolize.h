//===-- dasan_symbolize.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Map a device PC to a source location.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_SYMBOLIZE_H
#define DASAN_SYMBOLIZE_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __dasan {

using namespace __sanitizer;

SymbolizedStack* SymbolizeDevicePc(uptr Pc);

void PrintDeviceStack(SymbolizedStack* Frames, uptr Pc);

}  // namespace __dasan

#endif  // DASAN_SYMBOLIZE_H
