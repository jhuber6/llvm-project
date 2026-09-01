//===-- csan_offload_symbolize.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CSAN_OFFLOAD_SYMBOLIZE_H
#define CSAN_OFFLOAD_SYMBOLIZE_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __csan {

void TrackDeviceImage(__sanitizer::uptr LoadBase, __sanitizer::uptr LoadSize,
                      const void *Storage, __sanitizer::uptr StorageSize);
void ForgetDeviceImage(__sanitizer::uptr LoadBase);
void ForgetDeviceImages();

__sanitizer::SymbolizedStack *SymbolizeOffloadPc(__sanitizer::uptr PC);
bool SymbolizeOffloadData(__sanitizer::uptr Addr, __sanitizer::DataInfo *Info);

} // namespace __csan

#endif // CSAN_OFFLOAD_SYMBOLIZE_H
