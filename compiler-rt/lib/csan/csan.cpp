//===-- csan.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host-side CSan init. Device probes live in csan_gpu.cpp; this archive exists
// so the offload interceptors can reuse sanitizer_common.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

using namespace __sanitizer;

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __csan_init() {
  SanitizerToolName = "ConcurrencySanitizer";
}
