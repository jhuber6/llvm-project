//===-- csan_offload.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host offload reporting runtime for CSan.
//
//===----------------------------------------------------------------------===//

#ifndef CSAN_OFFLOAD_H
#define CSAN_OFFLOAD_H

#include "csan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __csan {

void Initialize();
void PrintOffloadReport(const __csan_gpu_race &R);

// Nest only after RpcMutex, the reverse deadlocks the report thread.
extern __sanitizer::Mutex CsanOffloadMutex;

} // namespace __csan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __csan_offload_init();
}

#endif // CSAN_OFFLOAD_H
