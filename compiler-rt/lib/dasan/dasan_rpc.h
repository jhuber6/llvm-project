//===-- dasan_rpc.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device report channel. If offload is present, register DASAN_REPORT with it
// and stop. Otherwise one buffer per GPU ordinal and a thread on an HSA
// doorbell.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_RPC_H
#define DASAN_RPC_H

#include "hsa.h"

namespace __dasan {

void StartRpc(hsa_executable_t Exec);
void FlushRpc();
void StopRpc();

}  // namespace __dasan

#endif  // DASAN_RPC_H
