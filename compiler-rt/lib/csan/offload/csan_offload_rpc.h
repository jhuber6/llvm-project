//===-- csan_offload_rpc.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CSAN_OFFLOAD_RPC_H
#define CSAN_OFFLOAD_RPC_H

#include "hsa.h"

namespace __csan {

void StartRpc(hsa_executable_t Exec);
void FlushRpc();
void StopRpc();

} // namespace __csan

#endif // CSAN_OFFLOAD_RPC_H
