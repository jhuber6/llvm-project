//===-- dasan_platform.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device VA: reserve a window, create backing, map it, and track who may
// touch it. Callers hold DasanMutex.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_PLATFORM_H
#define DASAN_PLATFORM_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_offload_hsa.h"

namespace __dasan {

using namespace __sanitizer;

// HSA pool handle, zero names the host pool. Allocator uses it as a key and as
// Create's argument.
using DeviceId = u64;

bool PlatformInit();
void PlatformShutdown();
bool PlatformReady();

// The pool each GPU places into, in agent order.
const InternalMmapVectorNoCtor<DeviceId>& PlatformDevices();
DeviceId DeviceForPool(hsa_amd_memory_pool_t Pool);
DeviceId DeviceForAgent(hsa_agent_t Agent);
bool AgentForDevice(DeviceId Device, hsa_agent_t* Out);
u32 PoolFlags(DeviceId Device);

uptr VaGranule();

bool VaReserve(uptr Addr, uptr Size);
void VaRelease(uptr Addr, uptr Size);

bool VaCreate(DeviceId Device, uptr Size, u32 Flags, u64* Handle);
bool VaShare(uptr Addr, uptr Size, u64* Handle, uptr* Offset);
void VaDestroy(u64 Handle);

bool VaMap(uptr Addr, uptr Size, u64 Handle, bool ReadOnly, DeviceId Device,
           uptr Offset = 0);
void VaUnmap(uptr Addr, uptr Size);
bool VaAllow(uptr Addr, uptr Size, DeviceId Device, bool ReadOnly);
bool VaAllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N,
                   bool ReadOnly);
bool VaAccessibleAgents(uptr Addr, void* (*Alloc)(size_t), u32* Num,
                        hsa_agent_t** Out);

bool VaProbe(uptr Addr);
bool VaWrite(uptr Dst, const void* Src, uptr N);
bool VaFill(uptr Dst, uptr N);
void VaPublish(uptr Touched);

}  // namespace __dasan

#endif  // DASAN_PLATFORM_H
