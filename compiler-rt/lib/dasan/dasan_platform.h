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

// HSA memory-pool handle. Zero is the host pool. The allocator uses it as
// the key for a device's free lists and as the pool Create backs with.
using PoolId = u64;

bool PlatformInit();
void PlatformShutdown();
bool PlatformReady();

// One placeable pool per GPU, in agent order. Coarse grain wins when an
// agent has several.
const InternalMmapVectorNoCtor<PoolId>& PlacedPools();
// The handle itself when this pool is one we place into, otherwise zero.
PoolId PlaceablePool(hsa_amd_memory_pool_t Pool);
PoolId PoolForAgent(hsa_agent_t Agent);
bool AgentForPool(PoolId Pool, hsa_agent_t* Out);
u32 PoolFlags(PoolId Pool);

uptr VaGranule();
// Allocation granule for metadata handles. VaGranule() is the recommended
// block.
uptr VaMetaGranule();

bool VaReserve(uptr Addr, uptr Size);
void VaRelease(uptr Addr, uptr Size);

bool VaCreate(PoolId Pool, uptr Size, u32 Flags, u64* Handle);
bool VaShare(uptr Addr, uptr Size, u64* Handle, uptr* Offset);
void VaDestroy(u64 Handle);

bool VaMap(uptr Addr, uptr Size, u64 Handle, PoolId Pool, uptr Offset = 0,
           bool Executable = false);
void VaUnmap(uptr Addr, uptr Size);
bool VaAllow(uptr Addr, uptr Size, PoolId Pool);
bool VaAllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N);
bool VaAccessibleAgents(uptr Addr, void* (*Alloc)(size_t), u32* Num,
                        hsa_agent_t** Out);

bool VaProbe(uptr Addr);
// PM4 store of [Addr, Addr + N) through the command processor, confirmed in
// L2 before return. A host store does not update the copy a device load
// reads. Null Src stores zeros.
bool VaPublish(uptr Addr, const void* Src, uptr N);

}  // namespace __dasan

#endif  // DASAN_PLATFORM_H
