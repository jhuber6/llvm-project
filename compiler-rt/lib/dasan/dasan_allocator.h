//===-- dasan_allocator.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placement into the reserved window.
//
// Not SizeClassAllocator64. The device check loads one word whose address is
// a function of the pointer, so the layout is the ABI in dasan_mapping.h:
//
//   * Classes are powers of two from 64 B. The class id sits in the pointer
//     and the device shifts by (6 + class). SizeClassMap is a different map.
//   * Entries grow down from the top of the region, as that allocator's
//     metadata does, but the VA holds no free array. The command processor
//     stores the word. A host store does not reach the copy the device loads.
//   * Backing is HSA virtual memory, and only for the bytes the object uses.
//   * One window, several devices. A metadata granule belongs to one device,
//     so a region has a free list per device.
//   * ELF is not a size class. A code object is a run of class-0 chunks. The
//     loader maps it there, or an old runtime aliases it through dma-buf.
//     Heap chunks start at the pointer alignment, so class 0 stays that run.
//
// The quarantine is a per-device FIFO of chunk indices. Every caller holds
// DasanMutex, and a chunk has to return to the device that backs it, so
// sanitizer_quarantine.h's per-thread pointer cache does not fit.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_ALLOCATOR_H
#define DASAN_ALLOCATOR_H

#include "dasan_mapping.h"
#include "dasan_platform.h"
#include "sanitizer_common/sanitizer_common.h"

namespace __dasan {

struct AllocInfo {
  uptr ChunkBeg;
  uptr ChunkSize;
  uptr Beg;
  uptr Size;
  PoolId Pool;
  bool Freed;
  u32 AllocStack;
  u32 FreeStack;
  u32 AllocTid;
  u32 FreeTid;
};

class Allocator {
 public:
  void Init();
  void Shutdown();

  bool IsPlaced(uptr Addr) const {
    return Reserved && !Failed && PointerIsMine(Addr);
  }

  // One chunk. Zero means the real runtime keeps the allocation.
  uptr Allocate(PoolId Pool, uptr Size, u32 AllocFlags = 0, u32 Stack = 0,
                u32 Tid = 0);
  bool Deallocate(uptr Addr, u32 Stack = 0, u32 Tid = 0);

  // Contiguous class-0 chunks, no user backing. The caller maps the bytes.
  uptr AllocateSpan(PoolId Pool, uptr Size);
  void DeallocateSpan(uptr Addr, uptr Size);
  // Entries[I] describes the I'th class-0 chunk from Addr. Published as one
  // command-processor store per block; the array grows downward, so the
  // block is reversed to land contiguous.
  void PublishEntries(uptr Addr, const u64* Entries, uptr N);

  bool Describe(uptr Addr, AllocInfo* Out) const;

  // Extends access to the chunk and to the entry the device loads.
  bool AllowAccess(uptr Addr, const hsa_agent_t* Agents, u32 N);

  struct Counters {
    uptr Placed;
    uptr LargerClass;
    uptr Freed;
    uptr TooLarge;
    uptr Exhausted;
    uptr Inactive;
    uptr CodeObjects;
    uptr CodeObjectChunks;
  } Count;

 private:
  struct ChunkState {
    u64 Size;
    u64 Mapped;
    u32 Offset;
    u32 List;
    u32 AllocStack;
    u32 FreeStack;
    u32 AllocTid;
    u32 FreeTid;
    bool Live;
    // Not a chunk. The index was skipped so this device's metadata starts
    // on a granule the previous device does not own.
    bool Gap;
    // One chunk of a code object. The heap must not recycle it.
    bool Span;
  };

  // Free array plus quarantine for one device inside one region.
  // SizeClassAllocator64 keeps a single free array per region, in the VA.
  // Chunks here are backed by one pool, so the array is per device and the
  // indices live in ordinary memory.
  struct FreeList {
    PoolId Pool;
    InternalMmapVectorNoCtor<uptr> Free;
    InternalMmapVectorNoCtor<uptr> Quarantine;
    uptr Head;
    uptr QuarantinedBytes;
  };

  struct Region {
    InternalMmapVectorNoCtor<ChunkState> Chunks;
    InternalMmapVectorNoCtor<FreeList> FreeLists;
    uptr NextChunk;
    uptr MetadataBytes;
    PoolId MetadataPool;
    bool MetadataPoolSet;
  };

  Region Regions[kNumSizeClasses];
  InternalMmapVectorNoCtor<u64> VmemHandles;
  uptr Granule;
  uptr MetaGranule;
  uptr QuarantinedBytes;

  bool Initialized;
  bool Reserved;
  bool Failed;
  bool MetadataOk;

  uptr ChunkBegin(uptr ClassId, uptr ChunkIdx) const {
    return RegionBeg(ClassId) + ChunkIdx * ClassIdToSize(ClassId);
  }
  // Metadata handles use the allocation granule, not the recommended block.
  uptr MetadataBytesFor(uptr Chunks) const {
    return RoundUpTo(Chunks * kMetadataSize, MetaGranule);
  }
  uptr MetadataBegin(uptr ClassId, uptr Bytes) const {
    return RegionBeg(ClassId) + kRegionSize - Bytes;
  }
  // Small classes are mapped a run at a time, like kUserMapSize in
  // SizeClassAllocator64. A larger chunk stays a VA reservation until the
  // allocation maps the prefix it uses.
  bool MapOnPopulate(uptr ClassId) const {
    return ClassIdToSize(ClassId) <= Granule;
  }
  uptr UserMapChunks(uptr ClassId) const;

  // The next device starts on a fresh metadata granule. Entry loads stay on
  // the device that owns the object; the skipped indices are VA only.
  void SkipToFreshGranule(uptr ClassId, PoolId Pool);

  uptr FreeListFor(uptr ClassId, PoolId Pool);
  bool CreateMetadata(PoolId Prefer, uptr Bytes, u64* Handle, PoolId* Home);
  bool Populate(PoolId Pool, uptr ClassId, uptr List, uptr Max, u32 AllocFlags);
  bool Recycle(uptr ClassId, uptr List);
  bool RecycleOne(uptr PreferClass, uptr PreferList);

  bool PublishEntry(uptr ClassId, uptr ChunkIdx, u64 Entry);
  bool EnsureMetadata(PoolId Pool, uptr ClassId, uptr Chunks);
  bool MapUserChunk(PoolId Pool, uptr ClassId, uptr ChunkIdx, uptr Need,
                    u32 AllocFlags);
  bool GetChunk(PoolId Pool, uptr ClassId, uptr* ChunkIdx, u32 AllocFlags);
};

Allocator& GetAllocator();

}  // namespace __dasan

#endif  // DASAN_ALLOCATOR_H
