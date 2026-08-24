//===-- dasan_allocator.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sub-allocates the reserved VA window. A live pointer's class and chunk index
// decode the metadata word that holds its bounds.
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
  DeviceId Device;
  bool Freed;
};

class Allocator {
 public:
  void Init();
  void Shutdown();

  bool IsPlaced(uptr Addr) const {
    return Reserved && !Broken && PointerIsMine(Addr);
  }

  uptr Allocate(DeviceId Device, uptr Size, u32 AllocFlags = 0);

  // Contiguous ClassId chunks, no backing. Images use class 0; heap never
  // chooses it (alignment bump is larger than a class-0 chunk).
  uptr Allocate(DeviceId Device, uptr Size, uptr ClassId);
  void Deallocate(uptr Addr, uptr Size, uptr ClassId);

  bool Deallocate(uptr Addr);

  bool Describe(uptr Addr, AllocInfo* Out) const;

  void SetImageEntry(uptr Addr, u64 Entry);
  void PublishEntries();

  uptr NumImages;
  uptr NumImageChunks;
  uptr NumAllocated;
  uptr NumFreed;
  uptr NumPassedTooBig;
  uptr NumPassedNoRoom;
  uptr NumPassedInactive;
  uptr NumFellForward;

 private:
  struct ChunkState {
    u64 Size;
    u32 Offset;
    bool Live;
    u64 Backed;
    u32 Slot;
  };

  struct Slot {
    DeviceId Device;
    InternalMmapVectorNoCtor<u32> Free;
    InternalMmapVectorNoCtor<u32> Quarantine;
    uptr Head;
    uptr QuarantinedBytes;
  };

  struct SizeClass {
    InternalMmapVectorNoCtor<ChunkState> Chunks;
    InternalMmapVectorNoCtor<Slot> Slots;
    uptr NextChunk;
    uptr EntryBytes;
  };

  SizeClass Classes[kNumSizeClasses];
  InternalMmapVectorNoCtor<u64> Handles;
  uptr Granule;
  uptr QuarantinedBytes;

  bool Initialized;
  bool Reserved;
  bool Broken;
  bool HaveEntries;
  volatile u64* Touched;

  uptr ChunkBeg(uptr ClassId, uptr ChunkIdx) const {
    return RegionBeg(ClassId) + ChunkIdx * ClassIdToSize(ClassId);
  }
  uptr EntryBytesFor(uptr Chunks) const {
    return RoundUpTo(Chunks * kMetadataSize, Granule);
  }
  uptr EntriesBeg(uptr ClassId, uptr Bytes) const {
    return RegionBeg(ClassId) + kRegionSize - Bytes;
  }
  bool IsRunBacked(uptr ClassId) const {
    return ClassIdToSize(ClassId) <= Granule;
  }
  uptr RunChunks(uptr ClassId) const;

  uptr FindSlot(uptr ClassId, DeviceId Device);
  bool CreateEntries(DeviceId Prefer, uptr Bytes, u64* Handle, DeviceId* Home);
  bool GrowClass(DeviceId Device, uptr ClassId, uptr SlotIdx, uptr Max,
                 u32 AllocFlags);
  bool GraduateOldest(uptr ClassId, uptr SlotIdx);
  bool GraduateOne(uptr PreferClass, uptr PreferSlot);

  void Publish();
  void SetEntry(uptr ClassId, uptr ChunkIdx, u64 Entry);
  bool EnsureEntries(DeviceId Device, uptr ClassId, uptr Chunks);
  bool EnsureBacking(DeviceId Device, uptr ClassId, uptr ChunkIdx, uptr Need,
                     u32 AllocFlags);
  bool TakeChunk(DeviceId Device, uptr ClassId, uptr* ChunkIdx, u32 AllocFlags);
};

Allocator& GetAllocator();

uptr LeadingRedzone(uptr Size, uptr ChunkSize, uptr Alignment);

uptr ChooseClass(uptr Size);

}  // namespace __dasan

#endif  // DASAN_ALLOCATOR_H
