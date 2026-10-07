//===-- dasan_allocator.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two placement paths share the same window:
//
//   Allocate(Device, Size)        heap. Size-class chunk, leading redzone,
//                                 backing grown to cover the object.
//   Allocate(Device, Size, Class) images. Contiguous Class chunks with no
//                                 backing; the HSA edge maps the alias itself.
//
//===----------------------------------------------------------------------===//

#include "dasan_allocator.h"

#include "dasan.h"
#include "dasan_flags.h"
#include "dasan_platform.h"
#include "dasan_report.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

using namespace __sanitizer;

namespace __dasan {

static const uptr kMaxRedzone = 2048;
static const uptr kRunBytes = 1 << 21;

static_assert(kMaxRedzone <= kOffsetMax, "a redzone can outgrow its entry");

static_assert(
    kAllocAlignment > (1ULL << kMinSizeLog),
    "class 0 is the image span; a heap chunk is at least the alignment");

uptr ChooseClass(uptr Size, uptr Alignment) {
  if (!Alignment)
    Alignment = kAllocAlignment;
  for (uptr ClassId = 0; ClassId < kNumSizeClasses; ++ClassId) {
    const uptr Chunk = ClassIdToSize(ClassId);
    // The chunk base is aligned to the chunk. A smaller chunk cannot hold
    // an Alignment-aligned pointer.
    if (Chunk < Alignment)
      continue;
    const uptr Off = LeadingRedzone(Size, Chunk, Alignment);
    if (Off + Size + kMinRedzone <= Chunk)
      return ClassId;
  }
  return kNumSizeClasses;
}

uptr LeadingRedzone(uptr Size, uptr ChunkSize, uptr Alignment) {
  if (Size + kMinRedzone >= ChunkSize)
    return 0;
  uptr Off = RoundDownTo(ChunkSize - Size - kMinRedzone, Alignment);
  return Off > kMaxRedzone ? RoundDownTo(kMaxRedzone, Alignment) : Off;
}

uptr Allocator::RunChunks(uptr ClassId) const {
  if (!IsRunBacked(ClassId))
    return 1;
  uptr Bytes = Granule > kRunBytes ? Granule : kRunBytes;
  return Bytes / ClassIdToSize(ClassId);
}

void Allocator::SeparateGranule(uptr ClassId, DeviceId Device) {
  SizeClass& SC = Classes[ClassId];
  const uptr Stride = Granule / kMetadataSize;
  if (!Stride)
    return;

  if (SC.NextChunk != 0 && SC.GranuleSet && SC.GranuleDevice != Device) {
    uptr Boundary = RoundUpTo(SC.NextChunk, Stride);
    // Boundary's granule is already homed on the other device.
    if (SC.EntryBytes >= EntryBytesFor(Boundary + 1))
      Boundary += Stride;
    SC.NextChunk = Boundary;
  }
  SC.GranuleDevice = Device;
  SC.GranuleSet = true;
  if (SC.NextChunk <= SC.Chunks.size())
    return;

  // Not allocatable. Those entries belong to the previous device.
  const uptr Mark = SC.Chunks.size();
  SC.Chunks.resize(SC.NextChunk);
  for (uptr I = Mark; I < SC.NextChunk; ++I) SC.Chunks[I].Live = true;
}

uptr Allocator::FindSlot(uptr ClassId, DeviceId Device) {
  SizeClass& SC = Classes[ClassId];
  for (uptr I = 0; I < SC.Slots.size(); ++I)
    if (SC.Slots[I].Device == Device)
      return I;
  uptr I = SC.Slots.size();
  SC.Slots.resize(I + 1);
  SC.Slots[I].Device = Device;
  SC.Slots[I].Head = 0;
  SC.Slots[I].QuarantinedBytes = 0;
  return I;
}

bool Allocator::CreateEntries(DeviceId Prefer, uptr Bytes, u64* Handle,
                              DeviceId* Home) {
  if (Prefer && VaCreate(Prefer, Bytes, 0, Handle)) {
    *Home = Prefer;
    return true;
  }
  for (DeviceId D : PlatformDevices()) {
    if (!D || D == Prefer)
      continue;
    if (VaCreate(D, Bytes, 0, Handle)) {
      *Home = D;
      return true;
    }
  }
  if (VaCreate(0, Bytes, 0, Handle)) {
    *Home = 0;
    return true;
  }
  return false;
}

bool Allocator::GrowClass(DeviceId Device, uptr ClassId, uptr SlotIdx, uptr Max,
                          u32 AllocFlags) {
  SizeClass& SC = Classes[ClassId];
  uptr ChunkSize = ClassIdToSize(ClassId);
  if (SC.NextChunk >= Max)
    return false;

  uptr Run = RunChunks(ClassId);
  if (Run > Max - SC.NextChunk)
    Run = Max - SC.NextChunk;

  const bool BackNow = IsRunBacked(ClassId);
  if (BackNow) {
    uptr PerGranule = Granule / ChunkSize;
    Run -= Run % PerGranule;
    if (Run == 0)
      return false;

    u64 H;
    if (!VaCreate(Device, Run * ChunkSize, AllocFlags, &H))
      return false;
    if (!VaMap(ChunkBeg(ClassId, SC.NextChunk), Run * ChunkSize, H,
               /*ReadOnly=*/false, Device)) {
      VaDestroy(H);
      return false;
    }
    Handles.push_back(H);
  }

  SC.Chunks.resize(SC.NextChunk + Run);
  for (uptr I = Run; I-- > 0;) {
    if (BackNow)
      SC.Chunks[SC.NextChunk + I].Backed = ChunkSize;
    SC.Chunks[SC.NextChunk + I].Slot = static_cast<u32>(SlotIdx);
    SC.Slots[SlotIdx].Free.push_back(SC.NextChunk + I);
  }
  SC.NextChunk += Run;
  return true;
}

bool Allocator::GraduateOldest(uptr ClassId, uptr SlotIdx) {
  Slot& S = Classes[ClassId].Slots[SlotIdx];
  if (S.Head >= S.Quarantine.size())
    return false;
  S.Free.push_back(S.Quarantine[S.Head++]);
  const uptr Bytes = ClassIdToSize(ClassId);
  S.QuarantinedBytes -= Bytes;
  QuarantinedBytes -= Bytes;
  if (S.Head == S.Quarantine.size()) {
    S.Quarantine.clear();
    S.Head = 0;
  }
  return true;
}

bool Allocator::GraduateOne(uptr PreferClass, uptr PreferSlot) {
  if (GraduateOldest(PreferClass, PreferSlot))
    return true;
  for (uptr C = 0; C < kNumSizeClasses; ++C) {
    SizeClass& SC = Classes[C];
    for (uptr S = 0; S < SC.Slots.size(); ++S) {
      if (C == PreferClass && S == PreferSlot)
        continue;
      if (GraduateOldest(C, S))
        return true;
    }
  }
  return false;
}

void Allocator::SetEntry(uptr ClassId, uptr ChunkIdx, u64 Entry) {
  uptr Addr = GetMetadata(RegionBeg(ClassId), ChunkIdx);
  Touched = reinterpret_cast<volatile u64*>(Addr);
  VaWrite(Addr, &Entry, sizeof(Entry));
}

void Allocator::Publish() {
  if (HaveEntries)
    VaPublish(reinterpret_cast<uptr>(Touched));
}

bool Allocator::EnsureEntries(DeviceId Device, uptr ClassId, uptr Chunks) {
  if (Broken || kMetadataBytes % Granule)
    return false;

  SizeClass& SC = Classes[ClassId];
  // Cover these chunks only. SeparateGranule keeps another device's indices
  // out of a granule this one already maps.
  uptr Want = EntryBytesFor(Chunks);
  uptr Max = EntryBytesFor(ReachableChunks(ClassId));
  if (Want > Max)
    Want = Max;
  if (SC.EntryBytes >= Want) {
    return !Device || VaAllow(EntriesBeg(ClassId, SC.EntryBytes), SC.EntryBytes,
                              Device, /*ReadOnly=*/true);
  }

  uptr Beg = EntriesBeg(ClassId, Want);
  uptr Bytes = Want - SC.EntryBytes;
  u64 H;
  DeviceId Home = Device;
  if (!CreateEntries(Device, Bytes, &H, &Home))
    return false;
  if (!VaMap(Beg, Bytes, H, /*ReadOnly=*/true, Home)) {
    VaDestroy(H);
    return false;
  }
  Handles.push_back(H);
  auto Undo = [&] {
    VaUnmap(Beg, Bytes);
    Handles.pop_back();
    VaDestroy(H);
    return false;
  };
  if (Device && Device != Home &&
      !VaAllow(Beg, Bytes, Device, /*ReadOnly=*/true))
    return Undo();

  if (!HaveEntries) {
    Broken = !VaProbe(Beg);
    if (Broken) {
      Report("%s: could not initialize metadata; allocations stay unplaced\n",
             SanitizerToolName);
      return Undo();
    }
  }

  if (!VaFill(Beg, Bytes))
    return Undo();
  Touched = reinterpret_cast<volatile u64*>(Beg + Bytes - sizeof(u64));
  Publish();
  HaveEntries = true;
  SC.EntryBytes = Want;
  VReport(2, "%s: class %zu holds %zu KiB of entries%s\n", SanitizerToolName,
          ClassId, Want >> 10,
          Home != Device ? " (spilled to another pool)" : "");
  return true;
}

bool Allocator::EnsureBacking(DeviceId Device, uptr ClassId, uptr ChunkIdx,
                              uptr Need, u32 AllocFlags) {
  ChunkState& S = Classes[ClassId].Chunks[ChunkIdx];
  uptr ChunkSize = ClassIdToSize(ClassId);
  uptr Want = RoundUpTo(Need, Granule);
  if (Want > ChunkSize)
    Want = ChunkSize;
  if (S.Backed >= Want)
    return true;

  uptr Beg = ChunkBeg(ClassId, ChunkIdx) + S.Backed;
  uptr Bytes = Want - S.Backed;
  u64 H;
  if (!VaCreate(Device, Bytes, AllocFlags, &H))
    return false;
  if (!VaMap(Beg, Bytes, H, /*ReadOnly=*/false, Device)) {
    VaDestroy(H);
    return false;
  }
  Handles.push_back(H);
  S.Backed = Want;
  return true;
}

bool Allocator::TakeChunk(DeviceId Device, uptr ClassId, uptr* ChunkIdx,
                          u32 AllocFlags) {
  SizeClass& SC = Classes[ClassId];
  uptr SlotIdx = FindSlot(ClassId, Device);
  if (SC.Slots[SlotIdx].Free.empty()) {
    SeparateGranule(ClassId, Device);
    uptr Max = MaxUsableChunks(ClassId);
    uptr Reach = SC.NextChunk + RunChunks(ClassId);
    if (Reach > Max)
      Reach = Max;
    bool Grew = EnsureEntries(Device, ClassId, Reach) &&
                GrowClass(Device, ClassId, SlotIdx, Max, AllocFlags);
    if (!Grew && !GraduateOldest(ClassId, SlotIdx))
      return false;
  }
  *ChunkIdx = SC.Slots[SlotIdx].Free.back();
  SC.Slots[SlotIdx].Free.pop_back();
  return true;
}

bool Allocator::AllowAccess(uptr Addr, const hsa_agent_t* Agents, u32 N) {
  if (!IsPlaced(Addr))
    return false;
  AllocInfo Info;
  uptr UserBeg = Addr;
  uptr UserSize = 1;
  if (Describe(Addr, &Info)) {
    UserBeg = Info.ChunkBeg;
    UserSize = Info.ChunkSize;
  }
  if (!VaAllowAgents(UserBeg, UserSize, Agents, N, /*ReadOnly=*/false))
    return false;
  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  return VaAllowAgents(GetMetadata(Addr, Idx), kMetadataSize, Agents, N,
                       /*ReadOnly=*/true);
}

void Allocator::Init() {
  if (Initialized)
    return;
  Initialized = true;

  Granule = VaGranule();
  if (!Granule)
    return;

  if (!VaReserve(kSpaceBeg, kSpaceUsed)) {
    VReport(1,
            "%s: could not reserve [0x%zx, 0x%zx); allocations stay unplaced\n",
            SanitizerToolName, (uptr)kSpaceBeg, (uptr)(kSpaceBeg + kSpaceUsed));
    return;
  }
  Reserved = true;
  VReport(1, "%s: reserved [0x%zx, 0x%zx)\n", SanitizerToolName,
          (uptr)kSpaceBeg, (uptr)(kSpaceBeg + kSpaceUsed));
}

void Allocator::Shutdown() {
  if (Reserved) {
    for (uptr C = 0; C < kNumSizeClasses; ++C) {
      SizeClass& SC = Classes[C];
      for (uptr I = 0; I < SC.NextChunk; ++I)
        if (SC.Chunks[I].Backed)
          VaUnmap(ChunkBeg(C, I), SC.Chunks[I].Backed);
      if (SC.EntryBytes)
        VaUnmap(EntriesBeg(C, SC.EntryBytes), SC.EntryBytes);
      for (uptr I = 0; I < SC.Slots.size(); ++I) {
        SC.Slots[I].Free.clear();
        SC.Slots[I].Quarantine.clear();
      }
      SC.Slots.clear();
      SC.Chunks.clear();
      SC.NextChunk = SC.EntryBytes = 0;
      SC.GranuleSet = false;
    }
    for (uptr I = 0; I < Handles.size(); ++I) VaDestroy(Handles[I]);
    Handles.clear();

    VaRelease(kSpaceBeg, kSpaceUsed);
    Reserved = false;
    Broken = false;
    HaveEntries = false;
    Touched = nullptr;
    Granule = 0;
    QuarantinedBytes = 0;
    NumImages = NumImageChunks = 0;
    NumAllocated = NumFreed = 0;
    NumPassedTooBig = NumPassedNoRoom = NumPassedInactive = NumFellForward = 0;
  }
  Initialized = false;
}

uptr Allocator::Allocate(DeviceId Device, uptr Size, u32 AllocFlags, u32 Stack,
                         u32 Tid) {
  if (!Reserved || Broken) {
    ++NumPassedInactive;
    return 0;
  }
  if (Size == 0)
    return 0;

  const uptr Keep = Size >= Granule ? Granule : kAllocAlignment;
  uptr First = ChooseClass(Size, Keep);
  if (First >= kNumSizeClasses) {
    ++NumPassedTooBig;
    return 0;
  }

  for (uptr ClassId = First; ClassId < kNumSizeClasses; ++ClassId) {
    uptr ChunkIdx;
    if (!TakeChunk(Device, ClassId, &ChunkIdx, AllocFlags)) {
      VReport(2, "%s: no chunk for class %zu\n", SanitizerToolName, ClassId);
      continue;
    }

    uptr Offset = LeadingRedzone(Size, ClassIdToSize(ClassId), Keep);
    if (!EnsureBacking(Device, ClassId, ChunkIdx, Offset + Size, AllocFlags)) {
      VReport(2, "%s: no backing for class %zu chunk %zu need %zu\n",
              SanitizerToolName, ClassId, ChunkIdx, Offset + Size);
      SizeClass& SC = Classes[ClassId];
      SC.Slots[SC.Chunks[ChunkIdx].Slot].Free.push_back(ChunkIdx);
      continue;
    }

    ChunkState& S = Classes[ClassId].Chunks[ChunkIdx];
    S.Size = Size;
    S.Offset = Offset;
    S.Live = true;
    S.AllocStack = Stack;
    S.FreeStack = 0;
    S.AllocTid = Tid;
    S.FreeTid = 0;
    SetEntry(ClassId, ChunkIdx, MakeEntry(Size, Offset));
    Publish();

    ++NumAllocated;
    if (ClassId != First)
      ++NumFellForward;
    VReport(3, "%s: placed %zu bytes at 0x%zx (class %zu chunk %zu +%zu)\n",
            SanitizerToolName, Size, ChunkBeg(ClassId, ChunkIdx) + Offset,
            ClassId, ChunkIdx, Offset);
    return ChunkBeg(ClassId, ChunkIdx) + Offset;
  }

  ++NumPassedNoRoom;
  return 0;
}

uptr Allocator::Allocate(DeviceId Device, uptr Size, uptr ClassId) {
  if (!Reserved || Broken || Size == 0 || ClassId >= kNumSizeClasses)
    return 0;

  const uptr ChunkSize = ClassIdToSize(ClassId);
  const uptr Page = GetPageSizeCached();
  const uptr Align = Page > ChunkSize ? Page : ChunkSize;
  const uptr Need = RoundUpTo(Size, ChunkSize) / ChunkSize;
  const uptr Stride = Align / ChunkSize;

  SizeClass& SC = Classes[ClassId];
  const uptr SlotIdx = FindSlot(ClassId, Device);

  uptr First = (uptr)-1;
  for (uptr C = 0; C + Need <= SC.NextChunk; C += Stride) {
    bool Free = true;
    for (uptr I = 0; I < Need; ++I) {
      const ChunkState& S = SC.Chunks[C + I];
      // A gap is live. Another device's free span stays on its granule.
      if (S.Live || SC.Slots[S.Slot].Device != Device) {
        Free = false;
        break;
      }
    }
    if (Free) {
      First = C;
      break;
    }
  }

  if (First == (uptr)-1) {
    SeparateGranule(ClassId, Device);
    First = RoundUpTo(SC.NextChunk, Stride);
    const uptr End = First + Need;
    if (End > MaxUsableChunks(ClassId) ||
        !EnsureEntries(Device, ClassId, End)) {
      ++NumPassedNoRoom;
      return 0;
    }
    SC.Chunks.resize(End);
    for (uptr I = SC.NextChunk; I < First; ++I)
      SC.Chunks[I].Slot = static_cast<u32>(SlotIdx);
    SC.NextChunk = End;
  } else if (!EnsureEntries(Device, ClassId, First + Need)) {
    ++NumPassedNoRoom;
    return 0;
  }

  for (uptr I = First; I < First + Need; ++I) {
    ChunkState& S = SC.Chunks[I];
    S.Slot = static_cast<u32>(SlotIdx);
    S.Live = true;
    S.Size = I == First ? Size : 0;
    S.Offset = 0;
    S.Backed = 0;
    S.AllocStack = S.FreeStack = 0;
    S.AllocTid = S.FreeTid = 0;
  }
  ++NumImages;
  NumImageChunks += Need;
  ++NumAllocated;
  return ChunkBeg(ClassId, First);
}

void Allocator::Deallocate(uptr Addr, uptr Size, uptr ClassId) {
  if (!Reserved || !Size || ClassId >= kNumSizeClasses)
    return;

  const uptr ChunkSize = ClassIdToSize(ClassId);
  const uptr Need = RoundUpTo(Size, ChunkSize) / ChunkSize;
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  SizeClass& SC = Classes[ClassId];
  const uptr N = Idx < SC.NextChunk ? Min(Need, SC.NextChunk - Idx) : 0;
  for (uptr C = Idx; C < Idx + N; ++C) {
    ChunkState& S = SC.Chunks[C];
    S.Live = false;
    S.Size = 0;
  }
  if (HaveEntries && N) {
    const uptr Low = GetMetadata(RegionBeg(ClassId), Idx + N - 1);
    VaFill(Low, N * kMetadataSize);
    Touched = reinterpret_cast<volatile u64*>(Low);
    Publish();
  }
  while (SC.NextChunk && !SC.Chunks[SC.NextChunk - 1].Live) --SC.NextChunk;
  if (NumImageChunks >= Need)
    NumImageChunks -= Need;
  if (NumImages)
    --NumImages;
}

bool Allocator::Deallocate(uptr Addr, u32 Stack, u32 Tid) {
  if (!IsPlaced(Addr))
    return false;

  uptr ClassId = GetSizeClass(Addr);
  uptr ChunkIdx = GetChunkIdx(Addr, ClassId);
  SizeClass& SC = Classes[ClassId];
  if (ChunkIdx >= SC.NextChunk)
    return false;

  ChunkState& S = SC.Chunks[ChunkIdx];
  if (!S.Live || Addr != ChunkBeg(ClassId, ChunkIdx) + S.Offset) {
    ReportInvalidFree(Addr, Stack, Tid);
    return true;
  }

  S.Live = false;
  S.FreeStack = Stack;
  S.FreeTid = Tid;
  SetEntry(ClassId, ChunkIdx, PoisonEntry(MakeEntry(0, S.Offset)));
  Publish();

  Slot& Sl = SC.Slots[S.Slot];
  const uptr Bytes = ClassIdToSize(ClassId);
  Sl.Quarantine.push_back(ChunkIdx);
  Sl.QuarantinedBytes += Bytes;
  QuarantinedBytes += Bytes;
  const uptr Budget = (uptr)flags()->quarantine_size_mb << 20;
  while (QuarantinedBytes > Budget && GraduateOne(ClassId, S.Slot)) {
  }

  ++NumFreed;
  return true;
}

bool Allocator::Describe(uptr Addr, AllocInfo* Out) const {
  if (!IsPlaced(Addr))
    return false;
  uptr ClassId = GetSizeClass(Addr);
  uptr ChunkIdx = GetChunkIdx(Addr, ClassId);
  if (ChunkIdx >= Classes[ClassId].NextChunk)
    return false;

  const ChunkState& S = Classes[ClassId].Chunks[ChunkIdx];
  Out->ChunkBeg = ChunkBeg(ClassId, ChunkIdx);
  Out->ChunkSize = ClassIdToSize(ClassId);
  Out->Beg = Out->ChunkBeg + S.Offset;
  Out->Size = S.Size;
  Out->Device = Classes[ClassId].Slots[S.Slot].Device;
  Out->Freed = !S.Live;
  Out->AllocStack = S.AllocStack;
  Out->FreeStack = S.FreeStack;
  Out->AllocTid = S.AllocTid;
  Out->FreeTid = S.FreeTid;
  return S.Size != 0;
}

void Allocator::SetImageEntries(uptr Addr, const u64* Entries, uptr N) {
  if (!HaveEntries || !N)
    return;
  // Entries grow downward from the region's top, so each block is reversed to
  // land as one contiguous write.
  const uptr First = GetChunkIdx(Addr, 0);
  u64 Block[1024];
  for (uptr Done = 0; Done < N;) {
    const uptr Step = Min(N - Done, (uptr)ARRAY_SIZE(Block));
    for (uptr I = 0; I < Step; ++I) Block[Step - 1 - I] = Entries[Done + I];
    VaWrite(GetMetadata(RegionBeg(0), First + Done + Step - 1), Block,
            Step * sizeof(u64));
    Done += Step;
  }
  Touched = reinterpret_cast<volatile u64*>(GetMetadata(RegionBeg(0), First));
  Publish();
}

static_assert(__is_trivially_constructible(Allocator),
              "Allocator would re-zero after OpenMP's Init");
static Allocator Instance;
Allocator& GetAllocator() { return Instance; }

}  // namespace __dasan
