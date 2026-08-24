//===-- dasan_allocator.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Heap chunks come from the per-device free array. A code object is a run of
// class-0 chunks with no backing of its own; see AllocateSpan.
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
// SizeClassAllocator64 maps 256 KB at a time. The floor here is the device
// granule, and 2 MB is that granule on current parts, so a small class is
// one handle per run.
static const uptr kUserMapSize = 1 << 21;
// Code objects are not a size class. 64 B chunks let a global start on any
// chunk boundary. Heap alignment is coarser, so this class stays theirs.
static const uptr kImageClass = 0;

static_assert(kMaxRedzone <= kOffsetMax, "a redzone can outgrow its entry");
static_assert(kAllocAlignment > ClassIdToSize(kImageClass),
              "class 0 is the image span; a heap chunk is at least the "
              "alignment");

static uptr LeadingRedzone(uptr Size, uptr ChunkSize, uptr Alignment) {
  if (Size + kMinRedzone >= ChunkSize)
    return 0;
  uptr Off = RoundDownTo(ChunkSize - Size - kMinRedzone, Alignment);
  return Off > kMaxRedzone ? RoundDownTo(kMaxRedzone, Alignment) : Off;
}

static uptr ChooseClass(uptr Size, uptr Alignment) {
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

uptr Allocator::UserMapChunks(uptr ClassId) const {
  if (!MapOnPopulate(ClassId))
    return 1;
  uptr Bytes = Granule > kUserMapSize ? Granule : kUserMapSize;
  return Bytes / ClassIdToSize(ClassId);
}

void Allocator::SkipToFreshGranule(uptr ClassId, PoolId Pool) {
  Region& R = Regions[ClassId];
  const uptr Stride = MetaGranule / kMetadataSize;
  if (!Stride)
    return;

  if (R.NextChunk != 0 && R.MetadataPoolSet && R.MetadataPool != Pool) {
    uptr Boundary = RoundUpTo(R.NextChunk, Stride);
    // Boundary's granule is already homed on the other device.
    if (R.MetadataBytes >= MetadataBytesFor(Boundary + 1))
      Boundary += Stride;
    R.NextChunk = Boundary;
  }
  R.MetadataPool = Pool;
  R.MetadataPoolSet = true;
  if (R.NextChunk <= R.Chunks.size())
    return;

  const uptr Mark = R.Chunks.size();
  R.Chunks.resize(R.NextChunk);
  for (uptr I = Mark; I < R.NextChunk; ++I) R.Chunks[I].Gap = true;
}

uptr Allocator::FreeListFor(uptr ClassId, PoolId Pool) {
  Region& R = Regions[ClassId];
  for (uptr I = 0; I < R.FreeLists.size(); ++I)
    if (R.FreeLists[I].Pool == Pool)
      return I;
  uptr I = R.FreeLists.size();
  R.FreeLists.resize(I + 1);
  R.FreeLists[I].Pool = Pool;
  R.FreeLists[I].Head = 0;
  R.FreeLists[I].QuarantinedBytes = 0;
  return I;
}

bool Allocator::CreateMetadata(PoolId Prefer, uptr Bytes, u64* Handle,
                               PoolId* Home) {
  if (Prefer && VaCreate(Prefer, Bytes, 0, Handle)) {
    *Home = Prefer;
    return true;
  }
  for (PoolId Other : PlacedPools()) {
    if (!Other || Other == Prefer)
      continue;
    if (VaCreate(Other, Bytes, 0, Handle)) {
      *Home = Other;
      return true;
    }
  }
  if (VaCreate(0, Bytes, 0, Handle)) {
    *Home = 0;
    return true;
  }
  return false;
}

bool Allocator::Populate(PoolId Pool, uptr ClassId, uptr List, uptr Max,
                         u32 AllocFlags) {
  Region& R = Regions[ClassId];
  uptr ChunkSize = ClassIdToSize(ClassId);
  if (R.NextChunk >= Max)
    return false;

  uptr Run = UserMapChunks(ClassId);
  if (Run > Max - R.NextChunk)
    Run = Max - R.NextChunk;

  const bool MapNow = MapOnPopulate(ClassId);
  if (MapNow) {
    uptr PerGranule = Granule / ChunkSize;
    Run -= Run % PerGranule;
    if (Run == 0)
      return false;

    u64 H;
    if (!VaCreate(Pool, Run * ChunkSize, AllocFlags, &H))
      return false;
    if (!VaMap(ChunkBegin(ClassId, R.NextChunk), Run * ChunkSize, H, Pool)) {
      VaDestroy(H);
      return false;
    }
    VmemHandles.push_back(H);
  }

  R.Chunks.resize(R.NextChunk + Run);
  for (uptr I = Run; I-- > 0;) {
    if (MapNow)
      R.Chunks[R.NextChunk + I].Mapped = ChunkSize;
    R.Chunks[R.NextChunk + I].List = static_cast<u32>(List);
    R.FreeLists[List].Free.push_back(R.NextChunk + I);
  }
  R.NextChunk += Run;
  return true;
}

bool Allocator::Recycle(uptr ClassId, uptr List) {
  FreeList& F = Regions[ClassId].FreeLists[List];
  if (F.Head >= F.Quarantine.size())
    return false;
  F.Free.push_back(F.Quarantine[F.Head++]);
  const uptr Bytes = ClassIdToSize(ClassId);
  F.QuarantinedBytes -= Bytes;
  QuarantinedBytes -= Bytes;
  if (F.Head == F.Quarantine.size()) {
    F.Quarantine.clear();
    F.Head = 0;
  }
  return true;
}

bool Allocator::RecycleOne(uptr PreferClass, uptr PreferList) {
  if (Recycle(PreferClass, PreferList))
    return true;
  for (uptr C = 0; C < kNumSizeClasses; ++C) {
    Region& R = Regions[C];
    for (uptr L = 0; L < R.FreeLists.size(); ++L) {
      if (C == PreferClass && L == PreferList)
        continue;
      if (Recycle(C, L))
        return true;
    }
  }
  return false;
}

bool Allocator::PublishEntry(uptr ClassId, uptr ChunkIdx, u64 Entry) {
  return VaPublish(GetMetadata(RegionBeg(ClassId), ChunkIdx), &Entry,
                   sizeof(Entry));
}

bool Allocator::EnsureMetadata(PoolId Pool, uptr ClassId, uptr Chunks) {
  if (Failed || !MetaGranule || kMetadataBytes % MetaGranule)
    return false;

  Region& R = Regions[ClassId];
  // Cover these chunks only. SkipToFreshGranule keeps another device's
  // indices out of a granule this one already maps.
  uptr Want = MetadataBytesFor(Chunks);
  uptr Max = MetadataBytesFor(ReachableChunks(ClassId));
  if (Want > Max)
    Want = Max;
  if (R.MetadataBytes >= Want) {
    return !Pool || VaAllow(MetadataBegin(ClassId, R.MetadataBytes),
                            R.MetadataBytes, Pool);
  }

  uptr Beg = MetadataBegin(ClassId, Want);
  uptr Bytes = Want - R.MetadataBytes;
  u64 H;
  PoolId Home = Pool;
  if (!CreateMetadata(Pool, Bytes, &H, &Home))
    return false;
  if (!VaMap(Beg, Bytes, H, Home)) {
    VaDestroy(H);
    return false;
  }
  VmemHandles.push_back(H);
  auto Undo = [&] {
    VaUnmap(Beg, Bytes);
    VmemHandles.pop_back();
    VaDestroy(H);
    return false;
  };
  if (Pool && Pool != Home && !VaAllow(Beg, Bytes, Pool))
    return Undo();

  if (!MetadataOk) {
    Failed = !VaProbe(Beg);
    if (Failed) {
      Report("%s: could not initialize metadata; allocations stay unplaced\n",
             SanitizerToolName);
      return Undo();
    }
  }

  if (!VaPublish(Beg, nullptr, Bytes)) {
    Failed = true;
    return Undo();
  }
  MetadataOk = true;
  R.MetadataBytes = Want;
  VReport(2, "%s: class %zu holds %zu KiB of entries%s\n", SanitizerToolName,
          ClassId, Want >> 10,
          Home != Pool ? " (spilled to another pool)" : "");
  return true;
}

bool Allocator::MapUserChunk(PoolId Pool, uptr ClassId, uptr ChunkIdx,
                             uptr Need, u32 AllocFlags) {
  ChunkState& S = Regions[ClassId].Chunks[ChunkIdx];
  uptr ChunkSize = ClassIdToSize(ClassId);
  uptr Want = RoundUpTo(Need, Granule);
  if (Want > ChunkSize)
    Want = ChunkSize;
  if (S.Mapped >= Want)
    return true;

  uptr Beg = ChunkBegin(ClassId, ChunkIdx) + S.Mapped;
  uptr Bytes = Want - S.Mapped;
  u64 H;
  if (!VaCreate(Pool, Bytes, AllocFlags, &H))
    return false;
  if (!VaMap(Beg, Bytes, H, Pool)) {
    VaDestroy(H);
    return false;
  }
  VmemHandles.push_back(H);
  S.Mapped = Want;
  return true;
}

bool Allocator::GetChunk(PoolId Pool, uptr ClassId, uptr* ChunkIdx,
                         u32 AllocFlags) {
  Region& R = Regions[ClassId];
  uptr List = FreeListFor(ClassId, Pool);
  if (R.FreeLists[List].Free.empty()) {
    SkipToFreshGranule(ClassId, Pool);
    uptr Max = MaxUsableChunks(ClassId);
    uptr Reach = R.NextChunk + UserMapChunks(ClassId);
    if (Reach > Max)
      Reach = Max;
    // Another device's quarantine is backed by that device. Recycling it
    // would not fill this free list.
    bool Grew = EnsureMetadata(Pool, ClassId, Reach) &&
                Populate(Pool, ClassId, List, Max, AllocFlags);
    if (!Grew && !Recycle(ClassId, List))
      return false;
  }
  *ChunkIdx = R.FreeLists[List].Free.back();
  R.FreeLists[List].Free.pop_back();
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
  if (!VaAllowAgents(UserBeg, UserSize, Agents, N))
    return false;
  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  return VaAllowAgents(GetMetadata(Addr, Idx), kMetadataSize, Agents, N);
}

void Allocator::Init() {
  if (Initialized)
    return;
  Initialized = true;

  Granule = VaGranule();
  MetaGranule = VaMetaGranule();
  if (!Granule || !MetaGranule)
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
      Region& R = Regions[C];
      for (uptr I = 0; I < R.NextChunk; ++I)
        if (R.Chunks[I].Mapped)
          VaUnmap(ChunkBegin(C, I), R.Chunks[I].Mapped);
      if (R.MetadataBytes)
        VaUnmap(MetadataBegin(C, R.MetadataBytes), R.MetadataBytes);
      for (uptr I = 0; I < R.FreeLists.size(); ++I) {
        R.FreeLists[I].Free.clear();
        R.FreeLists[I].Quarantine.clear();
      }
      R.FreeLists.clear();
      R.Chunks.clear();
      R.NextChunk = R.MetadataBytes = 0;
      R.MetadataPoolSet = false;
    }
    for (uptr I = 0; I < VmemHandles.size(); ++I) VaDestroy(VmemHandles[I]);
    VmemHandles.clear();

    VaRelease(kSpaceBeg, kSpaceUsed);
    Reserved = false;
    Failed = false;
    MetadataOk = false;
    Granule = MetaGranule = 0;
    QuarantinedBytes = 0;
    Count = {};
  }
  Initialized = false;
}

uptr Allocator::Allocate(PoolId Pool, uptr Size, u32 AllocFlags, u32 Stack,
                         u32 Tid) {
  if (!Reserved || Failed) {
    ++Count.Inactive;
    return 0;
  }
  if (Size == 0)
    return 0;

  const uptr Keep = Size >= Granule ? Granule : kAllocAlignment;
  uptr First = ChooseClass(Size, Keep);
  if (First >= kNumSizeClasses) {
    ++Count.TooLarge;
    return 0;
  }

  for (uptr ClassId = First; ClassId < kNumSizeClasses; ++ClassId) {
    uptr ChunkIdx;
    if (!GetChunk(Pool, ClassId, &ChunkIdx, AllocFlags)) {
      VReport(2, "%s: no chunk for class %zu\n", SanitizerToolName, ClassId);
      continue;
    }

    uptr Offset = LeadingRedzone(Size, ClassIdToSize(ClassId), Keep);
    if (!MapUserChunk(Pool, ClassId, ChunkIdx, Offset + Size, AllocFlags)) {
      VReport(2, "%s: no backing for class %zu chunk %zu need %zu\n",
              SanitizerToolName, ClassId, ChunkIdx, Offset + Size);
      Region& R = Regions[ClassId];
      R.FreeLists[R.Chunks[ChunkIdx].List].Free.push_back(ChunkIdx);
      continue;
    }

    ChunkState& S = Regions[ClassId].Chunks[ChunkIdx];
    S.Size = Size;
    S.Offset = Offset;
    S.Live = true;
    S.Gap = false;
    S.Span = false;
    S.AllocStack = Stack;
    S.FreeStack = 0;
    S.AllocTid = Tid;
    S.FreeTid = 0;
    if (!PublishEntry(ClassId, ChunkIdx, MakeEntry(Size, Offset))) {
      S.Live = false;
      S.Size = 0;
      S.Offset = 0;
      Regions[ClassId].FreeLists[S.List].Free.push_back(ChunkIdx);
      Failed = true;
      return 0;
    }

    ++Count.Placed;
    if (ClassId != First)
      ++Count.LargerClass;
    VReport(3, "%s: placed %zu bytes at 0x%zx (class %zu chunk %zu +%zu)\n",
            SanitizerToolName, Size, ChunkBegin(ClassId, ChunkIdx) + Offset,
            ClassId, ChunkIdx, Offset);
    return ChunkBegin(ClassId, ChunkIdx) + Offset;
  }

  ++Count.Exhausted;
  return 0;
}

uptr Allocator::AllocateSpan(PoolId Pool, uptr Size) {
  if (!Reserved || Failed || Size == 0)
    return 0;

  const uptr ClassId = kImageClass;
  const uptr ChunkSize = ClassIdToSize(ClassId);
  const uptr Page = GetPageSizeCached();
  const uptr Align = Page > ChunkSize ? Page : ChunkSize;
  const uptr Need = RoundUpTo(Size, ChunkSize) / ChunkSize;
  const uptr Stride = Align / ChunkSize;

  Region& R = Regions[ClassId];
  const uptr List = FreeListFor(ClassId, Pool);

  uptr First = (uptr)-1;
  for (uptr C = 0; C + Need <= R.NextChunk; C += Stride) {
    bool Free = true;
    for (uptr I = 0; I < Need; ++I) {
      const ChunkState& S = R.Chunks[C + I];
      // A gap stays on the device that owns its granule. Another device's
      // free span stays on its own.
      if (S.Live || S.Gap || R.FreeLists[S.List].Pool != Pool) {
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
    SkipToFreshGranule(ClassId, Pool);
    First = RoundUpTo(R.NextChunk, Stride);
    const uptr End = First + Need;
    if (End > MaxUsableChunks(ClassId) || !EnsureMetadata(Pool, ClassId, End)) {
      ++Count.Exhausted;
      return 0;
    }
    R.Chunks.resize(End);
    for (uptr I = R.NextChunk; I < First; ++I)
      R.Chunks[I].List = static_cast<u32>(List);
    R.NextChunk = End;
  } else if (!EnsureMetadata(Pool, ClassId, First + Need)) {
    ++Count.Exhausted;
    return 0;
  }

  for (uptr I = First; I < First + Need; ++I) {
    ChunkState& S = R.Chunks[I];
    S.List = static_cast<u32>(List);
    S.Live = true;
    S.Span = true;
    S.Gap = false;
    S.Size = I == First ? Size : 0;
    S.Offset = 0;
    S.Mapped = 0;
    S.AllocStack = S.FreeStack = 0;
    S.AllocTid = S.FreeTid = 0;
  }
  ++Count.CodeObjects;
  Count.CodeObjectChunks += Need;
  ++Count.Placed;
  return ChunkBegin(ClassId, First);
}

void Allocator::DeallocateSpan(uptr Addr, uptr Size) {
  if (!Reserved || !Size)
    return;

  const uptr ClassId = kImageClass;
  const uptr ChunkSize = ClassIdToSize(ClassId);
  const uptr Need = RoundUpTo(Size, ChunkSize) / ChunkSize;
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  Region& R = Regions[ClassId];
  const uptr N = Idx < R.NextChunk ? Min(Need, R.NextChunk - Idx) : 0;
  for (uptr C = Idx; C < Idx + N; ++C) {
    ChunkState& S = R.Chunks[C];
    S.Live = false;
    S.Span = false;
    S.Size = 0;
  }
  if (MetadataOk && N) {
    const uptr Low = GetMetadata(RegionBeg(ClassId), Idx + N - 1);
    if (!VaPublish(Low, nullptr, N * kMetadataSize))
      Failed = true;
  }
  while (R.NextChunk && !R.Chunks[R.NextChunk - 1].Live &&
         !R.Chunks[R.NextChunk - 1].Gap)
    --R.NextChunk;
  if (Count.CodeObjectChunks >= Need)
    Count.CodeObjectChunks -= Need;
  if (Count.CodeObjects)
    --Count.CodeObjects;
}

bool Allocator::Deallocate(uptr Addr, u32 Stack, u32 Tid) {
  if (!IsPlaced(Addr))
    return false;

  uptr ClassId = GetSizeClass(Addr);
  uptr ChunkIdx = GetChunkIdx(Addr, ClassId);
  Region& R = Regions[ClassId];
  if (ChunkIdx >= R.NextChunk)
    return false;

  ChunkState& S = R.Chunks[ChunkIdx];
  if (S.Span || S.Gap || !S.Live ||
      Addr != ChunkBegin(ClassId, ChunkIdx) + S.Offset) {
    ReportInvalidFree(Addr, Stack, Tid);
    return true;
  }

  S.Live = false;
  S.FreeStack = Stack;
  S.FreeTid = Tid;
  if (!PublishEntry(ClassId, ChunkIdx, PoisonEntry(MakeEntry(0, S.Offset))))
    Failed = true;

  FreeList& F = R.FreeLists[S.List];
  const uptr Bytes = ClassIdToSize(ClassId);
  F.Quarantine.push_back(ChunkIdx);
  F.QuarantinedBytes += Bytes;
  QuarantinedBytes += Bytes;
  const uptr Budget = (uptr)flags()->quarantine_size_mb << 20;
  while (QuarantinedBytes > Budget && RecycleOne(ClassId, S.List)) {
  }

  ++Count.Freed;
  return true;
}

bool Allocator::Describe(uptr Addr, AllocInfo* Out) const {
  if (!IsPlaced(Addr))
    return false;
  uptr ClassId = GetSizeClass(Addr);
  uptr ChunkIdx = GetChunkIdx(Addr, ClassId);
  if (ChunkIdx >= Regions[ClassId].NextChunk)
    return false;

  const ChunkState& S = Regions[ClassId].Chunks[ChunkIdx];
  Out->ChunkBeg = ChunkBegin(ClassId, ChunkIdx);
  Out->ChunkSize = ClassIdToSize(ClassId);
  Out->Beg = Out->ChunkBeg + S.Offset;
  Out->Size = S.Size;
  Out->Pool = Regions[ClassId].FreeLists[S.List].Pool;
  Out->Freed = !S.Live;
  Out->AllocStack = S.AllocStack;
  Out->FreeStack = S.FreeStack;
  Out->AllocTid = S.AllocTid;
  Out->FreeTid = S.FreeTid;
  return S.Size != 0;
}

void Allocator::PublishEntries(uptr Addr, const u64* Entries, uptr N) {
  if (!MetadataOk || !N)
    return;
  const uptr First = GetChunkIdx(Addr, kImageClass);
  u64 Block[1024];
  for (uptr Done = 0; Done < N;) {
    const uptr Step = Min(N - Done, (uptr)ARRAY_SIZE(Block));
    for (uptr I = 0; I < Step; ++I) Block[Step - 1 - I] = Entries[Done + I];
    uptr At = GetMetadata(RegionBeg(kImageClass), First + Done + Step - 1);
    if (!VaPublish(At, Block, Step * sizeof(u64))) {
      Failed = true;
      return;
    }
    Done += Step;
  }
}

static_assert(__is_trivially_constructible(Allocator),
              "Allocator would re-zero after OpenMP's Init");
static Allocator Instance;
Allocator& GetAllocator() { return Instance; }

}  // namespace __dasan
