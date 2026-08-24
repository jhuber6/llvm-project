//===-- dasan_image.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Load-at-address and the dma-buf alias share one table. ELF is only the
// symbol table in the storage blob: a global that owns a private run of
// class-0 chunks gets its own entries, and the rest keep one entry for the
// whole image. The image is not a power of two, which is why it is a run of
// the smallest chunks rather than one size class.
//
//===----------------------------------------------------------------------===//

#include "dasan_image.h"

#include <elf.h>

#include "dasan_allocator.h"
#include "dasan_flags.h"
#include "dasan_mapping.h"
#include "dasan_platform.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_offload.h"

using namespace __sanitizer;

namespace __dasan {

namespace {
const char kGuardName[] = "__dasan_guard";

struct Symbol {
  uptr Value;
  uptr Size;
  const char* Name;
  bool Guard;
};

// Base is the address in the window. LoadBase is the loader's address.
// They are equal when the loader placed the image itself.
struct TrackedImage {
  uptr LoadBase;
  uptr LoadSize;
  uptr Base;
  uptr MapSize;
  u64 Handle;
  bool Described;
};

struct ImageStats {
  uptr Described;
  uptr Shared;
};

InternalMmapVectorNoCtor<GlobalInfo> Globals;
InternalMmapVectorNoCtor<TrackedImage> Images;

bool InFile(uptr Size, const Elf64_Shdr& S) {
  return S.sh_type != SHT_NOBITS && S.sh_offset <= Size &&
         S.sh_size <= Size - S.sh_offset;
}

// STT_OBJECT symbols whose st_value fits in the loaded image. Names point
// into Storage.
bool ReadSymtab(const void* Storage, uptr Size, uptr LoadSize,
                InternalMmapVector<Symbol>* Out) {
  if (!Storage || Size < sizeof(Elf64_Ehdr))
    return false;
  auto* Hdr = reinterpret_cast<const Elf64_Ehdr*>(Storage);
  if (internal_memcmp(Hdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      Hdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      Hdr->e_shentsize != sizeof(Elf64_Shdr) || !Hdr->e_shnum)
    return false;
  const uptr ShdrBytes = Hdr->e_shnum * sizeof(Elf64_Shdr);
  if (Hdr->e_shoff > Size || ShdrBytes > Size - Hdr->e_shoff)
    return false;
  auto* Sections = reinterpret_cast<const Elf64_Shdr*>(
      static_cast<const char*>(Storage) + Hdr->e_shoff);

  const Elf64_Shdr* Symtab = nullptr;
  for (uptr S = 0; S < Hdr->e_shnum; ++S) {
    const Elf64_Shdr& Sec = Sections[S];
    if (!InFile(Size, Sec) || Sec.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (Sec.sh_type == SHT_SYMTAB)
      Symtab = &Sec;
    else if (Sec.sh_type == SHT_DYNSYM && !Symtab)
      Symtab = &Sec;
  }
  if (!Symtab || Symtab->sh_link >= Hdr->e_shnum)
    return false;

  const Elf64_Shdr& Strtab = Sections[Symtab->sh_link];
  const char* Names = InFile(Size, Strtab)
                          ? static_cast<const char*>(Storage) + Strtab.sh_offset
                          : nullptr;
  auto* Syms = reinterpret_cast<const Elf64_Sym*>(
      static_cast<const char*>(Storage) + Symtab->sh_offset);
  for (uptr N = 0; N < Symtab->sh_size / sizeof(Elf64_Sym); ++N) {
    const Elf64_Sym& Sym = Syms[N];
    if (ELF64_ST_TYPE(Sym.st_info) != STT_OBJECT || !Sym.st_size)
      continue;
    if (Sym.st_shndx == SHN_UNDEF || Sym.st_shndx >= Hdr->e_shnum)
      continue;
    if (!(Sections[Sym.st_shndx].sh_flags & SHF_ALLOC))
      continue;
    if (Sym.st_value > LoadSize || Sym.st_size > LoadSize - Sym.st_value)
      continue;
    const char* Name =
        Names && Sym.st_name < Strtab.sh_size ? Names + Sym.st_name : nullptr;
    const bool Guard =
        Name && !internal_strncmp(Name, kGuardName, sizeof(kGuardName) - 1);
    Out->push_back({(uptr)Sym.st_value, (uptr)Sym.st_size, Name, Guard});
  }
  return true;
}

bool ByValue(const Symbol& A, const Symbol& B) { return A.Value < B.Value; }

void DescribeGlobals(uptr Base, uptr LoadSize, const void* Storage,
                     uptr StorageSize, ImageStats* Stats) {
  const uptr ChunkSize = ClassIdToSize(0);
  const uptr Chunks = RoundUpTo(LoadSize, ChunkSize) / ChunkSize;
  InternalMmapVector<u64> Entries(Chunks);

  // Until a symbol claims them, every chunk names the whole image. The
  // offset points back at the image base, clamped to what an entry can
  // express, so an access off the end of the mapping still faults.
  const uptr Reach = static_cast<uptr>(-kOffsetMin);
  for (uptr C = 0; C < Chunks; ++C) {
    const uptr Back = Min(C * ChunkSize, Reach);
    Entries[C] =
        MakeEntry((Chunks - C) * ChunkSize + Back, -static_cast<sptr>(Back));
  }

  InternalMmapVector<Symbol> Symbols;
  if (!ReadSymtab(Storage, StorageSize, LoadSize, &Symbols)) {
    GetAllocator().PublishEntries(Base, Entries.data(), Chunks);
    return;
  }
  Sort(Symbols.data(), Symbols.size(), ByValue);

  uptr Reached = 0;
  for (uptr N = 0; N < Symbols.size(); ++N) {
    const Symbol& Sym = Symbols[N];
    if (Sym.Guard)
      continue;
    uptr Span = RoundUpTo(Sym.Size, ChunkSize);
    uptr Next = N + 1;
    if (Next < Symbols.size() && Symbols[Next].Guard &&
        Symbols[Next].Value == Sym.Value + Span) {
      Span += RoundUpTo(Symbols[Next].Size, ChunkSize);
      ++Next;
    }
    uptr End = Sym.Value + Sym.Size;
    // A private run: chunk-aligned, not overlapping a neighbor, and its
    // guard (when present) occupies the following chunks. Anything packed
    // into a shared chunk keeps the whole-image entry.
    bool Alone =
        Sym.Value % ChunkSize == 0 && Reached <= Sym.Value &&
        (Next == Symbols.size() || Symbols[Next].Value >= Sym.Value + Span);
    Reached = End > Reached ? End : Reached;
    if (!Alone) {
      ++Stats->Shared;
      continue;
    }

    const uptr Obj = Base + Sym.Value;
    const uptr First = RoundDownTo(Obj, ChunkSize);
    const uptr Last = RoundDownTo(Obj + Span - 1, ChunkSize);
    if (!OffsetFits(static_cast<sptr>(Obj) - static_cast<sptr>(First)) ||
        !OffsetFits(static_cast<sptr>(Obj) - static_cast<sptr>(Last))) {
      ++Stats->Shared;
      continue;
    }

    for (uptr Chunk = First; Chunk <= Last; Chunk += ChunkSize)
      Entries[(Chunk - Base) / ChunkSize] = MakeEntry(
          Sym.Size, static_cast<sptr>(Obj) - static_cast<sptr>(Chunk));
    Globals.push_back({Base + Sym.Value, Sym.Size,
                       Sym.Name ? internal_strdup(Sym.Name) : nullptr});
    ++Stats->Described;
  }

  GetAllocator().PublishEntries(Base, Entries.data(), Chunks);
}

void ForgetGlobals() {
  for (uptr N = 0; N < Globals.size(); ++N)
    if (Globals[N].Name)
      InternalFree(const_cast<char*>(Globals[N].Name));
  Globals.clear();
}

void ForgetGlobals(uptr Beg, uptr Size) {
  uptr Dst = 0;
  for (uptr N = 0; N < Globals.size(); ++N) {
    if (Globals[N].Beg >= Beg && Globals[N].Beg - Beg < Size) {
      if (Globals[N].Name)
        InternalFree(const_cast<char*>(Globals[N].Name));
      continue;
    }
    if (Dst != N)
      Globals[Dst] = Globals[N];
    ++Dst;
  }
  Globals.resize(Dst);
}

void DropImage(uptr I) {
  TrackedImage& Img = Images[I];
  VaUnmap(Img.Base, Img.MapSize);
  VaDestroy(Img.Handle);
  GetAllocator().DeallocateSpan(Img.Base, Img.MapSize);
  ForgetGlobals(Img.Base, Img.LoadSize);
  if (I + 1 != Images.size())
    Images[I] = Images.back();
  Images.pop_back();
}

void DescribePlaced(TrackedImage* Img, const Offload::CodeObject& Obj) {
  if (Img->Described)
    return;
  ImageStats Stats = {};
  DescribeGlobals(Img->Base, Obj.LoadSize, Obj.Storage, Obj.StorageSize,
                  &Stats);
  Img->Described = true;
  VReport(1,
          "%s: placed 0x%zx (+%zu) at 0x%zx; %zu globals described, %zu "
          "chunks shared\n",
          SanitizerToolName, Img->LoadBase, Img->LoadSize, Img->Base,
          Stats.Described, Stats.Shared);
}

// Second mapping of a code object the loader already placed, sharing its
// pages through dma-buf. The alias has to be executable or kernels cannot
// run from it, and then their PC-relative globals are unchecked.
void AliasWithDmaBuf(const Offload::CodeObject& Obj) {
  PoolId Pool = PoolForAgent(Obj.Agent);
  if (!Pool)
    return;

  Allocator& A = GetAllocator();
  uptr Base = A.AllocateSpan(Pool, Obj.LoadSize);
  if (!Base) {
    VReport(1, "%s: no room in class zero for a %zu byte code object\n",
            SanitizerToolName, Obj.LoadSize);
    return;
  }

  u64 Handle = 0;
  uptr Off = 0;
  if (!VaShare(Obj.LoadBase, Obj.LoadSize, &Handle, &Off)) {
    A.DeallocateSpan(Base, Obj.LoadSize);
    return;
  }
  if (!VaMap(Base, Obj.LoadSize, Handle, Pool, Off)) {
    VaDestroy(Handle);
    A.DeallocateSpan(Base, Obj.LoadSize);
    VReport(1, "%s: could not map the alias for 0x%zx (+%zu) at 0x%zx\n",
            SanitizerToolName, Obj.LoadBase, Obj.LoadSize, Base);
    return;
  }

  ImageStats Stats = {};
  DescribeGlobals(Base, Obj.LoadSize, Obj.Storage, Obj.StorageSize, &Stats);
  Images.push_back(
      {Obj.LoadBase, Obj.LoadSize, Base, Obj.LoadSize, Handle, true});
  VReport(1,
          "%s: aliased 0x%zx (+%zu) at 0x%zx; %zu globals described, %zu "
          "chunks shared\n",
          SanitizerToolName, Obj.LoadBase, Obj.LoadSize, Base, Stats.Described,
          Stats.Shared);
}

void OnCodeObject(const Offload::CodeObject& Obj) {
  for (TrackedImage& Img : Images) {
    if (Img.LoadBase != Obj.LoadBase)
      continue;
    DescribePlaced(&Img, Obj);
    return;
  }

  if (!Offload::Get().DmaBufAliasExecutable()) {
    VReport(1,
            "%s: leaving 0x%zx (+%zu) with the loader; a dma-buf alias is not "
            "executable\n",
            SanitizerToolName, Obj.LoadBase, Obj.LoadSize);
    return;
  }
  AliasWithDmaBuf(Obj);
}

}  // namespace

bool ReserveExecutable(hsa_agent_t Agent, uptr LoadSize, uptr* Addr, uptr* Span,
                       u64* Handle) {
  PoolId Pool = PoolForAgent(Agent);
  if (!Pool || !LoadSize)
    return false;
  uptr Granule = VaGranule();
  if (!Granule)
    Granule = GetPageSizeCached();
  const uptr MapSize = RoundUpTo(LoadSize, Granule);
  Allocator& A = GetAllocator();
  uptr Base = A.AllocateSpan(Pool, MapSize);
  if (!Base)
    return false;
  u64 H = 0;
  if (!VaCreate(Pool, MapSize, HSA_AMD_VMEM_HANDLE_CREATE_EXECUTABLE, &H) ||
      !VaMap(Base, MapSize, H, Pool, 0, /*Executable=*/true)) {
    if (H)
      VaDestroy(H);
    A.DeallocateSpan(Base, MapSize);
    return false;
  }
  *Addr = Base;
  *Span = MapSize;
  *Handle = H;
  return true;
}

void ReleaseExecutable(uptr Addr, uptr Span, u64 Handle) {
  VaUnmap(Addr, Span);
  VaDestroy(Handle);
  GetAllocator().DeallocateSpan(Addr, Span);
}

void NotePlacedCodeObject(uptr Addr, uptr LoadSize, uptr MapSize, u64 Handle) {
  Images.push_back({Addr, LoadSize, Addr, MapSize, Handle, false});
}

void TrackExecutable(hsa_executable_t Exec) {
  if (!flags()->alias_globals || !PlatformReady())
    return;
  Offload::Get().ForEachCodeObject(Exec, OnCodeObject);
}

void UntrackExecutable(hsa_executable_t Exec) {
  Offload::Get().ForEachCodeObject(Exec, [](const Offload::CodeObject& Obj) {
    for (uptr I = 0; I < Images.size(); ++I) {
      if (Images[I].LoadBase != Obj.LoadBase)
        continue;
      VReport(1, "%s: released 0x%zx (+%zu) at 0x%zx\n", SanitizerToolName,
              Images[I].LoadBase, Images[I].LoadSize, Images[I].Base);
      DropImage(I);
      return;
    }
  });
}

void UntrackAll() {
  while (Images.size()) DropImage(Images.size() - 1);
  ForgetGlobals();
}

uptr TrackedAddress(uptr Addr) {
  for (const TrackedImage& Img : Images)
    if (Addr >= Img.LoadBase && Addr < Img.LoadBase + Img.LoadSize)
      return Img.Base + (Addr - Img.LoadBase);
  return Addr;
}

uptr LoaderAddress(uptr Addr) {
  for (const TrackedImage& Img : Images)
    if (Addr >= Img.Base && Addr < Img.Base + Img.LoadSize)
      return Img.LoadBase + (Addr - Img.Base);
  return Addr;
}

bool LookupGlobal(uptr Beg, GlobalInfo* Out) {
  for (const GlobalInfo& G : Globals) {
    if (G.Beg == Beg) {
      *Out = G;
      return true;
    }
  }
  return false;
}

}  // namespace __dasan
