//===-- dasan_image.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On freeze: describe globals, keep the dma-buf alias. ELF is only the
// symbol table in the loader's storage blob.
//
//===----------------------------------------------------------------------===//

#include "dasan_image.h"

#include <elf.h>

#include "dasan_allocator.h"
#include "dasan_mapping.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_posix.h"

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

InternalMmapVectorNoCtor<GlobalInfo> Globals;
InternalMmapVectorNoCtor<AliasedImage> Images;

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

void DescribeGlobals(uptr Alias, uptr LoadSize, const void* Storage,
                     uptr StorageSize, ImageStats* Stats) {
  const uptr ChunkSize = ClassIdToSize(0);
  const uptr Chunks = RoundUpTo(LoadSize, ChunkSize) / ChunkSize;
  Allocator& A = GetAllocator();

  const uptr Reach = static_cast<uptr>(-kOffsetMin);
  for (uptr C = 0; C < Chunks; ++C) {
    const uptr Back = Min(C * ChunkSize, Reach);
    A.SetImageEntry(
        Alias + C * ChunkSize,
        MakeEntry((Chunks - C) * ChunkSize + Back, -static_cast<sptr>(Back)));
  }

  InternalMmapVector<Symbol> Symbols;
  if (!ReadSymtab(Storage, StorageSize, LoadSize, &Symbols)) {
    A.PublishEntries();
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
    bool Alone =
        Sym.Value % ChunkSize == 0 && Reached <= Sym.Value &&
        (Next == Symbols.size() || Symbols[Next].Value >= Sym.Value + Span);
    Reached = End > Reached ? End : Reached;
    if (!Alone) {
      ++Stats->Shared;
      continue;
    }

    const uptr Obj = Alias + Sym.Value;
    const uptr First = RoundDownTo(Obj, ChunkSize);
    const uptr Last = RoundDownTo(Obj + Span - 1, ChunkSize);
    bool Describable = true;
    for (uptr Chunk = First; Chunk <= Last; Chunk += ChunkSize)
      if (!OffsetFits(static_cast<sptr>(Obj) - static_cast<sptr>(Chunk)))
        Describable = false;

    if (!Describable) {
      ++Stats->Shared;
      continue;
    }

    for (uptr Chunk = First; Chunk <= Last; Chunk += ChunkSize)
      A.SetImageEntry(Chunk, MakeEntry(Sym.Size, static_cast<sptr>(Obj) -
                                                     static_cast<sptr>(Chunk)));
    Globals.push_back({Alias + Sym.Value, Sym.Size, Span,
                       Sym.Name ? internal_strdup(Sym.Name) : nullptr});
    ++Stats->Described;
  }

  A.PublishEntries();
}

}  // namespace

bool LookupGlobal(uptr Addr, GlobalInfo* Out) {
  const uptr ChunkSize = ClassIdToSize(0);
  const GlobalInfo* Over = nullptr;
  const GlobalInfo* Under = nullptr;
  for (uptr N = 0; N < Globals.size(); ++N) {
    const GlobalInfo& G = Globals[N];
    if (Addr >= G.Beg && Addr - G.Beg < G.Reach)
      Over = &G;
    else if (G.Beg > Addr && G.Beg - Addr <= ChunkSize &&
             (!Under || G.Beg < Under->Beg))
      Under = &G;
  }
  if (!Over) {
    if (!Under)
      return false;
    *Out = *Under;
    return true;
  }

  *Out = *Over;
  if (Under && Addr >= Over->Beg + Over->Size &&
      Under->Beg - Addr < Addr - (Over->Beg + Over->Size))
    *Out = *Under;
  return true;
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

bool AlreadyAliased(uptr LoadBase) {
  for (uptr I = 0; I < Images.size(); ++I)
    if (Images[I].LoadBase == LoadBase)
      return true;
  return false;
}

uptr AliasOf(uptr Addr) {
  for (uptr I = 0; I < Images.size(); ++I) {
    const AliasedImage& Img = Images[I];
    if (Addr >= Img.LoadBase && Addr < Img.LoadBase + Img.LoadSize)
      return Img.Alias + (Addr - Img.LoadBase);
  }
  return Addr;
}

void TrackImage(uptr LoadBase, uptr LoadSize, uptr Alias, u64 Handle,
                const void* Storage, uptr StorageSize, ImageStats* Stats) {
  ImageStats Local = {};
  if (!Stats)
    Stats = &Local;
  DescribeGlobals(Alias, LoadSize, Storage, StorageSize, Stats);

  AliasedImage Img{};
  Img.LoadBase = LoadBase;
  Img.LoadSize = LoadSize;
  Img.Alias = Alias;
  Img.Handle = Handle;
  if (common_flags()->symbolize && Storage && StorageSize) {
    Img.Bytes = MmapOrDie(StorageSize, "dasan image");
    Img.BytesSize = StorageSize;
    internal_memcpy(Img.Bytes, Storage, StorageSize);
  }
  Images.push_back(Img);
}

void DropAliasedImage(uptr I) {
  AliasedImage& Img = Images[I];
  if (Img.Path[0])
    internal_unlink(Img.Path);
  if (Img.Bytes)
    UnmapOrDie(Img.Bytes, Img.BytesSize);
  if (I + 1 != Images.size())
    Images[I] = Images.back();
  Images.pop_back();
}

uptr NumAliasedImages() { return Images.size(); }

AliasedImage& AliasedImageAt(uptr I) { return Images[I]; }

void ForgetImages() {
  while (Images.size()) DropAliasedImage(Images.size() - 1);
}

}  // namespace __dasan
