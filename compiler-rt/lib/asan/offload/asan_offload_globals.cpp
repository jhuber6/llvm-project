//===-- asan_offload_globals.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Poison the redzones of device global variables. The instrumentation emits a
// constructor that calls '__asan_register_globals', but HIP forbids dynamic
// initialization of device variables and never runs it, so the host walks the
// 'asan_globals' section of the loaded image and does the work itself.
//
//===----------------------------------------------------------------------===//

#include "asan_offload_globals.h"

#include "asan_interface_internal.h"
#include "asan_mapping.h"
#include "asan_poisoning.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <elf.h>

using namespace __sanitizer;

namespace __asan {
namespace {

// Relocated pointer fields read zero in the image, so the addend of the
// relative relocation covering a field is the only place its value exists.
constexpr u32 kRelative64 = 13;

struct Image {
  const char *Base;
  uptr Size;
  const Elf64_Shdr *Sections;
  u16 SectionCount;
  const char *Names;
  uptr NamesSize;

  bool InRange(uptr Off, uptr Len) const {
    return Off <= Size && Len <= Size - Off;
  }
};

bool OpenImage(const void *Bytes, uptr Size, Image *Img) {
  if (!Bytes || Size < sizeof(Elf64_Ehdr))
    return false;

  const char *Base = reinterpret_cast<const char *>(Bytes);
  const Elf64_Ehdr *Hdr = reinterpret_cast<const Elf64_Ehdr *>(Base);
  if (internal_memcmp(Hdr->e_ident, ELFMAG, SELFMAG) ||
      Hdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      Hdr->e_ident[EI_DATA] != ELFDATA2LSB)
    return false;
  if (!Hdr->e_shoff || Hdr->e_shentsize != sizeof(Elf64_Shdr) ||
      Hdr->e_shstrndx >= Hdr->e_shnum)
    return false;

  Img->Base = Base;
  Img->Size = Size;
  if (!Img->InRange(Hdr->e_shoff, (uptr)Hdr->e_shnum * sizeof(Elf64_Shdr)))
    return false;
  Img->Sections = reinterpret_cast<const Elf64_Shdr *>(Base + Hdr->e_shoff);
  Img->SectionCount = Hdr->e_shnum;

  const Elf64_Shdr &Str = Img->Sections[Hdr->e_shstrndx];
  if (!Img->InRange(Str.sh_offset, Str.sh_size))
    return false;
  Img->Names = Base + Str.sh_offset;
  Img->NamesSize = Str.sh_size;
  return true;
}

const Elf64_Shdr *FindSection(const Image &Img, const char *Name) {
  for (u16 I = 0; I < Img.SectionCount; ++I) {
    const Elf64_Shdr &S = Img.Sections[I];
    if (S.sh_name >= Img.NamesSize)
      continue;
    if (!internal_strcmp(Img.Names + S.sh_name, Name))
      return &S;
  }
  return nullptr;
}

// The value a relocated pointer field at 'Vaddr' will hold once loaded, or zero
// if nothing relocates it.
uptr RelocatedValue(const Image &Img, uptr Vaddr) {
  for (u16 I = 0; I < Img.SectionCount; ++I) {
    const Elf64_Shdr &S = Img.Sections[I];
    if (S.sh_type != SHT_RELA || S.sh_entsize != sizeof(Elf64_Rela))
      continue;
    if (!Img.InRange(S.sh_offset, S.sh_size))
      continue;
    const Elf64_Rela *R =
        reinterpret_cast<const Elf64_Rela *>(Img.Base + S.sh_offset);
    for (uptr J = 0, N = S.sh_size / sizeof(Elf64_Rela); J < N; ++J) {
      if (R[J].r_offset != Vaddr)
        continue;
      if (ELF64_R_TYPE(R[J].r_info) != kRelative64)
        continue;
      return (uptr)R[J].r_addend;
    }
  }
  return 0;
}

// Where a loaded address lives in the image file, for reading the strings that
// the descriptors point at.
const char *AtVaddr(const Image &Img, uptr Vaddr) {
  for (u16 I = 0; I < Img.SectionCount; ++I) {
    const Elf64_Shdr &S = Img.Sections[I];
    if (!(S.sh_flags & SHF_ALLOC) || S.sh_type == SHT_NOBITS || !S.sh_size)
      continue;
    if (Vaddr < S.sh_addr || Vaddr >= S.sh_addr + S.sh_size)
      continue;
    uptr Off = S.sh_offset + (Vaddr - S.sh_addr);
    return Img.InRange(Off, 1) ? Img.Base + Off : nullptr;
  }
  return nullptr;
}

void PoisonRedzones(uptr Beg, uptr Size, uptr SizeWithRedzone, bool Poison) {
  if (!Size || SizeWithRedzone < Size)
    return;
  if (!AddrIsInMem(Beg) || !AddrIsInMem(Beg + SizeWithRedzone - 1))
    return;
  if (!AddrIsAlignedByGranularity(Beg))
    return;

  if (!Poison) {
    FastPoisonShadow(Beg, RoundUpTo(SizeWithRedzone, ASAN_SHADOW_GRANULARITY),
                     0);
    return;
  }

  uptr Aligned = RoundUpTo(Size, ASAN_SHADOW_GRANULARITY);
  if (SizeWithRedzone > Aligned)
    FastPoisonShadow(Beg + Aligned, SizeWithRedzone - Aligned,
                     kAsanGlobalRedzoneMagic);
  if (Size != Aligned)
    FastPoisonShadowPartialRightRedzone(
        Beg + RoundDownTo(Size, ASAN_SHADOW_GRANULARITY),
        Size % ASAN_SHADOW_GRANULARITY, ASAN_SHADOW_GRANULARITY,
        kAsanGlobalRedzoneMagic);
}

struct Table {
  Image Img;
  const Elf64_Shdr *Sec;
  uptr Count;
};

bool OpenTable(const void *Bytes, uptr Size, Table *T) {
  if (!OpenImage(Bytes, Size, &T->Img))
    return false;
  T->Sec = FindSection(T->Img, "asan_globals");
  if (!T->Sec || T->Sec->sh_type != SHT_PROGBITS ||
      !T->Img.InRange(T->Sec->sh_offset, T->Sec->sh_size))
    return false;
  T->Count = T->Sec->sh_size / sizeof(__asan_global);
  return T->Count != 0;
}

// The 'beg' and 'name' fields are relocated pointers, so they read zero in the
// image and their values have to come from the relocation that covers them.
const __asan_global *ReadEntry(const Table &T, uptr I, uptr *Beg) {
  uptr Vaddr = T.Sec->sh_addr + I * sizeof(__asan_global);
  const __asan_global *G = reinterpret_cast<const __asan_global *>(
      T.Img.Base + T.Sec->sh_offset + I * sizeof(__asan_global));
  *Beg = G->beg ? (uptr)G->beg : RelocatedValue(T.Img, Vaddr);
  return *Beg ? G : nullptr;
}

} // namespace

void PoisonDeviceGlobals(uptr LoadBase, const void *Bytes, uptr Size,
                         bool Poison) {
  Table T;
  if (!OpenTable(Bytes, Size, &T))
    return;

  for (uptr I = 0; I < T.Count; ++I) {
    uptr Beg = 0;
    if (const __asan_global *G = ReadEntry(T, I, &Beg))
      PoisonRedzones(LoadBase + Beg, G->size, G->size_with_redzone, Poison);
  }

  VReport(2, "%s: %s %zu device global(s) in image 0x%zx\n", SanitizerToolName,
          Poison ? "poisoned" : "unpoisoned", T.Count, LoadBase);
}

bool FindDeviceGlobal(uptr LoadBase, const void *Bytes, uptr Size, uptr Addr,
                      DeviceGlobalInfo *Out) {
  Table T;
  if (!OpenTable(Bytes, Size, &T))
    return false;

  for (uptr I = 0; I < T.Count; ++I) {
    uptr Beg = 0;
    const __asan_global *G = ReadEntry(T, I, &Beg);
    if (!G)
      continue;
    uptr Run = LoadBase + Beg;
    if (Addr < Run || Addr >= Run + G->size_with_redzone)
      continue;

    Out->Beg = Run;
    Out->Size = G->size;
    Out->Name[0] = '\0';
    uptr NameVaddr =
        G->name ? (uptr)G->name
                : RelocatedValue(T.Img, T.Sec->sh_addr +
                                            I * sizeof(__asan_global) +
                                            __builtin_offsetof(__asan_global,
                                                               name));
    if (const char *Name = AtVaddr(T.Img, NameVaddr))
      internal_strlcpy(Out->Name, Name, sizeof(Out->Name));
    return true;
  }
  return false;
}

} // namespace __asan
