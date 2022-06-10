//===---------- OffloadBinary.h - Offload Binary Parser -- C++ ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains a generic class meant to parse data out of an input offloading
// binary. This binary format embeds an executable or linkable image inside
// along with some other metadata formatted as a string map.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <memory>
#include <unordered_map>

class OffloadBinary {
  struct Header {
    uint8_t Magic[4] = {0x10, 0xFF, 0x10, 0xAD}; // 0x10FF10AD magic bytes.
    uint32_t Version = 1;                        // Version identifier.
    uint64_t Size;        // Size in bytes of this entire binary.
    uint64_t EntryOffset; // Offset of the metadata entry in bytes.
    uint64_t EntrySize;   // Size of the metadata entry in bytes.
  };

  struct Entry {
    uint16_t TheImageKind;   // The kind of the image stored.
    uint16_t TheOffloadKind; // The producer of this image.
    uint32_t Flags;          // Additional flags associated with the image.
    uint64_t StringOffset;   // Offset in bytes to the string map.
    uint64_t NumStrings;     // Number of entries in the string map.
    uint64_t ImageOffset;    // Offset in bytes of the actual binary image.
    uint64_t ImageSize;      // Size in bytes of the binary image.
  };

  struct StringEntry {
    uint64_t KeyOffset;   // Offset of the key in the string table.
    uint64_t ValueOffset; // Offset of the value in the string table.
  };

public:
  static bool isValid(char *Buffer) {
    return !strncmp(Buffer, "\x10\xFF\x10\xAD", sizeof(Header::Magic)) &&
           reinterpret_cast<Header *>(Buffer)->Version == 1;
  }

  static std::unique_ptr<OffloadBinary> create(char *Buffer) {
    if (!isValid(Buffer))
      return nullptr;

    const Header *TheHeader = reinterpret_cast<const Header *>(Buffer);
    const Entry *TheEntry =
        reinterpret_cast<const Entry *>(&Buffer[TheHeader->EntryOffset]);

    return std::make_unique<OffloadBinary>(
        OffloadBinary(Buffer, TheHeader, TheEntry));
  }

  uint16_t getImageKind() const { return TheEntry->TheImageKind; }
  uint16_t getOffloadKind() const { return TheEntry->TheOffloadKind; }
  uint32_t getVersion() const { return TheHeader->Version; }
  uint32_t getFlags() const { return TheEntry->Flags; }
  uint64_t getSize() const { return TheHeader->Size; }

  const char *getTriple() const { return getString("triple"); }
  const char *getArch() const { return getString("arch"); }
  char *getImageStart() const { return &Buffer[TheEntry->ImageOffset]; }
  char *getImageEnd() const { return getImageStart() + getImageSize(); }
  uint64_t getImageSize() const { return TheEntry->ImageSize; }

private:
  OffloadBinary(char *Buffer, const Header *TheHeader, const Entry *TheEntry)
      : Buffer(Buffer), TheHeader(TheHeader), TheEntry(TheEntry) {
    const StringEntry *StringMapBegin =
        reinterpret_cast<const StringEntry *>(&Buffer[TheEntry->StringOffset]);
    for (uint64_t I = 0, E = TheEntry->NumStrings; I != E; ++I) {
      const char *Key = &Buffer[StringMapBegin[I].KeyOffset];
      StringData[Key] = &Buffer[StringMapBegin[I].ValueOffset];
    }
  }

  const char *getString(const char *Key) const {
    return (StringData.count(Key)) ? StringData.at(Key) : "";
  }

  /// Map from keys to offsets in the binary.
  std::unordered_map<std::string, const char *> StringData;
  /// Raw pointer to the MemoryBufferRef for convenience.
  char *Buffer;
  /// Location of the header within the binary.
  const Header *TheHeader;
  /// Location of the metadata entries within the binary.
  const Entry *TheEntry;
};
