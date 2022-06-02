#include "OffloadDump.h"
#include "llvm-objdump.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::objdump;

#define OFFLOAD_SECTION_MAGIC_STR ".llvm.offloading"

/// Get the printable name of the image kind.
static StringRef getImageName(const OffloadBinary *OB) {
  switch (OB->getImageKind()) {
  case IMG_Object:
    return "elf";
  case IMG_Bitcode:
    return "llvm ir";
  case IMG_Cubin:
    return "cubin";
  case IMG_Fatbinary:
    return "fatbinary";
  case IMG_PTX:
    return "ptx";
  default:
    return "<None>";
  }
}

static void printBinary(const OffloadBinary *OB, uint64_t Index) {
  outs() << "\nOFFLOADING IMAGE "
         << "[" << Index << "]:\n";
  outs() << left_justify("kind", 16) << getImageName(OB) << "\n";
  outs() << left_justify("arch", 16) << OB->getArch() << "\n";
  outs() << left_justify("triple", 16) << OB->getTriple() << "\n";
  outs() << left_justify("producer", 16)
         << getOffloadKindName(OB->getOffloadKind()) << "\n";
}

static Error visitAllBinaries(const OffloadBinary *OB) {
  uint64_t Offset = 0;
  uint64_t Index = 0;
  while (Offset < OB->getMemoryBufferRef().getBufferSize()) {
    MemoryBufferRef Buffer =
        MemoryBufferRef(OB->getData().drop_front(Offset), OB->getFileName());
    auto BinaryOrErr = OffloadBinary::create(Buffer);
    if (!BinaryOrErr)
      return BinaryOrErr.takeError();

    OffloadBinary &Binary = **BinaryOrErr;

    printBinary(&Binary, Index++);

    Offset += Binary.getSize();
  }
  return Error::success();
}

/// Print the embedded offloading contents of an ObjectFile \p O.
void llvm::dumpOffloadBinary(const ObjectFile *O) {
  for (const SectionRef &Sec : O->sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (!Name || !Name->startswith(OFFLOAD_SECTION_MAGIC_STR))
      continue;

    Expected<StringRef> Contents = Sec.getContents();
    if (!Contents)
      reportError(Contents.takeError(), O->getFileName());

    MemoryBufferRef Buffer = MemoryBufferRef(*Contents, O->getFileName());
    auto BinaryOrErr = OffloadBinary::create(Buffer);
    if (!BinaryOrErr)
      reportError(BinaryOrErr.takeError(), O->getFileName());
    OffloadBinary &Binary = **BinaryOrErr;

    if (Error Err = visitAllBinaries(&Binary))
      handleAllErrors(std::move(Err), [](const ErrorInfoBase &EI) {});
  }
}

/// Print the contents of an offload binary file \p OB. This may contain
/// multiple binaries stored in the same buffer.
void llvm::dumpOffloadSections(const OffloadBinary *OB) {
  if (Error Err = visitAllBinaries(OB))
    handleAllErrors(std::move(Err), [](const ErrorInfoBase &EI) {});
}
