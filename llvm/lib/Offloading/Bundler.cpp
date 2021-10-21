#include "llvm/Offloading/Bundler.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

/// Magic string that marks the existence of offloading data.
#define OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"

using namespace llvm;
using namespace llvm::object;

enum class CudaVersion {
  UNKNOWN,
  CUDA_70,
  CUDA_75,
  CUDA_80,
  CUDA_90,
  CUDA_91,
  CUDA_92,
  CUDA_100,
  CUDA_101,
  CUDA_102,
  CUDA_110,
  CUDA_111,
  CUDA_112,
  CUDA_113,
  CUDA_114,
  FULLY_SUPPORTED = CUDA_114,
  PARTIALLY_SUPPORTED =
      CUDA_114, // Partially supported. Proceed with a warning.
  NEW = 10000,  // Too new. Issue a warning, but allow using it.
};

enum class CudaArch {
  UNUSED,
  UNKNOWN,
  SM_20,
  SM_21,
  SM_30,
  SM_32,
  SM_35,
  SM_37,
  SM_50,
  SM_52,
  SM_53,
  SM_60,
  SM_61,
  SM_62,
  SM_70,
  SM_72,
  SM_75,
  SM_80,
  SM_86,
  GFX600,
  GFX601,
  GFX602,
  GFX700,
  GFX701,
  GFX702,
  GFX703,
  GFX704,
  GFX705,
  GFX801,
  GFX802,
  GFX803,
  GFX805,
  GFX810,
  GFX900,
  GFX902,
  GFX904,
  GFX906,
  GFX908,
  GFX909,
  GFX90a,
  GFX90c,
  GFX1010,
  GFX1011,
  GFX1012,
  GFX1013,
  GFX1030,
  GFX1031,
  GFX1032,
  GFX1033,
  GFX1034,
  GFX1035,
  LAST,
};

struct CudaArchToStringMap {
  CudaArch arch;
  const char *arch_name;
  const char *virtual_arch_name;
};

#define SM2(sm, ca)                                                            \
  { CudaArch::SM_##sm, "sm_" #sm, ca }
#define SM(sm) SM2(sm, "compute_" #sm)
#define GFX(gpu)                                                               \
  { CudaArch::GFX##gpu, "gfx" #gpu, "compute_amdgcn" }
static const CudaArchToStringMap arch_names[] = {
    // clang-format off
    {CudaArch::UNUSED, "", ""},
    SM2(20, "compute_20"), SM2(21, "compute_20"), // Fermi
    SM(30), SM(32), SM(35), SM(37),  // Kepler
    SM(50), SM(52), SM(53),          // Maxwell
    SM(60), SM(61), SM(62),          // Pascal
    SM(70), SM(72),                  // Volta
    SM(75),                          // Turing
    SM(80), SM(86),                  // Ampere
    GFX(600),  // gfx600
    GFX(601),  // gfx601
    GFX(602),  // gfx602
    GFX(700),  // gfx700
    GFX(701),  // gfx701
    GFX(702),  // gfx702
    GFX(703),  // gfx703
    GFX(704),  // gfx704
    GFX(705),  // gfx705
    GFX(801),  // gfx801
    GFX(802),  // gfx802
    GFX(803),  // gfx803
    GFX(805),  // gfx805
    GFX(810),  // gfx810
    GFX(900),  // gfx900
    GFX(902),  // gfx902
    GFX(904),  // gfx903
    GFX(906),  // gfx906
    GFX(908),  // gfx908
    GFX(909),  // gfx909
    GFX(90a),  // gfx90a
    GFX(90c),  // gfx90c
    GFX(1010), // gfx1010
    GFX(1011), // gfx1011
    GFX(1012), // gfx1012
    GFX(1013), // gfx1013
    GFX(1030), // gfx1030
    GFX(1031), // gfx1031
    GFX(1032), // gfx1032
    GFX(1033), // gfx1033
    GFX(1034), // gfx1034
    GFX(1035), // gfx1035
    // clang-format on
};

static CudaArch StringToCudaArch(llvm::StringRef S) {
  auto result = std::find_if(
      std::begin(arch_names), std::end(arch_names),
      [S](const CudaArchToStringMap &map) { return S == map.arch_name; });
  if (result == std::end(arch_names))
    return CudaArch::UNKNOWN;
  return result->arch;
}

struct OffloadTargetInfo {
  StringRef OffloadKind;
  llvm::Triple Triple;
  StringRef GPUArch;

  OffloadTargetInfo(const StringRef Target) {
    auto TargetFeatures = Target.split(':');
    auto TripleOrGPU = TargetFeatures.first.rsplit('-');

    if (StringToCudaArch(TripleOrGPU.second) != CudaArch::UNKNOWN) {
      auto KindTriple = TripleOrGPU.first.split('-');
      this->OffloadKind = KindTriple.first;
      this->Triple = llvm::Triple(KindTriple.second);
      this->GPUArch = Target.substr(Target.find(TripleOrGPU.second));
    } else {
      auto KindTriple = TargetFeatures.first.split('-');
      this->OffloadKind = KindTriple.first;
      this->Triple = llvm::Triple(KindTriple.second);
      this->GPUArch = "";
    }
  }

  bool hasHostKind() const { return this->OffloadKind == "host"; }

  bool isOffloadKindValid() const {
    return OffloadKind == "host" || OffloadKind == "openmp" ||
           OffloadKind == "hip" || OffloadKind == "hipv4";
  }

  bool isTripleValid() const {
    return !Triple.str().empty() && Triple.getArch() != Triple::UnknownArch;
  }

  bool operator==(const OffloadTargetInfo &Target) const {
    return OffloadKind == Target.OffloadKind &&
           Triple.isCompatibleWith(Target.Triple) && GPUArch == Target.GPUArch;
  }

  std::string str() {
    return Twine(OffloadKind + "-" + Triple.str() + "-" + GPUArch).str();
  }
};

/// Generic file handler interface.
class FileHandler {
public:
  struct BundleInfo {
    StringRef BundleID;
  };

  FileHandler() {}

  virtual ~FileHandler() {}

  /// Update the file handler with information from the header of the bundled
  /// file.
  virtual Error ReadHeader(MemoryBuffer &Input) = 0;

  /// Read the marker of the next bundled to be read in the file. The bundle
  /// name is returned if there is one in the file, or `None` if there are no
  /// more bundles to be read.
  virtual Expected<Optional<StringRef>>
  ReadBundleStart(MemoryBuffer &Input) = 0;

  /// Read the marker that closes the current bundle.
  virtual Error ReadBundleEnd(MemoryBuffer &Input) = 0;

  /// Read the current bundle and write the result into the stream \a OS.
  virtual Error ReadBundle(raw_ostream &OS, MemoryBuffer &Input) = 0;

  /// List bundle IDs in \a Input.
  virtual Error listBundleIDs(MemoryBuffer &Input) {
    if (Error Err = ReadHeader(Input))
      return Err;

    return forEachBundle(Input, [&](const BundleInfo &Info) -> Error {
      llvm::outs() << Info.BundleID << '\n';
      Error Err = listBundleIDsCallback(Input, Info);
      if (Err)
        return Err;
      return Error::success();
    });
  }

  /// For each bundle in \a Input, do \a Func.
  Error forEachBundle(MemoryBuffer &Input,
                      std::function<Error(const BundleInfo &)> Func) {
    while (true) {
      Expected<Optional<StringRef>> CurTripleOrErr = ReadBundleStart(Input);
      if (!CurTripleOrErr)
        return CurTripleOrErr.takeError();

      // No more bundles.
      if (!*CurTripleOrErr)
        break;

      StringRef CurTriple = **CurTripleOrErr;
      assert(!CurTriple.empty());

      BundleInfo Info{CurTriple};
      if (Error Err = Func(Info))
        return Err;
    }
    return Error::success();
  }

protected:
  virtual Error listBundleIDsCallback(MemoryBuffer &Input,
                                      const BundleInfo &Info) {
    return Error::success();
  }
};

/// Handler for object files. The bundles are organized by sections with a
/// designated name.
///
/// To unbundle, we just copy the contents of the designated section.
class ObjectFileHandler final : public FileHandler {

  /// The object file we are currently dealing with.
  std::unique_ptr<ObjectFile> Obj;

  /// Return the input file contents.
  StringRef getInputFileContents() const { return Obj->getData(); }

  /// Return bundle name (<kind>-<triple>) if the provided section is an offload
  /// section.
  static Expected<Optional<StringRef>> IsOffloadSection(SectionRef CurSection) {
    Expected<StringRef> NameOrErr = CurSection.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    // If it does not start with the reserved suffix, just skip this section.
    if (!NameOrErr->startswith(OFFLOAD_BUNDLER_MAGIC_STR))
      return None;

    // Return the triple that is right after the reserved prefix.
    return NameOrErr->substr(sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  }

  /// Total number of inputs.
  unsigned NumberOfInputs = 0;

  /// Total number of processed inputs, i.e, inputs that were already
  /// read from the buffers.
  unsigned NumberOfProcessedInputs = 0;

  /// Iterator of the current and next section.
  section_iterator CurrentSection;
  section_iterator NextSection;

public:
  ObjectFileHandler(std::unique_ptr<ObjectFile> ObjIn)
      : FileHandler(), Obj(std::move(ObjIn)),
        CurrentSection(Obj->section_begin()),
        NextSection(Obj->section_begin()) {}

  ~ObjectFileHandler() final {}

  Error ReadHeader(MemoryBuffer &Input) final { return Error::success(); }

  Expected<Optional<StringRef>> ReadBundleStart(MemoryBuffer &Input) final {
    while (NextSection != Obj->section_end()) {
      CurrentSection = NextSection;
      ++NextSection;

      // Check if the current section name starts with the reserved prefix. If
      // so, return the triple.
      Expected<Optional<StringRef>> TripleOrErr =
          IsOffloadSection(*CurrentSection);
      if (!TripleOrErr)
        return TripleOrErr.takeError();
      if (*TripleOrErr)
        return **TripleOrErr;
    }
    return None;
  }

  Error ReadBundleEnd(MemoryBuffer &Input) final { return Error::success(); }

  Error ReadBundle(raw_ostream &OS, MemoryBuffer &Input) final {
    Expected<StringRef> ContentOrErr = CurrentSection->getContents();
    if (!ContentOrErr)
      return ContentOrErr.takeError();
    StringRef Content = *ContentOrErr;

    // Copy fat object contents to the output when extracting host bundle.
    if (Content.size() == 1u && Content.front() == 0)
      Content = StringRef(Input.getBufferStart(), Input.getBufferSize());

    OS.write(Content.data(), Content.size());
    return Error::success();
  }

private:
  static Error executeObjcopy(StringRef Objcopy, ArrayRef<StringRef> Args) {
    // If the user asked for the commands to be printed out, we do that
    // instead of executing it.
    if (sys::ExecuteAndWait(Objcopy, Args))
      return createStringError(inconvertibleErrorCode(),
                               "'llvm-objcopy' tool failed");
    return Error::success();
  }
};

/// Read 8-byte integers from a buffer in little-endian format.
static uint64_t Read8byteIntegerFromBuffer(StringRef Buffer, size_t pos) {
  uint64_t Res = 0;
  const char *Data = Buffer.data();

  for (unsigned i = 0; i < 8; ++i) {
    Res <<= 8;
    uint64_t Char = (uint64_t)Data[pos + 7 - i];
    Res |= 0xffu & Char;
  }
  return Res;
}

class BinaryFileHandler final : public FileHandler {
  /// Information about the bundles extracted from the header.
  struct BinaryBundleInfo final : public BundleInfo {
    /// Size of the bundle.
    uint64_t Size = 0u;
    /// Offset at which the bundle starts in the bundled file.
    uint64_t Offset = 0u;

    BinaryBundleInfo() {}
    BinaryBundleInfo(uint64_t Size, uint64_t Offset)
        : Size(Size), Offset(Offset) {}
  };

  /// Map between a triple and the corresponding bundle information.
  StringMap<BinaryBundleInfo> BundlesInfo;

  /// Iterator for the bundle information that is being read.
  StringMap<BinaryBundleInfo>::iterator CurBundleInfo;
  StringMap<BinaryBundleInfo>::iterator NextBundleInfo;

public:
  BinaryFileHandler() : FileHandler() {}

  ~BinaryFileHandler() final {}

  Error ReadHeader(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Initialize the current bundle with the end of the container.
    CurBundleInfo = BundlesInfo.end();

    // Check if buffer is smaller than magic string.
    size_t ReadChars = sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;
    if (ReadChars > FC.size())
      return Error::success();

    // Check if no magic was found.
    StringRef Magic(FC.data(), sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
      return Error::success();

    // Read number of bundles.
    if (ReadChars + 8 > FC.size())
      return Error::success();

    uint64_t NumberOfBundles = Read8byteIntegerFromBuffer(FC, ReadChars);
    ReadChars += 8;

    // Read bundle offsets, sizes and triples.
    for (uint64_t i = 0; i < NumberOfBundles; ++i) {

      // Read offset.
      if (ReadChars + 8 > FC.size())
        return Error::success();

      uint64_t Offset = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read size.
      if (ReadChars + 8 > FC.size())
        return Error::success();

      uint64_t Size = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple size.
      if (ReadChars + 8 > FC.size())
        return Error::success();

      uint64_t TripleSize = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple.
      if (ReadChars + TripleSize > FC.size())
        return Error::success();

      StringRef Triple(&FC.data()[ReadChars], TripleSize);
      ReadChars += TripleSize;

      // Check if the offset and size make sense.
      if (!Offset || Offset + Size > FC.size())
        return Error::success();

      assert(BundlesInfo.find(Triple) == BundlesInfo.end() &&
             "Triple is duplicated??");
      BundlesInfo[Triple] = BinaryBundleInfo(Size, Offset);
    }
    // Set the iterator to where we will start to read.
    CurBundleInfo = BundlesInfo.end();
    NextBundleInfo = BundlesInfo.begin();
    return Error::success();
  }

  Expected<Optional<StringRef>> ReadBundleStart(MemoryBuffer &Input) final {
    if (NextBundleInfo == BundlesInfo.end())
      return None;
    CurBundleInfo = NextBundleInfo++;
    return CurBundleInfo->first();
  }

  Error ReadBundleEnd(MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    return Error::success();
  }

  Error ReadBundle(raw_ostream &OS, MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    StringRef FC = Input.getBuffer();
    OS.write(FC.data() + CurBundleInfo->second.Offset,
             CurBundleInfo->second.Size);
    return Error::success();
  }
};

/// Return an appropriate object file handler. We use the specific object
/// handler if we know how to deal with that format, otherwise we use a default
/// binary file handler.
static std::unique_ptr<FileHandler>
CreateObjectFileHandler(MemoryBuffer &FirstInput) {
  // Check if the input file format is one that we know how to deal with.
  Expected<std::unique_ptr<Binary>> BinaryOrErr = createBinary(FirstInput);

  // We only support regular object files. If failed to open the input as a
  // known binary or this is not an object file use the default binary handler.
  if (errorToBool(BinaryOrErr.takeError()) || !isa<ObjectFile>(*BinaryOrErr))
    return std::make_unique<BinaryFileHandler>();

  // Otherwise create an object file handler. The handler will be owned by the
  // client of this function.
  return std::make_unique<ObjectFileHandler>(
      std::unique_ptr<ObjectFile>(cast<ObjectFile>(BinaryOrErr->release())));
}

Error llvm::unbundleFiles(StringRef InputFile,
                          const SmallVector<StringRef> &OutputFiles,
                          const SmallVector<StringRef> &Triples) {
  // Open Input file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFile);
  if (std::error_code EC = CodeOrErr.getError())
    return createFileError(InputFile, EC);

  MemoryBuffer &Input = **CodeOrErr;

  // Select the right files handler.
  Expected<std::unique_ptr<FileHandler>> FileHandlerOrErr =
      CreateObjectFileHandler(Input);
  if (!FileHandlerOrErr)
    return FileHandlerOrErr.takeError();

  std::unique_ptr<FileHandler> &FH = *FileHandlerOrErr;
  assert(FH);

  // Read the header of the bundled file.
  if (Error Err = FH->ReadHeader(Input))
    return Err;

  // Create a work list that consist of the map triple/output file.
  StringMap<StringRef> Worklist;
  auto Output = OutputFiles.begin();
  for (auto &Triple : Triples) {
    Worklist[Triple] = *Output;
    ++Output;
  }

  // Read all the bundles that are in the work list. If we find no bundles we
  // assume the file is meant for the host target.
  bool FoundHostBundle = false;
  while (!Worklist.empty()) {
    Expected<Optional<StringRef>> CurTripleOrErr = FH->ReadBundleStart(Input);
    if (!CurTripleOrErr)
      return CurTripleOrErr.takeError();

    // We don't have more bundles.
    if (!*CurTripleOrErr)
      break;

    StringRef CurTriple = **CurTripleOrErr;
    assert(!CurTriple.empty());

    auto Output = Worklist.find(CurTriple);
    // The file may have more bundles for other targets, that we don't care
    // about. Therefore, move on to the next triple
    if (Output == Worklist.end())
      continue;

    // Check if the output file can be opened and copy the bundle to it.
    std::error_code EC;
    raw_fd_ostream OutputFile(Output->second, EC, sys::fs::OF_None);
    if (EC)
      return createFileError(Output->second, EC);
    if (Error Err = FH->ReadBundle(OutputFile, Input))
      return Err;
    if (Error Err = FH->ReadBundleEnd(Input))
      return Err;
    Worklist.erase(Output);

    // Record if we found the host bundle.
    auto OffloadInfo = OffloadTargetInfo(CurTriple);
    if (OffloadInfo.hasHostKind())
      FoundHostBundle = true;
  }

  // If no bundles were found, assume the input file is the host bundle and
  // create empty files for the remaining targets.
  if (Worklist.size() == Triples.size()) {
    for (auto &E : Worklist) {
      std::error_code EC;
      raw_fd_ostream OutputFile(E.second, EC, sys::fs::OF_None);
      if (EC)
        return createFileError(E.second, EC);

      // If this entry has a host kind, copy the input file to the output file.
      auto OffloadInfo = OffloadTargetInfo(E.getKey());
      if (OffloadInfo.hasHostKind())
        OutputFile.write(Input.getBufferStart(), Input.getBufferSize());
    }
    return Error::success();
  }

  // If we found elements, we emit an error if none of those were for the host
  // in case host bundle name was provided in command line.
  if (!FoundHostBundle)
    return createStringError(inconvertibleErrorCode(),
                             "Can't find bundle for the host target");

  // If we still have any elements in the worklist, create empty files for them.
  for (auto &E : Worklist) {
    std::error_code EC;
    raw_fd_ostream OutputFile(E.second, EC, sys::fs::OF_None);
    if (EC)
      return createFileError(E.second, EC);
  }

  return Error::success();
}
