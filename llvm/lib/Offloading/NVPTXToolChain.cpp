#include "llvm/Offloading/NVPTXToolChain.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace NVPTX;

static ErrorOr<std::string> getExecutablePath(const char *Executable) {
  SmallString<256> ExecutablePath;
  if (ErrorOr<std::string> ptxas = sys::findProgramByName(Executable))
    sys::fs::real_path(*ptxas, ExecutablePath);
  else
    return ptxas.getError();

  return static_cast<std::string>(ExecutablePath);
}

static ErrorOr<std::string> getTempFile(StringRef Input, StringRef Suffix) {
  int FD;
  SmallString<128> TempFile;
  StringRef Extension = sys::path::extension(Input);
  StringRef Prefix = sys::path::filename(Input).drop_back(Extension.size());

  if (std::error_code EC =
          sys::fs::createTemporaryFile(Prefix, Suffix, FD, TempFile))
    return EC;

  return static_cast<std::string>(TempFile);
}

Error CudaToolChain::run(Arg Input, Job JobType) {
  std::string LinkerInput;
  if (JobType == Job::Assembler) {
    ErrorOr<std::string> AssemblerTemp = getTempFile(Input, "cubin");
    if (std::error_code EC = AssemblerTemp.getError())
      return createFileError(Input, EC);

    TempFiles.push_back(*AssemblerTemp);
    if (Error Err = Assembler->run(Input, AssemblerTemp->c_str()))
      return Err;
    LinkerInput = *AssemblerTemp;
  } else {
    LinkerInput = Input;
  }

  ErrorOr<std::string> LinkerTemp = getTempFile(Input, "o");
  if (std::error_code EC = LinkerTemp.getError())
    return createFileError(LinkerInput, EC);

  TempFiles.push_back(*LinkerTemp);
  if (Error Err = Linker->run(LinkerInput.c_str(), LinkerTemp->c_str()))
    return Err;

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFile(*LinkerTemp);

  if (std::error_code EC = Buffer.getError())
    return createFileError(*LinkerTemp, EC);

  Output = std::move(*Buffer);
  OutputFile = *LinkerTemp;

  return Error::success();
}

Error Assembler::run(Arg Input, Arg Output) {
  const auto &TC = static_cast<const CudaToolChain &>(getToolChain());

  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  const char *GPUArchName = "sm_70";

  ErrorOr<std::string> Executable = getExecutablePath("ptxas");
  if (std::error_code EC = Executable.getError())
    return errorCodeToError(EC);

  SmallVector<StringRef, 16> CmdArgs{*Executable};

  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-m64" : "-m32");
  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(GPUArchName);
  CmdArgs.push_back("--output-file");
  CmdArgs.push_back(Output);
  CmdArgs.push_back(Input);
  CmdArgs.push_back("-O2");
  CmdArgs.push_back("-c");

  if (sys::ExecuteAndWait(*Executable, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'ptxas' failed");
  return Error::success();
}

Error Linker::run(Arg Input, Arg Output) {
  const auto &TC = static_cast<const CudaToolChain &>(getToolChain());

  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  const char *GPUArchName = "sm_70";

  ErrorOr<std::string> Executable = getExecutablePath("nvlink");
  if (std::error_code EC = Executable.getError())
    return errorCodeToError(EC);

  SmallVector<StringRef, 16> CmdArgs{*Executable};

  CmdArgs.push_back("-arch");
  CmdArgs.push_back(GPUArchName);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output);
  CmdArgs.push_back(Input);

  if (sys::ExecuteAndWait(*Executable, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'nvlink' failed");
  return Error::success();
}
