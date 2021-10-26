#include "llvm/Offloading/Wrapper.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

using namespace llvm;

namespace {

ErrorOr<std::string> getExecutablePath(const char *Executable) {
  SmallString<256> ExecutablePath;
  if (ErrorOr<std::string> ptxas = sys::findProgramByName(Executable))
    sys::fs::real_path(*ptxas, ExecutablePath);
  else
    return ptxas.getError();

  return static_cast<std::string>(ExecutablePath);
}

} // namespace

Error llvm::wrapFiles(StringRef InputFile, StringRef OutputFile,
                      Triple TheTriple) {
  ErrorOr<std::string> Executable = getExecutablePath("clang-offload-wrapper");
  if (std::error_code EC = Executable.getError())
    return errorCodeToError(EC);

  SmallVector<StringRef, 16> CmdArgs{*Executable};

  CmdArgs.push_back("-target");
  CmdArgs.push_back(TheTriple.getTriple());
  CmdArgs.push_back("-o");
  CmdArgs.push_back(OutputFile);
  CmdArgs.push_back(InputFile);

  if (sys::ExecuteAndWait(*Executable, CmdArgs))
    return createStringError(inconvertibleErrorCode(),
                             "'clang-offload-wrapper' failed");
  return Error::success();
}

Error llvm::compileWrappedFiles(StringRef InputFile, StringRef OutputFile) {
  ErrorOr<std::string> Executable = getExecutablePath("llc");
  if (std::error_code EC = Executable.getError())
    return errorCodeToError(EC);

  SmallVector<StringRef, 16> CmdArgs{*Executable};

  CmdArgs.push_back("-filetype=obj");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(OutputFile);
  CmdArgs.push_back(InputFile);

  if (sys::ExecuteAndWait(*Executable, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'llc' failed");
  return Error::success();
}
