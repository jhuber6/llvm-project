#include "llvm/Offloading/Bundler.h"
#include "llvm/Offloading/NVPTXToolChain.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::OptionCategory ToolChainOptions("llvm-vendor-toolchain options");

static cl::opt<std::string> InputFileName(cl::Positional, cl::OneOrMore,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::cat(ToolChainOptions));

static cl::opt<std::string> OutputFileName("o", cl::CommaSeparated,
                                           cl::Optional,
                                           cl::desc("<output file>"),
                                           cl::init("out.o"),
                                           cl::cat(ToolChainOptions));

int main(int argc, char **argv) {

  cl::ParseCommandLineOptions(argc, argv);

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    exit(1);
  };

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFile(InputFileName);

  if (std::error_code EC = Buffer.getError())
    reportError(createFileError(InputFileName, EC));

  NVPTX::CudaToolChain TC({"a"}, llvm::Triple("nvptx64-nvidia-cuda"));
  if (auto Err = TC.run(InputFileName.c_str()))
    reportError(std::move(Err));

  llvm::errs() << TC.getOutputFile() << " " << TC.getOutput() << "\n";
}
