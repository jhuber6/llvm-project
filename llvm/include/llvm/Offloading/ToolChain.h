#ifndef LLVM_OFFLOADING_TOOLCHAIN_H
#define LLVM_OFFLOADING_TOOLCHAIN_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

class Tool;
class ToolChain;

using ArgList = SmallVector<StringRef, 4>;
using Arg = StringRef;

class ToolChain {
protected:
  const ArgList Args;

  const llvm::Triple Triple;

  SmallVector<std::string, 4> TempFiles;

  std::unique_ptr<MemoryBuffer> Output;

  std::string OutputFile;

public:
  enum class Job {
    Assembler,
    LinkOnly,
  };

  ToolChain(const ArgList &Args, const llvm::Triple &Triple)
      : Args(Args), Triple(Triple) {}

  virtual ~ToolChain();

  llvm::Triple getTriple() const { return Triple; }
  ArgList getArgs() const { return Args; }
  const MemoryBuffer *getOutput() const { return Output.get(); }
  std::string getOutputFile() const { return OutputFile; }
  const SmallVectorImpl<std::string> &getTempFiles() const { return TempFiles; }

  virtual const Tool *getAssembler() const = 0;
  virtual const Tool *getLinker() const = 0;

  virtual Error run(Arg Input, Job JobType = Job::Assembler) = 0;
};

class Tool {
  const char *Name;

  const ToolChain &TheToolChain;

public:
  Tool(const char *Name, const ToolChain &TheToolChain)
      : Name(Name), TheToolChain(TheToolChain){};

  const ToolChain &getToolChain() const { return TheToolChain; };

  virtual ~Tool();

  virtual Error run(Arg Input, Arg Output) = 0;
};

} // namespace llvm

#endif
