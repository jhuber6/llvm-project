#ifndef LLVM_OFFLOADING_NVPTXTOOLCHAIN_H
#define LLVM_OFFLOADING_NVPTXTOOLCHAIN_H

#include "llvm/Offloading/ToolChain.h"

namespace llvm {
namespace NVPTX {

class Assembler final : public Tool {
public:
  Assembler(const ToolChain &TheToolChain)
      : Tool("NVPTX::Assembler", TheToolChain) {}

  Error run(Arg Input, Arg Output) override;
};

class Linker final : public Tool {
public:
  Linker(const ToolChain &TheToolChain) : Tool("NVPTX::Linker", TheToolChain) {}

  Error run(Arg Input, Arg Output) override;
};

class CudaToolChain final : public ToolChain {
  std::unique_ptr<Tool> Assembler;
  std::unique_ptr<Tool> Linker;

public:
  CudaToolChain(const ArgList &Args, const llvm::Triple &Triple)
      : ToolChain(Args, Triple),
        Assembler(std::make_unique<NVPTX::Assembler>(*this)),
        Linker(std::make_unique<NVPTX::Linker>(*this)) {}

  const Tool *getAssembler() const override { return Assembler.get(); }
  const Tool *getLinker() const override { return Linker.get(); }

  Error run(Arg Input, Job JobType = Job::Assembler) override;
};

} // namespace NVPTX
} // namespace llvm

#endif
