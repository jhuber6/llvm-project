#include "llvm/Offloading/ToolChain.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

ToolChain::~ToolChain() {
  for (auto &File : TempFiles)
    sys::fs::remove(File);
}

Tool::~Tool() {}
