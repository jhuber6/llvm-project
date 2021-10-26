#ifndef LLVM_OFFLOADING_WRAPPER_H
#define LLVM_OFFLOADING_WRAPPER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Error.h"

namespace llvm {

Error wrapFiles(StringRef InputFile, StringRef OutputFile, Triple TheTriple);

Error compileWrappedFiles(StringRef InputFile, StringRef OutputFile);

} // namespace llvm

#endif
