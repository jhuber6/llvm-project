#ifndef LLVM_OFFLOADING_BUNDLER_H
#define LLVM_OFFLOADING_BUNDLER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {

Error unbundleFiles(StringRef InputFile,
                    const SmallVector<StringRef> &OutputFiles,
                    const SmallVector<StringRef> &Triples);

} // namespace llvm

#endif
