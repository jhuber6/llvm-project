//===- MemoryAccessInstrumentation.h - shared sanitizer helpers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by ThreadSanitizer and ConcurrencySanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_INSTRUMENTATION_MEMORYACCESSINSTRUMENTATION_H
#define LLVM_LIB_TRANSFORMS_INSTRUMENTATION_MEMORYACCESSINSTRUMENTATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

namespace llvm {

class DataLayout;
class IRBuilderBase;
class Instruction;
class Module;
class Triple;
class Type;
class Value;

namespace memaccess {

/// Access sizes with dedicated callbacks: 1, 2, 4, 8, 16 bytes.
constexpr unsigned kNumAccessSizes = 5;

/// Which address spaces can participate in a data race.
enum class AddrSpaceRacePolicy {
  /// Host TSan: only the generic/flat space (AS 0).
  FlatOnly,
  /// GPU concurrent spaces: global/generic plus shared/LDS (and AMDGPU GDS).
  /// Private/scratch, constant, and buffer fat pointers are excluded.
  GPUConcurrent,
};

/// Index into a 1/2/4/8/16-byte callback table, or -1 if \p Ty is not a
/// supported access size.
int getAccessSizeIndex(Type *Ty, const DataLayout &DL);

bool isVtableAccess(const Instruction *I);

/// True if \p I is an atomic memory op (or fence) whose sync scope is not
/// singlethread. Loads/stores with a singlethread scope are not.
bool isAtomicMemoryAccess(const Instruction *I);

/// Skip the sanitizer module ctor, naked functions (no prologue/epilogue),
/// and disable_sanitizer_instrumentation.
inline bool skipInstrumentation(const Function &F, StringRef ModuleCtorName) {
  if (F.getName() == ModuleCtorName)
    return true;
  if (F.hasFnAttribute(Attribute::Naked))
    return true;
  return F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation);
}

bool addressSpaceMayRace(const Triple &T, unsigned AS,
                         AddrSpaceRacePolicy Policy);

/// Skip PGO counters and address spaces that cannot race under \p Policy.
bool shouldInstrumentAddress(const Module *M, Value *Addr,
                             AddrSpaceRacePolicy Policy);

/// Addrspace-cast \p Addr to generic ptr for a callback argument.
Value *genericCallbackPtr(IRBuilderBase &IRB, Value *Addr);

} // namespace memaccess
} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_INSTRUMENTATION_MEMORYACCESSINSTRUMENTATION_H
