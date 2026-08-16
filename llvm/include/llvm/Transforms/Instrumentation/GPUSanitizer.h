//===- GPUSanitizer.h - GPU memory safety instrumentation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bounds checking for AMDGPU.  The runtime places every device allocation in a
// power-of-two aligned slot inside a reserved virtual address region, so a
// pointer's allocation identity and base are recoverable by arithmetic and only
// the exact extent needs a table lookup.
//
// Device globals cannot be placed by an allocator, because the loader carves
// their segment out of its own storage.  They keep the addresses the loader
// gave them and get a table of their own: the instrumented ones are collected
// into a single section, and one entry per 16-byte granule of that section
// records which object owns the granule and how far it extends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_GPUSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_GPUSANITIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {

class Module;
class ModulePass;

/// Geometry shared between the pass and the runtime.  Any change here must be
/// mirrored in the host-side allocator.
namespace gpuasan {
/// Base of the reserved region.  The runtime asks for this address explicitly
/// and refuses to enable itself if it does not get it.
constexpr uint64_t kRegionBase = 1ULL << 45; // 32 TiB
/// Log2 of the per-class subregion size.  2^41 = 2^32 * 2^9 keeps the class
/// extraction a shift of the high dword alone.
constexpr unsigned kClassShift = 41;
/// Number of live size classes.  Class c holds slots of 2^(c+12) bytes.
constexpr unsigned kNumClasses = 25;
/// Smallest slot is a page.
constexpr unsigned kMinSlotLog2 = 12;
/// Log2 of the per-class stride in the metadata table, and so also of the
/// largest slot index a class can hand out: the check refuses an index at or
/// above this, and the runtime never allocates one.  That bound is what lets
/// the whole table be backed up front, which is in turn what stops a load from
/// it faulting -- a wild pointer has to produce a report, not a memory
/// violation inside the check that was supposed to diagnose it.
constexpr unsigned kTableClassShift = 21;
/// Most slots a class can hand out, which is the per-class stride above.
constexpr uint64_t kMaxSlots = 1ULL << kTableClassShift;
/// Base of the metadata table.  One 8-byte entry per slot.
constexpr uint64_t kTableBase = 3ULL << 45; // 96 TiB
/// Size of the table, which the runtime backs in full before any check runs.
constexpr uint64_t kTableSpan = kNumClasses * kMaxSlots * 8; // 400 MiB
/// Entry layout: [36:0] size, [47:37] color in 256 B units, [62:48] reserved,
/// [63] poisoned.  A poisoned entry keeps the extent of the allocation that was
/// there, so a use after free is reported against the object it used to be
/// rather than against an empty slot; the check only has to notice that the
/// entry went negative.
constexpr unsigned kSizeBits = 37;
constexpr unsigned kColorBits = 11;
constexpr unsigned kColorScaleLog2 = 8;
constexpr uint64_t kPoisonBit = 1ULL << 63;

/// Section every instrumented device global is collected into, so that one
/// range test covers all of them and a dense table can describe them.  The name
/// is a C identifier, which is what makes a linker synthesize the
/// `__start_`/`__stop_` pair that bounds it.  Those are the bounds the check
/// uses: the linker knows the extent of the section it laid out, so nothing has
/// to be computed at load time and told to the device, and the two addresses
/// materialize PC-relative into scalar registers instead of being loaded from
/// memory.  The table is then only a mirror of the layout, indexed by the
/// distance from `__start_`.
constexpr char kGlobalsSection[] = "__gpuasan_globals";

/// Bytes of the section described by one table entry.  Every object in the
/// section is aligned to this and padded out to it, so a granule belongs to
/// exactly one object and the granule a byte falls in is a shift away.
constexpr unsigned kGlobalGranuleLog2 = 4;
constexpr uint64_t kGlobalGranule = 1ULL << kGlobalGranuleLog2;

/// Bytes of poisoned padding appended to every instrumented global, so that
/// running off the end of one lands on a granule no object claims rather than
/// on the neighbour.
constexpr uint64_t kGlobalRedzone = kGlobalGranule;

/// Prefix of the per-image info block, which is the one symbol the host has to
/// find by name.  Everything else it needs is reachable from here.
///
///   struct { u64 base, table, granules, ndescs, descs; }
///
/// Nothing here is written by the host: the device does not read it at all.  It
/// exists because the host has to fill the table, and for that it needs the
/// origin the table is indexed from -- `base`, which the image initializes to
/// `__start_` of the section and the loader relocates -- along with the table,
/// its length, and the descriptors saying which object ended up where.
constexpr char kGlobalsInfoPrefix[] = "__gpuasan_globals_info.";

/// A granule entry: [31:0] the owning object's offset from `base`, [63:32] the
/// complement of the offset one past its end.  The complement is what makes a
/// zero entry mean "nothing known here, allow it": an unwritten table reads as
/// an object that starts at zero and ends 4 GiB later.  All ones is the
/// opposite, and is what the host writes over padding and redzones.
constexpr unsigned kGlobalEndShift = 32;
constexpr uint64_t kGlobalPoisonEntry = ~0ULL;

/// Fields of the info block, in 8-byte units.
enum GlobalsInfoField {
  kGlobalsInfoBase = 0,
  kGlobalsInfoTable = 1,
  kGlobalsInfoGranules = 2,
  kGlobalsInfoNumDescs = 3,
  kGlobalsInfoDescs = 4,
  kGlobalsInfoFields = 5,
};

/// Extra bits in the report handler's `flags` argument, above the object's
/// address space.
constexpr unsigned kFlagAddrSpaceMask = 0xff;
constexpr unsigned kFlagFreed = 1u << 8;
} // namespace gpuasan

/// Checks accesses to placed memory, and prepares LDS for the pass below by
/// widening each object with a redzone.
class GPUSanitizerPass : public PassInfoMixin<GPUSanitizerPass> {
public:
  /// \p Recover selects the report entry points that never end the program,
  /// which is what `-fsanitize-recover=gpuasan` asks for.  Without it a report
  /// is fatal unless the runtime says otherwise, as it is for every other
  /// address sanitizer.
  explicit GPUSanitizerPass(bool Recover = false) : Recover(Recover) {}

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  bool Recover;
};

/// Checks accesses to LDS.  Must run *after* AMDGPULowerModuleLDSPass, because
/// it needs the address each object was finally packed at in order to build the
/// shadow that describes them.
class GPUSanitizerLDSPass : public PassInfoMixin<GPUSanitizerLDSPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

/// Codegen still runs on the legacy pass manager, and LDS lowering happens
/// there for anything that does not go through LTO.
LLVM_ABI ModulePass *createGPUSanitizerLDSLegacyPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_GPUSANITIZER_H
