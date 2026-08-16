//===- GPUSanitizer.cpp - GPU memory safety instrumentation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bounds checking for AMDGPU, in two schemes that share only their reporting.
//
// Placed memory.  The host runtime reserves a large virtual address region
// divided into per-size-class subregions and puts every device allocation at
// the base of a power-of-two aligned slot.  A pointer therefore encodes its own
// size class and slot index, so the allocation base is recovered with a mask
// and the slot identity with a shift.  Only the exact byte extent is unknown,
// and that lives in a dense table indexed by the slot number the arithmetic
// produced.  Pointers outside the region -- host allocations, pinned memory,
// IPC imports, LDS and scratch -- fail the initial range compare and are left
// alone, which is what lets instrumented code consume memory the runtime never
// saw.
//
// Device globals cannot be placed by the allocator, because the loader carves
// their segment out of its own storage.  They are left where the loader put
// them and given a table of their own: the instrumented ones are collected
// into a single section, each padded with a redzone and aligned so that a
// 16-byte granule belongs to exactly one object, and one table entry per
// granule records the extent of its owner.  An access therefore pays one more
// range test, and the program's ABI is untouched -- every symbol keeps its
// address, its size and its linkage.  See GPUSanitizerLDS below, which is where
// that happens, because it needs the whole image.
//
// LDS has a scheme of its own, also described there, because its layout is
// decided at compile time rather than by an allocator.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/GPUSanitizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <string>

using namespace llvm;
using namespace llvm::gpuasan;

#define DEBUG_TYPE "gpuasan"

STATISTIC(NumInstrumented, "Number of memory accesses instrumented");
STATISTIC(NumSkippedAddrSpace, "Accesses skipped due to address space");
STATISTIC(NumSkippedGlobal, "Accesses to globals outside the placed region");
STATISTIC(NumGlobalsCollected, "Device globals given a redzone and a table");
STATISTIC(NumGlobalsOverBudget,
          "Device globals left unchecked because the table would be too large");
STATISTIC(NumGlobalsInstrumented,
          "Accesses checked against the device globals table");
STATISTIC(NumLDSPadded, "LDS objects given a redzone");
STATISTIC(NumLDSInstrumented, "LDS accesses checked against the shadow");
STATISTIC(NumLDSShadows, "LDS shadow maps emitted");
STATISTIC(NumLDSUnchecked, "Functions whose LDS layout could not be described");

static cl::opt<bool> ClInstrumentReads("gpuasan-instrument-reads",
                                       cl::init(true), cl::Hidden,
                                       cl::desc("Instrument loads"));
static cl::opt<bool> ClInstrumentWrites("gpuasan-instrument-writes",
                                        cl::init(true), cl::Hidden,
                                        cl::desc("Instrument stores"));
static cl::opt<bool> ClInstrumentAtomics("gpuasan-instrument-atomics",
                                         cl::init(true), cl::Hidden,
                                         cl::desc("Instrument atomics"));
static cl::opt<bool>
    ClRecover("gpuasan-recover", cl::init(false), cl::Hidden,
              cl::desc("Continue execution after reporting an error"));
static cl::opt<bool>
    ClInstrumentLDS("gpuasan-instrument-lds", cl::init(true), cl::Hidden,
                    cl::desc("Check accesses to statically sized LDS objects"));
static cl::opt<unsigned>
    ClLDSGranule("gpuasan-lds-granule", cl::init(16), cl::Hidden,
                 cl::desc("Bytes of LDS described by one shadow byte"));
static cl::opt<unsigned>
    ClLDSRedzone("gpuasan-lds-redzone", cl::init(16), cl::Hidden,
                 cl::desc("Poisoned bytes appended to each LDS object"));
static cl::opt<unsigned>
    ClLDSBudget("gpuasan-lds-budget", cl::init(65536), cl::Hidden,
                cl::desc("LDS a module may consume once objects are widened"));
static cl::opt<bool>
    ClInstrumentGlobals("gpuasan-instrument-globals", cl::init(true),
                        cl::Hidden,
                        cl::desc("Check accesses to device globals"));
static cl::opt<unsigned> ClGlobalsBudget(
    "gpuasan-globals-budget", cl::init(8 << 20), cl::Hidden,
    cl::desc("Bytes of granule table a module may spend on its globals"));

namespace {

constexpr unsigned kGlobalAddrSpace = 1;
constexpr unsigned kLDSAddrSpace = 3;
constexpr unsigned kConstantAddrSpace = 4;

/// Metadata the compile-time half leaves on a global it judges checkable, and
/// the only thing that survives to tell the post-link half which variables came
/// from a translation unit that was compiled with the sanitizer.
constexpr char kGlobalTag[] = "gpuasan.checkable";

/// Suffix given to the padded storage a checked global's contents move into.
/// The storage is private and the variable's own name becomes an alias of it,
/// which is what keeps the redzone out of the ABI.
constexpr char kGlobalStorageSuffix[] = ".gpuasan";

/// Name of the struct an LDS object is wrapped in when it is widened.  The
/// wrapper is what tells the LDS half of the tool, running after the objects
/// have been packed together, where the data ends and the redzone begins:
/// identified struct types survive packing, metadata does not.
constexpr char kLDSPadTypeName[] = "gpuasan.lds";

/// Left on every function this pass looked at, so the LDS half can tell them
/// from whatever LTO merged in beside them.
constexpr char kFuncMarker[] = "gpuasan-instrument";

/// Byte offset of `hidden_dynamic_lds_size` in the code object v5 implicit
/// arguments.  It is the one bound in LDS that is not a property of the
/// program.
constexpr uint64_t kHiddenDynLDSSizeOffset = 120;

/// One memory access to check.
struct AccessInfo {
  Instruction *I;
  Value *Ptr;
  Value *DynSize; // null when the width is a compile-time constant
  uint64_t Size;
  bool IsWrite;
};

class GPUSanitizer {
public:
  GPUSanitizer(Module &M, bool Recover)
      : M(M), Ctx(M.getContext()), DL(M.getDataLayout()),
        Int64Ty(Type::getInt64Ty(Ctx)),
        Recover(ClRecover.getNumOccurrences() ? ClRecover.getValue()
                                              : Recover) {}

  bool run();

private:
  bool tagGlobals();
  bool padLDSGlobals();

  Module &M;
  LLVMContext &Ctx;
  const DataLayout &DL;
  Type *Int64Ty;
  bool Recover;
  FunctionCallee LoadFn, StoreFn;
};

} // namespace

/// Declare the two report entry points.  Their bodies live in
/// libclang_rt.gpuasan, which the driver links after instrumentation, so the
/// transport to the host can change without touching the compiler.
///
///   void __gpuasan_report_{load,store}[_noabort](void *addr, u64 access_size,
///                                               void *base, u64 alloc_size,
///                                               u32 flags)
///
/// Named as the CPU sanitizers name theirs: the plain entry point may end the
/// program, and `-fsanitize-recover=gpuasan` asks for the one that never does.
/// Neither is `noreturn`, because whether a report is fatal is the runtime's
/// decision -- the host answers each report with it -- so the access the check
/// guards stays in the code and the stop happens inside the handler.
///
/// The recovered allocation is passed along because this is the side that
/// derived it and because the slot may be recycled before the host drains the
/// report.  `flags` carries the address space of the object in its low byte,
/// which the host cannot deduce -- a global and an LDS object are both simply
/// outside the placed region -- and above it whether the metadata the check
/// read was poisoned, which is the difference between an overflow and a use
/// after free.
static void declareRuntime(Module &M, bool Recover, FunctionCallee &LoadFn,
                           FunctionCallee &StoreFn) {
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = PointerType::get(Ctx, 0);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  StringRef Suffix = Recover ? "_noabort" : "";
  for (bool IsWrite : {false, true}) {
    FunctionCallee &Fn = IsWrite ? StoreFn : LoadFn;
    Fn = M.getOrInsertFunction(
        ("__gpuasan_report_" + Twine(IsWrite ? "store" : "load") + Suffix)
            .str(),
        Type::getVoidTy(Ctx), PtrTy, Int64Ty, PtrTy, Int64Ty, Int32Ty);
    auto *F = dyn_cast<Function>(Fn.getCallee());
    if (!F)
      continue;
    F->addFnAttr(Attribute::NoUnwind);
    F->addFnAttr(Attribute::Cold);
    F->addFnAttr(Attribute::DisableSanitizerInstrumentation);
  }
}

static void emitReport(FunctionCallee LoadFn, FunctionCallee StoreFn,
                       IRBuilder<> &IRB, const AccessInfo &A, Value *AddrPtr,
                       Value *Width, Value *BasePtr, Value *Size,
                       Value *Flags) {
  LLVMContext &Ctx = A.I->getContext();
  CallInst *Call = IRB.CreateCall(A.IsWrite ? StoreFn : LoadFn,
                                  {AddrPtr, Width, BasePtr, Size, Flags});
  Call->addFnAttr(Attribute::Cold);

  // The verifier rejects an inlinable call with no location inside a function
  // that has debug info, and the block this lands in was synthesized.  Borrow
  // the access's own line, falling back to the subprogram scope for an access
  // the compiler generated without one.
  DebugLoc Loc = A.I->getDebugLoc();
  if (!Loc)
    if (DISubprogram *SP = A.I->getFunction()->getSubprogram())
      Loc = DILocation::get(Ctx, 0, 0, SP);
  Call->setDebugLoc(Loc);
}

/// The report handlers take generic pointers so that one signature serves every
/// address space.
static Value *toGeneric(IRBuilder<> &IRB, Value *P) {
  Type *GenericPtrTy = PointerType::get(P->getContext(), 0);
  if (P->getType() == GenericPtrTy)
    return P;
  return IRB.CreateAddrSpaceCast(P, GenericPtrTy);
}

static bool shouldInstrumentFunction(Function &F) {
  if (F.isDeclaration())
    return false;
  // The runtime must not check itself.
  if (F.getName().starts_with("__gpuasan_"))
    return false;
  return !F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation);
}

static void collectAccesses(Function &F, const DataLayout &DL,
                            SmallVectorImpl<AccessInfo> &Out) {
  for (Instruction &I : instructions(F)) {
    if (I.hasMetadata(LLVMContext::MD_nosanitize))
      continue;

    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (ClInstrumentReads)
        Out.push_back({&I, LI->getPointerOperand(), nullptr,
                       DL.getTypeStoreSize(LI->getType()), false});
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (ClInstrumentWrites)
        Out.push_back({&I, SI->getPointerOperand(), nullptr,
                       DL.getTypeStoreSize(SI->getValueOperand()->getType()),
                       true});
    } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
      if (ClInstrumentAtomics)
        Out.push_back({&I, RMW->getPointerOperand(), nullptr,
                       DL.getTypeStoreSize(RMW->getValOperand()->getType()),
                       true});
    } else if (auto *XCHG = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (ClInstrumentAtomics)
        Out.push_back(
            {&I, XCHG->getPointerOperand(), nullptr,
             DL.getTypeStoreSize(XCHG->getCompareOperand()->getType()), true});
    } else if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
      // The bounds formula never adds to the offset, so a runtime length is
      // safe to pass through as-is.
      Value *Len = MI->getLength();
      if (ClInstrumentWrites)
        Out.push_back({&I, MI->getRawDest(), Len, 0, true});
      if (auto *MT = dyn_cast<MemTransferInst>(&I))
        if (ClInstrumentReads)
          Out.push_back({&I, MT->getRawSource(), Len, 0, false});
    }
  }
}

//===----------------------------------------------------------------------===//
// Device globals
//===----------------------------------------------------------------------===//

/// Whether a global can be given a redzone and a table entry, judged on what a
/// single translation unit can see.  Everything here is a property of the
/// variable itself, so the answer cannot change when the module is linked with
/// others -- which matters, because this decides only whether to *tag* the
/// variable; the transform happens after the link, where the table that
/// describes it can be built once for the whole image.
static bool canCheckGlobal(GlobalVariable &GV, const DataLayout &DL) {
  if (GV.isDeclaration())
    return false;
  unsigned AS = GV.getAddressSpace();
  if (AS != kGlobalAddrSpace && AS != kConstantAddrSpace)
    return false;

  // The contents move into padded storage that the variable's name aliases, and
  // an alias is not a definition anything else may claim to provide, nor one
  // another object may be substituted for.
  //
  // An externally initialized variable is fine, and has to be: every HIP
  // `__device__` variable is one.  All it says is that the host may write the
  // storage through the symbol before the program runs, and the symbol still
  // names the same address with the same size.  The flag travels to the storage
  // so that nothing folds a load of an initializer the host is going to
  // replace.
  if (GV.hasComdat() || GV.isInterposable() || GV.hasCommonLinkage() ||
      GV.hasAppendingLinkage())
    return false;
  // A read-only variable would drag the whole section into a read-only segment,
  // and the table the host writes lives nowhere near it.  Overreading one is
  // also the least interesting thing a program can do.
  if (GV.isConstant())
    return false;

  // The host reaches these through their symbols before any of our bookkeeping
  // exists, or reads them from a fixed address: the RPC client the report
  // channel is written into, the device library's configuration words that CLR
  // stamps at load, and our own state.
  StringRef Name = GV.getName();
  if (Name.starts_with("__gpuasan") || Name.starts_with("__llvm") ||
      Name.starts_with("llvm.") || Name.starts_with("__oclc_") ||
      Name.starts_with("__hip_cuid") || Name.starts_with("__omp_") ||
      Name.starts_with("__amdgcn_"))
    return false;
  // A global placed somewhere specific was put there for a reason.
  if (GV.hasSection() || GV.isThreadLocal())
    return false;

  Type *Ty = GV.getValueType();
  if (!Ty->isSized())
    return false;
  uint64_t Bytes = DL.getTypeAllocSize(Ty).getFixedValue();
  // A single object is described by as many granules as it spans, so the only
  // real bound is the table budget, which is a whole-image question and settled
  // where the table is built.
  if (Bytes == 0)
    return false;

  // The section is aligned to whatever it holds, so a large alignment costs
  // padding at the front rather than correctness.  Refuse the absurd ones,
  // which would make the table mostly holes.
  if (GV.getAlign().value_or(DL.getPreferredAlign(&GV)).value() > (1ULL << 16))
    return false;

  return true;
}

/// Mark the globals this module defines that the post-link half may check.
///
/// Only a mark.  Widening a variable and moving its contents into padded
/// storage is invisible to the rest of the program -- the name it is reached
/// through keeps its address, its size and its linkage -- but the table that
/// describes the result covers a whole section, so it can only be built once
/// every module that contributes to that section is present.  That is the
/// post-link module.
///
/// The mark is metadata on the variable, which survives linking, internalizing
/// and renaming, and is the only thing left that says the variable came from a
/// translation unit compiled with the sanitizer.  A device libc or a math
/// library linked in as bitcode carries none, and its globals are left alone.
bool GPUSanitizer::tagGlobals() {
  bool Changed = false;
  for (GlobalVariable &GV : M.globals()) {
    if (!canCheckGlobal(GV, DL))
      continue;
    GV.setMetadata(kGlobalTag, MDNode::get(Ctx, {}));
    Changed = true;
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// Placed memory
//===----------------------------------------------------------------------===//

/// Only global, constant and generic pointers can name a slot.  Constant is in
/// the list because a redirected `__constant__` variable keeps its address
/// space while its storage moves into the region.  Everything else is either
/// bounded by its own object or out of scope: private is not covered yet, and
/// the rest cannot reach placed memory at all.
static bool isCheckableAddrSpace(unsigned AS) {
  return AS == 0 || AS == kGlobalAddrSpace || AS == kConstantAddrSpace;
}

/// Emit the check for an access that may name placed memory.
static void emitPlacedCheck(const AccessInfo &A, Module &M,
                            FunctionCallee LoadFn, FunctionCallee StoreFn) {
  LLVMContext &Ctx = M.getContext();
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  unsigned AS = A.Ptr->getType()->getPointerAddressSpace();
  if (!isCheckableAddrSpace(AS)) {
    ++NumSkippedAddrSpace;
    return;
  }

  // A global is never inside the region -- the loader placed it -- so it can
  // never satisfy the region test.  Emitting the check anyway is a miscompile:
  // the region subtraction folds into a constant expression on the global's
  // address, which the DAG turns into a relocation with a 45-bit addend, and
  // AMDGPU asserts that global address offsets fit in 32 bits.  Globals are
  // checked against a table of their own instead.
  if (isa<GlobalValue>(getUnderlyingObject(A.Ptr))) {
    ++NumSkippedGlobal;
    return;
  }

  IRBuilder<> IRB(A.I);
  Value *Addr = IRB.CreatePtrToInt(A.Ptr, Int64Ty, "gpuasan.addr");

  // Region test.  d = p - base; c = d >> 41; in = c < NumClasses.
  //
  // A 2^41-aligned base means the low dword of the difference equals the low
  // dword of the pointer, so this is a 32-bit subtract on the high half once it
  // reaches the selection DAG.
  Value *Delta = IRB.CreateSub(Addr, ConstantInt::get(Int64Ty, kRegionBase),
                               "gpuasan.delta");
  Value *Class = IRB.CreateLShr(Delta, ConstantInt::get(Int64Ty, kClassShift),
                                "gpuasan.class");
  Value *InRegion = IRB.CreateICmpULT(
      Class, ConstantInt::get(Int64Ty, kNumClasses), "gpuasan.inregion");

  // Everything below only runs for pointers the runtime placed.
  Instruction *CheckTerm = SplitBlockAndInsertIfThen(InRegion, A.I, false);
  IRB.SetInsertPoint(CheckTerm);

  // Slot geometry.  k = c + 12 is the log2 of this class's slot size.
  Value *K = IRB.CreateAdd(Class, ConstantInt::get(Int64Ty, kMinSlotLog2),
                           "gpuasan.k");
  Value *SlotSize = IRB.CreateShl(ConstantInt::get(Int64Ty, 1), K);
  Value *SlotMask =
      IRB.CreateSub(SlotSize, ConstantInt::get(Int64Ty, 1), "gpuasan.slotmask");

  // Offset within the slot, and the slot's index inside its subregion.
  Value *SlotOff = IRB.CreateAnd(Addr, SlotMask, "gpuasan.slotoff");
  Value *SubMask = ConstantInt::get(Int64Ty, (1ULL << kClassShift) - 1);
  Value *Index =
      IRB.CreateLShr(IRB.CreateAnd(Delta, SubMask), K, "gpuasan.index");

  // Table index is (class << 21) | index.  The stride is a constant so this
  // stays a single shift-or rather than a second lookup.
  //
  // The index is masked into the class's own range, and an index that needed
  // masking is bad by construction: the runtime never hands out a slot at or
  // above the stride, so a pointer this far into a subregion belongs to no
  // allocation.  The mask is what makes the load safe -- every address it can
  // now produce is inside the table, which the runtime has backed in full, so
  // the load cannot fault however wild the pointer was.  Without it a stray
  // pointer would be diagnosed by a memory violation inside the check.
  Value *InRange = IRB.CreateICmpULT(
      Index, ConstantInt::get(Int64Ty, kMaxSlots), "gpuasan.inrange");
  Value *SafeIndex = IRB.CreateAnd(
      Index, ConstantInt::get(Int64Ty, kMaxSlots - 1), "gpuasan.safeindex");
  Value *Id = IRB.CreateOr(
      IRB.CreateShl(Class, ConstantInt::get(Int64Ty, kTableClassShift)),
      SafeIndex, "gpuasan.id");

  // Constant address space, which is what the table is from the device's point
  // of view: only the host ever writes it.  That is also what lets the backend
  // select a scalar load when the address turns out to be uniform, keeping the
  // whole check in SALU, where an ordinary global load would have to assume the
  // kernel's own stores could have clobbered the entry.
  //
  // Cacheable either way: the host invalidates device caches after every table
  // write, so a cached entry cannot outlive the free that zeroed it.
  Value *TableBase =
      ConstantExpr::getIntToPtr(ConstantInt::get(Int64Ty, kTableBase),
                                PointerType::get(Ctx, kConstantAddrSpace));
  Value *EntryPtr = IRB.CreateGEP(Int64Ty, TableBase, Id, "gpuasan.entryptr");
  LoadInst *Entry =
      IRB.CreateAlignedLoad(Int64Ty, EntryPtr, Align(8), "gpuasan.entry");
  // Not the program's memory: the post-link half must not check it, and no
  // later run of this pass may nest a check inside it.
  Entry->setMetadata(LLVMContext::MD_nosanitize, MDNode::get(Ctx, {}));

  // Entry layout: [36:0] size, [47:37] color in 256 B units, [63] poisoned.
  Value *Size =
      IRB.CreateAnd(Entry, ConstantInt::get(Int64Ty, (1ULL << kSizeBits) - 1),
                    "gpuasan.size");
  Value *Color = IRB.CreateShl(
      IRB.CreateAnd(IRB.CreateLShr(Entry, ConstantInt::get(Int64Ty, kSizeBits)),
                    ConstantInt::get(Int64Ty, (1ULL << kColorBits) - 1)),
      ConstantInt::get(Int64Ty, kColorScaleLog2), "gpuasan.color");

  // Offset relative to the allocation rather than the slot.
  Value *Off = IRB.CreateSub(SlotOff, Color, "gpuasan.off");

  Value *Width = A.DynSize ? IRB.CreateZExtOrTrunc(A.DynSize, Int64Ty)
                           : ConstantInt::get(Int64Ty, A.Size);

  // bad = (off >= size) || (size - off < width)
  //
  // Not `off + width > size`: `off` is unsigned, so an access below the base
  // wraps to a huge value, and the additive form wraps it back around to
  // something small and misses every underflow of `width` bytes or fewer.  The
  // first compare catches the wrapped case and guarantees the subtraction in
  // the second cannot itself wrap, which also makes an allocation smaller than
  // the access width report correctly.
  Value *PastEnd = IRB.CreateICmpUGE(Off, Size, "gpuasan.pastend");
  Value *Remain = IRB.CreateSub(Size, Off, "gpuasan.remain");
  Value *TooWide = IRB.CreateICmpULT(Remain, Width, "gpuasan.toowide");
  Value *Bad = IRB.CreateOr(PastEnd, TooWide, "gpuasan.bad");
  Bad = IRB.CreateOr(Bad, IRB.CreateNot(InRange), "gpuasan.bad.range");

  // A freed slot keeps the extent of what used to be there and sets the top bit
  // instead, so the report can name the allocation the pointer used to own
  // rather than an empty slot.  The check pays one compare for it: the entry is
  // negative exactly while the slot is poisoned.
  Value *Poisoned = IRB.CreateICmpSLT(Entry, ConstantInt::get(Int64Ty, 0),
                                      "gpuasan.poisoned");
  Bad = IRB.CreateOr(Bad, Poisoned, "gpuasan.bad.poison");

  // A zero-length transfer touches nothing, and one-past-the-end is a legal
  // address for it, so a `memcpy` of a computed length that happens to be zero
  // must not be reported.  Only worth a compare when the length is dynamic: a
  // constant-width access is never zero bytes wide.
  if (A.DynSize)
    Bad = IRB.CreateAnd(Bad,
                        IRB.CreateICmpNE(Width, ConstantInt::get(Int64Ty, 0)),
                        "gpuasan.bad.nonempty");

  Instruction *ReportTerm = SplitBlockAndInsertIfThen(Bad, CheckTerm, false);
  IRB.SetInsertPoint(ReportTerm);

  // The allocation base is the address less its offset, which the checks above
  // already produced.  Address space zero: a placed allocation is recognised by
  // its address, so the host needs no hint, but it does need to be told that
  // the metadata was poisoned, which is the whole difference between an
  // overflow and a use after free.
  Type *PtrTy = PointerType::get(Ctx, 0);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Value *Base = IRB.CreateIntToPtr(IRB.CreateSub(Addr, Off), PtrTy);
  Value *Flags = IRB.CreateShl(IRB.CreateZExt(Poisoned, Int32Ty),
                               Log2_32(kFlagFreed), "gpuasan.flags");
  emitReport(LoadFn, StoreFn, IRB, A, IRB.CreateIntToPtr(Addr, PtrTy), Width,
             Base, Size, Flags);

  ++NumInstrumented;
}

//===----------------------------------------------------------------------===//
// LDS, first half: widening
//===----------------------------------------------------------------------===//

/// Widen every static LDS object so that the shadow built after packing can
/// describe it.  Two properties are needed and neither can be added later: a
/// granule must belong to exactly one object, which the alignment provides, and
/// an overflow must have somewhere poisoned to land, which the trailing redzone
/// provides.  Both have to be in place before AMDGPULowerModuleLDSPass decides
/// the layout.
///
/// The object's true extent is recorded in its *type*, by wrapping it in a
/// two-member struct, because that pass builds its packed struct out of the
/// members' value types verbatim.
bool GPUSanitizer::padLDSGlobals() {
  const uint64_t Granule = ClLDSGranule;

  SmallVector<GlobalVariable *, 8> Targets;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != kLDSAddrSpace)
      continue;
    // An `extern __shared__` array is sized by the launch, so it has no
    // initializer and nothing here can widen it.  Bookkeeping variables the
    // lowering pass introduces are not objects.
    if (!GV.hasInitializer() || GV.getName().starts_with("llvm."))
      continue;
    if (DL.getTypeAllocSize(GV.getValueType()).getFixedValue() == 0)
      continue;
    Targets.push_back(&GV);
  }
  if (Targets.empty())
    return false;

  auto Consumed = [&](uint64_t Redzone) {
    uint64_t Total = 0;
    for (GlobalVariable *GV : Targets)
      Total += alignTo(DL.getTypeAllocSize(GV->getValueType()).getFixedValue(),
                       Granule) +
               Redzone;
    return Total;
  };

  // Widening costs occupancy and, past a point, the ability to launch at all.
  // Give up the redzones before giving up the build: without them an overflow
  // is only caught until the end of the object's own last granule, which is a
  // weaker check and not a broken one.
  uint64_t Redzone = alignTo<uint64_t>(ClLDSRedzone, Granule);
  if (Consumed(Redzone) > ClLDSBudget)
    Redzone = 0;
  if (Consumed(Redzone) > ClLDSBudget)
    return false;

  Type *Int8Ty = Type::getInt8Ty(Ctx);
  for (GlobalVariable *GV : Targets) {
    Type *Ty = GV->getValueType();
    uint64_t Size = DL.getTypeAllocSize(Ty).getFixedValue();
    uint64_t Pad = alignTo(Size, Granule) - Size + Redzone;

    StructType *PaddedTy =
        StructType::create({Ty, ArrayType::get(Int8Ty, Pad)}, kLDSPadTypeName);
    auto *NG =
        new GlobalVariable(M, PaddedTy, /*isConstant=*/false, GV->getLinkage(),
                           PoisonValue::get(PaddedTy), "", nullptr,
                           GlobalValue::NotThreadLocal, kLDSAddrSpace);
    NG->copyAttributesFrom(GV);
    NG->setAlignment(std::max(GV->getAlign().valueOrOne(), Align(Granule)));
    GV->replaceAllUsesWith(NG);
    NG->takeName(GV);
    GV->eraseFromParent();
    ++NumLDSPadded;
  }
  return true;
}

bool GPUSanitizer::run() {
  if (!M.getTargetTriple().isAMDGPU())
    return false;

  // The pass is hooked into both the regular and the LTO pipelines so that it
  // catches OpenMP and HIP alike; instrumenting twice would nest a check inside
  // its own table load.
  if (M.getModuleFlag("gpuasan.instrumented"))
    return false;
  M.addModuleFlag(Module::Override, "gpuasan.instrumented", 1);
  // Recorded rather than recomputed: the post-link half runs from the backend,
  // where the request that chose this is long gone, and both halves have to
  // call the same entry point.
  if (Recover)
    M.addModuleFlag(Module::Override, "gpuasan.recover", 1);

  declareRuntime(M, Recover, LoadFn, StoreFn);

  // The LDS half runs after LTO has already decided which symbols are live, and
  // the checks it adds may be the only callers of a handler.  Anchoring both
  // here is what keeps the runtime from being stripped out from under them.
  SmallVector<GlobalValue *, 2> Handlers;
  for (FunctionCallee Fn : {LoadFn, StoreFn})
    if (auto *GV = dyn_cast<GlobalValue>(Fn.getCallee()))
      Handlers.push_back(GV);
  appendToCompilerUsed(M, Handlers);

  // Only marked here; padded and described after the link.  See tagGlobals().
  if (ClInstrumentGlobals)
    tagGlobals();

  // Before anything reads an LDS address, and long before the objects are
  // packed.  The checks themselves are emitted by GPUSanitizerLDSPass once that
  // packing has happened.
  if (ClInstrumentLDS)
    padLDSGlobals();

  for (Function &F : M) {
    if (!shouldInstrumentFunction(F))
      continue;
    // The LDS half runs after LTO has merged the runtime and any uninstrumented
    // translation unit into this module, where neither is distinguishable by
    // anything but this mark.
    F.addFnAttr(kFuncMarker);
    SmallVector<AccessInfo, 32> Accesses;
    collectAccesses(F, DL, Accesses);
    for (const AccessInfo &A : Accesses)
      emitPlacedCheck(A, M, LoadFn, StoreFn);
  }
  return true;
}

PreservedAnalyses GPUSanitizerPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!GPUSanitizer(M, Recover).run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

//===----------------------------------------------------------------------===//
// LDS, second half: checking after the objects have been packed
//===----------------------------------------------------------------------===//

namespace {

/// A statically allocated LDS object, in the coordinates the hardware uses: a
/// byte offset from the base of the workgroup's block.
struct LDSObject {
  uint64_t Base;
  uint64_t Size; // data only; the redzone that follows is not part of it
};

/// One LDS variable as AMDGPULowerModuleLDSPass left it.  After that pass a
/// variable is usually a packed block holding many of the program's objects.
struct LDSVar {
  uint64_t Base = 0;
  uint64_t End = 0; // Base plus the total allocated size
  SmallVector<LDSObject, 4> Objects;
  /// Sized by the launch rather than by the program, so the shadow stops here.
  bool Dynamic = false;
  /// Whether the lowering pass recorded an address.  A sized variable without
  /// one leaves a hole in the layout; the dynamic region legitimately has none.
  bool Placed = false;
};

/// LDS is the one address space whose contents are entirely static.  Nothing is
/// allocated or freed while a kernel runs, so the map describing it is a
/// compile-time constant and can be *frozen*: it lives in constant memory,
/// costs no LDS, and needs no initialization, no barrier and no runtime
/// support.  That is the only real difference from ASan's shadow, which has to
/// be writable because heap and stack objects come and go.
///
/// One shadow byte covers one granule of LDS and holds the distance from that
/// granule's base to the end of the object occupying it, saturating at 255.  A
/// check is
///
///     (off & (G-1)) + width > shadow[off >> log2 G]
///
/// Saturation is exact rather than lossy: the offset within a granule is
/// bounded by the granule, so every entry above `G + max width` behaves
/// identically, and an access that far inside an object is in bounds however
/// much larger the object is.  Measuring to the end of the *object* rather than
/// of the granule is what lets an access straddling a granule boundary resolve
/// in a single lookup.
class GPUSanitizerLDS {
public:
  explicit GPUSanitizerLDS(Module &M)
      : M(M), Ctx(M.getContext()), DL(M.getDataLayout()),
        Int8Ty(Type::getInt8Ty(Ctx)), Int32Ty(Type::getInt32Ty(Ctx)),
        Int64Ty(Type::getInt64Ty(Ctx)) {}

  bool run();

private:
  using VarSet = SmallSetVector<GlobalVariable *, 8>;

  /// What one function's checks are built against.
  struct Geometry {
    GlobalVariable *Shadow = nullptr;
    Function *Lookup = nullptr;
    uint64_t Granules = 0;
    /// Offset past which the shadow says nothing.
    uint64_t Limit = 0;
    bool NeedsLimit = false;
    bool HasDynamic = false;
    /// Where the dynamic region begins, when it can be named at all.  Its
    /// extent is not in the shadow, so it is checked separately against what
    /// the dispatch asked for.
    Constant *DynBase = nullptr;

    /// Whether anything at all can be checked against this layout.
    bool usable() const { return Granules || DynBase; }
  };

  /// One checked global: the padded storage its contents live in, the extent
  /// the program asked for, and the name a report should use for it.
  struct GlobalObj {
    GlobalVariable *Storage;
    uint64_t Size;
    std::string Name;
  };

  bool collectGlobals();
  GlobalVariable *padGlobal(GlobalVariable &GV);
  void buildGlobalsTable();
  std::pair<Value *, Value *> globalsBounds(Function &F);
  void emitGlobalCheck(const AccessInfo &A);
  void collectVars();
  void collectReferences(GlobalVariable &GV);
  void collectKernelVars();
  Geometry geometryFor(Function &F, const SmallPtrSetImpl<Function *> &Kernels);
  GlobalVariable *buildShadow(const VarSet &Set, uint64_t Granules);
  Function *buildObjectLookup(const VarSet &Set);
  Value *loadDynLDSSize(IRBuilder<> &IRB);
  void emitCheck(const AccessInfo &A, const Geometry &G);

  const LDSVar &varFor(GlobalVariable *GV) const {
    auto It = Vars.find(GV);
    assert(It != Vars.end() && "LDS variable was never collected");
    return It->second;
  }

  Module &M;
  LLVMContext &Ctx;
  const DataLayout &DL;
  Type *Int8Ty, *Int32Ty, *Int64Ty;
  FunctionCallee LoadFn, StoreFn;

  /// The globals this half collected into the section, the table describing
  /// them, the linker's bounds on the section, and the info block that points
  /// the host at all of it.
  SmallVector<GlobalObj, 16> Globals;
  SmallPtrSet<const Value *, 16> GlobalStorage;
  GlobalVariable *GlobalsInfo = nullptr;
  GlobalVariable *GlobalsTable = nullptr;
  GlobalVariable *GlobalsStart = nullptr;
  GlobalVariable *GlobalsStop = nullptr;
  uint64_t GlobalsGranules = 0;
  DenseMap<Function *, std::pair<Value *, Value *>> GlobalsBounds;

  MapVector<GlobalVariable *, LDSVar> Vars;
  /// Whether the lowering pass built a static offset table, in which case any
  /// function that uses it names every kernel's block rather than just its own.
  bool HaveOffsetTable = false;
  /// Which functions name a variable, and which variables a kernel's call graph
  /// can reach.
  DenseMap<Function *, VarSet> Refs;
  DenseMap<Function *, VarSet> KernelVars;
  DenseMap<Function *, SmallPtrSet<Function *, 2>> Reachers;
  /// Shadows and lookups are shared between functions with identical layouts,
  /// keyed by their contents.
  StringMap<GlobalVariable *> Shadows;
  StringMap<Function *> Lookups;
};

} // namespace

//===----------------------------------------------------------------------===//
// Globals, second half: redzones and the granule table
//===----------------------------------------------------------------------===//

/// Move a global's contents into padded storage inside the section, leaving its
/// own name behind as an alias of it.
///
/// The padding is what gives an overflow somewhere to land that no object
/// claims.  Stock ASan gets it by replacing the variable with a wider one and
/// taking its name, which costs the symbol its size -- harmless on a host,
/// where nothing asks a symbol how large it is at runtime, and not harmless
/// here, where the loader and every hipMemcpyToSymbol do.  An alias keeps the
/// name, the linkage, the visibility and the address, and AsmPrinter takes an
/// alias's ELF size from its own value type whenever the aliasee is private,
/// which is exactly this shape.  Nothing else in the program can tell the
/// difference.
GlobalVariable *GPUSanitizerLDS::padGlobal(GlobalVariable &GV) {
  unsigned AS = GV.getAddressSpace();
  Type *Ty = GV.getValueType();
  uint64_t Size = DL.getTypeAllocSize(Ty).getFixedValue();
  uint64_t Pad = alignTo(Size, kGlobalGranule) - Size + kGlobalRedzone;

  Type *PadTy = ArrayType::get(Int8Ty, Pad);
  StructType *StorageTy = StructType::get(Ctx, {Ty, PadTy});
  Constant *Init = ConstantStruct::get(
      StorageTy, {GV.getInitializer(), Constant::getNullValue(PadTy)});

  auto *Storage = new GlobalVariable(M, StorageTy, /*isConstant=*/false,
                                     GlobalValue::PrivateLinkage, Init, "",
                                     nullptr, GlobalValue::NotThreadLocal, AS);
  Storage->setSection(kGlobalsSection);
  Storage->setExternallyInitialized(GV.isExternallyInitialized());
  // A granule has to belong to exactly one object for the table to describe it.
  Storage->setAlignment(Align(
      std::max<uint64_t>(kGlobalGranule, GV.getAlign().valueOrOne().value())));
  // The payload is at offset zero, so the variable's debug info describes the
  // storage unchanged.
  Storage->copyMetadata(&GV, 0);

  auto *Alias = GlobalAlias::create(Ty, AS, GV.getLinkage(), "", Storage, &M);
  Alias->setVisibility(GV.getVisibility());
  Alias->setUnnamedAddr(GV.getUnnamedAddr());
  Alias->setDSOLocal(GV.isDSOLocal());
  GV.replaceAllUsesWith(Alias);
  Alias->takeName(&GV);
  Storage->setName(Alias->getName() + kGlobalStorageSuffix);
  GV.eraseFromParent();

  ++NumGlobalsCollected;
  return Storage;
}

/// Collect the marked globals into the section and describe them.
///
/// Nothing here rewrites a reference: an alias is a constant, so every use --
/// including one inside another global's initializer, which the old scheme had
/// to refuse outright -- keeps working untouched.
bool GPUSanitizerLDS::collectGlobals() {
  if (!ClInstrumentGlobals)
    return false;

  const uint64_t Budget = ClGlobalsBudget / sizeof(uint64_t);
  SmallVector<GlobalVariable *, 16> Targets;
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.getMetadata(kGlobalTag))
      continue;
    GV.eraseMetadata(Ctx.getMDKindID(kGlobalTag));
    // Re-judged rather than trusted: optimization since the mark was left may
    // have given the variable a section, a comdat or a larger alignment.
    if (!canCheckGlobal(GV, DL))
      continue;

    // What the variable adds to the section: its own granules, its redzone, and
    // whatever the linker must insert to satisfy an alignment above a granule.
    // The sum bounds the section, which is what the table has to cover, and the
    // exact offsets are the linker's business rather than ours.
    uint64_t Size = DL.getTypeAllocSize(GV.getValueType()).getFixedValue();
    uint64_t Alignment =
        std::max<uint64_t>(kGlobalGranule, GV.getAlign().valueOrOne().value());
    uint64_t Span = alignTo(Size, kGlobalGranule) + kGlobalRedzone +
                    (Alignment - kGlobalGranule);
    if (GlobalsGranules + (Span >> kGlobalGranuleLog2) > Budget) {
      // The table is per-image and lives in the image, so a program with an
      // enormous amount of static data would pay for it in size.  Past the
      // budget a variable simply stays out of the section, unchecked and
      // untouched.
      ++NumGlobalsOverBudget;
      continue;
    }
    GlobalsGranules += Span >> kGlobalGranuleLog2;
    Targets.push_back(&GV);
  }
  if (Targets.empty()) {
    GlobalsGranules = 0;
    return false;
  }

  for (GlobalVariable *GV : Targets) {
    GlobalObj G{nullptr,
                DL.getTypeAllocSize(GV->getValueType()).getFixedValue(),
                GV->getName().str()};
    G.Storage = padGlobal(*GV);
    GlobalStorage.insert(G.Storage);
    Globals.push_back(std::move(G));
  }
  buildGlobalsTable();
  return true;
}

/// Emit the per-image metadata: the granule table, the descriptors that tell
/// the host what to write into it, and the info block that ties them together.
///
/// The host has to be able to find one symbol by name, and everything else
/// hangs off it.  Names are needed because a report should say which variable
/// was overrun, and the ELF symbol table cannot supply them: after linking most
/// device globals are internal and have no symbol at all.
void GPUSanitizerLDS::buildGlobalsTable() {
  // The linker's own bounds on the section, which is where the check gets the
  // range it tests against.  Hidden, so they resolve within the image and
  // materialize PC-relative rather than through the GOT; strong, because this
  // module is the one putting objects in the section, so the definitions exist.
  auto BoundSymbol = [&](StringRef Prefix) {
    auto *GV = new GlobalVariable(
        M, Int8Ty, /*isConstant=*/true, GlobalValue::ExternalLinkage, nullptr,
        Prefix + kGlobalsSection, nullptr, GlobalValue::NotThreadLocal,
        kGlobalAddrSpace);
    GV->setVisibility(GlobalValue::HiddenVisibility);
    return GV;
  };
  GlobalsStart = BoundSymbol("__start_");
  GlobalsStop = BoundSymbol("__stop_");

  auto *TableTy = ArrayType::get(Int64Ty, GlobalsGranules);
  GlobalsTable = new GlobalVariable(
      M, TableTy, /*isConstant=*/false, GlobalValue::PrivateLinkage,
      Constant::getNullValue(TableTy), "gpuasan.globals.table", nullptr,
      GlobalValue::NotThreadLocal, kGlobalAddrSpace);
  GlobalsTable->setAlignment(Align(16));
  // Written by the host before the first dispatch.  Without this the optimizer
  // is entitled to notice that nothing in the module ever stores to the table
  // and fold every load of it to zero.
  GlobalsTable->setExternallyInitialized(true);

  StructType *DescTy =
      StructType::get(Ctx, {Int64Ty, Int64Ty, Int64Ty, Int64Ty});
  SmallVector<Constant *, 16> Descs;
  for (const GlobalObj &G : Globals) {
    Constant *Str =
        ConstantDataArray::getString(Ctx, G.Name, /*AddNull=*/false);
    auto *Name = new GlobalVariable(
        M, Str->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage,
        Str, "gpuasan.globals.name", nullptr, GlobalValue::NotThreadLocal,
        kConstantAddrSpace);
    Name->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    // Addresses as integers, so one flat layout serves a global and a constant
    // variable alike.  Each is a relocation, which is how the host learns where
    // the loader put the storage without parsing the image.
    Descs.push_back(ConstantStruct::get(
        DescTy, {ConstantExpr::getPtrToInt(G.Storage, Int64Ty),
                 ConstantInt::get(Int64Ty, G.Size),
                 ConstantExpr::getPtrToInt(Name, Int64Ty),
                 ConstantInt::get(Int64Ty, G.Name.size())}));
  }
  auto *DescsTy = ArrayType::get(DescTy, Descs.size());
  auto *DescArray = new GlobalVariable(
      M, DescsTy, /*isConstant=*/false, GlobalValue::PrivateLinkage,
      ConstantArray::get(DescsTy, Descs), "gpuasan.globals.desc", nullptr,
      GlobalValue::NotThreadLocal, kGlobalAddrSpace);
  DescArray->setAlignment(Align(8));

  SmallVector<Constant *, kGlobalsInfoFields> Fields(
      kGlobalsInfoFields, ConstantInt::get(Int64Ty, 0));
  // The origin the table is indexed from, which the host needs in order to turn
  // a relocated descriptor address into a granule.  It is the same address the
  // check computes its distance from, and it arrives here the same way the
  // descriptors do: as a relocation the loader applies.
  Fields[kGlobalsInfoBase] = ConstantExpr::getPtrToInt(GlobalsStart, Int64Ty);
  Fields[kGlobalsInfoTable] = ConstantExpr::getPtrToInt(GlobalsTable, Int64Ty);
  Fields[kGlobalsInfoGranules] = ConstantInt::get(Int64Ty, GlobalsGranules);
  Fields[kGlobalsInfoNumDescs] = ConstantInt::get(Int64Ty, Descs.size());
  Fields[kGlobalsInfoDescs] = ConstantExpr::getPtrToInt(DescArray, Int64Ty);

  // An image whose table was never written reads as all zeroes, and a zero
  // entry is the permissive one, so a device with no host runtime behind it
  // checks nothing instead of reporting every global access.
  //
  // One image usually comes from one module, but nothing guarantees it, so the
  // name is unique per module and the host iterates rather than looks up: two
  // modules each describe their own contribution to the section.  Weak in case
  // that uniqueness ever fails, which costs the loser its coverage instead of
  // the link.
  auto *InfoTy = ArrayType::get(Int64Ty, kGlobalsInfoFields);
  std::string Id = getUniqueModuleId(&M);
  if (Id.empty())
    Id = "." + utohexstr(xxh3_64bits(M.getModuleIdentifier()));
  GlobalsInfo = new GlobalVariable(
      M, InfoTy, /*isConstant=*/false, GlobalValue::WeakODRLinkage,
      ConstantArray::get(InfoTy, Fields),
      Twine(kGlobalsInfoPrefix) + StringRef(Id).drop_front(), nullptr,
      GlobalValue::NotThreadLocal, kGlobalAddrSpace);
  GlobalsInfo->setVisibility(GlobalValue::ProtectedVisibility);
  GlobalsInfo->setAlignment(Align(8));
  // No device code reads it and the host only reads it, so it is ordinary
  // relocated data rather than something either side initializes.
  //
  // Nothing in the image refers to it by name, and the host has to find it.
  appendToCompilerUsed(M, {GlobalsInfo});
}

/// Where the section is and how far the table describes it, as one pair of
/// values per function.
///
/// Both come from the linker rather than from memory: `__start_`/`__stop_`
/// bound the section it laid out, and an address in an image is materialized
/// PC-relative into a pair of scalar registers, so the whole per-function cost
/// of checking globals is a handful of scalar instructions in the entry block
/// and no loads at all.
///
/// The span is clamped to what the table can name.  Within one module the
/// compiler's bound already covers the section, alignment padding included, but
/// an image can be several modules with a table apiece, and the section symbols
/// bound all of their contributions together.  The clamp is what keeps a check
/// from indexing off the end of its own table; the coverage it costs is the
/// tail of such a section, whose accesses are waved through instead.
std::pair<Value *, Value *> GPUSanitizerLDS::globalsBounds(Function &F) {
  auto It = GlobalsBounds.find(&F);
  if (It != GlobalsBounds.end())
    return It->second;

  BasicBlock &Entry = F.getEntryBlock();
  IRBuilder<> IRB(&Entry, Entry.getFirstNonPHIOrDbgOrAlloca());
  Value *Base =
      IRB.CreatePtrToInt(GlobalsStart, Int64Ty, "gpuasan.globals.base");
  Value *Span = IRB.CreateSub(IRB.CreatePtrToInt(GlobalsStop, Int64Ty), Base,
                              "gpuasan.globals.section");
  Span = IRB.CreateBinaryIntrinsic(
      Intrinsic::umin, Span,
      ConstantInt::get(Int64Ty, GlobalsGranules << kGlobalGranuleLog2), nullptr,
      "gpuasan.globals.span");
  std::pair<Value *, Value *> B{Base, Span};
  GlobalsBounds[&F] = B;
  return B;
}

/// Check an access that may name the globals section.
///
/// Two tests: whether the address is in the section the linker laid out, and
/// then what the granule it lands in says.  A granule entry holds the owning
/// object's offset from `__start_` and the complement of the offset one past
/// its end, so an unwritten entry reads as an object spanning the whole 4 GiB
/// the field can name and waves the access through, while all ones -- what the
/// host writes over every redzone and every byte of padding -- refuses
/// everything.
void GPUSanitizerLDS::emitGlobalCheck(const AccessInfo &A) {
  unsigned AS = A.Ptr->getType()->getPointerAddressSpace();
  if (!isCheckableAddrSpace(AS))
    return;

  // A pointer that provably names some other object cannot name the section.
  // Anything else has to be tested, including a pointer whose provenance is
  // unknown: a global's address that went through memory comes back as an
  // ordinary pointer, and it is still the global.
  const Value *Obj = getUnderlyingObject(A.Ptr);
  if ((isa<GlobalValue>(Obj) || isa<AllocaInst>(Obj)) &&
      !GlobalStorage.contains(Obj))
    return;

  IRBuilder<> IRB(A.I);
  auto [Base, Span] = globalsBounds(*A.I->getFunction());
  Value *Addr = IRB.CreatePtrToInt(A.Ptr, Int64Ty, "gpuasan.globals.addr");
  Value *Delta = IRB.CreateSub(Addr, Base, "gpuasan.globals.delta");
  Value *InSection =
      IRB.CreateICmpULT(Delta, Span, "gpuasan.globals.insection");

  Instruction *CheckTerm = SplitBlockAndInsertIfThen(InSection, A.I, false);
  IRB.SetInsertPoint(CheckTerm);

  Value *Granule = IRB.CreateLShr(Delta, kGlobalGranuleLog2, "gpuasan.granule");
  Value *EntryPtr =
      IRB.CreateGEP(Int64Ty, GlobalsTable, Granule, "gpuasan.globals.entryptr");
  LoadInst *Entry = IRB.CreateAlignedLoad(Int64Ty, EntryPtr, Align(8),
                                          "gpuasan.globals.entry");
  Entry->setMetadata(LLVMContext::MD_nosanitize, MDNode::get(Ctx, {}));
  Entry->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));

  Value *Begin = IRB.CreateAnd(Entry, ConstantInt::get(Int64Ty, UINT32_MAX),
                               "gpuasan.globals.begin");
  Value *End = IRB.CreateLShr(IRB.CreateNot(Entry),
                              ConstantInt::get(Int64Ty, kGlobalEndShift),
                              "gpuasan.globals.end");
  Value *Width = A.DynSize ? IRB.CreateZExtOrTrunc(A.DynSize, Int64Ty)
                           : ConstantInt::get(Int64Ty, A.Size);

  // Both offsets are section-relative and bounded by the span, so the sum
  // cannot wrap and the two-sided form is safe as written.
  Value *Under = IRB.CreateICmpULT(Delta, Begin, "gpuasan.globals.under");
  Value *Over = IRB.CreateICmpUGT(IRB.CreateAdd(Delta, Width), End,
                                  "gpuasan.globals.over");
  Value *Bad = IRB.CreateOr(Under, Over, "gpuasan.globals.bad");
  if (A.DynSize)
    Bad = IRB.CreateAnd(Bad,
                        IRB.CreateICmpNE(Width, ConstantInt::get(Int64Ty, 0)),
                        "gpuasan.globals.bad.nonempty");

  Instruction *ReportTerm = SplitBlockAndInsertIfThen(Bad, CheckTerm, false);
  IRB.SetInsertPoint(ReportTerm);

  // The object the granule named, which for a redzone is nonsense the host
  // replaces: it has the descriptors and can find the neighbour the access
  // really ran off.
  Type *PtrTy = PointerType::get(Ctx, 0);
  Value *ObjBase = IRB.CreateIntToPtr(IRB.CreateAdd(Base, Begin), PtrTy);
  emitReport(LoadFn, StoreFn, IRB, A, IRB.CreateIntToPtr(Addr, PtrTy), Width,
             ObjBase, IRB.CreateSub(End, Begin),
             ConstantInt::get(Int32Ty, kGlobalAddrSpace));

  ++NumGlobalsInstrumented;
}

/// Where the lowering pass finally placed a variable.  Everything downstream
/// depends on this, so a variable without one means the layout is unknown.
static bool absoluteAddress(const GlobalVariable &GV, uint64_t &Addr) {
  MDNode *MD = GV.getMetadata(LLVMContext::MD_absolute_symbol);
  if (!MD || MD->getNumOperands() == 0)
    return false;
  auto *C = mdconst::extract_or_null<ConstantInt>(MD->getOperand(0));
  if (!C)
    return false;
  Addr = C->getZExtValue();
  return true;
}

/// Recognise the wrapper the first half put around an object, which is how the
/// data extent survives being packed into a struct with its neighbours.
static bool paddedObjectSize(Type *Ty, const DataLayout &DL, uint64_t &Size) {
  auto *ST = dyn_cast<StructType>(Ty);
  if (!ST || !ST->hasName() || ST->getNumElements() != 2 ||
      !ST->getName().starts_with(kLDSPadTypeName))
    return false;
  Size = DL.getTypeAllocSize(ST->getElementType(0)).getFixedValue();
  return true;
}

void GPUSanitizerLDS::collectVars() {
  HaveOffsetTable = M.getNamedValue("llvm.amdgcn.lds.offset.table");

  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != kLDSAddrSpace)
      continue;

    Type *Ty = GV.getValueType();
    uint64_t Total =
        Ty->isSized() ? DL.getTypeAllocSize(Ty).getFixedValue() : 0;

    LDSVar V;
    V.Placed = absoluteAddress(GV, V.Base);
    V.End = V.Base + Total;

    // A zero-length variable is the dynamic region: sized by the launch, so the
    // shadow stops where it begins.  It is also the one kind of LDS variable
    // the lowering pass may leave unplaced, since its address is simply the end
    // of the static block.
    if (Total == 0) {
      V.Dynamic = true;
      Vars.insert({&GV, V});
      continue;
    }
    if (!V.Placed) {
      Vars.insert({&GV, V});
      continue;
    }

    if (uint64_t Size; paddedObjectSize(Ty, DL, Size)) {
      V.Objects.push_back({V.Base, Size});
    } else if (auto *ST = dyn_cast<StructType>(Ty)) {
      // The packed block, one member per object.  A member that is not one of
      // our wrappers was never widened, so all of it counts as data and it
      // simply has no redzone.
      const StructLayout *SL = DL.getStructLayout(ST);
      for (unsigned I = 0, E = ST->getNumElements(); I != E; ++I) {
        Type *MemTy = ST->getElementType(I);
        uint64_t MemSize;
        if (!paddedObjectSize(MemTy, DL, MemSize))
          MemSize = DL.getTypeAllocSize(MemTy).getFixedValue();
        V.Objects.push_back({V.Base + SL->getElementOffset(I), MemSize});
      }
    } else {
      V.Objects.push_back({V.Base, Total});
    }
    Vars.insert({&GV, V});
  }
}

/// Which functions name a variable, following constant expressions, since at
/// `-O0` every LDS address arrives as one.
void GPUSanitizerLDS::collectReferences(GlobalVariable &GV) {
  SmallVector<User *, 8> Worklist(GV.users());
  SmallPtrSet<User *, 8> Seen;
  while (!Worklist.empty()) {
    User *U = Worklist.pop_back_val();
    if (!Seen.insert(U).second)
      continue;
    if (auto *I = dyn_cast<Instruction>(U)) {
      Refs[I->getFunction()].insert(&GV);
      continue;
    }
    if (isa<Constant>(U))
      Worklist.append(U->user_begin(), U->user_end());
  }
}

/// A function is checked against the layout of the kernels that reach it, so
/// the call graph has to be walked: LDS addresses are absolute and every
/// kernel's block starts at zero, so an offset means nothing without knowing
/// whose block it indexes.
void GPUSanitizerLDS::collectKernelVars() {
  for (Function &K : M) {
    if (K.isDeclaration() || K.getCallingConv() != CallingConv::AMDGPU_KERNEL)
      continue;
    SmallVector<Function *, 8> Worklist{&K};
    SmallPtrSet<Function *, 8> Seen;
    VarSet &Reached = KernelVars[&K];
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      if (!Seen.insert(F).second)
        continue;
      Reachers[F].insert(&K);
      auto It = Refs.find(F);
      if (It != Refs.end())
        Reached.insert(It->second.begin(), It->second.end());
      for (Instruction &I : instructions(*F))
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction())
            if (!Callee->isDeclaration())
              Worklist.push_back(Callee);
    }
  }
}

GlobalVariable *GPUSanitizerLDS::buildShadow(const VarSet &Set,
                                             uint64_t Granules) {
  const uint64_t Granule = ClLDSGranule;

  // One granule longer than the block: a pointer past the end of LDS clamps
  // onto that entry and finds it poisoned, which costs a `umin` and saves a
  // branch.
  SmallVector<uint8_t, 64> Bytes(Granules + 1, 0);
  for (GlobalVariable *GV : Set) {
    for (const LDSObject &Obj : varFor(GV).Objects) {
      uint64_t ObjEnd = Obj.Base + Obj.Size;
      for (uint64_t G = Obj.Base / Granule; G * Granule < ObjEnd; ++G) {
        uint64_t Remain = ObjEnd - G * Granule;
        // Where two layouts disagree about a granule the permissive one wins,
        // which can only cost a report, never invent one.
        Bytes[G] = std::max<uint8_t>(Bytes[G], std::min<uint64_t>(Remain, 255));
      }
    }
  }

  StringRef Key(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  if (GlobalVariable *Shadow = Shadows.lookup(Key))
    return Shadow;

  Constant *Init = ConstantDataArray::get(Ctx, Bytes);
  auto *Shadow = new GlobalVariable(
      M, Init->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage,
      Init, "gpuasan.lds.shadow", nullptr, GlobalValue::NotThreadLocal,
      kConstantAddrSpace);
  Shadow->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Shadow->setAlignment(Align(16));
  ++NumLDSShadows;
  Shadows[Key] = Shadow;
  return Shadow;
}

/// The shadow answers "how far to the end of whatever is here", which is all a
/// check needs and not enough for a report: it cannot name the object or say
/// where it began.  This recovers both from the offset alone by picking the
/// nearest object, splitting each redzone down the middle so that a hit in the
/// lower half is an overflow of the object below and one in the upper half is
/// an underflow of the object above.  It is a chain of selects over
/// compile-time constants, reached only from a report block, so the fast path
/// pays nothing.
Function *GPUSanitizerLDS::buildObjectLookup(const VarSet &Set) {
  SmallVector<LDSObject, 8> Objs;
  for (GlobalVariable *GV : Set)
    append_range(Objs, varFor(GV).Objects);
  if (Objs.empty())
    return nullptr;
  sort(Objs,
       [](const LDSObject &A, const LDSObject &B) { return A.Base < B.Base; });

  std::string Key;
  raw_string_ostream OS(Key);
  for (const LDSObject &Obj : Objs)
    OS << Obj.Base << ':' << Obj.Size << ',';
  if (Function *Lookup = Lookups.lookup(Key))
    return Lookup;

  StructType *RetTy = StructType::get(Int32Ty, Int32Ty);
  auto *F =
      Function::Create(FunctionType::get(RetTy, {Int32Ty}, false),
                       GlobalValue::InternalLinkage, "gpuasan.lds.object", &M);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::NoInline);
  F->addFnAttr(Attribute::Cold);
  F->addFnAttr(Attribute::DisableSanitizerInstrumentation);
  F->setMemoryEffects(MemoryEffects::none());

  IRBuilder<> IRB(BasicBlock::Create(Ctx, "entry", F));
  Value *Off = F->getArg(0);
  // Anything below the lowest object is its underflow, so it is the default
  // rather than a case.
  Value *Base = ConstantInt::get(Int32Ty, Objs.front().Base);
  Value *Size = ConstantInt::get(Int32Ty, Objs.front().Size);
  uint64_t PrevEnd = Objs.front().Base + Objs.front().Size;
  for (const LDSObject &Obj : drop_begin(Objs)) {
    uint64_t From =
        Obj.Base > PrevEnd ? PrevEnd + (Obj.Base - PrevEnd) / 2 : Obj.Base;
    Value *Hit = IRB.CreateICmpUGE(Off, ConstantInt::get(Int32Ty, From));
    Base = IRB.CreateSelect(Hit, ConstantInt::get(Int32Ty, Obj.Base), Base);
    Size = IRB.CreateSelect(Hit, ConstantInt::get(Int32Ty, Obj.Size), Size);
    PrevEnd = std::max(PrevEnd, Obj.Base + Obj.Size);
  }
  Value *Ret = IRB.CreateInsertValue(PoisonValue::get(RetTy), Base, 0);
  IRB.CreateRet(IRB.CreateInsertValue(Ret, Size, 1));

  Lookups[Key] = F;
  return F;
}

/// The layout a function is checked against.  A function reachable from several
/// kernels has to answer for all of them, and their blocks overlap in address
/// space: the merged shadow takes the most permissive extent for every granule
/// and stops checking at the earliest point any of them stops being static.
/// Both merges lose reports and neither can invent one.
GPUSanitizerLDS::Geometry
GPUSanitizerLDS::geometryFor(Function &F,
                             const SmallPtrSetImpl<Function *> &Kernels) {
  const uint64_t Granule = ClLDSGranule;

  VarSet Reached, DynVars;
  uint64_t End = 0, Limit = UINT64_MAX, OwnDynBase = UINT64_MAX;
  bool UnknownDynBase = false;

  for (Function *K : Kernels) {
    // The placed alias the lowering pass gives this kernel for the dynamic
    // region.  Every kernel's alias is reachable through the offset table, so
    // the name is the only thing that says which one belongs to which kernel.
    SmallString<64> Alias;
    ("llvm.amdgcn." + K->getName() + ".dynlds").toVector(Alias);

    uint64_t KernelEnd = 0, AliasBase = UINT64_MAX, AnyBase = UINT64_MAX;
    bool KernelHasDynamic = false;
    for (GlobalVariable *GV : KernelVars[K]) {
      const LDSVar &V = varFor(GV);
      // A sized object nobody placed leaves a hole in the layout, and a shadow
      // built around it would describe the wrong memory.
      if (!V.Placed && !V.Dynamic) {
        LLVM_DEBUG(dbgs() << "gpuasan: unplaced LDS variable " << GV->getName()
                          << " leaves " << F.getName() << " unchecked\n");
        return Geometry();
      }
      Reached.insert(GV);
      if (!V.Dynamic) {
        KernelEnd = std::max(KernelEnd, V.End);
        continue;
      }
      KernelHasDynamic = true;
      DynVars.insert(GV);
      if (!V.Placed)
        continue;
      if (GV->getName() == Alias)
        AliasBase = V.Base;
      AnyBase = std::min(AnyBase, V.Base);
    }

    uint64_t DynBase = AliasBase != UINT64_MAX ? AliasBase : AnyBase;
    End = std::max(End, KernelEnd);
    // The shadow describes nothing past the end of a kernel's static block, and
    // nothing at or past where its dynamic region begins.  The latter can be
    // below the former, because a static offset table makes every kernel's
    // block reachable from any function that uses it and so inflates the end.
    Limit = std::min(Limit, std::min(KernelEnd, DynBase));
    if (KernelHasDynamic && DynBase == UINT64_MAX)
      UnknownDynBase = true;
    if (K == &F)
      OwnDynBase = DynBase;
  }

  // A dynamic region whose base nobody wrote down is only excluded from the
  // static check by the end of the static block, which is the one number the
  // offset table can inflate past it.  Nothing here can tell the two apart, so
  // this configuration goes unchecked rather than risk reporting a dynamic
  // access as an overflow of some other kernel's object.
  if (UnknownDynBase && HaveOffsetTable) {
    LLVM_DEBUG(dbgs() << "gpuasan: unnameable dynamic LDS base in "
                      << F.getName() << " with an offset table\n");
    return Geometry();
  }

  Geometry G;
  G.Granules = divideCeil(End, Granule);
  G.Limit = Limit;
  G.HasDynamic = !DynVars.empty();
  // Past the limit the shadow reads as poison, which is the right answer for a
  // wild pointer and the wrong one for the dynamic region or for another
  // kernel's larger block.
  G.NeedsLimit = G.HasDynamic || Limit < G.Granules * Granule;

  // The dynamic region is only checked inside a kernel.  Its extent comes from
  // the implicit arguments, which a callee is not guaranteed to have been
  // given, and forcing them onto one would mean rewriting every caller up the
  // stack.
  if (G.HasDynamic && F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
    if (OwnDynBase != UINT64_MAX)
      G.DynBase = ConstantInt::get(Int32Ty, OwnDynBase);
    else if (!G.Granules)
      // Nothing static is reachable, so the dynamic region starts at zero and
      // the check collapses to a single compare against its end.
      G.DynBase = ConstantInt::get(Int32Ty, 0);
    else if (DynVars.size() == 1)
      // Nothing wrote the address down, but the variable's own address is
      // materialised as the enclosing kernel's static LDS size, which is the
      // right number here and nowhere else.
      G.DynBase = ConstantExpr::getPtrToInt(DynVars.front(), Int32Ty);
  }

  if (G.Granules) {
    G.Shadow = buildShadow(Reached, G.Granules);
    G.Lookup = buildObjectLookup(Reached);
  }
  return G;
}

/// How much dynamic LDS this dispatch asked for.  Uniform, and marked invariant
/// so that the repeated reads collapse into one scalar load per region rather
/// than one per check.
Value *GPUSanitizerLDS::loadDynLDSSize(IRBuilder<> &IRB) {
  Function *ArgPtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_implicitarg_ptr);
  Value *Args = IRB.CreateCall(ArgPtr, {});
  Value *Slot =
      IRB.CreateConstInBoundsGEP1_64(Int8Ty, Args, kHiddenDynLDSSizeOffset);
  auto *Load =
      IRB.CreateAlignedLoad(Int32Ty, Slot, Align(4), "gpuasan.dynsize");
  Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));
  return Load;
}

void GPUSanitizerLDS::emitCheck(const AccessInfo &A, const Geometry &G) {
  unsigned AS = A.Ptr->getType()->getPointerAddressSpace();
  if (AS != kLDSAddrSpace && AS != 0)
    return;

  // A width the shadow cannot represent exactly would need a first-byte and a
  // last-byte check.  Until that exists, leaving it alone is the option that
  // cannot report an access which is in bounds.
  const uint64_t Granule = ClLDSGranule;
  if (A.DynSize || A.Size + Granule > 255)
    return;

  IRBuilder<> IRB(A.I);
  Instruction *SplitBefore = A.I;
  Value *Off = nullptr;
  if (AS == kLDSAddrSpace) {
    Off = IRB.CreatePtrToInt(A.Ptr, Int32Ty, "gpuasan.ldsoff");
  } else {
    // A generic pointer is LDS only some of the time, and at `-O0` it is how
    // every LDS access arrives.  The aperture test is also what makes the cast
    // below well defined.
    Function *IsShared =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_is_shared);
    Value *Shared = IRB.CreateCall(IsShared, {A.Ptr}, "gpuasan.isshared");
    SplitBefore = SplitBlockAndInsertIfThen(Shared, A.I, false);
    IRB.SetInsertPoint(SplitBefore);
    Value *Local =
        IRB.CreateAddrSpaceCast(A.Ptr, PointerType::get(Ctx, kLDSAddrSpace));
    Off = IRB.CreatePtrToInt(Local, Int32Ty, "gpuasan.ldsoff");
  }

  Value *Bad = nullptr, *Entry = nullptr;
  if (G.Granules) {
    Value *Index = IRB.CreateLShr(Off, Log2_64(Granule), "gpuasan.granule");
    Index = IRB.CreateBinaryIntrinsic(Intrinsic::umin, Index,
                                      ConstantInt::get(Int32Ty, G.Granules));
    Entry = IRB.CreateZExt(
        IRB.CreateAlignedLoad(Int8Ty,
                              IRB.CreateInBoundsGEP(Int8Ty, G.Shadow, Index),
                              Align(1), "gpuasan.ldsentry"),
        Int32Ty);

    Value *Rem = IRB.CreateAnd(Off, ConstantInt::get(Int32Ty, Granule - 1),
                               "gpuasan.rem");
    Value *Need = IRB.CreateAdd(Rem, ConstantInt::get(Int32Ty, A.Size));
    Bad = IRB.CreateICmpUGT(Need, Entry, "gpuasan.ldsbad");
    if (G.NeedsLimit)
      Bad = IRB.CreateAnd(
          Bad, IRB.CreateICmpULT(Off, ConstantInt::get(Int32Ty, G.Limit),
                                 "gpuasan.static"));
  }

  // Everything above the limit is the dynamic region, whose end is not a
  // property of the program at all.  Comparing against the remaining distance
  // rather than against `Off + Size` keeps a wild offset from wrapping back
  // into bounds.
  Value *DynSize = nullptr, *InDyn = nullptr;
  if (G.DynBase) {
    DynSize = loadDynLDSSize(IRB);
    Value *DynEnd = IRB.CreateAdd(G.DynBase, DynSize, "gpuasan.dynend");
    InDyn = IRB.CreateICmpUGE(Off, G.DynBase, "gpuasan.indyn");
    Value *Past = IRB.CreateICmpUGE(Off, DynEnd);
    Value *Straddles = IRB.CreateICmpULT(IRB.CreateSub(DynEnd, Off),
                                         ConstantInt::get(Int32Ty, A.Size));
    Value *DynBad = IRB.CreateAnd(InDyn, IRB.CreateOr(Past, Straddles));
    Bad = Bad ? IRB.CreateOr(Bad, DynBad, "gpuasan.ldsbad") : DynBad;
  }
  assert(Bad && "geometry with nothing to check");

  Instruction *ReportTerm = SplitBlockAndInsertIfThen(Bad, SplitBefore, false);
  IRB.SetInsertPoint(ReportTerm);

  Value *ObjBase = ConstantInt::get(Int32Ty, 0);
  Value *ObjSize = ConstantInt::get(Int64Ty, 0);
  if (G.Lookup) {
    Value *Obj = IRB.CreateCall(G.Lookup, {Off});
    ObjBase = IRB.CreateExtractValue(Obj, 0);
    ObjSize = IRB.CreateZExt(IRB.CreateExtractValue(Obj, 1), Int64Ty);
  } else if (Entry) {
    ObjBase = IRB.CreateAnd(
        Off, ConstantInt::get(Int32Ty, ~static_cast<uint32_t>(Granule - 1)));
    ObjSize = IRB.CreateZExt(Entry, Int64Ty);
  }
  // The lookup only knows the objects the compiler placed, so a hit above the
  // static block has to be named from the dispatch instead.
  if (InDyn) {
    ObjBase = IRB.CreateSelect(InDyn, G.DynBase, ObjBase);
    ObjSize =
        IRB.CreateSelect(InDyn, IRB.CreateZExt(DynSize, Int64Ty), ObjSize);
  }
  Value *BasePtr =
      IRB.CreateIntToPtr(ObjBase, PointerType::get(Ctx, kLDSAddrSpace));
  emitReport(LoadFn, StoreFn, IRB, A, toGeneric(IRB, A.Ptr),
             ConstantInt::get(Int64Ty, A.Size), toGeneric(IRB, BasePtr),
             ObjSize, ConstantInt::get(Int32Ty, kLDSAddrSpace));

  ++NumLDSInstrumented;
}

bool GPUSanitizerLDS::run() {
  if (!M.getTargetTriple().isAMDGPU() ||
      (!ClInstrumentLDS && !ClInstrumentGlobals))
    return false;

  // Only modules the first half prepared.  Without its LDS widening there are
  // no redzones to land in, without its marks there is nothing to say which
  // globals came from instrumented code, and without its module flag this is
  // not a module we are instrumenting at all.  Lowering runs in both the LTO
  // and the codegen pipelines, so this can also be reached twice.
  if (!M.getModuleFlag("gpuasan.instrumented") ||
      M.getModuleFlag("gpuasan.lds.instrumented"))
    return false;

  declareRuntime(M, M.getModuleFlag("gpuasan.recover") != nullptr, LoadFn,
                 StoreFn);

  // Whole-image: the section a global ends up in, and the table that describes
  // it, are only well defined once every module that contributes is present.
  bool HaveGlobals = collectGlobals();

  collectVars();
  bool HaveLDS = !Vars.empty();
  if (!HaveLDS && !HaveGlobals) {
    M.addModuleFlag(Module::Override, "gpuasan.lds.instrumented", 1);
    return false;
  }
  if (HaveLDS) {
    for (auto &V : Vars)
      collectReferences(*V.first);
    collectKernelVars();
  }

  // Snapshotted: the object lookups are appended to the module as they are
  // needed.
  SmallVector<Function *, 16> Functions;
  for (Function &F : M)
    Functions.push_back(&F);

  for (Function *F : Functions) {
    if (!shouldInstrumentFunction(*F) || !F->hasFnAttribute(kFuncMarker))
      continue;
    F->removeFnAttr(kFuncMarker);

    // One collection for both halves.  Splitting a block leaves the access
    // itself where it was, so the list stays good across the checks emitted
    // around it, and everything emitted here is marked so that neither half
    // instruments the other's metadata loads.
    SmallVector<AccessInfo, 32> Accesses;
    if (HaveGlobals) {
      collectAccesses(*F, DL, Accesses);
      for (const AccessInfo &A : Accesses)
        emitGlobalCheck(A);
    }

    if (!HaveLDS)
      continue;
    auto It = Reachers.find(F);
    if (It == Reachers.end())
      continue;

    Geometry G = geometryFor(*F, It->second);
    LLVM_DEBUG(dbgs() << "gpuasan: " << F->getName() << " under "
                      << It->second.size() << " kernel(s): " << G.Granules
                      << " granules, limit " << G.Limit
                      << (G.HasDynamic ? (G.DynBase ? ", dynamic checked"
                                                    : ", dynamic unchecked")
                                       : "")
                      << (G.usable() ? "" : ", unchecked") << "\n");
    if (!G.usable()) {
      ++NumLDSUnchecked;
      continue;
    }
    // Reading the launch's dynamic LDS request means the implicit arguments
    // have to reach this kernel, whatever an earlier attributor concluded.
    if (G.DynBase)
      F->removeFnAttr("amdgpu-no-implicitarg-ptr");

    if (!HaveGlobals)
      collectAccesses(*F, DL, Accesses);
    for (const AccessInfo &A : Accesses)
      emitCheck(A, G);
  }

  M.addModuleFlag(Module::Override, "gpuasan.lds.instrumented", 1);
  return true;
}

PreservedAnalyses GPUSanitizerLDSPass::run(Module &M, ModuleAnalysisManager &) {
  if (!GPUSanitizerLDS(M).run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

namespace {
class GPUSanitizerLDSLegacyPass : public ModulePass {
public:
  static char ID;
  GPUSanitizerLDSLegacyPass() : ModulePass(ID) {
    initializeGPUSanitizerLDSLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "GPU sanitizer LDS"; }
  bool runOnModule(Module &M) override { return GPUSanitizerLDS(M).run(); }
};
} // namespace

char GPUSanitizerLDSLegacyPass::ID = 0;

INITIALIZE_PASS(GPUSanitizerLDSLegacyPass, "gpuasan-lds",
                "GPU sanitizer LDS instrumentation", false, false)

ModulePass *llvm::createGPUSanitizerLDSLegacyPass() {
  return new GPUSanitizerLDSLegacyPass();
}
