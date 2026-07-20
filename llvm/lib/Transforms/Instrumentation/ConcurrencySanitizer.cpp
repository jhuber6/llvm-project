//===- ConcurrencySanitizer.cpp - watchpoint race detector ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert a lookup before every interesting memory operation and leave the
// operation itself in place. Atomics keep their ordering, syncscope, and
// address space; the runtime never executes them.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ConcurrencySanitizer.h"
#include "MemoryAccessInstrumentation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::memaccess;

#define DEBUG_TYPE "csan"

static cl::opt<bool> ClInstrumentMemoryAccesses(
    "csan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncEntryExit("csan-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool> ClHandleCxxExceptions(
    "csan-handle-cxx-exceptions", cl::init(true),
    cl::desc("Handle C++ exceptions (insert cleanup blocks for unwinding)"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("csan-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "csan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool>
    ClOmitNonCaptured("csan-omit-by-pointer-capturing", cl::init(true),
                      cl::desc("Omit accesses due to pointer capturing"),
                      cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");

static constexpr char kCsanModuleCtorName[] = "csan.module_ctor";
static constexpr char kCsanInitName[] = "__csan_init";

namespace {

/// Flags on the trailing i32 of __csan_{read,write}* / range probes.
/// Further bits (e.g. SCOPED for async copies) can be added without a new
/// callback family. TSan has separate atomic callbacks only because those
/// functions perform the memory op; CSan does not.
enum AccessFlags : unsigned {
  AF_None = 0,
  AF_Atomic = 1u << 0,
  AF_Compound = 1u << 1,
};

struct ConcurrencySanitizer {
  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);

private:
  void initialize(Module &M);
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(MemIntrinsic *M);
  void chooseInstructionsToInstrument(SmallVectorImpl<Instruction *> &Local,
                                      SmallVectorImpl<Instruction *> &All);
  void insertRuntimeIgnores(Function &F);
  FunctionCallee sizedAccessFn(bool IsWrite, bool Unaligned,
                               unsigned Idx) const;
  bool insertSizedProbe(Instruction *I, Value *Addr, Type *AccessTy,
                        const DataLayout &DL, bool IsWrite, unsigned Flags);

  Type *IntptrTy = nullptr;
  IntegerType *FlagsTy = nullptr;
  AddrSpaceRacePolicy RacePolicy = AddrSpaceRacePolicy::FlatOnly;
  FunctionCallee CsanFuncEntry;
  FunctionCallee CsanFuncExit;
  FunctionCallee CsanIgnoreBegin;
  FunctionCallee CsanIgnoreEnd;
  FunctionCallee CsanRead[kNumAccessSizes];
  FunctionCallee CsanWrite[kNumAccessSizes];
  FunctionCallee CsanUnalignedRead[kNumAccessSizes];
  FunctionCallee CsanUnalignedWrite[kNumAccessSizes];
  FunctionCallee CsanVptrUpdate;
  FunctionCallee CsanVptrLoad;
  FunctionCallee CsanReadRange;
  FunctionCallee CsanWriteRange;
};

static std::string csanName(const Twine &Suffix) {
  return ("__csan_" + Suffix).str();
}

static CallInst *insertProbe(Instruction *Before, FunctionCallee Fn,
                             Value *Addr, ArrayRef<Value *> ExtraArgs) {
  InstrumentationIRBuilder IRB(Before);
  SmallVector<Value *, 4> Args;
  Args.push_back(genericCallbackPtr(IRB, Addr));
  Args.append(ExtraArgs.begin(), ExtraArgs.end());
  return IRB.CreateCall(Fn, Args);
}

void insertModuleCtor(Module &M) {
  // Device runtimes do not run llvm.global_ctors; init is explicit.
  if (M.getTargetTriple().isGPU())
    return;
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kCsanModuleCtorName, kCsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}

} // namespace

PreservedAnalyses ConcurrencySanitizerPass::run(Function &F,
                                                FunctionAnalysisManager &FAM) {
  ConcurrencySanitizer CSan;
  if (CSan.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleConcurrencySanitizerPass::run(Module &M,
                                                      ModuleAnalysisManager &) {
  if (checkIfAlreadyInstrumented(M, "nosanitize_concurrency"))
    return PreservedAnalyses::all();
  insertModuleCtor(M);
  return PreservedAnalyses::none();
}

void ConcurrencySanitizer::initialize(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  IntptrTy = DL.getIntPtrType(Ctx);
  RacePolicy = M.getTargetTriple().isGPU() ? AddrSpaceRacePolicy::GPUConcurrent
                                           : AddrSpaceRacePolicy::FlatOnly;

  IRBuilder<> IRB(Ctx);
  AttributeList Attr;
  Attr = Attr.addFnAttribute(Ctx, Attribute::NoUnwind);
  FlagsTy = IRB.getInt32Ty();

  CsanFuncEntry = M.getOrInsertFunction(csanName("func_entry"), Attr,
                                        IRB.getVoidTy(), IRB.getPtrTy());
  CsanFuncExit =
      M.getOrInsertFunction(csanName("func_exit"), Attr, IRB.getVoidTy());
  CsanIgnoreBegin = M.getOrInsertFunction(csanName("ignore_thread_begin"), Attr,
                                          IRB.getVoidTy());
  CsanIgnoreEnd = M.getOrInsertFunction(csanName("ignore_thread_end"), Attr,
                                        IRB.getVoidTy());

  auto getAccessFn = [&](const Twine &Name) {
    return M.getOrInsertFunction(csanName(Name), Attr, IRB.getVoidTy(),
                                 IRB.getPtrTy(), FlagsTy);
  };

  for (unsigned I = 0; I < kNumAccessSizes; ++I) {
    std::string ByteSizeStr = utostr(1U << I);
    CsanRead[I] = getAccessFn("read" + ByteSizeStr);
    CsanWrite[I] = getAccessFn("write" + ByteSizeStr);
    CsanUnalignedRead[I] = getAccessFn("unaligned_read" + ByteSizeStr);
    CsanUnalignedWrite[I] = getAccessFn("unaligned_write" + ByteSizeStr);
  }

  CsanVptrUpdate =
      M.getOrInsertFunction(csanName("vptr_update"), Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), IRB.getPtrTy());
  CsanVptrLoad = M.getOrInsertFunction(csanName("vptr_read"), Attr,
                                       IRB.getVoidTy(), IRB.getPtrTy());
  CsanReadRange =
      M.getOrInsertFunction(csanName("read_range"), Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), IntptrTy, FlagsTy);
  CsanWriteRange =
      M.getOrInsertFunction(csanName("write_range"), Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), IntptrTy, FlagsTy);
}

FunctionCallee ConcurrencySanitizer::sizedAccessFn(bool IsWrite, bool Unaligned,
                                                   unsigned Idx) const {
  if (Unaligned)
    return IsWrite ? CsanUnalignedWrite[Idx] : CsanUnalignedRead[Idx];
  return IsWrite ? CsanWrite[Idx] : CsanRead[Idx];
}

bool ConcurrencySanitizer::insertSizedProbe(Instruction *I, Value *Addr,
                                            Type *AccessTy,
                                            const DataLayout &DL, bool IsWrite,
                                            unsigned Flags) {
  if (Addr->isSwiftError())
    return false;
  int Idx = getAccessSizeIndex(AccessTy, DL);
  if (Idx < 0)
    return false;

  Align Alignment = Align(1);
  if (auto *LI = dyn_cast<LoadInst>(I))
    Alignment = LI->getAlign();
  else if (auto *SI = dyn_cast<StoreInst>(I))
    Alignment = SI->getAlign();
  else if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
    Alignment = RMW->getAlign();
  else if (auto *CAS = dyn_cast<AtomicCmpXchgInst>(I))
    Alignment = CAS->getAlign();

  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(AccessTy);
  const bool Unaligned = !(Flags & AF_Atomic) && Alignment < Align(8) &&
                         (Alignment.value() % (TypeSize / 8)) != 0;

  insertProbe(I, sizedAccessFn(IsWrite, Unaligned, Idx), Addr,
              {ConstantInt::get(FlagsTy, Flags)});
  if (IsWrite)
    ++NumInstrumentedWrites;
  else
    ++NumInstrumentedReads;
  return true;
}

void ConcurrencySanitizer::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local,
    SmallVectorImpl<Instruction *> &All) {
  DenseMap<Value *, size_t> WriteTargets;
  for (Instruction *I : reverse(Local)) {
    const bool IsWrite = isa<StoreInst>(*I);
    Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                          : cast<LoadInst>(I)->getPointerOperand();

    if (!shouldInstrumentAddress(I->getModule(), Addr, RacePolicy))
      continue;

    if (!IsWrite) {
      if (WriteTargets.contains(Addr)) {
        ++NumOmittedReadsBeforeWrite;
        continue;
      }
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(
              isa<GetElementPtrInst>(Addr)
                  ? cast<GetElementPtrInst>(Addr)->getPointerOperand()
                  : Addr)) {
        if (GV->isConstant())
          continue;
      }
    }

    const AllocaInst *AI = findAllocaForValue(Addr);
    if (AI && !PointerMayBeCaptured(AI, /*ReturnCaptures=*/true) &&
        ClOmitNonCaptured) {
      ++NumOmittedNonCaptured;
      continue;
    }

    All.push_back(I);
    if (IsWrite)
      WriteTargets[Addr] = All.size() - 1;
  }
  Local.clear();
}

void ConcurrencySanitizer::insertRuntimeIgnores(Function &F) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  IRB.CreateCall(CsanIgnoreBegin);
  EscapeEnumerator EE(F, "csan_ignore_cleanup", ClHandleCxxExceptions);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(CsanIgnoreEnd);
  }
}

bool ConcurrencySanitizer::sanitizeFunction(Function &F,
                                            const TargetLibraryInfo &TLI) {
  if (skipInstrumentation(F, kCsanModuleCtorName))
    return false;

  initialize(*F.getParent());
  SmallVector<Instruction *, 8> AllLoadsAndStores;
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  bool Res = false;
  bool HasCalls = false;
  const bool SanitizeFunction =
      F.hasFnAttribute(Attribute::SanitizeConcurrency);
  const DataLayout &DL = F.getDataLayout();

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (Inst.hasMetadata(LLVMContext::MD_nosanitize))
        continue;
      if (isAtomicMemoryAccess(&Inst))
        AtomicAccesses.push_back(&Inst);
      else if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);
        if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst))
          MemIntrinCalls.push_back(MI);
        HasCalls = true;
        chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores);
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores);
  }

  if (ClInstrumentMemoryAccesses && SanitizeFunction)
    for (Instruction *I : AllLoadsAndStores)
      Res |= instrumentLoadOrStore(I, DL);

  if (ClInstrumentAtomics && SanitizeFunction)
    for (Instruction *I : AtomicAccesses)
      Res |= instrumentAtomic(I, DL);

  if (ClInstrumentMemIntrinsics && SanitizeFunction)
    for (Instruction *I : MemIntrinCalls)
      Res |= instrumentMemIntrinsic(cast<MemIntrinsic>(I));

  if (F.hasFnAttribute("sanitize_concurrency_no_checking_at_run_time")) {
    assert(!F.hasFnAttribute(Attribute::SanitizeConcurrency));
    if (HasCalls)
      insertRuntimeIgnores(F);
  }

  if ((Res || HasCalls) && ClInstrumentFuncEntryExit &&
      !F.getParent()->getTargetTriple().isGPU()) {
    InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                                 F.getEntryBlock().getFirstNonPHIIt());
    Type *ProgramAsPtrTy = PointerType::get(F.getParent()->getContext(),
                                            DL.getProgramAddressSpace());
    Value *ReturnAddress = IRB.CreateIntrinsic(
        Intrinsic::returnaddress, {ProgramAsPtrTy}, IRB.getInt32(0));
    IRB.CreateCall(CsanFuncEntry, ReturnAddress);

    EscapeEnumerator EE(F, "csan_cleanup", ClHandleCxxExceptions);
    while (IRBuilder<> *AtExit = EE.Next()) {
      InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
      AtExit->CreateCall(CsanFuncExit, {});
    }
    Res = true;
  }
  return Res;
}

bool ConcurrencySanitizer::instrumentLoadOrStore(Instruction *I,
                                                 const DataLayout &DL) {
  const bool IsWrite = isa<StoreInst>(*I);
  Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                        : cast<LoadInst>(I)->getPointerOperand();
  Type *OrigTy = getLoadStoreType(I);

  if (isVtableAccess(I)) {
    InstrumentationIRBuilder IRB(I);
    Value *Gen = genericCallbackPtr(IRB, Addr);
    if (IsWrite) {
      Value *StoredValue = cast<StoreInst>(I)->getValueOperand();
      if (isa<VectorType>(StoredValue->getType()))
        StoredValue = IRB.CreateExtractElement(
            StoredValue, ConstantInt::get(IRB.getInt32Ty(), 0));
      if (StoredValue->getType()->isIntegerTy())
        StoredValue = IRB.CreateIntToPtr(StoredValue, IRB.getPtrTy());
      IRB.CreateCall(CsanVptrUpdate, {Gen, StoredValue});
      ++NumInstrumentedWrites;
    } else {
      IRB.CreateCall(CsanVptrLoad, Gen);
      ++NumInstrumentedReads;
    }
    return true;
  }

  return insertSizedProbe(I, Addr, OrigTy, DL, IsWrite, AF_None);
}

bool ConcurrencySanitizer::instrumentAtomic(Instruction *I,
                                            const DataLayout &DL) {
  if (isa<FenceInst>(I))
    return false;

  Value *Addr = nullptr;
  Type *AccessTy = nullptr;
  bool IsWrite = true;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    Addr = LI->getPointerOperand();
    AccessTy = LI->getType();
    IsWrite = false;
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    Addr = SI->getPointerOperand();
    AccessTy = SI->getValueOperand()->getType();
  } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
    Addr = RMW->getPointerOperand();
    AccessTy = RMW->getValOperand()->getType();
  } else if (auto *CAS = dyn_cast<AtomicCmpXchgInst>(I)) {
    Addr = CAS->getPointerOperand();
    AccessTy = CAS->getNewValOperand()->getType();
  } else {
    return false;
  }

  if (!shouldInstrumentAddress(I->getModule(), Addr, RacePolicy))
    return false;
  unsigned Flags = AF_Atomic;
  if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
    Flags |= AF_Compound;
  return insertSizedProbe(I, Addr, AccessTy, DL, IsWrite, Flags);
}

bool ConcurrencySanitizer::instrumentMemIntrinsic(MemIntrinsic *M) {
  InstrumentationIRBuilder IRB(M);
  Value *Len = IRB.CreateIntCast(M->getLength(), IntptrTy, false);
  Value *Zero = ConstantInt::get(FlagsTy, AF_None);
  bool DidInstrument = false;
  if (auto *MS = dyn_cast<MemSetInst>(M)) {
    if (!shouldInstrumentAddress(M->getModule(), MS->getRawDest(), RacePolicy))
      return false;
    IRB.CreateCall(CsanWriteRange,
                   {genericCallbackPtr(IRB, MS->getRawDest()), Len, Zero});
    ++NumInstrumentedWrites;
    return true;
  }
  auto *MT = cast<MemTransferInst>(M);
  if (shouldInstrumentAddress(M->getModule(), MT->getRawSource(), RacePolicy)) {
    IRB.CreateCall(CsanReadRange,
                   {genericCallbackPtr(IRB, MT->getRawSource()), Len, Zero});
    ++NumInstrumentedReads;
    DidInstrument = true;
  }
  if (shouldInstrumentAddress(M->getModule(), MT->getRawDest(), RacePolicy)) {
    IRB.CreateCall(CsanWriteRange,
                   {genericCallbackPtr(IRB, MT->getRawDest()), Len, Zero});
    ++NumInstrumentedWrites;
    DidInstrument = true;
  }
  return DidInstrument;
}
