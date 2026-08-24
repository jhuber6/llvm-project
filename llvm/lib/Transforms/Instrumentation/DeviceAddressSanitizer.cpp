//===- DeviceAddressSanitizer.cpp - device address sanitizer --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bounds checking for device code. Allocations live in [2^45, 2^46). That
// window is 32 regions of 2^40 bytes, one per size class, in virtual memory:
//
//   bits [45]     = 1                      in the window
//   bits [44:40]  = C                      size class
//   bits [39:0]   = offset in region C
//
// Class C has chunks of 2^(8 + C) bytes. Chunk index is offset >> (8 + C).
// Metadata is an array of i64 at the high end of the region, one word per
// reachable chunk, indexed backwards from the last byte:
//
//   [ chunk 0 | chunk 1 | ... | unused | ... | md N-1 | ... | md 1 | md 0 ]
//   |<------ 2^40 bytes ----------------------------------------------- >|
//
// Accesses of the same inbounds root in the same original block share one
// bound.
//
//   %v = load i32, ptr addrspace(1) %p
//
// becomes:
//
//   ; bound: decode base(%p)
//   %addr     = ptrtoint ptr addrspace(1) %p to i64
//   %valid    = icmp eq i64 (lshr %addr, 45), 1      ; bit 45: in the window?
//   %class    = and i64 (lshr %addr, 40), 31         ; bits [44:40]: size class
//   %shift    = add i64 %class, 8                    ; chunk size 2^(8+C)
//   %inregion = and i64 %addr, (1<<40)-1             ; offset in this class
//   %idx      = lshr i64 %inregion, %shift           ; chunk index ("block")
//   %slot     = or i64 (shl %idx, 3), 7              ; last byte of md word idx
//   %meta     = sub i64 (%addr or (1<<40)-1), %slot  ; region top minus slot
//   %ep       = inttoptr (select %valid, %meta, @__dasan_no_entry)
//   %entry    = load i64, ptr addrspace(1) %ep, align 8, !invariant.load
//   %size     = and i64 %entry, (1<<40)-1            ; live allocation size
//   %off      = ashr i64 (shl %entry, 1), 41         ; signed chunk-base offset
//   %chunkoff = sub i64 %inregion, (shl %idx, %shift)
//   %start    = sub i64 %addr, (sub %chunkoff, %off) ; object base
//
//   ; check: this access against that bound
//   %oob = icmp uge i64 (%addr - %start + 3), %size   ; last byte of i32
//   %bad = and i1 %valid, %oob                        ; ignore non-window ptrs
//   %vote = call i64 @llvm.amdgcn.ballot.i64(i1 %bad) ; uniform branch
//   br i1 (%vote != 0), label %fail, label %go, !prof !unlikely
// fail:
//   call void @__dasan_report_load(ptr %p, i64 4, i64 %vote)
//   unreachable
// go:
//   %v = load i32, ptr addrspace(1) %p
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/DeviceAddressSanitizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "dasan"

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumDecodes, "Number of base pointer decodes emitted");

// Constants describing the layout of the tracked memory region.
static constexpr uint64_t kSpaceSizeLog = 45;
static constexpr uint64_t kNumClassesLog = 5;
static constexpr uint64_t kClassMask = (1ULL << kNumClassesLog) - 1;
static constexpr uint64_t kRegionSizeLog = kSpaceSizeLog - kNumClassesLog;
static constexpr uint64_t kRegionMask = (1ULL << kRegionSizeLog) - 1;
static constexpr uint64_t kMinSizeLog = 8;
static constexpr uint64_t kMetadataSizeLog = 3;
static constexpr uint64_t kMetadataSize = 1ULL << kMetadataSizeLog;
static constexpr uint64_t kSizeBits = 40;
static constexpr uint64_t kSizeMask = (1ULL << kSizeBits) - 1;
static constexpr uint64_t kOffsetBits = 23;
static constexpr uint64_t kAllocAlignment = 256;
static constexpr unsigned kWordBits = 64;
static_assert(kSizeBits + kOffsetBits <= kWordBits, "an entry is one word");

static constexpr uint64_t kMetadataBytes =
    (1ULL << (kRegionSizeLog - kMinSizeLog)) << kMetadataSizeLog;

static constexpr uint64_t countSizeClasses() {
  uint64_t N = 0;
  while (N <= kClassMask &&
         (1ULL << (kMinSizeLog + N)) <= (kRegionMask + 1) - kMetadataBytes)
    ++N;
  return N;
}
static_assert(countSizeClasses() << kRegionSizeLog == 1ULL << kSpaceSizeLog,
              "the layout must keep every size class");

static constexpr char kReportLoadName[] = "__dasan_report_load";
static constexpr char kReportStoreName[] = "__dasan_report_store";
static constexpr char kReportSharedLoadName[] = "__dasan_report_shared_load";
static constexpr char kReportSharedStoreName[] = "__dasan_report_shared_store";
static constexpr char kRecoverSuffix[] = "_noabort";
static constexpr char kModuleMarkerName[] = "__dasan_instrumented";
static constexpr char kGlobalGuardName[] = "__dasan_guard";
static constexpr char kNoEntryName[] = "__dasan_no_entry";
static constexpr char kGlobalsSectionName[] = ".rodata.dasan.globals";
static constexpr char kGlobalsDataSectionName[] = ".data.dasan.globals";
static constexpr char kGlobalsRelRoSectionName[] = ".data.rel.ro.dasan.globals";
static constexpr char kGlobalsBssSectionName[] = ".sbss.dasan.globals";

static cl::opt<bool> ClInstrumentReads("dasan-instrument-reads", cl::Hidden,
                                       cl::init(true));
static cl::opt<bool> ClInstrumentWrites("dasan-instrument-writes", cl::Hidden,
                                        cl::init(true));
static cl::opt<bool> ClInstrumentAtomics("dasan-instrument-atomics", cl::Hidden,
                                         cl::init(true));
static cl::opt<bool> ClRecover("dasan-recover", cl::Hidden, cl::init(false));
static cl::opt<bool> ClTrapOnError("dasan-trap-on-error", cl::Hidden,
                                   cl::init(false));
static cl::opt<bool> ClInstrumentGlobals("dasan-instrument-globals", cl::Hidden,
                                         cl::init(true));
static cl::opt<bool> ClInstrumentShared("dasan-instrument-shared", cl::Hidden,
                                        cl::init(true));

namespace {

// A single memory access.
struct Access {
  Instruction *I;
  BasicBlock *Block;
  Value *Ptr;
  Value *Len;
  uint64_t FixedLen;
  MaybeAlign Alignment;
  bool IsWrite;
};

// Access information from its associated metadata.
struct Bound {
  Value *Start;
  Value *Size;
  Value *Valid;
};

class NoSanitizeInserter final : public IRBuilderDefaultInserter {
public:
  void InsertHelper(Instruction *I, const Twine &Name,
                    BasicBlock::iterator IP) const override {
    IRBuilderDefaultInserter::InsertHelper(I, Name, IP);
    I->setMetadata(LLVMContext::MD_nosanitize,
                   MDNode::get(I->getContext(), {}));
  }
};

using BuilderTy = IRBuilder<InstSimplifyFolder, NoSanitizeInserter>;

// We emit all failures through a single common branch in the function. Ideally
// this could be inferred by optimizations but for now this is crucial to
// minimizing the register pressure created from the instrumentation.
struct FailSink {
  BasicBlock *BB = nullptr;
  PHINode *Addr = nullptr;
  PHINode *Len = nullptr;
  PHINode *Mask = nullptr;
  PHINode *PC = nullptr;
  PHINode *Start = nullptr;
  PHINode *Size = nullptr;
};

class DeviceAddressSanitizer {
public:
  DeviceAddressSanitizer(Module &M, const DeviceAddressSanitizerOptions &Opts)
      : M(M), C(M.getContext()), DL(M.getDataLayout()),
        IntptrTy(DL.getIntPtrType(C)), Int32Ty(Type::getInt32Ty(C)),
        Recover(ClRecover.getNumOccurrences() ? ClRecover : Opts.Recover),
        CheckShared(ClInstrumentShared) {}

  bool run();

private:
  void initializeCallbacks();
  bool placeGlobals();
  void emitModuleMarker();
  bool instrumentFunction(Function &F);
  void collect(Instruction &I, SmallVectorImpl<Access> &Accesses) const;
  bool instrumentPointer(Value *Ptr) const;
  bool shouldPlaceGlobal(const GlobalVariable &G) const;

  Bound heapBound(BuilderTy &IRB, Value *Base);
  Bound sharedBound(BuilderTy &IRB, Value *Ptr, Value *Limit);
  void check(BuilderTy &IRB, const Access &A, const Bound &B, Instruction *At,
             bool Shared, FailSink Fail[2][2], BasicBlock *Trap);
  Value *sharedIntPtr(BuilderTy &IRB, Value *P);
  Value *groupSegmentLimit(BuilderTy &IRB, Function &F);
  Value *vote(BuilderTy &IRB, Value *Bad, Value *&FailMask);
  void splitTo(BuilderTy &IRB, Instruction *At, Value *Any, BasicBlock *Fail);
  BasicBlock *createTrap(Function &F);
  FailSink createFail(Function &F, bool Shared, bool IsWrite);
  void emitReport(BuilderTy &IRB, Value *Addr, Value *Len, Value *Mask,
                  Value *PC, Value *Start, Value *Size, bool Shared,
                  bool IsWrite);
  GlobalVariable *sizedSharedGV(const Value *P) const;
  bool aborting() const { return !Recover || ClTrapOnError; }

  Constant *intptr(uint64_t V) const { return ConstantInt::get(IntptrTy, V); }
  Constant *int32(uint32_t V) const { return ConstantInt::get(Int32Ty, V); }
  void setPoint(BuilderTy &IRB, Instruction *At) const {
    IRB.SetInsertPoint(At);
    IRB.SetCurrentDebugLocation(Loc);
  }

  Module &M;
  LLVMContext &C;
  const DataLayout &DL;
  IntegerType *IntptrTy;
  Type *Int32Ty;
  bool Recover;
  bool CheckShared;
  FunctionCallee Report[2];
  FunctionCallee ReportShared[2];
  Constant *NoEntry = nullptr;
  DebugLoc Loc;
};

} // namespace

// Aliases to the relevant address spaces for the target.
static constexpr unsigned FlatAS = AMDGPUAS::FLAT_ADDRESS;
static constexpr unsigned GlobalAS = AMDGPUAS::GLOBAL_ADDRESS;
static constexpr unsigned LocalAS = AMDGPUAS::LOCAL_ADDRESS;
static constexpr unsigned ConstantAS = AMDGPUAS::CONSTANT_ADDRESS;
static constexpr unsigned PrivateAS = AMDGPUAS::PRIVATE_ADDRESS;

static bool isLocalAS(unsigned AS) { return AS == LocalAS; }
static bool isPrivateAS(unsigned AS) { return AS == PrivateAS; }
static bool isHeapAS(unsigned AS) {
  return AS == FlatAS || AS == GlobalAS || AS == ConstantAS;
}

// Special accesses that should not be instrumented like the dispatch packet.
static bool isSpecialMemory(const Value *Obj) {
  const auto *II = dyn_cast<IntrinsicInst>(Obj);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::amdgcn_dispatch_ptr:
  case Intrinsic::amdgcn_implicitarg_ptr:
  case Intrinsic::amdgcn_kernarg_segment_ptr:
  case Intrinsic::amdgcn_queue_ptr:
  case Intrinsic::amdgcn_implicit_buffer_ptr:
    return true;
  default:
    return false;
  }
}

bool DeviceAddressSanitizer::instrumentPointer(Value *Ptr) const {
  unsigned AS = Ptr->getType()->getPointerAddressSpace();
  if (!isHeapAS(AS) && !isLocalAS(AS) && !isPrivateAS(AS))
    return false;
  if (isLocalAS(AS))
    return CheckShared;
  if (isPrivateAS(AS))
    return false;
  const Value *Obj = getUnderlyingObject(Ptr);
  return !isa<AllocaInst>(Obj) && !isSpecialMemory(Obj);
}

void DeviceAddressSanitizer::collect(Instruction &I,
                                     SmallVectorImpl<Access> &Accesses) const {
  auto Add = [&](Value *Ptr, Value *Len, uint64_t Fixed, MaybeAlign A,
                 bool IsWrite) {
    if (!(IsWrite ? ClInstrumentWrites : ClInstrumentReads))
      return;
    if (!instrumentPointer(Ptr))
      return;
    Accesses.push_back({&I, I.getParent(), Ptr, Len,
                        Fixed <= UINT32_MAX ? Fixed : 0, A, IsWrite});
  };
  auto AddTyped = [&](Value *Ptr, Type *Ty, MaybeAlign A, bool IsWrite) {
    TypeSize Size = DL.getTypeStoreSize(Ty);
    if (Size.isScalable() || Size.isZero())
      return;
    Add(Ptr, intptr(Size.getFixedValue()), Size.getFixedValue(), A, IsWrite);
  };

  // Collect every interesting instruction that touches memory.
  if (auto *LI = dyn_cast<LoadInst>(&I))
    AddTyped(LI->getPointerOperand(), LI->getType(), LI->getAlign(), false);
  else if (auto *SI = dyn_cast<StoreInst>(&I))
    AddTyped(SI->getPointerOperand(), SI->getValueOperand()->getType(),
             SI->getAlign(), true);
  else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
    if (ClInstrumentAtomics)
      AddTyped(RMW->getPointerOperand(), RMW->getValOperand()->getType(),
               std::nullopt, true);
  } else if (auto *XCHG = dyn_cast<AtomicCmpXchgInst>(&I)) {
    if (ClInstrumentAtomics)
      AddTyped(XCHG->getPointerOperand(), XCHG->getCompareOperand()->getType(),
               std::nullopt, true);
  } else if (auto *MI = dyn_cast<AnyMemIntrinsic>(&I)) {
    Value *Len = MI->getLength();
    if (Len->getType()->getIntegerBitWidth() > IntptrTy->getBitWidth())
      return;
    auto *CI = dyn_cast<ConstantInt>(Len);
    if (CI && CI->isZero())
      return;
    uint64_t Fixed = CI ? CI->getZExtValue() : 0;
    Add(MI->getRawDest(), Len, Fixed, MI->getDestAlign(), true);
    if (auto *MT = dyn_cast<AnyMemTransferInst>(MI))
      Add(MT->getRawSource(), Len, Fixed, MT->getSourceAlign(), false);
  }
}

static bool isSafeAccess(ObjectSizeOffsetVisitor &Vis, Value *Ptr,
                         uint64_t Bytes) {
  // Do not check accesses that are provably inbounds their respecitve object.
  SizeOffsetAPInt SO = Vis.compute(Ptr);
  if (!SO.bothKnown())
    return false;
  uint64_t Size = SO.Size.getZExtValue();
  int64_t Off = SO.Offset.getSExtValue();

  // Three checks are required to ensure safety:
  // . Offset >= 0  (since the offset is given from the base ptr)
  // . Size >= Offset  (unsigned)
  // . Size - Offset >= NeededSize  (unsigned)
  bool Safe =
      Off >= 0 && Size >= uint64_t(Off) && Size - uint64_t(Off) >= Bytes;
  return Safe;
}

static bool lastByteSuffices(const Access &A, uint64_t Bound) {
  return A.FixedLen && isPowerOf2_64(A.FixedLen) && A.FixedLen <= Bound &&
         A.Alignment && A.Alignment->value() >= A.FixedLen;
}

static uint64_t sharedAllocSize(const GlobalVariable &G, const DataLayout &DL) {
  if (!isLocalAS(G.getAddressSpace()) || G.isDeclaration())
    return 0;
  Type *Ty = G.getValueType();
  if (!Ty->isSized())
    return 0;
  uint64_t Size = DL.getTypeAllocSize(Ty).getFixedValue();
  return Size && Size <= UINT32_MAX ? Size : 0;
}

GlobalVariable *DeviceAddressSanitizer::sizedSharedGV(const Value *P) const {
  auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(P));
  return GV && sharedAllocSize(*GV, DL) ? const_cast<GlobalVariable *>(GV)
                                        : nullptr;
}

static GlobalVariable *anySharedGV(const Value *P) {
  auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(P));
  return GV && isLocalAS(GV->getAddressSpace())
             ? const_cast<GlobalVariable *>(GV)
             : nullptr;
}

Bound DeviceAddressSanitizer::heapBound(BuilderTy &IRB, Value *Base) {
  // Compute the address in the metadata region and return its pointer bounds.
  Value *Addr = IRB.CreatePtrToInt(Base, IntptrTy);
  Value *Valid =
      IRB.CreateICmpEQ(IRB.CreateLShr(Addr, kSpaceSizeLog), intptr(1));
  Value *Class =
      IRB.CreateAnd(IRB.CreateLShr(Addr, kRegionSizeLog), intptr(kClassMask));
  Value *Shift = IRB.CreateAdd(Class, intptr(kMinSizeLog));
  Value *InRegion = IRB.CreateAnd(Addr, intptr(kRegionMask));
  Value *Idx = IRB.CreateLShr(InRegion, Shift);
  Value *Slot = IRB.CreateOr(IRB.CreateShl(Idx, kMetadataSizeLog),
                             intptr(kMetadataSize - 1));
  LoadInst *Entry = IRB.CreateAlignedLoad(
      IntptrTy,
      IRB.CreateIntToPtr(
          IRB.CreateSelect(
              Valid,
              IRB.CreateSub(IRB.CreateOr(Addr, intptr(kRegionMask)), Slot),
              NoEntry),
          IRB.getPtrTy(GlobalAS)),
      Align(kMetadataSize));

  // The value can change underneath us in the case of a use-after-free. This
  // trades that coverage for being able to hoist this out of any loops.
  Entry->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, {}));
  Value *Size = IRB.CreateAnd(Entry, intptr(kSizeMask));
  Value *Off =
      IRB.CreateAShr(IRB.CreateShl(Entry, kWordBits - kOffsetBits - kSizeBits),
                     kWordBits - kOffsetBits);
  Value *ChunkOff = IRB.CreateSub(InRegion, IRB.CreateShl(Idx, Shift));
  Value *Start = IRB.CreateSub(Addr, IRB.CreateSub(ChunkOff, Off));

  ++NumDecodes;
  return {Start, Size, Valid};
}

Value *DeviceAddressSanitizer::sharedIntPtr(BuilderTy &IRB, Value *P) {
  return IRB.CreateZExtOrTrunc(
      IRB.CreatePtrToInt(P, DL.getIntPtrType(P->getType())), Int32Ty);
}

Bound DeviceAddressSanitizer::sharedBound(BuilderTy &IRB, Value *Ptr,
                                          Value *Limit) {
  if (GlobalVariable *GV = sizedSharedGV(Ptr))
    return {ConstantExpr::getPtrToInt(GV, Int32Ty),
            int32(sharedAllocSize(*GV, DL)), nullptr};
  return {int32(0), Limit, nullptr};
}

Value *DeviceAddressSanitizer::groupSegmentLimit(BuilderTy &IRB, Function &F) {
  IRB.SetInsertPoint(&*F.getEntryBlock().getFirstInsertionPt());
  CallInst *Packet =
      IRB.CreateIntrinsicWithoutFolding(Intrinsic::amdgcn_dispatch_ptr, {});
  Packet->addRetAttr(Attribute::NoAlias);
  Packet->addRetAttr(Attribute::NonNull);
  Packet->addDereferenceableRetAttr(64);

  // Offset of the group_segment_size field in the HSA dispatch pointer.
  static constexpr uint64_t kGroupSegmentSizeOffset = 28;
  auto *Load =
      IRB.CreateAlignedLoad(Int32Ty,
                            IRB.CreateConstInBoundsGEP1_64(
                                Int32Ty, Packet, kGroupSegmentSizeOffset / 4),
                            Align(4));
  Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, {}));
  return Load;
}

Value *DeviceAddressSanitizer::vote(BuilderTy &IRB, Value *Bad,
                                    Value *&FailMask) {
  FailMask = IRB.getInt64(0);
  if (!M.getTargetTriple().isAMDGPU())
    return Bad;
  FailMask =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_ballot, {IRB.getInt64Ty()}, {Bad});
  return IRB.CreateIsNotNull(FailMask);
}

void DeviceAddressSanitizer::splitTo(BuilderTy &IRB, Instruction *At,
                                     Value *Any, BasicBlock *Fail) {
  BasicBlock *Orig = At->getParent();
  Orig->splitBasicBlock(At, "dasan.cont");
  Instruction *Uncond = Orig->getTerminator();
  IRB.SetInsertPoint(Uncond);
  IRB.SetCurrentDebugLocation(Loc);
  IRB.CreateCondBr(Any, Fail, At->getParent(),
                   MDBuilder(C).createUnlikelyBranchWeights());
  Uncond->eraseFromParent();
  setPoint(IRB, At);
}

BasicBlock *DeviceAddressSanitizer::createTrap(Function &F) {
  BasicBlock *BB = BasicBlock::Create(C, "dasan.trap", &F);
  BuilderTy IRB(C, InstSimplifyFolder(DL));
  IRB.SetInsertPoint(BB);
  if (DISubprogram *SP = F.getSubprogram())
    IRB.SetCurrentDebugLocation(DILocation::get(C, SP->getScopeLine(), 0, SP));
  IRB.CreateIntrinsic(Intrinsic::trap, {});
  IRB.CreateUnreachable();
  return BB;
}

FailSink DeviceAddressSanitizer::createFail(Function &F, bool Shared,
                                            bool IsWrite) {
  FailSink S;
  S.BB = BasicBlock::Create(C, "dasan.fail", &F);
  BuilderTy IRB(C, InstSimplifyFolder(DL));
  IRB.SetInsertPoint(S.BB);
  if (DISubprogram *SP = F.getSubprogram())
    IRB.SetCurrentDebugLocation(DILocation::get(C, SP->getScopeLine(), 0, SP));
  S.Addr = IRB.CreatePHI(IntptrTy, 8);
  S.Len = IRB.CreatePHI(IntptrTy, 8);
  S.Mask = IRB.CreatePHI(IRB.getInt64Ty(), 8);
  S.PC = IRB.CreatePHI(IntptrTy, 8);
  if (Shared) {
    S.Start = IRB.CreatePHI(IntptrTy, 8);
    S.Size = IRB.CreatePHI(IntptrTy, 8);
  }
  emitReport(IRB, S.Addr, S.Len, S.Mask, S.PC, S.Start, S.Size, Shared,
             IsWrite);
  IRB.CreateUnreachable();
  return S;
}

void DeviceAddressSanitizer::emitReport(BuilderTy &IRB, Value *Addr, Value *Len,
                                        Value *Mask, Value *PC, Value *Start,
                                        Value *Size, bool Shared,
                                        bool IsWrite) {
  Value *P = IRB.CreateIntToPtr(Addr, IRB.getPtrTy());
  Value *Pc = IRB.CreateIntToPtr(PC, IRB.getPtrTy());
  if (Shared)
    IRB.CreateCall(ReportShared[IsWrite], {P, Len, Start, Size, Mask, Pc});
  else
    IRB.CreateCall(Report[IsWrite], {P, Len, Mask, Pc});
}

static Value *outOfBounds(BuilderTy &IRB, Value *Addr, Value *Len, Value *Start,
                          Value *Size, bool LastByteOnly) {
  Value *One = ConstantInt::get(Len->getType(), 1);
  Value *First = IRB.CreateSub(Addr, Start);
  Value *Last = IRB.CreateAdd(First, IRB.CreateSub(Len, One));
  return IRB.CreateICmpUGE(
      LastByteOnly ? Last
                   : IRB.CreateBinaryIntrinsic(Intrinsic::umax, First, Last),
      Size);
}

void DeviceAddressSanitizer::check(BuilderTy &IRB, const Access &A,
                                   const Bound &B, Instruction *At, bool Shared,
                                   FailSink Fail[2][2], BasicBlock *Trap) {
  // Check if the given access fits within the known offset and size.
  setPoint(IRB, At);
  Value *Addr =
      Shared ? sharedIntPtr(IRB, A.Ptr) : IRB.CreatePtrToInt(A.Ptr, IntptrTy);
  Value *Len = IRB.CreateZExtOrTrunc(A.Len, IntptrTy);
  Value *LenCmp = Shared ? IRB.CreateTrunc(Len, Int32Ty) : Len;
  Value *Bad = outOfBounds(IRB, Addr, LenCmp, B.Start, B.Size,
                           !Shared && lastByteSuffices(A, kAllocAlignment));
  if (!A.FixedLen) {
    Bad = IRB.CreateAnd(Bad, IRB.CreateIsNotNull(Len));
    Value *Sz = Shared ? IRB.CreateZExt(B.Size, IntptrTy) : B.Size;
    Bad = IRB.CreateOr(Bad, IRB.CreateICmpUGT(Len, Sz));
  }
  if (B.Valid)
    Bad = IRB.CreateAnd(B.Valid, Bad);

  Value *FailMask = nullptr;
  Value *Any = vote(IRB, Bad, FailMask);
  if (Trap) {
    splitTo(IRB, At, Any, Trap);
    return;
  }

  Value *AddrI = IRB.CreateZExtOrTrunc(Addr, IntptrTy);
  Value *StartI = Shared ? IRB.CreateZExt(B.Start, IntptrTy) : nullptr;
  Value *SizeI = Shared ? IRB.CreateZExt(B.Size, IntptrTy) : nullptr;
  Value *PCI = IRB.CreateIntrinsic(Intrinsic::amdgcn_s_getpc, {});
  if (aborting()) {
    FailSink &S = Fail[Shared][A.IsWrite];
    if (!S.BB)
      S = createFail(*At->getFunction(), Shared, A.IsWrite);
    BasicBlock *Orig = At->getParent();
    splitTo(IRB, At, Any, S.BB);
    S.Addr->addIncoming(AddrI, Orig);
    S.Len->addIncoming(Len, Orig);
    S.Mask->addIncoming(FailMask, Orig);
    S.PC->addIncoming(PCI, Orig);
    if (Shared) {
      S.Start->addIncoming(StartI, Orig);
      S.Size->addIncoming(SizeI, Orig);
    }
    return;
  }

  Instruction *Then =
      SplitBlockAndInsertIfThen(Any, At, /*Unreachable=*/false,
                                MDBuilder(C).createUnlikelyBranchWeights());
  Then->setDebugLoc(Loc);
  setPoint(IRB, Then);
  emitReport(IRB, AddrI, Len, FailMask, PCI, StartI, SizeI, Shared, A.IsWrite);
}

bool DeviceAddressSanitizer::instrumentFunction(Function &F) {
  if (F.isDeclaration() || F.getName().starts_with("__dasan_"))
    return false;
  if (!F.hasFnAttribute(Attribute::SanitizeDeviceAddress) ||
      F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation) ||
      F.hasFnAttribute(Attribute::Naked))
    return false;

  SmallVector<Access, 16> Accesses;
  for (Instruction &I : instructions(F))
    if (!I.hasMetadata(LLVMContext::MD_nosanitize))
      collect(I, Accesses);

  ObjectSizeOffsetVisitor Vis(DL, /*TLI=*/nullptr, C);
  erase_if(Accesses, [&](const Access &A) {
    return A.FixedLen && isSafeAccess(Vis, A.Ptr, A.FixedLen);
  });
  if (Accesses.empty())
    return false;

  if (!NoEntry)
    initializeCallbacks();
  removeASanIncompatibleFnAttributes(F, /*ReadsArgMem=*/false);

  FailSink Fail[2][2] = {};
  BasicBlock *Trap = ClTrapOnError ? createTrap(F) : nullptr;

  BuilderTy IRB(C, InstSimplifyFolder(DL));
  Value *Limit = nullptr;
  auto NeedsLimit = [this](const Access &A) {
    if (isLocalAS(A.Ptr->getType()->getPointerAddressSpace()))
      return !sizedSharedGV(A.Ptr);
    return anySharedGV(A.Ptr) && !sizedSharedGV(A.Ptr);
  };
  if (CheckShared && any_of(Accesses, NeedsLimit) &&
      !F.hasFnAttribute("amdgpu-no-dispatch-ptr"))
    Limit = groupSegmentLimit(IRB, F);

  auto TakeLoc = [&](const Access &A) {
    Loc = A.I->getDebugLoc();
    if (!Loc)
      if (DISubprogram *SP = A.I->getFunction()->getSubprogram())
        Loc = DILocation::get(C, SP->getScopeLine(), 0, SP);
  };

  // For each access were generate a bound and an check on that bound. The
  // bounds are available anywhere inbounds from the access, so we only need one
  // load for each pointer base.
  DenseMap<Value *, Bound> HeapB;
  DenseMap<Value *, Bound> SharedB;
  BasicBlock *Block = nullptr;
  for (const Access &A : Accesses) {
    if (A.Block != Block) {
      Block = A.Block;
      HeapB.clear();
      SharedB.clear();
    }
    Instruction *At = A.I;
    TakeLoc(A);
    unsigned AS = A.Ptr->getType()->getPointerAddressSpace();
    if (CheckShared && (isLocalAS(AS) || anySharedGV(A.Ptr))) {
      // TODO: LDS instrumentation relies entirely on finding the underlying
      //       object. This is sufficient to catch most uses, but a real
      //       solution will require some form of pointer tagging.
      GlobalVariable *GV = sizedSharedGV(A.Ptr);
      if (!GV && !Limit)
        continue;
      Value *Root = A.Ptr;
      if (GV)
        Root = GV;
      auto [It, New] = SharedB.try_emplace(Root);
      if (New) {
        setPoint(IRB, At);
        It->second = sharedBound(IRB, A.Ptr, Limit);
      }
      check(IRB, A, It->second, At, true, Fail, Trap);
    } else if (isHeapAS(AS)) {
      Value *Root = A.Ptr->stripInBoundsOffsets();
      auto [It, New] = HeapB.try_emplace(Root);
      if (New) {
        setPoint(IRB, At);
        It->second = heapBound(IRB, Root);
      }
      check(IRB, A, It->second, At, false, Fail, Trap);
    }
    ++(A.IsWrite ? NumInstrumentedWrites : NumInstrumentedReads);
  }
  return true;
}

void DeviceAddressSanitizer::initializeCallbacks() {
  IRBuilder<> IRB(C);
  AttributeList AL = AttributeList()
                         .addFnAttribute(C, Attribute::NoUnwind)
                         .addFnAttribute(C, Attribute::Cold);
  if (!Recover)
    AL = AL.addFnAttribute(C, Attribute::NoReturn);
  StringRef Suffix = Recover ? kRecoverSuffix : "";
  if (!ClTrapOnError) {
    FunctionType *Ty = FunctionType::get(
        IRB.getVoidTy(),
        {IRB.getPtrTy(), IRB.getInt64Ty(), IRB.getInt64Ty(), IRB.getPtrTy()},
        false);
    FunctionType *SharedTy =
        FunctionType::get(IRB.getVoidTy(),
                          {IRB.getPtrTy(), IRB.getInt64Ty(), IRB.getInt64Ty(),
                           IRB.getInt64Ty(), IRB.getInt64Ty(), IRB.getPtrTy()},
                          false);
    Report[0] =
        M.getOrInsertFunction((Twine(kReportLoadName) + Suffix).str(), Ty, AL);
    Report[1] =
        M.getOrInsertFunction((Twine(kReportStoreName) + Suffix).str(), Ty, AL);
    ReportShared[0] = M.getOrInsertFunction(
        (Twine(kReportSharedLoadName) + Suffix).str(), SharedTy, AL);
    ReportShared[1] = M.getOrInsertFunction(
        (Twine(kReportSharedStoreName) + Suffix).str(), SharedTy, AL);
  }
  auto *Word = IRB.getInt64Ty();
  auto *GV = new GlobalVariable(M, Word, /*isConstant=*/true,
                                GlobalValue::PrivateLinkage,
                                ConstantInt::get(Word, 0), kNoEntryName,
                                nullptr, GlobalValue::NotThreadLocal, GlobalAS);
  GV->setAlignment(Align(kMetadataSize));
  NoEntry = ConstantExpr::getPtrToInt(GV, IntptrTy);
}

void DeviceAddressSanitizer::emitModuleMarker() {
  Type *Ty = Type::getInt8Ty(C);
  auto *GV = new GlobalVariable(M, Ty, /*isConstant=*/true,
                                GlobalValue::LinkOnceODRLinkage,
                                ConstantInt::get(Ty, 1), kModuleMarkerName,
                                nullptr, GlobalValue::NotThreadLocal, GlobalAS);
  GV->setVisibility(GlobalValue::ProtectedVisibility);
  GV->setAlignment(Align(1ULL << kMinSizeLog));
  appendToUsed(M, {GV});
}

static bool isReservedName(StringRef N) {
  return N.starts_with("__dasan_") || N.starts_with("llvm.");
}

bool DeviceAddressSanitizer::shouldPlaceGlobal(const GlobalVariable &G) const {
  if (!isHeapAS(G.getAddressSpace()))
    return false;
  if (G.isDeclarationForLinker() || G.isThreadLocal() || G.hasCommonLinkage() ||
      G.hasAppendingLinkage() || G.hasSection() ||
      G.hasMetadata(LLVMContext::MD_type))
    return false;
  if (isReservedName(G.getName()))
    return false;
  return G.getParent()->getDataLayout().getTypeAllocSize(G.getValueType()) != 0;
}

static StringRef sectionForGlobal(const GlobalVariable &G) {
  const Constant *Init = G.getInitializer();
  if (Init->isNullValue())
    return kGlobalsBssSectionName;
  if (!G.isConstant())
    return kGlobalsDataSectionName;
  if (Init->needsRelocation())
    return kGlobalsRelRoSectionName;
  return kGlobalsSectionName;
}

static uint64_t globalRedzoneSize(uint64_t Size, uint64_t ChunkSize) {
  constexpr uint64_t kMaxGlobalRedzone = 1ULL << 18;
  if (Size <= ChunkSize / 2)
    return ChunkSize - Size;
  uint64_t RZ = std::clamp((Size / ChunkSize / 4) * ChunkSize, ChunkSize,
                           kMaxGlobalRedzone);
  if (Size % ChunkSize)
    RZ += ChunkSize - (Size % ChunkSize);
  return RZ;
}

bool DeviceAddressSanitizer::placeGlobals() {
  if (!ClInstrumentGlobals)
    return false;

  // Global placement works by loading the executable itself inside of the
  // tracked region. This requires that globals to be instrumented are aligned
  // at least to the minimum chunk size. A single global can span many chunks,
  // but the signed offset points back to the original offset.
  constexpr uint64_t ChunkSize = 1ULL << kMinSizeLog;
  const Align ChunkAlign(ChunkSize);
  SmallVector<GlobalVariable *, 8> GlobalsToPlace;
  SmallVector<GlobalVariable *, 8> GlobalsToPin;
  for (GlobalVariable &G : M.globals()) {
    if (!shouldPlaceGlobal(G))
      continue;
    if (G.getName().starts_with("__") || G.getName().starts_with(".") ||
        G.hasComdat() || !G.hasExactDefinition())
      GlobalsToPin.push_back(&G);
    else
      GlobalsToPlace.push_back(&G);
  }

  // We size globals based off of their symbol `st_size`. Private symbols do not
  // get an entry in the symbol table so we convert them to internal symbols.
  // These are globals we do not own the layout for so we only align them.
  for (GlobalVariable *G : GlobalsToPin) {
    if (G->hasPrivateLinkage())
      G->setLinkage(GlobalValue::InternalLinkage);
    G->setAlignment(std::max(G->getAlign().valueOrOne(), ChunkAlign));
    G->setSection(sectionForGlobal(*G));
  }

  // Add additional redzones to globals to detect buffer overruns without
  // stepping into the next valid object.
  Type *Int8Ty = Type::getInt8Ty(C);
  SmallVector<GlobalValue *, 16> Kept;
  for (GlobalVariable *G : GlobalsToPlace) {
    Type *Ty = G->getValueType();
    const unsigned AS = G->getAddressSpace();
    const uint64_t Size = DL.getTypeAllocSize(Ty).getFixedValue();
    const Align Alignment = std::max(G->getAlign().valueOrOne(), ChunkAlign);
    const uint64_t Total =
        alignTo(Size + globalRedzoneSize(Size, ChunkSize), Alignment);
    const uint64_t GuardAt = alignTo(Size, ChunkSize);
    auto *PadTy = ArrayType::get(Int8Ty, Total - Size);
    auto *WholeTy = StructType::get(C, {Ty, PadTy});

    // Generate a new global variable with sufficient redzone padding.
    auto *Whole = new GlobalVariable(
        M, WholeTy, G->isConstant(), GlobalValue::PrivateLinkage,
        ConstantStruct::get(
            WholeTy, {G->getInitializer(), Constant::getNullValue(PadTy)}),
        G->getName() + ".dasan", G, GlobalValue::NotThreadLocal, AS);
    Whole->setAlignment(Alignment);
    Whole->setSection(sectionForGlobal(*G));
    Whole->setExternallyInitialized(G->isExternallyInitialized());
    Whole->copyMetadata(G, 0);

    // Replace the original global with an alias into the created fat global.
    auto *Obj = GlobalAlias::create(
        Ty, AS,
        G->hasPrivateLinkage() ? GlobalValue::InternalLinkage : G->getLinkage(),
        "", Whole, &M);
    Obj->setVisibility(G->getVisibility());
    Obj->setDSOLocal(G->isDSOLocal());
    G->replaceAllUsesWith(Obj);
    Obj->takeName(G);
    G->eraseFromParent();
    Kept.push_back(Whole);
    if (Total > GuardAt)
      Kept.push_back(GlobalAlias::create(
          ArrayType::get(Int8Ty, Total - GuardAt), AS,
          GlobalValue::InternalLinkage, kGlobalGuardName,
          ConstantExpr::getGetElementPtr(Int8Ty, Whole, intptr(GuardAt)), &M));
  }
  appendToCompilerUsed(M, Kept);
  return !GlobalsToPlace.empty() || !GlobalsToPin.empty();
}

bool DeviceAddressSanitizer::run() {
  bool Changed = false;
  for (Function &F : M)
    Changed |= instrumentFunction(F);
  Changed |= placeGlobals();
  if (Changed && !M.getNamedValue(kModuleMarkerName))
    emitModuleMarker();
  return Changed;
}

PreservedAnalyses DeviceAddressSanitizerPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  if (checkIfAlreadyInstrumented(M, "nosanitize_device_address"))
    return PreservedAnalyses::all();
  const Triple &T = M.getTargetTriple();
  if (!T.isAMDGPU() || M.getDataLayout().getPointerSizeInBits() != kWordBits)
    return PreservedAnalyses::all();

  if (!DeviceAddressSanitizer(M, Options).run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
