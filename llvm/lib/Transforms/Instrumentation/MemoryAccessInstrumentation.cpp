//===- MemoryAccessInstrumentation.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryAccessInstrumentation.h"
#include "llvm/ADT/bit.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::memaccess;

int memaccess::getAccessSizeIndex(Type *Ty, const DataLayout &DL) {
  assert(Ty->isSized());
  if (Ty->isScalableTy())
    return -1;
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(Ty);
  if (TypeSize != 8 && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 &&
      TypeSize != 128)
    return -1;
  unsigned Idx = llvm::countr_zero(TypeSize / 8);
  assert(Idx < kNumAccessSizes);
  return static_cast<int>(Idx);
}

bool memaccess::isVtableAccess(const Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

bool memaccess::isAtomicMemoryAccess(const Instruction *I) {
  auto SSID = getAtomicSyncScopeID(I);
  if (!SSID)
    return false;
  if (isa<LoadInst>(I) || isa<StoreInst>(I))
    return *SSID != SyncScope::SingleThread;
  return true;
}

bool memaccess::addressSpaceMayRace(const Triple &T, unsigned AS,
                                    AddrSpaceRacePolicy Policy) {
  if (Policy == AddrSpaceRacePolicy::FlatOnly)
    return AS == 0;

  if (T.isAMDGPU()) {
    switch (AS) {
    case AMDGPUAS::FLAT_ADDRESS:
    case AMDGPUAS::GLOBAL_ADDRESS:
    case AMDGPUAS::REGION_ADDRESS:
    case AMDGPUAS::LOCAL_ADDRESS:
      return true;
    default:
      // Private, constant, and buffer fat/strided pointers (AS 7/8/9) are
      // skipped until the probe ABI can form a 64-bit watchpoint key.
      return false;
    }
  }
  if (T.isNVPTX()) {
    switch (AS) {
    case NVPTXAS::ADDRESS_SPACE_GENERIC:
    case NVPTXAS::ADDRESS_SPACE_GLOBAL:
    case NVPTXAS::ADDRESS_SPACE_SHARED:
    case NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER:
      return true;
    default:
      return false;
    }
  }
  return AS == 0;
}

bool memaccess::shouldInstrumentAddress(const Module *M, Value *Addr,
                                        AddrSpaceRacePolicy Policy) {
  // Peel GEPs/bitcasts, but not for the address-space check: the space is a
  // property of the access, not of the underlying object.
  Value *PeeledAddr = Addr->stripInBoundsOffsets();
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(PeeledAddr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      auto OF = M->getTargetTriple().getObjectFormat();
      if (SectionName.ends_with(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }
  }

  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  return addressSpaceMayRace(M->getTargetTriple(),
                             PtrTy->getPointerAddressSpace(), Policy);
}

Value *memaccess::genericCallbackPtr(IRBuilderBase &IRB, Value *Addr) {
  return IRB.CreateAddrSpaceCast(Addr, IRB.getPtrTy());
}
