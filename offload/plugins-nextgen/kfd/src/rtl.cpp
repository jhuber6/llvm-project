//===----RTLs/kfd/src/rtl.cpp - Target RTL for Linux KFD --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>

#include "Shared/Debug.h"
#include "Shared/Environment.h"
#include "Shared/EnvironmentVar.h"

#include "ErrorReporting.h"
#include "GlobalHandler.h"
#include "OffloadAPI.h"
#include "OffloadError.h"
#include "PluginInterface.h"
#include "RPC.h"
#include "omptarget.h"

#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"

#include "libkfd/libkfd.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

using namespace error;

/// Map a libkfd error and its carried errno onto the closest offload error.
template <typename T>
static Error toErr(const std::expected<T, kfd::Error> &E, const char *Ctx) {
  ErrorCode Code = ErrorCode::UNKNOWN;
  switch (E.error().code) {
  case ENOENT:
    Code = ErrorCode::NOT_FOUND;
    break;
  case ENOMEM:
    Code = ErrorCode::OUT_OF_RESOURCES;
    break;
  case EINVAL:
    Code = ErrorCode::INVALID_ARGUMENT;
    break;
  default:
    break;
  }
  return Plugin::error(Code, "%s: %s", Ctx, kfd::strerror(E.error()));
}

struct KFDDeviceTy;
struct KFDPluginTy;
struct KFDKernelTy;

/// Pooled kernarg allocator shared by every launch path.
struct KFDArgsManagerTy : public DeviceAllocatorTy {
  explicit KFDArgsManagerTy(kfd::Device *Dev) : KFDDevice(Dev) {}

  Error init() {
    Manager =
        std::make_unique<MemoryManagerTy>(*this, /*Threshold=*/size_t(1) << 30);
    return Plugin::success();
  }

  void deinit() {
    Manager.reset();
    std::lock_guard<std::mutex> Lock(Mtx);
    Buffers.clear();
  }

  Expected<kfd::Buffer *> allocate(size_t Size) {
    auto PtrOrErr = Manager->allocate(Size, nullptr, /*Alignment=*/0);
    if (!PtrOrErr)
      return PtrOrErr.takeError();
    std::lock_guard<std::mutex> Lock(Mtx);
    auto It = Buffers.find(*PtrOrErr);
    if (It == Buffers.end())
      return Plugin::error(ErrorCode::UNKNOWN, "pooled kernarg buffer missing");
    return &It->second;
  }

  void deallocate(void *Ptr) {
    if (Error Err = Manager->free(Ptr))
      consumeError(std::move(Err));
  }

private:
  // KFD buffers are page-aligned, so the alignment hint is met implicitly.
  Expected<void *> allocate(size_t Size, void *, TargetAllocTy,
                            size_t /*Alignment*/) override {
    auto BufOrErr = kfd::Buffer::allocate(
        *KFDDevice, Size, kfd::MemType::GTT,
        kfd::MemFlags::WRITABLE | kfd::MemFlags::COHERENT |
            kfd::MemFlags::HOST_ACCESS | kfd::MemFlags::UNCACHED);
    if (!BufOrErr)
      return toErr(BufOrErr, "failed to allocate kernarg buffer");
    kfd::Buffer Buf = std::move(*BufOrErr);
    if (auto Res = Buf.map(*KFDDevice); !Res)
      return toErr(Res, "failed to map kernarg buffer");
    void *Ptr = Buf.data();
    std::lock_guard<std::mutex> Lock(Mtx);
    Buffers.emplace(Ptr, std::move(Buf));
    return Ptr;
  }

  Error free(void *Ptr, TargetAllocTy) override {
    std::lock_guard<std::mutex> Lock(Mtx);
    Buffers.erase(Ptr);
    return Plugin::success();
  }

  kfd::Device *KFDDevice;
  std::unique_ptr<MemoryManagerTy> Manager;
  std::mutex Mtx;
  // std::map keeps node references stable as other threads insert.
  std::map<void *, kfd::Buffer> Buffers;
};

/// Thread-safe free-list of completion signals, one borrowed per in-flight op.
struct KFDSignalPoolTy {
  void init(kfd::Context *Context) { Ctx = Context; }
  void deinit() {
    std::lock_guard<std::mutex> Lock(Mtx);
    Free.clear();
  }

  /// Borrow a signal, creating one only when the free-list is empty.
  Expected<kfd::Signal> acquire() {
    {
      std::lock_guard<std::mutex> Lock(Mtx);
      if (!Free.empty()) {
        kfd::Signal Sig = std::move(Free.back());
        Free.pop_back();
        return Sig;
      }
    }
    auto SignalOrErr = kfd::Signal::create(*Ctx);
    if (!SignalOrErr)
      return toErr(SignalOrErr, "failed to create completion signal");
    return std::move(*SignalOrErr);
  }

  void release(kfd::Signal &&Sig) {
    std::lock_guard<std::mutex> Lock(Mtx);
    Free.emplace_back(std::move(Sig));
  }

private:
  kfd::Context *Ctx = nullptr;
  std::mutex Mtx;
  llvm::SmallVector<kfd::Signal> Free;
};

/// A PM4 hardware queue and the lock serializing the software streams
/// multiplexed onto it. Hardware queues are the scarce, oversubscribed resource.
struct KFDHwQueueTy {
  explicit KFDHwQueueTy(kfd::ComputeQueue &&Q) : Queue(std::move(Q)) {}

  kfd::ComputeQueue Queue;
  // Held across an op and its completion signal to keep them contiguous.
  std::mutex Mtx;
  // Streams currently assigned here; steers least-used assignment.
  unsigned Users = 0;
};

/// Thread-safe pool of hardware queues, grown lazily to a cap
/// (LIBOMPTARGET_KFD_NUM_QUEUES) and shared by the software streams thereafter.
struct KFDQueuePoolTy {
  Error init(kfd::Device *Dev, unsigned Max) {
    Device = Dev;
    MaxQueues = Max ? Max : 1;
    return Plugin::success();
  }
  void deinit() {
    std::lock_guard<std::mutex> Lock(Mtx);
    Queues.clear();
  }

  /// Assign the least-used queue, growing the pool up to its cap first.
  Expected<KFDHwQueueTy *> assign() {
    std::lock_guard<std::mutex> Lock(Mtx);
    KFDHwQueueTy *Best = nullptr;
    for (auto &Q : Queues)
      if (!Best || Q->Users < Best->Users)
        Best = Q.get();
    if ((!Best || Best->Users > 0) && Queues.size() < MaxQueues) {
      auto QueueOrErr = kfd::ComputeQueue::create(*Device);
      if (!QueueOrErr)
        return toErr(QueueOrErr, "failed to create hardware queue");
      Queues.emplace_back(
          std::make_unique<KFDHwQueueTy>(std::move(*QueueOrErr)));
      Best = Queues.back().get();
    }
    ++Best->Users;
    return Best;
  }

  void release(KFDHwQueueTy *HW) {
    if (!HW)
      return;
    std::lock_guard<std::mutex> Lock(Mtx);
    if (HW->Users)
      --HW->Users;
  }

private:
  kfd::Device *Device = nullptr;
  unsigned MaxQueues = 1;
  std::mutex Mtx;
  // unique_ptr keeps queue addresses stable as the pool grows.
  llvm::SmallVector<std::unique_ptr<KFDHwQueueTy>> Queues;
};

/// Max ops a stream keeps in flight before submission blocks on the oldest.
static constexpr unsigned KFDStreamMaxInFlight = 64;

/// Ordered async stream (HSA/CUDA model): a software FIFO multiplexed onto a
/// hardware queue from the device's bounded pool. Ops submit in order with a
/// borrowed completion signal; the host blocks only on backpressure or at
/// synchronize/query.
///
/// Many streams share one in-order ring, so a ring must only wait on a signal
/// whose producer packet is already enqueued -- else the producer could sit
/// behind the waiter and deadlock (upheld by waitEventImpl and the host-fn
/// gates).
struct KFDStreamTy {
  using HostFn = std::function<Error()>;

  KFDStreamTy() = default;

  KFDStreamTy(const KFDStreamTy &) = delete;
  KFDStreamTy &operator=(const KFDStreamTy &) = delete;

  /// Retire any ops left outstanding (only reached on an errored teardown),
  /// returning their kernargs and signals rather than leaking.
  ~KFDStreamTy() {
    std::lock_guard<std::mutex> Lock(Mtx);
    for (PendingOp &Op : InFlight)
      finishOpLocked(Op);
    InFlight.clear();
  }

  /// Attach the host RPC server; each in-flight dispatch keeps it polling.
  void setRPCServer(RPCServerTy *Server) { RPC = Server; }

  /// Attach the device kernarg manager that owns the dispatched kernargs.
  void setArgsManager(KFDArgsManagerTy *Manager) { Args = Manager; }

  /// Attach the device signal pool completion signals are borrowed from.
  void setSignalPool(KFDSignalPoolTy *Pool) { Signals = Pool; }

  /// Bind the hardware queue borrowed for this acquire/recycle cycle.
  void setHwQueue(KFDHwQueueTy *Queue) { HW = Queue; }
  KFDHwQueueTy *hwQueue() const { return HW; }

  /// Submit a kernel; \p KernargPtr is manager-owned and retired with the op.
  Error enqueueDispatch(const kfd::Kernel &Kernel,
                        const kfd::DispatchConfig &Config, void *KernargPtr,
                        const kfd::Buffer &KernargBuf) {
    return enqueueOp(
        [&](kfd::ComputeQueue &Q) -> Error {
          if (auto Res = Q.dispatch(Kernel, Config, KernargBuf); !Res)
            return toErr(Res, "failed to dispatch kernel");
          return Plugin::success();
        },
        KernargPtr, /*RPCUser=*/true);
  }

  /// Stream-ordered copy via a CP DMA. Both operands must be GPU-reachable.
  Error enqueueCopy(void *Dst, const void *Src, uint64_t Bytes) {
    if (!Bytes)
      return Plugin::success();
    return enqueueOp(
        [=](kfd::ComputeQueue &Q) -> Error {
          // CP DMA byte_count is 32-bit; chunk large copies.
          for (uint64_t Off = 0; Off < Bytes;) {
            uint32_t N = uint32_t(std::min<uint64_t>(Bytes - Off, 1ull << 30));
            if (auto Res = Q.dma_copy(static_cast<char *>(Dst) + Off,
                                      static_cast<const char *>(Src) + Off, N);
                !Res)
              return toErr(Res, "failed to enqueue copy");
            Off += N;
          }
          return Plugin::success();
        },
        nullptr, /*RPCUser=*/false);
  }

  /// Stream-ordered fill of a 32-bit word via a CP DMA (\p Bytes multiple of 4).
  Error enqueueFill(void *Dst, uint32_t Word, uint64_t Bytes) {
    if (!Bytes)
      return Plugin::success();
    return enqueueOp(
        [=](kfd::ComputeQueue &Q) -> Error {
          for (uint64_t Off = 0; Off < Bytes;) {
            uint32_t N = uint32_t(std::min<uint64_t>(Bytes - Off, 1ull << 30));
            if (auto Res = Q.dma_fill(static_cast<char *>(Dst) + Off, Word, N);
                !Res)
              return toErr(Res, "failed to enqueue fill");
            Off += N;
          }
          return Plugin::success();
        },
        nullptr, /*RPCUser=*/false);
  }

  /// Record an event: decrement \p EventSig on the GPU after prior stream work.
  Error enqueueMarker(kfd::Signal &EventSig) {
    return enqueueOp(
        [&](kfd::ComputeQueue &Q) -> Error {
          if (auto Res = Q.signal(EventSig); !Res)
            return toErr(Res, "failed to enqueue event marker");
          return Plugin::success();
        },
        nullptr, /*RPCUser=*/false);
  }

  /// Stall the queue on the GPU until \p EventSig reaches 0.
  Error enqueueWaitOn(kfd::Signal &EventSig) {
    return enqueueOp(
        [&](kfd::ComputeQueue &Q) -> Error {
          if (auto Res = Q.wait(EventSig, kfd::Condition::EQ, 0); !Res)
            return toErr(Res, "failed to enqueue event wait");
          return Plugin::success();
        },
        nullptr, /*RPCUser=*/false);
  }

  /// Gate a host callback: signal \p Before after prior work, then wait on
  /// \p After. The owned refs keep both signals alive until the op retires, so
  /// their packets can never outlive them.
  Error enqueueGate(kfd::Signal &Before, kfd::Signal &After,
                    std::shared_ptr<kfd::Signal> BeforeOwned,
                    std::shared_ptr<kfd::Signal> AfterOwned) {
    return enqueueOp(
        [&](kfd::ComputeQueue &Q) -> Error {
          if (auto Res = Q.signal(Before); !Res)
            return toErr(Res, "failed to enqueue host-fn gate-in");
          if (auto Res = Q.wait(After, kfd::Condition::EQ, 0); !Res)
            return toErr(Res, "failed to enqueue host-fn gate-out");
          return Plugin::success();
        },
        nullptr, /*RPCUser=*/false,
        {std::move(BeforeOwned), std::move(AfterOwned)});
  }

  /// Drain the stream, then run \p Fn inline (staged pageable transfers). Busy
  /// keeps idle() false while it runs.
  Error enqueue(HostFn Fn) {
    {
      std::lock_guard<std::mutex> Lock(Mtx);
      if (Error Err = drainLocked())
        return Err;
      ++Busy;
    }
    Error Err = Fn();
    std::lock_guard<std::mutex> Lock(Mtx);
    --Busy;
    return Err;
  }

  /// Block until every submitted op has retired.
  Error synchronize() {
    std::lock_guard<std::mutex> Lock(Mtx);
    return drainLocked();
  }

  bool idle() {
    std::lock_guard<std::mutex> Lock(Mtx);
    pollLocked();
    return InFlight.empty() && Busy == 0;
  }

private:
  // One in-flight op: completion signal, kernarg to free, RPC-user flag, and
  // any signals kept alive until retire (the host-fn gates).
  struct PendingOp {
    kfd::Signal Signal;
    void *Kernarg;
    bool RPCUser;
    llvm::SmallVector<std::shared_ptr<kfd::Signal>, 2> Owned;
  };

  /// Emit packets onto the shared queue with a trailing completion signal and
  /// track the op. Blocks only on backpressure.
  template <typename EmitFn>
  Error enqueueOp(EmitFn Emit, void *Kernarg, bool RPCUser,
                  llvm::SmallVector<std::shared_ptr<kfd::Signal>, 2> Owned =
                      {}) {
    std::lock_guard<std::mutex> Lock(Mtx);
    if (InFlight.size() >= KFDStreamMaxInFlight)
      if (Error E = retireOldestLocked())
        return E;
    auto SignalOrErr = Signals->acquire();
    if (!SignalOrErr)
      return SignalOrErr.takeError();
    kfd::Signal Sig = std::move(*SignalOrErr);
    if (auto Res = Sig.reset(/*value=*/1); !Res)
      return toErr(Res, "failed to reset completion signal");
    {
      // Keep the op and its completion signal contiguous on the shared ring.
      std::lock_guard<std::mutex> QLock(HW->Mtx);
      if (Error E = Emit(HW->Queue))
        return E;
      if (auto Res = HW->Queue.signal(Sig); !Res)
        return toErr(Res, "failed to submit completion signal");
    }
    bool Rpc = RPCUser && RPC;
    if (Rpc)
      RPC->Thread->notify();
    InFlight.push_back(
        PendingOp{std::move(Sig), Kernarg, Rpc, std::move(Owned)});
    return Plugin::success();
  }

  /// Return an op's resources once its signal has fired. Caller holds Mtx.
  void finishOpLocked(PendingOp &Op) {
    if (Op.Kernarg)
      Args->deallocate(Op.Kernarg);
    if (Op.RPCUser && RPC)
      RPC->Thread->finish();
    Signals->release(std::move(Op.Signal));
    // Op.Owned signals, if any, drop here -- their gate packets have retired.
  }

  /// Block on the oldest op, then retire it. Caller holds Mtx. On wait failure
  /// the op stays outstanding so its kernarg is not reused under a live wave.
  Error retireOldestLocked() {
    PendingOp &Op = InFlight.front();
    if (auto Res = Op.Signal.wait(kfd::Condition::EQ, 0, UINT64_MAX); !Res)
      return toErr(Res, "failed to wait on kernel completion");
    finishOpLocked(Op);
    InFlight.pop_front();
    return Plugin::success();
  }

  /// Retire every op whose signal has fired; stop at the first still pending.
  void pollLocked() {
    while (!InFlight.empty()) {
      PendingOp &Op = InFlight.front();
      if (Op.Signal.value() != 0)
        break;
      finishOpLocked(Op);
      InFlight.pop_front();
    }
  }

  Error drainLocked() {
    while (!InFlight.empty())
      if (Error Err = retireOldestLocked())
        return Err;
    return Plugin::success();
  }

  std::mutex Mtx;
  // Outstanding ops in submission order; bounded to KFDStreamMaxInFlight.
  std::deque<PendingOp> InFlight;
  // Inline ops currently running (see enqueue), to keep idle() honest.
  unsigned Busy = 0;

  // Device-owned resources outliving the stream (RPC server may be null).
  RPCServerTy *RPC = nullptr;
  KFDArgsManagerTy *Args = nullptr;
  KFDSignalPoolTy *Signals = nullptr;
  KFDHwQueueTy *HW = nullptr;
};

/// CUDA-style event: a GPU signal armed to 1 on record and decremented by the
/// GPU. Created complete (0) so an un-recorded event never stalls a waiter.
struct KFDEventTy {
  explicit KFDEventTy(kfd::Signal &&Sig) : Signal(std::move(Sig)) {}
  kfd::Signal Signal;
};

/// Device image wrapping a libkfd Executable loaded into GPU memory.
struct KFDDeviceImageTy : public DeviceImageTy {
  KFDDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                   std::unique_ptr<MemoryBuffer> &&TgtImage)
      : DeviceImageTy(ImageId, Device, std::move(TgtImage)) {}

  void setExecutable(kfd::Executable &&Exe) {
    Executable.emplace(std::move(Exe));
  }
  kfd::Executable &getExecutable() { return *Executable; }

  /// Parse code-object metadata: per-kernel arg offsets and max flat WG size.
  Error parseMetadata() {
    return llvm::offloading::amdgpu::getAMDGPUMetaDataFromImage(
        getMemoryBuffer(), KernelInfoMap, ELFABIVersion);
  }

  Expected<const llvm::offloading::amdgpu::AMDGPUKernelMetaData *>
  getKernelInfo(StringRef Name) const {
    auto It = KernelInfoMap.find(Name);
    if (It == KernelInfoMap.end())
      return Plugin::error(ErrorCode::INVALID_BINARY,
                           "could not find metadata for kernel '%s'",
                           Name.str().c_str());
    return &It->second;
  }

private:
  std::optional<kfd::Executable> Executable;
  llvm::StringMap<llvm::offloading::amdgpu::AMDGPUKernelMetaData> KernelInfoMap;
  uint16_t ELFABIVersion = 0;
};

/// Kernel functionality backed by a libkfd Kernel handle.
struct KFDKernelTy : public GenericKernelTy {
  KFDKernelTy(const char *Name) : GenericKernelTy(Name) {}

  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    auto &KImage = static_cast<KFDDeviceImageTy &>(Image);
    // AMDGPU kernel descriptors carry a ".kd" suffix in the symbol table.
    std::string KdName = std::string(getName()) + ".kd";
    auto KernelOrErr = KImage.getExecutable().kernel(KdName);
    if (!KernelOrErr)
      return toErr(KernelOrErr, "failed to look up kernel descriptor");
    Kernel.emplace(std::move(*KernelOrErr));

    // Kernarg layout and max flat work-group size come from the code-object
    // metadata, not the descriptor.
    auto InfoOrErr = KImage.getKernelInfo(getName());
    if (!InfoOrErr)
      return InfoOrErr.takeError();
    const auto &Info = **InfoOrErr;
    ArgMDs.assign(Info.ArgMDs.begin(), Info.ArgMDs.end());

    constexpr uint32_t Invalid =
        llvm::offloading::amdgpu::AMDGPUKernelMetaData::KInvalidValue;
    uint32_t MaxFlat = Info.MaxFlatWorkgroupSize;
    MaxFlatWorkgroupSize = (MaxFlat && MaxFlat != Invalid) ? MaxFlat : 1024;
    // Clamp (not clobber) against the flat-workgroup ceiling; init() already
    // applied the launch bounds and device thread limit.
    MaxNumThreads = std::min(MaxNumThreads, MaxFlatWorkgroupSize);
    PreferredNumThreads = std::min(PreferredNumThreads, MaxNumThreads);
    return Plugin::success();
  }

  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], uint32_t DynBlockMemSize,
                   KernelArgsTy &KernelArgs, KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  Expected<uint64_t> maxGroupSize(GenericDeviceTy &Device,
                                  uint64_t DynamicMemSize) const override {
    return MaxNumThreads;
  }

  /// Match the HSA plugin's LIBOMPTARGET_INFO launch trace.
  Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                               KernelArgsTy &KernelArgs, uint32_t NumThreads[3],
                               uint32_t NumBlocks[3]) const override {
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, GenericDevice.getDeviceId(),
         "#Args: %d Teams x Thrds: %4ux%4u (MaxFlatWorkGroupSize: %u)\n",
         KernelArgs.NumArgs, NumBlocks[0] * NumBlocks[1] * NumBlocks[2],
         NumThreads[0] * NumThreads[1] * NumThreads[2], MaxFlatWorkgroupSize);
    return Plugin::success();
  }

private:
  std::optional<kfd::Kernel> Kernel;
  // Per-argument {offset, size} from the code object's ".args" metadata.
  llvm::SmallVector<std::pair<uint32_t, uint32_t>, 8> ArgMDs;
  uint32_t MaxFlatWorkgroupSize = 1024;
};

/// Heap context for a stream-ordered host callback, freed by the worker.
struct KFDHostFnCtx {
  std::function<Error()> Work;
  // Gate-in (waited on) and gate-out (released to unblock later GPU work).
  std::shared_ptr<kfd::Signal> Before;
  std::shared_ptr<kfd::Signal> After;
  KFDDeviceTy *Device;
};

/// Runs on the libkfd watcher when the gate-in signal fires; only hands the
/// context to the Tier-1 worker so user code never runs on the watcher.
/// Defined after KFDDeviceTy.
static void kfdHostFnTrampoline(void *P);

/// Device functionality for a single GPU node exposed by KFD.
struct KFDDeviceTy : public GenericDeviceTy {
  KFDDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
              kfd::Device &Dev)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices,
                        llvm::omp::AMDGPUGridValues32),
        KFDDevice(&Dev), ArgsManager(&Dev) {}

  kfd::Context &getKFDContext();

  /// KFD topology id of this device, used to match asynchronous fault reports.
  uint32_t getGpuId() const { return KFDDevice->gpu_id(); }

  /// Create the compute queue (load + dispatch) and the SDMA queue (copies).
  Error initImpl(GenericPluginTy &Plugin) override {
    auto QueueOrErr = kfd::ComputeQueue::create(*KFDDevice);
    if (!QueueOrErr)
      return toErr(QueueOrErr, "failed to create compute queue");
    Queue.emplace(std::move(*QueueOrErr));

    auto SDMAOrErr = kfd::SDMAQueue::create(*KFDDevice);
    if (!SDMAOrErr)
      return toErr(SDMAOrErr, "failed to create SDMA queue");
    SDMA.emplace(std::move(*SDMAOrErr));

    // One signal per staging slot, so a chunk's SDMA overlaps the next copy.
    for (unsigned I = 0; I < NumStaging; ++I) {
      auto SignalOrErr = kfd::Signal::create(getKFDContext());
      if (!SignalOrErr)
        return toErr(SignalOrErr, "failed to create transfer signal");
      XferSignal[I].emplace(std::move(*SignalOrErr));
    }

    auto ComputeSignalOrErr = kfd::Signal::create(getKFDContext());
    if (!ComputeSignalOrErr)
      return toErr(ComputeSignalOrErr, "failed to create compute signal");
    ComputeSignal.emplace(std::move(*ComputeSignalOrErr));

    const auto &Props = KFDDevice->properties();
    uint32_t Lds = Props.lds_size_in_kb << 10;
    MaxBlockSharedMemSize = Lds ? Lds : (64u << 10);

    // Use the device's actual wavefront size for launch sizing and RPC lanes.
    if (Props.wave_front_size)
      GridValues.GV_Warp_Size = Props.wave_front_size;

    // PM4 workgroup counts are full 32-bit fields, so teams are bounded only by
    // total grid extent, not the base default of 1<<16.
    GridValues.GV_Max_Teams =
        UINT32_MAX /
        (GridValues.GV_Max_WG_Size ? GridValues.GV_Max_WG_Size : 1);

    // Compute units and resident waves size the default team and RPC-port count.
    uint32_t ComputeUnits =
        Props.simd_per_cu ? Props.simd_count / Props.simd_per_cu : 0;
    if (!ComputeUnits)
      ComputeUnits = 1;
    uint32_t WavesPerCU = Props.max_waves_per_simd
                              ? Props.max_waves_per_simd *
                                    (Props.simd_per_cu ? Props.simd_per_cu : 1)
                              : 0;
    if (!WavesPerCU)
      WavesPerCU = 1;
    HardwareParallelism = uint64_t(ComputeUnits) * WavesPerCU;
    GridValues.GV_Default_Num_Teams = ComputeUnits * 4;

    if (Error Err = ArgsManager.init())
      return Err;
    SignalPool.init(&getKFDContext());

    // Default 4 matches HIP's GPU_MAX_HW_QUEUES; more oversubscribes the runlist.
    UInt32Envar NumQueues("LIBOMPTARGET_KFD_NUM_QUEUES", 4);
    if (Error Err = QueuePool.init(KFDDevice, NumQueues.get()))
      return Err;

    // Tier-1 host-callback worker: runs user code off the libkfd watcher. One
    // worker matches CUDA's per-context host-function serialism.
    CallbackWorker = std::thread(&KFDDeviceTy::callbackLoop, this);
    return Plugin::success();
  }

  /// Total resident wavefronts: compute units times waves per CU.
  uint64_t getHardwareParallelism() const override {
    return HardwareParallelism;
  }

  /// Device host services (printf, malloc, assert) use the HSA RPC ABI.
  bool shouldSetupRPCServer() const override { return true; }

  /// One port per resident wave; the GPU has no inter-wave forward progress.
  uint64_t requestedRPCPortCount() const override {
    return getHardwareParallelism();
  }

  /// Global ctors/dtors run via the compiler-emitted device.init/fini kernels.
  Error callGlobalConstructors(GenericPluginTy &Plugin,
                               DeviceImageTy &Image) override {
    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
  }
  Error callGlobalDestructors(GenericPluginTy &Plugin,
                              DeviceImageTy &Image) override {
    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
  }

  Error deinitImpl() override {
    // Stop the worker first; it drains queued callbacks (releasing gate-out
    // signals) before exiting, so no stream is left wedged.
    {
      std::lock_guard<std::mutex> Lock(CallbackMtx);
      CallbackStop = true;
      CallbackCV.notify_all();
    }
    if (CallbackWorker.joinable())
      CallbackWorker.join();
    // The runtime synchronizes streams before teardown, so every gate has run.
    {
      std::lock_guard<std::mutex> Lock(StreamPoolMtx);
      for (auto *Stream : StreamPool)
        delete Stream;
      StreamPool.clear();
    }
    // Order: queues down after streams (waves preempted), before the kernargs
    // and signals they touch are freed.
    QueuePool.deinit();
    ArgsManager.deinit();
    SignalPool.deinit();
    std::lock_guard<std::mutex> Lock(AllocMutex);
    PinnedLocks.clear();
    PinnedPageRefs.clear();
    PinnedRuns.clear();
    Allocations.clear();
    for (unsigned I = 0; I < NumStaging; ++I) {
      Staging[I].reset();
      XferSignal[I].reset();
    }
    ComputeSignal.reset();
    SDMA.reset();
    Queue.reset();
    return Plugin::success();
  }

  Error setContext() override { return Plugin::success(); }

  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    KFDKernelTy *Kernel = Plugin.allocate<KFDKernelTy>();
    if (!Kernel)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "failed to allocate memory for kernel");
    new (Kernel) KFDKernelTy(Name);
    return *Kernel;
  }

  Expected<DeviceImageTy *>
  loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                 int32_t ImageId) override {
    KFDDeviceImageTy *Image = Plugin.allocate<KFDDeviceImageTy>();
    new (Image) KFDDeviceImageTy(ImageId, *this, std::move(TgtImage));

    std::span<const std::byte> Bytes(
        reinterpret_cast<const std::byte *>(Image->getStart()),
        Image->getSize());
    auto ExeOrErr = kfd::Executable::load(*KFDDevice, Bytes, *Queue);
    if (!ExeOrErr)
      return toErr(ExeOrErr, "failed to load executable image");
    Image->setExecutable(std::move(*ExeOrErr));
    if (auto Err = Image->parseMetadata())
      return std::move(Err);
    return Image;
  }

  Error unloadBinaryImpl(DeviceImageTy *Image) override {
    Plugin.free(Image);
    return Plugin::success();
  }

  /// Device allocations are device-local VRAM; host/shared allocations are GTT
  /// system memory the CPU touches directly. KFD backs every buffer with at
  /// least a page, so alignment is met implicitly.
  Expected<void *> allocate(size_t Size, void *, TargetAllocTy Kind,
                            size_t /*Alignment*/) override {
    if (Size == 0)
      return nullptr;

    bool HostVisible = Kind == TARGET_ALLOC_HOST || Kind == TARGET_ALLOC_SHARED;
    kfd::MemType Type = HostVisible ? kfd::MemType::GTT : kfd::MemType::VRAM;
    kfd::MemFlags Flags = kfd::MemFlags::WRITABLE;
    if (HostVisible)
      Flags = Flags | kfd::MemFlags::HOST_ACCESS | kfd::MemFlags::COHERENT;

    auto BufOrErr = kfd::Buffer::allocate(*KFDDevice, Size, Type, Flags);
    if (!BufOrErr)
      return toErr(BufOrErr, "failed to allocate device memory");

    kfd::Buffer Buffer = std::move(*BufOrErr);
    if (auto Res = Buffer.map(*KFDDevice); !Res)
      return toErr(Res, "failed to map device memory");

    void *Ptr = Buffer.data();
    std::lock_guard<std::mutex> Lock(AllocMutex);
    Allocations.emplace(Ptr, std::move(Buffer));
    // Host/shared allocations are GPU-mapped; record them so SDMA can reach
    // them directly and skip the staging bounce (see stagedTransfer).
    if (HostVisible)
      HostRanges.emplace(Ptr, Size);
    return Ptr;
  }

  Error free(void *TgtPtr, TargetAllocTy Kind) override {
    if (!TgtPtr)
      return Plugin::success();
    std::lock_guard<std::mutex> Lock(AllocMutex);
    Allocations.erase(TgtPtr);
    HostRanges.erase(TgtPtr);
    return Plugin::success();
  }

  // Host<->device copy: stream-ordered when async and the host buffer is
  // GPU-reachable, else the synchronous staged path (as CUDA does for pageable).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &Async) override {
    void *Hst = const_cast<void *>(HstPtr);
    if (auto *Stream = getStream(Async); Stream && hostReachable(Hst, Size))
      return Stream->enqueueCopy(TgtPtr, Hst, uint64_t(Size));
    return runOrEnqueue(Async, [this, TgtPtr, Hst, Size] {
      return stagedTransfer(TgtPtr, Hst, Size, /*ToDevice=*/true);
    });
  }
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &Async) override {
    void *Tgt = const_cast<void *>(TgtPtr);
    if (auto *Stream = getStream(Async); Stream && hostReachable(HstPtr, Size))
      return Stream->enqueueCopy(HstPtr, Tgt, uint64_t(Size));
    return runOrEnqueue(Async, [this, HstPtr, Tgt, Size] {
      return stagedTransfer(Tgt, HstPtr, Size, /*ToDevice=*/false);
    });
  }
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &Async) override {
    void *Src = const_cast<void *>(SrcPtr);
    auto &Dst = static_cast<KFDDeviceTy &>(DstDev);
    // Same device: stream-ordered CP DMA when async, else an inline SDMA copy.
    if (&Dst == this) {
      if (auto *Stream = getStream(Async))
        return Stream->enqueueCopy(DstPtr, Src, uint64_t(Size));
      return runOrEnqueue(Async, [this, Src, DstPtr, Size] {
        std::lock_guard<std::mutex> Lock(XferMutex);
        return sdmaCopy(DstPtr, Src, Size);
      });
    }
    // Peer device: a direct SDMA over the link after mapping the peer's buffer
    // (see crossDeviceExchange). Always synchronous; it spans two devices.
    return runOrEnqueue(Async, [this, &Dst, Src, DstPtr, Size] {
      return crossDeviceExchange(Dst, DstPtr, Src, Size);
    });
  }
  Error dataFillImpl(void *TgtPtr, const void *PatternPtr, int64_t PatternSize,
                     int64_t Size, AsyncInfoWrapperTy &Async) override {
    if (Size <= 0 || PatternSize <= 0)
      return Plugin::success();
    KFDStreamTy *Stream = getStream(Async);
    // Fast path: a broadcastable pattern at a word-multiple size is a CP fill;
    // sub-word sizes take the staged path.
    if (Stream && (Size % 4) == 0)
      if (auto FW = broadcastPattern(PatternPtr, PatternSize);
          FW && (Size % FW->Period) == 0)
        return Stream->enqueueFill(TgtPtr, FW->Word, uint64_t(Size));
    if (static_cast<size_t>(PatternSize) > StagingChunkBytes)
      return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                           "fill pattern exceeds staging buffer size");
    // Staged fallback; copy the pattern into the closure so it may run async.
    std::string Pattern(static_cast<const char *>(PatternPtr), PatternSize);
    auto Work = [this, TgtPtr, Pattern = std::move(Pattern), Size]() -> Error {
      std::lock_guard<std::mutex> Lock(XferMutex);
      if (auto Err = ensureStaging())
        return Err;
      // Tile the pattern across a staging slot; DMA out in whole-pattern chunks.
      size_t PatSize = Pattern.size();
      size_t ChunkBytes = (StagingChunkBytes / PatSize) * PatSize;
      char *Stage = static_cast<char *>(Staging[0]->data());
      for (size_t Off = 0; Off < ChunkBytes; Off += PatSize)
        std::memcpy(Stage + Off, Pattern.data(), PatSize);
      for (int64_t Off = 0; Off < Size; Off += ChunkBytes) {
        size_t N = std::min<int64_t>(ChunkBytes, Size - Off);
        if (auto Err = sdmaCopy(static_cast<char *>(TgtPtr) + Off, Stage, N))
          return Err;
      }
      return Plugin::success();
    };
    // Transfer work, not a host callback: run inline via the drain path.
    return runOrEnqueue(Async, std::move(Work));
  }
  Error dataFence(__tgt_async_info *) override { return Plugin::success(); }

  Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                        bool ReleaseQueue) override {
    auto *Stream = static_cast<KFDStreamTy *>(AsyncInfo.Queue);
    if (!Stream)
      return Plugin::success();
    Error Err = Stream->synchronize();
    if (ReleaseQueue) {
      recycleStream(Stream, !Err);
      AsyncInfo.Queue = nullptr;
    }
    return Err;
  }
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo, bool ReleaseQueue,
                       bool *IsQueueWorkCompleted) override {
    auto *Stream = static_cast<KFDStreamTy *>(AsyncInfo.Queue);
    bool Done = !Stream || Stream->idle();
    if (IsQueueWorkCompleted)
      *IsQueueWorkCompleted = Done;
    if (Done && ReleaseQueue && Stream) {
      Error Err = Stream->synchronize();
      recycleStream(Stream, !Err);
      AsyncInfo.Queue = nullptr;
      return Err;
    }
    return Plugin::success();
  }
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &Async) override {
    auto StreamOrErr = acquireStream();
    if (!StreamOrErr)
      return StreamOrErr.takeError();
    Async.setQueueAs<KFDStreamTy *>(*StreamOrErr);
    return Plugin::success();
  }
  Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                            AsyncInfoWrapperTy &Async) override {
    // cudaLaunchHostFunc: stream-ordered and non-blocking on the submitter.
    if (auto *Stream = getStream(Async))
      return deferHostFn(*Stream, [Callback, UserData] {
        Callback(UserData);
        return Plugin::success();
      });
    Callback(UserData);
    return Plugin::success();
  }

  /// Run \p Work on the Tier-1 worker, gated on the stream: it runs after prior
  /// stream work, and later GPU work waits for it (cudaLaunchHostFunc).
  Error deferHostFn(KFDStreamTy &Stream, std::function<Error()> Work) {
    auto BeforeOrErr = kfd::Signal::create(getKFDContext(), /*initial=*/1);
    if (!BeforeOrErr)
      return toErr(BeforeOrErr, "failed to create worker gate-in signal");
    auto AfterOrErr = kfd::Signal::create(getKFDContext(), /*initial=*/1);
    if (!AfterOrErr)
      return toErr(AfterOrErr, "failed to create worker gate-out signal");
    auto Before = std::make_shared<kfd::Signal>(std::move(*BeforeOrErr));
    auto After = std::make_shared<kfd::Signal>(std::move(*AfterOrErr));
    // Stream-order the callback: signal Before after prior work, stall later
    // work on After. The ring serializes successive gates.
    if (Error Err = Stream.enqueueGate(*Before, *After, Before, After))
      return Err;
    // The watcher fires when Before reaches 0 and hands Ctx to the worker.
    auto *Ctx = new KFDHostFnCtx{std::move(Work), Before, After, this};
    if (auto Res = getKFDContext().register_handler(
            *Before, kfd::Condition::EQ, 0, kfdHostFnTrampoline, Ctx);
        !Res) {
      delete Ctx;
      // The gate op still owns both signals, so deleting Ctx is safe. Release
      // After so later GPU work is not wedged behind a callback that won't run.
      consumeError(toErr(After->reset(/*value=*/0), "gate-out release failed"));
      return toErr(Res, "failed to register host-fn completion handler");
    }
    return Plugin::success();
  }

  /// Hand a ready callback context to the Tier-1 worker.
  void enqueueCallback(KFDHostFnCtx *Ctx) {
    std::lock_guard<std::mutex> Lock(CallbackMtx);
    CallbackQueue.push_back(Ctx);
    CallbackCV.notify_one();
  }

  // Pin host memory in place via register_host so it becomes SDMA-reachable
  // (zero-copy). Pages are refcounted, with one buffer per run of not-yet-pinned
  // pages. Best-effort: memory that cannot be a userptr falls back to unpinned.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    const uintptr_t PageSize = sysconf(_SC_PAGESIZE);
    uintptr_t Addr = reinterpret_cast<uintptr_t>(HstPtr);
    uintptr_t PBeg = Addr & ~(PageSize - 1);
    uintptr_t PEnd = (Addr + size_t(Size) + PageSize - 1) & ~(PageSize - 1);

    std::lock_guard<std::mutex> Lock(AllocMutex);
    llvm::SmallVector<uintptr_t, 4> NewRuns;
    auto unpinnedFallback = [&]() -> void * {
      for (uintptr_t R : NewRuns)
        PinnedRuns.erase(R);
      return HstPtr;
    };
    for (uintptr_t P = PBeg; P < PEnd;) {
      if (PinnedPageRefs.count(P)) {
        P += PageSize;
        continue;
      }
      uintptr_t RunBeg = P;
      while (P < PEnd && !PinnedPageRefs.count(P))
        P += PageSize;
      auto BufOrErr = kfd::Buffer::register_host(
          *KFDDevice, reinterpret_cast<void *>(RunBeg), size_t(P - RunBeg),
          kfd::MemFlags::WRITABLE | kfd::MemFlags::COHERENT);
      if (!BufOrErr)
        return unpinnedFallback();
      if (auto Res = BufOrErr->map(*KFDDevice); !Res)
        return unpinnedFallback();
      PinnedRuns.emplace(RunBeg, std::move(*BufOrErr));
      NewRuns.push_back(RunBeg);
    }
    for (uintptr_t P = PBeg; P < PEnd; P += PageSize)
      ++PinnedPageRefs[P];
    PinnedLocks.emplace(HstPtr, std::make_pair(PBeg, PEnd));
    HostRanges.emplace(HstPtr, size_t(Size));
    return HstPtr;
  }
  Error dataUnlockImpl(void *HstPtr) override {
    const uintptr_t PageSize = sysconf(_SC_PAGESIZE);
    std::lock_guard<std::mutex> Lock(AllocMutex);
    auto It = PinnedLocks.find(HstPtr);
    if (It == PinnedLocks.end())
      return Plugin::success();
    auto [PBeg, PEnd] = It->second;
    for (uintptr_t P = PBeg; P < PEnd; P += PageSize) {
      auto R = PinnedPageRefs.find(P);
      if (R != PinnedPageRefs.end() && --R->second == 0)
        PinnedPageRefs.erase(R);
    }
    // Free a run's buffer once none of its pages are referenced any longer.
    for (auto RIt = PinnedRuns.begin(); RIt != PinnedRuns.end();) {
      uintptr_t RBeg = RIt->first;
      uintptr_t REnd = RBeg + RIt->second.size();
      if (RBeg >= PEnd || REnd <= PBeg) {
        ++RIt;
        continue;
      }
      bool AnyRef = false;
      for (uintptr_t P = RBeg; P < REnd; P += PageSize)
        if (PinnedPageRefs.count(P)) {
          AnyRef = true;
          break;
        }
      RIt = AnyRef ? std::next(RIt) : PinnedRuns.erase(RIt);
    }
    PinnedLocks.erase(It);
    HostRanges.erase(HstPtr);
    return Plugin::success();
  }
  // Pinning is tracked in the generic PinnedAllocationMapTy via dataLockImpl;
  // there is no external KFD source of pinned memory to query here.
  Expected<bool> isPinnedPtrImpl(void *, void *&, void *&,
                                 size_t &) const override {
    return false;
  }

  // Events are GPU-signal completion markers (see KFDEventTy).
  Error createEventImpl(void **EventPtrStorage, bool) override {
    auto SignalOrErr = kfd::Signal::create(getKFDContext(), /*initial=*/0);
    if (!SignalOrErr)
      return toErr(SignalOrErr, "failed to create event signal");
    *EventPtrStorage = new KFDEventTy(std::move(*SignalOrErr));
    return Plugin::success();
  }
  Error destroyEventImpl(void *EventPtr, bool) override {
    delete static_cast<KFDEventTy *>(EventPtr);
    return Plugin::success();
  }
  Error recordEventImpl(void *EventPtr, AsyncInfoWrapperTy &Async,
                        bool) override {
    auto &Ev = *static_cast<KFDEventTy *>(EventPtr);
    if (auto *Stream = getStream(Async)) {
      // Arm on the host, then decrement on the GPU after prior stream work.
      if (auto Res = Ev.Signal.reset(/*value=*/1); !Res)
        return toErr(Res, "failed to reset event signal");
      return Stream->enqueueMarker(Ev.Signal);
    }
    // No stream: prior synchronous work has already completed, so mark done.
    if (auto Res = Ev.Signal.reset(/*value=*/0); !Res)
      return toErr(Res, "failed to reset event signal");
    return Plugin::success();
  }
  Error waitEventImpl(void *EventPtr, AsyncInfoWrapperTy &Async) override {
    auto &Ev = *static_cast<KFDEventTy *>(EventPtr);
    if (auto *Stream = getStream(Async)) {
      // Shared-ring safety: recordEvent submits the marker before this wait, so
      // the producer is always ahead on the ring; a fired event needs no wait.
      if (Ev.Signal.value() == 0)
        return Plugin::success();
      return Stream->enqueueWaitOn(Ev.Signal);
    }
    if (auto Res = Ev.Signal.wait(kfd::Condition::EQ, 0, UINT64_MAX); !Res)
      return toErr(Res, "failed to wait on event");
    return Plugin::success();
  }
  Expected<bool> isEventCompleteImpl(void *EventPtr,
                                     AsyncInfoWrapperTy &) override {
    return static_cast<KFDEventTy *>(EventPtr)->Signal.value() == 0;
  }
  Error syncEventImpl(void *EventPtr) override {
    auto &Ev = *static_cast<KFDEventTy *>(EventPtr);
    if (auto Res = Ev.Signal.wait(kfd::Condition::EQ, 0, UINT64_MAX); !Res)
      return toErr(Res, "failed to wait on event");
    return Plugin::success();
  }
  Expected<float> getEventElapsedTimeImpl(void *, void *) override {
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "event elapsed time is not supported");
  }
  Expected<bool> hasPendingWorkImpl(AsyncInfoWrapperTy &Async) override {
    if (!Async.hasQueue())
      return false;
    auto *Stream = Async.getQueueAs<KFDStreamTy *>();
    return Stream && !Stream->idle();
  }

  std::string getComputeUnitKind() const override {
    return std::string(KFDDevice->name());
  }

  Error getDeviceMemorySize(uint64_t &DSize) override {
    DSize = 0;
    for (const auto &Bank : KFDDevice->memory_banks())
      DSize += Bank.size_in_bytes;
    if (DSize == 0)
      DSize = 1;
    return Plugin::success();
  }

  Error getDeviceStackSize(uint64_t &V) override {
    V = 0;
    return Plugin::success();
  }
  Error setDeviceStackSize(uint64_t) override { return Plugin::success(); }

  Expected<InfoTreeNode> obtainInfoImpl() override {
    InfoTreeNode Info;
    const auto &Props = KFDDevice->properties();
    std::string Name(KFDDevice->name());

    Info.add("Device Type", "AMDGPU (KFD)");
    Info.add("Product Name", Name, "", DeviceInfo::PRODUCT_NAME);
    Info.add("Device Name", Name, "", DeviceInfo::NAME);
    Info.add("Vendor", "AMD", "", DeviceInfo::VENDOR);
    Info.add("Vendor ID", uint64_t(Props.vendor_id), "", DeviceInfo::VENDOR_ID);
    Info.add("Driver Version", "KFD-" + std::to_string(Props.fw_version), "",
             DeviceInfo::DRIVER_VERSION);
    Info.add("GFX Version", uint64_t(Props.gfx_target_version));
    Info.add("Warp Size", uint64_t(getWarpSize()));
    Info.add("Memory Address Size", uint64_t(48), "bits",
             DeviceInfo::ADDRESS_BITS);

    uint32_t ComputeUnits =
        Props.simd_per_cu ? Props.simd_count / Props.simd_per_cu : 0;
    Info.add("Compute Units", uint64_t(ComputeUnits ? ComputeUnits : 1), "",
             DeviceInfo::NUM_COMPUTE_UNITS);

    // Real hardware reports nonzero clocks; clamp to satisfy "> 0" consumers.
    uint64_t EngineClk =
        Props.max_engine_clk_fcompute ? Props.max_engine_clk_fcompute : 1;
    Info.add("Max Clock Freq", EngineClk, "MHz",
             DeviceInfo::MAX_CLOCK_FREQUENCY);
    Info.add("Memory Clock Rate", EngineClk, "MHz",
             DeviceInfo::MEMORY_CLOCK_RATE);

    uint64_t LdsBytes = uint64_t(Props.lds_size_in_kb) << 10;
    Info.add("Max Shared Memory per Work Group", LdsBytes ? LdsBytes : 1,
             "bytes", DeviceInfo::WORK_GROUP_LOCAL_MEM_SIZE);

    uint64_t MaxWG = 1024;
    Info.add("Workgroup Max Size", MaxWG, "", DeviceInfo::MAX_WORK_GROUP_SIZE);
    auto &WGDim =
        *Info.add("Workgroup Max Size per Dimension", std::monostate{}, "",
                  DeviceInfo::MAX_WORK_GROUP_SIZE_PER_DIMENSION);
    WGDim.add("x", MaxWG);
    WGDim.add("y", MaxWG);
    WGDim.add("z", MaxWG);

    uint64_t MaxGrid = UINT32_MAX;
    Info.add("Grid Max Size", MaxGrid, "", DeviceInfo::MAX_WORK_SIZE);
    auto &GridDim = *Info.add("Grid Max Size per Dimension", std::monostate{},
                              "", DeviceInfo::MAX_WORK_SIZE_PER_DIMENSION);
    GridDim.add("x", MaxGrid);
    GridDim.add("y", MaxGrid);
    GridDim.add("z", MaxGrid);

    ol_device_fp_capability_flags_t FPFlags =
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT |
        OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
        OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
        OL_DEVICE_FP_CAPABILITY_FLAG_DENORM | OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
    Info.add("Single FP Support", true, "", DeviceInfo::SINGLE_FP_SUPPORT);
    Info.add("Single FP Capabilities", uint64_t(FPFlags), "",
             DeviceInfo::SINGLE_FP_CONFIG);
    Info.add("Double FP Support", true, "", DeviceInfo::DOUBLE_FP_SUPPORT);
    Info.add("Double FP Capabilities", uint64_t(FPFlags), "",
             DeviceInfo::DOUBLE_FP_CONFIG);
    Info.add("Half FP Support", false, "", DeviceInfo::HALF_FP_SUPPORT);
    Info.add("Half FP Capabilities", uint64_t(0), "",
             DeviceInfo::HALF_FP_CONFIG);
    return Info;
  }

  /// Run \p Fn inline, draining the stream first if \p Async has one.
  Error runOrEnqueue(AsyncInfoWrapperTy &Async, KFDStreamTy::HostFn Fn) {
    if (!Async.hasQueue())
      return Fn();
    return Async.getQueueAs<KFDStreamTy *>()->enqueue(std::move(Fn));
  }

  /// Dispatch on the shared compute queue and block on its completion signal.
  /// Used only for queue-less (synchronous) launches.
  Error dispatchAndWait(const kfd::Kernel &Kernel,
                        const kfd::DispatchConfig &Config,
                        const kfd::Buffer &Kernarg) {
    std::lock_guard<std::mutex> Lock(ComputeMutex);
    // Keep the RPC server awake across the wait so host calls make progress.
    RPCServerTy *Rpc = getRPCServer();
    if (Rpc)
      Rpc->Thread->notify();
    Error Err = [&]() -> Error {
      if (auto Res = ComputeSignal->reset(/*value=*/1); !Res)
        return toErr(Res, "failed to reset completion signal");
      if (auto Res = Queue->dispatch(Kernel, Config, Kernarg, *ComputeSignal);
          !Res)
        return toErr(Res, "failed to dispatch kernel");
      if (auto Res = ComputeSignal->wait(kfd::Condition::EQ, 0, UINT64_MAX);
          !Res)
        return toErr(Res, "failed to wait on kernel completion");
      return Plugin::success();
    }();
    if (Rpc)
      Rpc->Thread->finish();
    return Err;
  }

  KFDStreamTy *getStream(AsyncInfoWrapperTy &Async) {
    return Async.hasQueue() ? Async.getQueueAs<KFDStreamTy *>() : nullptr;
  }

  /// Take an idle stream from the pool (or build one) and bind it a hardware
  /// queue from the bounded pool, which streams may share.
  Expected<KFDStreamTy *> acquireStream() {
    KFDStreamTy *Stream = nullptr;
    {
      std::lock_guard<std::mutex> Lock(StreamPoolMtx);
      if (!StreamPool.empty()) {
        Stream = StreamPool.back();
        StreamPool.pop_back();
      }
    }
    if (!Stream)
      Stream = new KFDStreamTy();

    auto HwOrErr = QueuePool.assign();
    if (!HwOrErr) {
      std::lock_guard<std::mutex> Lock(StreamPoolMtx);
      StreamPool.push_back(Stream);
      return HwOrErr.takeError();
    }
    Stream->setHwQueue(*HwOrErr);
    Stream->setRPCServer(getRPCServer());
    Stream->setArgsManager(&ArgsManager);
    Stream->setSignalPool(&SignalPool);
    return Stream;
  }

  /// Release the stream's hardware queue and pool the stream. If \p Idle is
  /// false (failed synchronize) it may still have live waves, so it is deleted.
  void recycleStream(KFDStreamTy *Stream, bool Idle) {
    QueuePool.release(Stream->hwQueue());
    Stream->setHwQueue(nullptr);
    if (Idle) {
      std::lock_guard<std::mutex> Lock(StreamPoolMtx);
      if (StreamPool.size() < StreamPoolCap) {
        StreamPool.push_back(Stream);
        return;
      }
    }
    delete Stream;
  }

  /// The device's canonical kernarg allocator, shared by every launch path.
  KFDArgsManagerTy &getArgsManager() { return ArgsManager; }

private:
  /// Launch the ctor/dtor kernel emitted by 'amdgpu-lower-ctor-dtor' if it is
  /// present, synchronously with a 1x1x1 grid and no arguments.
  Error callGlobalCtorDtorCommon(GenericPluginTy &Plugin, DeviceImageTy &Image,
                                 bool IsCtor) {
    const char *KernelName =
        IsCtor ? "amdgcn.device.init" : "amdgcn.device.fini";
    GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
    if (!Handler.isSymbolInImage(*this, Image, KernelName))
      return Plugin::success();

    KFDKernelTy Kernel(KernelName);
    if (auto Err = Kernel.init(*this, Image))
      return Err;

    AsyncInfoWrapperTy AsyncInfoWrapper(*this, nullptr);
    KernelArgsTy KernelArgs = {};
    uint32_t NumBlocksAndThreads[3] = {1u, 1u, 1u};
    Error Err =
        Kernel.launchImpl(*this, NumBlocksAndThreads, NumBlocksAndThreads,
                          /*DynBlockMemSize=*/0, KernelArgs,
                          KernelLaunchParamsTy{}, AsyncInfoWrapper);
    AsyncInfoWrapper.finalize(Err);
    return Err;
  }

  /// Tier-1 worker: run queued callbacks and release their gate-out signals,
  /// draining remaining work on shutdown so no gated stream is left stalled.
  void callbackLoop() {
    while (true) {
      KFDHostFnCtx *Ctx = nullptr;
      {
        std::unique_lock<std::mutex> Lock(CallbackMtx);
        CallbackCV.wait(Lock,
                        [&] { return CallbackStop || !CallbackQueue.empty(); });
        if (CallbackQueue.empty())
          return;
        Ctx = CallbackQueue.front();
        CallbackQueue.pop_front();
      }
      consumeError(Ctx->Work());
      if (auto Res = Ctx->After->reset(/*value=*/0); !Res)
        consumeError(toErr(Res, "host-fn gate-out release failed"));
      delete Ctx;
    }
  }

  /// Lazily allocate the GTT staging buffers. Caller must hold XferMutex.
  Error ensureStaging() {
    for (unsigned I = 0; I < NumStaging; ++I) {
      if (Staging[I])
        continue;
      auto BufOrErr = kfd::Buffer::allocate(
          *KFDDevice, StagingChunkBytes, kfd::MemType::GTT,
          kfd::MemFlags::WRITABLE | kfd::MemFlags::HOST_ACCESS |
              kfd::MemFlags::COHERENT);
      if (!BufOrErr)
        return toErr(BufOrErr, "failed to allocate staging buffer");
      kfd::Buffer Buf = std::move(*BufOrErr);
      if (auto Res = Buf.map(*KFDDevice); !Res)
        return toErr(Res, "failed to map staging buffer");
      Staging[I].emplace(std::move(Buf));
    }
    return Plugin::success();
  }

  /// Submit an SDMA linear copy plus completion signal without waiting, so the
  /// caller can overlap it with CPU work. Caller must hold XferMutex.
  Error sdmaSubmit(void *Dst, void *Src, int64_t Size, kfd::Signal &Sig) {
    if (auto Res = Sig.reset(/*value=*/1); !Res)
      return toErr(Res, "failed to reset transfer signal");
    if (auto Res = SDMA->copy_linear(Dst, Src, Size); !Res)
      return toErr(Res, "failed to submit SDMA copy");
    if (auto Res = SDMA->signal(Sig); !Res)
      return toErr(Res, "failed to submit SDMA completion signal");
    return Plugin::success();
  }
  Error sdmaWait(kfd::Signal &Sig) {
    if (auto Res = Sig.wait(kfd::Condition::EQ, 0, UINT64_MAX); !Res)
      return toErr(Res, "failed to wait on SDMA completion");
    return Plugin::success();
  }

  /// Submit a single SDMA linear copy and block until it completes. Used for
  /// device-to-device and fill, where there is no host copy to overlap. Caller
  /// must hold XferMutex.
  Error sdmaCopy(void *Dst, void *Src, int64_t Size) {
    if (auto Err = sdmaSubmit(Dst, Src, Size, *XferSignal[0]))
      return Err;
    return sdmaWait(*XferSignal[0]);
  }

  /// Copy this device's VRAM to peer \p Dst's VRAM with a single SDMA over the
  /// link, after mapping the destination buffer into this device.
  Error crossDeviceExchange(KFDDeviceTy &Dst, void *DstPtr, void *Src,
                            int64_t Size) {
    if (Size <= 0)
      return Plugin::success();
    {
      std::lock_guard<std::mutex> Lock(Dst.AllocMutex);
      auto It = Dst.Allocations.upper_bound(DstPtr);
      if (It == Dst.Allocations.begin())
        return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                             "peer destination is not a device allocation");
      --It;
      char *Base = static_cast<char *>(It->first);
      if (static_cast<char *>(DstPtr) + Size > Base + It->second.size())
        return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                             "peer destination overruns its allocation");
      if (auto Res = It->second.map(*KFDDevice); !Res)
        return toErr(Res, "failed to map peer destination for access");
    }
    std::lock_guard<std::mutex> Lock(XferMutex);
    return sdmaCopy(DstPtr, Src, Size);
  }

  // A fill pattern collapsed to the 32-bit word a CP DMA fill broadcasts, plus
  // the byte period it repeats at (the total size must be a multiple of it).
  struct FillWord {
    uint32_t Word;
    int Period;
  };

  /// Collapse a fill pattern into the 32-bit word a CP fill broadcasts. Handles
  /// 1-byte (any size), 2-, and 4-byte patterns; nullopt for anything else.
  static std::optional<FillWord> broadcastPattern(const void *Pattern,
                                                  int64_t PatternSize) {
    auto *B = static_cast<const unsigned char *>(Pattern);
    bool AllEqual = true;
    for (int64_t I = 1; I < PatternSize; ++I)
      if (B[I] != B[0]) {
        AllEqual = false;
        break;
      }
    if (AllEqual)
      return FillWord{uint32_t(B[0]) * 0x01010101u, 1};
    if (PatternSize == 2) {
      uint32_t H = uint32_t(B[0]) | (uint32_t(B[1]) << 8);
      return FillWord{H | (H << 16), 2};
    }
    if (PatternSize == 4)
      return FillWord{uint32_t(B[0]) | (uint32_t(B[1]) << 8) |
                          (uint32_t(B[2]) << 16) | (uint32_t(B[3]) << 24),
                      4};
    return std::nullopt;
  }

  /// True when [Ptr, Ptr+Size) lies wholly within a host/shared allocation,
  /// which is already GPU-mapped and thus directly SDMA-reachable.
  bool hostReachable(void *Ptr, int64_t Size) {
    std::lock_guard<std::mutex> Lock(AllocMutex);
    auto It = HostRanges.upper_bound(Ptr);
    if (It == HostRanges.begin())
      return false;
    --It;
    char *Base = static_cast<char *>(It->first);
    return static_cast<char *>(Ptr) >= Base &&
           static_cast<char *>(Ptr) + Size <= Base + It->second;
  }

  /// Copy between host memory and device VRAM: a single zero-copy DMA for
  /// SDMA-reachable host memory, else a pipelined double-buffered bounce.
  Error stagedTransfer(void *DevPtr, void *HstPtr, int64_t Size,
                       bool ToDevice) {
    if (hostReachable(HstPtr, Size)) {
      std::lock_guard<std::mutex> Lock(XferMutex);
      return ToDevice ? sdmaCopy(DevPtr, HstPtr, Size)
                      : sdmaCopy(HstPtr, DevPtr, Size);
    }
    std::lock_guard<std::mutex> Lock(XferMutex);
    if (auto Err = ensureStaging())
      return Err;
    char *Dev = static_cast<char *>(DevPtr);
    char *Hst = static_cast<char *>(HstPtr);
    const int64_t Chunk = StagingChunkBytes;
    const int64_t N = (Size + Chunk - 1) / Chunk;
    auto len = [&](int64_t I) {
      return static_cast<size_t>(std::min(Chunk, Size - I * Chunk));
    };

    if (ToDevice) {
      // memcpy(chunk i) overlaps SDMA(chunk i-1); reclaim a slot once its DMA
      // has retired.
      for (int64_t I = 0; I < N; ++I) {
        unsigned B = I & 1;
        if (I >= NumStaging)
          if (auto Err = sdmaWait(*XferSignal[B]))
            return Err;
        std::memcpy(Staging[B]->data(), Hst + I * Chunk, len(I));
        if (auto Err = sdmaSubmit(Dev + I * Chunk, Staging[B]->data(), len(I),
                                  *XferSignal[B]))
          return Err;
      }
      for (int64_t I = std::max<int64_t>(0, N - NumStaging); I < N; ++I)
        if (auto Err = sdmaWait(*XferSignal[I & 1]))
          return Err;
      return Plugin::success();
    }

    // From device: SDMA(chunk i+1) overlaps memcpy(chunk i); prefetch chunk 0.
    auto fetch = [&](int64_t I) {
      unsigned B = I & 1;
      return sdmaSubmit(Staging[B]->data(), Dev + I * Chunk, len(I),
                        *XferSignal[B]);
    };
    if (N > 0)
      if (auto Err = fetch(0))
        return Err;
    for (int64_t I = 0; I < N; ++I) {
      unsigned B = I & 1;
      if (I + 1 < N)
        if (auto Err = fetch(I + 1))
          return Err;
      if (auto Err = sdmaWait(*XferSignal[B]))
        return Err;
      std::memcpy(Hst + I * Chunk, Staging[B]->data(), len(I));
    }
    return Plugin::success();
  }

  // Two staging slots suffice to overlap a CPU copy with one in-flight DMA.
  static constexpr size_t StagingChunkBytes = 8ull << 20;
  static constexpr unsigned NumStaging = 2;

  kfd::Device *KFDDevice;
  uint64_t HardwareParallelism = 1;
  std::optional<kfd::ComputeQueue> Queue;
  std::optional<kfd::SDMAQueue> SDMA;

  // Idle streams kept for reuse; the scarce hardware queues live in QueuePool.
  std::mutex StreamPoolMtx;
  llvm::SmallVector<KFDStreamTy *> StreamPool;
  static constexpr size_t StreamPoolCap = 64;

  // Bounded pool of hardware queues that streams are multiplexed onto.
  KFDQueuePoolTy QueuePool;

  // Tier-1 host-callback worker; the watcher only pushes ready contexts here.
  std::thread CallbackWorker;
  std::mutex CallbackMtx;
  std::condition_variable CallbackCV;
  std::deque<KFDHostFnCtx *> CallbackQueue;
  bool CallbackStop = false;

  // Canonical kernarg allocator shared by every launch path (sync and async).
  KFDArgsManagerTy ArgsManager;

  // Device-wide free-list of per-op completion signals borrowed by streams.
  KFDSignalPoolTy SignalPool;

  // Serializes submissions on the shared compute queue; never held across a wait.
  std::mutex ComputeMutex;
  // Reused completion signal for synchronous (queue-less) launches.
  std::optional<kfd::Signal> ComputeSignal;

  std::mutex XferMutex;
  std::optional<kfd::Signal> XferSignal[NumStaging];
  std::optional<kfd::Buffer> Staging[NumStaging];

  std::mutex AllocMutex;
  std::map<void *, kfd::Buffer> Allocations;
  // In-place host pinning state (see dataLockImpl), all under AllocMutex.
  std::map<uintptr_t, kfd::Buffer> PinnedRuns;
  std::map<uintptr_t, unsigned> PinnedPageRefs;
  std::map<void *, std::pair<uintptr_t, uintptr_t>> PinnedLocks;
  // Base -> size of GPU-reachable host memory (host/shared allocations and
  // pinned host pointers), for the zero-copy transfer path.
  std::map<void *, size_t> HostRanges;
};

/// Global handler resolving device symbols from the loaded Executable.
struct KFDGlobalHandlerTy final : public GenericGlobalHandlerTy {
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    auto &KImage = static_cast<KFDDeviceImageTy &>(Image);
    auto SymOrErr = KImage.getExecutable().symbol(DeviceGlobal.getName());
    if (!SymOrErr)
      return toErr(SymOrErr, "failed to resolve device global");

    std::span<std::byte> Sym = *SymOrErr;
    if (DeviceGlobal.getSize() && DeviceGlobal.getSize() != Sym.size())
      return Plugin::error(ErrorCode::INVALID_BINARY,
                           "global '%s' size mismatch",
                           DeviceGlobal.getName().c_str());
    DeviceGlobal.setSize(Sym.size());
    DeviceGlobal.setPtr(Sym.data());
    return Plugin::success();
  }
};

/// Watcher-thread handler for an unrecoverable GPU fault: report it and abort,
/// as the HSA runtime does, rather than letting in-flight waits hang forever.
static void kfdFaultHandler(const kfd::FaultInfo &Fault, void *PluginPtr) {
  auto *Plugin = reinterpret_cast<GenericPluginTy *>(PluginPtr);

  // Pause briefly so the RPC server can flush any final printf/assert output.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  if (Fault.kind == kfd::FaultInfo::Kind::MemoryViolation) {
    const auto &M = Fault.memory;
    // Match the HSA plugin's wording so shared diagnostics/tests apply.
    std::string Reasons;
    auto Add = [&](const char *R) {
      if (!Reasons.empty())
        Reasons += ", ";
      Reasons += R;
    };
    if (M.reason & kfd::MemoryFaultInfo::NotPresent)
      Add("Page not present or supervisor privilege");
    if (M.reason & kfd::MemoryFaultInfo::ReadOnly)
      Add("Write access to a read-only page");
    if (M.reason & kfd::MemoryFaultInfo::NoExecute)
      Add("Execute access to a page marked NX");
    if (M.reason & kfd::MemoryFaultInfo::Imprecise)
      Add("Can't determine the exact fault address");
    if (Reasons.empty())
      Add("Unknown");

    // Surface gpu_id as the agent handle to match the HSA plugin's message.
    char Buf[256];
    snprintf(Buf, sizeof(Buf),
             "memory access fault by GPU %u (agent 0x%lx) at virtual address "
             "%p. Reasons: %s",
             Fault.gpu_id, (unsigned long)Fault.gpu_id, (void *)M.va,
             Reasons.c_str());
    std::string S(Buf);

    // Report against the faulting device: kernel traces, then the fault, abort.
    if (Plugin) {
      for (int32_t I = 0, E = Plugin->getNumDevices(); I < E; ++I) {
        auto &Device = static_cast<KFDDeviceTy &>(Plugin->getDevice(I));
        if (Device.getGpuId() != Fault.gpu_id)
          continue;
        auto KTIR = Device.KernelLaunchTraces.getExclusiveAccessor();
        ErrorReporter::reportKernelTraces(Device, *KTIR);
        ErrorReporter::reportMemoryAccessError(Device, (void *)M.va, S,
                                               /*Abort=*/true);
      }
    }
    // No matching device: still surface the fault before aborting.
    fprintf(stderr, "OFFLOAD ERROR: %s\n", S.c_str());
    fflush(stderr);
  } else {
    fprintf(stderr,
            "KFD: hardware exception on GPU node %u (reset_type:%u cause:%u "
            "memory_lost:%u)\n",
            Fault.gpu_id, Fault.hardware.reset_type, Fault.hardware.reset_cause,
            Fault.hardware.memory_lost);
    fflush(stderr);
  }
  std::abort();
}

/// Plugin holding the libkfd context and enumerated devices.
struct KFDPluginTy final : public GenericPluginTy {
  KFDPluginTy() : GenericPluginTy(getTripleArch()) {}

  KFDPluginTy(const KFDPluginTy &) = delete;
  KFDPluginTy(KFDPluginTy &&) = delete;

  kfd::Context &getKFDContext() { return *Context; }

  Expected<int32_t> initImpl() override {
    auto CtxOrErr = kfd::Context::create();
    if (!CtxOrErr)
      return toErr(CtxOrErr, "failed to open KFD context");
    Context.emplace(std::move(*CtxOrErr));
    // Install our fault handler; the plugin lets it map gpu_id to a device.
    if (auto Res = Context->register_handler(kfdFaultHandler, this); !Res)
      return toErr(Res, "failed to register fault handler");
    return static_cast<int32_t>(Context->num_devices());
  }

  Error deinitImpl() override {
    Context.reset();
    return Plugin::success();
  }

  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new KFDDeviceTy(Plugin, DeviceId, NumDevices,
                           Context->devices()[DeviceId]);
  }

  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new KFDGlobalHandlerTy();
  }

  // EM_AMDGPU, numeric to avoid clashing with libkfd's own ELF macros.
  uint16_t getMagicElfBits() const override { return /*EM_AMDGPU=*/224; }

  Triple::ArchType getTripleArch() const override {
    return llvm::Triple::amdgcn;
  }

  const char *getName() const override { return GETNAME(TARGET_NAME); }

  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    auto *Self = const_cast<KFDPluginTy *>(this);
    kfd::Device &Dev = Self->Context->devices()[DeviceId];
    std::span<const std::byte> Bytes(
        reinterpret_cast<const std::byte *>(Image.data()), Image.size());
    return Dev.loadable(Bytes);
  }

private:
  std::optional<kfd::Context> Context;
};

kfd::Context &KFDDeviceTy::getKFDContext() {
  return static_cast<KFDPluginTy &>(Plugin).getKFDContext();
}

static void kfdHostFnTrampoline(void *P) {
  auto *Ctx = static_cast<KFDHostFnCtx *>(P);
  Ctx->Device->enqueueCallback(Ctx);
}

Error KFDKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                              uint32_t NumThreads[3], uint32_t NumBlocks[3],
                              uint32_t DynBlockMemSize,
                              KernelArgsTy &KernelArgs,
                              KernelLaunchParamsTy LaunchParams,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  auto &Device = static_cast<KFDDeviceTy &>(GenericDevice);

  kfd::DispatchConfig Config;
  Config.grid = {NumBlocks[0], NumBlocks[1], NumBlocks[2]};
  Config.block = {NumThreads[0], NumThreads[1], NumThreads[2]};
  Config.dynamic_lds = DynBlockMemSize;

  const kfd::abi::KernelDescriptor &Descriptor = Kernel->descriptor();
  size_t Total = kfd::abi::kernarg_alloc_size(Descriptor.kernarg_size);

  // Kernarg buffer from the shared manager; holds explicit + implicit (Total).
  auto KernargOrErr = Device.getArgsManager().allocate(Total);
  if (!KernargOrErr)
    return KernargOrErr.takeError();
  kfd::Buffer &Kernarg = **KernargOrErr;
  void *KernargPtr = Kernarg.data();

  std::memset(KernargPtr, 0, Total);

  // Each LaunchParams.Args[I] points at argument I's value; place it at its
  // AMDHSA ".args" offset. ArgMDs may list hidden args past NumArgs.
  size_t ExplicitSize = 0;
  if (LaunchParams.Args) {
    if (LaunchParams.NumArgs > ArgMDs.size())
      return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                           "kernel '%s' got %u args but expects at most %zu",
                           getName(), LaunchParams.NumArgs, ArgMDs.size());
    char *Base = static_cast<char *>(Kernarg.data());
    for (uint32_t I = 0; I < LaunchParams.NumArgs; ++I) {
      auto [Offset, Size] = ArgMDs[I];
      // Widen to 64-bit so a malformed offset cannot wrap the bound check.
      if (LaunchParams.Args[I] && uint64_t(Offset) + Size <= Total)
        std::memcpy(Base + Offset, LaunchParams.Args[I], Size);
    }
    if (LaunchParams.NumArgs) {
      auto [Offset, Size] = ArgMDs[LaunchParams.NumArgs - 1];
      ExplicitSize = uint64_t(Offset) + Size;
    }
  }
  kfd::abi::fill_implicit_args(KernargPtr, ExplicitSize, Descriptor, Config);

  // Stream the dispatch when a queue is present, else dispatch synchronously.
  KFDStreamTy *Stream = Device.getStream(AsyncInfoWrapper);
  if (Stream)
    return Stream->enqueueDispatch(*Kernel, Config, KernargPtr, Kernarg);
  // dispatchAndWait blocks until done, so the kernarg can be returned here.
  Error Err = Device.dispatchAndWait(*Kernel, Config, Kernarg);
  Device.getArgsManager().deallocate(KernargPtr);
  return Err;
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_kfd() {
  return new llvm::omp::target::plugin::KFDPluginTy();
}
}
