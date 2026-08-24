//===-- dasan_rpc.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Same shape as offload's RPC.cpp: one opcode handler, one buffer per GPU
// ordinal. If __tgt_register_rpc_callback is present, register that handler
// and do nothing else. Otherwise own the buffers, doorbell, and a thread that
// walks the live device list under RpcMutex.
//
// Lock order: RpcMutex then DasanMutex. StartRpc holds both after offload
// probe so Init cannot rediscover Gpus/HostPool under a concurrent setup.
//
//===----------------------------------------------------------------------===//

#include "dasan_rpc.h"

#include <dlfcn.h>

#include "dasan.h"
#include "dasan_hsa.h"
#include "dasan_platform.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "shared/rpc.h"

using namespace __sanitizer;

namespace __dasan {
namespace {

struct DeviceRpc {
  void* Buffer;
  u32 Ports;
  u32 Lanes;
};

Mutex RpcMutex;  // Nest DasanMutex after this; never the reverse.
InternalMmapVectorNoCtor<DeviceRpc> Devices;
void* Thread;
hsa_signal_t Doorbell;
u64* DoorbellValue;
u64* DoorbellMailbox;
u32 DoorbellEvent;
bool Offload;
atomic_uint8_t Stop;

uint32_t handleReport(void* PortPtr, uint32_t) {
  auto& Port = *reinterpret_cast<rpc::Server::Port*>(PortPtr);
  if (Port.get_opcode() != DASAN_REPORT)
    return rpc::RPC_UNHANDLED_OPCODE;

  bool Halt = false;
  Port.recv([&](rpc::Buffer* Buffer, uint32_t) {
    dasan_report_t R;
    internal_memcpy(&R, Buffer->data, sizeof(R));
    Lock L(&DasanMutex);
    Halt = PrintReport(R);
  });
  if (Halt)
    Die();
  Port.send([](rpc::Buffer*, uint32_t) {});
  return rpc::RPC_SUCCESS;
}

bool tryOffload() {
  if (Offload)
    return true;
  using RegisterFn = void (*)(uint32_t (*)(void*, uint32_t));
  auto Register = reinterpret_cast<RegisterFn>(
      dlsym(RTLD_DEFAULT, "__tgt_register_rpc_callback"));
  if (!Register)
    return false;
  Register(handleReport);
  Offload = true;
  VReport(1, "%s: reporting through the offload runtime's server\n",
          SanitizerToolName);
  return true;
}

bool executableNeedsRPC(hsa_executable_t Exec, const Device& Gpu, u64* Addr) {
  return GetHsa().SymbolAddr(Exec, "__llvm_rpc_client", Gpu.Agent, Addr);
}

void drainDevice(DeviceRpc& D) {
  if (!D.Buffer)
    return;
  rpc::Server Server(D.Ports, D.Buffer);
  while (auto Port = Server.try_open(D.Lanes)) {
    if (handleReport(&*Port, D.Lanes) == rpc::RPC_UNHANDLED_OPCODE)
      VReport(1, "%s: unexpected opcode 0x%x on the report channel\n",
              SanitizerToolName, Port->get_opcode());
  }
}

void drainAll() {
  for (uptr I = 0; I < Devices.size(); ++I) drainDevice(Devices[I]);
}

void* ServerLoop(void*) {
  Hsa& H = GetHsa();
  for (;;) {
    if (!atomic_load_relaxed(&Stop))
      H.WaitSignalNE(Doorbell, 0);
    Lock L(&RpcMutex);
    drainAll();
    if (atomic_load_relaxed(&Stop))
      break;
  }
  return nullptr;
}

bool initDoorbell() {
  if (Doorbell.handle)
    return true;
  return GetHsa().InitDoorbell(&Doorbell, &DoorbellValue, &DoorbellMailbox,
                               &DoorbellEvent);
}

void plantDoorbell(void* Buffer) {
  if (!Doorbell.handle)
    return;
  auto* Bell = reinterpret_cast<rpc::Doorbell*>(static_cast<u8*>(Buffer) +
                                                rpc::Server::doorbell_offset());
  Bell->value = reinterpret_cast<uint64_t*>(DoorbellValue);
  Bell->mailbox = reinterpret_cast<uint64_t*>(DoorbellMailbox);
  Bell->event_id = DoorbellEvent;
}

bool startThread() {
  if (Thread)
    return true;
  if (atomic_load_relaxed(&Stop))
    return false;
  if (!initDoorbell())
    return false;
  Thread = internal_start_thread(ServerLoop, nullptr);
  return Thread;
}

bool initDevice(uptr Ordinal, const Device& Gpu) {
  while (Devices.size() < GetHsa().Gpus.size()) {
    DeviceRpc Empty = {};
    Devices.push_back(Empty);
  }
  if (Ordinal >= Devices.size())
    return false;

  DeviceRpc& D = Devices[Ordinal];
  if (D.Buffer)
    return true;

  D.Lanes = Gpu.Lanes;
  D.Ports = Gpu.Waves;
  if (D.Ports > rpc::MAX_PORT_COUNT)
    D.Ports = rpc::MAX_PORT_COUNT;
  if (!D.Ports)
    D.Ports = 64;

  const uptr Bytes = rpc::Server::allocation_size(D.Lanes, D.Ports);
  void* Buffer = nullptr;
  if (!GetHsa().AllocHost(Bytes, &Buffer) || !Buffer)
    return false;
  internal_memset(Buffer, 0, Bytes);
  if (!initDoorbell()) {
    GetHsa().FreeHost(Buffer);
    return false;
  }
  plantDoorbell(Buffer);
  D.Buffer = Buffer;
  VReport(1, "%s: serving reports on GPU %zu, %u ports, %u lanes\n",
          SanitizerToolName, Ordinal, D.Ports, D.Lanes);
  return true;
}

}  // namespace

void FlushRpc() {
  Lock L(&RpcMutex);
  if (Offload)
    return;
  drainAll();
}

void StartRpc(hsa_executable_t Exec) {
  Lock Rpc(&RpcMutex);
  if (!GetHsa().Ready() || atomic_load_relaxed(&Stop))
    return;
  // Retry each freeze: libomptarget may appear after a HIP module already
  // loaded. Once it is there it owns the server; do not write a second client.
  // Register may call handleReport, which takes DasanMutex, so not yet.
  if (tryOffload())
    return;

  Lock Dasan(&DasanMutex);
  Hsa& H = GetHsa();
  if (!H.Ready() || atomic_load_relaxed(&Stop))
    return;
  for (uptr I = 0; I < H.Gpus.size(); ++I) {
    u64 Addr = 0;
    if (!executableNeedsRPC(Exec, H.Gpus[I], &Addr))
      continue;
    if (!initDevice(I, H.Gpus[I]) || !startThread())
      continue;
    DeviceRpc& D = Devices[I];
    rpc::Client Client(D.Ports, D.Buffer);
    if (!H.Copy(reinterpret_cast<void*>(Addr), &Client, sizeof(Client)))
      continue;
    VaPublish(Addr);
  }
}

void StopRpc() {
  void* Join = nullptr;
  {
    Lock L(&RpcMutex);
    Offload = false;
    atomic_store_relaxed(&Stop, 1);
    if (Thread) {
      if (Doorbell.handle)
        GetHsa().StoreSignal(Doorbell, 1);
      Join = Thread;
      Thread = nullptr;
    }
  }
  if (Join)
    internal_join_thread(Join);

  Lock L(&RpcMutex);
  if (Thread)
    return;
  if (Doorbell.handle) {
    GetHsa().DestroySignal(Doorbell);
    Doorbell = {};
    DoorbellValue = nullptr;
    DoorbellMailbox = nullptr;
    DoorbellEvent = 0;
  }
  for (uptr I = 0; I < Devices.size(); ++I) {
    if (Devices[I].Buffer)
      GetHsa().FreeHost(Devices[I].Buffer);
  }
  Devices.clear();
  atomic_store_relaxed(&Stop, 0);
}

}  // namespace __dasan
