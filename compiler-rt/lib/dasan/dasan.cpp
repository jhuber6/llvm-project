//===-- dasan.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Startup and flags.
//
//===----------------------------------------------------------------------===//

#include "dasan.h"

#include <pthread.h>

#include "dasan_flags.h"
#include "dasan_image.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_mutex.h"

using namespace __sanitizer;

namespace __sanitizer {

int internal_pthread_create(void* Th, void* Attr, void* (*Callback)(void*),
                            void* Param) {
  return pthread_create(reinterpret_cast<pthread_t*>(Th),
                        reinterpret_cast<const pthread_attr_t*>(Attr), Callback,
                        Param);
}

int internal_pthread_join(void* Th, void** Ret) {
  return pthread_join(reinterpret_cast<pthread_t>(Th), Ret);
}

}  // namespace __sanitizer

namespace __dasan {

Mutex DasanMutex;
Flags DasanFlags;

void Flags::SetDefaults() {
#define DASAN_FLAG(Type, Name, Default, Description) Name = Default;
#include "dasan_flags.inc"
#undef DASAN_FLAG
}

static void RegisterDasanFlags(FlagParser* Parser, Flags* F) {
#define DASAN_FLAG(Type, Name, Default, Description) \
  RegisterFlag(Parser, #Name, Description, &F->Name);
#include "dasan_flags.inc"
#undef DASAN_FLAG
}

void InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags Cf;
    Cf.CopyFrom(*common_flags());
    Cf.external_symbolizer_path = GetEnv("DASAN_SYMBOLIZER_PATH");
    OverrideCommonFlags(Cf);
  }

  Flags* F = flags();
  F->SetDefaults();

  FlagParser Parser;
  RegisterDasanFlags(&Parser, F);
  RegisterCommonFlags(&Parser);

  Parser.ParseString(__dasan_default_options());
  Parser.ParseStringFromEnv("DASAN_OPTIONS");

  InitializeCommonFlags();

  if (Verbosity())
    ReportUnrecognizedFlags();

  if (common_flags()->help)
    Parser.PrintFlagDescriptions();
}

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;

  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;

  SanitizerToolName = "DeviceAddressSanitizer";
  InitializeFlags();

  // Need to clean up any temp files before exit.
  AddDieCallback(ForgetImages);

  VReport(1, "%s: initialized\n", SanitizerToolName);
  atomic_store(&Initialized, 1, memory_order_release);
}

}  // namespace __dasan

SANITIZER_INTERFACE_WEAK_DEF(const char*, __dasan_default_options, void) {
  return "";
}

extern "C" void __dasan_init() { __dasan::Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"),
               used)) static void (*preinit)(void) = __dasan_init;
#endif

__attribute__((constructor(0))) static void DasanDynInit() { __dasan_init(); }
