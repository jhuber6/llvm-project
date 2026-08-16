//===-- sanitizer/gpuasan_interface.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared between the GPU address sanitizer runtime and whatever services its
// RPC opcode on the host, so the two cannot drift.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_GPUASAN_INTERFACE_H
#define SANITIZER_GPUASAN_INTERFACE_H

#include <stdint.h>

// Opcode space is keyed by a leading byte, as libc does with 'c'.
#define GPUASAN_RPC_BASE 'g'
#define GPUASAN_OPCODE(n) ((GPUASAN_RPC_BASE << 24) | (n))

typedef enum {
  GPUASAN_REPORT = GPUASAN_OPCODE(0),
} gpuasan_opcode_t;

#ifdef __cplusplus
extern "C" {
#endif

/// One failed access. Must fit in a single rpc::Buffer, which is 64 bytes.
///
/// The device sends the derived allocation rather than only the address,
/// because it is the side that already performed the derivation, and because
/// the allocation may be recycled before the host drains the report.
typedef struct gpuasan_report_t {
  uint64_t pc;         // Caller PC, for the host to symbolize.
  uint64_t addr;       // Address accessed.
  uint64_t base;       // Base of the allocation the address derives to.
  uint64_t alloc_size; // Its size.
  uint32_t access_size;
  uint32_t block[3];
  uint16_t thread[3];
  uint8_t lane;
  uint8_t is_write;
  // AMDGPU address space of the object, so the host can name the error
  // without guessing from the address. Only meaningful for the compile-time
  // bounded objects; a placed heap allocation is identified by its address
  // falling inside the reserved region and leaves this zero.
  uint8_t addr_space;
  // The metadata the check read was poisoned: the allocation it describes has
  // been freed, and `base` and `alloc_size` are the extent it used to have.
  uint8_t freed;
  // The reporter was compiled with `-fsanitize-recover=gpuasan` and will carry
  // on whatever the host answers, so the host does not promise to stop.
  uint8_t recover;
} gpuasan_report_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_GPUASAN_INTERFACE_H
