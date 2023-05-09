//===-- Generic RPC server interface --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_LOADER_RPC_H
#define LLVM_LIBC_UTILS_GPU_LOADER_RPC_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stddef.h>

#include "src/__support/RPC/rpc.h"

static __llvm_libc::rpc::Server server;

static __llvm_libc::cpp::Atomic<uint32_t>
    lock[__llvm_libc::rpc::default_port_count] = {0};

/// Queries the RPC client at least once and performs server-side work if there
/// are any active requests.
void handle_server() {
  // Continue servicing the client until there is no work left and we return.
  for (;;) {
    auto port = server.try_open();
    if (!port)
      return;

    switch (port->get_opcode()) {
    case __llvm_libc::rpc::Opcode::PRINT_TO_STDERR: {
      uint64_t size[__llvm_libc::rpc::MAX_LANE_SIZE] = {0};
      void *dst[__llvm_libc::rpc::MAX_LANE_SIZE] = {nullptr};

      port->recv_n(dst, size, [](uint64_t size) { return new char[size]; });
      for (uint64_t i = 0; i < server.lane_size; ++i) {
        if (port->get_lane_mask() & 1ul << i) {
          fwrite(dst[i], size[i], 1, stderr);
          delete[] reinterpret_cast<uint8_t *>(dst[i]);
        }
      }
      break;
    }
    case __llvm_libc::rpc::Opcode::EXIT: {
      port->recv([](__llvm_libc::rpc::Buffer *buffer) {
        exit(reinterpret_cast<uint32_t *>(buffer->data)[0]);
      });
      break;
    }
    case __llvm_libc::rpc::Opcode::TEST_INCREMENT: {
      port->recv_and_send([](__llvm_libc::rpc::Buffer *buffer) {
        reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
      });
      break;
    }
    case __llvm_libc::rpc::Opcode::TEST_STREAM: {
      const char *test_str =
          "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy";

      const void **send_ptr = reinterpret_cast<const void **>(&test_str);
      uint64_t size = 51;
      port->send_n(send_ptr, &size);
      break;
    }
    default:
      port->recv([](__llvm_libc::rpc::Buffer *buffer) {});
    }
    port->close();
  }
}
#endif
