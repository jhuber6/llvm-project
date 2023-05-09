//===-- Loader test to check the RPC streaming interface with the loader --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/string/memory_utils/memcmp_implementations.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include "test/IntegrationTest/test.h"

extern "C" void *malloc(uint64_t);

using namespace __llvm_libc;

static void test_stream_simple() {
  void *recv_ptr;
  uint64_t recv_size;
  rpc::Client::Port port = rpc::client.open<rpc::TEST_STREAM>();
  port.recv_n(&recv_ptr, &recv_size,
              [](uint64_t size) { return malloc(size); });
  port.close();
}

TEST_MAIN(int argc, char **argv, char **envp) {
  if (gpu::get_thread_id() == 0)
    test_stream_simple();

  return 0;
}
