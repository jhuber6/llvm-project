//===-- Loader test to check the RPC interface with the loader ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc_client.h"
#include "test/IntegrationTest/test.h"

using namespace __llvm_libc;

// TODO: These should be put in a common utility header.
#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
uint32_t __nvvm_read_ptx_sreg_ctaid_x();
uint32_t get_block_id() { return __nvvm_read_ptx_sreg_ctaid_x(); }
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
uint32_t __builtin_amdgcn_workitem_id_x();
uint32_t get_block_id() { return __builtin_amdgcn_workgroup_id_x(); }
#else
uint32_t get_block_id() { return 0; }
#endif

static void test_add_simple() {
  uint32_t num_additions = 1000 + 10 * get_block_id();
  uint64_t cnt = 0;
  for (uint32_t i = 0; i < num_additions; ++i) {
    rpc::Port port = rpc::client.open(rpc::TEST_INCREMENT);
    port.send_and_recv(
        [=](rpc::Buffer *buffer) {
          reinterpret_cast<uint64_t *>(buffer->data)[0] = cnt;
        },
        [&](rpc::Buffer *buffer) {
          cnt = reinterpret_cast<uint64_t *>(buffer->data)[0];
        });
    port.close();
  }
  ASSERT_TRUE(cnt == num_additions && "Incorrect sum");
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_add_simple();

  return 0;
}
