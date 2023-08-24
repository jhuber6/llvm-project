//===--------------- Printf format parsing for the GPU --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc_client.h"
#include "src/__support/arg_list.h"
#include "src/stdio/gpu/parser.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

template <unsigned opcode>
LIBC_INLINE uint64_t printf_impl(::FILE *__restrict stream,
                                 const char *__restrict format,
                                 internal::ArgList &args) {
  rpc::Client::Port port = rpc::client.open<opcode>();
  if constexpr (opcode == RPC_PRINTF_TO_STREAM)
    port.send([&](rpc::Buffer *buffer) {
      buffer->data[0] = reinterpret_cast<uintptr_t>(stream);
    });

  port.send_n(format, internal::string_length(format) + 1);

  uint64_t mask = gpu::get_lane_mask();
  gpu::MicroParser<internal::ArgList> parser(format, args);
  for (gpu::Specifier cur = parser.get_next_specifier();
       gpu::ballot(mask, !parser.end(cur)); cur = parser.get_next_specifier())
    port.send_n(parser.get_pointer(cur), parser.get_size(cur));

  uint64_t ret = 0;
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<uint64_t *>(buffer->data)[0];
  });
  port.close();

  return ret;
}

LIBC_INLINE uint64_t printf_common(::FILE *__restrict stream,
                                   const char *__restrict format,
                                   internal::ArgList &args) {
  if (stream == stdout)
    return printf_impl<RPC_PRINTF_TO_STDOUT>(stdout, format, args);
  else if (stream == stderr)
    return printf_impl<RPC_PRINTF_TO_STDERR>(stderr, format, args);
  else
    return printf_impl<RPC_PRINTF_TO_STREAM>(stream, format, args);
}

} // namespace LIBC_NAMESPACE
