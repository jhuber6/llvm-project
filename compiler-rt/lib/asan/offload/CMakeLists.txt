include(FindLibcCommonUtils)
if(NOT TARGET llvm-libc-common-utilities)
  return()
endif()

include_directories(..)
include_directories(../..)

set(ASAN_OFFLOAD_SOURCES
  asan_offload_hsa.cpp
  asan_offload_hsa_interceptors.cpp
  asan_offload_rpc.cpp
  asan_offload_report.cpp
  asan_offload_symbolize.cpp
  )

set(ASAN_OFFLOAD_HEADERS
  hsa.h
  asan_offload.h
  asan_offload_hsa.h
  asan_offload_rpc.h
  asan_offload_symbolize.h
  ../asan_offload_packet.h
  )

# No OBJECT_LIBS: sanitizer_common already comes from the host ASan runtime.
add_compiler_rt_runtime(clang_rt.asan_offload
  STATIC
  ARCHS ${ASAN_SUPPORTED_ARCH}
  SOURCES ${ASAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${ASAN_OFFLOAD_HEADERS}
  CFLAGS ${ASAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET asan)
