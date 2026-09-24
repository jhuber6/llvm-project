include(FindLibcCommonUtils)
# The device instrumentation assumes the x86-64 Linux shadow mapping.
if(NOT OS_NAME MATCHES "Linux" OR NOT TARGET llvm-libc-common-utilities
   OR NOT x86_64 IN_LIST ASAN_SUPPORTED_ARCH)
  return()
endif()

include_directories(..)

set(ASAN_OFFLOAD_SOURCES
  asan_offload_hsa_interceptors.cpp
  asan_offload_report.cpp
  )

set(ASAN_OFFLOAD_HEADERS
  asan_offload.h
  ../asan_offload_packet.h
  )

# The ASan internals come from the host runtime this is linked against.
add_compiler_rt_runtime(clang_rt.asan_offload
  STATIC
  ARCHS x86_64
  SOURCES ${ASAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${ASAN_OFFLOAD_HEADERS}
  OBJECT_LIBS RTSanitizerOffload
  CFLAGS ${ASAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET asan)
