set(CSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Device compiler-rt builds the GPU archive; the tests run from the host.
if(COMPILER_RT_GPU_BUILD)
  return()
endif()

set(CSAN_TESTSUITES)
set(CSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND CSAN_TEST_DEPS csan)

set(COMPILER_RT_CSAN_CAN_RUN_HIP FALSE)
set(COMPILER_RT_CSAN_CAN_RUN_OPENMP_OFFLOAD FALSE)
set(CSAN_GPU_ARCH "")
set(CSAN_HIP_LIB_DIR "")

if(LLVM_RUNTIMES_BUILD AND COMPILER_RT_TEST_COMPILER_ID MATCHES "Clang")
  find_program(CSAN_OFFLOAD_ARCH offload-arch PATHS ${LLVM_TOOLS_BINARY_DIR})
  if(CSAN_OFFLOAD_ARCH)
    execute_process(COMMAND ${CSAN_OFFLOAD_ARCH} --only=amdgpu
                    OUTPUT_VARIABLE _csan_amdgpu_archs
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REPLACE "\n" ";" _csan_amdgpu_archs "${_csan_amdgpu_archs}")
    if(_csan_amdgpu_archs)
      list(GET _csan_amdgpu_archs 0 CSAN_GPU_ARCH)
    endif()
  endif()

  if(CSAN_GPU_ARCH)
    find_library(CSAN_TEST_HIP_LIBRARY NAMES amdhip64
      HINTS "$ENV{ROCM_PATH}/lib" "$ENV{HIP_PATH}/lib" /opt/rocm/lib)
    if(CSAN_TEST_HIP_LIBRARY)
      set(COMPILER_RT_CSAN_CAN_RUN_HIP TRUE)
      get_filename_component(CSAN_HIP_LIB_DIR "${CSAN_TEST_HIP_LIBRARY}" DIRECTORY)
    endif()
    if(HAVE_OFFLOAD)
      set(COMPILER_RT_CSAN_CAN_RUN_OPENMP_OFFLOAD TRUE)
    endif()
  endif()
endif()

pythonize_bool(COMPILER_RT_CSAN_CAN_RUN_HIP)
pythonize_bool(COMPILER_RT_CSAN_CAN_RUN_OPENMP_OFFLOAD)

set(CSAN_TEST_ARCH ${UBSAN_SUPPORTED_ARCH})
foreach(arch ${CSAN_TEST_ARCH})
  set(CSAN_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} CSAN_TEST_TARGET_CC CSAN_TEST_TARGET_CFLAGS)
  set(CONFIG_NAME ${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND CSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-csan "Running ConcurrencySanitizer tests"
  ${CSAN_TESTSUITES}
  DEPENDS ${CSAN_TEST_DEPS})

function(_csan_depend_on_offload)
  if(NOT TARGET check-csan)
    return()
  endif()
  foreach(dep omp omptarget)
    if(TARGET ${dep})
      add_dependencies(check-csan ${dep})
    endif()
  endforeach()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _csan_depend_on_offload)
