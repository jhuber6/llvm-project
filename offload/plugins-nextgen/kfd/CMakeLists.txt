include(FetchContent)

set(LIBKFD_GIT_REPOSITORY "https://github.com/jhuber6/libkfd.git" CACHE STRING
    "Git repository for the libkfd dependency of the KFD plugin.")
set(LIBKFD_GIT_TAG "master" CACHE STRING
    "Git tag/branch/commit of libkfd to build the KFD plugin against.")

# Allow users to point to an external root directory.
if(LIBKFD_ROOT)
  set(FETCHCONTENT_SOURCE_DIR_LIBKFD "${LIBKFD_ROOT}" CACHE PATH
      "Local source override for libkfd." FORCE)
endif()

FetchContent_Declare(libkfd
  GIT_REPOSITORY ${LIBKFD_GIT_REPOSITORY}
  GIT_TAG ${LIBKFD_GIT_TAG}
  EXCLUDE_FROM_ALL
  SYSTEM)

FetchContent_MakeAvailable(libkfd)

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.kfd kfd)
target_sources(omptarget.rtl.kfd PRIVATE src/rtl.cpp)
target_include_directories(omptarget.rtl.kfd PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})
target_link_libraries(omptarget.rtl.kfd PRIVATE libkfd::kfd)

message(STATUS "Building experimental KFD plugin against libkfd "
               "(${LIBKFD_GIT_REPOSITORY}@${LIBKFD_GIT_TAG})")
