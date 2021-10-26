add_llvm_component_library(LLVMOffloading
  Wrapper.cpp
  Bundler.cpp
  ToolChain.cpp
  NVPTXToolChain.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Offloading

  DEPENDS
  intrinsics_gen
  omp_gen

  COMPONENT_NAME
  Offloading

  LINK_COMPONENTS
  Core
  FrontendOpenMP
  Linker
  Object
  Support
  Instrumentation
  )
