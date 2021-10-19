set(LLVM_LINK_COMPONENTS
  Core
  FrontendOpenMP
  Linker
  Object
  Support
  Instrumentation
  Offloading
  )

add_llvm_tool(llvm-vendor-toolchain
  ToolChain.cpp

  DEPENDS
  intrinsics_gen
  )
