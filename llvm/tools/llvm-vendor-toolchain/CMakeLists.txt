set(LLVM_LINK_COMPONENTS
  Core
  FrontendOpenMP
  Linker
  Object
  Support
  Instrumentation
  Offloading
  BinaryFormat
  )

add_llvm_tool(llvm-vendor-toolchain
  ToolChain.cpp

  DEPENDS
  intrinsics_gen
  )
