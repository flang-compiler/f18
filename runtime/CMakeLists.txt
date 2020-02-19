# TODO: do we need/borrow functionality similar to llvm/clang's 
# macros here? 
add_library(FortranRuntime
  ISO_Fortran_binding.cpp
  buffer.cpp
  derived-type.cpp
  descriptor.cpp
  environment.cpp
  file.cpp
  format.cpp
  io-api.cpp
  io-error.cpp
  io-stmt.cpp
  main.cpp
  memory.cpp
  stop.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  unit.cpp
)

target_compile_features(FortranRuntime PUBLIC cxx_std_17)

target_link_libraries(FortranRuntime
  FortranCommon
  FortranDecimal
)
