add_library(EvaluateObjects OBJECT
  call.cpp
  characteristics.cpp
  check-expression.cpp
  common.cpp
  complex.cpp
  constant.cpp
  expression.cpp
  fold.cpp
  fold-character.cpp
  fold-complex.cpp
  fold-integer.cpp
  fold-logical.cpp
  fold-real.cpp
  formatting.cpp
  host.cpp
  integer.cpp
  intrinsics.cpp
  intrinsics-library.cpp
  logical.cpp
  real.cpp
  shape.cpp
  static-data.cpp
  tools.cpp
  type.cpp
  variable.cpp
)

target_compile_features(EvaluateObjects PUBLIC cxx_std_17)
