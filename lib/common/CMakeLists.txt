
add_library(FortranCommon
  Fortran.cpp
  Fortran-features.cpp
  default-kinds.cpp
  idioms.cpp
)

install (TARGETS FortranCommon
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
