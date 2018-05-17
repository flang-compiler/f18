add_library(FortranBasic 
    target-info.cc
    targets/target-info-from-clang.cc
)

target_link_libraries(FortranBasic 
  clangBasic
)
