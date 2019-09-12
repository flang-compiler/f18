# Copyright (c) 2019, Arm.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(flang-fc1
  main.cc
  main_fc1.cc
)

set_target_properties(flang-fc1 PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)

set(COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize)

llvm_map_components_to_libnames(FLANG_FC1_LIBS ${COMPONENTS})

target_link_directories(flang-fc1
  PUBLIC
  ${LLVM_DIR}/../..)

target_link_libraries(flang-fc1
  PRIVATE
  ${FLANG_FC1_LIBS}
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangSerialization
  )

message("Flang link libraries: ${FLANG_FC1_LIBS}")
