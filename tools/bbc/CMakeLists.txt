# Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-string-conversion")

add_executable(bbc
  bbc.cc
)

set(LIBS
  FIR
  FIRTransforms
  MLIRAffineOps
  MLIRLLVMIR
  MLIRLoopOps
  MLIRStandardOps
)

whole_archive_link(bbc ${LIBS})
target_link_libraries(bbc
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranBurnside
  FIRTransforms
  ${LIBS}
)

install(TARGETS bbc DESTINATION bin)
