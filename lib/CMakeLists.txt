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

if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-string-conversion -Wno-covered-switch-default")
  endif()
endif()

add_subdirectory(burnside)
add_subdirectory(common)
add_subdirectory(decimal)
add_subdirectory(evaluate)
add_subdirectory(fir)
add_subdirectory(parser)
add_subdirectory(semantics)
