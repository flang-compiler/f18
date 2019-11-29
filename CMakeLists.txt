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

cmake_minimum_required(VERSION 3.9.0)

project(flang CXX)

cmake_policy(SET CMP0068 NEW)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimplicit-fallthrough")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdelete-non-virtual-dtor")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "{CMAKE_CXX_FLAGS_MINSIZEREL} '-DCHECK=(void)'")
  set(CMAKE_CXX_FLAGS_DEBUG      "{CMAKE_CXX_FLAGS_DEBUG} -DDEBUGF18")

   # Building shared libraries is death on performance with GCC by default
   # due to the need to preserve the right to override external entry points
   # at dynamic link time.  -fno-semantic-interposition waives that right and
   # recovers a little bit of that performance.
   if (BUILD_SHARED_LIBS AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
   endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstring-conversion")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcovered-switch-default")
endif()

# Set RPATH in every executable, overriding the default setting.
# If you set this first variable back to true (the default),
# also set the second one.
set(CMAKE_SKIP_BUILD_RPATH false)
set(CMAKE_BUILD_WITH_INSTALL_RPATH false)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" "${CMAKE_INSTALL_RPATH}")

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}" )

if( NOT DEFINED PACKAGE_VERSION )

  # Out of tree

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")

  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})

  # Find the libraries that correspond to the LLVM components
  # that we wish to use
  llvm_map_components_to_libnames(llvm_libs support core irreader)

# FIXME?? No Python bindings...

# The libdir suffix must exactly match whatever LLVM's configuration used.
set(FLANG_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. "
"Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

# Override version information to reflect early stages of flang vs.
# following LLVM's versioning info...
set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")

# If FLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if (NOT DEFINED FLANG_VERSION_MAJOR)
  set(FLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if (NOT DEFINED FLANG_VERSION_MINOR)
  set(FLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if (NOT DEFINED FLANG_VERSION_PATCHLEVEL)
  set(FLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
# Unlike PACKAGE_VERSION, FLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "Flang version: ${FLANG_VERSION}")

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/flang/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/flang/Version.inc)

if (FLANG_LINK_WITH_FIR)
  message(STATUS "Linking Flang driver with FIR and LLVM")
  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support)
  message(STATUS "Flang using LLVM libraries: ${LLVM_COMMON_LIBS}")
endif()

#set(FLANG_CXX_FLAGS "-std=c++17")

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti -fno-exceptions -Wall")
  if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -fno-semantic-interposition")
  endif ()

  # Enable -pedantic for Flang even if it's not enabled for LLVM.
  if (NOT LLVM_ENABLE_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  endif ()

  if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    if (APPLE)
      # don't add --gcc-toolchain
    else()
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${CMAKE_CXX_COMPILER}")
    endif()
  endif()
  
  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if (CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types")
  endif()

  set(MLIR_SOURCE_DIR ${LLVM_INCLUDE_DIR})
  set(MLIR_MAIN_SRC_DIR ${MLIR_SOURCE_DIR})
  set(MLIR_INCLUDE_DIR ${LLVM_INCLUDE_DIR})
  set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

else()

  # In tree
  set(MLIR_SOURCE_DIR ${LLVM_SOURCE_DIR}/projects/mlir)
  set(MLIR_MAIN_SRC_DIR ${LLVM_BINARY_DIR}/projects/mlir/include)
  set(MLIR_INCLUDE_DIR ${MLIR_SOURCE_DIR}/include)
  set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

endif()

message(STATUS "LLVM_BINARY_DIR: ${LLVM_BINARY_DIR}") # debug
message(STATUS "MLIR_SOURCE_DIR: ${MLIR_SOURCE_DIR}")
message(STATUS "MLIR_INCLUDE_DIR: ${MLIR_INCLUDE_DIR}")
message(STATUS "MLIR_MAIN_SRC_DIR: ${MLIR_MAIN_SRC_DIR}") # debug

set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX} )

# Copied from mlir
function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

# Copied from mlir
function(whole_archive_link target)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib ")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/lib/lib${LIB}.a ")
    ENDFOREACH(LIB)
  elseif(MSVC)
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "/WHOLEARCHIVE:${LIB} ")
    ENDFOREACH(LIB)
  else()
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib -Wl,--whole-archive,")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    ENDFOREACH(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if(BUILD_WITH_CLANG_LIBRARIES)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -I${BUILD_WITH_CLANG_LIBRARIES}/include/c++/v1 -DCLANG_LIBRARIES")
         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -Wl,-rpath,${BUILD_WITH_CLANG_LIBRARIES}/lib -L${BUILD_WITH_CLANG_LIBRARIES}/lib")
      else()
         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
      endif()
      if(GCC)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
      endif()
   endif()
endif()

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(BEFORE
  ${FLANG_BINARY_DIR}/include
  ${FLANG_SOURCE_DIR}/include
  ${LLVM_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIR}
  ${MLIR_MAIN_SRC_DIR}
  )

enable_testing()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(test)
add_subdirectory(tools)

set(FLANG_VERSION "${LLVM_VERSION}")
message(STATUS "FLANG version: ${FLANG_VERSION}")

configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h
  )
