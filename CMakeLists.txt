cmake_minimum_required(VERSION 3.9.0)


#
# Reminder: The normal way to set the compiler is via the CXX environment variable
#  
#   CXX=/opt/gcc-7.2.0/bin/g++ cmake .../f18  
#
# but for convenience we provide the following cmake variables to 
# use an existing gcc installation directory.
#

# Pass '-DPGI=1' to cmake to use the latest gcc installation at PGI 
if( PGI )  
  set(GCC /home/sw/thirdparty/gcc/gcc-7.3.0/linux86-64/redhat)
endif()

# Pass -DGCC=... to cmake to use a specific gcc installation 
if( GCC ) 
  set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
  set(CMAKE_CC_COMPILER "${GCC}/bin/gcc")
  set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH true)
endif()

# Reminder: Setting CMAKE_CXX_COMPILER must be done before calling project() 

project(f18 CXX)

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}" )

find_package(Clang REQUIRED CONFIG)
include_directories(${CLANG_INSTALL_PREFIX/include})
message(STATUS "Found Clang in ${CLANG_INSTALL_PREFIX}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Get names for the LLVM libraries
#
# The full list of LLVM components can be obtained with
#
#   llvm-config --components
#
# Similarly, the (static) libraries corresponding to some
# components (default is 'all') can be obtained with
#
#   llvm-config --libs --link-static [component ...]
#
# See also
#  http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
#  https://stackoverflow.com/questions/41924375/llvm-how-to-specify-all-link-libraries-as-input-to-llvm-map-components-to-libna
#  https://stackoverflow.com/questions/33948633/how-do-i-link-when-building-with-llvm-libraries
# 
#  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support target option) 

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
   set(CMAKE_CXX_FLAGS_RELEASE    "-O2 -DDEBUG")
   set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 '-DCHECK=(void)'")
   set(CMAKE_CXX_FLAGS_DEBUG      "-g -DDEBUG")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
   set(CMAKE_CXX_FLAGS_RELEASE    "-O2 -DDEBUG")
   set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 '-DCHECK=(void)'")
   set(CMAKE_CXX_FLAGS_DEBUG      "-g -DDEBUG")

endif()

set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "FLANG version: ${FLANG_VERSION}")

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(BEFORE
  ${FLANG_BINARY_DIR}/include
  ${FLANG_SOURCE_DIR}/include
  )

add_subdirectory(include/flang)
add_subdirectory(lib)
add_subdirectory(tools)

configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h)
