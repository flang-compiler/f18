<!--
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
-->

# f18

## Selection of the C/C++ compiler

F18 requires a C++17 compiler. As of today, the code was only tested with g++ 7.2.0 and g++ 7.3.0  

For a proper installation, we assume that the PATH and LD_LIBRARY_PATH environment variables 
are properly set to use gcc, g++ and the associated libraries.   

cmake will require that the environement variables CC and CXX are properly set (else it will 
search for use the 'cc' and 'c++' program which are likely /usr/bin/cc and /usr/bin/c++) that 
can be done now or while calling cmake 

    export CC=gcc
    export CXX=g++

## Installation of LLVM 6.0

    ############ Extract LLVM and Clang from git in current directory. 
    ############       

    ROOT=$(pwd)
    REL=release_60
   
    # To build LLVM and Clang, we only need the head of the requested branch. 
    # Remove --single-branch --depth=1 if you want access to the whole git history. 
   
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/llvm.git/       llvm
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/clang.git/      llvm/tools/clang
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/openmp.git/     llvm/projects/openmp
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/libcxx.git/     llvm/projects/libcxx
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/libcxxabi.git/  llvm/projects/libcxxabi

    # List the version of all git sub-directories. They should all match $REL
    for dir in $(find "$ROOT/llvm" -name .git) ; do 
      cd $dir/.. ; 
      printf " %-15s %s\n" "$(git rev-parse --abbrev-ref HEAD)" "$(pwd)" ; 
    done
   
    ###########  Build LLVM & CLANG in $LLM_PREFIX 
    ###########  A Debug build can take a long time and a lot of disk space
    ###########  so I recommend making a Release  build.
       
    LLVM_PREFIX=...... 
    mkdir $LLVM_PREFIX
    
    mkdir $ROOT/llvm/build
    cd  $ROOT/llvm/build 
    CC=gcc CXX+g++ Ccmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX ..
    make -j 4
    make install
   

## Installation of F18

    ######## The installation will be done in $F18_PREFIX
    ######## That directory can be equal to different to $LLVM_PREFIX
   
    F18_PREFIX=$ROOT/usr   

    ######### Add $LLVM_PREFIX/bin to PATH so that cmake finds llvm-config   

    export "PATH=$LLVM_PREFIX/bin:$PATH"

    ######## Get Flang sources 
    git clone https://github.com/ThePortlandGroup/f18.git

    ######## Create a build directory for f18 and build it 
    mkdir $ROOT/f18-build
    cd $ROOT/f18-build
    CC=gcc CXX+g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ../f18 
    make -j 4
    make install 
