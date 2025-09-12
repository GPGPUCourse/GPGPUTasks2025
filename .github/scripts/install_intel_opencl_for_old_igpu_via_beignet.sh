#!/bin/bash

# exit script on failure
set -ev

njobs=`grep -c '^processor' /proc/cpuinfo`

# Tested on Ubuntu 24.04
# Install LLVM-3.7 (Beignet is compatible with LLVM-3.6 ad LLVM-3.7)
sudo mkdir -p /opt/llvm-3.7 && cd /opt/llvm-3.7
sudo wget https://releases.llvm.org/3.7.1/clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz
sudo tar -xJf clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz --strip-components=1

# Automatically replace line 99: (to fix Beignet build with gcc-11)
# - bool hasMD() const { return MDMap; }
# with:
# + bool hasMD() const { return MDMap.get() != nullptr; }
sudo sed -i 's/return MDMap;/return MDMap.get() != nullptr;/' /opt/llvm-3.7/include/llvm/IR/ValueMap.h

# Install other dependencies https://github.com/intel/beignet
sudo apt install -yq cmake pkg-config ocl-icd-dev libegl1-mesa-dev ocl-icd-opencl-dev libdrm-dev libxfixes-dev libxext-dev libtinfo-dev libedit-dev zlib1g-dev

# Download Beignet source code and compile it
git clone https://github.com/intel/beignet
cd beignet
mkdir build
cd build
cmake -DLLVM_INSTALL_DIR=/opt/llvm-3.7/bin/ -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" ..
make -j$njobs
sudo make install
