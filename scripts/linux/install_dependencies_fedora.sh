#!/bin/bash

# exit script on failure
set -ev

njobs=`grep -c '^processor' /proc/cpuinfo`

install_prefix=/usr/local

sudo dnf install ImageMagick-c++-devel # we need Magick++.h so that CImg.h can load jpg files

googletest_version=1.17.0 # 1.10.0 does not compile, lol

echo "Downloading sources"
wget https://github.com/google/googletest/archive/refs/tags/v${googletest_version}.zip # new beautiful tags naming

echo "Installing googletest"
unzip v${googletest_version}.zip
rm v${googletest_version}.zip
pushd googletest-${googletest_version}
mkdir releasebuild
cd releasebuild
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${install_prefix} ..
make -j${njobs} install
popd
rm -rf googletest-${googletest_version}
