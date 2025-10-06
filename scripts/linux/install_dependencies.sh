#!/bin/bash

# exit script on failure
set -ev

njobs=`grep -c '^processor' /proc/cpuinfo`

install_prefix=/usr/local

sudo apt update
sudo apt install -yq graphicsmagick-libmagick-dev-compat # we need Magick++.h so that CImg.h can load jpg files

googletest_version=1.10.0

echo "Downloading sources"
wget https://github.com/google/googletest/archive/refs/tags/release-${googletest_version}.zip

echo "Installing googletest"
unzip release-${googletest_version}.zip
rm release-${googletest_version}.zip
pushd googletest-release-${googletest_version}
mkdir releasebuild
cd releasebuild
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${install_prefix} ..
make -j${njobs} install
popd
rm -rf googletest-release-${googletest_version}
