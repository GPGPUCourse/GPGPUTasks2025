#!/bin/bash

# exit script on failure
set -ev

njobs=`sysctl -n hw.ncpu`
njobs=`expr $njobs + $njobs`

install_prefix=/opt

googletest_version=1.10.0

echo "Downloading sources"
# curl -O https://github.com/google/googletest/archive/refs/tags/release-${googletest_version}.zip
curl -o release-${googletest_version}.zip https://codeload.github.com/google/googletest/zip/refs/tags/release-${googletest_version}

# Alternatively you can install googletest simply via: "brew install googletest" - but if you do - you do it on your own risk (version incompatibility is possible)
echo "Installing googletest"
unzip release-${googletest_version}.zip
rm release-${googletest_version}.zip
pushd googletest-release-${googletest_version}
mkdir releasebuild
cd releasebuild
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${install_prefix} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j${njobs} install
popd
rm -rf googletest-release-${googletest_version}
