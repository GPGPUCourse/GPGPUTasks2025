#!/usr/bin/env bash

set -euo pipefail

NDK=$HOME/Home/apps/android-ndk-r27d/

mkdir -p build_android
cd build_android

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-27 \
    -G Ninja

ninja
