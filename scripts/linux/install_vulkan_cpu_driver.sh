#!/bin/bash

# exit script on failure
set -ev

sudo apt update
sudo apt install -yq mesa-vulkan-drivers vulkan-tools # includes lavapipe software driver and vulkaninfo

vulkaninfo --summary # should show:
# deviceType = PHYSICAL_DEVICE_TYPE_CPU
# deviceName = llvmpipe (LLVM 15.0.6, 256 bits)
# driverInfo = Mesa 22.2.5 (LLVM 15.0.6)
