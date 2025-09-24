{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    cmake
    gnumake

    # Google Tests
    gtest
    xorg.libX11

    # OpenCL
    ocl-icd

    # CUDA Toolkit
    cudaPackages.cudatoolkit
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cccl
    cudaPackages.cuda_cudart

    git gitRepo gnupg autoconf curl
    procps gnumake util-linux m4 gperf unzip
    linuxPackages.nvidia_x11
    libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    ncurses5 stdenv.cc binutils

    # Vulkan SDK
    vulkan-tools
    vulkan-headers
    vulkan-loader
    vulkan-memory-allocator
    vulkan-validation-layers
    shaderc

    ninja
    pkg-config
    clang-tools
  ];

  shellHook = with pkgs; ''
    export CMAKE_EXPORT_COMPILE_COMMANDS=ON

    # OpenCL
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ocl-icd}/lib

    # CUDA
    export CUDA_PATH=${cudaPackages.cudatoolkit}
    export CUDA_HOME=${cudaPackages.cudatoolkit}
    export PATH=$PATH:${cudaPackages.cuda_nvcc}/bin
    export CMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc
    export LD_LIBRARY_PATH=${cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${linuxPackages.nvidia_x11}/lib:${ncurses5}/lib:$LD_LIBRARY_PATH

    # Vulkan
    export CMAKE_INCLUDE_PATH=$CMAKE_INCLUDE_PATH:${vulkan-memory-allocator}/include
    export VK_LAYER_PATH=${vulkan-validation-layers}/etc/vulkan/explicit_layer.d
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${vulkan-loader}/lib
    # disable non gpu Vulkan
    export VK_ICD_FILENAMES=${linuxPackages.nvidia_x11}/share/vulkan/icd.d/nvidia_icd.x86_64.json
  '';
}
