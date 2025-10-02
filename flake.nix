{
  description = "C++ OpenCL-Vulkan dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      platform = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${platform};
    in
    {
      devShells.${platform}.default = pkgs.mkShell {
        name = "c++-opencl-vulkan-dev-shell";

        buildInputs = with pkgs; [
          gcc
          cmake
          clinfo
          ocl-icd
          vulkan-tools
          vulkan-headers
          vulkan-loader
          vulkan-memory-allocator
          vulkan-validation-layers
          shaderc
          gtest
          pkg-config
          xorg.libX11
          xorg.libXrandr
          xorg.libXinerama
          xorg.libXcursor
          xorg.libXi
        ];

        VULKAN_SDK = "${pkgs.vulkan-loader}";
        VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
          pkgs.ocl-icd
          pkgs.vulkan-loader
          pkgs.xorg.libX11
          pkgs.xorg.libXrandr
          pkgs.xorg.libXinerama
          pkgs.xorg.libXcursor
          pkgs.xorg.libXi
        ]}";

        CMAKE_INCLUDE_PATH = "${pkgs.vulkan-memory-allocator}/include";

        CMAKE_EXPORT_COMPILE_COMMANDS = "ON";

        shellHook = ''
          echo "C++ OpenCL-Vulkan dev shell started. OpenCL platforms:"
          clinfo -l
          echo "Vulkan info:"
          vulkaninfo --summary
        '';
      };
    };
}
