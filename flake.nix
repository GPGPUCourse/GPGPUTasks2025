{
  description = "GPGPU Course";

  outputs = inputs@{ flake-parts, devenv-root, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = [ "x86_64-linux" "i686-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];

      perSystem = { config, self, inputs', pkgs, lib, system, ... }: {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        devenv.shells.default = {
          name = "gpgpu";
          packages = [
            pkgs.bear pkgs.cmake pkgs.gtest
            # CUDA
            pkgs.cudatoolkit pkgs.cudaPackages.nsight_compute
            # Vulkan
            pkgs.xorg.xorgserver pkgs.glslang pkgs.shaderc pkgs.vulkan-headers pkgs.vulkan-loader pkgs.vulkan-tools pkgs.vulkan-validation-layers pkgs.vulkan-memory-allocator
          ];
          env = {
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.ocl-icd
              pkgs.vulkan-loader
              pkgs.cudatoolkit
              pkgs.linuxPackages.nvidia_x11
              # pkgs.ncurses5
            ];
            OCL_ICD_VENDORS = "/run/opengl-driver/etc/OpenCL/vendors/";
            VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
            # CUDA_PATH="${pkgs.cudatoolkit}";
          };
          languages.c.enable = true;
          languages.cplusplus.enable = true;
        };
      };
    };

  inputs = {
    devenv-root = {
      url = "file+file:///dev/null";
      flake = false;
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    devenv.url = "github:cachix/devenv";
    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
  };
  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };
}
