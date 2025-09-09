{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    ocl-icd # libOpenCL.so
  ];

  shellHook = with pkgs; ''
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ocl-icd}/lib
  '';
}
