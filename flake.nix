{
  inputs.fenix = {
    inputs.nixpkgs.follows = "nixpkgs";
    url = github:nix-community/fenix;
  };
  inputs.rust-manifest = {
    flake = false;
    url = "https://static.rust-lang.org/dist/2024-03-02/channel-rust-nightly.toml";
  };
  description = "index_camera_passthrough";

  outputs = { self, nixpkgs, fenix, ... } @ inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; overlays = [ fenix.overlays.default ]; };
    rust-toolchain = (pkgs.fenix.fromManifestFile inputs.rust-manifest).withComponents [
      "rustfmt"
      "rust-src"
      "clippy"
      "rustc"
      "cargo"
    ];
  in with pkgs; {
    devShells.${system}.default = mkShell {
      nativeBuildInputs = [ pkg-config cmake rust-toolchain cargo-bloat ];
      buildInputs = [ systemdLibs linuxHeaders openvr ];
      shellHook = ''
        export LD_LIBRARY_PATH="${lib.makeLibraryPath [ libglvnd vulkan-loader ]}:$LD_LIBRARY_PATH"
      '';
      LIBCLANG_PATH = lib.makeLibraryPath [ llvmPackages_17.libclang.lib ];

      BINDGEN_EXTRA_CLANG_ARGS =
      # Includes with normal include path
      (builtins.map (a: ''-I"${a}/include"'') [
        # add dev libraries here (e.g. pkgs.libvmi.dev)
        linuxHeaders
      ]) ++ [
        ''-isystem "${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${lib.versions.major pkgs.llvmPackages_latest.libclang.version}/include"''
        ''-isystem "${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}"''
        ''-isystem "${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config}"''
        ''-isystem "${glibc.dev}/include"''
      ];
    };
  };
}
