{
  description = "vibeboy development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # Native libraries and tools for `cargo build`. The Rust toolchain itself
      # comes from rustup, outside this shell.
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell (
          {
            packages =
              with pkgs;
              [
                pkg-config
                shader-slang
                sdl3
                # gtk-ui
                gtk4
                # winit-ui: bindgen (v4l2-sys-mit) needs libclang
                rustPlatform.bindgenHook
                # scripts/build-web.sh; also needs
                # `rustup target add wasm32-unknown-unknown`
                wasm-pack
                # scripts/fetch-pdroms.sh (build-web.sh --roms),
                # fetch-test-roms.sh and vectorize_comparison.sh
                curl
                unzip
                xxd
                # tools/*.py and scripts/*.py; Pillow for generate_icon.py.
                # On macOS the shell's nix DEVELOPER_DIR/SDKROOT break Apple's
                # /usr/bin/python3 shim.
                (python3.withPackages (ps: [ ps.pillow ]))
              ]
              ++ lib.optionals stdenv.hostPlatform.isLinux [
                alsa-lib
                systemdLibs # libudev, for gilrs
              ];
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            # A distro's ALSA config may route the default device through
            # PipeWire's plugin, which nix's alsa-lib looks for in its own
            # plugin directory. Without this, audio fails to open.
            ALSA_PLUGIN_DIR = "${pkgs.pipewire}/lib/alsa-lib";

            # wgpu and winit dlopen these at run time rather than linking them,
            # so nothing puts them on a binary's RUNPATH and a nix build cannot
            # find them. Add them to every binary linked in this shell.
            shellHook = ''
              export NIX_LDFLAGS="$NIX_LDFLAGS -rpath ${
                pkgs.lib.makeLibraryPath (
                  with pkgs;
                  [
                    vulkan-loader
                    libGL
                    libxkbcommon
                    wayland
                    libx11
                    libxcursor
                    libxi
                    libxrandr
                  ]
                )
              }"
            '';
          }
        );
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
