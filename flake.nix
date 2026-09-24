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
          }
        );
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
