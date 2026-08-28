# Development environment setup

Per-platform prerequisites for building vibeboy. Once these are in place, see
the build and test commands in [CLAUDE.md](../CLAUDE.md).

## All platforms

- **Rust 1.98 or newer**, 2024 edition.
- **SDL3 >= 3.4.** `build.rs` locates it through
  [`system-deps`](https://github.com/gdesmott/system-deps)/pkg-config, so any
  install shipping `sdl3.pc` works. The declaration lives in
  `[package.metadata.system-deps]` in `Cargo.toml`. A missing SDL3 fails the
  build with a pkg-config error naming what it looked for.
- **[Slang](https://github.com/shader-slang/slang/releases)** (`slangc` on
  PATH), required for the GPU compute shaders.

## macOS

```bash
brew install sdl3
```

Or with nix, add `sdl3.dev` to your packages. Note that nixpkgs' `sdl3` is
multi-output and installs only `out` (share/ files) by default; the `dev`
output is the one carrying `sdl3.pc`, and it pulls in the `lib` output it
points at.

## Linux

Install the distribution's SDL3 development package (`libsdl3-dev` on
Debian/Ubuntu, `sdl3-devel` on Fedora), which ships `sdl3.pc`.

## Windows

Also needs the MSVC Rust toolchain and Visual Studio Build Tools. `dxc` is
optional, for DXIL output (Direct3D 12).

The official `SDL3-devel-VC` zip ships **no** `sdl3.pc`, so use vcpkg, which
does (its port runs `vcpkg_fixup_pkgconfig`). Windows also has no system
pkg-config, so install pkgconf alongside it:

```
vcpkg install sdl3:x64-windows pkgconf:x64-windows
set PKG_CONFIG=%VCPKG_ROOT%\installed\x64-windows\tools\pkgconf\pkgconf.exe
set PKG_CONFIG_PATH=%VCPKG_ROOT%\installed\x64-windows\lib\pkgconfig
```

`SDL3.dll` (in `installed\x64-windows\bin`) must be on PATH or sit next to the
built `.exe`.

**MSYS2/MinGW** is an alternative needing no environment variables, since the
official mingw distribution ships `sdl3.pc` and pkgconf is on PATH:

```
pacman -S mingw-w64-x86_64-SDL3 mingw-w64-x86_64-pkgconf
```

It targets `x86_64-pc-windows-gnu` rather than MSVC.
