use std::process::Command;
use std::path::Path;

// Boot ROM generators (build-time only, not part of the runtime library).
#[path = "build_support/bootrom/mod.rs"]
mod bootrom;

/// All compute shaders, by base name. Source: src/shaders/{name}.slang
/// Output: {OUT_DIR}/{name}_comp.{spv,metal,wgsl,dxil}
const SHADERS: &[&str] = &[
    "nearest",
    "nearest_aa",
    "bilinear",
    "bicubic",
    "eagle",
    "epx",
    "scale3x",
    "lcd_grid",
    "mmpx",
    "sai2x",
    "super_sai2x",
    "super_eagle",
    "dcci",
    "nedi",
    "edi",
    "hqx",
    "xbr",
    "xbrz",
    "omniscale",
    "omniscale_legacy",
    "super_xbr",
    "similarity_graph",
    "resolve_crossings",
    "cell_graph",
    "cell_rasterizer",
    "optimize_energy",
    "update_tjunction",
    "vectorize_raster",
    "vectorize_to_buf",
    "spline_diffusion",
    "diffusion_raster",
];

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Generate boot ROMs
    generate_boot_roms(&out_dir);

    // Compile shaders when any GPU feature is enabled
    if std::env::var("CARGO_FEATURE_SDL3_GPU_SHADERS").is_err()
        && std::env::var("CARGO_FEATURE_WEB").is_err()
        && std::env::var("CARGO_FEATURE_GPU").is_err()
    {
        return;
    }

    let shader_dir = Path::new("src/shaders");
    let has_slangc = Command::new("slangc").arg("-v").output().is_ok();

    if !has_slangc {
        panic!(
            "slangc not found. Install the Slang shader compiler:\n\
             https://github.com/shader-slang/slang/releases\n\
             Add the bin/ directory to your PATH."
        );
    }

    let has_dxc = Command::new("dxc").arg("--version").output().is_ok();

    // Track shared module files so shaders recompile when modules change.
    let modules_dir = shader_dir.join("modules");
    if modules_dir.is_dir() {
        println!("cargo:rerun-if-changed={}", modules_dir.display());
        if let Ok(entries) = std::fs::read_dir(&modules_dir) {
            for entry in entries.flatten() {
                println!("cargo:rerun-if-changed={}", entry.path().display());
            }
        }
    }

    for name in SHADERS {
        let src = shader_dir.join(format!("{name}.slang"));
        let spv = Path::new(&out_dir).join(format!("{name}_comp.spv"));
        let msl = Path::new(&out_dir).join(format!("{name}_comp.metal"));
        let wgsl = Path::new(&out_dir).join(format!("{name}_comp.wgsl"));
        let dxil = Path::new(&out_dir).join(format!("{name}_comp.dxil"));

        println!("cargo:rerun-if-changed={}", src.display());

        // SPIR-V
        run_slangc(&src, "spirv", &spv);

        // Metal
        run_slangc(&src, "metal", &msl);

        // WGSL — WTexture2D emits write access directly, no post-processing needed.
        run_slangc(&src, "wgsl", &wgsl);

        // DXIL (requires dxc, only available on Windows).
        // The .slang sources have explicit register(tN,space0) / register(uN,space1) /
        // register(bN,space2) annotations matching SDL3 D3D12's type-based space grouping.
        if has_dxc {
            run_slangc(&src, "dxil", &dxil);
        } else {
            // Empty stub so include_bytes! compiles on all platforms
            let _ = std::fs::write(&dxil, b"");
        }
    }
}

/// Run slangc to compile a shader to a specific target.
fn run_slangc(src: &Path, target: &str, out: &Path) {
    let mut cmd = Command::new("slangc");
    cmd.arg(src.to_str().unwrap())
        .arg("-I")
        .arg("src/shaders") // Resolve `import modules.color;` etc.
        .arg("-target")
        .arg(target);
    // DXIL needs an explicit compute shader profile; without it slangc
    // defaults to a library target (lib_6_x) which dxc cannot validate.
    if target == "dxil" {
        cmd.arg("-profile").arg("cs_6_0");
    }
    let output = cmd.arg("-o")
        .arg(out.to_str().unwrap())
        .output();

    match output {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!(
                "slangc failed for {} -> {target}:\n{stderr}",
                src.display()
            );
        }
        Err(e) => panic!("Failed to run slangc for {}: {e}", src.display()),
    }
}

fn generate_boot_roms(out_dir: &str) {
    let boot_dir = Path::new(out_dir).join("bootroms");
    std::fs::create_dir_all(&boot_dir).unwrap();

    let cgb = bootrom::cgb_boot_rom();
    std::fs::write(boot_dir.join("cgb_boot.bin"), &cgb).unwrap();

    let agb = bootrom::agb_boot_rom();
    std::fs::write(boot_dir.join("agb_boot.bin"), &agb).unwrap();

    let dmg = bootrom::dmg_boot_rom();
    std::fs::write(boot_dir.join("dmg_boot.bin"), &dmg).unwrap();

    let mgb = bootrom::mgb_boot_rom();
    std::fs::write(boot_dir.join("mgb_boot.bin"), &mgb).unwrap();

    println!("cargo:rerun-if-changed=build_support/bootrom");
}
