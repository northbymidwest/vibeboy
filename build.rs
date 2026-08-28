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
    "scalefx",
    "similarity_graph",
    "resolve_crossings",
    "cell_graph",
    "cell_rasterizer",
    "optimize_energy",
    "picard_step",
    "gradient_correction",
    "update_tjunction",
    "crossing_pack",
];

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Generate boot ROMs
    generate_boot_roms(&out_dir);

    // Emits the SDL3 link search path; see [package.metadata.system-deps].
    if let Err(err) = system_deps::Config::new().probe() {
        eprintln!("{err}");
        std::process::exit(1);
    }

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

    // Collect all shader compilation jobs, then run in parallel.
    struct ShaderJob {
        src: std::path::PathBuf,
        target: &'static str,
        out: std::path::PathBuf,
        profile: Option<&'static str>,
        define: Option<&'static str>,
    }

    let mut jobs: Vec<ShaderJob> = Vec::new();

    for name in SHADERS {
        let src = shader_dir.join(format!("{name}.slang"));
        println!("cargo:rerun-if-changed={}", src.display());

        jobs.push(ShaderJob {
            src: src.clone(), target: "spirv",
            out: Path::new(&out_dir).join(format!("{name}_comp.spv")),
            profile: None, define: None,
        });
        jobs.push(ShaderJob {
            src: src.clone(), target: "metal",
            out: Path::new(&out_dir).join(format!("{name}_comp.metal")),
            profile: None, define: None,
        });
        jobs.push(ShaderJob {
            src: src.clone(), target: "wgsl",
            out: Path::new(&out_dir).join(format!("{name}_comp.wgsl")),
            profile: None, define: None,
        });

        // DXIL (requires dxc, only available on Windows).
        let dxil = Path::new(&out_dir).join(format!("{name}_comp.dxil"));
        if has_dxc {
            jobs.push(ShaderJob {
                src: src.clone(), target: "dxil", out: dxil,
                profile: Some("cs_6_0"), define: Some("DXIL_TARGET"),
            });
        } else {
            let _ = std::fs::write(&dxil, b"");
        }
    }

    // Run slangc invocations in parallel, limited to available CPUs to avoid
    // hitting the open file descriptor limit (~112 simultaneous processes).
    let max_parallel = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let results: Vec<_> = jobs.chunks(max_parallel).flat_map(|chunk| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunk.iter().map(|job| {
                scope.spawn(|| {
                    let mut cmd = Command::new("slangc");
                    cmd.arg(job.src.to_str().unwrap())
                        .arg("-I").arg("src/shaders")
                        .arg("-target").arg(job.target);
                    if let Some(profile) = job.profile {
                        cmd.arg("-profile").arg(profile);
                    }
                    if let Some(define) = job.define {
                        cmd.arg(format!("-D{define}"));
                    }
                    cmd.arg("-o").arg(job.out.to_str().unwrap());
                    cmd.output()
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
        })
    }).collect();

    for (job, output) in jobs.iter().zip(results) {
        match output {
            Ok(o) if !o.status.success() => {
                eprintln!("slangc failed for {} -> {}:",
                    job.src.display(), job.target);
                eprintln!("{}", String::from_utf8_lossy(&o.stderr));
                panic!("Shader compilation failed");
            }
            Err(e) => panic!("Failed to run slangc: {e}"),
            _ => {}
        }
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
