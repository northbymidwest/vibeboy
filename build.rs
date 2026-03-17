use std::process::Command;
use std::path::Path;

/// Shader stage info for cross-compilation.
struct ShaderInfo {
    glsl_src: &'static str,
    spv_name: &'static str,
    msl_name: &'static str,
    /// MSL buffer index remapping: maps spirv-cross index → SDL3-expected index.
    /// spirv-cross assigns Metal buffer indices by descriptor set/binding order,
    /// but SDL3 MSL expects: uniform buffers first, then storage buffers.
    /// Each entry (from, to) means [[buffer(from)]] → [[buffer(to)]].
    msl_buffer_remap: &'static [(u32, u32)],
}

const SHADERS: &[ShaderInfo] = &[
    ShaderInfo {
        glsl_src: "fullscreen.vert",
        spv_name: "fullscreen_vert.spv",
        msl_name: "fullscreen_vert.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "omniscale.frag",
        spv_name: "omniscale_frag.spv",
        msl_name: "omniscale_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "hqx.frag",
        spv_name: "hqx_frag.spv",
        msl_name: "hqx_frag.metal",
        // spirv-cross already generates correct order for fragment:
        //   buffer(0)=uniform, buffer(1)=storage — matches SDL3 MSL convention
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "bicubic.frag",
        spv_name: "bicubic_frag.spv",
        msl_name: "bicubic_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "omniscale_legacy.frag",
        spv_name: "omniscale_legacy_frag.spv",
        msl_name: "omniscale_legacy_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "scale3x.frag",
        spv_name: "scale3x_frag.spv",
        msl_name: "scale3x_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "eagle.frag",
        spv_name: "eagle_frag.spv",
        msl_name: "eagle_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "aa_nearest.frag",
        spv_name: "aa_nearest_frag.spv",
        msl_name: "aa_nearest_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "epx.frag",
        spv_name: "epx_frag.spv",
        msl_name: "epx_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "xbr.frag",
        spv_name: "xbr_frag.spv",
        msl_name: "xbr_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "xbrz.frag",
        spv_name: "xbrz_frag.spv",
        msl_name: "xbrz_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "super_xbr.frag",
        spv_name: "super_xbr_frag.spv",
        msl_name: "super_xbr_frag.metal",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "vectorize_raster.comp",
        spv_name: "vectorize_raster_comp.spv",
        msl_name: "vectorize_raster_comp.metal",
        // spirv-cross generates: buffer(0)=edges, buffer(1)=rows, buffer(2)=uniforms, buffer(3)=indices
        // SDL3 MSL compute expects: uniform buffers first, then readonly storage buffers:
        //   buffer(0)=uniforms, buffer(1)=edges, buffer(2)=rows, buffer(3)=indices
        msl_buffer_remap: &[(0, 1), (1, 2), (2, 0)],
    },
    ShaderInfo {
        glsl_src: "diffusion_raster.comp",
        spv_name: "diffusion_raster_comp.spv",
        msl_name: "diffusion_raster_comp.metal",
        // spirv-cross generates: buffer(0)=diags, buffer(1)=uniforms, buffer(2)=regions, buffer(3)=pixels
        // SDL3 MSL compute expects: buffer(0)=uniforms, buffer(1)=pixels, buffer(2)=regions, buffer(3)=diags
        msl_buffer_remap: &[(0, 3), (1, 0), (3, 1)],
    },
    ShaderInfo {
        glsl_src: "vectorize_to_buf.comp",
        spv_name: "vectorize_to_buf_comp.spv",
        msl_name: "vectorize_to_buf_comp.metal",
        // spirv-cross: buffer(0)=edges, buffer(1)=rows, buffer(2)=uniforms, buffer(3)=indices, buffer(4)=output
        // SDL3 expects: buffer(0)=uniforms, buffer(1)=edges, buffer(2)=rows, buffer(3)=indices, buffer(4)=output
        msl_buffer_remap: &[(0, 1), (1, 2), (2, 0)],
    },
    ShaderInfo {
        glsl_src: "spline_diffusion.comp",
        spv_name: "spline_diffusion_comp.spv",
        msl_name: "spline_diffusion_comp.metal",
        // spirv-cross: buffer(0)=uniforms, buffer(1)=region_colors, buffer(2)=pixels
        // SDL3 expects: buffer(0)=uniforms, buffer(1)=pixels, buffer(2)=region_colors
        msl_buffer_remap: &[(1, 2), (2, 1)],
    },
    ShaderInfo {
        glsl_src: "edge_raster.comp",
        spv_name: "edge_raster_comp.spv",
        msl_name: "edge_raster_comp.metal",
        // spirv-cross: buffer(0)=edges, buffer(1)=uniforms, buffer(2)=nn_colors,
        //              buffer(3)=grid_offsets, buffer(4)=grid_data
        // SDL3 MSL compute: buffer(0)=uniforms, buffer(1)=edges, buffer(2)=grid_data,
        //                    buffer(3)=grid_offsets, buffer(4)=nn_colors
        msl_buffer_remap: &[(0, 1), (1, 0), (2, 4), (4, 2)],
    },
];

fn main() {
    // Only compile shaders when the GPU shader feature is enabled
    if std::env::var("CARGO_FEATURE_SDL3_GPU_SHADERS").is_err() {
        return;
    }

    let shader_dir = Path::new("src/shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let glsl_compiler = find_glsl_compiler();
    let has_spirv_cross = Command::new("spirv-cross").arg("--version").output().is_ok();

    for shader in SHADERS {
        let src_path = shader_dir.join(shader.glsl_src);
        let spv_path = Path::new(&out_dir).join(shader.spv_name);
        let msl_path = Path::new(&out_dir).join(shader.msl_name);

        println!("cargo:rerun-if-changed={}", src_path.display());

        // Step 1: GLSL → SPIR-V
        match &glsl_compiler {
            Some((cmd, is_glslc)) => {
                let status = if *is_glslc {
                    Command::new(cmd)
                        .args(["-O", "--target-env=vulkan1.0"])
                        .arg(src_path.to_str().unwrap())
                        .arg("-o")
                        .arg(spv_path.to_str().unwrap())
                        .status()
                } else {
                    Command::new(cmd)
                        .args(["-V", "--target-env", "vulkan1.0"])
                        .arg(src_path.to_str().unwrap())
                        .arg("-o")
                        .arg(spv_path.to_str().unwrap())
                        .status()
                };
                match status {
                    Ok(s) if s.success() => {}
                    Ok(s) => panic!("Shader compilation failed for {} (exit code: {s})", shader.glsl_src),
                    Err(e) => panic!("Failed to run shader compiler for {}: {e}", shader.glsl_src),
                }
            }
            None => {
                panic!(
                    "No GLSL compiler found. Install the Vulkan SDK (glslc) or glslangValidator.\n\
                     On macOS: brew install shaderc\n\
                     On Linux: apt install glslc\n\
                     On Windows: install LunarG Vulkan SDK"
                );
            }
        }

        // Step 2: SPIR-V → MSL via spirv-cross (for Metal backends)
        if has_spirv_cross {
            let output = Command::new("spirv-cross")
                .arg("--msl")
                .arg("--msl-version")
                .arg("20100")
                .arg(spv_path.to_str().unwrap())
                .output()
                .expect("spirv-cross failed");

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("spirv-cross MSL failed for {}: {stderr}", shader.spv_name);
            }

            let mut msl = String::from_utf8(output.stdout)
                .expect("spirv-cross produced non-UTF8 MSL");

            // Remap buffer indices for SDL3 MSL compatibility
            if !shader.msl_buffer_remap.is_empty() {
                msl = remap_msl_buffer_indices(&msl, shader.msl_buffer_remap);
            }

            std::fs::write(&msl_path, msl.as_bytes())
                .unwrap_or_else(|e| panic!("Failed to write {}: {e}", msl_path.display()));
        } else {
            eprintln!(
                "cargo:warning=spirv-cross not found, MSL shaders will not be generated. \
                 Install with: brew install spirv-cross"
            );
        }
    }
}

/// Remap MSL [[buffer(N)]] indices so resources match SDL3's expected layout.
/// Each entry (from, to) replaces [[buffer(from)]] with [[buffer(to)]].
/// All replacements happen simultaneously (via placeholders) to avoid conflicts.
fn remap_msl_buffer_indices(msl: &str, remap: &[(u32, u32)]) -> String {
    let mut result = msl.to_string();
    // Phase 1: replace all source indices with placeholders
    for &(from, _) in remap {
        let tag = format!("[[buffer({from})]]");
        let placeholder = format!("[[buffer(__REMAP_{from}__)]]");
        result = result.replace(&tag, &placeholder);
    }
    // Phase 2: replace placeholders with target indices
    for &(from, to) in remap {
        let placeholder = format!("[[buffer(__REMAP_{from}__)]]");
        let tag = format!("[[buffer({to})]]");
        result = result.replace(&placeholder, &tag);
    }
    result
}

fn find_glsl_compiler() -> Option<(String, bool)> {
    if Command::new("glslc").arg("--version").output().is_ok() {
        return Some(("glslc".to_string(), true));
    }
    if Command::new("glslangValidator").arg("--version").output().is_ok() {
        return Some(("glslangValidator".to_string(), false));
    }
    None
}
