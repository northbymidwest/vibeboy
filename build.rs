use std::process::Command;
use std::path::Path;

/// Shader stage info for cross-compilation.
struct ShaderInfo {
    glsl_src: &'static str,
    spv_name: &'static str,
    msl_name: &'static str,
    dxil_name: &'static str,
    wgsl_name: &'static str,
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
        dxil_name: "fullscreen_vert.dxil",
        wgsl_name: "fullscreen_vert.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "omniscale.frag",
        spv_name: "omniscale_frag.spv",
        msl_name: "omniscale_frag.metal",
        dxil_name: "omniscale_frag.dxil",
        wgsl_name: "omniscale_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "hqx.frag",
        spv_name: "hqx_frag.spv",
        msl_name: "hqx_frag.metal",
        dxil_name: "hqx_frag.dxil",
        wgsl_name: "hqx_frag.wgsl",
        // spirv-cross already generates correct order for fragment:
        //   buffer(0)=uniform, buffer(1)=storage — matches SDL3 MSL convention
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "bicubic.frag",
        spv_name: "bicubic_frag.spv",
        msl_name: "bicubic_frag.metal",
        dxil_name: "bicubic_frag.dxil",
        wgsl_name: "bicubic_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "omniscale_legacy.frag",
        spv_name: "omniscale_legacy_frag.spv",
        msl_name: "omniscale_legacy_frag.metal",
        dxil_name: "omniscale_legacy_frag.dxil",
        wgsl_name: "omniscale_legacy_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "scale3x.frag",
        spv_name: "scale3x_frag.spv",
        msl_name: "scale3x_frag.metal",
        dxil_name: "scale3x_frag.dxil",
        wgsl_name: "scale3x_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "eagle.frag",
        spv_name: "eagle_frag.spv",
        msl_name: "eagle_frag.metal",
        dxil_name: "eagle_frag.dxil",
        wgsl_name: "eagle_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "aa_nearest.frag",
        spv_name: "aa_nearest_frag.spv",
        msl_name: "aa_nearest_frag.metal",
        dxil_name: "aa_nearest_frag.dxil",
        wgsl_name: "aa_nearest_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "epx.frag",
        spv_name: "epx_frag.spv",
        msl_name: "epx_frag.metal",
        dxil_name: "epx_frag.dxil",
        wgsl_name: "epx_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "xbr.frag",
        spv_name: "xbr_frag.spv",
        msl_name: "xbr_frag.metal",
        dxil_name: "xbr_frag.dxil",
        wgsl_name: "xbr_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "xbrz.frag",
        spv_name: "xbrz_frag.spv",
        msl_name: "xbrz_frag.metal",
        dxil_name: "xbrz_frag.dxil",
        wgsl_name: "xbrz_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "super_xbr.frag",
        spv_name: "super_xbr_frag.spv",
        msl_name: "super_xbr_frag.metal",
        dxil_name: "super_xbr_frag.dxil",
        wgsl_name: "super_xbr_frag.wgsl",
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "vectorize_raster.comp",
        spv_name: "vectorize_raster_comp.spv",
        msl_name: "vectorize_raster_comp.metal",
        dxil_name: "vectorize_raster_comp.dxil",
        wgsl_name: "vectorize_raster_comp.wgsl",
        // spirv-cross generates: buffer(0)=edges, buffer(1)=rows, buffer(2)=uniforms, buffer(3)=indices
        // SDL3 MSL compute expects: uniform buffers first, then readonly storage buffers:
        //   buffer(0)=uniforms, buffer(1)=edges, buffer(2)=rows, buffer(3)=indices
        msl_buffer_remap: &[(0, 1), (1, 2), (2, 0)],
    },
    ShaderInfo {
        glsl_src: "diffusion_raster.comp",
        spv_name: "diffusion_raster_comp.spv",
        msl_name: "diffusion_raster_comp.metal",
        dxil_name: "diffusion_raster_comp.dxil",
        wgsl_name: "diffusion_raster_comp.wgsl",
        // spirv-cross generates: buffer(0)=diags, buffer(1)=uniforms, buffer(2)=regions, buffer(3)=pixels
        // SDL3 MSL compute expects: buffer(0)=uniforms, buffer(1)=pixels, buffer(2)=regions, buffer(3)=diags
        msl_buffer_remap: &[(0, 3), (1, 0), (3, 1)],
    },
    ShaderInfo {
        glsl_src: "vectorize_to_buf.comp",
        spv_name: "vectorize_to_buf_comp.spv",
        msl_name: "vectorize_to_buf_comp.metal",
        dxil_name: "vectorize_to_buf_comp.dxil",
        wgsl_name: "vectorize_to_buf_comp.wgsl",
        // spirv-cross: buffer(0)=edges, buffer(1)=rows, buffer(2)=uniforms, buffer(3)=indices, buffer(4)=output
        // SDL3 expects: buffer(0)=uniforms, buffer(1)=edges, buffer(2)=rows, buffer(3)=indices, buffer(4)=output
        msl_buffer_remap: &[(0, 1), (1, 2), (2, 0)],
    },
    ShaderInfo {
        glsl_src: "spline_diffusion.comp",
        spv_name: "spline_diffusion_comp.spv",
        msl_name: "spline_diffusion_comp.metal",
        dxil_name: "spline_diffusion_comp.dxil",
        wgsl_name: "spline_diffusion_comp.wgsl",
        // spirv-cross: buffer(0)=uniforms, buffer(1)=region_colors, buffer(2)=pixels
        // SDL3 expects: buffer(0)=uniforms, buffer(1)=pixels, buffer(2)=region_colors
        msl_buffer_remap: &[(1, 2), (2, 1)],
    },
    ShaderInfo {
        glsl_src: "cell_rasterizer.comp",
        spv_name: "cell_rasterizer_comp.spv",
        msl_name: "cell_rasterizer_comp.metal",
        dxil_name: "cell_rasterizer_comp.dxil",
        wgsl_name: "cell_rasterizer_comp.wgsl",
        // Raw: 0=cp_positions, 1=orig_positions, 2=uniforms, 3=flags, 4=neighbors, 5=edge_colors, 6=pixels
        // SDL3: 0=uniforms, 1=pixels, 2=cp_positions, 3=orig_positions, 4=flags, 5=neighbors, 6=edge_colors
        msl_buffer_remap: &[(0,2),(1,3),(2,0),(3,4),(4,5),(5,6),(6,1)],
    },
    ShaderInfo {
        glsl_src: "similarity_graph.comp",
        spv_name: "similarity_graph_comp.spv",
        msl_name: "similarity_graph_comp.metal",
        dxil_name: "similarity_graph_comp.dxil",
        wgsl_name: "similarity_graph_comp.wgsl",
        // spirv-cross: buffer(0)=pixels, buffer(1)=uniforms, buffer(2)=graph
        // SDL3: buffer(0)=uniforms, buffer(1)=pixels(ro), buffer(2)=graph(rw)
        msl_buffer_remap: &[(0, 1), (1, 0)],
    },
    ShaderInfo {
        glsl_src: "resolve_crossings.comp",
        spv_name: "resolve_crossings_comp.spv",
        msl_name: "resolve_crossings_comp.metal",
        dxil_name: "resolve_crossings_comp.dxil",
        wgsl_name: "resolve_crossings_comp.wgsl",
        // spirv-cross: buffer(0)=graph_in(ro), buffer(1)=uniforms, buffer(2)=graph_out(rw)
        // SDL3: buffer(0)=uniforms, buffer(1)=graph_in(ro), buffer(2)=graph_out(rw)
        msl_buffer_remap: &[(0, 1), (1, 0)],
    },
    ShaderInfo {
        glsl_src: "cell_graph.comp",
        spv_name: "cell_graph_comp.spv",
        msl_name: "cell_graph_comp.metal",
        dxil_name: "cell_graph_comp.dxil",
        wgsl_name: "cell_graph_comp.wgsl",
        // spirv-cross: buffer(0)=uniforms, buffer(1)=graph, buffer(2)=pos, buffer(3)=nbr, buffer(4)=flags
        // Already correct — no remap needed
        msl_buffer_remap: &[],
    },
    ShaderInfo {
        glsl_src: "optimize_energy.comp",
        spv_name: "optimize_energy_comp.spv",
        msl_name: "optimize_energy_comp.metal",
        dxil_name: "optimize_energy_comp.dxil",
        wgsl_name: "optimize_energy_comp.wgsl",
        // spirv-cross: buffer(0)=pos_in, buffer(1)=flags, buffer(2)=orig,
        //              buffer(3)=uniforms, buffer(4)=pos_out, buffer(5)=neighbors
        // SDL3: buffer(0)=uniforms, buffer(1)=pos_in, buffer(2)=orig,
        //        buffer(3)=neighbors, buffer(4)=flags, buffer(5)=pos_out
        msl_buffer_remap: &[(0, 1), (1, 4), (3, 0), (4, 5), (5, 3)],
    },
    ShaderInfo {
        glsl_src: "update_tjunction.comp",
        spv_name: "update_tjunction_comp.spv",
        msl_name: "update_tjunction_comp.metal",
        dxil_name: "update_tjunction_comp.dxil",
        wgsl_name: "update_tjunction_comp.wgsl",
        // Raw: 0=uniforms, 1=flags, 2=neighbors, 3=positions
        // SDL3: 0=uniforms, 1=neighbors, 2=flags, 3=positions
        msl_buffer_remap: &[(1, 2), (2, 1)],
    },
];

fn main() {
    // Compile shaders when GPU features are enabled
    if std::env::var("CARGO_FEATURE_SDL3_GPU_SHADERS").is_err()
        && std::env::var("CARGO_FEATURE_WEB").is_err()
    {
        return;
    }

    let shader_dir = Path::new("src/shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let glsl_compiler = find_glsl_compiler();
    let has_spirv_cross = Command::new("spirv-cross").arg("--version").output().is_ok();
    let has_dxc = Command::new("dxc").arg("--version").output().is_ok();

    for shader in SHADERS {
        let src_path = shader_dir.join(shader.glsl_src);
        let spv_path = Path::new(&out_dir).join(shader.spv_name);
        let msl_path = Path::new(&out_dir).join(shader.msl_name);
        let dxil_path = Path::new(&out_dir).join(shader.dxil_name);

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

        // Step 3: SPIR-V → HLSL → DXIL via spirv-cross + dxc (for Direct3D 12 backends)
        if has_spirv_cross && has_dxc {
            let hlsl_path = Path::new(&out_dir).join(
                shader.dxil_name.replace(".dxil", ".hlsl")
            );

            let output = Command::new("spirv-cross")
                .arg("--hlsl")
                .arg("--shader-model")
                .arg("60")
                .arg(spv_path.to_str().unwrap())
                .arg("--output")
                .arg(hlsl_path.to_str().unwrap())
                .output()
                .expect("spirv-cross HLSL failed");

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("cargo:warning=spirv-cross HLSL failed for {}: {stderr}", shader.spv_name);
                // Write empty stub so include_bytes! compiles
                let _ = std::fs::write(&dxil_path, b"");
                continue;
            }

            // Remap HLSL register spaces to match SDL3 D3D12 conventions:
            //   t[n] (SRVs: textures, readonly storage) → space0
            //   s[n] (samplers)                         → space0
            //   u[n] (UAVs: readwrite storage)          → space1
            //   b[n] (CBVs: uniform/constant buffers)   → space2 (compute)
            //                                             space3 (vertex/fragment — after sampler space)
            let is_compute = shader.glsl_src.ends_with(".comp");
            let hlsl = std::fs::read_to_string(&hlsl_path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", hlsl_path.display()));
            let hlsl = remap_hlsl_registers(&hlsl, is_compute);
            std::fs::write(&hlsl_path, hlsl.as_bytes())
                .unwrap_or_else(|e| panic!("Failed to write {}: {e}", hlsl_path.display()));

            // Determine DXC profile from shader stage
            let dxc_profile = if shader.glsl_src.ends_with(".vert") {
                "vs_6_0"
            } else if shader.glsl_src.ends_with(".frag") {
                "ps_6_0"
            } else {
                "cs_6_0"
            };

            let status = Command::new("dxc")
                .arg("-T").arg(dxc_profile)
                .arg("-E").arg("main")
                .arg(hlsl_path.to_str().unwrap())
                .arg("-Fo").arg(dxil_path.to_str().unwrap())
                .status();

            match status {
                Ok(s) if s.success() => {}
                _ => {
                    // Write empty stub so include_bytes! compiles
                    let _ = std::fs::write(&dxil_path, b"");
                }
            }
        } else {
            // Write empty stub so include_bytes! compiles on all platforms
            let _ = std::fs::write(&dxil_path, b"");
            if cfg!(target_os = "windows") && !has_dxc {
                eprintln!(
                    "cargo:warning=dxc not found, DXIL shaders will not be generated. \
                     Install the LunarG Vulkan SDK for dxc."
                );
            }
        }

        // Step 4: SPIR-V → WGSL via naga (for wgpu/WebGPU backends)
        let wgsl_path = Path::new(&out_dir).join(shader.wgsl_name);
        if spv_path.exists() {
            let spv_bytes = std::fs::read(&spv_path).unwrap_or_default();
            if !spv_bytes.is_empty() {
                match naga_spv_to_wgsl(&spv_bytes) {
                    Ok(wgsl) => { let _ = std::fs::write(&wgsl_path, wgsl); }
                    Err(e) => {
                        eprintln!("cargo:warning=naga WGSL conversion failed for {}: {e}", shader.spv_name);
                        let _ = std::fs::write(&wgsl_path, "");
                    }
                }
            } else {
                let _ = std::fs::write(&wgsl_path, "");
            }
        } else {
            let _ = std::fs::write(&wgsl_path, "");
        }
    }
}

/// Convert SPIR-V bytecode to WGSL using naga.
fn naga_spv_to_wgsl(spv_bytes: &[u8]) -> Result<String, String> {
    let options = naga::front::spv::Options {
        adjust_coordinate_space: false,
        ..Default::default()
    };
    let module = naga::front::spv::parse_u8_slice(spv_bytes, &options)
        .map_err(|e| format!("SPIR-V parse error: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("Validation error: {e}"))?;
    let wgsl = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| format!("WGSL write error: {e}"))?;
    Ok(wgsl)
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

/// Remap HLSL register spaces to match SDL3 D3D12 conventions.
///
/// spirv-cross assigns HLSL register spaces based on SPIR-V descriptor sets,
/// but SDL3's D3D12 backend expects resources grouped by type:
///   - t[n], space0: SRV (sampled textures, readonly storage textures, readonly storage buffers)
///   - s[n], space0: Samplers (vertex/fragment only)
///   - u[n], space1: UAV (readwrite storage textures, readwrite storage buffers)
///   - b[n], space2: CBV for compute / space3 for vertex+fragment (uniform buffers)
///
/// Additionally, register indices within each type must be contiguous starting at 0.
fn remap_hlsl_registers(hlsl: &str, is_compute: bool) -> String {
    // Collect all register declarations and build remapping
    let mut t_regs: Vec<(u32, u32)> = Vec::new(); // (old_idx, old_space)
    let mut s_regs: Vec<(u32, u32)> = Vec::new();
    let mut u_regs: Vec<(u32, u32)> = Vec::new();
    let mut b_regs: Vec<(u32, u32)> = Vec::new();

    // Find all register(xN, spaceM) patterns
    let mut i = 0;
    while i < hlsl.len() {
        if let Some(pos) = hlsl[i..].find("register(") {
            let start = i + pos + 9; // after "register("
            if let Some(end) = hlsl[start..].find(')') {
                let content = &hlsl[start..start + end];
                // Parse "tN, spaceM" or "tN"
                let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
                if !parts.is_empty() && parts[0].len() >= 2 {
                    let reg_type = parts[0].as_bytes()[0];
                    if let Ok(reg_idx) = parts[0][1..].parse::<u32>() {
                        let space = if parts.len() > 1 && parts[1].starts_with("space") {
                            parts[1][5..].parse::<u32>().unwrap_or(0)
                        } else { 0 };
                        match reg_type {
                            b't' => t_regs.push((reg_idx, space)),
                            b's' => s_regs.push((reg_idx, space)),
                            b'u' => u_regs.push((reg_idx, space)),
                            b'b' => b_regs.push((reg_idx, space)),
                            _ => {}
                        }
                    }
                }
                i = start + end + 1;
            } else { break; }
        } else { break; }
    }

    // Deduplicate and sort by (space, index) to get stable ordering
    t_regs.sort();
    t_regs.dedup();
    s_regs.sort();
    s_regs.dedup();
    u_regs.sort();
    u_regs.dedup();
    b_regs.sort();
    b_regs.dedup();

    let cbv_space = if is_compute { 2 } else { 3 };

    // Build replacement map: old pattern → new pattern
    let mut result = hlsl.to_string();

    // Phase 1: Replace with placeholders to avoid conflicts
    for (new_idx, &(idx, space)) in t_regs.iter().enumerate() {
        let old = format!("register(t{idx}, space{space})");
        let placeholder = format!("register(__T{new_idx}__, __SPACE0__)");
        result = result.replace(&old, &placeholder);
    }
    for (new_idx, &(idx, space)) in s_regs.iter().enumerate() {
        let old = format!("register(s{idx}, space{space})");
        let placeholder = format!("register(__S{new_idx}__, __SPACE0S__)");
        result = result.replace(&old, &placeholder);
    }
    for (new_idx, &(idx, space)) in u_regs.iter().enumerate() {
        let old = format!("register(u{idx}, space{space})");
        let placeholder = format!("register(__U{new_idx}__, __SPACE1__)");
        result = result.replace(&old, &placeholder);
    }
    for (new_idx, &(idx, space)) in b_regs.iter().enumerate() {
        let old = format!("register(b{idx}, space{space})");
        let placeholder = format!("register(__B{new_idx}__, __SPACECBV__)");
        result = result.replace(&old, &placeholder);
    }

    // Phase 2: Replace placeholders with final values
    for new_idx in 0..t_regs.len() {
        let placeholder = format!("register(__T{new_idx}__, __SPACE0__)");
        let final_val = format!("register(t{new_idx}, space0)");
        result = result.replace(&placeholder, &final_val);
    }
    for new_idx in 0..s_regs.len() {
        let placeholder = format!("register(__S{new_idx}__, __SPACE0S__)");
        let final_val = format!("register(s{new_idx}, space0)");
        result = result.replace(&placeholder, &final_val);
    }
    for new_idx in 0..u_regs.len() {
        let placeholder = format!("register(__U{new_idx}__, __SPACE1__)");
        let final_val = format!("register(u{new_idx}, space1)");
        result = result.replace(&placeholder, &final_val);
    }
    for new_idx in 0..b_regs.len() {
        let placeholder = format!("register(__B{new_idx}__, __SPACECBV__)");
        let final_val = format!("register(b{new_idx}, space{cbv_space})");
        result = result.replace(&placeholder, &final_val);
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
