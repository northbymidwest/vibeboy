//! GPU scaling filter pipeline using wgpu (WebGPU-compatible).
//!
//! Runs pixel-art scaling filters as compute shaders:
//! EPX, Eagle, Scale3x, Bicubic, Nearest AA, OmniScale.
//!
//! All shaders read from a u32 pixel storage buffer and write to an rgba8 output texture.

use wgpu;

/// Available compute-based scaling filters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WgpuScaleFilter {
    Nearest,
    Bilinear,
    Epx,
    Eagle,
    Scale3x,
    Bicubic,
    NearestAa,
    OmniScale,
    Hqx,
    Xbr,
    Xbrz,
    SuperXbr,
    OmniScaleLegacy,
    Edi,
    Nedi,
    Dcci,
    Mmpx,
    LcdGrid,
    Sai2x,
    Super2xSai,
    SuperEagle,
    ScaleFx,
}

impl WgpuScaleFilter {
    /// Map a ScaleFilter to the corresponding WgpuScaleFilter, if one exists.
    pub fn from_scale_filter(f: super::ScaleFilter) -> Option<Self> {
        use super::ScaleFilter as SF;
        match f {
            SF::Nearest => Some(Self::Nearest),
            SF::Bilinear => Some(Self::Bilinear),
            SF::Bicubic => Some(Self::Bicubic),
            SF::Epx | SF::Scale2x | SF::Scale4x => Some(Self::Epx),
            SF::Scale3x => Some(Self::Scale3x),
            SF::Eagle => Some(Self::Eagle),
            SF::Hqx(_) => Some(Self::Hqx),
            SF::Xbr(_) | SF::SuperXbr => Some(Self::Xbr),
            SF::Xbrz(_) => Some(Self::Xbrz),
            SF::NearestAa => Some(Self::NearestAa),
            SF::OmniScale => Some(Self::OmniScale),
            SF::OmniScaleLegacy => Some(Self::OmniScaleLegacy),
            SF::Edi => Some(Self::Edi),
            SF::Nedi => Some(Self::Nedi),
            SF::Dcci => Some(Self::Dcci),
            SF::Mmpx => Some(Self::Mmpx),
            SF::LcdGrid => Some(Self::LcdGrid),
            SF::Sai2x => Some(Self::Sai2x),
            SF::Super2xSai => Some(Self::Super2xSai),
            SF::SuperEagle => Some(Self::SuperEagle),
            SF::ScaleFx | SF::ScaleFx9x => Some(Self::ScaleFx),
            _ => None,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nearest" => Some(Self::Nearest),
            "bilinear" => Some(Self::Bilinear),
            "epx" => Some(Self::Epx),
            "eagle" => Some(Self::Eagle),
            "scale3x" => Some(Self::Scale3x),
            "bicubic" => Some(Self::Bicubic),
            "nearest-aa" => Some(Self::NearestAa),
            "omniscale" => Some(Self::OmniScale),
            "hqx" => Some(Self::Hqx),
            "xbr" => Some(Self::Xbr),
            "xbrz" => Some(Self::Xbrz),
            "super-xbr" => Some(Self::SuperXbr),
            "omniscale-legacy" => Some(Self::OmniScaleLegacy),
            "edi" => Some(Self::Edi),
            "nedi" => Some(Self::Nedi),
            "dcci" => Some(Self::Dcci),
            "mmpx" => Some(Self::Mmpx),
            "lcd-grid" => Some(Self::LcdGrid),
            "2xsai" => Some(Self::Sai2x),
            "super-2xsai" => Some(Self::Super2xSai),
            "super-eagle" => Some(Self::SuperEagle),
            "scalefx" => Some(Self::ScaleFx),
            _ => None,
        }
    }

    pub fn all_names() -> &'static [&'static str] {
        &["nearest", "bilinear", "epx", "eagle", "scale3x", "bicubic", "nearest-aa",
          "omniscale", "hqx", "xbr", "xbrz", "super-xbr", "omniscale-legacy",
          "edi", "nedi", "dcci", "mmpx", "lcd-grid",
          "2xsai", "super-2xsai", "super-eagle"]
    }

    /// For integer-scale filters, compute the native output dimensions.
    /// Returns None for resolution-independent filters (use requested size).
    /// Compute output dimensions. If `factor` is non-zero, use it directly
    /// (from ScaleFilter::factor()). Otherwise use the requested size.
    pub fn native_size(&self, src_w: u32, src_h: u32, out_w: u32, out_h: u32, factor: u32) -> (u32, u32) {
        if factor > 0 {
            return (src_w * factor, src_h * factor);
        }
        // Resolution-independent — use requested size
        (out_w, out_h)
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    wgsl: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let module = unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Cached wgpu compute pipelines and buffers for scaling filters.
pub struct WgpuScalePipeline {
    nearest: wgpu::ComputePipeline,
    bilinear: wgpu::ComputePipeline,
    epx: wgpu::ComputePipeline,
    eagle: wgpu::ComputePipeline,
    scale3x: wgpu::ComputePipeline,
    bicubic: wgpu::ComputePipeline,
    nearest_aa: wgpu::ComputePipeline,
    omniscale: wgpu::ComputePipeline,
    hqx: wgpu::ComputePipeline,
    xbr: wgpu::ComputePipeline,
    xbrz: wgpu::ComputePipeline,
    super_xbr: wgpu::ComputePipeline,
    omniscale_legacy: wgpu::ComputePipeline,
    edi: wgpu::ComputePipeline,
    nedi: wgpu::ComputePipeline,
    dcci: wgpu::ComputePipeline,
    mmpx: wgpu::ComputePipeline,
    lcd_grid: wgpu::ComputePipeline,
    sai2x: wgpu::ComputePipeline,
    super_sai2x: wgpu::ComputePipeline,
    super_eagle: wgpu::ComputePipeline,
    scalefx: wgpu::ComputePipeline,
    bufs: Option<ScaleBufs>,
}

struct ScaleBufs {
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    px_buf: wgpu::Buffer,
    uni_buf: wgpu::Buffer,
    output_tex: wgpu::Texture,
}

impl WgpuScalePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let nearest_wgsl = include_str!(concat!(env!("OUT_DIR"), "/nearest_comp.wgsl"));
        let bilinear_wgsl = include_str!(concat!(env!("OUT_DIR"), "/bilinear_comp.wgsl"));
        let epx_wgsl = include_str!(concat!(env!("OUT_DIR"), "/epx_comp.wgsl"));
        let eagle_wgsl = include_str!(concat!(env!("OUT_DIR"), "/eagle_comp.wgsl"));
        let scale3x_wgsl = include_str!(concat!(env!("OUT_DIR"), "/scale3x_comp.wgsl"));
        let bicubic_wgsl = include_str!(concat!(env!("OUT_DIR"), "/bicubic_comp.wgsl"));
        let nearest_aa_wgsl = include_str!(concat!(env!("OUT_DIR"), "/nearest_aa_comp.wgsl"));
        let omniscale_wgsl = include_str!(concat!(env!("OUT_DIR"), "/omniscale_comp.wgsl"));
        let hqx_wgsl = include_str!(concat!(env!("OUT_DIR"), "/hqx_comp.wgsl"));
        let xbr_wgsl = include_str!(concat!(env!("OUT_DIR"), "/xbr_comp.wgsl"));
        let xbrz_wgsl = include_str!(concat!(env!("OUT_DIR"), "/xbrz_comp.wgsl"));
        let super_xbr_wgsl = include_str!(concat!(env!("OUT_DIR"), "/super_xbr_comp.wgsl"));
        let omniscale_legacy_wgsl = include_str!(concat!(env!("OUT_DIR"), "/omniscale_legacy_comp.wgsl"));
        let edi_wgsl = include_str!(concat!(env!("OUT_DIR"), "/edi_comp.wgsl"));
        let nedi_wgsl = include_str!(concat!(env!("OUT_DIR"), "/nedi_comp.wgsl"));
        let dcci_wgsl = include_str!(concat!(env!("OUT_DIR"), "/dcci_comp.wgsl"));
        let mmpx_wgsl = include_str!(concat!(env!("OUT_DIR"), "/mmpx_comp.wgsl"));
        let lcd_grid_wgsl = include_str!(concat!(env!("OUT_DIR"), "/lcd_grid_comp.wgsl"));
        let sai2x_wgsl = include_str!(concat!(env!("OUT_DIR"), "/sai2x_comp.wgsl"));
        let super_sai2x_wgsl = include_str!(concat!(env!("OUT_DIR"), "/super_sai2x_comp.wgsl"));
        let super_eagle_wgsl = include_str!(concat!(env!("OUT_DIR"), "/super_eagle_comp.wgsl"));
        let scalefx_wgsl = include_str!(concat!(env!("OUT_DIR"), "/scalefx_comp.wgsl"));

        WgpuScalePipeline {
            nearest: create_compute_pipeline(device, nearest_wgsl, "nearest"),
            bilinear: create_compute_pipeline(device, bilinear_wgsl, "bilinear"),
            epx: create_compute_pipeline(device, epx_wgsl, "epx"),
            eagle: create_compute_pipeline(device, eagle_wgsl, "eagle"),
            scale3x: create_compute_pipeline(device, scale3x_wgsl, "scale3x"),
            bicubic: create_compute_pipeline(device, bicubic_wgsl, "bicubic"),
            nearest_aa: create_compute_pipeline(device, nearest_aa_wgsl, "nearest_aa"),
            omniscale: create_compute_pipeline(device, omniscale_wgsl, "omniscale"),
            hqx: create_compute_pipeline(device, hqx_wgsl, "hqx"),
            xbr: create_compute_pipeline(device, xbr_wgsl, "xbr"),
            xbrz: create_compute_pipeline(device, xbrz_wgsl, "xbrz"),
            super_xbr: create_compute_pipeline(device, super_xbr_wgsl, "super_xbr"),
            omniscale_legacy: create_compute_pipeline(device, omniscale_legacy_wgsl, "omniscale_legacy"),
            edi: create_compute_pipeline(device, edi_wgsl, "edi"),
            nedi: create_compute_pipeline(device, nedi_wgsl, "nedi"),
            dcci: create_compute_pipeline(device, dcci_wgsl, "dcci"),
            mmpx: create_compute_pipeline(device, mmpx_wgsl, "mmpx"),
            lcd_grid: create_compute_pipeline(device, lcd_grid_wgsl, "lcd_grid"),
            sai2x: create_compute_pipeline(device, sai2x_wgsl, "sai2x"),
            super_sai2x: create_compute_pipeline(device, super_sai2x_wgsl, "super_sai2x"),
            super_eagle: create_compute_pipeline(device, super_eagle_wgsl, "super_eagle"),
            scalefx: create_compute_pipeline(device, scalefx_wgsl, "scalefx"),
            bufs: None,
        }
    }

    fn pipeline_for(&self, filter: WgpuScaleFilter) -> &wgpu::ComputePipeline {
        match filter {
            WgpuScaleFilter::Nearest => &self.nearest,
            WgpuScaleFilter::Bilinear => &self.bilinear,
            WgpuScaleFilter::Epx => &self.epx,
            WgpuScaleFilter::Eagle => &self.eagle,
            WgpuScaleFilter::Scale3x => &self.scale3x,
            WgpuScaleFilter::Bicubic => &self.bicubic,
            WgpuScaleFilter::NearestAa => &self.nearest_aa,
            WgpuScaleFilter::OmniScale => &self.omniscale,
            WgpuScaleFilter::Hqx => &self.hqx,
            WgpuScaleFilter::Xbr => &self.xbr,
            WgpuScaleFilter::Xbrz => &self.xbrz,
            WgpuScaleFilter::SuperXbr => &self.super_xbr,
            WgpuScaleFilter::OmniScaleLegacy => &self.omniscale_legacy,
            WgpuScaleFilter::Edi => &self.edi,
            WgpuScaleFilter::Nedi => &self.nedi,
            WgpuScaleFilter::Dcci => &self.dcci,
            WgpuScaleFilter::Mmpx => &self.mmpx,
            WgpuScaleFilter::LcdGrid => &self.lcd_grid,
            WgpuScaleFilter::Sai2x => &self.sai2x,
            WgpuScaleFilter::Super2xSai => &self.super_sai2x,
            WgpuScaleFilter::SuperEagle => &self.super_eagle,
            WgpuScaleFilter::ScaleFx => &self.scalefx,
        }
    }

    /// Encode a scaling compute pass. Returns a reference to the output texture.
    /// The texture is valid until the next call to `encode`.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        filter: WgpuScaleFilter,
        pixels: &[u32],
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) -> &wgpu::Texture {

        let px_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                pixels.as_ptr() as *const u8,
                pixels.len() * 4,
            )
        };

        // Reallocate buffers if dimensions changed
        let need_realloc = self.bufs.as_ref().map_or(true, |b| {
            b.src_w != src_w || b.src_h != src_h || b.out_w != out_w || b.out_h != out_h
        });

        if need_realloc {
            let storage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

            let px_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scale_pixels"),
                size: (src_w * src_h * 4) as u64,
                usage: storage_ro,
                mapped_at_creation: false,
            });

            // Uniform buffer: 8 u32s to cover all filter variants
            let uni_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scale_uniforms"),
                size: 32, // 8 × u32
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let output_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scale_output"),
                size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                     | wgpu::TextureUsages::TEXTURE_BINDING
                     | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            self.bufs = Some(ScaleBufs {
                src_w, src_h, out_w, out_h,
                px_buf, uni_buf, output_tex,
            });
        }

        let bufs = self.bufs.as_ref().unwrap();

        // Upload pixel data
        queue.write_buffer(&bufs.px_buf, 0, px_bytes);

        // Upload uniforms (layout depends on filter)
        let iscale = out_w / src_w;
        let uni_data: [u32; 8] = match filter {
            WgpuScaleFilter::Epx | WgpuScaleFilter::Hqx
            | WgpuScaleFilter::Xbr | WgpuScaleFilter::Xbrz
            | WgpuScaleFilter::LcdGrid => {
                [src_w, src_h, out_w, out_h, iscale, 0, 0, 0]
            }
            WgpuScaleFilter::OmniScale => {
                let sx = src_w as f32 / out_w as f32;
                let sy = src_h as f32 / out_h as f32;
                let pixel_size = (sx * sx + sy * sy).sqrt();
                [src_w, src_h, out_w, out_h,
                 f32::to_bits(pixel_size), 0, 0, 0]
            }
            _ => [src_w, src_h, out_w, out_h, 0, 0, 0, 0],
        };
        let uni_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(uni_data.as_ptr() as *const u8, 32)
        };
        queue.write_buffer(&bufs.uni_buf, 0, uni_bytes);

        let pipeline = self.pipeline_for(filter);
        let tex_view = bufs.output_tex.create_view(&Default::default());
        let dispatch_x = (out_w + 15) / 16;
        let dispatch_y = (out_h + 15) / 16;

        if filter == WgpuScaleFilter::ScaleFx {
            // ScaleFX: 5-pass pipeline with 4 intermediate float4 buffers + px_out for 9x chaining.
            // Shader layout: group 0=pixels, 1=buf0..buf3+px_out(rw), 2=texture, 3=uniforms
            // For 9x (out_w == src_w*9): run pipeline twice, second pass reads from px_out.
            let is_9x = out_w == src_w * 9;
            let mid_w = src_w * 3;
            let mid_h = src_h * 3;

            // Intermediate float4 buffers: sized for the larger dimension set (3x src for 9x)
            let max_pixels = if is_9x { mid_w * mid_h } else { src_w * src_h };
            let buf_size = (max_pixels as u64) * 16;
            let make_buf = |label| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label), size: buf_size,
                usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
            });
            let buf0 = make_buf("sfx_buf0");
            let buf1 = make_buf("sfx_buf1");
            let buf2 = make_buf("sfx_buf2");
            let buf3 = make_buf("sfx_buf3");

            // px_out: pass 4 writes packed XRGB here; for 9x the second pass reads this as pixel input.
            // px_out2: dummy write target for the second pass (avoids aliasing px_out as both RO and RW).
            // Always sized for the first pass 4 output — shader writes unconditionally.
            let px_out_size = (mid_w * mid_h * 4) as u64;
            let px_out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sfx_px_out"), size: px_out_size,
                usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
            });
            let px_out2_size = if is_9x { (out_w * out_h * 4) as u64 } else { 4 };
            let px_out2 = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sfx_px_out2"), size: px_out2_size,
                usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
            });

            let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0, resource: wgpu::BindingResource::TextureView(&tex_view),
                }],
            });

            // Helper: create bind groups and uniform buffers, dispatch 5 passes
            let dispatch_5 = |encoder: &mut wgpu::CommandEncoder,
                              px_src: &wgpu::Buffer,
                              px_dst: &wgpu::Buffer,
                              sw: u32, sh: u32, ow: u32, oh: u32| {
                let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0, resource: px_src.as_entire_binding(),
                    }],
                });
                let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(1),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf0.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf1.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf2.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: buf3.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: px_dst.as_entire_binding() },
                    ],
                });

                let mut pass_bgs = Vec::new();
                for pass_idx in 0u32..5 {
                    let pass_uni = [sw, sh, ow, oh, pass_idx, 0, 0, 0];
                    let pass_uni_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(pass_uni.as_ptr() as *const u8, 32)
                    };
                    let pass_uni_buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("sfx_uni"), size: 32,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    queue.write_buffer(&pass_uni_buf, 0, pass_uni_bytes);
                    pass_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(3),
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0, resource: pass_uni_buf.as_entire_binding(),
                        }],
                    }));
                }

                let sd_x = (sw + 15) / 16;
                let sd_y = (sh + 15) / 16;
                let od_x = (ow + 15) / 16;
                let od_y = (oh + 15) / 16;

                for pass_idx in 0..5u32 {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("scalefx"), timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bg0, &[]);
                    pass.set_bind_group(1, &bg1, &[]);
                    pass.set_bind_group(2, &bg2, &[]);
                    pass.set_bind_group(3, &pass_bgs[pass_idx as usize], &[]);
                    let (dx, dy) = if pass_idx < 4 { (sd_x, sd_y) } else { (od_x, od_y) };
                    pass.dispatch_workgroups(dx, dy, 1);
                }
            };

            // First 3x pass: reads uploaded pixels, writes packed output to px_out
            dispatch_5(encoder, &bufs.px_buf, &px_out, src_w, src_h, mid_w, mid_h);

            if is_9x {
                // Second 3x pass: reads from px_out, writes to px_out2 (separate buffer to avoid aliasing)
                dispatch_5(encoder, &px_out, &px_out2, mid_w, mid_h, out_w, out_h);
            }
        } else if filter == WgpuScaleFilter::SuperXbr {
            // Super xBR: 3-pass pipeline with intermediate buffer
            // Shader layout: group 0=pixels, 1=intermed(rw), 2=texture, 3=uniforms
            let intermed_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sxbr_intermed"),
                size: (out_w * out_h * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let bg0_sxbr = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.px_buf.as_entire_binding(),
                }],
            });

            let bg1_sxbr = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: intermed_buf.as_entire_binding(),
                }],
            });

            let bg2_sxbr = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                }],
            });

            // Create separate uniform buffers for each pass
            let mut pass_bgs = Vec::new();
            for pass_idx in 0u32..3 {
                let pass_uni = [src_w, src_h, out_w, out_h, pass_idx, 0, 0, 0];
                let pass_uni_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(pass_uni.as_ptr() as *const u8, 32)
                };
                let pass_uni_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sxbr_uni"),
                    size: 32,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&pass_uni_buf, 0, pass_uni_bytes);

                pass_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(3),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pass_uni_buf.as_entire_binding(),
                    }],
                }));
            }

            for pass_idx in 0..3 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("super_xbr"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg0_sxbr, &[]);
                pass.set_bind_group(1, &bg1_sxbr, &[]);
                pass.set_bind_group(2, &bg2_sxbr, &[]);
                pass.set_bind_group(3, &pass_bgs[pass_idx], &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
        } else {
            // Standard single-pass filters: group 0=pixels, 1=texture, 2=uniforms
            let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.px_buf.as_entire_binding(),
                }],
            });
            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                }],
            });
            let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.uni_buf.as_entire_binding(),
                }],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale_filter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        &self.bufs.as_ref().unwrap().output_tex
    }
}
