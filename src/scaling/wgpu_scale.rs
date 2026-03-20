//! GPU scaling filter pipeline using wgpu (WebGPU-compatible).
//!
//! Runs pixel-art scaling filters as compute shaders:
//! EPX, Eagle, Scale3x, Bicubic, AA Nearest, OmniScale.
//!
//! All shaders read from a u32 pixel storage buffer and write to an rgba8 output texture.

use wgpu;

/// Available compute-based scaling filters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WgpuScaleFilter {
    Nearest,
    Epx,
    Eagle,
    Scale3x,
    Bicubic,
    AaNearest,
    OmniScale,
    Hqx,
    Xbr,
    Xbrz,
    SuperXbr,
    OmniScaleLegacy,
}

impl WgpuScaleFilter {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nearest" => Some(Self::Nearest),
            "epx" => Some(Self::Epx),
            "eagle" => Some(Self::Eagle),
            "scale3x" => Some(Self::Scale3x),
            "bicubic" => Some(Self::Bicubic),
            "aa-nearest" => Some(Self::AaNearest),
            "omniscale" => Some(Self::OmniScale),
            "hqx" => Some(Self::Hqx),
            "xbr" => Some(Self::Xbr),
            "xbrz" => Some(Self::Xbrz),
            "super-xbr" => Some(Self::SuperXbr),
            "omniscale-legacy" => Some(Self::OmniScaleLegacy),
            _ => None,
        }
    }

    pub fn all_names() -> &'static [&'static str] {
        &["nearest", "epx", "eagle", "scale3x", "bicubic", "aa-nearest",
          "omniscale", "hqx", "xbr", "xbrz", "super-xbr", "omniscale-legacy"]
    }

    /// For integer-scale filters, compute the native output dimensions.
    /// Returns None for resolution-independent filters (use requested size).
    pub fn native_size(&self, src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> (u32, u32) {
        match self {
            // Fixed 2x
            Self::Eagle | Self::SuperXbr => (src_w * 2, src_h * 2),
            // Fixed 3x
            Self::Scale3x => (src_w * 3, src_h * 3),
            // 2x/3x/4x — pick best integer fit
            Self::Epx | Self::Hqx | Self::Xbr => {
                let s = (out_w / src_w).max(1).min(4);
                let s = if s <= 2 { 2 } else if s <= 3 { 3 } else { 4 };
                (src_w * s, src_h * s)
            }
            // 2x-6x
            Self::Xbrz => {
                let s = (out_w / src_w).max(2).min(6);
                (src_w * s, src_h * s)
            }
            // Resolution-independent — use requested size
            Self::Nearest | Self::Bicubic | Self::AaNearest
            | Self::OmniScaleLegacy => (out_w, out_h),
            // OmniScale works best at moderate integer scale
            Self::OmniScale => {
                let s = (out_w / src_w).max(2).min(8);
                (src_w * s, src_h * s)
            }
        }
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
    epx: wgpu::ComputePipeline,
    eagle: wgpu::ComputePipeline,
    scale3x: wgpu::ComputePipeline,
    bicubic: wgpu::ComputePipeline,
    aa_nearest: wgpu::ComputePipeline,
    omniscale: wgpu::ComputePipeline,
    hqx: wgpu::ComputePipeline,
    xbr: wgpu::ComputePipeline,
    xbrz: wgpu::ComputePipeline,
    super_xbr: wgpu::ComputePipeline,
    omniscale_legacy: wgpu::ComputePipeline,
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
        let epx_wgsl = include_str!(concat!(env!("OUT_DIR"), "/epx_comp.wgsl"));
        let eagle_wgsl = include_str!(concat!(env!("OUT_DIR"), "/eagle_comp.wgsl"));
        let scale3x_wgsl = include_str!(concat!(env!("OUT_DIR"), "/scale3x_comp.wgsl"));
        let bicubic_wgsl = include_str!(concat!(env!("OUT_DIR"), "/bicubic_comp.wgsl"));
        let aa_nearest_wgsl = include_str!(concat!(env!("OUT_DIR"), "/aa_nearest_comp.wgsl"));
        let omniscale_wgsl = include_str!(concat!(env!("OUT_DIR"), "/omniscale_comp.wgsl"));
        let hqx_wgsl = include_str!(concat!(env!("OUT_DIR"), "/hqx_comp.wgsl"));
        let xbr_wgsl = include_str!(concat!(env!("OUT_DIR"), "/xbr_comp.wgsl"));
        let xbrz_wgsl = include_str!(concat!(env!("OUT_DIR"), "/xbrz_comp.wgsl"));
        let super_xbr_wgsl = include_str!(concat!(env!("OUT_DIR"), "/super_xbr_comp.wgsl"));
        let omniscale_legacy_wgsl = include_str!(concat!(env!("OUT_DIR"), "/omniscale_legacy_comp.wgsl"));

        WgpuScalePipeline {
            epx: create_compute_pipeline(device, epx_wgsl, "epx"),
            eagle: create_compute_pipeline(device, eagle_wgsl, "eagle"),
            scale3x: create_compute_pipeline(device, scale3x_wgsl, "scale3x"),
            bicubic: create_compute_pipeline(device, bicubic_wgsl, "bicubic"),
            aa_nearest: create_compute_pipeline(device, aa_nearest_wgsl, "aa_nearest"),
            omniscale: create_compute_pipeline(device, omniscale_wgsl, "omniscale"),
            hqx: create_compute_pipeline(device, hqx_wgsl, "hqx"),
            xbr: create_compute_pipeline(device, xbr_wgsl, "xbr"),
            xbrz: create_compute_pipeline(device, xbrz_wgsl, "xbrz"),
            super_xbr: create_compute_pipeline(device, super_xbr_wgsl, "super_xbr"),
            omniscale_legacy: create_compute_pipeline(device, omniscale_legacy_wgsl, "omniscale_legacy"),
            bufs: None,
        }
    }

    fn pipeline_for(&self, filter: WgpuScaleFilter) -> &wgpu::ComputePipeline {
        match filter {
            WgpuScaleFilter::Nearest => unreachable!("nearest uses direct blit"),
            WgpuScaleFilter::Epx => &self.epx,
            WgpuScaleFilter::Eagle => &self.eagle,
            WgpuScaleFilter::Scale3x => &self.scale3x,
            WgpuScaleFilter::Bicubic => &self.bicubic,
            WgpuScaleFilter::AaNearest => &self.aa_nearest,
            WgpuScaleFilter::OmniScale => &self.omniscale,
            WgpuScaleFilter::Hqx => &self.hqx,
            WgpuScaleFilter::Xbr => &self.xbr,
            WgpuScaleFilter::Xbrz => &self.xbrz,
            WgpuScaleFilter::SuperXbr => &self.super_xbr,
            WgpuScaleFilter::OmniScaleLegacy => &self.omniscale_legacy,
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
        // Integer-scale filters render at native size; blit shader stretches to window
        let (out_w, out_h) = filter.native_size(src_w, src_h, out_w, out_h);

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
            | WgpuScaleFilter::Xbr | WgpuScaleFilter::Xbrz => {
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

        if filter == WgpuScaleFilter::SuperXbr {
            // Super xBR: 3-pass pipeline with intermediate buffer
            // Shader layout: group 0 = pixels, group 1 = intermed(rw),
            //                group 2 = output texture, group 3 = uniforms
            let bg0_sxbr = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.px_buf.as_entire_binding(),
                }],
            });
            let intermed_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sxbr_intermed"),
                size: (out_w * out_h * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
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

            // Create separate uniform buffers for each pass (wgpu queue.write_buffer
            // executes before the encoder, so we can't update the same buffer per-pass)
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
