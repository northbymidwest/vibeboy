//! GPU scaling filter pipeline using wgpu (WebGPU-compatible).
//!
//! Runs every pixel-art scaling filter that has a compute shader. Which
//! shader a filter uses, how it is dispatched and what its uniforms hold
//! come from the filter registry (`ScaleFilter::gpu()`); this module only
//! loads the WGSL and records the passes.
//!
//! All shaders read from a u32 pixel storage buffer and write to an rgba8 output texture.

use super::{GpuPass, ScaleFilter, ScaleShader, multipass_uniforms};
use wgpu;

/// A scaling filter that has a wgpu compute path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WgpuScaleFilter(ScaleFilter);

impl WgpuScaleFilter {
    /// Map a ScaleFilter to its compute path, if it has one.
    pub fn from_scale_filter(f: ScaleFilter) -> Option<Self> {
        f.gpu().map(|_| Self(f))
    }

    /// The filter this compute path renders.
    pub fn filter(self) -> ScaleFilter {
        self.0
    }
}

macro_rules! wgsl_code {
    ($($variant:ident => $module:literal),* $(,)?) => {
        /// WGSL source of a scaling shader.
        fn wgsl(shader: ScaleShader) -> &'static str {
            match shader {
                $(ScaleShader::$variant => {
                    include_str!(concat!(env!("OUT_DIR"), "/", $module, "_comp.wgsl"))
                }),*
            }
        }
    };
}
crate::scale_shader_list!(wgsl_code);

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
    /// One pipeline per scaling shader, indexed by `ScaleShader::index()`.
    pipelines: Vec<wgpu::ComputePipeline>,
    bufs: Option<ScaleBufs>,
}

/// Source/output resources, re-created only when a size changes.
struct ScaleBufs {
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    px_buf: wgpu::Buffer,
    output_tex: wgpu::Texture,
    output_view: wgpu::TextureView,
    /// Recorded passes of the last filter, re-created when the filter changes.
    passes: Option<FilterPasses>,
}

/// Everything one filter's dispatches bind: intermediates, uniform buffers
/// (written once, since they only depend on the filter and sizes) and bind
/// groups.
struct FilterPasses {
    filter: ScaleFilter,
    label: &'static str,
    passes: Vec<Dispatch>,
}

struct Dispatch {
    bind_groups: Vec<wgpu::BindGroup>,
    workgroups: (u32, u32),
}

fn workgroups(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(16), h.div_ceil(16))
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn uniform_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: [u32; 8],
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 32, // 8 × u32
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 32) };
    queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Bind group `group` of `pipeline` with the given resources at bindings 0..n.
fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    group: u32,
    resources: &[wgpu::BindingResource],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: r.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(group),
        entries: &entries,
    })
}

impl ScaleBufs {
    fn new(device: &wgpu::Device, src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Self {
        let px_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scale_pixels"),
            size: (src_w * src_h * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scale_output"),
            size: wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&Default::default());
        ScaleBufs {
            src_w,
            src_h,
            out_w,
            out_h,
            px_buf,
            output_tex,
            output_view,
            passes: None,
        }
    }

    /// Build the bind groups and uniforms for every dispatch of `filter`.
    fn build_passes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        filter: ScaleFilter,
    ) -> FilterPasses {
        let shader = filter.gpu().expect("filter has no GPU shader");
        let (src_w, src_h, out_w, out_h) = (self.src_w, self.src_h, self.out_w, self.out_h);
        let tex = wgpu::BindingResource::TextureView(&self.output_view);

        match shader.pass {
            GpuPass::Single => {
                // Standard single-pass filters: group 0=pixels, 1=texture, 2=uniforms
                let uni = uniform_buffer(
                    device,
                    queue,
                    "scale_uniforms",
                    shader.uniforms(src_w, src_h, out_w, out_h),
                );
                let bind_groups = vec![
                    bind_group(device, pipeline, 0, &[self.px_buf.as_entire_binding()]),
                    bind_group(device, pipeline, 1, &[tex]),
                    bind_group(device, pipeline, 2, &[uni.as_entire_binding()]),
                ];
                FilterPasses {
                    filter,
                    label: "scale_filter",
                    passes: vec![Dispatch {
                        bind_groups,
                        workgroups: workgroups(out_w, out_h),
                    }],
                }
            }
            GpuPass::SuperXbr => {
                // Super xBR: 3-pass pipeline with intermediate buffer
                // Shader layout: group 0=pixels, 1=intermed(rw), 2=texture, 3=uniforms
                let intermed = storage_buffer(device, "sxbr_intermed", (out_w * out_h * 4) as u64);
                let bg0 = bind_group(device, pipeline, 0, &[self.px_buf.as_entire_binding()]);
                let bg1 = bind_group(device, pipeline, 1, &[intermed.as_entire_binding()]);
                let bg2 = bind_group(device, pipeline, 2, &[tex]);
                let passes = (0u32..3)
                    .map(|pass| {
                        let uni = uniform_buffer(
                            device,
                            queue,
                            "sxbr_uni",
                            multipass_uniforms(src_w, src_h, out_w, out_h, pass),
                        );
                        Dispatch {
                            bind_groups: vec![
                                bg0.clone(),
                                bg1.clone(),
                                bg2.clone(),
                                bind_group(device, pipeline, 3, &[uni.as_entire_binding()]),
                            ],
                            workgroups: workgroups(out_w, out_h),
                        }
                    })
                    .collect();
                FilterPasses {
                    filter,
                    label: "super_xbr",
                    passes,
                }
            }
            GpuPass::ScaleFx { chained } => {
                // ScaleFX: 5-pass pipeline with 4 intermediate float4 buffers + px_out for 9x chaining.
                // Shader layout: group 0=pixels, 1=buf0..buf3+px_out(rw), 2=texture, 3=uniforms
                // For 9x: run the five passes twice, the second reading from px_out.
                let mid_w = src_w * 3;
                let mid_h = src_h * 3;

                // Intermediate float4 buffers: sized for the larger dimension set (3x src for 9x)
                let max_pixels = if chained {
                    mid_w * mid_h
                } else {
                    src_w * src_h
                };
                let buf_size = (max_pixels as u64) * 16;
                let intermediates = [
                    storage_buffer(device, "sfx_buf0", buf_size),
                    storage_buffer(device, "sfx_buf1", buf_size),
                    storage_buffer(device, "sfx_buf2", buf_size),
                    storage_buffer(device, "sfx_buf3", buf_size),
                ];

                // px_out: pass 4 writes packed XRGB here; for 9x the second pass reads this as pixel input.
                // px_out2: write target for the second pass (avoids aliasing px_out as both RO and RW).
                // Always sized for the first pass 4 output; the shader writes unconditionally.
                let px_out = storage_buffer(device, "sfx_px_out", (mid_w * mid_h * 4) as u64);
                let px_out2_size = if chained {
                    (out_w * out_h * 4) as u64
                } else {
                    4
                };
                let px_out2 = storage_buffer(device, "sfx_px_out2", px_out2_size);

                let bg2 = bind_group(device, pipeline, 2, &[tex]);

                // Five passes: 0-3 over the source, 4 over the 3x output.
                let chain = |px_src: &wgpu::Buffer,
                             px_dst: &wgpu::Buffer,
                             sw: u32,
                             sh: u32,
                             ow: u32,
                             oh: u32| {
                    let bg0 = bind_group(device, pipeline, 0, &[px_src.as_entire_binding()]);
                    let mut rw: Vec<_> = intermediates
                        .iter()
                        .map(|b| b.as_entire_binding())
                        .collect();
                    rw.push(px_dst.as_entire_binding());
                    let bg1 = bind_group(device, pipeline, 1, &rw);
                    (0u32..5)
                        .map(|pass| {
                            let uni = uniform_buffer(
                                device,
                                queue,
                                "sfx_uni",
                                multipass_uniforms(sw, sh, ow, oh, pass),
                            );
                            Dispatch {
                                bind_groups: vec![
                                    bg0.clone(),
                                    bg1.clone(),
                                    bg2.clone(),
                                    bind_group(device, pipeline, 3, &[uni.as_entire_binding()]),
                                ],
                                workgroups: if pass < 4 {
                                    workgroups(sw, sh)
                                } else {
                                    workgroups(ow, oh)
                                },
                            }
                        })
                        .collect::<Vec<_>>()
                };

                // First 3x pass: reads uploaded pixels, writes packed output to px_out
                let mut passes = chain(&self.px_buf, &px_out, src_w, src_h, mid_w, mid_h);
                if chained {
                    // Second 3x pass: reads from px_out, writes to px_out2
                    passes.extend(chain(&px_out, &px_out2, mid_w, mid_h, out_w, out_h));
                }
                FilterPasses {
                    filter,
                    label: "scalefx",
                    passes,
                }
            }
        }
    }
}

impl WgpuScalePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let pipelines = ScaleShader::ALL
            .iter()
            .map(|&s| create_compute_pipeline(device, wgsl(s), s.module()))
            .collect();
        WgpuScalePipeline {
            pipelines,
            bufs: None,
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
        let filter = filter.filter();
        let shader = filter.gpu().expect("filter has no GPU shader");
        let pipeline = &self.pipelines[shader.shader.index()];

        // Reallocate buffers if dimensions changed
        let need_realloc = self.bufs.as_ref().is_none_or(|b| {
            b.src_w != src_w || b.src_h != src_h || b.out_w != out_w || b.out_h != out_h
        });
        if need_realloc {
            self.bufs = Some(ScaleBufs::new(device, src_w, src_h, out_w, out_h));
        }
        let bufs = self.bufs.as_mut().unwrap();

        // Rebuild bind groups and uniforms if the filter changed
        if bufs.passes.as_ref().is_none_or(|p| p.filter != filter) {
            bufs.passes = Some(bufs.build_passes(device, queue, pipeline, filter));
        }

        // Upload pixel data
        let px_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        queue.write_buffer(&bufs.px_buf, 0, px_bytes);

        let passes = bufs.passes.as_ref().unwrap();
        for d in &passes.passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(passes.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for (i, bg) in d.bind_groups.iter().enumerate() {
                pass.set_bind_group(i as u32, bg, &[]);
            }
            pass.dispatch_workgroups(d.workgroups.0, d.workgroups.1, 1);
        }

        &bufs.output_tex
    }
}
