use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, ns_string};
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use super::scaling;
use super::scaling::{GpuPass, ScaleShader, multipass_uniforms};
use super::vectorize_metal::MetalVectorizePipeline;

pub(super) const METAL_SHADERS: &str = "
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex VertexOut vertex_main(uint vid [[vertex_id]],
                             constant float4 *viewport [[buffer(0)]]) {
    // viewport.x = x_offset (NDC), viewport.y = y_offset (NDC)
    // viewport.z = width (NDC), viewport.w = height (NDC)
    float4 vp = viewport[0];
    float2 positions[4] = {
        float2(vp.x,        vp.y),
        float2(vp.x + vp.z, vp.y),
        float2(vp.x,        vp.y + vp.w),
        float2(vp.x + vp.z, vp.y + vp.w),
    };
    float2 texcoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.texcoord = texcoords[vid];
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::nearest, min_filter::nearest);
    return tex.sample(s, in.texcoord);
}

fragment float4 fragment_linear(VertexOut in [[stage_in]],
                                texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    return tex.sample(s, in.texcoord);
}
";

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type CmdQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type RenderPipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;
type ComputePipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct MetalRenderer {
    pub device: Device,
    pub layer: Retained<CAMetalLayer>,
    pub command_queue: CmdQueue,
    pipeline_state: RenderPipeline,
    pipeline_state_linear: RenderPipeline,
    pub use_linear_blit: bool,
    pub texture: Texture,
    pub tex_w: u32,
    pub tex_h: u32,
    // Compute scaling pipelines (lazily initialized)
    scale_compute: [Option<ComputePipeline>; ScaleShader::COUNT],
    pub compute_out_tex: Option<Texture>,
    pub compute_out_w: u32,
    pub compute_out_h: u32,
    // Full GPU vectorize pipeline
    pub vectorize_pipeline: Option<MetalVectorizePipeline>,
}

fn make_buf(dev: &ProtocolObject<dyn MTLDevice>, data: *const u8, len: usize) -> Buffer {
    unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::new_unchecked(data as *mut _),
            len,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("failed to create buffer")
    }
}

fn make_buf_empty(dev: &ProtocolObject<dyn MTLDevice>, len: usize) -> Buffer {
    dev.newBufferWithLength_options(len.max(4), MTLResourceOptions::StorageModeShared)
        .expect("failed to create buffer")
}

fn make_texture(
    dev: &ProtocolObject<dyn MTLDevice>,
    w: u32,
    h: u32,
    usage: MTLTextureUsage,
) -> Texture {
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    unsafe {
        desc.setWidth(w as usize);
        desc.setHeight(h as usize);
    }
    desc.setUsage(usage);
    dev.newTextureWithDescriptor(&desc)
        .expect("failed to create texture")
}

macro_rules! msl_code {
    ($($variant:ident => $module:literal),* $(,)?) => {
        /// MSL source of a scaling shader.
        fn msl(shader: ScaleShader) -> &'static [u8] {
            match shader {
                $(ScaleShader::$variant => {
                    include_bytes!(concat!(env!("OUT_DIR"), "/", $module, "_comp.metal"))
                }),*
            }
        }
    };
}
vibeboy_core::scale_shader_list!(msl_code);

/// Encode one 16x16-threadgroup compute dispatch covering `grid` threads,
/// binding `buffers` at indices 0.. and `tex` at texture index 0.
fn encode_dispatch(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    tex: &ProtocolObject<dyn MTLTexture>,
    (w, h): (u32, u32),
) {
    let encoder = cmd.computeCommandEncoder().unwrap();
    encoder.setComputePipelineState(pipeline);
    unsafe {
        for (i, buf) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf), 0, i);
        }
        encoder.setTexture_atIndex(Some(tex), 0);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: w.div_ceil(16) as usize,
                height: h.div_ceil(16) as usize,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            },
        );
    }
    encoder.endEncoding();
}

fn load_compute_pipeline(
    dev: &ProtocolObject<dyn MTLDevice>,
    msl: &[u8],
) -> Option<ComputePipeline> {
    let src = std::str::from_utf8(msl).ok()?;
    let ns_src = NSString::from_str(src);
    let lib = dev
        .newLibraryWithSource_options_error(&ns_src, None)
        .map_err(|e| eprintln!("MSL compile error: {e}"))
        .ok()?;
    let func_name = ns_string!("main_0");
    let func = lib.newFunctionWithName(func_name)?;
    dev.newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| eprintln!("Pipeline error: {e}"))
        .ok()
}

impl MetalRenderer {
    pub fn new(tex_w: u32, tex_h: u32) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setPresentsWithTransaction(false);

        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        // Compile shaders
        let ns_src = NSString::from_str(METAL_SHADERS);
        let library = device
            .newLibraryWithSource_options_error(&ns_src, None)
            .expect("Failed to compile Metal shaders");
        let vert_fn = library
            .newFunctionWithName(ns_string!("vertex_main"))
            .unwrap();
        let frag_fn = library
            .newFunctionWithName(ns_string!("fragment_main"))
            .unwrap();

        let pipeline_desc = MTLRenderPipelineDescriptor::new();
        pipeline_desc.setVertexFunction(Some(&vert_fn));
        pipeline_desc.setFragmentFunction(Some(&frag_fn));
        unsafe {
            pipeline_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0)
                .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        }

        let pipeline_state = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
            .expect("Failed to create render pipeline state");

        let frag_linear_fn = library
            .newFunctionWithName(ns_string!("fragment_linear"))
            .unwrap();
        pipeline_desc.setFragmentFunction(Some(&frag_linear_fn));
        let pipeline_state_linear = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
            .expect("Failed to create linear render pipeline state");

        let texture = make_texture(&device, tex_w, tex_h, MTLTextureUsage::ShaderRead);

        MetalRenderer {
            device,
            layer,
            command_queue,
            pipeline_state,
            pipeline_state_linear,
            use_linear_blit: false,
            texture,
            tex_w,
            tex_h,
            scale_compute: Default::default(),
            compute_out_tex: None,
            compute_out_w: 0,
            compute_out_h: 0,
            vectorize_pipeline: None,
        }
    }

    /// Compiled pipeline for a scaling shader, built on first use.
    fn ensure_scale_compute(
        &mut self,
        shader: ScaleShader,
    ) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
        let idx = shader.index();
        if self.scale_compute[idx].is_none() {
            self.scale_compute[idx] = load_compute_pipeline(&self.device, msl(shader));
        }
        self.scale_compute[idx].as_deref()
    }

    /// Run a compute scaling filter and return the output texture.
    pub fn run_scale_compute(
        &mut self,
        filter: scaling::ScaleFilter,
        pixels: &[u32],
        src_w: u32,
        src_h: u32,
        disp_w: u32,
        disp_h: u32,
    ) -> Option<(&ProtocolObject<dyn MTLTexture>, u32, u32)> {
        let gpu = filter.gpu()?;

        // Integer filters render at their factor; adaptive filters at
        // aspect-correct display dimensions.
        let adaptive = {
            let s = (disp_w as f64 / src_w as f64)
                .min(disp_h as f64 / src_h as f64)
                .max(1.0);
            (
                (src_w as f64 * s).round() as u32,
                (src_h as f64 * s).round() as u32,
            )
        };
        let (out_w, out_h) = filter.output_size(src_w, src_h, adaptive);

        self.ensure_scale_compute(gpu.shader)?;

        // Create/resize output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            self.compute_out_tex = Some(make_texture(
                &self.device,
                out_w,
                out_h,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            ));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }

        let cmd = match self.command_queue.commandBuffer() {
            Some(c) => c,
            None => {
                log::error!("Metal: failed to create command buffer");
                return None;
            }
        };
        let pipeline = self.scale_compute[gpu.shader.index()].as_deref()?;
        let out_tex = self.compute_out_tex.as_deref()?;
        let px_buf = make_buf(&self.device, pixels.as_ptr() as *const u8, pixels.len() * 4);

        let (src, out) = ((src_w, src_h), (out_w, out_h));
        match gpu.pass {
            GpuPass::Single => {
                let uniforms = gpu.uniforms(src_w, src_h, out_w, out_h);
                let uni_buf = make_buf(&self.device, uniforms.as_ptr() as *const u8, 32);
                encode_dispatch(&cmd, pipeline, &[&uni_buf, &px_buf], out_tex, out);
            }
            GpuPass::SuperXbr => {
                // Metal buffer layout: 0=uniforms, 1=pixels, 2=intermed, texture(0)=output
                let intermed = make_buf_empty(&self.device, (out_w * out_h * 4) as usize);
                for pass in 0u32..3 {
                    let uniforms = multipass_uniforms(src_w, src_h, out_w, out_h, pass);
                    let uni_buf = make_buf(&self.device, uniforms.as_ptr() as *const u8, 32);
                    encode_dispatch(
                        &cmd,
                        pipeline,
                        &[&uni_buf, &px_buf, &intermed],
                        out_tex,
                        out,
                    );
                }
            }
            GpuPass::ScaleFx { chained } => {
                self.encode_scalefx(&cmd, pipeline, out_tex, &px_buf, src, out, chained);
            }
        }
        cmd.commit();

        Some((self.compute_out_tex.as_deref()?, out_w, out_h))
    }

    /// ScaleFX 5-pass (3x) or 10-pass (9x, `chained`) compute dispatch.
    /// Metal buffer layout: 0=uniforms, 1=pixels, 2=buf0, 3=buf1, 4=buf2,
    /// 5=buf3, 6=px_out, texture(0)=output
    fn encode_scalefx(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        out_tex: &ProtocolObject<dyn MTLTexture>,
        px_buf: &ProtocolObject<dyn MTLBuffer>,
        (src_w, src_h): (u32, u32),
        (out_w, out_h): (u32, u32),
        chained: bool,
    ) {
        let mid_w = src_w * 3;
        let mid_h = src_h * 3;

        // Intermediate float4 buffers (sized for the larger pass in 9x mode)
        let max_pixels = if chained {
            mid_w * mid_h
        } else {
            src_w * src_h
        };
        let f4_size = (max_pixels as usize) * 16;
        let buf0 = make_buf_empty(&self.device, f4_size);
        let buf1 = make_buf_empty(&self.device, f4_size);
        let buf2 = make_buf_empty(&self.device, f4_size);
        let buf3 = make_buf_empty(&self.device, f4_size);

        // px_out / px_out2: packed XRGB output for chaining 9x.
        // Always sized for the pass 4 output; the shader writes to px_out unconditionally.
        let px_out_size = (mid_w * mid_h * 4) as usize;
        let px_out = make_buf_empty(&self.device, px_out_size);
        let px_out2_size = if chained {
            (out_w * out_h * 4) as usize
        } else {
            4
        };
        let px_out2 = make_buf_empty(&self.device, px_out2_size);

        // Five passes: 0-3 over the source, 4 over the 3x output.
        let dispatch_5 = |px_src: &ProtocolObject<dyn MTLBuffer>,
                          px_dst: &ProtocolObject<dyn MTLBuffer>,
                          sw: u32,
                          sh: u32,
                          ow: u32,
                          oh: u32| {
            for pass in 0u32..5 {
                let uniforms = multipass_uniforms(sw, sh, ow, oh, pass);
                let uni_buf = make_buf(&self.device, uniforms.as_ptr() as *const u8, 32);
                let grid = if pass < 4 { (sw, sh) } else { (ow, oh) };
                encode_dispatch(
                    cmd,
                    pipeline,
                    &[&uni_buf, px_src, &buf0, &buf1, &buf2, &buf3, px_dst],
                    out_tex,
                    grid,
                );
            }
        };

        // First 3x pass
        dispatch_5(px_buf, &px_out, src_w, src_h, mid_w, mid_h);

        if chained {
            // Second 3x pass: read from px_out, write to px_out2
            dispatch_5(&px_out, &px_out2, mid_w, mid_h, out_w, out_h);
        }
    }

    pub fn update_texture(&self, pixels: &[u32]) {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: self.tex_w as usize,
                height: self.tex_h as usize,
                depth: 1,
            },
        };
        unsafe {
            self.texture
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    NonNull::new_unchecked(pixels.as_ptr() as *mut _),
                    (self.tex_w * 4) as usize,
                );
        }
    }

    pub fn render(&self) {
        objc2::rc::autoreleasepool(|_| {
            let drawable = match self.layer.nextDrawable() {
                Some(d) => d,
                None => return,
            };

            let dst_tex = drawable.texture();
            let dst_w = dst_tex.width() as f32;
            let dst_h = dst_tex.height() as f32;

            // Compute aspect-ratio-correct viewport in NDC (-1..1)
            let tex_aspect = self.tex_w as f32 / self.tex_h as f32;
            let dst_aspect = dst_w / dst_h;
            let (ndc_w, ndc_h) = if dst_aspect > tex_aspect {
                (2.0 * tex_aspect / dst_aspect, 2.0)
            } else {
                (2.0, 2.0 * dst_aspect / tex_aspect)
            };
            let ndc_x = -ndc_w / 2.0;
            let ndc_y = -ndc_h / 2.0;
            let viewport: [f32; 4] = [ndc_x, ndc_y, ndc_w, ndc_h];

            let rpd = MTLRenderPassDescriptor::new();
            let ca = unsafe { rpd.colorAttachments().objectAtIndexedSubscript(0) };
            ca.setTexture(Some(&dst_tex));
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setStoreAction(MTLStoreAction::Store);
            ca.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            });

            let cmd_buf = match self.command_queue.commandBuffer() {
                Some(c) => c,
                None => {
                    log::error!("Metal: failed to create command buffer");
                    return;
                }
            };
            let Some(encoder) = cmd_buf.renderCommandEncoderWithDescriptor(&rpd) else {
                return;
            };

            let ps = if self.use_linear_blit {
                &self.pipeline_state_linear
            } else {
                &self.pipeline_state
            };
            encoder.setRenderPipelineState(ps);
            unsafe {
                encoder.setVertexBytes_length_atIndex(
                    NonNull::new_unchecked(viewport.as_ptr() as *mut _),
                    std::mem::size_of::<[f32; 4]>(),
                    0,
                );
                encoder.setFragmentTexture_atIndex(Some(&self.texture), 0);
            }
            unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                );
            }
            encoder.endEncoding();

            cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));
            cmd_buf.commit();
        });
    }
}
