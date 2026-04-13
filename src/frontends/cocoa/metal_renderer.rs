use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{NSString, ns_string};
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use super::scaling;
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
    pub texture: Texture,
    pub tex_w: u32,
    pub tex_h: u32,
    // Compute scaling pipelines (lazily initialized)
    scale_compute: [Option<ComputePipeline>; 22],
    pub compute_out_tex: Option<Texture>,
    pub compute_out_w: u32,
    pub compute_out_h: u32,
    // Full GPU vectorize pipeline
    pub vectorize_pipeline: Option<MetalVectorizePipeline>,
}

fn safe_buf(dev: &ProtocolObject<dyn MTLDevice>, data: &[u8]) -> Buffer {
    unsafe {
        if data.is_empty() {
            dev.newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
                .expect("failed to create buffer")
        } else {
            dev.newBufferWithBytes_length_options(
                NonNull::new_unchecked(data.as_ptr() as *mut _),
                data.len(),
                MTLResourceOptions::StorageModeShared,
            ).expect("failed to create buffer")
        }
    }
}

fn make_buf(dev: &ProtocolObject<dyn MTLDevice>, data: *const u8, len: usize) -> Buffer {
    unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::new_unchecked(data as *mut _),
            len,
            MTLResourceOptions::StorageModeShared,
        ).expect("failed to create buffer")
    }
}

fn make_buf_empty(dev: &ProtocolObject<dyn MTLDevice>, len: usize) -> Buffer {
    dev.newBufferWithLength_options(len.max(4), MTLResourceOptions::StorageModeShared)
        .expect("failed to create buffer")
}

fn make_texture(dev: &ProtocolObject<dyn MTLDevice>, w: u32, h: u32, usage: MTLTextureUsage) -> Texture {
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    unsafe {
        desc.setWidth(w as usize);
        desc.setHeight(h as usize);
    }
    desc.setUsage(usage);
    dev.newTextureWithDescriptor(&desc).expect("failed to create texture")
}

fn load_compute_pipeline(
    dev: &ProtocolObject<dyn MTLDevice>,
    msl: &[u8],
) -> Option<ComputePipeline> {
    let src = std::str::from_utf8(msl).ok()?;
    let ns_src = NSString::from_str(src);
    let lib = dev.newLibraryWithSource_options_error(&ns_src, None)
        .map_err(|e| eprintln!("MSL compile error: {e}")).ok()?;
    let func_name = ns_string!("main_0");
    let func = lib.newFunctionWithName(func_name)?;
    dev.newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| eprintln!("Pipeline error: {e}")).ok()
}

impl MetalRenderer {
    pub fn new(tex_w: u32, tex_h: u32) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setPresentsWithTransaction(false);

        let command_queue = device.newCommandQueue().expect("Failed to create command queue");

        // Compile shaders
        let ns_src = NSString::from_str(METAL_SHADERS);
        let library = device
            .newLibraryWithSource_options_error(&ns_src, None)
            .expect("Failed to compile Metal shaders");
        let vert_fn = library.newFunctionWithName(ns_string!("vertex_main")).unwrap();
        let frag_fn = library.newFunctionWithName(ns_string!("fragment_main")).unwrap();

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

        let texture = make_texture(&device, tex_w, tex_h, MTLTextureUsage::ShaderRead);

        MetalRenderer {
            device,
            layer,
            command_queue,
            pipeline_state,
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

    pub fn ensure_scale_compute(&mut self, idx: usize, msl: &[u8]) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
        if self.scale_compute[idx].is_none() && !msl.is_empty() {
            self.scale_compute[idx] = load_compute_pipeline(&self.device, msl);
        }
        self.scale_compute[idx].as_deref()
    }

    /// Run a compute scaling filter and return the output texture.
    pub fn run_scale_compute(
        &mut self,
        filter: scaling::ScaleFilter,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        disp_w: u32, disp_h: u32,
    ) -> Option<(&ProtocolObject<dyn MTLTexture>, u32, u32)> {
        use scaling::ScaleFilter;

        let (idx, msl): (usize, &[u8]) = match filter {
            ScaleFilter::OmniScale =>
                (0, include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_comp.metal"))),
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x =>
                (1, include_bytes!(concat!(env!("OUT_DIR"), "/epx_comp.metal"))),
            ScaleFilter::Eagle =>
                (2, include_bytes!(concat!(env!("OUT_DIR"), "/eagle_comp.metal"))),
            ScaleFilter::Scale3x =>
                (3, include_bytes!(concat!(env!("OUT_DIR"), "/scale3x_comp.metal"))),
            ScaleFilter::Bicubic =>
                (4, include_bytes!(concat!(env!("OUT_DIR"), "/bicubic_comp.metal"))),
            ScaleFilter::NearestAa =>
                (5, include_bytes!(concat!(env!("OUT_DIR"), "/nearest_aa_comp.metal"))),
            ScaleFilter::Hqx(_) =>
                (6, include_bytes!(concat!(env!("OUT_DIR"), "/hqx_comp.metal"))),
            ScaleFilter::Xbr(_) =>
                (7, include_bytes!(concat!(env!("OUT_DIR"), "/xbr_comp.metal"))),
            ScaleFilter::Xbrz(_) =>
                (8, include_bytes!(concat!(env!("OUT_DIR"), "/xbrz_comp.metal"))),
            ScaleFilter::OmniScaleLegacy =>
                (9, include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_legacy_comp.metal"))),
            ScaleFilter::SuperXbr =>
                return self.run_super_xbr_compute(pixels, src_w, src_h),
            ScaleFilter::Edi =>
                (11, include_bytes!(concat!(env!("OUT_DIR"), "/edi_comp.metal"))),
            ScaleFilter::Nedi =>
                (12, include_bytes!(concat!(env!("OUT_DIR"), "/nedi_comp.metal"))),
            ScaleFilter::Dcci =>
                (13, include_bytes!(concat!(env!("OUT_DIR"), "/dcci_comp.metal"))),
            ScaleFilter::Mmpx =>
                (14, include_bytes!(concat!(env!("OUT_DIR"), "/mmpx_comp.metal"))),
            ScaleFilter::LcdGrid =>
                (15, include_bytes!(concat!(env!("OUT_DIR"), "/lcd_grid_comp.metal"))),
            ScaleFilter::Nearest =>
                (16, include_bytes!(concat!(env!("OUT_DIR"), "/nearest_comp.metal"))),
            ScaleFilter::Bilinear =>
                (17, include_bytes!(concat!(env!("OUT_DIR"), "/bilinear_comp.metal"))),
            ScaleFilter::Sai2x =>
                (18, include_bytes!(concat!(env!("OUT_DIR"), "/sai2x_comp.metal"))),
            ScaleFilter::Super2xSai =>
                (19, include_bytes!(concat!(env!("OUT_DIR"), "/super_sai2x_comp.metal"))),
            ScaleFilter::SuperEagle =>
                (20, include_bytes!(concat!(env!("OUT_DIR"), "/super_eagle_comp.metal"))),
            ScaleFilter::ScaleFx | ScaleFilter::ScaleFx9x =>
                return self.run_scalefx_compute(filter, pixels, src_w, src_h, disp_w, disp_h),
            _ => return None,
        };

        self.ensure_scale_compute(idx, msl)?;
        let pipeline = self.scale_compute[idx].as_deref()?;

        // Compute output dimensions
        let (out_w, out_h) = match filter {
            ScaleFilter::Eagle | ScaleFilter::SuperXbr |
            ScaleFilter::Epx | ScaleFilter::Scale2x |
            ScaleFilter::Edi | ScaleFilter::Nedi | ScaleFilter::Dcci
            | ScaleFilter::Mmpx
            | ScaleFilter::Sai2x | ScaleFilter::Super2xSai | ScaleFilter::SuperEagle
            => (src_w * 2, src_h * 2),
            ScaleFilter::LcdGrid => (src_w * 4, src_h * 4),
            ScaleFilter::Scale3x => (src_w * 3, src_h * 3),
            ScaleFilter::Scale4x => (src_w * 4, src_h * 4),
            ScaleFilter::Hqx(h) => { let f = h.factor(); (src_w * f, src_h * f) }
            ScaleFilter::Xbr(x) => { let f = x.factor(); (src_w * f, src_h * f) }
            ScaleFilter::Xbrz(x) => { let f = x.factor(); (src_w * f, src_h * f) }
            _ => {
                // Adaptive filters: compute aspect-correct output dimensions
                let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64).max(1.0);
                ((src_w as f64 * s).round() as u32, (src_h as f64 * s).round() as u32)
            }
        };

        // Create/resize output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            self.compute_out_tex = Some(make_texture(
                &self.device, out_w, out_h,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            ));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        let px_buf = make_buf(&self.device, pixels.as_ptr() as *const u8, pixels.len() * 4);

        // Build uniforms
        let iscale = if src_w > 0 { out_w / src_w } else { 1 };
        let extra = match filter {
            ScaleFilter::OmniScale => {
                let sx = src_w as f32 / out_w as f32;
                let sy = src_h as f32 / out_h as f32;
                f32::to_bits((sx * sx + sy * sy).sqrt())
            }
            ScaleFilter::Epx | ScaleFilter::Scale4x
            | ScaleFilter::Hqx(_) | ScaleFilter::Xbr(_) | ScaleFilter::Xbrz(_) => iscale,
            _ => 0,
        };
        let uniforms: [u32; 8] = [src_w, src_h, out_w, out_h, extra, 0, 0, 0];
        let uni_buf = make_buf(&self.device, uniforms.as_ptr() as *const u8, 32);

        // Dispatch
        let cmd = match self.command_queue.commandBuffer() { Some(c) => c, None => { log::error!("Metal: failed to create command buffer"); return None; } };
        let encoder = cmd.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&px_buf), 0, 1);
            encoder.setTexture_atIndex(Some(out_tex), 0);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: ((out_w + 15) / 16) as usize,
                    height: ((out_h + 15) / 16) as usize,
                    depth: 1,
                },
                MTLSize { width: 16, height: 16, depth: 1 },
            );
        }
        encoder.endEncoding();
        cmd.commit();

        Some((self.compute_out_tex.as_deref()?, out_w, out_h))
    }

    /// Super xBR 3-pass compute dispatch.
    /// Metal buffer layout: 0=uniforms, 1=pixels, 2=intermed, texture(0)=output
    fn run_super_xbr_compute(
        &mut self,
        pixels: &[u32],
        src_w: u32, src_h: u32,
    ) -> Option<(&ProtocolObject<dyn MTLTexture>, u32, u32)> {
        let out_w = src_w * 2;
        let out_h = src_h * 2;

        let msl: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_comp.metal"));
        self.ensure_scale_compute(10, msl)?;
        let pipeline = self.scale_compute[10].as_deref()?;

        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            self.compute_out_tex = Some(make_texture(
                &self.device, out_w, out_h,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            ));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        let px_buf = make_buf(&self.device, pixels.as_ptr() as *const u8, pixels.len() * 4);
        let intermed = make_buf_empty(&self.device, (out_w * out_h * 4) as usize);

        #[repr(C)]
        struct Uniforms { src_w: u32, src_h: u32, out_w: u32, out_h: u32, pass: u32, _pad: [u32; 3] }

        let cmd = self.command_queue.commandBuffer()?;
        let dispatch_x = ((out_w + 15) / 16) as usize;
        let dispatch_y = ((out_h + 15) / 16) as usize;

        for pass_idx in 0u32..3 {
            let unis = Uniforms { src_w, src_h, out_w, out_h, pass: pass_idx, _pad: [0; 3] };
            let uni_buf = make_buf(&self.device, &unis as *const _ as *const u8, 32);
            let encoder = cmd.computeCommandEncoder().unwrap();
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&px_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&intermed), 0, 2);
                encoder.setTexture_atIndex(Some(out_tex), 0);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: dispatch_x, height: dispatch_y, depth: 1 },
                    MTLSize { width: 16, height: 16, depth: 1 },
                );
            }
            encoder.endEncoding();
        }
        cmd.commit();

        Some((self.compute_out_tex.as_deref()?, out_w, out_h))
    }

    /// ScaleFX 5-pass (3x) or 10-pass (9x) compute dispatch.
    fn run_scalefx_compute(
        &mut self,
        filter: scaling::ScaleFilter,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        _disp_w: u32, _disp_h: u32,
    ) -> Option<(&ProtocolObject<dyn MTLTexture>, u32, u32)> {
        use scaling::ScaleFilter;
        let is_9x = matches!(filter, ScaleFilter::ScaleFx9x);
        let mid_w = src_w * 3;
        let mid_h = src_h * 3;
        let out_w = if is_9x { src_w * 9 } else { mid_w };
        let out_h = if is_9x { src_h * 9 } else { mid_h };

        let msl: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scalefx_comp.metal"));
        self.ensure_scale_compute(21, msl)?;
        let pipeline = self.scale_compute[21].as_deref()?;

        // Resize output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            self.compute_out_tex = Some(make_texture(
                &self.device, out_w, out_h,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            ));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        // Upload source pixels
        let px_buf = make_buf(&self.device, pixels.as_ptr() as *const u8, pixels.len() * 4);

        // Intermediate float4 buffers (sized for the larger pass in 9x mode)
        let max_pixels = if is_9x { mid_w * mid_h } else { src_w * src_h };
        let f4_size = (max_pixels as usize) * 16;
        let buf0 = make_buf_empty(&self.device, f4_size);
        let buf1 = make_buf_empty(&self.device, f4_size);
        let buf2 = make_buf_empty(&self.device, f4_size);
        let buf3 = make_buf_empty(&self.device, f4_size);

        // px_out / px_out2: packed XRGB output for chaining 9x.
        // Always sized for the pass 4 output — the shader writes to px_out unconditionally.
        let px_out_size = (mid_w * mid_h * 4) as usize;
        let px_out = make_buf_empty(&self.device, px_out_size);
        let px_out2_size = if is_9x { (out_w * out_h * 4) as usize } else { 4 };
        let px_out2 = make_buf_empty(&self.device, px_out2_size);

        #[repr(C)]
        struct Uniforms { src_w: u32, src_h: u32, out_w: u32, out_h: u32, pass: u32, _pad: [u32; 3] }

        let cmd = self.command_queue.commandBuffer()?;

        // Helper: dispatch 5 ScaleFX passes
        // Metal buffer layout: 0=uniforms, 1=pixels, 2=buf0, 3=buf1, 4=buf2, 5=buf3, 6=px_out, texture(0)=output
        let dispatch_5 = |cmd: &ProtocolObject<dyn MTLCommandBuffer>,
                          px_src: &ProtocolObject<dyn MTLBuffer>,
                          px_dst: &ProtocolObject<dyn MTLBuffer>,
                          sw: u32, sh: u32, ow: u32, oh: u32| {
            for pass_idx in 0u32..5 {
                let unis = Uniforms { src_w: sw, src_h: sh, out_w: ow, out_h: oh, pass: pass_idx, _pad: [0; 3] };
                let uni_buf = make_buf(&self.device, &unis as *const _ as *const u8, 32);
                let encoder = cmd.computeCommandEncoder().unwrap();
                encoder.setComputePipelineState(pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(px_src), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&buf0), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&buf1), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&buf2), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&buf3), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(px_dst), 0, 6);
                    encoder.setTexture_atIndex(Some(out_tex), 0);
                    let (dx, dy) = if pass_idx < 4 {
                        (((sw + 15) / 16) as usize, ((sh + 15) / 16) as usize)
                    } else {
                        (((ow + 15) / 16) as usize, ((oh + 15) / 16) as usize)
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize { width: dx, height: dy, depth: 1 },
                        MTLSize { width: 16, height: 16, depth: 1 },
                    );
                }
                encoder.endEncoding();
            }
        };

        // First 3x pass
        dispatch_5(&cmd, &px_buf, &px_out, src_w, src_h, mid_w, mid_h);

        if is_9x {
            // Second 3x pass: read from px_out, write to px_out2
            dispatch_5(&cmd, &px_out, &px_out2, mid_w, mid_h, out_w, out_h);
        }

        cmd.commit();
        Some((self.compute_out_tex.as_deref()?, out_w, out_h))
    }

    pub fn update_texture(&self, pixels: &[u32]) {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: self.tex_w as usize, height: self.tex_h as usize, depth: 1 },
        };
        unsafe {
            self.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
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
            ca.setClearColor(MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0 });

            let cmd_buf = match self.command_queue.commandBuffer() { Some(c) => c, None => { log::error!("Metal: failed to create command buffer"); return; } };
            let Some(encoder) = cmd_buf.renderCommandEncoderWithDescriptor(&rpd) else { return; };

            encoder.setRenderPipelineState(&self.pipeline_state);
            unsafe {
                encoder.setVertexBytes_length_atIndex(
                    NonNull::new_unchecked(viewport.as_ptr() as *mut _),
                    std::mem::size_of::<[f32; 4]>(),
                    0,
                );
                encoder.setFragmentTexture_atIndex(Some(&self.texture), 0);
            }
            unsafe { encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4); }
            encoder.endEncoding();

            cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));
            cmd_buf.commit();
        });
    }
}
