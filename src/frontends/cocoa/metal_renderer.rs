use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{NSString, ns_string};
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use super::scaling;
use super::vectorize;

/// Gaussian blur kernel width for diffusion rasterizers.
const GAUSS_K: f32 = 2.5;
/// Gaussian blur radius for diffusion rasterizers.
const GAUSS_RADIUS: f32 = 2.0;
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
    scale_compute: [Option<ComputePipeline>; 21],
    pub compute_out_tex: Option<Texture>,
    pub compute_out_w: u32,
    pub compute_out_h: u32,
    // Full GPU vectorize pipeline
    pub vectorize_pipeline: Option<MetalVectorizePipeline>,
    // GPU scanline rasterizer for vectorize filters
    scanline_rasterizer: Option<ComputePipeline>,
    // Diffusion rasterizer (single-pass)
    diffusion_pipeline: Option<ComputePipeline>,
    // Spline-diffusion (2-pass: vectorize_to_buf + spline_diffusion)
    spline_diff_pass1: Option<ComputePipeline>,
    spline_diff_pass2: Option<ComputePipeline>,
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
    let func_name = ns_string!("main0");
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
            scanline_rasterizer: None,
            diffusion_pipeline: None,
            spline_diff_pass1: None,
            spline_diff_pass2: None,
        }
    }

    /// Run the GPU scanline rasterizer on pre-computed vectorize edges.
    pub fn run_scanline_rasterize(
        &mut self,
        edges: &[vectorize::rasterize::GpuEdgeV2],
        row_ranges: &[vectorize::rasterize::GpuRowRange],
        edge_indices: &[u32],
        out_w: u32, out_h: u32,
        bg_color: u32,
    ) -> Option<()> {
        if out_w == 0 || out_h == 0 { return None; }
        // Lazily init pipeline
        if self.scanline_rasterizer.is_none() {
            let msl = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.metal"));
            self.scanline_rasterizer = load_compute_pipeline(&self.device, msl);
        }
        let pipeline = self.scanline_rasterizer.as_ref()?;

        // Ensure output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            self.compute_out_tex = Some(make_texture(
                &self.device, out_w, out_h,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            ));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        // Upload edge data
        let edge_bytes = unsafe {
            std::slice::from_raw_parts(edges.as_ptr() as *const u8,
                edges.len() * std::mem::size_of::<vectorize::rasterize::GpuEdgeV2>())
        };
        let row_bytes = unsafe {
            std::slice::from_raw_parts(row_ranges.as_ptr() as *const u8,
                row_ranges.len() * std::mem::size_of::<vectorize::rasterize::GpuRowRange>())
        };
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(edge_indices.as_ptr() as *const u8,
                edge_indices.len() * 4)
        };

        let edge_buf = safe_buf(&self.device, edge_bytes);
        let row_buf = safe_buf(&self.device, row_bytes);
        let idx_buf = safe_buf(&self.device, idx_bytes);

        // Uniforms: out_w, out_h, num_edges, bg_color
        let uniforms: [u32; 4] = [out_w, out_h, edges.len() as u32, bg_color];
        let uni_buf = make_buf(&self.device, uniforms.as_ptr() as *const u8, 16);

        // Dispatch
        let cmd = match self.command_queue.commandBuffer() { Some(c) => c, None => { log::error!("Metal: failed to create command buffer"); return None; } };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&edge_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&row_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&idx_buf), 0, 3);
            enc.setTexture_atIndex(Some(out_tex), 0);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: ((out_w + 15) / 16) as usize, height: ((out_h + 15) / 16) as usize, depth: 1 },
                MTLSize { width: 16, height: 16, depth: 1 },
            );
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        self.tex_w = out_w;
        self.tex_h = out_h;
        self.texture = out_tex.clone();
        Some(())
    }

    /// Run the diffusion rasterizer (single-pass Gaussian blending).
    pub fn run_diffusion_rasterize(
        &mut self,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        scale: u32,
    ) -> Option<()> {
        if out_w == 0 || out_h == 0 { return None; }
        use vectorize::rasterize::build_graph_regions;
        use vectorize::graph;

        // Lazily init pipeline
        if self.diffusion_pipeline.is_none() {
            let msl = include_bytes!(concat!(env!("OUT_DIR"), "/diffusion_raster_comp.metal"));
            self.diffusion_pipeline = load_compute_pipeline(&self.device, msl);
        }
        let pipeline = self.diffusion_pipeline.as_ref()?;

        // Prepare data on CPU
        let g = graph::build(pixels, src_w as usize, src_h as usize);
        let regions = build_graph_regions(src_w as usize, src_h as usize, &g);
        let mut diags = vec![0u32; (src_w * src_h) as usize];
        for py in 0..src_h as usize {
            for px in 0..src_w as usize {
                let tl = Self::corner_diag(&g, px, py) as u32;
                let tr = Self::corner_diag(&g, px + 1, py) as u32;
                let br = Self::corner_diag(&g, px + 1, py + 1) as u32;
                let bl = Self::corner_diag(&g, px, py + 1) as u32;
                diags[py * src_w as usize + px] = tl | (tr << 2) | (br << 4) | (bl << 6);
            }
        }

        // Ensure output texture
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
        let reg_buf = make_buf(&self.device, regions.as_ptr() as *const u8, regions.len() * 4);
        let diag_buf = make_buf(&self.device, diags.as_ptr() as *const u8, diags.len() * 4);

        let inv_scale = 1.0f32 / scale as f32;
        let uniforms: [u32; 8] = [out_w, out_h, src_w, src_h,
            f32::to_bits(inv_scale), f32::to_bits(GAUSS_K), f32::to_bits(GAUSS_RADIUS), 0];
        let uni_buf = make_buf(&self.device, uniforms.as_ptr() as *const u8, 32);

        let cmd = match self.command_queue.commandBuffer() { Some(c) => c, None => { log::error!("Metal: failed to create command buffer"); return None; } };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&diag_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&reg_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&px_buf), 0, 3);
            enc.setTexture_atIndex(Some(out_tex), 0);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: ((out_w + 15) / 16) as usize, height: ((out_h + 15) / 16) as usize, depth: 1 },
                MTLSize { width: 16, height: 16, depth: 1 },
            );
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        self.tex_w = out_w;
        self.tex_h = out_h;
        self.texture = out_tex.clone();
        Some(())
    }

    fn corner_diag(g: &vectorize::graph::SimilarityGraph, cx: usize, cy: usize) -> u8 {
        if cx == 0 || cy == 0 || cx >= g.width || cy >= g.height { return 0; }
        if g.edge(cx - 1, cy - 1).down_right { return 1; }
        if g.edge(cx, cy - 1).down_left { return 2; }
        0
    }

    /// Run spline-diffusion (2-pass: vectorize_to_buf -> spline_diffusion).
    pub fn run_spline_diffusion(
        &mut self,
        edges: &[vectorize::rasterize::GpuEdgeV2],
        row_ranges: &[vectorize::rasterize::GpuRowRange],
        edge_indices: &[u32],
        pixels: &[u32],
        out_w: u32, out_h: u32,
        src_w: u32, src_h: u32,
        bg_color: u32,
        scale: u32,
    ) -> Option<()> {
        // Lazily init pipelines
        if self.spline_diff_pass1.is_none() {
            let msl = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_to_buf_comp.metal"));
            self.spline_diff_pass1 = load_compute_pipeline(&self.device, msl);
        }
        if self.spline_diff_pass2.is_none() {
            let msl = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.metal"));
            self.spline_diff_pass2 = load_compute_pipeline(&self.device, msl);
        }
        let p1 = self.spline_diff_pass1.as_ref()?;
        let p2 = self.spline_diff_pass2.as_ref()?;

        // Ensure output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            self.compute_out_tex = Some(make_texture(
                &self.device, out_w, out_h,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            ));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        // Upload buffers
        let edge_bytes = unsafe { std::slice::from_raw_parts(
            edges.as_ptr() as *const u8,
            edges.len() * std::mem::size_of::<vectorize::rasterize::GpuEdgeV2>()) };
        let row_bytes = unsafe { std::slice::from_raw_parts(
            row_ranges.as_ptr() as *const u8,
            row_ranges.len() * std::mem::size_of::<vectorize::rasterize::GpuRowRange>()) };
        let idx_bytes = unsafe { std::slice::from_raw_parts(
            edge_indices.as_ptr() as *const u8, edge_indices.len() * 4) };

        let edge_buf = safe_buf(&self.device, edge_bytes);
        let row_buf = safe_buf(&self.device, row_bytes);
        let idx_buf = safe_buf(&self.device, idx_bytes);
        let px_buf = make_buf(&self.device, pixels.as_ptr() as *const u8, pixels.len() * 4);

        // Intermediate region buffer
        let region_buf = make_buf_empty(&self.device, (out_w * out_h * 4).max(4) as usize);

        let cmd = match self.command_queue.commandBuffer() { Some(c) => c, None => { log::error!("Metal: failed to create command buffer"); return None; } };

        // Pass 1: vectorize_to_buf
        {
            let uni1: [u32; 4] = [out_w, out_h, edges.len() as u32, bg_color];
            let uni_buf = make_buf(&self.device, uni1.as_ptr() as *const u8, 16);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(p1);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&edge_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&row_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&idx_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&region_buf), 0, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((out_w + 15) / 16) as usize, height: ((out_h + 15) / 16) as usize, depth: 1 },
                    MTLSize { width: 16, height: 16, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Pass 2: spline_diffusion
        {
            let inv_scale = 1.0f32 / scale as f32;
            let uni2: [u32; 8] = [out_w, out_h, src_w, src_h,
                f32::to_bits(inv_scale), f32::to_bits(GAUSS_K), f32::to_bits(GAUSS_RADIUS), scale];
            let uni_buf = make_buf(&self.device, uni2.as_ptr() as *const u8, 32);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(p2);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&px_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&region_buf), 0, 2);
                enc.setTexture_atIndex(Some(out_tex), 0);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((out_w + 15) / 16) as usize, height: ((out_h + 15) / 16) as usize, depth: 1 },
                    MTLSize { width: 16, height: 16, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        self.tex_w = out_w;
        self.tex_h = out_h;
        self.texture = out_tex.clone();
        Some(())
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
            ScaleFilter::AaNearestNeighbor =>
                (5, include_bytes!(concat!(env!("OUT_DIR"), "/aa_nearest_comp.metal"))),
            ScaleFilter::Hqx(_) =>
                (6, include_bytes!(concat!(env!("OUT_DIR"), "/hqx_comp.metal"))),
            ScaleFilter::Xbr(_) =>
                (7, include_bytes!(concat!(env!("OUT_DIR"), "/xbr_comp.metal"))),
            ScaleFilter::Xbrz(_) =>
                (8, include_bytes!(concat!(env!("OUT_DIR"), "/xbrz_comp.metal"))),
            ScaleFilter::OmniScaleLegacy =>
                (9, include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_legacy_comp.metal"))),
            ScaleFilter::SuperXbr =>
                (10, include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_comp.metal"))),
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
            _ => (disp_w, disp_h),
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
        cmd.waitUntilCompleted();

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
