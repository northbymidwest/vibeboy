use metal::*;
use objc::rc::autoreleasepool;

use super::scaling;
use super::vectorize;
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

pub(super) struct MetalRenderer {
    pub device: Device,
    pub layer: MetalLayer,
    pub command_queue: CommandQueue,
    pipeline_state: RenderPipelineState,
    pub texture: Texture,
    pub tex_w: u32,
    pub tex_h: u32,
    // Compute scaling pipelines (lazily initialized)
    scale_compute: [Option<ComputePipelineState>; 14],
    pub compute_out_tex: Option<Texture>,
    pub compute_out_w: u32,
    pub compute_out_h: u32,
    // Full GPU vectorize pipeline
    pub vectorize_pipeline: Option<MetalVectorizePipeline>,
    // GPU scanline rasterizer for vectorize filters
    scanline_rasterizer: Option<ComputePipelineState>,
    // Diffusion rasterizer (single-pass)
    diffusion_pipeline: Option<ComputePipelineState>,
    // Spline-diffusion (2-pass: vectorize_to_buf + spline_diffusion)
    spline_diff_pass1: Option<ComputePipelineState>,
    spline_diff_pass2: Option<ComputePipelineState>,
}

fn safe_buf(dev: &Device, data: &[u8]) -> Buffer {
    if data.is_empty() {
        dev.new_buffer(4, MTLResourceOptions::StorageModeShared)
    } else {
        dev.new_buffer_with_data(data.as_ptr() as *const _, data.len() as u64, MTLResourceOptions::StorageModeShared)
    }
}

impl MetalRenderer {
    pub fn new(tex_w: u32, tex_h: u32) -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = device
            .new_library_with_source(METAL_SHADERS, &CompileOptions::new())
            .expect("Failed to compile Metal shaders");
        let vert_fn = library.get_function("vertex_main", None).unwrap();
        let frag_fn = library.get_function("fragment_main", None).unwrap();

        let pipeline_desc = RenderPipelineDescriptor::new();
        pipeline_desc.set_vertex_function(Some(&vert_fn));
        pipeline_desc.set_fragment_function(Some(&frag_fn));
        pipeline_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let pipeline_state = device
            .new_render_pipeline_state(&pipeline_desc)
            .expect("Failed to create render pipeline state");

        // Create framebuffer texture
        let tex_desc = TextureDescriptor::new();
        tex_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        tex_desc.set_width(tex_w as u64);
        tex_desc.set_height(tex_h as u64);
        tex_desc.set_usage(MTLTextureUsage::ShaderRead);

        let texture = device.new_texture(&tex_desc);

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
            let src = std::str::from_utf8(msl).ok()?;
            let lib = self.device.new_library_with_source(src, &CompileOptions::new()).ok()?;
            let func = lib.get_function("main0", None).ok()?;
            self.scanline_rasterizer = Some(
                self.device.new_compute_pipeline_state_with_function(&func).ok()?
            );
        }
        let pipeline = self.scanline_rasterizer.as_ref()?;

        // Ensure output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            let desc = TextureDescriptor::new();
            desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            desc.set_width(out_w as u64);
            desc.set_height(out_h as u64);
            desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            self.compute_out_tex = Some(self.device.new_texture(&desc));
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
        let uni_buf = self.device.new_buffer_with_data(
            uniforms.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);

        // Dispatch
        // MSL buffer order (after remap): 0=uniforms, 1=edges, 2=row_ranges, 3=edge_indices
        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&uni_buf), 0);
        enc.set_buffer(1, Some(&edge_buf), 0);
        enc.set_buffer(2, Some(&row_buf), 0);
        enc.set_buffer(3, Some(&idx_buf), 0);
        enc.set_texture(0, Some(out_tex));
        enc.dispatch_thread_groups(
            MTLSize::new(((out_w + 15) / 16) as u64, ((out_h + 15) / 16) as u64, 1),
            MTLSize::new(16, 16, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

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
            let src = std::str::from_utf8(msl).ok()?;
            let lib = self.device.new_library_with_source(src, &CompileOptions::new()).ok()?;
            let func = lib.get_function("main0", None).ok()?;
            self.diffusion_pipeline = Some(
                self.device.new_compute_pipeline_state_with_function(&func).ok()?);
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
            let desc = TextureDescriptor::new();
            desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            desc.set_width(out_w as u64);
            desc.set_height(out_h as u64);
            desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            self.compute_out_tex = Some(self.device.new_texture(&desc));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        let px_buf = self.device.new_buffer_with_data(
            pixels.as_ptr() as *const _, (pixels.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let reg_buf = self.device.new_buffer_with_data(
            regions.as_ptr() as *const _, (regions.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let diag_buf = self.device.new_buffer_with_data(
            diags.as_ptr() as *const _, (diags.len() * 4) as u64, MTLResourceOptions::StorageModeShared);

        // MSL: buffer(0)=uniforms, buffer(1)=diag_states, buffer(2)=regions, buffer(3)=pixels
        let inv_scale = 1.0f32 / scale as f32;
        let uniforms: [u32; 8] = [out_w, out_h, src_w, src_h,
            f32::to_bits(inv_scale), f32::to_bits(2.5), f32::to_bits(2.0), 0];
        let uni_buf = self.device.new_buffer_with_data(
            uniforms.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared);

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&uni_buf), 0);
        enc.set_buffer(1, Some(&diag_buf), 0);
        enc.set_buffer(2, Some(&reg_buf), 0);
        enc.set_buffer(3, Some(&px_buf), 0);
        enc.set_texture(0, Some(out_tex));
        enc.dispatch_thread_groups(
            MTLSize::new(((out_w + 15) / 16) as u64, ((out_h + 15) / 16) as u64, 1),
            MTLSize::new(16, 16, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

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
            let src = std::str::from_utf8(msl).ok()?;
            let lib = self.device.new_library_with_source(src, &CompileOptions::new()).ok()?;
            let func = lib.get_function("main0", None).ok()?;
            self.spline_diff_pass1 = Some(
                self.device.new_compute_pipeline_state_with_function(&func).ok()?);
        }
        if self.spline_diff_pass2.is_none() {
            let msl = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.metal"));
            let src = std::str::from_utf8(msl).ok()?;
            let lib = self.device.new_library_with_source(src, &CompileOptions::new()).ok()?;
            let func = lib.get_function("main0", None).ok()?;
            self.spline_diff_pass2 = Some(
                self.device.new_compute_pipeline_state_with_function(&func).ok()?);
        }
        let p1 = self.spline_diff_pass1.as_ref()?;
        let p2 = self.spline_diff_pass2.as_ref()?;

        // Ensure output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            let desc = TextureDescriptor::new();
            desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            desc.set_width(out_w as u64);
            desc.set_height(out_h as u64);
            desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            self.compute_out_tex = Some(self.device.new_texture(&desc));
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

        let edge_buf = self.device.new_buffer_with_data(
            edge_bytes.as_ptr() as *const _, edge_bytes.len().max(4) as u64, MTLResourceOptions::StorageModeShared);
        let row_buf = self.device.new_buffer_with_data(
            row_bytes.as_ptr() as *const _, row_bytes.len().max(4) as u64, MTLResourceOptions::StorageModeShared);
        let idx_buf = self.device.new_buffer_with_data(
            idx_bytes.as_ptr() as *const _, idx_bytes.len().max(4) as u64, MTLResourceOptions::StorageModeShared);
        let px_buf = self.device.new_buffer_with_data(
            pixels.as_ptr() as *const _, (pixels.len() * 4) as u64, MTLResourceOptions::StorageModeShared);

        // Intermediate region buffer
        let region_buf = self.device.new_buffer(
            (out_w * out_h * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.command_queue.new_command_buffer();

        // Pass 1: vectorize_to_buf — scanline rasterize edges -> region_buf
        // MSL: buffer(0)=uniforms, buffer(1)=edges(rw), buffer(2)=rows(rw),
        //      buffer(3)=indices, buffer(4)=region_out(rw)
        {
            let uni1: [u32; 4] = [out_w, out_h, edges.len() as u32, bg_color];
            let uni_buf = self.device.new_buffer_with_data(
                uni1.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p1);
            enc.set_buffer(0, Some(&uni_buf), 0);
            enc.set_buffer(1, Some(&edge_buf), 0);
            enc.set_buffer(2, Some(&row_buf), 0);
            enc.set_buffer(3, Some(&idx_buf), 0);
            enc.set_buffer(4, Some(&region_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(((out_w + 15) / 16) as u64, ((out_h + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        // Pass 2: spline_diffusion — Gaussian blending from region_buf -> output texture
        // MSL: buffer(0)=uniforms, buffer(1)=pixels, buffer(2)=region_colors
        {
            let inv_scale = 1.0f32 / scale as f32;
            let uni2: [u32; 8] = [out_w, out_h, src_w, src_h,
                f32::to_bits(inv_scale), f32::to_bits(2.5), f32::to_bits(2.0), scale];
            let uni_buf = self.device.new_buffer_with_data(
                uni2.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p2);
            enc.set_buffer(0, Some(&uni_buf), 0);
            enc.set_buffer(1, Some(&px_buf), 0);
            enc.set_buffer(2, Some(&region_buf), 0);
            enc.set_texture(0, Some(out_tex));
            enc.dispatch_thread_groups(
                MTLSize::new(((out_w + 15) / 16) as u64, ((out_h + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        self.tex_w = out_w;
        self.tex_h = out_h;
        self.texture = out_tex.clone();
        Some(())
    }

    pub fn ensure_scale_compute(&mut self, idx: usize, msl: &[u8]) -> Option<&ComputePipelineState> {
        if self.scale_compute[idx].is_none() && !msl.is_empty() {
            let msl_str = std::str::from_utf8(msl).ok()?;
            let lib = self.device.new_library_with_source(msl_str, &CompileOptions::new()).ok()?;
            let func = lib.get_function("main0", None).ok()?;
            let pipeline = self.device.new_compute_pipeline_state_with_function(&func).ok()?;
            self.scale_compute[idx] = Some(pipeline);
        }
        self.scale_compute[idx].as_ref()
    }

    /// Run a compute scaling filter and return the output texture.
    /// Returns (texture_ref, out_w, out_h) or None if the filter isn't GPU-accelerated.
    pub fn run_scale_compute(
        &mut self,
        filter: scaling::ScaleFilter,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        disp_w: u32, disp_h: u32,
    ) -> Option<(&Texture, u32, u32)> {
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
            _ => return None,
        };

        self.ensure_scale_compute(idx, msl)?;
        let pipeline = self.scale_compute[idx].as_ref()?;

        // Compute output dimensions (integer scale for fixed-factor filters)
        let (out_w, out_h) = match filter {
            ScaleFilter::Eagle | ScaleFilter::SuperXbr |
            ScaleFilter::Epx | ScaleFilter::Scale2x |
            ScaleFilter::Edi | ScaleFilter::Nedi | ScaleFilter::Dcci => (src_w * 2, src_h * 2),
            ScaleFilter::Scale3x => (src_w * 3, src_h * 3),
            ScaleFilter::Scale4x => (src_w * 4, src_h * 4),
            ScaleFilter::Hqx(h) => { let f = h.factor(); (src_w * f, src_h * f) }
            ScaleFilter::Xbr(x) => { let f = x.factor(); (src_w * f, src_h * f) }
            ScaleFilter::Xbrz(x) => { let f = x.factor(); (src_w * f, src_h * f) }
            _ => (disp_w, disp_h),
        };

        // Create/resize output texture
        if self.compute_out_w != out_w || self.compute_out_h != out_h {
            let desc = TextureDescriptor::new();
            desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            desc.set_width(out_w as u64);
            desc.set_height(out_h as u64);
            desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            self.compute_out_tex = Some(self.device.new_texture(&desc));
            self.compute_out_w = out_w;
            self.compute_out_h = out_h;
        }
        let out_tex = self.compute_out_tex.as_ref()?;

        // Create pixel buffer
        let px_buf = self.device.new_buffer_with_data(
            pixels.as_ptr() as *const _,
            (pixels.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

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
        let uni_buf = self.device.new_buffer_with_data(
            uniforms.as_ptr() as *const _,
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch
        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&uni_buf), 0);
        encoder.set_buffer(1, Some(&px_buf), 0);
        encoder.set_texture(0, Some(out_tex));
        let tw = MTLSize::new(16, 16, 1);
        let tg = MTLSize::new(
            ((out_w + 15) / 16) as u64,
            ((out_h + 15) / 16) as u64,
            1,
        );
        encoder.dispatch_thread_groups(tg, tw);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some((self.compute_out_tex.as_ref()?, out_w, out_h))
    }

    pub fn update_texture(&self, pixels: &[u32]) {
        let region = MTLRegion::new_2d(0, 0, self.tex_w as u64, self.tex_h as u64);
        self.texture.replace_region(
            region,
            0,
            pixels.as_ptr() as *const _,
            (self.tex_w * 4) as u64,
        );
    }

    pub fn render(&self) {
        autoreleasepool(|| {
            let drawable = match self.layer.next_drawable() {
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
                // Window wider than texture: pillarbox
                (2.0 * tex_aspect / dst_aspect, 2.0)
            } else {
                // Window taller than texture: letterbox
                (2.0, 2.0 * dst_aspect / tex_aspect)
            };
            let ndc_x = -ndc_w / 2.0;
            let ndc_y = -ndc_h / 2.0;
            let viewport: [f32; 4] = [ndc_x, ndc_y, ndc_w, ndc_h];

            let rpd = RenderPassDescriptor::new();
            let ca = rpd.color_attachments().object_at(0).unwrap();
            ca.set_texture(Some(dst_tex));
            ca.set_load_action(MTLLoadAction::Clear);
            ca.set_store_action(MTLStoreAction::Store);
            ca.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_render_command_encoder(rpd);

            encoder.set_render_pipeline_state(&self.pipeline_state);
            encoder.set_vertex_bytes(
                0,
                std::mem::size_of::<[f32; 4]>() as u64,
                viewport.as_ptr() as *const _,
            );
            encoder.set_fragment_texture(0, Some(&self.texture));
            encoder.draw_primitives(MTLPrimitiveType::TriangleStrip, 0, 4);
            encoder.end_encoding();

            cmd_buf.present_drawable(&drawable);
            cmd_buf.commit();
        });
    }
}
