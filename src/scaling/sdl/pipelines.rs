//! SDL3 GPU pipeline manager.
//!
//! Encapsulates all GPU resources (device, textures, transfer buffers, shader
//! pipelines) behind a single struct, replacing the large destructured tuple
//! that previously lived in `main.rs`.

use sdl3::gpu;
use crate::scaling::ScaleFilter;
use crate::vectorize::rasterize::{GpuEdgeV2, GpuRowRange};

/// Index into `scale_pipelines` array for each compute-based scaling filter.
const SP_OMNISCALE: usize = 0;
const SP_EPX: usize = 1;
const SP_EAGLE: usize = 2;
const SP_SCALE3X: usize = 3;
const SP_BICUBIC: usize = 4;
const SP_AA_NEAREST: usize = 5;
const SP_HQX: usize = 6;
const SP_XBR: usize = 7;
const SP_XBRZ: usize = 8;
const SP_SUPER_XBR: usize = 9;
const SP_OMNISCALE_LEGACY: usize = 10;
const SP_EDI: usize = 11;
const SP_NEDI: usize = 12;
const SP_DCCI: usize = 13;
const SP_MMPX: usize = 14;
const SP_LCD_GRID: usize = 15;
const SP_NEAREST: usize = 16;
const SP_BILINEAR: usize = 17;

/// Owns the SDL3 GPU device and all lazily-initialized shader pipelines.
pub struct GpuPipelines {
    pub device: gpu::Device,
    pub tex: gpu::Texture<'static>,
    pub tex_w: u32,
    pub tex_h: u32,
    pub transfer_buf: gpu::TransferBuffer,
    pub transfer_buf_size: u32,
    pub sampler: gpu::Sampler,

    // Lazily-initialized graphics pipelines (fragment shaders — legacy, kept for reference)
    // All scaling filters now use compute pipelines instead.

    // Lazily-initialized compute pipelines (scaling filters)
    scale_pipelines: [Option<gpu::ComputePipeline>; 18],

    // Vectorize compute pipelines
    vectorize_compute: Option<gpu::ComputePipeline>,
    diffusion_compute: Option<gpu::ComputePipeline>,
    spline_diff: Option<(gpu::ComputePipeline, gpu::ComputePipeline)>,
    full_vectorize: Option<super::GpuVectorizePipelines>,
}

impl GpuPipelines {
    /// Create a new GPU pipeline manager for the given window and source dimensions.
    pub fn new(window: &sdl3::video::Window, src_w: u32, src_h: u32) -> Self {
        let all_formats = gpu::ShaderFormat::PRIVATE
            | gpu::ShaderFormat::SPIRV
            | gpu::ShaderFormat::MSL
            | gpu::ShaderFormat::DXBC
            | gpu::ShaderFormat::DXIL;
        let dev = gpu::Device::new(all_formats, false)
            .expect("Failed to create GPU device")
            .with_window(window)
            .expect("Failed to claim window for GPU device");
        let _ = dev.set_swapchain_parameters(
            window,
            gpu::PresentMode::Vsync,
            gpu::SwapchainComposition::Sdr,
        );
        let tex = super::create_texture(&dev, src_w, src_h);
        let max_xfer = src_w * src_h * 4;
        let xfer = dev
            .create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
            .with_size(max_xfer)
            .build()
            .expect("Failed to create transfer buffer");
        let sampler = dev
            .create_sampler(
                gpu::SamplerCreateInfo::new()
                    .with_min_filter(gpu::Filter::Nearest)
                    .with_mag_filter(gpu::Filter::Nearest),
            )
            .expect("Failed to create sampler");

        GpuPipelines {
            device: dev,
            tex,
            tex_w: src_w,
            tex_h: src_h,
            transfer_buf: xfer,
            transfer_buf_size: max_xfer,
            sampler,
            scale_pipelines: Default::default(),
            vectorize_compute: None,
            diffusion_compute: None,
            spline_diff: None,
            full_vectorize: None,
        }
    }

    /// Resize the GPU texture if dimensions changed.
    pub fn resize_texture(&mut self, w: u32, h: u32) {
        if w != self.tex_w || h != self.tex_h {
            self.tex_w = w;
            self.tex_h = h;
            self.tex = super::create_texture(&self.device, w, h);
        }
    }

    /// Ensure the transfer buffer is large enough for the given pixel count.
    pub fn ensure_transfer_buf(&mut self, needed: u32) {
        if needed > self.transfer_buf_size {
            self.transfer_buf = self.device
                .create_transfer_buffer()
                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
                .with_size(needed)
                .build()
                .expect("transfer buf");
            self.transfer_buf_size = needed;
        }
    }

    /// Lazily initialize the GPU pipeline for the given filter.
    /// Returns true if the filter has a GPU pipeline available.
    /// `force_cpu` prevents initialization of shader-based pipelines.
    pub fn ensure_pipeline(
        &mut self,
        filter: ScaleFilter,
        window: &sdl3::video::Window,
        force_cpu: bool,
    ) -> GpuRenderMode {
        if force_cpu {
            return GpuRenderMode::Cpu;
        }
        // Map scaling filters to compute pipeline index + init function
        let scale_idx: Option<(usize, fn(&gpu::Device) -> Option<gpu::ComputePipeline>)> = match filter {
            ScaleFilter::OmniScale =>
                Some((SP_OMNISCALE, super::init_omniscale_compute_pipeline)),
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x =>
                Some((SP_EPX, super::init_epx_compute_pipeline)),
            ScaleFilter::Eagle =>
                Some((SP_EAGLE, super::init_eagle_compute_pipeline)),
            ScaleFilter::Scale3x =>
                Some((SP_SCALE3X, super::init_scale3x_compute_pipeline)),
            ScaleFilter::Bicubic =>
                Some((SP_BICUBIC, super::init_bicubic_compute_pipeline)),
            ScaleFilter::AaNearestNeighbor =>
                Some((SP_AA_NEAREST, super::init_aa_nearest_compute_pipeline)),
            ScaleFilter::Hqx(_) =>
                Some((SP_HQX, super::init_hqx_compute_pipeline)),
            ScaleFilter::Xbr(_) =>
                Some((SP_XBR, super::init_xbr_compute_pipeline)),
            ScaleFilter::Xbrz(_) =>
                Some((SP_XBRZ, super::init_xbrz_compute_pipeline)),
            ScaleFilter::SuperXbr =>
                Some((SP_SUPER_XBR, super::init_super_xbr_compute_pipeline)),
            ScaleFilter::OmniScaleLegacy =>
                Some((SP_OMNISCALE_LEGACY, super::init_omniscale_legacy_compute_pipeline)),
            ScaleFilter::Edi =>
                Some((SP_EDI, super::init_edi_compute_pipeline)),
            ScaleFilter::Nedi =>
                Some((SP_NEDI, super::init_nedi_compute_pipeline)),
            ScaleFilter::Dcci =>
                Some((SP_DCCI, super::init_dcci_compute_pipeline)),
            ScaleFilter::Mmpx =>
                Some((SP_MMPX, super::init_mmpx_compute_pipeline)),
            ScaleFilter::LcdGrid =>
                Some((SP_LCD_GRID, super::init_lcd_grid_compute_pipeline)),
            ScaleFilter::Nearest =>
                Some((SP_NEAREST, super::init_nearest_compute_pipeline)),
            ScaleFilter::Bilinear =>
                Some((SP_BILINEAR, super::init_bilinear_compute_pipeline)),
            _ => None,
        };

        if let Some((idx, init_fn)) = scale_idx {
            if self.scale_pipelines[idx].is_none() {
                self.scale_pipelines[idx] = init_fn(&self.device);
            }
            return if self.scale_pipelines[idx].is_some() {
                GpuRenderMode::ScaleCompute
            } else {
                GpuRenderMode::Cpu
            };
        }

        match filter {
            ScaleFilter::VectorizeLegacy | ScaleFilter::VectorizeLegacyAdaptive => {
                if self.vectorize_compute.is_none() {
                    self.vectorize_compute =
                        super::init_vectorize_compute_pipeline(&self.device);
                }
                if self.vectorize_compute.is_some() {
                    GpuRenderMode::Vectorize
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::VectorizeDiffusion => {
                if self.diffusion_compute.is_none() {
                    self.diffusion_compute =
                        super::init_diffusion_compute_pipeline(&self.device);
                }
                if self.diffusion_compute.is_some() {
                    GpuRenderMode::Diffusion
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::VectorizeSplineDiffusion
            | ScaleFilter::VectorizeSplineDiffusionAdaptive => {
                if self.spline_diff.is_none() {
                    self.spline_diff =
                        super::init_spline_diffusion_pipelines(&self.device);
                }
                if self.spline_diff.is_some() {
                    GpuRenderMode::SplineDiffusion
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Vectorize | ScaleFilter::VectorizeAdaptive => {
                if self.vectorize_compute.is_none() {
                    self.vectorize_compute =
                        super::init_vectorize_compute_pipeline(&self.device);
                }
                if self.vectorize_compute.is_some() {
                    GpuRenderMode::VectorizeSharedChain
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::VectorizeGpu => {
                if self.full_vectorize.is_none() {
                    self.full_vectorize = super::init_full_gpu_pipeline(&self.device);
                }
                if self.full_vectorize.is_some() {
                    GpuRenderMode::FullGpuVectorize
                } else {
                    GpuRenderMode::Cpu
                }
            }
            _ => GpuRenderMode::Cpu,
        }
    }

    /// Render using a compute scaling filter. Computes uniforms and dispatches.
    pub fn render_scale_compute(
        &mut self,
        filter: ScaleFilter,
        window: &sdl3::video::Window,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
    ) {
        // Determine which pipeline index this filter uses
        let idx = match filter {
            ScaleFilter::OmniScale => SP_OMNISCALE,
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x => SP_EPX,
            ScaleFilter::Eagle => SP_EAGLE,
            ScaleFilter::Scale3x => SP_SCALE3X,
            ScaleFilter::Bicubic => SP_BICUBIC,
            ScaleFilter::AaNearestNeighbor => SP_AA_NEAREST,
            ScaleFilter::Hqx(_) => SP_HQX,
            ScaleFilter::Xbr(_) => SP_XBR,
            ScaleFilter::Xbrz(_) => SP_XBRZ,
            ScaleFilter::SuperXbr => SP_SUPER_XBR,
            ScaleFilter::OmniScaleLegacy => SP_OMNISCALE_LEGACY,
            ScaleFilter::Edi => SP_EDI,
            ScaleFilter::Nedi => SP_NEDI,
            ScaleFilter::Dcci => SP_DCCI,
            ScaleFilter::Mmpx => SP_MMPX,
            ScaleFilter::LcdGrid => SP_LCD_GRID,
            _ => return,
        };

        // Integer-scale filters render at native dimensions
        let (out_w, out_h) = match filter {
            ScaleFilter::Eagle | ScaleFilter::SuperXbr
            | ScaleFilter::Edi | ScaleFilter::Nedi | ScaleFilter::Dcci
            | ScaleFilter::Mmpx => (src_w * 2, src_h * 2),
            ScaleFilter::LcdGrid => (src_w * 4, src_h * 4),
            ScaleFilter::Scale3x => (src_w * 3, src_h * 3),
            ScaleFilter::Epx | ScaleFilter::Scale2x => (src_w * 2, src_h * 2),
            ScaleFilter::Scale4x => (src_w * 4, src_h * 4),
            ScaleFilter::Hqx(h) => { let f = h.factor(); (src_w * f, src_h * f) }
            ScaleFilter::Xbr(x) => { let f = x.factor(); (src_w * f, src_h * f) }
            ScaleFilter::Xbrz(x) => { let f = x.factor(); (src_w * f, src_h * f) }
            _ => (out_w, out_h), // resolution-independent filters use display size
        };

        self.resize_texture(out_w, out_h);
        let pipeline = self.scale_pipelines[idx].as_ref().unwrap();

        // Super xBR uses a special 3-pass dispatch
        if matches!(filter, ScaleFilter::SuperXbr) {
            super::super_xbr_compute_and_blit(
                &self.device, window, &self.tex, pipeline,
                pixels, src_w, src_h, out_w, out_h,
            );
            return;
        }

        // Build uniforms: [src_w, src_h, out_w, out_h, extra, 0, 0, 0]
        // The 5th u32 is filter-specific: iscale for integer filters, pixel_size for omniscale
        let extra = match filter {
            ScaleFilter::OmniScale => {
                let sx = src_w as f32 / out_w as f32;
                let sy = src_h as f32 / out_h as f32;
                f32::to_bits((sx * sx + sy * sy).sqrt())
            }
            ScaleFilter::Epx | ScaleFilter::Scale4x
            | ScaleFilter::Hqx(_) | ScaleFilter::Xbr(_) | ScaleFilter::Xbrz(_)
            | ScaleFilter::LcdGrid => {
                out_w / src_w
            }
            _ => 0,
        };
        let uniforms = [src_w, src_h, out_w, out_h, extra, 0, 0, 0];

        super::scale_compute_and_blit(
            &self.device, window, &self.tex, pipeline,
            pixels, out_w, out_h, &uniforms,
        );
    }

    /// Render with simple texture blit (nearest/bilinear).
    pub fn render_blit(
        &mut self,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        window: &sdl3::video::Window,
        filter_mode: gpu::Filter,
    ) {
        self.resize_texture(src_w, src_h);
        let needed = src_w * src_h * 4;
        self.ensure_transfer_buf(needed);
        super::upload_and_blit(
            &self.device, window, &self.tex, &self.transfer_buf,
            pixels, src_w, src_h, filter_mode,
        );
    }

    /// Full vectorize render path: prepare edges, upload to GPU, blit to window.
    pub fn render_vectorize_to_window(
        &mut self,
        window: &sdl3::video::Window,
        gpu_edges: &[GpuEdgeV2],
        row_ranges: &[GpuRowRange],
        edge_indices: &[u32],
        out_w: u32,
        out_h: u32,
        bg_color: u32,
    ) {
        self.resize_texture(out_w, out_h);
        super::vectorize_and_blit(
            &self.device,
            window,
            &self.tex,
            self.vectorize_compute.as_ref().unwrap(),
            gpu_edges,
            row_ranges,
            edge_indices,
            out_w,
            out_h,
            bg_color,
        );
    }

    /// Run the full GPU vectorize pipeline (all 5 stages on GPU, no CPU readback).
    pub fn render_full_vectorize_to_window(
        &mut self,
        window: &sdl3::video::Window,
        pixels: &[u32],
        img_w: u32, img_h: u32,
        out_w: u32, out_h: u32,
        scale: f32,
    ) {
        if self.full_vectorize.is_none() {
            self.full_vectorize = super::init_full_gpu_pipeline(&self.device);
        }
        self.resize_texture(out_w, out_h);
        // Need to borrow pipelines and tex separately from self
        let pipelines = self.full_vectorize.as_mut().unwrap();
        super::gpu_vectorize_full_pipeline(
            &self.device, window, &self.tex, pipelines,
            pixels, img_w, img_h, out_w, out_h, scale,
        );
    }

    /// Full diffusion render path.
    pub fn render_diffusion_to_window(
        &mut self,
        window: &sdl3::video::Window,
        src_pixels: &[u32],
        src_regions: &[u32],
        diag_states: &[u32],
        sw: u32,
        sh: u32,
        out_w: u32,
        out_h: u32,
        scale: f32,
    ) {
        self.resize_texture(out_w, out_h);
        super::diffusion_and_blit(
            &self.device,
            window,
            &self.tex,
            self.diffusion_compute.as_ref().unwrap(),
            src_pixels,
            src_regions,
            diag_states,
            sw,
            sh,
            out_w,
            out_h,
            scale,
        );
    }

    /// Full spline-diffusion render path.
    pub fn render_spline_diffusion_to_window(
        &mut self,
        window: &sdl3::video::Window,
        gpu_edges: &[GpuEdgeV2],
        row_ranges: &[GpuRowRange],
        edge_indices: &[u32],
        src_pixels: &[u32],
        out_w: u32,
        out_h: u32,
        sw: u32,
        sh: u32,
        bg_color: u32,
        scale: u32,
    ) {
        self.resize_texture(out_w, out_h);
        let (p1, p2) = self.spline_diff.as_ref().unwrap();
        super::spline_diffusion_and_blit(
            &self.device,
            window,
            &self.tex,
            p1,
            p2,
            gpu_edges,
            row_ranges,
            edge_indices,
            src_pixels,
            out_w,
            out_h,
            sw,
            sh,
            bg_color,
            scale,
        );
    }

    /// Upload pre-scaled CPU pixels and blit to the window.
    pub fn upload_and_blit(
        &mut self,
        pixels: &[u32],
        w: u32,
        h: u32,
        window: &sdl3::video::Window,
    ) {
        self.resize_texture(w, h);
        let needed = w * h * 4;
        self.ensure_transfer_buf(needed);
        super::upload_and_blit(
            &self.device,
            window,
            &self.tex,
            &self.transfer_buf,
            pixels,
            w,
            h,
            gpu::Filter::Nearest,
        );
    }
}

/// Result of `ensure_pipeline` — tells the caller which render path to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuRenderMode {
    /// Use GPU texture blit (nearest/bilinear — no shader).
    Native,
    /// Use GPU compute scaling filter pipeline.
    ScaleCompute,
    /// Use GPU compute vectorize pipeline.
    Vectorize,
    /// Use GPU compute diffusion pipeline.
    Diffusion,
    /// Use GPU compute spline-diffusion pipeline.
    SplineDiffusion,
    /// Use GPU compute shared-chain vectorize pipeline.
    VectorizeSharedChain,
    /// Full GPU vectorize pipeline (all stages on GPU).
    FullGpuVectorize,
    /// No GPU pipeline available; use CPU scaling + upload_and_blit.
    Cpu,
}
