//! SDL3 GPU pipeline manager.
//!
//! Encapsulates all GPU resources (device, textures, transfer buffers, shader
//! pipelines) behind a single struct, replacing the large destructured tuple
//! that previously lived in `main.rs`.

use sdl3::gpu;
use super::ScaleFilter;
use crate::vectorize::rasterize::{GpuEdgeV2, GpuRowRange};

/// Owns the SDL3 GPU device and all lazily-initialized shader pipelines.
pub struct GpuPipelines {
    pub device: gpu::Device,
    pub tex: gpu::Texture<'static>,
    pub tex_w: u32,
    pub tex_h: u32,
    pub transfer_buf: gpu::TransferBuffer,
    pub transfer_buf_size: u32,
    pub sampler: gpu::Sampler,

    // Lazily-initialized graphics pipelines (fragment shaders)
    omniscale: Option<gpu::GraphicsPipeline>,
    hqx: Option<gpu::GraphicsPipeline>,
    bicubic: Option<gpu::GraphicsPipeline>,
    omniscale_legacy: Option<gpu::GraphicsPipeline>,
    scale3x: Option<gpu::GraphicsPipeline>,
    eagle: Option<gpu::GraphicsPipeline>,
    aa_nearest: Option<gpu::GraphicsPipeline>,
    epx: Option<gpu::GraphicsPipeline>,
    xbr: Option<gpu::GraphicsPipeline>,
    xbrz: Option<gpu::GraphicsPipeline>,
    super_xbr: Option<gpu::GraphicsPipeline>,

    // Lazily-initialized compute pipelines
    vectorize_compute: Option<gpu::ComputePipeline>,
    diffusion_compute: Option<gpu::ComputePipeline>,
    spline_diff: Option<(gpu::ComputePipeline, gpu::ComputePipeline)>,
    optimizer_compute: Option<gpu::ComputePipeline>,
    full_vectorize: Option<super::gpu::GpuVectorizePipelines>,
}

/// References to GPU device and optimizer pipeline for passing to vectorization.
pub struct GpuOptRefs<'a> {
    pub device: &'a gpu::Device,
    pub pipeline: &'a gpu::ComputePipeline,
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
        if dev
            .set_swapchain_parameters(
                window,
                gpu::PresentMode::Mailbox,
                gpu::SwapchainComposition::Sdr,
            )
            .is_err()
        {
            let _ = dev.set_swapchain_parameters(
                window,
                gpu::PresentMode::Immediate,
                gpu::SwapchainComposition::Sdr,
            );
        }
        let tex = super::gpu::create_texture(&dev, src_w, src_h);
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
            omniscale: None,
            hqx: None,
            bicubic: None,
            omniscale_legacy: None,
            scale3x: None,
            eagle: None,
            aa_nearest: None,
            epx: None,
            xbr: None,
            xbrz: None,
            super_xbr: None,
            vectorize_compute: None,
            diffusion_compute: None,
            spline_diff: None,
            optimizer_compute: None,
            full_vectorize: None,
        }
    }

    /// Resize the GPU texture if dimensions changed.
    pub fn resize_texture(&mut self, w: u32, h: u32) {
        if w != self.tex_w || h != self.tex_h {
            self.tex_w = w;
            self.tex_h = h;
            self.tex = super::gpu::create_texture(&self.device, w, h);
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
        match filter {
            ScaleFilter::Nearest | ScaleFilter::Bilinear => GpuRenderMode::Native,
            ScaleFilter::OmniScale => {
                if self.omniscale.is_none() {
                    self.omniscale =
                        super::gpu::init_omniscale_pipeline(&self.device, window);
                }
                if self.omniscale.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Hqx(_) => {
                if self.hqx.is_none() {
                    self.hqx = super::gpu::init_hqx_pipeline(&self.device, window);
                }
                if self.hqx.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Bicubic => {
                if self.bicubic.is_none() {
                    self.bicubic =
                        super::gpu::init_bicubic_pipeline(&self.device, window);
                }
                if self.bicubic.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::OmniScaleLegacy => {
                if self.omniscale_legacy.is_none() {
                    self.omniscale_legacy =
                        super::gpu::init_omniscale_legacy_pipeline(&self.device, window);
                }
                if self.omniscale_legacy.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Scale3x => {
                if self.scale3x.is_none() {
                    self.scale3x =
                        super::gpu::init_scale3x_pipeline(&self.device, window);
                }
                if self.scale3x.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Eagle => {
                if self.eagle.is_none() {
                    self.eagle =
                        super::gpu::init_eagle_pipeline(&self.device, window);
                }
                if self.eagle.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::AaNearestNeighbor => {
                if self.aa_nearest.is_none() {
                    self.aa_nearest =
                        super::gpu::init_aa_nearest_pipeline(&self.device, window);
                }
                if self.aa_nearest.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x => {
                if self.epx.is_none() {
                    self.epx = super::gpu::init_epx_pipeline(&self.device, window);
                }
                if self.epx.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Xbr(_) => {
                if self.xbr.is_none() {
                    self.xbr = super::gpu::init_xbr_pipeline(&self.device, window);
                }
                if self.xbr.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Xbrz(_) => {
                if self.xbrz.is_none() {
                    self.xbrz =
                        super::gpu::init_xbrz_pipeline(&self.device, window);
                }
                if self.xbrz.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::SuperXbr => {
                if self.super_xbr.is_none() {
                    self.super_xbr =
                        super::gpu::init_super_xbr_pipeline(&self.device, window);
                }
                if self.super_xbr.is_some() {
                    GpuRenderMode::Native
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::VectorizeLegacy | ScaleFilter::VectorizeLegacyAdaptive => {
                if self.vectorize_compute.is_none() {
                    self.vectorize_compute =
                        super::gpu::init_vectorize_compute_pipeline(&self.device);
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
                        super::gpu::init_diffusion_compute_pipeline(&self.device);
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
                        super::gpu::init_spline_diffusion_pipelines(&self.device);
                }
                if self.spline_diff.is_some() {
                    GpuRenderMode::SplineDiffusion
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::Vectorize | ScaleFilter::VectorizeAdaptive => {
                // Reuses the vectorize compute pipeline — same winding-number
                // fill shader, just different input data (shared-chain paths).
                if self.vectorize_compute.is_none() {
                    self.vectorize_compute =
                        super::gpu::init_vectorize_compute_pipeline(&self.device);
                }
                if self.vectorize_compute.is_some() {
                    GpuRenderMode::VectorizeSharedChain
                } else {
                    GpuRenderMode::Cpu
                }
            }
            ScaleFilter::VectorizeGpu => {
                if self.full_vectorize.is_none() {
                    self.full_vectorize = super::gpu::init_full_gpu_pipeline(&self.device);
                }
                if self.full_vectorize.is_some() {
                    GpuRenderMode::FullGpuVectorize
                } else {
                    GpuRenderMode::Cpu
                }
            }
            // Filters without GPU shaders fall back to CPU
            _ => GpuRenderMode::Cpu,
        }
    }

    /// Render a frame using the appropriate GPU shader pipeline.
    /// Assumes `ensure_pipeline` was called and returned a non-Cpu mode.
    pub fn render_native(
        &mut self,
        filter: ScaleFilter,
        pixels: &[u32],
        src_w: u32,
        src_h: u32,
        window: &sdl3::video::Window,
    ) {
        self.resize_texture(src_w, src_h);
        let needed = src_w * src_h * 4;
        self.ensure_transfer_buf(needed);

        match filter {
            ScaleFilter::Scale3x => {
                super::gpu::render_scale3x(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.scale3x.as_ref().unwrap(), &self.sampler,
                );
            }
            ScaleFilter::Eagle => {
                super::gpu::render_eagle(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.eagle.as_ref().unwrap(), &self.sampler,
                );
            }
            ScaleFilter::AaNearestNeighbor => {
                let (ww, wh) = window.size();
                super::gpu::render_aa_nearest(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.aa_nearest.as_ref().unwrap(), &self.sampler,
                    ww, wh,
                );
            }
            ScaleFilter::Bicubic => {
                let (ww, wh) = window.size();
                super::gpu::render_bicubic(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.bicubic.as_ref().unwrap(), &self.sampler,
                    ww, wh,
                );
            }
            ScaleFilter::OmniScaleLegacy => {
                super::gpu::render_omniscale_legacy(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.omniscale_legacy.as_ref().unwrap(), &self.sampler,
                );
            }
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x => {
                let epx_scale = match filter {
                    ScaleFilter::Scale4x => 4.0,
                    _ => 2.0,
                };
                super::gpu::render_epx(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.epx.as_ref().unwrap(), &self.sampler, epx_scale,
                );
            }
            ScaleFilter::Hqx(h) => {
                let hqx_scale = h.factor() as f32;
                super::gpu::render_hqx(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.hqx.as_ref().unwrap(), &self.sampler, hqx_scale,
                );
            }
            ScaleFilter::Xbr(x) => {
                let xbr_scale = x.factor() as f32;
                super::gpu::render_xbr(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.xbr.as_ref().unwrap(), &self.sampler, xbr_scale,
                );
            }
            ScaleFilter::Xbrz(x) => {
                let xbrz_scale = x.factor() as f32;
                super::gpu::render_xbrz(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.xbrz.as_ref().unwrap(), &self.sampler, xbrz_scale,
                );
            }
            ScaleFilter::SuperXbr => {
                super::gpu::render_super_xbr(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h,
                    self.super_xbr.as_ref().unwrap(), &self.sampler,
                );
            }
            ScaleFilter::OmniScale => {
                if let Some(ref pipeline) = self.omniscale {
                    super::gpu::render_omniscale(
                        &self.device, window, &self.tex, &self.transfer_buf,
                        pixels, src_w, src_h, pipeline, &self.sampler,
                    );
                } else {
                    super::gpu::upload_and_blit(
                        &self.device, window, &self.tex, &self.transfer_buf,
                        pixels, src_w, src_h, gpu::Filter::Nearest,
                    );
                }
            }
            ScaleFilter::Bilinear => {
                super::gpu::upload_and_blit(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h, gpu::Filter::Linear,
                );
            }
            _ => {
                // Nearest or unsupported — plain blit
                super::gpu::upload_and_blit(
                    &self.device, window, &self.tex, &self.transfer_buf,
                    pixels, src_w, src_h, gpu::Filter::Nearest,
                );
            }
        }
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
        super::gpu::vectorize_and_blit(
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

    /// Initialize and return GPU optimizer references for use by vectorization.
    /// Returns None if pipeline creation fails.
    pub fn gpu_optimizer(&mut self) -> Option<GpuOptRefs> {
        if self.optimizer_compute.is_none() {
            self.optimizer_compute = super::gpu::init_optimizer_compute_pipeline(&self.device);
        }
        self.optimizer_compute.as_ref().map(|p| GpuOptRefs {
            device: &self.device,
            pipeline: p,
        })
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
            self.full_vectorize = super::gpu::init_full_gpu_pipeline(&self.device);
        }
        self.resize_texture(out_w, out_h);
        // Need to borrow pipelines and tex separately from self
        let pipelines = self.full_vectorize.as_ref().unwrap();
        super::gpu::gpu_vectorize_full_pipeline(
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
        super::gpu::diffusion_and_blit(
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
        super::gpu::spline_diffusion_and_blit(
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
        super::gpu::upload_and_blit(
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
    /// Use GPU fragment shader pipeline (render_native).
    Native,
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
