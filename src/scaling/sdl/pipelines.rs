//! SDL3 GPU pipeline manager.
//!
//! Encapsulates all GPU resources (device, textures, transfer buffers, shader
//! pipelines) behind a single struct, replacing the large destructured tuple
//! that previously lived in `main.rs`.

use crate::scaling::{ScaleFilter, ScaleShader};
use sdl3::gpu;

/// Owns the SDL3 GPU device and all lazily-initialized shader pipelines.
pub struct GpuPipelines {
    pub device: gpu::Device,
    pub tex: gpu::Texture<'static>,
    pub tex_w: u32,
    pub tex_h: u32,
    pub transfer_buf: gpu::TransferBuffer,
    pub transfer_buf_size: u32,
    pub sampler: gpu::Sampler,

    // Lazily-initialized compute pipelines (scaling filters), per shader
    scale_pipelines: [Option<gpu::ComputePipeline>; ScaleShader::COUNT],
    // Per-frame scaling buffers, re-created only on size change
    scale_bufs: super::ScaleBufCache,

    // Full GPU vectorize pipeline
    full_vectorize: Option<super::GpuVectorizePipelines>,
}

impl GpuPipelines {
    /// Create a new GPU pipeline manager for the given window and source dimensions.
    pub fn new(window: &sdl3::video::Window, src_w: u32, src_h: u32) -> Self {
        let formats = if std::env::var("VIBEBOY_FORCE_VULKAN").is_ok() {
            // Request only SPIRV to force Vulkan backend on Windows
            gpu::ShaderFormat::SPIRV
        } else {
            gpu::ShaderFormat::PRIVATE
                | gpu::ShaderFormat::SPIRV
                | gpu::ShaderFormat::MSL
                | gpu::ShaderFormat::DXBC
                | gpu::ShaderFormat::DXIL
        };
        let dev = gpu::Device::new(formats, false)
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
            scale_bufs: Default::default(),
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
            self.transfer_buf = self
                .device
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
    pub fn ensure_pipeline(&mut self, filter: ScaleFilter, force_cpu: bool) -> GpuRenderMode {
        if force_cpu {
            return GpuRenderMode::Cpu;
        }
        if let Some(shader) = filter.gpu() {
            let idx = shader.shader.index();
            if self.scale_pipelines[idx].is_none() {
                self.scale_pipelines[idx] = super::init_scale_pipeline(&self.device, shader);
            }
            return if self.scale_pipelines[idx].is_some() {
                GpuRenderMode::ScaleCompute
            } else {
                GpuRenderMode::Cpu
            };
        }

        match filter {
            ScaleFilter::Vectorize => {
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
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) {
        let Some(shader) = filter.gpu() else {
            return;
        };

        // Integer-scale filters render at native dimensions; resolution-
        // independent filters use the display size.
        let (out_w, out_h) = filter.output_size(src_w, src_h, (out_w, out_h));

        self.resize_texture(out_w, out_h);
        let pipeline = self.scale_pipelines[shader.shader.index()]
            .as_ref()
            .unwrap();

        super::scale_compute_and_blit(
            &self.device,
            window,
            &self.tex,
            pipeline,
            shader,
            &mut self.scale_bufs,
            pixels,
            src_w,
            src_h,
            out_w,
            out_h,
        );
    }

    /// Run the full GPU vectorize pipeline (all stages on GPU, no CPU readback).
    pub fn render_full_vectorize_to_window(
        &mut self,
        window: &sdl3::video::Window,
        pixels: &[u32],
        img_w: u32,
        img_h: u32,
        out_w: u32,
        out_h: u32,
        scale: f32,
    ) {
        if self.full_vectorize.is_none() {
            self.full_vectorize = super::init_full_gpu_pipeline(&self.device);
        }
        self.resize_texture(out_w, out_h);
        // Need to borrow pipelines and tex separately from self
        let pipelines = self.full_vectorize.as_mut().unwrap();
        super::gpu_vectorize_full_pipeline(
            &self.device,
            window,
            &self.tex,
            pipelines,
            pixels,
            img_w,
            img_h,
            out_w,
            out_h,
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
    /// Use GPU compute scaling filter pipeline.
    ScaleCompute,
    /// Full GPU vectorize pipeline (all stages on GPU).
    FullGpuVectorize,
    /// No GPU pipeline available; use CPU scaling + upload_and_blit.
    Cpu,
}
