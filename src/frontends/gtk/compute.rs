//! GPU compute scaling for the GTK4 frontend via wgpu.
//!
//! Uses wgpu's GL backend with the GtkGLArea's GL context for zero-copy
//! compute → display. The compute output texture is a GL texture in the
//! same context, which the blit shader can bind directly.

use crate::scaling::wgpu_scale::{WgpuScaleFilter, WgpuScalePipeline};
use crate::scaling::ScaleFilter;
use glow::HasContext;

pub struct GpuCompute {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: WgpuScalePipeline,
    /// Last output dimensions (for cache invalidation).
    last_out_w: u32,
    last_out_h: u32,
}

impl GpuCompute {
    /// Create a wgpu device using the GL backend from the current GL context.
    /// Must be called while GtkGLArea's GL context is current (e.g., in `connect_realize`).
    pub fn new(gl_loader: impl FnMut(&str) -> *const std::ffi::c_void) -> Option<Self> {
        // Create wgpu gles adapter from the current GL context
        let hal_adapter = unsafe {
            wgpu::hal::gles::Adapter::new_external(
                gl_loader,
                wgpu::GlBackendOptions::default(),
            )?
        };

        // Create a minimal wgpu Instance as container for the external adapter.
        // Use Backends::empty() to avoid wgpu trying to init its own EGL display
        // (which produces MESA warnings since GTK already owns the GL context).
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::empty(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = unsafe { instance.create_adapter_from_hal(hal_adapter) };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gtk-gl-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .ok()?;

        let pipeline = WgpuScalePipeline::new(&device);

        eprintln!("GPU compute initialized (GL shared context)");

        Some(GpuCompute {
            instance,
            device,
            queue,
            pipeline,
            last_out_w: 0,
            last_out_h: 0,
        })
    }

    /// Run a scaling filter. Returns the GL texture ID and dimensions of the output.
    /// Must be called while the GtkGLArea's GL context is current.
    pub fn scale(
        &mut self,
        filter: WgpuScaleFilter,
        pixels: &[u32],
        src_w: u32,
        src_h: u32,
        fit_w: u32,
        fit_h: u32,
    ) -> Option<(glow::Texture, u32, u32)> {
        let (out_w, out_h) = filter.native_size(src_w, src_h, fit_w, fit_h);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let output_tex = self.pipeline.encode(
            &self.device,
            &self.queue,
            &mut encoder,
            filter,
            pixels,
            src_w,
            src_h,
            out_w,
            out_h,
        );

        // Extract the GL texture ID from the wgpu texture
        let gl_texture = unsafe {
            let hal_tex = output_tex.as_hal::<wgpu::hal::api::Gles>()?;
            match &hal_tex.inner {
                wgpu::hal::gles::TextureInner::Texture { raw, .. } => Some(*raw),
                _ => None,
            }
        }?;

        self.queue.submit(std::iter::once(encoder.finish()));
        // Wait for compute to finish so the texture is ready for GL sampling
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        self.last_out_w = out_w;
        self.last_out_h = out_h;

        Some((gl_texture, out_w, out_h))
    }
}

/// Map a ScaleFilter to a WgpuScaleFilter, if it has a GPU compute path.
pub fn to_wgpu_filter(filter: ScaleFilter) -> Option<WgpuScaleFilter> {
    match filter {
        ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x => {
            Some(WgpuScaleFilter::Epx)
        }
        ScaleFilter::Eagle => Some(WgpuScaleFilter::Eagle),
        ScaleFilter::Scale3x => Some(WgpuScaleFilter::Scale3x),
        ScaleFilter::Bicubic => Some(WgpuScaleFilter::Bicubic),
        ScaleFilter::AaNearestNeighbor => Some(WgpuScaleFilter::AaNearest),
        ScaleFilter::OmniScale => Some(WgpuScaleFilter::OmniScale),
        ScaleFilter::OmniScaleLegacy => Some(WgpuScaleFilter::OmniScaleLegacy),
        ScaleFilter::Hqx(_) => Some(WgpuScaleFilter::Hqx),
        ScaleFilter::Xbr(_) => Some(WgpuScaleFilter::Xbr),
        ScaleFilter::Xbrz(_) => Some(WgpuScaleFilter::Xbrz),
        ScaleFilter::SuperXbr => Some(WgpuScaleFilter::SuperXbr),
        ScaleFilter::Nedi => Some(WgpuScaleFilter::Nedi),
        ScaleFilter::Dcci => Some(WgpuScaleFilter::Dcci),
        ScaleFilter::Edi => Some(WgpuScaleFilter::Edi),
        ScaleFilter::Mmpx => Some(WgpuScaleFilter::Mmpx),
        ScaleFilter::LcdGrid => Some(WgpuScaleFilter::LcdGrid),
        _ => None,
    }
}
