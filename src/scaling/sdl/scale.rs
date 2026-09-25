//! Scaling-filter compute pipelines: creation, cached per-frame buffers and
//! dispatch. Which shader a filter uses and how it is dispatched comes from
//! the filter registry (`ScaleFilter::gpu()`); this module only knows how to
//! load the bytecode and record the passes on SDL3 GPU.

use super::common::*;
use crate::scaling::{GpuPass, GpuShader, ScaleShader, multipass_uniforms};
use sdl3::gpu;

macro_rules! sdl_shader_code {
    ($($variant:ident => $module:literal),* $(,)?) => {
        /// SPIR-V, MSL and DXIL bytecode of a scaling shader.
        fn shader_code(shader: ScaleShader) -> (&'static [u8], &'static [u8], &'static [u8]) {
            match shader {
                $(ScaleShader::$variant => (
                    include_bytes!(concat!(env!("OUT_DIR"), "/", $module, "_comp.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/", $module, "_comp.metal")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/", $module, "_comp.dxil")),
                )),*
            }
        }
    };
}
crate::scale_shader_list!(sdl_shader_code);

/// Create the compute pipeline for a filter's shader.
///
/// All scaling shaders share one descriptor layout: 1 readonly storage
/// buffer (pixels), 1 readwrite storage texture (output), 1 uniform buffer,
/// plus the readwrite storage buffers of multi-pass kinds (1 for Super xBR,
/// 5 for ScaleFX).
pub fn init_scale_pipeline(device: &gpu::Device, gpu: &GpuShader) -> Option<gpu::ComputePipeline> {
    let (spirv, msl, dxil) = shader_code(gpu.shader);
    let label = gpu.shader.module();
    let rw_buffers = gpu.pass.rw_storage_buffers();
    let build = |format, code: &[u8], entry: &std::ffi::CStr| {
        device
            .create_compute_pipeline()
            .with_code(format, code)
            .with_entrypoint(entry)
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(1)
            .with_readwrite_storage_buffers(rw_buffers)
            .with_readwrite_storage_textures(1)
            .with_thread_count(16, 16, 1)
            .build()
    };
    let pipeline = build(gpu::ShaderFormat::SPIRV, spirv, c"main")
        .or_else(|_| {
            if !dxil.is_empty() {
                build(gpu::ShaderFormat::DXIL, dxil, c"main")
            } else {
                Err(sdl3::get_error())
            }
        })
        .or_else(|_| build(gpu::ShaderFormat::MSL, msl, c"main_0"));

    match pipeline {
        Ok(p) => {
            eprintln!("{label} compute pipeline ready");
            Some(p)
        }
        Err(e) => {
            eprintln!("{label} compute pipeline failed: {e}");
            None
        }
    }
}

/// GPU buffers for the scaling filters, kept across frames and re-created
/// only when a required size changes.
#[derive(Default)]
pub struct ScaleBufCache {
    px_xfer: Option<(u32, gpu::TransferBuffer)>,
    px_buf: Option<(u32, gpu::Buffer)>,
    /// Read-write intermediates of the multi-pass kinds: Super xBR's one
    /// buffer, or ScaleFX's buf0..buf3, px_out and px_out2.
    rw_bufs: Vec<(u32, gpu::Buffer)>,
}

impl ScaleBufCache {
    /// Upload `pixels` into the cached source storage buffer.
    fn upload(&mut self, device: &gpu::Device, cmd: &gpu::CommandBuffer, pixels: &[u32]) {
        let px_bytes =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        let px_size = px_bytes.len().max(4) as u32;

        if self.px_xfer.as_ref().is_none_or(|(s, _)| *s != px_size) {
            let xfer = device
                .create_transfer_buffer()
                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
                .with_size(px_size)
                .build()
                .expect("px transfer");
            self.px_xfer = Some((px_size, xfer));
        }
        if self.px_buf.as_ref().is_none_or(|(s, _)| *s != px_size) {
            let buf = device
                .create_buffer()
                .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)
                .with_size(px_size)
                .build()
                .expect("px buf");
            self.px_buf = Some((px_size, buf));
        }
        let (_, xfer) = self.px_xfer.as_ref().unwrap();
        let (_, buf) = self.px_buf.as_ref().unwrap();

        {
            let mut map = xfer.map::<u8>(device, true);
            map.mem_mut()[..px_bytes.len()].copy_from_slice(px_bytes);
            map.unmap();
        }
        let cp = device.begin_copy_pass(cmd).expect("copy pass");
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(xfer),
            gpu::BufferRegion::new().with_buffer(buf).with_size(px_size),
            true,
        );
        device.end_copy_pass(cp);
    }

    /// Read-write intermediates with the given byte sizes.
    fn rw_bufs(&mut self, device: &gpu::Device, sizes: &[u32]) -> Vec<gpu::Buffer> {
        let matches = self.rw_bufs.len() == sizes.len()
            && self.rw_bufs.iter().zip(sizes).all(|((s, _), n)| s == n);
        if !matches {
            let rw = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ
                | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE;
            self.rw_bufs = sizes
                .iter()
                .map(|&size| {
                    let buf = device
                        .create_buffer()
                        .with_usage(rw)
                        .with_size(size)
                        .build()
                        .expect("scale intermediate buf");
                    (size, buf)
                })
                .collect();
        }
        self.rw_bufs.iter().map(|(_, b)| b.clone()).collect()
    }
}

/// Record the pixel upload and every compute dispatch of a scaling filter
/// into `cmd`, leaving the `out_w` x `out_h` result in `out_tex`.
pub(super) fn encode_scale(
    device: &gpu::Device,
    cmd: &gpu::CommandBuffer,
    pipeline: &gpu::ComputePipeline,
    shader: &GpuShader,
    cache: &mut ScaleBufCache,
    out_tex: &gpu::Texture,
    pixels: &[u32],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
) {
    cache.upload(device, cmd, pixels);
    let px_buf = cache.px_buf.as_ref().unwrap().1.clone();

    // One compute pass per dispatch. The output texture and the read-write
    // buffers are cycled on the first dispatch of the frame only; later
    // dispatches read what earlier ones wrote.
    let dispatch = |src: &gpu::Buffer,
                    rw: &[gpu::Buffer],
                    uniforms: [u32; 8],
                    groups: (u32, u32),
                    first: bool| {
        let rw_bindings: Vec<_> = rw
            .iter()
            .map(|b| {
                gpu::StorageBufferReadWriteBinding::new()
                    .with_buffer(b)
                    .with_cycle(first)
            })
            .collect();
        let cp = device
            .begin_compute_pass(
                cmd,
                &[gpu::StorageTextureReadWriteBinding::new()
                    .with_texture(out_tex)
                    .with_cycle(first)],
                &rw_bindings,
            )
            .expect("compute pass");
        cp.bind_compute_pipeline(pipeline);
        cp.bind_compute_storage_buffers(0, std::slice::from_ref(src));
        #[repr(C)]
        struct RawUniforms([u32; 8]);
        cmd.push_compute_uniform_data(0, &RawUniforms(uniforms));
        cp.dispatch(groups.0, groups.1, 1);
        device.end_compute_pass(cp);
    };
    let groups = |w: u32, h: u32| (w.div_ceil(16), h.div_ceil(16));

    match shader.pass {
        GpuPass::Single => {
            let uniforms = shader.uniforms(src_w, src_h, out_w, out_h);
            dispatch(&px_buf, &[], uniforms, groups(out_w, out_h), true);
        }
        GpuPass::SuperXbr => {
            // Intermediate buffer (out_w * out_h * 4 bytes)
            let rw = cache.rw_bufs(device, &[(out_w * out_h * 4).max(4)]);
            for pass in 0u32..3 {
                let uniforms = multipass_uniforms(src_w, src_h, out_w, out_h, pass);
                dispatch(&px_buf, &rw, uniforms, groups(out_w, out_h), pass == 0);
            }
        }
        GpuPass::ScaleFx { chained } => {
            // For 9x: intermediate buffers need to be sized for the larger (3x) second pass.
            // First pass: src_w × src_h intermediates, 3x output.
            // Second pass: (src_w*3) × (src_h*3) intermediates, 9x output.
            let mid_w = src_w * 3;
            let mid_h = src_h * 3;
            let max_intermediate = if chained {
                mid_w * mid_h
            } else {
                src_w * src_h
            };
            let buf_size = (max_intermediate * 16).max(16);
            // px_out: pass 4 writes packed XRGB pixels here; for 9x, second pass reads this as input
            // px_out2: separate write target for the second pass (avoids RO/RW aliasing on px_out)
            // Always sized for the first pass 4 output; the shader writes unconditionally.
            let px_out_size = (mid_w * mid_h * 4).max(4);
            let px_out2_size = if chained {
                (out_w * out_h * 4).max(4)
            } else {
                4
            };
            let bufs = cache.rw_bufs(
                device,
                &[
                    buf_size,
                    buf_size,
                    buf_size,
                    buf_size,
                    px_out_size,
                    px_out2_size,
                ],
            );
            let (intermediates, outs) = bufs.split_at(4);
            let (px_out, px_out2) = (&outs[0], &outs[1]);

            // Five ScaleFX passes: 0-3 over the source, 4 over the 3x output.
            let dispatch_5 = |px_src: &gpu::Buffer,
                              px_dst: &gpu::Buffer,
                              sw: u32,
                              sh: u32,
                              ow: u32,
                              oh: u32,
                              first: bool| {
                let mut rw = intermediates.to_vec();
                rw.push(px_dst.clone());
                for pass in 0u32..5 {
                    let g = if pass < 4 {
                        groups(sw, sh)
                    } else {
                        groups(ow, oh)
                    };
                    let uniforms = multipass_uniforms(sw, sh, ow, oh, pass);
                    dispatch(px_src, &rw, uniforms, g, first && pass == 0);
                }
            };

            // First 3x pass: writes packed output to px_out
            dispatch_5(&px_buf, px_out, src_w, src_h, mid_w, mid_h, true);
            if chained {
                // Second 3x pass: reads px_out, writes to px_out2 (separate buffer to avoid aliasing)
                dispatch_5(px_out, px_out2, mid_w, mid_h, out_w, out_h, false);
            }
        }
    }
}

/// Dispatch a scaling filter's compute shader(s) and blit the result to the
/// swapchain.
pub fn scale_compute_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipeline: &gpu::ComputePipeline,
    shader: &GpuShader,
    cache: &mut ScaleBufCache,
    pixels: &[u32],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    encode_scale(
        device, &cmd, pipeline, shader, cache, gpu_tex, pixels, src_w, src_h, out_w, out_h,
    );

    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let (vx, vy, vw, vh) = aspect_viewport(out_w, out_h, sw_w, sw_h);
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = out_w;
        blit_info.source.h = out_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = vx as u32;
        blit_info.destination.y = vy as u32;
        blit_info.destination.w = vw as u32;
        blit_info.destination.h = vh as u32;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(gpu::Filter::Linear as i32);
        unsafe {
            sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info);
        }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}
