//\! Compute pipeline init and dispatch for vectorize, diffusion, and spline-diffusion.

use sdl3::gpu;
use super::common::*;

// ── Vectorize compute pipeline ──────────────────────────────────────────────

pub fn init_vectorize_compute_pipeline(device: &gpu::Device) -> Option<gpu::ComputePipeline> {
    let comp_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.spv"));
    let comp_msl = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.metal"));
    let comp_dxil = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.dxil"));

    let pipeline = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, comp_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(3)
        .with_readwrite_storage_textures(1)
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| if !comp_dxil.is_empty() {
            device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::DXIL, comp_dxil)
                .with_entrypoint(c"main")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(3)
                .with_readwrite_storage_textures(1)
                .with_thread_count(16, 16, 1)
                .build()
        } else { Err(sdl3::get_error()) })
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, comp_msl)
            .with_entrypoint(c"main_0")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(3)
            .with_readwrite_storage_textures(1)
            .with_thread_count(16, 16, 1)
            .build());

    match pipeline {
        Ok(p) => { eprintln!("Vectorize GPU compute pipeline ready"); Some(p) }
        Err(e) => { eprintln!("Vectorize GPU compute: pipeline creation failed: {e}"); None }
    }
}

pub fn vectorize_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipeline: &gpu::ComputePipeline,
    edges: &[crate::vectorize::rasterize::GpuEdgeV2],
    row_ranges: &[crate::vectorize::rasterize::GpuRowRange],
    edge_indices: &[u32],
    out_w: u32, out_h: u32,
    bg_color: u32,
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");

    fn upload_buffer(
        device: &gpu::Device, _cmd: &gpu::CommandBuffer,
        data: &[u8], usage: gpu::BufferUsageFlags,
    ) -> (gpu::TransferBuffer, gpu::Buffer) {
        let size = data.len().max(4) as u32;
        let transfer = device.create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
            .with_size(size)
            .build().expect("transfer buf");
        {
            let mut map = transfer.map::<u8>(device, true);
            map.mem_mut()[..data.len()].copy_from_slice(data);
            map.unmap();
        }
        let buf = device.create_buffer()
            .with_usage(usage)
            .with_size(size)
            .build().expect("storage buf");
        (transfer, buf)
    }

    let edge_bytes = unsafe {
        std::slice::from_raw_parts(
            edges.as_ptr() as *const u8,
            edges.len() * std::mem::size_of::<crate::vectorize::rasterize::GpuEdgeV2>(),
        )
    };
    let row_bytes = unsafe {
        std::slice::from_raw_parts(
            row_ranges.as_ptr() as *const u8,
            row_ranges.len() * std::mem::size_of::<crate::vectorize::rasterize::GpuRowRange>(),
        )
    };
    let idx_bytes = unsafe {
        std::slice::from_raw_parts(edge_indices.as_ptr() as *const u8, edge_indices.len() * 4)
    };

    let (edge_xfer, edge_buf) = upload_buffer(device, &cmd, edge_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);
    let (row_xfer, row_buf) = upload_buffer(device, &cmd, row_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);
    let (idx_xfer, idx_buf) = upload_buffer(device, &cmd, idx_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);

    {
        let copy_pass = device.begin_copy_pass(&cmd).expect("copy pass");
        copy_pass.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&edge_xfer),
            gpu::BufferRegion::new().with_buffer(&edge_buf).with_size(edge_bytes.len().max(4) as u32),
            false,
        );
        copy_pass.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&row_xfer),
            gpu::BufferRegion::new().with_buffer(&row_buf).with_size(row_bytes.len().max(4) as u32),
            false,
        );
        copy_pass.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&idx_xfer),
            gpu::BufferRegion::new().with_buffer(&idx_buf).with_size(idx_bytes.len().max(4) as u32),
            false,
        );
        device.end_copy_pass(copy_pass);
    }

    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex).with_cycle(true)],
            &[],
        ).expect("compute pass");
        compute_pass.bind_compute_pipeline(pipeline);
        compute_pass.bind_compute_storage_buffers(0, &[edge_buf, row_buf, idx_buf]);

        #[repr(C)]
        struct Uniforms { out_w: u32, out_h: u32, num_edges: u32, bg_color: u32 }
        cmd.push_compute_uniform_data(0, &Uniforms {
            out_w, out_h, num_edges: edges.len() as u32, bg_color,
        });
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let (dx, dy, dw, dh) = {
            let src_aspect = out_w as f32 / out_h as f32;
            let dst_aspect = sw_w as f32 / sw_h as f32;
            if dst_aspect > src_aspect {
                let dh = sw_h; let dw = (sw_h as f32 * src_aspect) as u32;
                ((sw_w - dw) / 2, 0, dw, dh)
            } else {
                let dw = sw_w; let dh = (sw_w as f32 / src_aspect) as u32;
                (0, (sw_h - dh) / 2, dw, dh)
            }
        };
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = out_w;
        blit_info.source.h = out_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = dx;
        blit_info.destination.y = dy;
        blit_info.destination.w = dw;
        blit_info.destination.h = dh;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(gpu::Filter::Nearest as i32);
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── OmniScale compute pipeline ──────────────────────────────────────────────

/// Create a scaling filter compute pipeline from pre-compiled shader bytecode.
/// All scaling compute shaders share the same descriptor layout:
///   set 0 = 1 readonly storage buffer (pixels)
///   set 1 = 1 readwrite storage texture (output)
///   set 2 = 1 uniform buffer
fn init_scale_compute(
    device: &gpu::Device,
    spirv: &[u8], msl: &[u8], dxil: &[u8],
    label: &str,
) -> Option<gpu::ComputePipeline> {
    let pipeline = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(1)
        .with_readwrite_storage_textures(1)
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| if !dxil.is_empty() {
            device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::DXIL, dxil)
                .with_entrypoint(c"main")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(1)
                .with_readwrite_storage_textures(1)
                .with_thread_count(16, 16, 1)
                .build()
        } else { Err(sdl3::get_error()) })
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, msl)
            .with_entrypoint(c"main_0")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(1)
            .with_readwrite_storage_textures(1)
            .with_thread_count(16, 16, 1)
            .build());

    match pipeline {
        Ok(p) => { eprintln!("{label} compute pipeline ready"); Some(p) }
        Err(e) => { eprintln!("{label} compute pipeline failed: {e}"); None }
    }
}

macro_rules! init_scale_pipeline {
    ($name:ident, $base:literal) => {
        pub fn $name(device: &gpu::Device) -> Option<gpu::ComputePipeline> {
            init_scale_compute(
                device,
                include_bytes!(concat!(env!("OUT_DIR"), "/", $base, "_comp.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/", $base, "_comp.metal")),
                include_bytes!(concat!(env!("OUT_DIR"), "/", $base, "_comp.dxil")),
                $base,
            )
        }
    };
}

init_scale_pipeline!(init_omniscale_compute_pipeline, "omniscale");
init_scale_pipeline!(init_epx_compute_pipeline, "epx");
init_scale_pipeline!(init_eagle_compute_pipeline, "eagle");
init_scale_pipeline!(init_scale3x_compute_pipeline, "scale3x");
init_scale_pipeline!(init_bicubic_compute_pipeline, "bicubic");
init_scale_pipeline!(init_nearest_aa_compute_pipeline, "nearest_aa");
init_scale_pipeline!(init_hqx_compute_pipeline, "hqx");
init_scale_pipeline!(init_xbr_compute_pipeline, "xbr");
init_scale_pipeline!(init_xbrz_compute_pipeline, "xbrz");
// Super xBR uses a different pipeline descriptor (2 storage buffers for 3-pass)
pub fn init_super_xbr_compute_pipeline(device: &gpu::Device) -> Option<gpu::ComputePipeline> {
    let comp_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_comp.spv"));
    let comp_msl = include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_comp.metal"));
    let comp_dxil = include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_comp.dxil"));

    let pipeline = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, comp_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(1)
        .with_readwrite_storage_buffers(1)
        .with_readwrite_storage_textures(1)
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| if !comp_dxil.is_empty() {
            device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::DXIL, comp_dxil)
                .with_entrypoint(c"main")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(1)
                .with_readwrite_storage_buffers(1)
                .with_readwrite_storage_textures(1)
                .with_thread_count(16, 16, 1)
                .build()
        } else { Err(sdl3::get_error()) })
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, comp_msl)
            .with_entrypoint(c"main_0")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(1)
            .with_readwrite_storage_buffers(1)
            .with_readwrite_storage_textures(1)
            .with_thread_count(16, 16, 1)
            .build());

    match pipeline {
        Ok(p) => { eprintln!("super_xbr compute pipeline ready"); Some(p) }
        Err(e) => { eprintln!("super_xbr compute pipeline failed: {e}"); None }
    }
}
init_scale_pipeline!(init_omniscale_legacy_compute_pipeline, "omniscale_legacy");
init_scale_pipeline!(init_edi_compute_pipeline, "edi");
init_scale_pipeline!(init_nedi_compute_pipeline, "nedi");
init_scale_pipeline!(init_dcci_compute_pipeline, "dcci");
init_scale_pipeline!(init_mmpx_compute_pipeline, "mmpx");
init_scale_pipeline!(init_lcd_grid_compute_pipeline, "lcd_grid");
init_scale_pipeline!(init_nearest_compute_pipeline, "nearest");
init_scale_pipeline!(init_bilinear_compute_pipeline, "bilinear");
init_scale_pipeline!(init_sai2x_compute_pipeline, "sai2x");
init_scale_pipeline!(init_super_sai2x_compute_pipeline, "super_sai2x");
init_scale_pipeline!(init_super_eagle_compute_pipeline, "super_eagle");

/// Dispatch a scaling compute shader and blit to the swapchain.
/// `uniforms` must be exactly 32 bytes (8 × u32) matching the shader's uniform block.
pub fn scale_compute_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipeline: &gpu::ComputePipeline,
    pixels: &[u32],
    out_w: u32, out_h: u32,
    uniforms: &[u32; 8],
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");

    let px_bytes = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
    };

    let px_size = px_bytes.len().max(4) as u32;
    let px_xfer = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(px_size)
        .build().expect("px transfer");
    {
        let mut map = px_xfer.map::<u8>(device, true);
        map.mem_mut()[..px_bytes.len()].copy_from_slice(px_bytes);
        map.unmap();
    }
    let px_buf = device.create_buffer()
        .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)
        .with_size(px_size)
        .build().expect("px buf");

    {
        let cp = device.begin_copy_pass(&cmd).expect("copy pass");
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
            gpu::BufferRegion::new().with_buffer(&px_buf).with_size(px_size),
            false,
        );
        device.end_copy_pass(cp);
    }

    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex).with_cycle(true)],
            &[],
        ).expect("compute pass");
        compute_pass.bind_compute_pipeline(pipeline);
        compute_pass.bind_compute_storage_buffers(0, &[px_buf]);

        #[repr(C)]
        struct RawUniforms([u32; 8]);
        cmd.push_compute_uniform_data(0, &RawUniforms(*uniforms));
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

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
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

/// Dispatch Super xBR 3-pass compute pipeline and blit to swapchain.
pub fn super_xbr_compute_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipeline: &gpu::ComputePipeline,
    pixels: &[u32],
    src_w: u32, src_h: u32,
    out_w: u32, out_h: u32,
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");

    // Upload pixel data
    let px_bytes = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
    };
    let px_size = px_bytes.len().max(4) as u32;
    let px_xfer = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(px_size).build().expect("px transfer");
    {
        let mut map = px_xfer.map::<u8>(device, true);
        map.mem_mut()[..px_bytes.len()].copy_from_slice(px_bytes);
        map.unmap();
    }
    let px_buf = device.create_buffer()
        .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)
        .with_size(px_size).build().expect("px buf");

    // Intermediate buffer (out_w * out_h * 4 bytes)
    let intermed_size = out_w * out_h * 4;
    let intermed_buf = device.create_buffer()
        .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE)
        .with_size(intermed_size.max(4)).build().expect("intermed buf");

    #[repr(C)]
    struct Uniforms { src_w: u32, src_h: u32, out_w: u32, out_h: u32, pass: u32, _pad: [u32; 3] }

    let dispatch_x = (out_w + 15) / 16;
    let dispatch_y = (out_h + 15) / 16;

    // Upload pixels
    {
        let cp = device.begin_copy_pass(&cmd).expect("copy pass");
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
            gpu::BufferRegion::new().with_buffer(&px_buf).with_size(px_size), false);
        device.end_copy_pass(cp);
    }

    // 3 passes in one command buffer — clone handles for each bind call
    for pass_idx in 0u32..3 {
        let cp = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex)
                .with_cycle(pass_idx == 0)],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&intermed_buf.clone())
                .with_cycle(pass_idx == 0)],
        ).expect("compute pass");
        cp.bind_compute_pipeline(pipeline);
        cp.bind_compute_storage_buffers(0, &[px_buf.clone()]);
        cmd.push_compute_uniform_data(0, &Uniforms {
            src_w, src_h, out_w, out_h, pass: pass_idx, _pad: [0; 3],
        });
        cp.dispatch(dispatch_x, dispatch_y, 1);
        device.end_compute_pass(cp);
    }
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
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── Full GPU vectorize pipeline ─────────────────────────────────────────────
//
// Six-stage pipeline: similarity_graph → resolve_crossings → cell_graph →
// optimize_energy → tjunction_correction → cell_rasterizer.
// All stages run on GPU with no CPU readback between stages.
// Buffers are cached between frames for zero per-frame allocation.

/// Cached GPU buffers for the cell rasterizer pipeline.
/// Allocated once and reused each frame (GB image dimensions never change).
struct CellRastBufCache {
    img_w: u32,
    img_h: u32,
    px_buf: gpu::Buffer,
    graph_buf: gpu::Buffer,
    graph_snapshot: gpu::Buffer,
    pos_buf: gpu::Buffer,
    nbr_buf: gpu::Buffer,
    flag_buf: gpu::Buffer,
    opt_out_buf: gpu::Buffer,
    orig_pos_buf: gpu::Buffer,
    px_xfer: gpu::TransferBuffer,
}

/// All pipelines for the full GPU vectorize pipeline.
/// Six stages: similarity_graph → resolve_crossings → cell_graph →
/// optimize_energy → tjunction_correction → cell_rasterizer.
pub struct GpuVectorizePipelines {
    sim_graph: gpu::ComputePipeline,
    resolve: gpu::ComputePipeline,
    cell_graph: gpu::ComputePipeline,
    optimizer: gpu::ComputePipeline,
    tjunction: gpu::ComputePipeline,
    rasterizer: gpu::ComputePipeline,
    buf_cache: Option<CellRastBufCache>,
}

pub fn init_full_gpu_pipeline(device: &gpu::Device) -> Option<GpuVectorizePipelines> {
    fn make(device: &gpu::Device, spirv: &[u8], msl: &[u8], dxil: &[u8],
            ro_bufs: u32, rw_bufs: u32, rw_tex: u32, threads: (u32,u32,u32),
            label: &str) -> Option<gpu::ComputePipeline> {
        let result = device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::SPIRV, spirv)
            .with_entrypoint(c"main")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(ro_bufs)
            .with_readwrite_storage_buffers(rw_bufs)
            .with_readwrite_storage_textures(rw_tex)
            .with_thread_count(threads.0, threads.1, threads.2)
            .build()
            .or_else(|_| if !dxil.is_empty() {
                device.create_compute_pipeline()
                    .with_code(gpu::ShaderFormat::DXIL, dxil)
                    .with_entrypoint(c"main")
                    .with_uniform_buffers(1)
                    .with_readonly_storage_buffers(ro_bufs)
                    .with_readwrite_storage_buffers(rw_bufs)
                    .with_readwrite_storage_textures(rw_tex)
                    .with_thread_count(threads.0, threads.1, threads.2)
                    .build()
            } else { Err(sdl3::get_error()) })
            .or_else(|_| device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::MSL, msl)
                .with_entrypoint(c"main_0")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(ro_bufs)
                .with_readwrite_storage_buffers(rw_bufs)
                .with_readwrite_storage_textures(rw_tex)
                .with_thread_count(threads.0, threads.1, threads.2)
                .build());
        match result {
            Ok(p) => Some(p),
            Err(e) => { eprintln!("vectorize-gpu: {label} pipeline failed: {e}"); None }
        }
    }

    let sim = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.dxil")),
        1, 1, 0, (16, 16, 1), "similarity_graph")?;

    let resolve = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.dxil")),
        1, 1, 0, (16, 16, 1), "resolve_crossings")?;

    let cell = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.dxil")),
        1, 3, 0, (16, 16, 1), "cell_graph")?;

    let opt = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.dxil")),
        4, 1, 0, (256, 1, 1), "optimize_energy")?;

    let tjunc = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.dxil")),
        2, 1, 0, (256, 1, 1), "update_tjunction")?;

    let rast = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.dxil")),
        5, 0, 1, (256, 1, 1), "cell_rasterizer")?;

    eprintln!("Full GPU vectorize pipeline ready (6 stages)");
    Some(GpuVectorizePipelines { sim_graph: sim, resolve, cell_graph: cell, optimizer: opt, tjunction: tjunc, rasterizer: rast, buf_cache: None })
}

/// Dispatch stages 1-4b (similarity graph through T-junction correction).
/// Shared between live pipeline and screenshot pipeline.
/// Returns the buffer containing the optimized+corrected positions.
fn dispatch_stages_1_4b(
    device: &gpu::Device,
    cmd: &gpu::CommandBuffer,
    pipelines: &GpuVectorizePipelines,
    px_buf: &gpu::Buffer,
    graph_buf: &gpu::Buffer,
    graph_snapshot: &gpu::Buffer,
    pos_buf: &gpu::Buffer,
    nbr_buf: &gpu::Buffer,
    flag_buf: &gpu::Buffer,
    opt_out_buf: &gpu::Buffer,
    orig_pos_buf: &gpu::Buffer,
    img_w: u32, img_h: u32,
) -> gpu::Buffer {
    let graph_stride = 2 * img_w + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;
    let graph_size = (graph_stride * (2 * img_h + 1) * 4).max(4) as u32;
    let pos_size = (num_cps * 2 * 4).max(4) as u32;

    // Stage 1: Similarity graph
    {
        let cp = device.begin_compute_pass(cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(graph_buf).with_cycle(false)]).expect("sim pass");
        cp.bind_compute_pipeline(&pipelines.sim_graph);
        cp.bind_compute_storage_buffers(0, &[px_buf.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, graph_stride: u32, _p: u32 }
        cmd.push_compute_uniform_data(0, &U { img_w, img_h, graph_stride, _p: 0 });
        cp.dispatch((img_w + 15) / 16, (img_h + 15) / 16, 1);
        device.end_compute_pass(cp);
    }

    // Stage 2: Resolve crossings
    {
        let cp = device.begin_copy_pass(cmd).expect("graph copy");
        unsafe {
            let src = sdl3::sys::gpu::SDL_GPUBufferLocation { buffer: graph_buf.raw(), offset: 0 };
            let dst = sdl3::sys::gpu::SDL_GPUBufferLocation { buffer: graph_snapshot.raw(), offset: 0 };
            sdl3::sys::gpu::SDL_CopyGPUBufferToBuffer(cp.raw(), &src, &dst, graph_size, false);
        }
        device.end_copy_pass(cp);
    }
    {
        let cp = device.begin_compute_pass(cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(graph_buf).with_cycle(false)]).expect("resolve pass");
        cp.bind_compute_pipeline(&pipelines.resolve);
        cp.bind_compute_storage_buffers(0, &[graph_snapshot.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, graph_stride: u32, _p: u32 }
        cmd.push_compute_uniform_data(0, &U { img_w, img_h, graph_stride, _p: 0 });
        cp.dispatch((img_w.saturating_sub(1) + 15) / 16, (img_h.saturating_sub(1) + 15) / 16, 1);
        device.end_compute_pass(cp);
    }

    // Stage 3: Cell graph
    {
        let cp = device.begin_compute_pass(cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(pos_buf).with_cycle(false),
              gpu::StorageBufferReadWriteBinding::new().with_buffer(nbr_buf).with_cycle(false),
              gpu::StorageBufferReadWriteBinding::new().with_buffer(flag_buf).with_cycle(false),
            ]).expect("cell pass");
        cp.bind_compute_pipeline(&pipelines.cell_graph);
        cp.bind_compute_storage_buffers(0, &[graph_buf.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, graph_stride: u32, corners_w: u32 }
        cmd.push_compute_uniform_data(0, &U { img_w, img_h, graph_stride, corners_w });
        cp.dispatch((corners_w + 15) / 16, (corners_h + 15) / 16, 1);
        device.end_compute_pass(cp);
    }

    // Save original positions before optimization
    {
        let cp = device.begin_copy_pass(cmd).expect("orig pos copy");
        unsafe {
            let src = sdl3::sys::gpu::SDL_GPUBufferLocation { buffer: pos_buf.raw(), offset: 0 };
            let dst = sdl3::sys::gpu::SDL_GPUBufferLocation { buffer: orig_pos_buf.raw(), offset: 0 };
            sdl3::sys::gpu::SDL_CopyGPUBufferToBuffer(cp.raw(), &src, &dst, pos_size, false);
        }
        device.end_copy_pass(cp);
    }

    // Stage 4: Optimize energy (2-pass ping-pong)
    let mut cur_in = pos_buf.clone();
    let mut cur_out = opt_out_buf.clone();
    for _ in 0..2u32 {
        let cp = device.begin_compute_pass(cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&cur_out).with_cycle(false)],
        ).expect("opt pass");
        cp.bind_compute_pipeline(&pipelines.optimizer);
        cp.bind_compute_storage_buffers(0, &[cur_in.clone(), pos_buf.clone(), nbr_buf.clone(), flag_buf.clone()]);
        #[repr(C)] struct U { num_nodes: u32, gradient_step: f32, max_move: f32, positional_scale: f32 }
        cmd.push_compute_uniform_data(0, &U {
            num_nodes: num_cps, gradient_step: 0.01, max_move: 0.25, positional_scale: 2.5 });
        cp.dispatch((num_cps + 255) / 256, 1, 1);
        device.end_compute_pass(cp);
        std::mem::swap(&mut cur_in, &mut cur_out);
    }
    let optimized_pos = cur_in;

    // Stage 4b: T-junction position correction + stem CP alignment
    {
        let cp = device.begin_compute_pass(cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&optimized_pos).with_cycle(false)],
        ).expect("tjunc pass");
        cp.bind_compute_pipeline(&pipelines.tjunction);
        cp.bind_compute_storage_buffers(0, &[nbr_buf.clone(), flag_buf.clone()]);
        #[repr(C)] struct U { num_nodes: u32, _p0: u32, _p1: u32, _p2: u32 }
        cmd.push_compute_uniform_data(0, &U { num_nodes: num_cps, _p0: 0, _p1: 0, _p2: 0 });
        cp.dispatch((num_cps + 255) / 256, 1, 1);
        device.end_compute_pass(cp);
    }

    optimized_pos
}

/// Run the full GPU vectorize pipeline and blit to window.
/// Called every frame from the emulator's render loop. Uses cached GPU
/// buffers for zero per-frame allocation. No CPU computation after the
/// initial pixel upload — all stages run on GPU back-to-back.
pub fn gpu_vectorize_full_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipelines: &mut GpuVectorizePipelines,
    pixels: &[u32],
    img_w: u32, img_h: u32,
    out_w: u32, out_h: u32,
    scale: f32,
) {
    let graph_stride = 2 * img_w + 1;
    let graph_h = 2 * img_h + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;
    let px_size = img_w * img_h * 4;
    let graph_size = (graph_stride * graph_h * 4).max(4);
    let pos_size = (num_cps * 2 * 4).max(4);

    // Ensure GPU buffers are cached (allocated once, reused each frame)
    if pipelines.buf_cache.as_ref().map_or(true, |c| c.img_w != img_w || c.img_h != img_h) {
        let rw = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE;
        let ro = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ;
        let nbr_size = (num_cps * 4 * 4).max(4);
        let flag_size = (num_cps * 4).max(4);
        pipelines.buf_cache = Some(CellRastBufCache {
            img_w, img_h,
            px_buf: device.create_buffer().with_usage(ro).with_size(px_size).build().expect("px buf"),
            graph_buf: device.create_buffer().with_usage(rw).with_size(graph_size).build().expect("graph buf"),
            graph_snapshot: device.create_buffer().with_usage(ro).with_size(graph_size).build().expect("graph snapshot"),
            pos_buf: device.create_buffer().with_usage(rw).with_size(pos_size).build().expect("pos buf"),
            nbr_buf: device.create_buffer().with_usage(rw).with_size(nbr_size).build().expect("nbr buf"),
            flag_buf: device.create_buffer().with_usage(rw).with_size(flag_size).build().expect("flag buf"),
            opt_out_buf: device.create_buffer().with_usage(rw).with_size(pos_size).build().expect("opt out buf"),
            orig_pos_buf: device.create_buffer().with_usage(rw).with_size(pos_size).build().expect("orig pos buf"),
            px_xfer: device.create_transfer_buffer()
                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
                .with_size(px_size).build().expect("px xfer"),
        });
        eprintln!("Cell rasterizer buffer cache allocated for {}x{}", img_w, img_h);
    }
    let b = pipelines.buf_cache.as_ref().unwrap();

    let cmd = device.acquire_command_buffer().expect("cmd buf");

    // Upload pixel data (reuse cached transfer buffer)
    {
        let mut map = b.px_xfer.map::<u8>(device, true);
        let bytes = unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        map.mem_mut()[..bytes.len()].copy_from_slice(bytes);
        map.unmap();
    }
    {
        let cp = device.begin_copy_pass(&cmd).expect("copy pass");
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&b.px_xfer),
            gpu::BufferRegion::new().with_buffer(&b.px_buf).with_size(px_size), false);
        device.end_copy_pass(cp);
    }

    // Stages 1-4b: vectorize pipeline (shared with screenshot path)
    let optimized_pos = dispatch_stages_1_4b(
        device, &cmd, pipelines,
        &b.px_buf, &b.graph_buf, &b.graph_snapshot,
        &b.pos_buf, &b.nbr_buf, &b.flag_buf,
        &b.opt_out_buf, &b.orig_pos_buf, img_w, img_h,
    );

    // Stage 5: Tile-based cell rasterizer (one workgroup per 2×2 source tile)
    {
        let tiles_w = (img_w + 1) / 2;
        let tiles_h = (img_h + 1) / 2;
        let total_tiles = tiles_w * tiles_h;
        let cp = device.begin_compute_pass(&cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex).with_cycle(true)],
            &[]).expect("rast pass");
        cp.bind_compute_pipeline(&pipelines.rasterizer);
        cp.bind_compute_storage_buffers(0, &[b.px_buf.clone(), optimized_pos.clone(), b.orig_pos_buf.clone(), b.flag_buf.clone(), b.nbr_buf.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, out_w: u32, out_h: u32,
                               scale: f32, corners_w: u32, tiles_w: u32, tiles_h: u32 }
        cmd.push_compute_uniform_data(0, &U {
            img_w, img_h, out_w, out_h, scale, corners_w, tiles_w, tiles_h });
        cp.dispatch(total_tiles, 1, 1);
        device.end_compute_pass(cp);
    }

    // Blit to swapchain
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let src_aspect = out_w as f32 / out_h as f32;
        let dst_aspect = sw_w as f32 / sw_h as f32;
        let (dx, dy, dw, dh) = if dst_aspect > src_aspect {
            let dh = sw_h; let dw = (sw_h as f32 * src_aspect) as u32;
            ((sw_w - dw) / 2, 0, dw, dh)
        } else {
            let dw = sw_w; let dh = (sw_w as f32 / src_aspect) as u32;
            (0, (sw_h - dh) / 2, dw, dh)
        };
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = out_w;
        blit_info.source.h = out_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = dx;
        blit_info.destination.y = dy;
        blit_info.destination.w = dw;
        blit_info.destination.h = dh;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(gpu::Filter::Nearest as i32);
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

/// CPU mirror of cell_rasterizer.comp for debugging.
/// Activated by setting the `CPU_RASTER` environment variable.
/// Set `CPU_RASTER_PX=x,y` to debug a specific pixel.
#[allow(dead_code)]
/// Takes the same buffers the GPU rasterizer uses and runs the identical logic
/// on CPU, printing debug info for specific artifact pixels.
#[allow(clippy::too_many_arguments)]
fn cpu_rasterize_debug(
    pixels: &[u32], img_w: u32, img_h: u32,
    cp_positions: &[f32],   // optimized positions, 2 floats per CP
    orig_positions: &[f32], // original positions, 2 floats per CP
    cp_flags: &[u32],       // 1 u32 per CP
    cp_neighbors: &[i32],   // 4 i32 per CP
    out_w: u32, out_h: u32, scale: f32, corners_w: u32,
) {
    fn px_color(pixels: &[u32], x: i32, y: i32, img_w: u32, img_h: u32) -> u32 {
        let x = x.clamp(0, img_w as i32 - 1);
        let y = y.clamp(0, img_h as i32 - 1);
        pixels[(y as usize) * (img_w as usize) + x as usize]
    }
    fn read_pos(positions: &[f32], idx: i32) -> (f32, f32) {
        if idx < 0 { return (-1e10, -1e10); }
        let i = idx as usize;
        (positions[i * 2], positions[i * 2 + 1])
    }
    fn beval(p0: (f32,f32), p1: (f32,f32), p2: (f32,f32), t: f32) -> (f32,f32) {
        let u = 1.0 - t;
        (0.5*u*u*p0.0 + (u*t+0.5)*p1.0 + 0.5*t*t*p2.0,
         0.5*u*u*p0.1 + (u*t+0.5)*p1.1 + 0.5*t*t*p2.1)
    }
    fn beval_deriv(p0: (f32,f32), p1: (f32,f32), p2: (f32,f32), t: f32) -> (f32,f32) {
        ((t-1.0)*p0.0 + (1.0-2.0*t)*p1.0 + t*p2.0,
         (t-1.0)*p0.1 + (1.0-2.0*t)*p1.1 + t*p2.1)
    }
    fn dot2(a: (f32,f32), b: (f32,f32)) -> f32 { a.0*b.0 + a.1*b.1 }
    fn len2(a: (f32,f32)) -> f32 { a.0*a.0 + a.1*a.1 }
    fn color_str(c: u32) -> String {
        format!("#{:02x}{:02x}{:02x}", (c>>16)&0xff, (c>>8)&0xff, c&0xff)
    }
    fn edge_colors_for_cp(pixels: &[u32], img_w: u32, img_h: u32, cp_neighbors: &[i32], ci: usize) -> (u32, u32, u32, u32) {
        let prev_dir = cp_neighbors[ci * 4 + 2];
        let next_dir = cp_neighbors[ci * 4 + 3];
        let icx = (ci / 2 % (img_w as usize + 1)) as i32;
        let icy = (ci / 2 / (img_w as usize + 1)) as i32;
        let get_px = |px: i32, py: i32| -> u32 {
            let px = px.clamp(0, img_w as i32 - 1) as usize;
            let py = py.clamp(0, img_h as i32 - 1) as usize;
            pixels[py * img_w as usize + px]
        };
        let edge_col = |dir: i32| -> (u32, u32) {
            match dir {
                0 => (get_px(icx-1, icy-1), get_px(icx, icy-1)),
                1 => (get_px(icx, icy-1), get_px(icx, icy)),
                2 => (get_px(icx, icy), get_px(icx-1, icy)),
                3 => (get_px(icx-1, icy), get_px(icx-1, icy-1)),
                _ => (0, 0),
            }
        };
        let (pl, pr) = if prev_dir >= 0 { edge_col(prev_dir) } else { (0, 0) };
        let (nl, nr) = if next_dir >= 0 { edge_col(next_dir) } else { (0, 0) };
        (pl, pr, nl, nr)
    }

    let num_cps = cp_flags.len();
    let mut mismatches = 0u32;
    let mut output = vec![0u32; (out_w * out_h) as usize];

    // Dump CP info near the debug pixel
    if let Ok(px_str) = std::env::var("CPU_RASTER_PX") {
        let parts: Vec<&str> = px_str.split(',').collect();
        if parts.len() == 2 {
            let dpx: f32 = parts[0].parse().unwrap_or(0.0);
            let dpy: f32 = parts[1].parse().unwrap_or(0.0);
            let src_x = (dpx + 0.5) / scale;
            let src_y = (dpy + 0.5) / scale;
            let scx = src_x.floor() as i32;
            let scy = src_y.floor() as i32;
            eprintln!("DEBUG: pixel ({},{}) → src ({:.2},{:.2}), searching CPs near corner ({},{})", dpx, dpy, src_x, src_y, scx, scy);
            for cy in (scy-1)..=(scy+1) {
                for cx in (scx-1)..=(scx+1) {
                    for slot in 0..2 {
                        if cx < 0 || cy < 0 || cx >= corners_w as i32 || cy > img_h as i32 { continue; }
                        let ci = (cy as usize * corners_w as usize + cx as usize) * 2 + slot;
                        if ci >= cp_flags.len() { continue; }
                        let flag = cp_flags[ci];
                        if flag == 0 { continue; }
                        let prev = cp_neighbors[ci * 4];
                        let next = cp_neighbors[ci * 4 + 1];
                        let pos = read_pos(cp_positions, ci as i32);
                        let (pl, pr, nl, nr) = edge_colors_for_cp(pixels, img_w, img_h, cp_neighbors, ci);
                        eprintln!("  ci={} corner({},{}) slot={} flag={} prev={} next={} pos=({:.2},{:.2}) pl={} pr={} nl={} nr={}",
                            ci, cx, cy, slot, flag, prev, next, pos.0, pos.1,
                            color_str(pl), color_str(pr), color_str(nl), color_str(nr));
                    }
                }
            }
        }
    }

    // Rasterize every output pixel using same logic as GPU shader
    let debug_all = std::env::var("CPU_RASTER_ALL").is_ok();
    // Debug specific pixel: CPU_RASTER_PX=x,y
    let debug_px: Option<(u32,u32)> = std::env::var("CPU_RASTER_PX").ok().and_then(|s| {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 2 {
            Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
        } else { None }
    });

    for opy in 0..out_h {
        for opx in 0..out_w {
            let fx = (opx as f32 + 0.5) / scale;
            let fy = (opy as f32 + 0.5) / scale;
            let pt = (fx, fy);

            let nn_x = (fx.floor() as i32).clamp(0, img_w as i32 - 1);
            let nn_y = (fy.floor() as i32).clamp(0, img_h as i32 - 1);
            let nn_color = px_color(pixels, nn_x, nn_y, img_w, img_h);

            // Collect hits (same as GPU)
            struct Hit { dist2: f32, t: f32, orig_t: f32, ci: i32, prev_ci: i32, next_ci: i32 }
            let mut hits: Vec<Hit> = Vec::new();

            let search_cx = (fx.floor() as i32).clamp(0, img_w as i32);
            let search_cy = (fy.floor() as i32).clamp(0, img_h as i32);

            for cy in (search_cy-2)..=(search_cy+2) {
                for cx in (search_cx-2)..=(search_cx+2) {
                    for slot in 0..2 {
                        if cx < 0 || cy < 0 || cx >= corners_w as i32 || cy > img_h as i32 { continue; }
                        let ci = (cy * corners_w as i32 + cx) * 2 + slot;
                        if ci < 0 || ci >= num_cps as i32 { continue; }
                        let flag = cp_flags[ci as usize];
                        if flag == 0 { continue; }
                        let prev_ci = cp_neighbors[ci as usize * 4];
                        let next_ci = cp_neighbors[ci as usize * 4 + 1];
                        if prev_ci < 0 && next_ci < 0 { continue; }

                        let cp = read_pos(cp_positions, ci);
                        let pp = if prev_ci >= 0 { read_pos(cp_positions, prev_ci) } else { cp };
                        let pn = if next_ci >= 0 { read_pos(cp_positions, next_ci) } else { cp };

                        let ocp = read_pos(orig_positions, ci);
                        let opp = if prev_ci >= 0 { read_pos(orig_positions, prev_ci) } else { ocp };
                        let opn = if next_ci >= 0 { read_pos(orig_positions, next_ci) } else { ocp };

                        let mut best_d2 = 1e10f32;
                        let mut best_t = 0.0f32;
                        let mut best_orig_t = 0.0f32;
                        let mut orig_best_d2 = 1e10f32;
                        let n = 16;
                        for s in 0..=n {
                            let t = s as f32 / n as f32;
                            let bp = beval(pp, cp, pn, t);
                            let d2 = len2((pt.0-bp.0, pt.1-bp.1));
                            if d2 < best_d2 { best_d2 = d2; best_t = t; }
                            let obp = beval(opp, ocp, opn, t);
                            let od2 = len2((pt.0-obp.0, pt.1-obp.1));
                            if od2 < orig_best_d2 { orig_best_d2 = od2; best_orig_t = t; }
                        }

                        if best_d2 < 1.0 {
                            let pos = hits.iter().position(|h| best_d2 < h.dist2).unwrap_or(hits.len());
                            if hits.len() < 4 || pos < 4 {
                                hits.insert(pos, Hit { dist2: best_d2, t: best_t, orig_t: best_orig_t, ci, prev_ci, next_ci });
                                hits.truncate(4);
                            }
                        }
                    }
                }
            }

            // Process hits (same logic as GPU shader)
            let mut final_color = nn_color;
            let mut resolved = false;
            let mut debug_info: Vec<String> = Vec::new();

            for (h, hit) in hits.iter().enumerate() {
                let ci = hit.ci as usize;
                let t = hit.t;

                let (pl, pr, nl, nr) = edge_colors_for_cp(pixels, img_w, img_h, cp_neighbors, ci);

                let prev_valid = pl != pr;
                let next_valid = nl != nr;

                let (color_left, color_right, ref_t);
                if hit.prev_ci < 0 {
                    // Endpoint: skip if pixel is beyond start
                    let ep = read_pos(cp_positions, hit.ci);
                    let toward = (read_pos(cp_positions, hit.next_ci).0 - ep.0,
                                  read_pos(cp_positions, hit.next_ci).1 - ep.1);
                    if (pt.0 - ep.0) * toward.0 + (pt.1 - ep.1) * toward.1 < 0.0 { continue; }
                    if next_valid { color_left = nr; color_right = nl; ref_t = 1.0; }
                    else { continue; }
                } else if hit.next_ci < 0 {
                    // Endpoint: skip if pixel is beyond end
                    let ep = read_pos(cp_positions, hit.ci);
                    let toward = (read_pos(cp_positions, hit.prev_ci).0 - ep.0,
                                  read_pos(cp_positions, hit.prev_ci).1 - ep.1);
                    if (pt.0 - ep.0) * toward.0 + (pt.1 - ep.1) * toward.1 < 0.0 { continue; }
                    if prev_valid { color_left = pl; color_right = pr; ref_t = 0.0; }
                    else { continue; }
                } else if t < 0.5 {
                    if prev_valid { color_left = pl; color_right = pr; ref_t = 0.0; }
                    else if next_valid { color_left = nr; color_right = nl; ref_t = 1.0; }
                    else { continue; }
                } else {
                    if next_valid { color_left = nr; color_right = nl; ref_t = 1.0; }
                    else if prev_valid { color_left = pl; color_right = pr; ref_t = 0.0; }
                    else { continue; }
                }

                let orig_cp = read_pos(orig_positions, hit.ci);
                let orig_pp = if hit.prev_ci >= 0 { read_pos(orig_positions, hit.prev_ci) } else { orig_cp };
                let orig_pn = if hit.next_ci >= 0 { read_pos(orig_positions, hit.next_ci) } else { orig_cp };
                let orig_tangent = beval_deriv(orig_pp, orig_cp, orig_pn, ref_t);
                let tl2 = len2(orig_tangent).sqrt();
                if tl2 < 1e-8 { continue; }
                let orig_tangent = (orig_tangent.0/tl2, orig_tangent.1/tl2);
                let orig_normal = (-orig_tangent.1, orig_tangent.0);

                let opt_cp = read_pos(cp_positions, hit.ci);
                let opt_pp = if hit.prev_ci >= 0 { read_pos(cp_positions, hit.prev_ci) } else { opt_cp };
                let opt_pn = if hit.next_ci >= 0 { read_pos(cp_positions, hit.next_ci) } else { opt_cp };
                let cpt = beval(opt_pp, opt_cp, opt_pn, t);

                let opt_tangent = beval_deriv(opt_pp, opt_cp, opt_pn, t);
                let otl = len2(opt_tangent).sqrt();
                if otl < 1e-8 { continue; }
                let opt_tangent = (opt_tangent.0/otl, opt_tangent.1/otl);
                let opt_normal = (-opt_tangent.1, opt_tangent.0);

                let normals_agree = dot2(opt_normal, orig_normal) > 0.0;
                let side = dot2((pt.0-cpt.0, pt.1-cpt.1), opt_normal);
                let assigned = if normals_agree {
                    if side > 0.0 { color_left } else { color_right }
                } else {
                    if side > 0.0 { color_right } else { color_left }
                };

                debug_info.push(format!(
                    "  hit[{}]: ci={} t={:.3} d2={:.4} pl={} pr={} nl={} nr={} L={} R={} side={:.4} agree={} → {}",
                    h, hit.ci, t, hit.dist2,
                    color_str(pl), color_str(pr), color_str(nl), color_str(nr),
                    color_str(color_left), color_str(color_right),
                    side, normals_agree, color_str(assigned)
                ));
                debug_info.push(format!(
                    "    orig_cp=({:.2},{:.2}) opt_cp=({:.2},{:.2}) cpt=({:.2},{:.2})",
                    orig_cp.0, orig_cp.1, opt_cp.0, opt_cp.1, cpt.0, cpt.1
                ));
                debug_info.push(format!(
                    "    orig_tang=({:.3},{:.3}) opt_tang=({:.3},{:.3}) orig_n=({:.3},{:.3}) opt_n=({:.3},{:.3})",
                    orig_tangent.0, orig_tangent.1, opt_tangent.0, opt_tangent.1,
                    orig_normal.0, orig_normal.1, opt_normal.0, opt_normal.1
                ));

                if !resolved {
                    final_color = assigned;
                    resolved = true;
                }
            }

            output[(opy * out_w + opx) as usize] = final_color;

            // Debug output for specific pixel or all suspicious pixels
            let is_target = debug_px.map_or(false, |(dx,dy)| opx == dx && opy == dy);
            let is_suspicious = resolved && final_color != nn_color;
            if (is_target || (debug_all && is_suspicious)) && !debug_info.is_empty() {
                eprintln!("pixel ({},{}) → {} (nn={}) hits={}{}",
                    opx, opy, color_str(final_color), color_str(nn_color),
                    hits.len(),
                    if is_suspicious { " *** DIFFERS FROM NN ***" } else { "" });
                for line in &debug_info {
                    eprintln!("{}", line);
                }
            }
            if is_suspicious { mismatches += 1; }
        }
    }
    eprintln!("CPU rasterizer: {} pixels differ from nearest-neighbor", mismatches);

    // Save CPU rasterizer output as PNG
    let path = "/tmp/cpu_rasterizer_output.png";
    let mut rgb = vec![0u8; (out_w * out_h * 3) as usize];
    for (i, &c) in output.iter().enumerate() {
        rgb[i*3]   = ((c >> 16) & 0xff) as u8;
        rgb[i*3+1] = ((c >> 8) & 0xff) as u8;
        rgb[i*3+2] = (c & 0xff) as u8;
    }
    image::save_buffer(path, &rgb, out_w, out_h, image::ColorType::Rgb8).unwrap();
    eprintln!("CPU rasterizer output saved to {}", path);
}

/// Headless GPU full-pipeline screenshot (creates own device).
/// Used by the test_runner for offline vectorization. Creates a temporary
/// SDL context and GPU device, runs the full pipeline, and downloads the
/// result. Uses `dispatch_stages_1_4b()` shared with the live pipeline.
/// Includes optional CPU debug rasterizer (set `CPU_RASTER` env var).
pub fn gpu_full_pipeline_screenshot(
    src: &[u32], src_w: usize, src_h: usize, scale: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    let img_w = src_w as u32;
    let img_h = src_h as u32;
    let out_w = (src_w * scale) as u32;
    let out_h = (src_h * scale) as u32;
    if out_w == 0 || out_h == 0 { return None; }

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_full", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false).ok()?.with_window(&window).ok()?;

    let pipelines = init_full_gpu_pipeline(&device)?;

    let out_tex = device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
            .with_width(out_w).with_height(out_h)
            .with_layer_count_or_depth(1).with_num_levels(1)
    ).ok()?;

    // Run the full pipeline (reuse the same dispatch function but with our tex)
    // We need to duplicate the dispatch logic without the blit-to-swapchain part.
    // For simplicity, call the existing function which blits to a null swapchain,
    // then download from the texture.
    let cmd = device.acquire_command_buffer().ok()?;

    let graph_stride = 2 * img_w + 1;
    let graph_h = 2 * img_h + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;
    let rw = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE;
    let ro = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ;

    let px_size = img_w * img_h * 4;
    let graph_size = graph_stride * graph_h * 4;
    let pos_size = num_cps * 2 * 4;
    let nbr_size = num_cps * 4 * 4;
    let flag_size = num_cps * 4;

    // Upload pixels
    let px_xfer = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(px_size).build().ok()?;
    {
        let mut map = px_xfer.map::<u8>(&device, true);
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        map.mem_mut()[..bytes.len()].copy_from_slice(bytes);
        map.unmap();
    }
    let px_buf = device.create_buffer().with_usage(ro).with_size(px_size).build().ok()?;
    let graph_buf = device.create_buffer().with_usage(rw).with_size(graph_size.max(4)).build().ok()?;
    let pos_buf = device.create_buffer().with_usage(rw).with_size(pos_size.max(4)).build().ok()?;
    let nbr_buf = device.create_buffer().with_usage(rw).with_size(nbr_size.max(4)).build().ok()?;
    let flag_buf = device.create_buffer().with_usage(rw).with_size(flag_size.max(4)).build().ok()?;
    let opt_out_buf = device.create_buffer().with_usage(rw).with_size(pos_size.max(4)).build().ok()?;

    { let cp = device.begin_copy_pass(&cmd).ok()?;
      cp.upload_to_gpu_buffer(
          gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
          gpu::BufferRegion::new().with_buffer(&px_buf).with_size(px_size), false);
      device.end_copy_pass(cp); }

    let graph_snapshot = device.create_buffer().with_usage(ro).with_size(graph_size.max(4)).build().ok()?;
    let orig_pos_buf = device.create_buffer().with_usage(rw).with_size(pos_size.max(4)).build().ok()?;

    // Stages 1-4b: shared vectorize pipeline dispatch
    dispatch_stages_1_4b(
        &device, &cmd, &pipelines,
        &px_buf, &graph_buf, &graph_snapshot,
        &pos_buf, &nbr_buf, &flag_buf,
        &opt_out_buf, &orig_pos_buf, img_w, img_h,
    );

    // Debug: download positions before and after optimizer
    let pos_dl_size = num_cps * 2 * 4;
    let pos_dl = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(pos_dl_size).build().ok()?;
    let opt_dl = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(pos_dl_size).build().ok()?;
    let nbr_dl = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(num_cps * 4 * 4).build().ok()?;
    { let cp = device.begin_copy_pass(&cmd).ok()?;
      unsafe {
          let mut src = sdl3::sys::gpu::SDL_GPUBufferRegion::default();
          src.buffer = pos_buf.raw(); src.size = pos_dl_size;
          let mut dst = sdl3::sys::gpu::SDL_GPUTransferBufferLocation::default();
          dst.transfer_buffer = pos_dl.raw();
          sdl3::sys::gpu::SDL_DownloadFromGPUBuffer(cp.raw(), &src, &dst);

          src.buffer = opt_out_buf.raw();
          dst.transfer_buffer = opt_dl.raw();
          sdl3::sys::gpu::SDL_DownloadFromGPUBuffer(cp.raw(), &src, &dst);

          let mut src2 = sdl3::sys::gpu::SDL_GPUBufferRegion::default();
          src2.buffer = nbr_buf.raw(); src2.size = num_cps * 4 * 4;
          let mut dst2 = sdl3::sys::gpu::SDL_GPUTransferBufferLocation::default();
          dst2.transfer_buffer = nbr_dl.raw();
          sdl3::sys::gpu::SDL_DownloadFromGPUBuffer(cp.raw(), &src2, &dst2);
      }
      device.end_copy_pass(cp); }

    // Submit and read back positions before running rasterizer
    let fence0 = cmd.submit_and_acquire_fence(&device).ok()?;
    device.wait_fences(true, &[fence0]).ok()?;

    {
        let pos_map = pos_dl.map::<f32>(&device, false);
        let opt_map = opt_dl.map::<f32>(&device, false);
        let nbr_map = nbr_dl.map::<i32>(&device, false);
        let pos_data = pos_map.mem();
        let opt_data = opt_map.mem();
        let nbr_data = nbr_map.mem();
        let mut diff_count = 0u32;
        let mut nonzero_pos = 0u32;
        let mut valid_nbr = 0u32;
        for i in 0..num_cps as usize {
            let px = pos_data[i * 2];
            let py = pos_data[i * 2 + 1];
            let ox = opt_data[i * 2];
            let oy = opt_data[i * 2 + 1];
            if (px - ox).abs() > 0.001 || (py - oy).abs() > 0.001 { diff_count += 1; }
            if px.abs() > 0.001 || py.abs() > 0.001 { nonzero_pos += 1; }
            let n0 = nbr_data[i * 4];
            let n1 = nbr_data[i * 4 + 1];
            if n0 >= 0 || n1 >= 0 { valid_nbr += 1; }
        }
        // Also download flags
        let flag_dl = device.create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
            .with_size(num_cps * 4).build().ok();
        if let Some(ref fdl) = flag_dl {
            let cmd2 = device.acquire_command_buffer().ok().unwrap();
            let cp2 = device.begin_copy_pass(&cmd2).ok().unwrap();
            unsafe {
                let mut src = sdl3::sys::gpu::SDL_GPUBufferRegion::default();
                src.buffer = flag_buf.raw(); src.size = num_cps * 4;
                let mut dst = sdl3::sys::gpu::SDL_GPUTransferBufferLocation::default();
                dst.transfer_buffer = fdl.raw();
                sdl3::sys::gpu::SDL_DownloadFromGPUBuffer(cp2.raw(), &src, &dst);
            }
            device.end_copy_pass(cp2);
            let f2 = cmd2.submit_and_acquire_fence(&device).ok().unwrap();
            let _ = device.wait_fences(true, &[f2]);
            let fmap = fdl.map::<u32>(&device, false);
            let fdata = fmap.mem();
            let pinned_count = fdata.iter().filter(|&&f| f & 1 != 0).count();
            let active_count = fdata.iter().filter(|&&f| f != 0).count();
            eprintln!("GPU debug: flags: {} pinned, {} with any flag set", pinned_count, active_count);
            drop(fmap);
        }

        let nonzero_opt = opt_data.chunks(2).filter(|c| c[0].abs() > 0.001 || c[1].abs() > 0.001).count();
        eprintln!("GPU debug: {} CPs, {} nonzero pos, {} nonzero opt, {} with valid neighbors, {} changed by optimizer",
            num_cps, nonzero_pos, nonzero_opt, valid_nbr, diff_count);

        // Check graph buffer too
        eprintln!("GPU debug: graph_size={}, graph_stride={}", graph_size, graph_stride);
        // Dump all CPs for visualization
        let mut integer_count = 0u32;
        let mut fractional_count = 0u32;
        if std::env::var("DUMP_CPS").is_ok() {
            for i in 0..num_cps as usize {
                let px = pos_data[i * 2]; let py = pos_data[i * 2 + 1];
                let ox = opt_data[i * 2]; let oy = opt_data[i * 2 + 1];
                let n0 = nbr_data[i * 4]; let n1 = nbr_data[i * 4 + 1];
                let flag = if let Some(ref fdl) = flag_dl {
                    let fmap = fdl.map::<u32>(&device, false);
                    let f = fmap.mem()[i];
                    drop(fmap);
                    f
                } else { 0 };
                println!("{} {} {} {} {} {} {} {}", i, px, py, ox, oy, n0, n1, flag);
            }
        }
        let mut disconnected_count = 0u32;
        for i in 0..num_cps as usize {
            let px = pos_data[i * 2]; let py = pos_data[i * 2 + 1];
            if px.abs() < 0.001 && py.abs() < 0.001 { continue; }
            let is_frac = (px.fract().abs() > 0.01) || (py.fract().abs() > 0.01);
            if is_frac { fractional_count += 1; } else { integer_count += 1; }
            let n0 = nbr_data[i * 4]; let n1 = nbr_data[i * 4 + 1];
            let flag = if let Some(ref fdl) = flag_dl {
                let fmap = fdl.map::<u32>(&device, false);
                let f = fmap.mem()[i];
                drop(fmap);
                f
            } else { 0 };
            // A CP is "disconnected" if it has a position but no neighbors and isn't a border CP
            if n0 < 0 && n1 < 0 && flag != 1 { disconnected_count += 1; }
        }
        eprintln!("GPU debug: {} integer CPs, {} fractional (diagonal) CPs, Disconnected: {}",
            integer_count, fractional_count, disconnected_count);

        // CPU rasterizer: mirror the GPU cell_rasterizer logic for debugging.
        // Downloads orig_positions, then runs the same algorithm
        // on CPU with debug output for artifact pixels.
        if std::env::var("CPU_RASTER").is_ok() {
            // Download orig_positions
            let orig_dl_size = num_cps * 2 * 4; // 2 f32 per CP
            let orig_dl = device.create_transfer_buffer()
                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
                .with_size(orig_dl_size).build().ok();
            if let Some(odl) = &orig_dl {
                let cmd3 = device.acquire_command_buffer().ok().unwrap();
                let cp3 = device.begin_copy_pass(&cmd3).ok().unwrap();
                unsafe {
                    let mut src = sdl3::sys::gpu::SDL_GPUBufferRegion::default();
                    src.buffer = orig_pos_buf.raw(); src.size = orig_dl_size;
                    let mut dst = sdl3::sys::gpu::SDL_GPUTransferBufferLocation::default();
                    dst.transfer_buffer = odl.raw();
                    sdl3::sys::gpu::SDL_DownloadFromGPUBuffer(cp3.raw(), &src, &dst);
                }
                device.end_copy_pass(cp3);
                let f3 = cmd3.submit_and_acquire_fence(&device).ok().unwrap();
                let _ = device.wait_fences(true, &[f3]);

                let omap = odl.map::<f32>(&device, false);
                let fmap2 = if let Some(ref fdl) = flag_dl {
                    Some(fdl.map::<u32>(&device, false))
                } else { None };

                let orig_pos = omap.mem();
                let flags_slice = fmap2.as_ref().map(|m| m.mem());

                cpu_rasterize_debug(
                    src, img_w, img_h,
                    pos_data, orig_pos, flags_slice.unwrap_or(&[]),
                    nbr_data,
                    out_w, out_h, scale as f32, corners_w,
                );
                drop(omap); drop(fmap2);
            }
        }

        drop(pos_map); drop(opt_map); drop(nbr_map);
    }

    // New command buffer for rasterizer
    let cmd = device.acquire_command_buffer().ok()?;

    // Tile-based rasterizer: one workgroup per 2×2 source tile
    { let tiles_w = (img_w + 1) / 2;
      let tiles_h = (img_h + 1) / 2;
      let total_tiles = tiles_w * tiles_h;
      let cp = device.begin_compute_pass(&cmd,
          &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex).with_cycle(true)],
          &[]).ok()?;
      cp.bind_compute_pipeline(&pipelines.rasterizer);
      cp.bind_compute_storage_buffers(0, &[px_buf.clone(), pos_buf.clone(), orig_pos_buf.clone(), flag_buf.clone(), nbr_buf.clone()]);
      #[repr(C)] struct U{iw:u32,ih:u32,ow:u32,oh:u32,s:f32,cw:u32,tw:u32,th:u32}
      cmd.push_compute_uniform_data(0,&U{iw:img_w,ih:img_h,ow:out_w,oh:out_h,s:scale as f32,cw:corners_w,tw:tiles_w,th:tiles_h});
      cp.dispatch(total_tiles,1,1); device.end_compute_pass(cp); }

    // Download from texture
    let dl_buf = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w * out_h * 4).build().ok()?;
    { let cp = device.begin_copy_pass(&cmd).ok()?;
      unsafe {
          let mut src_r = sdl3::sys::gpu::SDL_GPUTextureRegion::default();
          src_r.texture = out_tex.raw(); src_r.w = out_w; src_r.h = out_h; src_r.d = 1;
          let mut dst = sdl3::sys::gpu::SDL_GPUTextureTransferInfo::default();
          dst.transfer_buffer = dl_buf.raw();
          sdl3::sys::gpu::SDL_DownloadFromGPUTexture(cp.raw(), &src_r, &dst);
      }
      device.end_copy_pass(cp); }

    let fence = cmd.submit_and_acquire_fence(&device).ok()?;
    device.wait_fences(true, &[fence]).ok()?;

    let map = dl_buf.map::<u8>(&device, false);
    let bytes = map.mem();
    let mut pixels = vec![0u32; (out_w * out_h) as usize];
    for i in 0..pixels.len() {
        let off = i * 4;
        let b = bytes[off] as u32;
        let g = bytes[off + 1] as u32;
        let r = bytes[off + 2] as u32;
        pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
    }
    drop(map);
    Some((pixels, out_w, out_h))
}

// ── Diffusion compute pipeline ──────────────────────────────────────────────

pub fn init_diffusion_compute_pipeline(device: &gpu::Device) -> Option<gpu::ComputePipeline> {
    let comp_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/diffusion_raster_comp.spv"));
    let comp_msl = include_bytes!(concat!(env!("OUT_DIR"), "/diffusion_raster_comp.metal"));
    let comp_dxil = include_bytes!(concat!(env!("OUT_DIR"), "/diffusion_raster_comp.dxil"));

    let pipeline = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, comp_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(3) // pixels, regions, ownership
        .with_readwrite_storage_textures(1)
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| if !comp_dxil.is_empty() {
            device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::DXIL, comp_dxil)
                .with_entrypoint(c"main")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(3)
                .with_readwrite_storage_textures(1)
                .with_thread_count(16, 16, 1)
                .build()
        } else { Err(sdl3::get_error()) })
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, comp_msl)
            .with_entrypoint(c"main_0")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(3)
            .with_readwrite_storage_textures(1)
            .with_thread_count(16, 16, 1)
            .build());

    match pipeline {
        Ok(p) => { eprintln!("Diffusion GPU compute pipeline ready"); Some(p) }
        Err(e) => { eprintln!("Diffusion GPU compute: pipeline creation failed: {e}"); None }
    }
}

/// Prepare CPU-side buffers for the diffusion compute shader.
/// Returns (src_pixels, src_regions, packed_diags, out_w, out_h).
///
/// packed_diags: 1 uint per source pixel with 4 corner diagonal states
/// packed into bits [1:0]=TL, [3:2]=TR, [5:4]=BR, [7:6]=BL.
/// Each 2-bit value: 0=none, 1=backslash, 2=slash.
pub fn prepare_diffusion_data(
    pixels: &[u32], width: usize, height: usize,
    scale: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32, u32) {
    use crate::vectorize::rasterize::build_graph_regions;
    use crate::vectorize::graph;

    let graph = graph::build(pixels, width, height);
    let regions = build_graph_regions(width, height, &graph);

    // Pack diagonal states: 4 corners per pixel, 2 bits each
    let mut diags = vec![0u32; width * height];
    for py in 0..height {
        for px in 0..width {
            let tl = corner_diag_state(&graph, px, py) as u32;
            let tr = corner_diag_state(&graph, px + 1, py) as u32;
            let br = corner_diag_state(&graph, px + 1, py + 1) as u32;
            let bl = corner_diag_state(&graph, px, py + 1) as u32;
            diags[py * width + px] = tl | (tr << 2) | (br << 4) | (bl << 6);
        }
    }

    let out_w = (width * scale) as u32;
    let out_h = (height * scale) as u32;

    (pixels.to_vec(), regions, diags, out_w, out_h)
}

/// Get diagonal state at grid corner (cx, cy): 0=none, 1=backslash, 2=slash.
fn corner_diag_state(graph: &crate::vectorize::graph::SimilarityGraph, cx: usize, cy: usize) -> u8 {
    let w = graph.width;
    let h = graph.height;
    if cx == 0 || cy == 0 || cx >= w || cy >= h { return 0; }
    if graph.edge(cx - 1, cy - 1).down_right { return 1; }
    if graph.edge(cx, cy - 1).down_left { return 2; }
    0
}

pub fn diffusion_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipeline: &gpu::ComputePipeline,
    src_pixels: &[u32],
    src_regions: &[u32],
    diag_states: &[u32],
    src_w: u32, src_h: u32,
    out_w: u32, out_h: u32,
    scale: f32,
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");

    fn upload_u32_buffer(
        device: &gpu::Device, data: &[u32],
    ) -> (gpu::TransferBuffer, gpu::Buffer) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        let size = bytes.len().max(4) as u32;
        let transfer = device.create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
            .with_size(size)
            .build().expect("transfer buf");
        {
            let mut map = transfer.map::<u8>(device, true);
            map.mem_mut()[..bytes.len()].copy_from_slice(bytes);
            map.unmap();
        }
        let buf = device.create_buffer()
            .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)
            .with_size(size)
            .build().expect("storage buf");
        (transfer, buf)
    }

    let (px_xfer, px_buf) = upload_u32_buffer(device, src_pixels);
    let (reg_xfer, reg_buf) = upload_u32_buffer(device, src_regions);
    let (diag_xfer, diag_buf) = upload_u32_buffer(device, diag_states);

    {
        let copy_pass = device.begin_copy_pass(&cmd).expect("copy pass");
        for (xfer, buf, data) in [
            (&px_xfer, &px_buf, src_pixels),
            (&reg_xfer, &reg_buf, src_regions),
            (&diag_xfer, &diag_buf, diag_states),
        ] {
            copy_pass.upload_to_gpu_buffer(
                gpu::TransferBufferLocation::new().with_transfer_buffer(xfer),
                gpu::BufferRegion::new().with_buffer(buf).with_size((data.len() * 4).max(4) as u32),
                false,
            );
        }
        device.end_copy_pass(copy_pass);
    }

    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex).with_cycle(true)],
            &[],
        ).expect("compute pass");
        compute_pass.bind_compute_pipeline(pipeline);
        compute_pass.bind_compute_storage_buffers(0, &[px_buf, reg_buf, diag_buf]);

        #[repr(C)]
        struct Uniforms {
            out_w: u32, out_h: u32, src_w: u32, src_h: u32,
            inv_scale: f32, gauss_k: f32, radius: f32, _pad: u32,
        }
        cmd.push_compute_uniform_data(0, &Uniforms {
            out_w, out_h, src_w, src_h,
            inv_scale: 1.0 / scale,
            gauss_k: 2.5,
            radius: 2.0,
            _pad: 0,
        });
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let (dx, dy, dw, dh) = {
            let src_aspect = out_w as f32 / out_h as f32;
            let dst_aspect = sw_w as f32 / sw_h as f32;
            if dst_aspect > src_aspect {
                let dh = sw_h; let dw = (sw_h as f32 * src_aspect) as u32;
                ((sw_w - dw) / 2, 0, dw, dh)
            } else {
                let dw = sw_w; let dh = (sw_w as f32 / src_aspect) as u32;
                (0, (sw_h - dh) / 2, dw, dh)
            }
        };
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = out_w;
        blit_info.source.h = out_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = dx;
        blit_info.destination.y = dy;
        blit_info.destination.w = dw;
        blit_info.destination.h = dh;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(gpu::Filter::Nearest as i32);
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── Spline-diffusion compute pipeline (2-pass) ─────────────────────────────

pub fn init_spline_diffusion_pipelines(device: &gpu::Device) -> Option<(gpu::ComputePipeline, gpu::ComputePipeline)> {
    let pass1_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_to_buf_comp.spv"));
    let pass1_msl = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_to_buf_comp.metal"));
    let pass1_dxil = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_to_buf_comp.dxil"));
    let pass2_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.spv"));
    let pass2_msl = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.metal"));
    let pass2_dxil = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.dxil"));

    // Pass 1: scanline rasterize edges → storage buffer
    let p1 = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, pass1_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(3)        // edges, rows, indices
        .with_readwrite_storage_buffers(1)       // output color buffer
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| if !pass1_dxil.is_empty() {
            device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::DXIL, pass1_dxil)
                .with_entrypoint(c"main")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(3)
                .with_readwrite_storage_buffers(1)
                .with_thread_count(16, 16, 1)
                .build()
        } else { Err(sdl3::get_error()) })
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, pass1_msl)
            .with_entrypoint(c"main_0")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(3)
            .with_readwrite_storage_buffers(1)
            .with_thread_count(16, 16, 1)
            .build());

    // Pass 2: Gaussian diffusion reading buffer → texture
    let p2 = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, pass2_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(2)        // src_pixels, region_colors
        .with_readwrite_storage_textures(1)      // output texture
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| if !pass2_dxil.is_empty() {
            device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::DXIL, pass2_dxil)
                .with_entrypoint(c"main")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(2)
                .with_readwrite_storage_textures(1)
                .with_thread_count(16, 16, 1)
                .build()
        } else { Err(sdl3::get_error()) })
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, pass2_msl)
            .with_entrypoint(c"main_0")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(2)
            .with_readwrite_storage_textures(1)
            .with_thread_count(16, 16, 1)
            .build());

    match (p1, p2) {
        (Ok(a), Ok(b)) => { eprintln!("Spline-diffusion GPU pipeline ready (2-pass)"); Some((a, b)) }
        _ => { eprintln!("Spline-diffusion GPU: pipeline creation failed"); None }
    }
}

pub fn spline_diffusion_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pass1: &gpu::ComputePipeline,
    pass2: &gpu::ComputePipeline,
    edges: &[crate::vectorize::rasterize::GpuEdgeV2],
    row_ranges: &[crate::vectorize::rasterize::GpuRowRange],
    edge_indices: &[u32],
    src_pixels: &[u32],
    out_w: u32, out_h: u32,
    src_w: u32, src_h: u32,
    bg_color: u32,
    scale: u32,
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");

    fn upload_buf(device: &gpu::Device, data: &[u8], usage: gpu::BufferUsageFlags)
        -> (gpu::TransferBuffer, gpu::Buffer)
    {
        let size = data.len().max(4) as u32;
        let transfer = device.create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
            .with_size(size)
            .build().expect("transfer buf");
        {
            let mut map = transfer.map::<u8>(device, true);
            map.mem_mut()[..data.len()].copy_from_slice(data);
            map.unmap();
        }
        let buf = device.create_buffer()
            .with_usage(usage)
            .with_size(size)
            .build().expect("storage buf");
        (transfer, buf)
    }

    fn as_bytes<T>(slice: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * std::mem::size_of::<T>()) }
    }

    // Upload edge data (readonly)
    let (edge_xfer, edge_buf) = upload_buf(device, as_bytes(edges), gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);
    let (row_xfer, row_buf) = upload_buf(device, as_bytes(row_ranges), gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);
    let (idx_xfer, idx_buf) = upload_buf(device, as_bytes(edge_indices), gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);
    // Upload source pixels (readonly)
    let (px_xfer, px_buf) = upload_buf(device, as_bytes(src_pixels), gpu::BufferUsageFlags::COMPUTE_STORAGE_READ);


    // Intermediate buffer: flat colors from scanline rasterization (GPU-resident, readwrite then readonly)
    let region_buf_size = (out_w * out_h * 4).max(4);
    let region_buf = device.create_buffer()
        .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE)
        .with_size(region_buf_size)
        .build().expect("region buf");

    // Copy pass: upload all readonly data
    {
        let copy_pass = device.begin_copy_pass(&cmd).expect("copy pass");
        for (xfer, buf, size) in [
            (&edge_xfer, &edge_buf, as_bytes(edges).len()),
            (&row_xfer, &row_buf, as_bytes(row_ranges).len()),
            (&idx_xfer, &idx_buf, as_bytes(edge_indices).len()),
            (&px_xfer, &px_buf, as_bytes(src_pixels).len()),
        ] {
            copy_pass.upload_to_gpu_buffer(
                gpu::TransferBufferLocation::new().with_transfer_buffer(xfer),
                gpu::BufferRegion::new().with_buffer(buf).with_size(size.max(4) as u32),
                false,
            );
        }
        device.end_copy_pass(copy_pass);
    }

    // Pass 1: scanline rasterize edges → region_buf
    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[], // no storage textures
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&region_buf).with_cycle(false)],
        ).expect("compute pass 1");
        compute_pass.bind_compute_pipeline(pass1);
        compute_pass.bind_compute_storage_buffers(0, &[edge_buf, row_buf, idx_buf]);

        #[repr(C)]
        struct Pass1Uniforms { out_w: u32, out_h: u32, num_edges: u32, bg_color: u32 }
        cmd.push_compute_uniform_data(0, &Pass1Uniforms {
            out_w, out_h, num_edges: edges.len() as u32, bg_color,
        });
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

    // Pass 2: Gaussian diffusion reading region_buf → output texture
    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex).with_cycle(true)],
            &[],
        ).expect("compute pass 2");
        compute_pass.bind_compute_pipeline(pass2);
        compute_pass.bind_compute_storage_buffers(0, &[px_buf, region_buf]);

        #[repr(C)]
        struct Pass2Uniforms {
            out_w: u32, out_h: u32, src_w: u32, src_h: u32,
            inv_scale: f32, gauss_k: f32, radius: f32, scale_int: u32,
        }
        cmd.push_compute_uniform_data(0, &Pass2Uniforms {
            out_w, out_h, src_w, src_h,
            inv_scale: 1.0 / scale as f32,
            gauss_k: 2.5,
            radius: 2.0,
            scale_int: scale,
        });
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

    // Blit to swapchain
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let (dx, dy, dw, dh) = {
            let src_aspect = out_w as f32 / out_h as f32;
            let dst_aspect = sw_w as f32 / sw_h as f32;
            if dst_aspect > src_aspect {
                let dh = sw_h; let dw = (sw_h as f32 * src_aspect) as u32;
                ((sw_w - dw) / 2, 0, dw, dh)
            } else {
                let dw = sw_w; let dh = (sw_w as f32 / src_aspect) as u32;
                (0, (sw_h - dh) / 2, dw, dh)
            }
        };
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = out_w;
        blit_info.source.h = out_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = dx;
        blit_info.destination.y = dy;
        blit_info.destination.w = dw;
        blit_info.destination.h = dh;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(gpu::Filter::Nearest as i32);
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

