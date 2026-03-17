//\! Compute pipeline init and dispatch for vectorize, diffusion, and spline-diffusion.

use sdl3::gpu;
use super::common::*;

// ── Vectorize compute pipeline ──────────────────────────────────────────────

pub fn init_vectorize_compute_pipeline(device: &gpu::Device) -> Option<gpu::ComputePipeline> {
    let comp_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.spv"));
    let comp_msl = include_bytes!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.metal"));

    let pipeline = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, comp_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(3)
        .with_readwrite_storage_textures(1)
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, comp_msl)
            .with_entrypoint(c"main0")
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

// ── Full GPU vectorize pipeline ─────────────────────────────────────────────
//
// Five-stage pipeline: similarity_graph → resolve_crossings → cell_graph →
// optimize_energy → cell_rasterizer. All stages run on GPU with no CPU
// readback between stages.

/// All pipelines for the full GPU vectorize pipeline.
pub struct GpuVectorizePipelines {
    pub sim_graph: gpu::ComputePipeline,
    pub resolve: gpu::ComputePipeline,
    pub cell_graph: gpu::ComputePipeline,
    pub optimizer: gpu::ComputePipeline,
    pub rasterizer: gpu::ComputePipeline,
}

pub fn init_full_gpu_pipeline(device: &gpu::Device) -> Option<GpuVectorizePipelines> {
    fn make(device: &gpu::Device, spirv: &[u8], msl: &[u8],
            ro_bufs: u32, rw_bufs: u32, rw_tex: u32, threads: (u32,u32,u32)) -> Option<gpu::ComputePipeline> {
        device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::SPIRV, spirv)
            .with_entrypoint(c"main")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(ro_bufs)
            .with_readwrite_storage_buffers(rw_bufs)
            .with_readwrite_storage_textures(rw_tex)
            .with_thread_count(threads.0, threads.1, threads.2)
            .build()
            .or_else(|_| device.create_compute_pipeline()
                .with_code(gpu::ShaderFormat::MSL, msl)
                .with_entrypoint(c"main0")
                .with_uniform_buffers(1)
                .with_readonly_storage_buffers(ro_bufs)
                .with_readwrite_storage_buffers(rw_bufs)
                .with_readwrite_storage_textures(rw_tex)
                .with_thread_count(threads.0, threads.1, threads.2)
                .build()).ok()
    }

    let sim = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.metal")),
        1, 1, 0, (16, 16, 1))?; // ro: pixels, rw: graph

    let resolve = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.metal")),
        0, 1, 0, (16, 16, 1))?; // rw: graph (set 1)

    let cell = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.metal")),
        1, 3, 0, (16, 16, 1))?; // ro: graph, rw: positions, neighbors, flags

    let opt = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.metal")),
        4, 1, 0, (256, 1, 1))?; // ro: pos_in, orig, neighbors, flags; rw: pos_out

    let rast = make(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.metal")),
        4, 0, 1, (16, 16, 1))?; // ro: pixels, positions, neighbors, flags; rw_tex: output

    eprintln!("Full GPU vectorize pipeline ready (5 stages)");
    Some(GpuVectorizePipelines { sim_graph: sim, resolve, cell_graph: cell, optimizer: opt, rasterizer: rast })
}

/// Run the full GPU vectorize pipeline and blit to window.
pub fn gpu_vectorize_full_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipelines: &GpuVectorizePipelines,
    pixels: &[u32],
    img_w: u32, img_h: u32,
    out_w: u32, out_h: u32,
    scale: f32,
) {
    let cmd = device.acquire_command_buffer().expect("cmd buf");

    let graph_stride = 2 * img_w + 1;
    let graph_h = 2 * img_h + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;

    // Create GPU buffers
    let rw = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE;
    let ro = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ;

    let px_size = (img_w * img_h * 4) as u32;
    let graph_size = (graph_stride * graph_h * 4) as u32;
    let pos_size = (num_cps * 2 * 4) as u32; // 2 floats per CP
    let nbr_size = (num_cps * 4 * 4) as u32; // 4 ints per CP
    let flag_size = (num_cps * 4) as u32;    // 1 uint per CP

    // Upload pixel data
    let px_xfer = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(px_size).build().expect("px xfer");
    {
        let mut map = px_xfer.map::<u8>(device, true);
        let bytes = unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        map.mem_mut()[..bytes.len()].copy_from_slice(bytes);
        map.unmap();
    }
    let px_buf = device.create_buffer().with_usage(ro).with_size(px_size).build().expect("px buf");

    // Create graph buffer (rw for sim_graph and resolve_crossings)
    let graph_buf = device.create_buffer().with_usage(rw).with_size(graph_size.max(4)).build().expect("graph buf");

    // Create cell graph output buffers
    let pos_buf = device.create_buffer().with_usage(rw).with_size(pos_size.max(4)).build().expect("pos buf");
    let nbr_buf = device.create_buffer().with_usage(rw).with_size(nbr_size.max(4)).build().expect("nbr buf");
    let flag_buf = device.create_buffer().with_usage(rw).with_size(flag_size.max(4)).build().expect("flag buf");

    // Optimizer output buffer (ping-pong)
    let opt_out_buf = device.create_buffer().with_usage(rw).with_size(pos_size.max(4)).build().expect("opt out buf");

    // Upload pixels
    {
        let cp = device.begin_copy_pass(&cmd).expect("copy pass");
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
            gpu::BufferRegion::new().with_buffer(&px_buf).with_size(px_size), false);
        device.end_copy_pass(cp);
    }

    // Stage 1: Similarity graph
    {
        let cp = device.begin_compute_pass(&cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&graph_buf).with_cycle(true)]).expect("sim pass");
        cp.bind_compute_pipeline(&pipelines.sim_graph);
        cp.bind_compute_storage_buffers(0, &[px_buf.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, graph_stride: u32, _p: u32 }
        cmd.push_compute_uniform_data(0, &U { img_w, img_h, graph_stride, _p: 0 });
        cp.dispatch((img_w + 15) / 16, (img_h + 15) / 16, 1);
        device.end_compute_pass(cp);
    }

    // Stage 2: Resolve crossings
    {
        let cp = device.begin_compute_pass(&cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&graph_buf).with_cycle(false)]).expect("resolve pass");
        cp.bind_compute_pipeline(&pipelines.resolve);
        #[repr(C)] struct U { img_w: u32, img_h: u32, graph_stride: u32, _p: u32 }
        cmd.push_compute_uniform_data(0, &U { img_w, img_h, graph_stride, _p: 0 });
        cp.dispatch((img_w.saturating_sub(1) + 15) / 16, (img_h.saturating_sub(1) + 15) / 16, 1);
        device.end_compute_pass(cp);
    }

    // Stage 3: Cell graph
    {
        let cp = device.begin_compute_pass(&cmd, &[],
            &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&pos_buf).with_cycle(true),
              gpu::StorageBufferReadWriteBinding::new().with_buffer(&nbr_buf).with_cycle(true),
              gpu::StorageBufferReadWriteBinding::new().with_buffer(&flag_buf).with_cycle(true),
            ]).expect("cell pass");
        cp.bind_compute_pipeline(&pipelines.cell_graph);
        cp.bind_compute_storage_buffers(0, &[graph_buf.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, graph_stride: u32, corners_w: u32 }
        cmd.push_compute_uniform_data(0, &U { img_w, img_h, graph_stride, corners_w });
        cp.dispatch((corners_w + 15) / 16, (corners_h + 15) / 16, 1);
        device.end_compute_pass(cp);
    }

    // Stage 4: Optimize energy (multi-pass ping-pong)
    // orig_positions always reads pos_buf; pos_in/pos_out alternate
    let num_opt_passes = 1u32;
    let mut cur_in = pos_buf.clone();
    let mut cur_out = opt_out_buf.clone();
    for _pass in 0..num_opt_passes {
        let cp = device.begin_compute_pass(&cmd, &[],
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
    // After loop: result is in cur_in (last swap moved output→input)
    let optimized_pos = cur_in;

    // Stage 5: Cell rasterizer
    {
        let cp = device.begin_compute_pass(&cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(gpu_tex).with_cycle(true)],
            &[]).expect("rast pass");
        cp.bind_compute_pipeline(&pipelines.rasterizer);
        // Bind order must match spirv-cross MSL: pixels, positions, flags, neighbors
        cp.bind_compute_storage_buffers(0, &[px_buf.clone(), optimized_pos.clone(), flag_buf.clone(), nbr_buf.clone()]);
        #[repr(C)] struct U { img_w: u32, img_h: u32, out_w: u32, out_h: u32,
                               scale: f32, corners_w: u32, _p0: u32, _p1: u32 }
        cmd.push_compute_uniform_data(0, &U {
            img_w, img_h, out_w, out_h, scale, corners_w, _p0: 0, _p1: 0 });
        cp.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
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

/// Headless GPU full-pipeline screenshot (creates own device).
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
    let nbr_buf = device.create_buffer().with_usage(ro).with_size(nbr_size.max(4)).build().ok()?;
    let flag_buf = device.create_buffer().with_usage(ro).with_size(flag_size.max(4)).build().ok()?;
    let opt_out_buf = device.create_buffer().with_usage(rw).with_size(pos_size.max(4)).build().ok()?;

    { let cp = device.begin_copy_pass(&cmd).ok()?;
      cp.upload_to_gpu_buffer(
          gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
          gpu::BufferRegion::new().with_buffer(&px_buf).with_size(px_size), false);
      device.end_copy_pass(cp); }

    // Stage 1-5 (same as gpu_vectorize_full_pipeline)
    { let cp = device.begin_compute_pass(&cmd, &[],
          &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&graph_buf).with_cycle(true)]).ok()?;
      cp.bind_compute_pipeline(&pipelines.sim_graph);
      cp.bind_compute_storage_buffers(0, &[px_buf.clone()]);
      #[repr(C)] struct U{w:u32,h:u32,s:u32,p:u32}
      cmd.push_compute_uniform_data(0,&U{w:img_w,h:img_h,s:graph_stride,p:0});
      cp.dispatch((img_w+15)/16,(img_h+15)/16,1); device.end_compute_pass(cp); }

    { let cp = device.begin_compute_pass(&cmd, &[],
          &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&graph_buf).with_cycle(false)]).ok()?;
      cp.bind_compute_pipeline(&pipelines.resolve);
      #[repr(C)] struct U{w:u32,h:u32,s:u32,p:u32}
      cmd.push_compute_uniform_data(0,&U{w:img_w,h:img_h,s:graph_stride,p:0});
      cp.dispatch((img_w.saturating_sub(1)+15)/16,(img_h.saturating_sub(1)+15)/16,1); device.end_compute_pass(cp); }

    { let cp = device.begin_compute_pass(&cmd, &[],
          &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&pos_buf).with_cycle(true),
            gpu::StorageBufferReadWriteBinding::new().with_buffer(&nbr_buf).with_cycle(true),
            gpu::StorageBufferReadWriteBinding::new().with_buffer(&flag_buf).with_cycle(true)]).ok()?;
      cp.bind_compute_pipeline(&pipelines.cell_graph);
      cp.bind_compute_storage_buffers(0, &[graph_buf.clone()]);
      #[repr(C)] struct U{w:u32,h:u32,s:u32,c:u32}
      cmd.push_compute_uniform_data(0,&U{w:img_w,h:img_h,s:graph_stride,c:corners_w});
      cp.dispatch((corners_w+15)/16,(corners_h+15)/16,1); device.end_compute_pass(cp); }

    { let cp = device.begin_compute_pass(&cmd, &[],
          &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&opt_out_buf).with_cycle(false)]).ok()?;
      cp.bind_compute_pipeline(&pipelines.optimizer);
      cp.bind_compute_storage_buffers(0, &[pos_buf.clone(), pos_buf.clone(), nbr_buf.clone(), flag_buf.clone()]);
      #[repr(C)] struct U{n:u32,g:f32,m:f32,s:f32}
      cmd.push_compute_uniform_data(0,&U{n:num_cps,g:0.01,m:0.25,s:2.5});
      cp.dispatch((num_cps+255)/256,1,1); device.end_compute_pass(cp); }

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
        drop(pos_map); drop(opt_map); drop(nbr_map);
    }

    // New command buffer for rasterizer
    let cmd = device.acquire_command_buffer().ok()?;

    { let cp = device.begin_compute_pass(&cmd,
          &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex).with_cycle(true)],
          &[]).ok()?;
      cp.bind_compute_pipeline(&pipelines.rasterizer);
      // Use optimizer output (smoothed positions)
      // Bind order must match spirv-cross MSL: pixels, positions, flags, neighbors
      cp.bind_compute_storage_buffers(0, &[px_buf.clone(), opt_out_buf.clone(), flag_buf.clone(), nbr_buf.clone()]);
      #[repr(C)] struct U{iw:u32,ih:u32,ow:u32,oh:u32,s:f32,cw:u32,p0:u32,p1:u32}
      cmd.push_compute_uniform_data(0,&U{iw:img_w,ih:img_h,ow:out_w,oh:out_h,s:scale as f32,cw:corners_w,p0:0,p1:0});
      cp.dispatch((out_w+15)/16,(out_h+15)/16,1); device.end_compute_pass(cp); }

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

    let pipeline = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, comp_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(3) // pixels, regions, ownership
        .with_readwrite_storage_textures(1)
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, comp_msl)
            .with_entrypoint(c"main0")
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
    let pass2_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.spv"));
    let pass2_msl = include_bytes!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.metal"));

    // Pass 1: scanline rasterize edges → storage buffer
    let p1 = device.create_compute_pipeline()
        .with_code(gpu::ShaderFormat::SPIRV, pass1_spirv)
        .with_entrypoint(c"main")
        .with_uniform_buffers(1)
        .with_readonly_storage_buffers(3)        // edges, rows, indices
        .with_readwrite_storage_buffers(1)       // output color buffer
        .with_thread_count(16, 16, 1)
        .build()
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, pass1_msl)
            .with_entrypoint(c"main0")
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
        .or_else(|_| device.create_compute_pipeline()
            .with_code(gpu::ShaderFormat::MSL, pass2_msl)
            .with_entrypoint(c"main0")
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

