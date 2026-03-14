//! SDL3 GPU pipeline management for scaling filters.
//!
//! Contains all GPU texture, pipeline, and render functions used by the
//! SDL3 frontend. Separated from main.rs so the code can be shared
//! and eventually adapted for other backends (Metal, wgpu).

use sdl3::gpu;

// ── Texture helpers ─────────────────────────────────────────────────────────

pub fn create_texture(device: &gpu::Device, w: u32, h: u32) -> gpu::Texture<'static> {
    device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
            .with_width(w)
            .with_height(h)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
    ).expect("Failed to create GPU texture")
}

// ── Pixel upload helper ─────────────────────────────────────────────────────

/// Upload 0x00RRGGBB pixel data to a transfer buffer, setting alpha to 0xFF.
fn upload_pixels(
    device: &gpu::Device,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
) {
    let mut map = transfer_buf.map::<u8>(device, true);
    let dst = map.mem_mut();
    let byte_count = (tex_w * tex_h * 4) as usize;
    let src = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, byte_count)
    };
    dst[..byte_count].copy_from_slice(src);
    for i in (3..byte_count).step_by(4) {
        dst[i] = 0xFF;
    }
    map.unmap();
}

/// Copy pass: transfer buffer → GPU texture.
fn copy_to_texture(
    device: &gpu::Device,
    cmd: &gpu::CommandBuffer,
    transfer_buf: &gpu::TransferBuffer,
    gpu_tex: &gpu::Texture<'static>,
    tex_w: u32, tex_h: u32,
) {
    let copy_pass = device.begin_copy_pass(cmd).expect("Failed to begin copy pass");
    copy_pass.upload_to_gpu_texture(
        gpu::TextureTransferInfo::new()
            .with_transfer_buffer(transfer_buf),
        gpu::TextureRegion::new()
            .with_texture(gpu_tex)
            .with_width(tex_w)
            .with_height(tex_h)
            .with_depth(1),
        false,
    );
    device.end_copy_pass(copy_pass);
}

// ── Swapchain helpers ───────────────────────────────────────────────────────

/// Acquire swapchain texture. Returns (raw_ptr, width, height) or null ptr.
fn acquire_swapchain(
    cmd: &gpu::CommandBuffer,
    window: &sdl3::video::Window,
) -> (*mut sdl3::sys::gpu::SDL_GPUTexture, u32, u32) {
    let mut raw = std::ptr::null_mut();
    let mut w = 0u32;
    let mut h = 0u32;
    let got = unsafe {
        sdl3::sys::gpu::SDL_AcquireGPUSwapchainTexture(
            cmd.raw(), window.raw(), &mut raw, &mut w, &mut h,
        )
    };
    if got { (raw, w, h) } else { (std::ptr::null_mut(), 0, 0) }
}

/// Compute aspect-correct viewport for source aspect ratio within swapchain.
fn aspect_viewport(tex_w: u32, tex_h: u32, sw_w: u32, sw_h: u32) -> (f32, f32, f32, f32) {
    let src_aspect = tex_w as f32 / tex_h as f32;
    let dst_aspect = sw_w as f32 / sw_h as f32;
    if dst_aspect > src_aspect {
        let h = sw_h as f32;
        let w = h * src_aspect;
        ((sw_w as f32 - w) / 2.0, 0.0, w, h)
    } else {
        let w = sw_w as f32;
        let h = w / src_aspect;
        (0.0, (sw_h as f32 - h) / 2.0, w, h)
    }
}

/// Begin a render pass on the swapchain texture. Returns None if swapchain unavailable.
fn begin_swapchain_render_pass(
    cmd: &gpu::CommandBuffer,
    swapchain_raw: *mut sdl3::sys::gpu::SDL_GPUTexture,
) -> Option<gpu::RenderPass> {
    if swapchain_raw.is_null() { return None; }
    let mut color_info = sdl3::sys::gpu::SDL_GPUColorTargetInfo::default();
    color_info.texture = swapchain_raw;
    color_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
    color_info.store_op = sdl3::sys::gpu::SDL_GPUStoreOp::STORE;
    let raw = unsafe {
        sdl3::sys::gpu::SDL_BeginGPURenderPass(cmd.raw(), &color_info, 1, std::ptr::null())
    };
    if raw.is_null() { return None; }
    Some(unsafe { std::mem::transmute::<_, gpu::RenderPass>(raw) })
}

/// Submit command buffer and wait if swapchain was unavailable.
fn submit_and_sync(device: &gpu::Device, cmd: gpu::CommandBuffer, swapchain_was_null: bool) {
    cmd.submit().expect("Failed to submit GPU command buffer");
    if swapchain_was_null {
        unsafe { sdl3::sys::gpu::SDL_WaitForGPUIdle(device.raw()); }
    }
}

// ── Shader loading helper ───────────────────────────────────────────────────

/// Create a vertex shader with SPIR-V primary + MSL fallback.
fn load_vertex_shader(device: &gpu::Device) -> Result<gpu::Shader, sdl3::Error> {
    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen_vert.spv"));
    let msl = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen_vert.metal"));
    device.create_shader()
        .with_code(gpu::ShaderFormat::SPIRV, spirv, gpu::ShaderStage::Vertex)
        .with_entrypoint(c"main")
        .build()
        .or_else(|_| device.create_shader()
            .with_code(gpu::ShaderFormat::MSL, msl, gpu::ShaderStage::Vertex)
            .with_entrypoint(c"main0")
            .build())
}

/// Create a fragment shader with SPIR-V primary + MSL fallback.
fn load_fragment_shader(
    device: &gpu::Device,
    spirv: &[u8], msl: &[u8],
    samplers: u32, storage_buffers: u32, uniform_buffers: u32,
) -> Result<gpu::Shader, sdl3::Error> {
    device.create_shader()
        .with_code(gpu::ShaderFormat::SPIRV, spirv, gpu::ShaderStage::Fragment)
        .with_entrypoint(c"main")
        .with_samplers(samplers)
        .with_storage_buffers(storage_buffers)
        .with_uniform_buffers(uniform_buffers)
        .build()
        .or_else(|_| device.create_shader()
            .with_code(gpu::ShaderFormat::MSL, msl, gpu::ShaderStage::Fragment)
            .with_entrypoint(c"main0")
            .with_samplers(samplers)
            .with_storage_buffers(storage_buffers)
            .with_uniform_buffers(uniform_buffers)
            .build())
}

/// Create a fullscreen-triangle graphics pipeline with the given fragment shader.
fn create_fullscreen_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    vs: &gpu::Shader,
    fs: &gpu::Shader,
) -> Result<gpu::GraphicsPipeline, sdl3::Error> {
    let swapchain_fmt = device.get_swapchain_texture_format(window);
    device.create_graphics_pipeline()
        .with_vertex_shader(vs)
        .with_fragment_shader(fs)
        .with_primitive_type(gpu::PrimitiveType::TriangleList)
        .with_target_info(
            gpu::GraphicsPipelineTargetInfo::new()
                .with_color_target_descriptions(&[
                    gpu::ColorTargetDescription::new().with_format(swapchain_fmt)
                ])
        )
        .build()
}

// ── Blit (Nearest / Bilinear) ───────────────────────────────────────────────

pub fn upload_and_blit(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
    filter: gpu::Filter,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);

    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let (dx, dy, dw, dh) = {
            let src_aspect = tex_w as f32 / tex_h as f32;
            let dst_aspect = sw_w as f32 / sw_h as f32;
            if dst_aspect > src_aspect {
                let dh = sw_h;
                let dw = (sw_h as f32 * src_aspect) as u32;
                ((sw_w - dw) / 2, 0, dw, dh)
            } else {
                let dw = sw_w;
                let dh = (sw_w as f32 / src_aspect) as u32;
                (0, (sw_h - dh) / 2, dw, dh)
            }
        };
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = tex_w;
        blit_info.source.h = tex_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = dx;
        blit_info.destination.y = dy;
        blit_info.destination.w = dw;
        blit_info.destination.h = dh;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(filter as i32);
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── OmniScale pipeline ──────────────────────────────────────────────────────

pub fn init_omniscale_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("OmniScale GPU: vertex shader failed: {e}")).ok()?;
    let fs = load_fragment_shader(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_frag.metal")),
        1, 0, 1,
    ).map_err(|e| eprintln!("OmniScale GPU: fragment shader failed: {e}")).ok()?;

    match create_fullscreen_pipeline(device, window, &vs, &fs) {
        Ok(p) => { eprintln!("OmniScale GPU shader pipeline ready"); Some(p) }
        Err(e) => { eprintln!("OmniScale GPU: pipeline creation failed: {e}"); None }
    }
}

pub fn render_omniscale(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline,
    sampler: &gpu::Sampler,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);

    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(render_pass) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        render_pass.bind_graphics_pipeline(pipeline);
        device.set_viewport(&render_pass, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        render_pass.bind_fragment_samplers(0, &[
            gpu::TextureSamplerBinding::new()
                .with_texture(gpu_tex)
                .with_sampler(sampler)
        ]);

        #[repr(C)]
        struct Uniforms { src_size: [f32; 2], dst_size: [f32; 2], pixel_size: f32, pad: [f32; 3] }
        let pixel_size = ((tex_w as f32 / vw).powi(2) + (tex_h as f32 / vh).powi(2)).sqrt();
        cmd.push_fragment_uniform_data(0, &Uniforms {
            src_size: [tex_w as f32, tex_h as f32],
            dst_size: [vw, vh],
            pixel_size,
            pad: [0.0; 3],
        });
        render_pass.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(render_pass);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── HQx pipeline ────────────────────────────────────────────────────────────

pub fn init_hqx_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("HQx GPU: vertex shader failed: {e}")).ok()?;
    let fs = load_fragment_shader(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/hqx_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/hqx_frag.metal")),
        1, 0, 1,
    ).map_err(|e| eprintln!("HQx GPU: fragment shader failed: {e}")).ok()?;

    match create_fullscreen_pipeline(device, window, &vs, &fs) {
        Ok(p) => { eprintln!("HQx GPU shader pipeline ready"); Some(p) }
        Err(e) => { eprintln!("HQx GPU: pipeline creation failed: {e}"); None }
    }
}

pub fn render_hqx(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline,
    sampler: &gpu::Sampler,
    hqx_scale: f32,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);

    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(render_pass) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        render_pass.bind_graphics_pipeline(pipeline);
        device.set_viewport(&render_pass, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        render_pass.bind_fragment_samplers(0, &[
            gpu::TextureSamplerBinding::new()
                .with_texture(gpu_tex)
                .with_sampler(sampler)
        ]);

        #[repr(C)]
        struct HqxUniforms { src_size: [f32; 2], scale: f32, pad0: f32 }
        cmd.push_fragment_uniform_data(0, &HqxUniforms {
            src_size: [tex_w as f32, tex_h as f32],
            scale: hqx_scale,
            pad0: 0.0,
        });
        render_pass.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(render_pass);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

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
