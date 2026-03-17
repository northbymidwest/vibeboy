//! Shared GPU helpers: texture creation, pixel upload, swapchain, shader loading.

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
pub(super) fn upload_pixels(
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
pub(super) fn copy_to_texture(
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
pub(super) fn acquire_swapchain(
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
pub(super) fn aspect_viewport(tex_w: u32, tex_h: u32, sw_w: u32, sw_h: u32) -> (f32, f32, f32, f32) {
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
pub(super) fn begin_swapchain_render_pass(
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
pub(super) fn submit_and_sync(device: &gpu::Device, cmd: gpu::CommandBuffer, swapchain_was_null: bool) {
    cmd.submit().expect("Failed to submit GPU command buffer");
    if swapchain_was_null {
        unsafe { sdl3::sys::gpu::SDL_WaitForGPUIdle(device.raw()); }
    }
}

// ── Shader loading helper ───────────────────────────────────────────────────

/// Create a vertex shader with SPIR-V primary + MSL fallback.
pub(super) fn load_vertex_shader(device: &gpu::Device) -> Result<gpu::Shader, sdl3::Error> {
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
pub(super) fn load_fragment_shader(
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
pub(super) fn create_fullscreen_pipeline(
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

