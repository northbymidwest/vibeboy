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
        sdl3::sys::gpu::SDL_WaitAndAcquireGPUSwapchainTexture(
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

/// Submit command buffer and wait if swapchain was unavailable.
pub(super) fn submit_and_sync(device: &gpu::Device, cmd: gpu::CommandBuffer, swapchain_was_null: bool) {
    cmd.submit().expect("Failed to submit GPU command buffer");
    if swapchain_was_null {
        unsafe { sdl3::sys::gpu::SDL_WaitForGPUIdle(device.raw()); }
    }
}

// ── Upload and blit ─────────────────────────────────────────────────────────

/// Upload pixel data to a texture and blit to the swapchain with the given filter.
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
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = tex_w;
        blit_info.source.h = tex_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = vx as u32;
        blit_info.destination.y = vy as u32;
        blit_info.destination.w = vw as u32;
        blit_info.destination.h = vh as u32;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(filter as i32);
        unsafe { sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info); }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

