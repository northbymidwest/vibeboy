//! Headless GPU screenshot and rendering functions.

use super::scale::{ScaleBufCache, encode_scale, init_scale_pipeline};
use sdl3::gpu;

// ── Headless GPU screenshot (compute shader path) ─────────────────────────

/// Render a scaling filter on the GPU via compute shader and read pixels back.
///
/// Creates a hidden SDL window and GPU device, dispatches the compute shader
/// to an offscreen texture, downloads the result. Returns (pixels, width, height).
pub fn gpu_screenshot(
    src: &[u32],
    src_w: u32,
    src_h: u32,
    filter: super::super::ScaleFilter,
) -> Option<(Vec<u32>, u32, u32)> {
    let shader = filter.gpu()?;
    // Adaptive filters render at 4x.
    let (out_w, out_h) = filter.output_size(src_w, src_h, (src_w * 4, src_h * 4));

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_screenshot", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV
        | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC
        | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false)
        .ok()?
        .with_window(&window)
        .ok()?;

    let pipeline = init_scale_pipeline(&device, shader)?;

    // Create output storage texture
    let out_tex = device
        .create_texture(
            gpu::TextureCreateInfo::new()
                .with_type(gpu::TextureType::_2D)
                .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
                .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
                .with_width(out_w)
                .with_height(out_h)
                .with_layer_count_or_depth(1)
                .with_num_levels(1),
        )
        .ok()?;

    // Upload pixels + dispatch compute + download
    let cmd = device.acquire_command_buffer().ok()?;
    let mut bufs = ScaleBufCache::default();
    encode_scale(
        &device, &cmd, &pipeline, shader, &mut bufs, &out_tex, src, src_w, src_h, out_w, out_h,
    );

    // Download pixels
    let dl_buf = device
        .create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w * out_h * 4)
        .build()
        .ok()?;
    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        unsafe {
            let src_region = sdl3::sys::gpu::SDL_GPUTextureRegion {
                texture: out_tex.raw(),
                w: out_w,
                h: out_h,
                d: 1,
                ..Default::default()
            };
            let dst_info = sdl3::sys::gpu::SDL_GPUTextureTransferInfo {
                transfer_buffer: dl_buf.raw(),
                ..Default::default()
            };
            sdl3::sys::gpu::SDL_DownloadFromGPUTexture(cp.raw(), &src_region, &dst_info);
        }
        device.end_copy_pass(cp);
    }
    let fence = cmd.submit_and_acquire_fence(&device).ok()?;
    device.wait_fences(true, &[fence]).ok()?;

    // Read back pixels (BGRA -> ARGB)
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
    map.unmap();

    Some((pixels, out_w, out_h))
}
