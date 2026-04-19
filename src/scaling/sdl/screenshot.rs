//! Headless GPU screenshot and rendering functions.

use sdl3::gpu;
use super::compute::*;

// ── Headless GPU screenshot (compute shader path) ─────────────────────────

/// Render a scaling filter on the GPU via compute shader and read pixels back.
///
/// Creates a hidden SDL window and GPU device, dispatches the compute shader
/// to an offscreen texture, downloads the result. Returns (pixels, width, height).
pub fn gpu_screenshot(
    src: &[u32], src_w: u32, src_h: u32,
    filter: super::super::ScaleFilter,
) -> Option<(Vec<u32>, u32, u32)> {
    use super::super::ScaleFilter;

    let factor = filter.factor();
    let (out_w, out_h) = if factor > 1 {
        (src_w * factor, src_h * factor)
    } else {
        (src_w * 4, src_h * 4)
    };

    let pipeline_init: fn(&gpu::Device) -> Option<gpu::ComputePipeline> = match filter {
        ScaleFilter::OmniScale => init_omniscale_compute_pipeline,
        ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Scale4x => init_epx_compute_pipeline,
        ScaleFilter::Eagle => init_eagle_compute_pipeline,
        ScaleFilter::Scale3x => init_scale3x_compute_pipeline,
        ScaleFilter::Bicubic => init_bicubic_compute_pipeline,
        ScaleFilter::NearestAa => init_nearest_aa_compute_pipeline,
        ScaleFilter::Hqx(_) => init_hqx_compute_pipeline,
        ScaleFilter::Xbr(_) => init_xbr_compute_pipeline,
        ScaleFilter::Xbrz(_) => init_xbrz_compute_pipeline,
        ScaleFilter::SuperXbr => init_super_xbr_compute_pipeline,
        ScaleFilter::OmniScaleLegacy => init_omniscale_legacy_compute_pipeline,
        ScaleFilter::Edi => init_edi_compute_pipeline,
        ScaleFilter::Nedi => init_nedi_compute_pipeline,
        ScaleFilter::Dcci => init_dcci_compute_pipeline,
        ScaleFilter::Mmpx => init_mmpx_compute_pipeline,
        ScaleFilter::LcdGrid => init_lcd_grid_compute_pipeline,
        _ => return None,
    };

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_screenshot", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false).ok()?.with_window(&window).ok()?;

    let pipeline = pipeline_init(&device)?;

    // Create output storage texture
    let out_tex = device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
            .with_width(out_w).with_height(out_h)
            .with_layer_count_or_depth(1).with_num_levels(1)
    ).ok()?;

    // Build uniforms
    let iscale = out_w / src_w;
    let extra = match filter {
        ScaleFilter::OmniScale => {
            let sx = src_w as f32 / out_w as f32;
            let sy = src_h as f32 / out_h as f32;
            f32::to_bits((sx * sx + sy * sy).sqrt())
        }
        ScaleFilter::Epx | ScaleFilter::Scale4x
        | ScaleFilter::Hqx(_) | ScaleFilter::Xbr(_) | ScaleFilter::Xbrz(_) => iscale,
        _ => 0,
    };
    let uniforms = [src_w, src_h, out_w, out_h, extra, 0, 0, 0];

    // Upload pixels + dispatch compute + download
    let cmd = device.acquire_command_buffer().ok()?;

    let px_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4)
    };
    let px_size = px_bytes.len().max(4) as u32;
    let px_xfer = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(px_size).build().ok()?;
    {
        let mut map = px_xfer.map::<u8>(&device, true);
        map.mem_mut()[..px_bytes.len()].copy_from_slice(px_bytes);
        map.unmap();
    }
    let px_buf = device.create_buffer()
        .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)
        .with_size(px_size).build().ok()?;

    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
            gpu::BufferRegion::new().with_buffer(&px_buf).with_size(px_size), false);
        device.end_copy_pass(cp);
    }

    let dispatch_x = out_w.div_ceil(16);
    let dispatch_y = out_h.div_ceil(16);

    if matches!(filter, ScaleFilter::SuperXbr) {
        // Super xBR: 3-pass pipeline with intermediate buffer
        let intermed_size = out_w * out_h * 4;
        let intermed_buf = device.create_buffer()
            .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE)
            .with_size(intermed_size.max(4)).build().ok()?;

        #[repr(C)]
        struct SxbrUniforms { src_w: u32, src_h: u32, out_w: u32, out_h: u32, pass: u32, _pad: [u32; 3] }

        for pass_idx in 0u32..3 {
            let cp = device.begin_compute_pass(
                &cmd,
                &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex)
                    .with_cycle(pass_idx == 0)],
                &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&intermed_buf.clone())
                    .with_cycle(pass_idx == 0)],
            ).ok()?;
            cp.bind_compute_pipeline(&pipeline);
            cp.bind_compute_storage_buffers(0, &[px_buf.clone()]);
            cmd.push_compute_uniform_data(0, &SxbrUniforms {
                src_w, src_h, out_w, out_h, pass: pass_idx, _pad: [0; 3],
            });
            cp.dispatch(dispatch_x, dispatch_y, 1);
            device.end_compute_pass(cp);
        }
    } else {
        // Standard single-pass filters
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex).with_cycle(true)],
            &[],
        ).ok()?;
        compute_pass.bind_compute_pipeline(&pipeline);
        compute_pass.bind_compute_storage_buffers(0, &[px_buf]);
        #[repr(C)] struct RawUniforms([u32; 8]);
        cmd.push_compute_uniform_data(0, &RawUniforms(uniforms));
        compute_pass.dispatch(dispatch_x, dispatch_y, 1);
        device.end_compute_pass(compute_pass);
    }

    // Download pixels
    let dl_buf = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w * out_h * 4).build().ok()?;
    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        unsafe {
            let mut src_region = sdl3::sys::gpu::SDL_GPUTextureRegion::default();
            src_region.texture = out_tex.raw();
            src_region.w = out_w; src_region.h = out_h; src_region.d = 1;
            let mut dst_info = sdl3::sys::gpu::SDL_GPUTextureTransferInfo::default();
            dst_info.transfer_buffer = dl_buf.raw();
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
    drop(map);

    Some((pixels, out_w, out_h))
}
