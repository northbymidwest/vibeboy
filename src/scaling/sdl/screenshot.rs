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
        ScaleFilter::AaNearestNeighbor => init_aa_nearest_compute_pipeline,
        ScaleFilter::Hqx(_) => init_hqx_compute_pipeline,
        ScaleFilter::Xbr(_) => init_xbr_compute_pipeline,
        ScaleFilter::Xbrz(_) => init_xbrz_compute_pipeline,
        ScaleFilter::SuperXbr => init_super_xbr_compute_pipeline,
        ScaleFilter::OmniScaleLegacy => init_omniscale_legacy_compute_pipeline,
        ScaleFilter::Edi => init_edi_compute_pipeline,
        ScaleFilter::Nedi => init_nedi_compute_pipeline,
        ScaleFilter::Dcci => init_dcci_compute_pipeline,
        ScaleFilter::Mmpx => init_mmpx_compute_pipeline,
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

    let dispatch_x = (out_w + 15) / 16;
    let dispatch_y = (out_h + 15) / 16;

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

    // Read back pixels (BGRA → ARGB)
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

/// GPU-accelerated vectorize screenshot.
///
/// Runs CPU vectorization to extract paths, then dispatches the compute
/// shader to rasterize them on the GPU, downloads the result.
pub fn gpu_vectorize_screenshot(
    src: &[u32], src_w: usize, src_h: usize, scale: f64, adaptive: bool,
) -> Option<(Vec<u32>, u32, u32)> {
    let mut cache = crate::vectorize::VectorizeCache::new_legacy(adaptive);
    let (paths, bg_color) = cache.get_paths(src, src_w, src_h);
    let (gpu_edges, row_ranges, edge_indices, out_w, out_h) =
        crate::vectorize::rasterize::prepare_gpu_edges_v2(paths, bg_color, scale, src_w, src_h);

    if out_w == 0 || out_h == 0 || gpu_edges.is_empty() {
        return None;
    }

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_vectorize", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false).ok()?.with_window(&window).ok()?;

    let compute_pipeline = init_vectorize_compute_pipeline(&device)?;

    // Create the output storage texture
    let out_tex = device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
            .with_width(out_w)
            .with_height(out_h)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
    ).ok()?;

    let cmd = device.acquire_command_buffer().ok()?;

    // Upload edge data to GPU storage buffers
    fn upload_buf(
        device: &gpu::Device, data: &[u8], usage: gpu::BufferUsageFlags,
    ) -> Option<(gpu::TransferBuffer, gpu::Buffer)> {
        let size = data.len().max(4) as u32;
        let xfer = device.create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
            .with_size(size)
            .build().ok()?;
        {
            let mut map = xfer.map::<u8>(device, true);
            map.mem_mut()[..data.len()].copy_from_slice(data);
            map.unmap();
        }
        let buf = device.create_buffer().with_usage(usage).with_size(size).build().ok()?;
        Some((xfer, buf))
    }

    let edge_bytes = unsafe {
        std::slice::from_raw_parts(
            gpu_edges.as_ptr() as *const u8,
            gpu_edges.len() * std::mem::size_of::<crate::vectorize::rasterize::GpuEdgeV2>(),
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

    let (edge_xfer, edge_buf) = upload_buf(&device, edge_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)?;
    let (row_xfer, row_buf) = upload_buf(&device, row_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)?;
    let (idx_xfer, idx_buf) = upload_buf(&device, idx_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)?;

    // Upload buffers to GPU
    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&edge_xfer),
            gpu::BufferRegion::new().with_buffer(&edge_buf).with_size(edge_bytes.len().max(4) as u32),
            false,
        );
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&row_xfer),
            gpu::BufferRegion::new().with_buffer(&row_buf).with_size(row_bytes.len().max(4) as u32),
            false,
        );
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&idx_xfer),
            gpu::BufferRegion::new().with_buffer(&idx_buf).with_size(idx_bytes.len().max(4) as u32),
            false,
        );
        device.end_copy_pass(cp);
    }

    // Dispatch compute shader
    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex).with_cycle(true)],
            &[],
        ).ok()?;
        compute_pass.bind_compute_pipeline(&compute_pipeline);
        compute_pass.bind_compute_storage_buffers(0, &[edge_buf, row_buf, idx_buf]);

        #[repr(C)]
        struct Uniforms { out_w: u32, out_h: u32, num_edges: u32, bg_color: u32 }
        cmd.push_compute_uniform_data(0, &Uniforms {
            out_w, out_h, num_edges: gpu_edges.len() as u32, bg_color,
        });
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

    // Download pixels
    let dl_buf = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w * out_h * 4)
        .build().ok()?;
    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        unsafe {
            let mut src_region = sdl3::sys::gpu::SDL_GPUTextureRegion::default();
            src_region.texture = out_tex.raw();
            src_region.w = out_w;
            src_region.h = out_h;
            src_region.d = 1;
            let mut dst_info = sdl3::sys::gpu::SDL_GPUTextureTransferInfo::default();
            dst_info.transfer_buffer = dl_buf.raw();
            sdl3::sys::gpu::SDL_DownloadFromGPUTexture(cp.raw(), &src_region, &dst_info);
        }
        device.end_copy_pass(cp);
    }

    let fence = cmd.submit_and_acquire_fence(&device).ok()?;
    device.wait_fences(true, &[fence]).ok()?;

    // Read back (BGRA → ARGB)
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

/// GPU-accelerated shared-chain vectorize screenshot (headless).
/// Uses the winding-number fill pipeline with shared boundary chains
/// for gap-free rendering.
pub fn gpu_vectorize_shared_screenshot(
    src: &[u32], src_w: usize, src_h: usize, scale: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    let graph = crate::vectorize::graph::build(src, src_w, src_h);
    let (paths, bg_color) = crate::vectorize::contour::extract_shared_edge_paths(src, &graph);
    gpu_vectorize_screenshot_with_paths(&paths, bg_color, scale as f64, src_w, src_h)
}

/// Shared implementation for GPU vectorize screenshots given pre-built paths.
fn gpu_vectorize_screenshot_with_paths(
    paths: &[crate::vectorize::contour::ColorPath],
    bg_color: u32, scale: f64, src_w: usize, src_h: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    let (gpu_edges, row_ranges, edge_indices, out_w, out_h) =
        crate::vectorize::rasterize::prepare_gpu_edges_v2(paths, bg_color, scale, src_w, src_h);

    if out_w == 0 || out_h == 0 || gpu_edges.is_empty() {
        return None;
    }

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_vectorize", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false).ok()?.with_window(&window).ok()?;

    let compute_pipeline = init_vectorize_compute_pipeline(&device)?;

    let out_tex = device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
            .with_width(out_w)
            .with_height(out_h)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
    ).ok()?;

    let cmd = device.acquire_command_buffer().ok()?;

    fn upload_buf(
        device: &gpu::Device, data: &[u8], usage: gpu::BufferUsageFlags,
    ) -> Option<(gpu::TransferBuffer, gpu::Buffer)> {
        let size = data.len().max(4) as u32;
        let xfer = device.create_transfer_buffer()
            .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
            .with_size(size).build().ok()?;
        { let mut map = xfer.map::<u8>(device, true);
          map.mem_mut()[..data.len()].copy_from_slice(data); map.unmap(); }
        let buf = device.create_buffer().with_usage(usage).with_size(size).build().ok()?;
        Some((xfer, buf))
    }

    let edge_bytes = unsafe { std::slice::from_raw_parts(
        gpu_edges.as_ptr() as *const u8,
        gpu_edges.len() * std::mem::size_of::<crate::vectorize::rasterize::GpuEdgeV2>(),
    )};
    let row_bytes = unsafe { std::slice::from_raw_parts(
        row_ranges.as_ptr() as *const u8,
        row_ranges.len() * std::mem::size_of::<crate::vectorize::rasterize::GpuRowRange>(),
    )};
    let idx_bytes = unsafe { std::slice::from_raw_parts(
        edge_indices.as_ptr() as *const u8, edge_indices.len() * 4,
    )};

    let (edge_xfer, edge_buf) = upload_buf(&device, edge_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)?;
    let (row_xfer, row_buf) = upload_buf(&device, row_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)?;
    let (idx_xfer, idx_buf) = upload_buf(&device, idx_bytes, gpu::BufferUsageFlags::COMPUTE_STORAGE_READ)?;

    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&edge_xfer),
            gpu::BufferRegion::new().with_buffer(&edge_buf).with_size(edge_bytes.len().max(4) as u32), false);
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&row_xfer),
            gpu::BufferRegion::new().with_buffer(&row_buf).with_size(row_bytes.len().max(4) as u32), false);
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&idx_xfer),
            gpu::BufferRegion::new().with_buffer(&idx_buf).with_size(idx_bytes.len().max(4) as u32), false);
        device.end_copy_pass(cp);
    }

    {
        let compute_pass = device.begin_compute_pass(
            &cmd,
            &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex).with_cycle(true)],
            &[],
        ).ok()?;
        compute_pass.bind_compute_pipeline(&compute_pipeline);
        compute_pass.bind_compute_storage_buffers(0, &[edge_buf, row_buf, idx_buf]);

        #[repr(C)]
        struct Uniforms { out_w: u32, out_h: u32, num_edges: u32, bg_color: u32 }
        cmd.push_compute_uniform_data(0, &Uniforms {
            out_w, out_h, num_edges: gpu_edges.len() as u32, bg_color,
        });
        compute_pass.dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
        device.end_compute_pass(compute_pass);
    }

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

/// GPU-accelerated spline-diffusion screenshot.
pub fn gpu_spline_diffusion_screenshot(
    src: &[u32], src_w: usize, src_h: usize, scale: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    crate::vectorize::contour::YUV_VISIBLE_EDGES
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let mut cache = crate::vectorize::VectorizeCache::new_legacy(false);
    let (paths, bg_color) = cache.get_paths(src, src_w, src_h);
    crate::vectorize::contour::YUV_VISIBLE_EDGES
        .store(false, std::sync::atomic::Ordering::Relaxed);

    let (gpu_edges, row_ranges, edge_indices, out_w, out_h) =
        crate::vectorize::rasterize::prepare_gpu_edges_v2(paths, bg_color, scale as f64, src_w, src_h);
    if out_w == 0 || out_h == 0 || gpu_edges.is_empty() { return None; }

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_sdiff", 1, 1).hidden().build().ok()?;
    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false).ok()?.with_window(&window).ok()?;
    let (p1, p2) = init_spline_diffusion_pipelines(&device)?;

    let out_tex = device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
            .with_width(out_w).with_height(out_h)
            .with_layer_count_or_depth(1).with_num_levels(1)
    ).ok()?;

    let cmd = device.acquire_command_buffer().ok()?;
    fn b<T>(s: &[T]) -> &[u8] { unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * std::mem::size_of::<T>()) } }
    fn u(d: &gpu::Device, data: &[u8], usage: gpu::BufferUsageFlags) -> Option<(gpu::TransferBuffer, gpu::Buffer)> {
        let sz = data.len().max(4) as u32;
        let x = d.create_transfer_buffer().with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD).with_size(sz).build().ok()?;
        { let mut m = x.map::<u8>(d, true); m.mem_mut()[..data.len()].copy_from_slice(data); m.unmap(); }
        let buf = d.create_buffer().with_usage(usage).with_size(sz).build().ok()?;
        Some((x, buf))
    }
    let rd = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ;
    let (ex,eb) = u(&device, b(&gpu_edges), rd)?;
    let (rx,rb) = u(&device, b(&row_ranges), rd)?;
    let (ix,ib) = u(&device, b(&edge_indices), rd)?;
    let (px,pb) = u(&device, b(src), rd)?;
    let region_buf = device.create_buffer()
        .with_usage(gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE)
        .with_size((out_w * out_h * 4).max(4)).build().ok()?;

    { let cp = device.begin_copy_pass(&cmd).ok()?;
      for (xf,bf,sz) in [(&ex,&eb,b(&gpu_edges).len()),(&rx,&rb,b(&row_ranges).len()),
                          (&ix,&ib,b(&edge_indices).len()),(&px,&pb,b(src).len())] {
        cp.upload_to_gpu_buffer(gpu::TransferBufferLocation::new().with_transfer_buffer(xf),
            gpu::BufferRegion::new().with_buffer(bf).with_size(sz.max(4) as u32), false);
      }
      device.end_copy_pass(cp);
    }
    { let cp = device.begin_compute_pass(&cmd, &[],
          &[gpu::StorageBufferReadWriteBinding::new().with_buffer(&region_buf).with_cycle(false)]).ok()?;
      cp.bind_compute_pipeline(&p1);
      cp.bind_compute_storage_buffers(0, &[eb, rb, ib]);
      #[repr(C)] struct U1 { ow:u32, oh:u32, ne:u32, bg:u32 }
      cmd.push_compute_uniform_data(0, &U1{ow:out_w,oh:out_h,ne:gpu_edges.len() as u32,bg:bg_color});
      cp.dispatch((out_w+15)/16,(out_h+15)/16,1);
      device.end_compute_pass(cp);
    }
    { let cp = device.begin_compute_pass(&cmd,
          &[gpu::StorageTextureReadWriteBinding::new().with_texture(&out_tex).with_cycle(true)], &[]).ok()?;
      cp.bind_compute_pipeline(&p2);
      cp.bind_compute_storage_buffers(0, &[pb, region_buf]);
      #[repr(C)] struct U2 { ow:u32, oh:u32, sw:u32, sh:u32, is:f32, s2:f32, r:f32, si:u32 }
      cmd.push_compute_uniform_data(0, &U2{ow:out_w,oh:out_h,sw:src_w as u32,sh:src_h as u32,
          is:1.0/scale as f32,s2:2.5,r:2.0,si:scale as u32});
      cp.dispatch((out_w+15)/16,(out_h+15)/16,1);
      device.end_compute_pass(cp);
    }
    let dl = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w*out_h*4).build().ok()?;
    { let cp = device.begin_copy_pass(&cmd).ok()?;
      unsafe {
        let mut tr = sdl3::sys::gpu::SDL_GPUTextureRegion::default();
        tr.texture = out_tex.raw(); tr.w = out_w; tr.h = out_h; tr.d = 1;
        let mut di = sdl3::sys::gpu::SDL_GPUTextureTransferInfo::default();
        di.transfer_buffer = dl.raw();
        sdl3::sys::gpu::SDL_DownloadFromGPUTexture(cp.raw(), &tr, &di);
      }
      device.end_copy_pass(cp);
    }
    let fence = cmd.submit_and_acquire_fence(&device).ok()?;
    device.wait_fences(true, &[fence]).ok()?;
    let map = dl.map::<u8>(&device, false);
    let byt = map.mem();
    let mut out_px = vec![0u32; (out_w*out_h) as usize];
    for i in 0..out_px.len() {
        let o = i*4;
        out_px[i] = 0xFF000000|((byt[o+2] as u32)<<16)|((byt[o+1] as u32)<<8)|byt[o] as u32;
    }
    drop(map);
    Some((out_px, out_w, out_h))
}
