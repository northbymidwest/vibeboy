//\! Headless GPU screenshot and rendering functions.

use sdl3::gpu;
use super::common::*;
use super::graphics::*;
use super::compute::*;

// ── Headless GPU screenshot ────────────────────────────────────────────────

/// Render a scaling filter on the GPU and read pixels back to CPU.
///
/// Creates a hidden SDL window and GPU device, renders the filter to an
/// offscreen texture, downloads the result. Returns (pixels, width, height).
pub fn gpu_screenshot(
    src: &[u32], src_w: u32, src_h: u32,
    filter: super::super::ScaleFilter,
) -> Option<(Vec<u32>, u32, u32)> {
    let factor = filter.factor();
    let (out_w, out_h) = if factor > 1 {
        (src_w * factor, src_h * factor)
    } else {
        (src_w * 4, src_h * 4)
    };

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_screenshot", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false).ok()?.with_window(&window).ok()?;

    let src_tex = create_texture(&device, src_w, src_h);
    let xfer = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(src_w * src_h * 4)
        .build().ok()?;
    upload_pixels(&device, &xfer, src, src_w, src_h);

    // Create offscreen render target
    let rt_tex = device.create_texture(
        gpu::TextureCreateInfo::new()
            .with_type(gpu::TextureType::_2D)
            .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
            .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COLOR_TARGET)
            .with_width(out_w).with_height(out_h)
            .with_layer_count_or_depth(1).with_num_levels(1)
    ).ok()?;

    let sampler = device.create_sampler(
        gpu::SamplerCreateInfo::new()
            .with_min_filter(gpu::Filter::Nearest)
            .with_mag_filter(gpu::Filter::Nearest)
    ).ok()?;

    let pipeline = match filter {
        super::super::ScaleFilter::Hqx(_) => init_hqx_pipeline(&device, &window),
        super::super::ScaleFilter::Xbr(_) => init_xbr_pipeline(&device, &window),
        super::super::ScaleFilter::Xbrz(_) => init_xbrz_pipeline(&device, &window),
        super::super::ScaleFilter::SuperXbr => init_super_xbr_pipeline(&device, &window),
        super::super::ScaleFilter::Epx | super::super::ScaleFilter::Scale2x | super::super::ScaleFilter::Scale4x
            => init_epx_pipeline(&device, &window),
        super::super::ScaleFilter::Scale3x => init_scale3x_pipeline(&device, &window),
        super::super::ScaleFilter::Eagle => init_eagle_pipeline(&device, &window),
        super::super::ScaleFilter::AaNearestNeighbor => init_aa_nearest_pipeline(&device, &window),
        super::super::ScaleFilter::Bicubic => init_bicubic_pipeline(&device, &window),
        super::super::ScaleFilter::OmniScale => init_omniscale_pipeline(&device, &window),
        super::super::ScaleFilter::OmniScaleLegacy => init_omniscale_legacy_pipeline(&device, &window),
        _ => return None,
    }?;

    // Single command buffer: upload → render → download
    let cmd = device.acquire_command_buffer().ok()?;
    copy_to_texture(&device, &cmd, &xfer, &src_tex, src_w, src_h);
    let mut color_info = sdl3::sys::gpu::SDL_GPUColorTargetInfo::default();
    color_info.texture = rt_tex.raw();
    color_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
    color_info.store_op = sdl3::sys::gpu::SDL_GPUStoreOp::STORE;
    let rp_raw = unsafe {
        sdl3::sys::gpu::SDL_BeginGPURenderPass(cmd.raw(), &color_info, 1, std::ptr::null())
    };
    if rp_raw.is_null() { return None; }
    let rp: gpu::RenderPass = unsafe { std::mem::transmute(rp_raw) };

    rp.bind_graphics_pipeline(&pipeline);
    device.set_viewport(&rp, gpu::Viewport::new(0.0, 0.0, out_w as f32, out_h as f32, 0.0, 1.0));
    rp.bind_fragment_samplers(0, &[
        gpu::TextureSamplerBinding::new().with_texture(&src_tex).with_sampler(&sampler)
    ]);

    // Push uniforms
    #[repr(C)] struct Uniforms4 { a: f32, b: f32, c: f32, d: f32 }
    let scale_f = match filter {
        super::super::ScaleFilter::Hqx(h) => h.factor() as f32,
        super::super::ScaleFilter::Xbr(x) => x.factor() as f32,
        super::super::ScaleFilter::Xbrz(x) => x.factor() as f32,
        super::super::ScaleFilter::Scale4x => 4.0,
        super::super::ScaleFilter::Epx | super::super::ScaleFilter::Scale2x => 2.0,
        _ => 0.0,
    };
    let needs_dst = matches!(filter,
        super::super::ScaleFilter::AaNearestNeighbor | super::super::ScaleFilter::Bicubic
        | super::super::ScaleFilter::OmniScale | super::super::ScaleFilter::OmniScaleLegacy);
    if needs_dst {
        cmd.push_fragment_uniform_data(0, &Uniforms4 {
            a: src_w as f32, b: src_h as f32, c: out_w as f32, d: out_h as f32,
        });
    } else {
        cmd.push_fragment_uniform_data(0, &Uniforms4 {
            a: src_w as f32, b: src_h as f32, c: scale_f, d: 0.0,
        });
    }

    rp.draw_primitives(3, 1, 0, 0);
    device.end_render_pass(rp);

    // Download pixels from render target
    let dl_buf = device.create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w * out_h * 4)
        .build().ok()?;
    let copy_pass = device.begin_copy_pass(&cmd).ok()?;
    // Use raw SDL3 API — the Rust wrapper doesn't expose download yet
    unsafe {
        let mut src_region = sdl3::sys::gpu::SDL_GPUTextureRegion::default();
        src_region.texture = rt_tex.raw();
        src_region.w = out_w;
        src_region.h = out_h;
        src_region.d = 1;
        let mut dst_info = sdl3::sys::gpu::SDL_GPUTextureTransferInfo::default();
        dst_info.transfer_buffer = dl_buf.raw();
        sdl3::sys::gpu::SDL_DownloadFromGPUTexture(copy_pass.raw(), &src_region, &dst_info);
    }
    device.end_copy_pass(copy_pass);
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
