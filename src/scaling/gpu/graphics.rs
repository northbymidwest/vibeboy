//\! Graphics pipeline init and render functions for scaling filters.

use sdl3::gpu;
use super::common::*;

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
        include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_frag.dxil")),
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
        include_bytes!(concat!(env!("OUT_DIR"), "/hqx_frag.dxil")),
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

// ── Bicubic pipeline ────────────────────────────────────────────────────────

pub fn init_bicubic_pipeline(
    device: &gpu::Device, window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("Bicubic GPU: vs failed: {e}")).ok()?;
    let fs = load_fragment_shader(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/bicubic_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/bicubic_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/bicubic_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("Bicubic GPU: fs failed: {e}")).ok()?;
    create_fullscreen_pipeline(device, window, &vs, &fs)
        .map_err(|e| eprintln!("Bicubic GPU: pipeline failed: {e}")).ok()
}

pub fn render_bicubic(
    device: &gpu::Device, window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>, transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32], tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline, sampler: &gpu::Sampler,
    dst_w: u32, dst_h: u32,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(rp) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        rp.bind_graphics_pipeline(pipeline);
        device.set_viewport(&rp, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        rp.bind_fragment_samplers(0, &[gpu::TextureSamplerBinding::new().with_texture(gpu_tex).with_sampler(sampler)]);
        #[repr(C)] struct U { src_size: [f32; 2], dst_size: [f32; 2] }
        cmd.push_fragment_uniform_data(0, &U {
            src_size: [tex_w as f32, tex_h as f32],
            dst_size: [dst_w as f32, dst_h as f32],
        });
        rp.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(rp);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── OmniScale Legacy pipeline ───────────────────────────────────────────────

pub fn init_omniscale_legacy_pipeline(
    device: &gpu::Device, window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("OmniScale Legacy GPU: vs failed: {e}")).ok()?;
    let fs = load_fragment_shader(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_legacy_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_legacy_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/omniscale_legacy_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("OmniScale Legacy GPU: fs failed: {e}")).ok()?;
    create_fullscreen_pipeline(device, window, &vs, &fs)
        .map_err(|e| eprintln!("OmniScale Legacy GPU: pipeline failed: {e}")).ok()
}

pub fn render_omniscale_legacy(
    device: &gpu::Device, window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>, transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32], tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline, sampler: &gpu::Sampler,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(rp) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        rp.bind_graphics_pipeline(pipeline);
        device.set_viewport(&rp, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        rp.bind_fragment_samplers(0, &[gpu::TextureSamplerBinding::new().with_texture(gpu_tex).with_sampler(sampler)]);
        #[repr(C)] struct U { src_size: [f32; 2], dst_size: [f32; 2] }
        cmd.push_fragment_uniform_data(0, &U {
            src_size: [tex_w as f32, tex_h as f32],
            dst_size: [sw_w as f32, sw_h as f32],
        });
        rp.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(rp);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── Scale3x pipeline ────────────────────────────────────────────────────────

pub fn init_scale3x_pipeline(
    device: &gpu::Device, window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("Scale3x GPU: vs failed: {e}")).ok()?;
    let fs = load_fragment_shader(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/scale3x_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/scale3x_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/scale3x_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("Scale3x GPU: fs failed: {e}")).ok()?;
    create_fullscreen_pipeline(device, window, &vs, &fs)
        .map_err(|e| eprintln!("Scale3x GPU: pipeline failed: {e}")).ok()
}

pub fn render_scale3x(
    device: &gpu::Device, window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>, transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32], tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline, sampler: &gpu::Sampler,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(rp) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        rp.bind_graphics_pipeline(pipeline);
        device.set_viewport(&rp, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        rp.bind_fragment_samplers(0, &[gpu::TextureSamplerBinding::new().with_texture(gpu_tex).with_sampler(sampler)]);
        #[repr(C)] struct U { src_size: [f32; 2], pad0: f32, pad1: f32 }
        cmd.push_fragment_uniform_data(0, &U { src_size: [tex_w as f32, tex_h as f32], pad0: 0.0, pad1: 0.0 });
        rp.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(rp);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── Eagle pipeline ──────────────────────────────────────────────────────────

pub fn init_eagle_pipeline(
    device: &gpu::Device, window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("Eagle GPU: vs failed: {e}")).ok()?;
    let fs = load_fragment_shader(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/eagle_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/eagle_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/eagle_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("Eagle GPU: fs failed: {e}")).ok()?;
    create_fullscreen_pipeline(device, window, &vs, &fs)
        .map_err(|e| eprintln!("Eagle GPU: pipeline failed: {e}")).ok()
}

pub fn render_eagle(
    device: &gpu::Device, window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>, transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32], tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline, sampler: &gpu::Sampler,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(rp) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        rp.bind_graphics_pipeline(pipeline);
        device.set_viewport(&rp, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        rp.bind_fragment_samplers(0, &[gpu::TextureSamplerBinding::new().with_texture(gpu_tex).with_sampler(sampler)]);
        #[repr(C)] struct U { src_size: [f32; 2], pad0: f32, pad1: f32 }
        cmd.push_fragment_uniform_data(0, &U { src_size: [tex_w as f32, tex_h as f32], pad0: 0.0, pad1: 0.0 });
        rp.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(rp);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── AA Nearest pipeline ─────────────────────────────────────────────────────

pub fn init_aa_nearest_pipeline(
    device: &gpu::Device, window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("AA Nearest GPU: vs failed: {e}")).ok()?;
    let fs = load_fragment_shader(device,
        include_bytes!(concat!(env!("OUT_DIR"), "/aa_nearest_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/aa_nearest_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/aa_nearest_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("AA Nearest GPU: fs failed: {e}")).ok()?;
    create_fullscreen_pipeline(device, window, &vs, &fs)
        .map_err(|e| eprintln!("AA Nearest GPU: pipeline failed: {e}")).ok()
}

pub fn render_aa_nearest(
    device: &gpu::Device, window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>, transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32], tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline, sampler: &gpu::Sampler,
    dst_w: u32, dst_h: u32,
) {
    upload_pixels(device, transfer_buf, pixels, tex_w, tex_h);
    let cmd = device.acquire_command_buffer().expect("cmd buf");
    copy_to_texture(device, &cmd, transfer_buf, gpu_tex, tex_w, tex_h);
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if let Some(rp) = begin_swapchain_render_pass(&cmd, swapchain_raw) {
        let (vx, vy, vw, vh) = aspect_viewport(tex_w, tex_h, sw_w, sw_h);
        rp.bind_graphics_pipeline(pipeline);
        device.set_viewport(&rp, gpu::Viewport::new(vx, vy, vw, vh, 0.0, 1.0));
        rp.bind_fragment_samplers(0, &[gpu::TextureSamplerBinding::new().with_texture(gpu_tex).with_sampler(sampler)]);
        #[repr(C)] struct U { src_size: [f32; 2], dst_size: [f32; 2] }
        cmd.push_fragment_uniform_data(0, &U {
            src_size: [tex_w as f32, tex_h as f32],
            dst_size: [dst_w as f32, dst_h as f32],
        });
        rp.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(rp);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── EPX / Scale2x / Scale4x pipeline ────────────────────────────────────────

pub fn init_epx_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("EPX GPU: vertex shader failed: {e}")).ok()?;
    let fs = load_fragment_shader(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/epx_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/epx_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/epx_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("EPX GPU: fragment shader failed: {e}")).ok()?;

    match create_fullscreen_pipeline(device, window, &vs, &fs) {
        Ok(p) => { eprintln!("EPX GPU shader pipeline ready"); Some(p) }
        Err(e) => { eprintln!("EPX GPU: pipeline creation failed: {e}"); None }
    }
}

pub fn render_epx(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline,
    sampler: &gpu::Sampler,
    epx_scale: f32,
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
        struct EpxUniforms { src_size: [f32; 2], scale: f32, pad0: f32 }
        cmd.push_fragment_uniform_data(0, &EpxUniforms {
            src_size: [tex_w as f32, tex_h as f32],
            scale: epx_scale,
            pad0: 0.0,
        });
        render_pass.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(render_pass);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── xBR pipeline ────────────────────────────────────────────────────────────

pub fn init_xbr_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("xBR GPU: vertex shader failed: {e}")).ok()?;
    let fs = load_fragment_shader(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/xbr_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/xbr_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/xbr_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("xBR GPU: fragment shader failed: {e}")).ok()?;

    match create_fullscreen_pipeline(device, window, &vs, &fs) {
        Ok(p) => { eprintln!("xBR GPU shader pipeline ready"); Some(p) }
        Err(e) => { eprintln!("xBR GPU: pipeline creation failed: {e}"); None }
    }
}

pub fn render_xbr(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline,
    sampler: &gpu::Sampler,
    xbr_scale: f32,
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
        struct XbrUniforms { src_size: [f32; 2], scale: f32, pad0: f32 }
        cmd.push_fragment_uniform_data(0, &XbrUniforms {
            src_size: [tex_w as f32, tex_h as f32],
            scale: xbr_scale,
            pad0: 0.0,
        });
        render_pass.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(render_pass);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── xBRZ pipeline ───────────────────────────────────────────────────────────

pub fn init_xbrz_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("xBRZ GPU: vertex shader failed: {e}")).ok()?;
    let fs = load_fragment_shader(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/xbrz_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/xbrz_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/xbrz_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("xBRZ GPU: fragment shader failed: {e}")).ok()?;

    match create_fullscreen_pipeline(device, window, &vs, &fs) {
        Ok(p) => { eprintln!("xBRZ GPU shader pipeline ready"); Some(p) }
        Err(e) => { eprintln!("xBRZ GPU: pipeline creation failed: {e}"); None }
    }
}

pub fn render_xbrz(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    transfer_buf: &gpu::TransferBuffer,
    pixels: &[u32],
    tex_w: u32, tex_h: u32,
    pipeline: &gpu::GraphicsPipeline,
    sampler: &gpu::Sampler,
    xbrz_scale: f32,
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
        struct XbrzUniforms { src_size: [f32; 2], scale: f32, pad0: f32 }
        cmd.push_fragment_uniform_data(0, &XbrzUniforms {
            src_size: [tex_w as f32, tex_h as f32],
            scale: xbrz_scale,
            pad0: 0.0,
        });
        render_pass.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(render_pass);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

// ── Super xBR pipeline ──────────────────────────────────────────────────────

pub fn init_super_xbr_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
) -> Option<gpu::GraphicsPipeline> {
    let vs = load_vertex_shader(device).map_err(|e| eprintln!("Super xBR GPU: vertex shader failed: {e}")).ok()?;
    let fs = load_fragment_shader(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_frag.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_frag.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/super_xbr_frag.dxil")),
        1, 0, 1,
    ).map_err(|e| eprintln!("Super xBR GPU: fragment shader failed: {e}")).ok()?;

    match create_fullscreen_pipeline(device, window, &vs, &fs) {
        Ok(p) => { eprintln!("Super xBR GPU shader pipeline ready"); Some(p) }
        Err(e) => { eprintln!("Super xBR GPU: pipeline creation failed: {e}"); None }
    }
}

pub fn render_super_xbr(
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
        struct SuperXbrUniforms { src_size: [f32; 2], pad0: f32, pad1: f32 }
        cmd.push_fragment_uniform_data(0, &SuperXbrUniforms {
            src_size: [tex_w as f32, tex_h as f32],
            pad0: 0.0, pad1: 0.0,
        });
        render_pass.draw_primitives(3, 1, 0, 0);
        device.end_render_pass(render_pass);
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

