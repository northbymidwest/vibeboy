//! GPU rendering for the GTK4 frontend using wgpu.
//!
//! Creates a wgpu surface from the GTK4 window's native handle (macOS NSView)
//! and renders frames via a fullscreen blit shader, bypassing Cairo for display.
//! Cairo remains available as a fallback when GPU init fails.

use wgpu::util::DeviceExt;

const BLIT_SHADER: &str = r#"
struct Uniforms {
    scale: vec2<f32>,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct VsOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOutput {
    let x = f32(vi & 1u);
    let y = f32((vi >> 1u) & 1u);
    let uv = vec2<f32>(x, y);
    let pos = vec2<f32>(
        (x * 2.0 - 1.0) * uniforms.scale.x,
        (1.0 - y * 2.0) * uniforms.scale.y,
    );
    var out: VsOutput;
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, tex_sampler, in.uv);
}
"#;

pub(super) struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    texture: Option<wgpu::Texture>,
    tex_w: u32,
    tex_h: u32,
    scale_factor: u32,
    #[cfg(target_os = "macos")]
    ns_view: std::ptr::NonNull<std::ffi::c_void>,
}

impl GpuRenderer {
    /// Try to create a GPU renderer from a GTK4 window's native handle.
    /// Returns None if the platform is unsupported or GPU init fails.
    pub fn from_gtk_window(window: &gtk4::ApplicationWindow) -> Option<Self> {
        use gtk4::prelude::*;

        // Get the GDK surface from the GTK native interface
        let native = window.native()?;
        let gdk_surface = native.surface()?;

        // Get native window handle (platform-specific)
        let handles = get_native_handles(&gdk_surface)?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let surface = unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(handles.raw_display),
                raw_window_handle: handles.raw_window,
            }).ok()?
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })).ok()?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        )).ok()?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        // Get initial size in physical pixels (GTK4 reports logical points)
        let scale = gdk_surface.scale_factor() as u32;
        let width = (window.width().max(1) as u32) * scale;
        let height = (window.height().max(1) as u32) * scale;

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        eprintln!("GPU renderer initialized ({:?})", adapter.get_info().backend);

        Some(GpuRenderer {
            device,
            queue,
            surface,
            surface_config,
            pipeline,
            sampler,
            bind_group_layout,
            texture: None,
            tex_w: 0,
            tex_h: 0,
            scale_factor: scale,
            #[cfg(target_os = "macos")]
            ns_view: handles.ns_view,
        })
    }

    /// Resize the surface. `width`/`height` are in logical points (GTK4 units).
    pub fn resize(&mut self, width: u32, height: u32) {
        let pw = width * self.scale_factor;
        let ph = height * self.scale_factor;
        if pw == 0 || ph == 0 {
            return;
        }
        if pw == self.surface_config.width && ph == self.surface_config.height {
            return;
        }
        self.surface_config.width = pw;
        self.surface_config.height = ph;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Render a frame buffer directly to the window surface.
    /// `pixels` is the (possibly scaled) pixel buffer, `frame_w x frame_h`.
    /// `src_w x src_h` is the logical source size for aspect ratio calculation.
    pub fn render(&mut self, pixels: &[u32], frame_w: u32, frame_h: u32, src_w: u32, src_h: u32) {
        // Recreate texture if dimensions changed
        if self.tex_w != frame_w || self.tex_h != frame_h {
            self.texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: frame_w, height: frame_h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
            self.tex_w = frame_w;
            self.tex_h = frame_h;
        }

        let texture = self.texture.as_ref().unwrap();

        // Convert 0x00RRGGBB to BGRA bytes
        let bgra: Vec<u8> = pixels.iter().flat_map(|&p| {
            let r = ((p >> 16) & 0xFF) as u8;
            let g = ((p >> 8) & 0xFF) as u8;
            let b = (p & 0xFF) as u8;
            [b, g, r, 0xFF]
        }).collect();

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bgra,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame_w * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d { width: frame_w, height: frame_h, depth_or_array_layers: 1 },
        );

        // Aspect-correct scaling
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let src_aspect = src_w as f32 / src_h as f32;
        let win_aspect = win_w / win_h;
        let (scale_x, scale_y) = if win_aspect > src_aspect {
            (src_aspect / win_aspect, 1.0)
        } else {
            (1.0, win_aspect / src_aspect)
        };

        let uniform_data = [scale_x, scale_y, 0.0f32, 0.0f32];
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let surface_view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

/// Native handle info extracted from GDK surface.
struct NativeHandles {
    raw_window: raw_window_handle::RawWindowHandle,
    raw_display: raw_window_handle::RawDisplayHandle,
    #[cfg(target_os = "macos")]
    ns_view: std::ptr::NonNull<std::ffi::c_void>,
}

/// Extract raw window and display handles from a GDK surface.
#[cfg(target_os = "macos")]
fn get_native_handles(gdk_surface: &gtk4::gdk::Surface) -> Option<NativeHandles> {
    use gtk4::glib::prelude::*;
    use raw_window_handle::{AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle};

    // Downcast GdkSurface to GdkMacosSurface
    let macos_surface: &gdk4_macos::MacosSurface = gdk_surface.downcast_ref()?;

    // Get the NSWindow pointer
    let ns_window = macos_surface.native();
    if ns_window.is_null() {
        return None;
    }

    // Get contentView from NSWindow
    let ns_view: *mut std::ffi::c_void = unsafe {
        objc2::msg_send![ns_window as *const objc2::runtime::AnyObject, contentView]
    };
    let ns_view = std::ptr::NonNull::new(ns_view)?;

    Some(NativeHandles {
        raw_window: RawWindowHandle::AppKit(AppKitWindowHandle::new(ns_view)),
        raw_display: RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
        ns_view,
    })
}

/// Extract raw window and display handles on Wayland.
#[cfg(all(not(target_os = "macos"), feature = "gdk4-wayland"))]
fn get_wayland_handles(gdk_surface: &gtk4::gdk::Surface, gdk_display: &gtk4::gdk::Display) -> Option<NativeHandles> {
    use gtk4::glib::prelude::*;
    use raw_window_handle::{WaylandDisplayHandle, WaylandWindowHandle, RawDisplayHandle, RawWindowHandle};

    let wl_surface: &gdk4_wayland::WaylandSurface = gdk_surface.downcast_ref()?;
    let wl_display: &gdk4_wayland::WaylandDisplay = gdk_display.downcast_ref()?;

    let surface_ptr = unsafe { gdk4_wayland::ffi::gdk_wayland_surface_get_wl_surface(
        wl_surface.as_ptr() as *mut _
    ) };
    let display_ptr = unsafe { gdk4_wayland::ffi::gdk_wayland_display_get_wl_display(
        wl_display.as_ptr() as *mut _
    ) };

    if surface_ptr.is_null() || display_ptr.is_null() {
        return None;
    }

    Some(NativeHandles {
        raw_window: RawWindowHandle::Wayland(WaylandWindowHandle::new(
            std::ptr::NonNull::new(surface_ptr as *mut _)?,
        )),
        raw_display: RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
            std::ptr::NonNull::new(display_ptr as *mut _)?,
        )),
    })
}

/// Extract raw window and display handles on X11.
#[cfg(all(not(target_os = "macos"), feature = "gdk4-x11"))]
fn get_x11_handles(gdk_surface: &gtk4::gdk::Surface, gdk_display: &gtk4::gdk::Display) -> Option<NativeHandles> {
    use gtk4::glib::prelude::*;
    use raw_window_handle::{XlibDisplayHandle, XlibWindowHandle, RawDisplayHandle, RawWindowHandle};

    let x11_surface: &gdk4_x11::X11Surface = gdk_surface.downcast_ref()?;
    let x11_display: &gdk4_x11::X11Display = gdk_display.downcast_ref()?;

    let xid = unsafe { gdk4_x11::ffi::gdk_x11_surface_get_xid(
        x11_surface.as_ptr() as *mut _
    ) };
    let xdisplay = unsafe { gdk4_x11::ffi::gdk_x11_display_get_xdisplay(
        x11_display.as_ptr() as *mut _
    ) };

    if xdisplay.is_null() {
        return None;
    }

    let mut window_handle = XlibWindowHandle::new(xid as _);
    let display_handle = XlibDisplayHandle::new(
        std::ptr::NonNull::new(xdisplay as *mut _),
        0, // screen number — GDK doesn't easily expose this, 0 is usually correct
    );

    Some(NativeHandles {
        raw_window: RawWindowHandle::Xlib(window_handle),
        raw_display: RawDisplayHandle::Xlib(display_handle),
    })
}

/// Linux: try Wayland first, then X11.
#[cfg(not(target_os = "macos"))]
fn get_native_handles(gdk_surface: &gtk4::gdk::Surface) -> Option<NativeHandles> {
    use gtk4::prelude::*;
    let display = gdk_surface.display();

    #[cfg(feature = "gdk4-wayland")]
    if display.is_wayland() {
        if let Some(h) = get_wayland_handles(gdk_surface, &display) {
            return Some(h);
        }
    }

    #[cfg(feature = "gdk4-x11")]
    if display.is_x11() {
        if let Some(h) = get_x11_handles(gdk_surface, &display) {
            return Some(h);
        }
    }

    eprintln!("GPU rendering: unsupported display server (need Wayland or X11)");
    None
}
