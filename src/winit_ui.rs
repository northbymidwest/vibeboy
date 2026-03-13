mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod model;
mod ppu;
mod printer;
mod serial;
mod sgb;
mod snapshot;
mod snes;
mod timer;
mod scaling;
mod vectorize;

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use emulator::Emulator;
use muda::{
    AboutMetadata, CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu,
    accelerator::{Accelerator, Code, Modifiers},
};
use model::GbModel;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};
use gilrs::{Gilrs, GamepadId, Button as GilButton, Axis as GilAxis};

const SCALE: u32 = 3;
const GB_W: u32 = 160;
const GB_H: u32 = 144;
const SGB_W: u32 = 256;
const SGB_H: u32 = 224;
const AUDIO_SAMPLE_RATE: u32 = 96_000;

fn frame_duration(model: GbModel) -> Duration {
    let nanos = 70_224u64 * 1_000_000_000 / model.cpu_clock_rate() as u64;
    Duration::from_nanos(nanos)
}

// ── GPU Renderer ─────────────────────────────────────────────────────────────

struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    // Cached texture + bind group, recreated when frame dimensions change
    texture: Option<wgpu::Texture>,
    bind_group: Option<wgpu::BindGroup>,
    tex_w: u32,
    tex_h: u32,
}

impl GpuRenderer {
    fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .unwrap();

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
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
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        GpuRenderer {
            device,
            queue,
            surface,
            surface_config,
            pipeline,
            sampler,
            bind_group_layout,
            texture: None,
            bind_group: None,
            tex_w: 0,
            tex_h: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.bind_group = None;
    }

    fn render(&mut self, pixels: &[u32], frame_w: u32, frame_h: u32, src_w: u32, src_h: u32) {
        // Recreate texture if frame dimensions changed
        if self.tex_w != frame_w || self.tex_h != frame_h {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: frame_w,
                    height: frame_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.texture = Some(texture);
            self.tex_w = frame_w;
            self.tex_h = frame_h;
            // bind_group will be recreated below
            self.bind_group = None;
        }

        let texture = self.texture.as_ref().unwrap();

        // Convert 0x00RRGGBB to BGRA bytes for upload
        let bgra: Vec<u8> = pixels
            .iter()
            .flat_map(|&p| {
                let r = ((p >> 16) & 0xFF) as u8;
                let g = ((p >> 8) & 0xFF) as u8;
                let b = (p & 0xFF) as u8;
                [b, g, r, 0xFF]
            })
            .collect();

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
            wgpu::Extent3d {
                width: frame_w,
                height: frame_h,
                depth_or_array_layers: 1,
            },
        );

        // Compute aspect-correct viewport as normalized rect [-1,1]
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let src_aspect = src_w as f32 / src_h as f32;
        let win_aspect = win_w / win_h;
        let (scale_x, scale_y) = if win_aspect > src_aspect {
            (src_aspect / win_aspect, 1.0)
        } else {
            (1.0, win_aspect / src_aspect)
        };

        // Uniform buffer: [scale_x, scale_y, 0, 0]
        let uniform_data = [scale_x, scale_y, 0.0f32, 0.0f32];
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        if self.bind_group.is_none() {
            // Recreate bind group (texture or uniform changed)
            let view = texture.create_view(&Default::default());
            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..4, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

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
    // Fullscreen quad as triangle strip: 0=TL, 1=TR, 2=BL, 3=BR
    let x = f32(vi & 1u);
    let y = f32((vi >> 1u) & 1u);
    let uv = vec2<f32>(x, y);
    // Map to clip space with aspect-correct scaling
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

#[derive(Parser)]
#[command(name = "vibeboy", about = "Game Boy / Game Boy Color emulator (winit frontend)")]
struct Cli {
    /// Path to ROM file (.gb / .gbc). If omitted, a file dialog will open.
    rom: Option<PathBuf>,

    /// Path to boot ROM file (auto-detected if not specified)
    #[arg(long)]
    bootrom: Option<PathBuf>,

    /// Hardware model (auto-detected from cart header if not specified)
    #[arg(long, value_parser = |s: &str| s.parse::<GbModel>())]
    model: Option<GbModel>,

    /// Skip boot ROM
    #[arg(long)]
    no_boot: bool,
}

// ── Menu item IDs ────────────────────────────────────────────────────────────

const ID_OPEN: &str = "open_rom";
const ID_QUIT: &str = "quit";
const ID_PAUSE: &str = "pause";
const ID_RESET: &str = "reset";

fn slot_save_id(n: usize) -> String {
    format!("slot_save_{}", n)
}
fn slot_load_id(n: usize) -> String {
    format!("slot_load_{}", n)
}

/// All filter menu entries: (menu_id, display_name, ScaleFilter).
/// Filters that support arbitrary resolution (Bilinear, Bicubic, OmniScale,
/// AA Nearest, Vectorize) appear as a single entry — no per-scale variants.
fn filter_entries() -> Vec<(&'static str, &'static str, scaling::ScaleFilter)> {
    use scaling::*;
    vec![
        ("filter_nearest",     "Nearest",       ScaleFilter::Nearest),
        ("filter_bilinear",    "Bilinear",      ScaleFilter::Bilinear),
        ("filter_bicubic",     "Bicubic",       ScaleFilter::Bicubic),
        ("filter_epx",         "EPX / Scale2x", ScaleFilter::Epx),
        ("filter_scale3x",    "Scale3x",       ScaleFilter::Scale3x),
        ("filter_scale4x",    "Scale4x",       ScaleFilter::Scale4x),
        ("filter_eagle",       "Eagle",         ScaleFilter::Eagle),
        ("filter_2xsai",       "2xSaI",         ScaleFilter::Sai2x),
        ("filter_s2xsai",     "Super 2xSaI",   ScaleFilter::Super2xSai),
        ("filter_seagle",      "Super Eagle",   ScaleFilter::SuperEagle),
        ("filter_hq2x",       "HQ2x",          ScaleFilter::Hqx(HqxScale::Hq2x)),
        ("filter_hq3x",       "HQ3x",          ScaleFilter::Hqx(HqxScale::Hq3x)),
        ("filter_hq4x",       "HQ4x",          ScaleFilter::Hqx(HqxScale::Hq4x)),
        ("filter_xbr2x",      "xBR 2x",        ScaleFilter::Xbr(XbrScale::Xbr2x)),
        ("filter_xbr3x",      "xBR 3x",        ScaleFilter::Xbr(XbrScale::Xbr3x)),
        ("filter_xbr4x",      "xBR 4x",        ScaleFilter::Xbr(XbrScale::Xbr4x)),
        ("filter_xbr_hybrid", "xBR Hybrid",    ScaleFilter::XbrHybrid),
        ("filter_super_xbr",  "Super xBR",     ScaleFilter::SuperXbr),
        ("filter_xbrz2x",     "xBRZ 2x",       ScaleFilter::Xbrz(XbrzScale::Xbrz2x)),
        ("filter_xbrz3x",     "xBRZ 3x",       ScaleFilter::Xbrz(XbrzScale::Xbrz3x)),
        ("filter_xbrz4x",     "xBRZ 4x",       ScaleFilter::Xbrz(XbrzScale::Xbrz4x)),
        ("filter_xbrz5x",     "xBRZ 5x",       ScaleFilter::Xbrz(XbrzScale::Xbrz5x)),
        ("filter_xbrz6x",     "xBRZ 6x",       ScaleFilter::Xbrz(XbrzScale::Xbrz6x)),
        ("filter_nedi",        "NEDI",          ScaleFilter::Nedi),
        ("filter_dcci",        "DCCI",          ScaleFilter::Dcci),
        ("filter_edi",         "EDI",           ScaleFilter::Edi),
        ("filter_omniscale",   "OmniScale",     ScaleFilter::OmniScale),
        ("filter_omniscale_l", "OmniScale Legacy", ScaleFilter::OmniScaleLegacy),
        ("filter_aanear",      "AA Nearest",    ScaleFilter::AaNearestNeighbor),
        ("filter_vectorize",   "Vectorize",           ScaleFilter::Vectorize),
        ("filter_vec_adapt",   "Vectorize Adaptive",  ScaleFilter::VectorizeAdaptive),
    ]
}

fn filter_id_to_filter(id: &str) -> Option<scaling::ScaleFilter> {
    filter_entries().iter().find(|(mid, _, _)| *mid == id).map(|(_, _, f)| *f)
}

// ── Audio ────────────────────────────────────────────────────────────────────

struct AudioRing {
    buf: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
}

impl AudioRing {
    fn new(capacity: usize) -> Self {
        AudioRing {
            buf: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            capacity,
        }
    }

    fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            let next = (self.write_pos + 1) % self.capacity;
            if next == self.read_pos {
                self.read_pos = (self.read_pos + 1) % self.capacity;
            }
            self.buf[self.write_pos] = s;
            self.write_pos = next;
        }
    }

    fn pop(&mut self) -> f32 {
        if self.read_pos == self.write_pos {
            return 0.0;
        }
        let val = self.buf[self.read_pos];
        self.read_pos = (self.read_pos + 1) % self.capacity;
        val
    }
}

fn start_audio(ring: Arc<Mutex<AudioRing>>) -> Option<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;
    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(AUDIO_SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Fixed(512),
    };

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut ring = ring.lock().unwrap();
                for sample in data.iter_mut() {
                    *sample = ring.pop();
                }
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )
        .ok()?;
    stream.play().ok()?;
    Some(stream)
}

// ── Camera (nokhwa) ──────────────────────────────────────────────────────────

struct CameraThread {
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CameraThread {
    fn start() -> Option<Self> {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};

        // Check if any camera is available
        let devices = nokhwa::query(nokhwa::utils::ApiBackend::Auto).ok()?;
        if devices.is_empty() {
            log::info!("No cameras found — using noise generator for Pocket Camera");
            return None;
        }

        let buffer: Arc<Mutex<[u8; 128 * 112]>> = Arc::new(Mutex::new([0u8; 128 * 112]));
        let has_new_frame = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let buf_clone = Arc::clone(&buffer);
        let frame_clone = Arc::clone(&has_new_frame);
        let stop_clone = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                camera_thread_main(buf_clone, frame_clone, stop_clone);
            })
            .expect("failed to spawn camera thread");

        log::info!("Camera capture thread started");
        Some(CameraThread {
            buffer,
            has_new_frame,
            stop,
            handle: Some(handle),
        })
    }

    fn read_frame(&self, buf: &mut [u8; 128 * 112]) -> bool {
        if self.has_new_frame.swap(false, Ordering::Acquire) {
            let lock = self.buffer.lock().unwrap();
            buf.copy_from_slice(&*lock);
            true
        } else {
            false
        }
    }
}

impl Drop for CameraThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn camera_thread_main(
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
    use nokhwa::Camera;

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestResolution(
        Resolution::new(640, 480),
    ));
    let mut camera = match Camera::new(CameraIndex::Index(0), format) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("Camera thread: failed to open camera: {}", e);
            return;
        }
    };

    if let Err(e) = camera.open_stream() {
        log::warn!("Camera thread: failed to start stream: {}", e);
        return;
    }

    log::info!("Camera thread: webcam opened");

    while !stop.load(Ordering::Acquire) {
        match camera.frame() {
            Ok(frame) => {
                let decoded = match frame.decode_image::<RgbFormat>() {
                    Ok(img) => img,
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                };

                let src_w = decoded.width();
                let src_h = decoded.height();

                // Convert RGB to RGBA for the image crate
                let rgba = image::RgbaImage::from_fn(src_w, src_h, |x, y| {
                    let p = decoded.get_pixel(x, y);
                    image::Rgba([p[0], p[1], p[2], 255])
                });

                // Crop to 8:7 aspect ratio (128:112)
                let target_ratio = 128.0 / 112.0;
                let src_ratio = src_w as f64 / src_h as f64;
                let (crop_w, crop_h) = if src_ratio > target_ratio {
                    let cw = (src_h as f64 * target_ratio) as u32;
                    (cw, src_h)
                } else {
                    let ch = (src_w as f64 / target_ratio) as u32;
                    (src_w, ch)
                };
                let crop_x = (src_w - crop_w) / 2;
                let crop_y = (src_h - crop_h) / 2;
                let cropped =
                    image::imageops::crop_imm(&rgba, crop_x, crop_y, crop_w, crop_h).to_image();

                // Convert to grayscale and resize
                let gray = image::imageops::grayscale(&cropped);
                let resized = image::imageops::resize(
                    &gray,
                    128,
                    112,
                    image::imageops::FilterType::Lanczos3,
                );

                {
                    let mut lock = buffer.lock().unwrap();
                    lock.copy_from_slice(resized.as_raw());
                }
                has_new_frame.store(true, Ordering::Release);
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    let _ = camera.stop_stream();
    log::info!("Camera thread: shut down");
}

// ── Menu builder ─────────────────────────────────────────────────────────────

fn build_menu() -> (Menu, Vec<(CheckMenuItem, scaling::ScaleFilter)>) {
    let menu = Menu::new();

    // File menu
    let file_menu = Submenu::new("File", true);
    file_menu
        .append_items(&[
            &MenuItem::with_id(
                ID_OPEN,
                "Open ROM...",
                true,
                Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyO)),
            ),
            &PredefinedMenuItem::separator(),
            &MenuItem::with_id(
                ID_QUIT,
                "Quit",
                true,
                Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyQ)),
            ),
        ])
        .unwrap();

    // Emulation menu
    let emu_menu = Submenu::new("Emulation", true);
    emu_menu
        .append_items(&[
            &MenuItem::with_id(
                ID_PAUSE,
                "Pause",
                true,
                Some(Accelerator::new(None, Code::F6)),
            ),
            &MenuItem::with_id(ID_RESET, "Reset", true, None::<Accelerator>),
        ])
        .unwrap();

    // State menu with Save/Load submenus
    let state_menu = Submenu::new("State", true);
    let save_sub = Submenu::new("Save State", true);
    let load_sub = Submenu::new("Load State", true);
    for i in 1..=9 {
        save_sub
            .append(&MenuItem::with_id(
                slot_save_id(i),
                format!("Slot {}", i),
                true,
                if i == 1 {
                    Some(Accelerator::new(None, Code::F5))
                } else {
                    None
                },
            ))
            .unwrap();
        load_sub
            .append(&MenuItem::with_id(
                slot_load_id(i),
                format!("Slot {}", i),
                true,
                if i == 1 {
                    Some(Accelerator::new(None, Code::F7))
                } else {
                    None
                },
            ))
            .unwrap();
    }
    state_menu
        .append_items(&[&save_sub, &load_sub])
        .unwrap();

    // Filter menu — use CheckMenuItem for checkmark support
    let filter_menu = Submenu::new("Filter", true);
    let mut filter_items = Vec::new();
    {
        let hqx_sub = Submenu::new("HQx", true);
        let xbr_sub = Submenu::new("xBR", true);
        let xbrz_sub = Submenu::new("xBRZ", true);
        let edge_sub = Submenu::new("Edge-Directed", true);

        for (id, name, filter) in filter_entries() {
            let checked = filter == scaling::ScaleFilter::Nearest;
            let item = CheckMenuItem::with_id(id, name, true, checked, None::<Accelerator>);
            match filter {
                scaling::ScaleFilter::Hqx(_) => { hqx_sub.append(&item).unwrap(); }
                scaling::ScaleFilter::Xbr(_)
                | scaling::ScaleFilter::XbrHybrid
                | scaling::ScaleFilter::SuperXbr => { xbr_sub.append(&item).unwrap(); }
                scaling::ScaleFilter::Xbrz(_) => { xbrz_sub.append(&item).unwrap(); }
                scaling::ScaleFilter::Nedi
                | scaling::ScaleFilter::Dcci
                | scaling::ScaleFilter::Edi => { edge_sub.append(&item).unwrap(); }
                _ => { filter_menu.append(&item).unwrap(); }
            }
            filter_items.push((item, filter));
        }

        filter_menu.append(&PredefinedMenuItem::separator()).unwrap();
        filter_menu.append_items(&[&hqx_sub, &xbr_sub, &xbrz_sub, &edge_sub]).unwrap();
    }

    // Help menu
    let help_menu = Submenu::new("Help", true);
    help_menu
        .append(&PredefinedMenuItem::about(
            Some("About VibeBoy"),
            Some(AboutMetadata {
                name: Some("VibeBoy".into()),
                version: Some(env!("CARGO_PKG_VERSION").into()),
                ..Default::default()
            }),
        ))
        .unwrap();

    #[cfg(target_os = "macos")]
    {
        let app_menu = Submenu::new("VibeBoy", true);
        app_menu
            .append_items(&[
                &PredefinedMenuItem::about(
                    Some("About VibeBoy"),
                    Some(AboutMetadata {
                        name: Some("VibeBoy".into()),
                        version: Some(env!("CARGO_PKG_VERSION").into()),
                        ..Default::default()
                    }),
                ),
                &PredefinedMenuItem::separator(),
                &PredefinedMenuItem::quit(Some("Quit VibeBoy")),
            ])
            .unwrap();
        menu.append(&app_menu).unwrap();
    }

    menu.append_items(&[&file_menu, &emu_menu, &state_menu, &filter_menu, &help_menu])
        .unwrap();

    (menu, filter_items)
}

// ── Application ──────────────────────────────────────────────────────────────

struct App {
    rom_path: Option<PathBuf>,
    cli: Cli,
    emu: Option<Emulator>,
    model: GbModel,
    window: Option<Arc<Window>>,
    gpu: Option<GpuRenderer>,
    _menu: Option<Menu>,
    filter_items: Vec<(CheckMenuItem, scaling::ScaleFilter)>,
    audio_ring: Arc<Mutex<AudioRing>>,
    _audio_stream: Option<cpal::Stream>,
    camera_thread: Option<CameraThread>,
    camera_buf: [u8; 128 * 112],
    scale_filter: scaling::ScaleFilter,
    vec_cache: Option<vectorize::VectorizeCache>,
    frame_start: Instant,
    frame_dur: Duration,
    paused: bool,
    current_slot: usize,
    src_w: u32,
    src_h: u32,
    fps_timer: Instant,
    fps_count: u32,
    fps_emu_total: Duration,
    gilrs: Option<Gilrs>,
    active_gamepad: Option<GamepadId>,
    kb_buttons: u8,  // bitmask of keyboard-pressed buttons
    gp_buttons: u8,  // bitmask of gamepad-pressed buttons
}

impl App {
    fn new(cli: Cli) -> Self {
        let model = cli.model.unwrap_or(GbModel::Cgb);

        let audio_ring = Arc::new(Mutex::new(AudioRing::new(AUDIO_SAMPLE_RATE as usize * 2 /* channels */)));
        let stream = start_audio(Arc::clone(&audio_ring));

        App {
            rom_path: cli.rom.clone(),
            cli,
            emu: None,
            model,
            window: None,
            gpu: None,
            _menu: None,
            filter_items: Vec::new(),
            audio_ring,
            _audio_stream: stream,
            camera_thread: None,
            camera_buf: [0u8; 128 * 112],
            scale_filter: scaling::ScaleFilter::Nearest,
            vec_cache: None,
            frame_start: Instant::now(),
            frame_dur: frame_duration(model),
            paused: false,
            current_slot: 0,
            src_w: GB_W,
            src_h: GB_H,
            fps_timer: Instant::now(),
            fps_count: 0,
            fps_emu_total: Duration::ZERO,
            gilrs: Gilrs::new().ok(),
            active_gamepad: None,
            kb_buttons: 0,
            gp_buttons: 0,
        }
    }

    fn load_rom(&mut self, path: &PathBuf) {
        let rom = fs::read(path).unwrap_or_else(|e| {
            eprintln!("Failed to read ROM '{}': {}", path.display(), e);
            Vec::new()
        });
        if rom.is_empty() {
            return;
        }

        let boot_rom: Option<Vec<u8>> = if self.cli.no_boot {
            None
        } else if let Some(ref p) = self.cli.bootrom {
            fs::read(p).ok()
        } else {
            let path = match self.model {
                GbModel::Dmg0 => "bootroms/dmg0_boot.bin",
                GbModel::Dmg => "bootroms/dmg_boot.bin",
                GbModel::Mgb => "bootroms/mgb_boot.bin",
                GbModel::Sgb => "bootroms/sgb_boot.bin",
                GbModel::Sgb2 => "bootroms/sgb2_boot.bin",
                GbModel::Cgb0 => "bootroms/cgb0_boot.bin",
                GbModel::Cgb => "bootroms/cgb_boot.bin",
                GbModel::Agb => "bootroms/cgb_agb_boot.bin",
            };
            fs::read(path).ok()
        };

        let emu = Emulator::new(rom, boot_rom, Some(path.as_path()), self.model, None);
        let is_sgb = emu.is_sgb();
        self.src_w = if is_sgb { SGB_W } else { GB_W };
        self.src_h = if is_sgb { SGB_H } else { GB_H };

        // Start camera thread if cart has camera (Pocket Camera)
        if emu.bus.cart.has_camera() && self.camera_thread.is_none() {
            self.camera_thread = CameraThread::start();
        }

        self.emu = Some(emu);
        self.rom_path = Some(path.clone());
        self.frame_dur = frame_duration(self.model);

        if let Some(ref window) = self.window {
            let size = LogicalSize::new(self.src_w * SCALE, self.src_h * SCALE);
            let _ = window.request_inner_size(size);
            window.set_title(&format!(
                "VibeBoy — {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
    }

    fn update_filter_checkmarks(&self) {
        for (item, filter) in &self.filter_items {
            item.set_checked(*filter == self.scale_filter);
        }
    }

    fn handle_menu_event(&mut self, id: &str) {
        match id {
            ID_OPEN => {
                let file = rfd::FileDialog::new()
                    .add_filter("Game Boy ROMs", &["gb", "gbc"])
                    .add_filter("All files", &["*"])
                    .pick_file();
                if let Some(path) = file {
                    self.load_rom(&path);
                }
            }
            ID_QUIT => {
                if let Some(ref emu) = self.emu {
                    emu.save();
                }
                std::process::exit(0);
            }
            ID_PAUSE => {
                self.paused = !self.paused;
                eprintln!("{}", if self.paused { "Paused" } else { "Resumed" });
            }
            ID_RESET => {
                if let Some(path) = self.rom_path.clone() {
                    self.load_rom(&path);
                }
            }
            other => {
                // Check filter menu items
                if let Some(filter) = filter_id_to_filter(other) {
                    self.scale_filter = filter;
                    self.update_filter_checkmarks();
                    match filter {
                        scaling::ScaleFilter::Vectorize => {
                            self.vec_cache = Some(vectorize::VectorizeCache::new(false));
                        }
                        scaling::ScaleFilter::VectorizeAdaptive => {
                            self.vec_cache = Some(vectorize::VectorizeCache::new(true));
                        }
                        _ => {}
                    }
                    eprintln!("Filter: {:?}", filter);
                    return;
                }

                for i in 1..=9 {
                    if other == slot_save_id(i) {
                        if let Some(ref mut emu) = self.emu {
                            emu.save_state(i - 1);
                            eprintln!("State saved to slot {}", i);
                        }
                        return;
                    }
                    if other == slot_load_id(i) {
                        if let Some(ref mut emu) = self.emu {
                            if emu.load_state(i - 1) {
                                eprintln!("State loaded from slot {}", i);
                            } else {
                                eprintln!("Slot {} is empty", i);
                            }
                        }
                        return;
                    }
                }
            }
        }
    }

    fn step_and_render(&mut self) {
        if self.paused {
            return;
        }

        let emu = match self.emu.as_mut() {
            Some(e) => e,
            None => return,
        };

        // Feed webcam frames to Pocket Camera
        if let Some(ref ct) = self.camera_thread {
            if ct.read_frame(&mut self.camera_buf) {
                emu.bus.cart.set_camera_image(&self.camera_buf);
            }
        }

        emu.step_frame();

        // Push audio samples directly to ring buffer (96kHz stereo, matching APU output)
        let samples = emu.bus.apu.drain_samples();
        if !samples.is_empty() {
            let mut ring = self.audio_ring.lock().unwrap();
            ring.push(&samples);
        }

        // Render via wgpu
        let gpu = match self.gpu.as_mut() {
            Some(g) => g,
            None => return,
        };
        let window = match self.window.as_ref() {
            Some(w) => w,
            None => return,
        };

        let is_sgb = emu.is_sgb();
        let fb: &[u32] = if is_sgb {
            emu.sgb_composited_frame()
        } else {
            emu.frame_buffer()
        };
        let sw = self.src_w as usize;
        let sh = self.src_h as usize;

        let phys = window.inner_size();
        let win_w = phys.width as usize;
        let win_h = phys.height as usize;
        if win_w == 0 || win_h == 0 {
            return;
        }

        gpu.resize(win_w as u32, win_h as u32);

        // Apply scaling filter
        let disp_w = win_w;
        let disp_h = win_h;

        let scaled;
        let (frame_pixels, frame_w, frame_h): (&[u32], usize, usize) = match self.scale_filter {
            scaling::ScaleFilter::Nearest => {
                (fb, sw, sh)
            }
            scaling::ScaleFilter::Bilinear => {
                scaled = scaling::bilinear::scale_to(fb, sw, sh, disp_w, disp_h);
                (&scaled, disp_w, disp_h)
            }
            scaling::ScaleFilter::Bicubic => {
                scaled = scaling::bicubic::scale_to(fb, sw, sh, disp_w, disp_h);
                (&scaled, disp_w, disp_h)
            }
            scaling::ScaleFilter::Epx | scaling::ScaleFilter::Scale2x => {
                scaled = scaling::epx::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Scale3x => {
                scaled = scaling::scale3x::scale(fb, sw, sh);
                (&scaled, sw * 3, sh * 3)
            }
            scaling::ScaleFilter::Scale4x => {
                scaled = scaling::epx::scale4x(fb, sw, sh);
                (&scaled, sw * 4, sh * 4)
            }
            scaling::ScaleFilter::Eagle => {
                scaled = scaling::eagle::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Sai2x => {
                scaled = scaling::sai::scale_2xsai(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Super2xSai => {
                scaled = scaling::sai::scale_super2xsai(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::SuperEagle => {
                scaled = scaling::sai::scale_super_eagle(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Hqx(mode) => {
                let f = mode.factor() as usize;
                scaled = scaling::hqx::scale(fb, sw, sh, mode);
                (&scaled, sw * f, sh * f)
            }
            scaling::ScaleFilter::Xbr(mode) => {
                let f = mode.factor() as usize;
                scaled = scaling::xbr::scale(fb, sw, sh, mode);
                (&scaled, sw * f, sh * f)
            }
            scaling::ScaleFilter::Xbrz(mode) => {
                let f = mode.factor() as usize;
                scaled = scaling::xbrz::scale(fb, sw, sh, mode);
                (&scaled, sw * f, sh * f)
            }
            scaling::ScaleFilter::XbrHybrid => {
                scaled = scaling::xbr_hybrid::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::SuperXbr => {
                scaled = scaling::super_xbr::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Nedi => {
                scaled = scaling::nedi::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Dcci => {
                scaled = scaling::dcci::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::Edi => {
                scaled = scaling::edi::scale(fb, sw, sh);
                (&scaled, sw * 2, sh * 2)
            }
            scaling::ScaleFilter::OmniScale => {
                scaled = scaling::omniscale::scale_to(fb, sw, sh, disp_w, disp_h);
                (&scaled, disp_w, disp_h)
            }
            scaling::ScaleFilter::OmniScaleLegacy => {
                scaled = scaling::omniscale_legacy::scale_to(fb, sw, sh, disp_w, disp_h);
                (&scaled, disp_w, disp_h)
            }
            scaling::ScaleFilter::AaNearestNeighbor => {
                scaled = scaling::aa_nearest::scale(fb, sw, sh, disp_w, disp_h);
                (&scaled, disp_w, disp_h)
            }
            scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive => {
                let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                let adaptive = matches!(self.scale_filter, scaling::ScaleFilter::VectorizeAdaptive);
                let cache = self.vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new(adaptive));
                let (raster, vw, vh) = cache.rasterize(fb, sw, sh, scale);
                (raster, vw, vh)
            }
        };

        gpu.render(frame_pixels, frame_w as u32, frame_h as u32, self.src_w, self.src_h);

        // FPS counter
        let emu_time = self.frame_start.elapsed();
        self.fps_count += 1;
        self.fps_emu_total += emu_time;
        let elapsed = self.fps_timer.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = self.fps_count as f64 / elapsed.as_secs_f64();
            let avg_emu_ms = self.fps_emu_total.as_secs_f64() * 1000.0 / self.fps_count as f64;
            eprintln!("FPS: {:.1}  emu: {:.2}ms/frame", fps, avg_emu_ms);
            self.fps_count = 0;
            self.fps_emu_total = Duration::ZERO;
            self.fps_timer = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let win_w = self.src_w * SCALE;
        let win_h = self.src_h * SCALE;

        let attrs = Window::default_attributes()
            .with_title("VibeBoy")
            .with_inner_size(LogicalSize::new(win_w, win_h))
            .with_min_inner_size(LogicalSize::new(GB_W, GB_H))
            .with_resizable(true);

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        // Set up menu bar
        let (menu, filter_items) = build_menu();
        #[cfg(target_os = "macos")]
        {
            menu.init_for_nsapp();
        }
        #[cfg(target_os = "windows")]
        {
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::Win32(h) = handle.as_raw() {
                    let _ = menu.init_for_hwnd(h.hwnd.get() as _);
                }
            }
        }
        #[cfg(target_os = "linux")]
        {
            // On Linux with GTK, muda needs gtk initialization
            // For now, skip — muda will use gtk_application_set_menubar if available
        }

        let gpu = GpuRenderer::new(window.clone());

        self._menu = Some(menu);
        self.filter_items = filter_items;
        self.window = Some(window);
        self.gpu = Some(gpu);

        // Load ROM if provided on command line, otherwise show file dialog
        if let Some(path) = self.rom_path.clone() {
            self.load_rom(&path);
        } else {
            let file = rfd::FileDialog::new()
                .add_filter("Game Boy ROMs", &["gb", "gbc"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(path) = file {
                self.load_rom(&path);
            }
        }

        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ref emu) = self.emu {
                    emu.save();
                }
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    {
                        let btn = match key {
                            KeyCode::KeyZ => Some(Emulator::BTN_B),
                            KeyCode::KeyX => Some(Emulator::BTN_A),
                            KeyCode::Enter => Some(Emulator::BTN_START),
                            KeyCode::ShiftRight => Some(Emulator::BTN_SELECT),
                            KeyCode::ArrowRight => Some(Emulator::BTN_RIGHT),
                            KeyCode::ArrowLeft => Some(Emulator::BTN_LEFT),
                            KeyCode::ArrowUp => Some(Emulator::BTN_UP),
                            KeyCode::ArrowDown => Some(Emulator::BTN_DOWN),
                            _ => None,
                        };
                        if let Some(b) = btn {
                            if pressed {
                                self.kb_buttons |= b;
                            } else {
                                self.kb_buttons &= !b;
                            }
                        }
                    }

                    if let Some(ref mut emu) = self.emu {
                        // Apply combined keyboard + gamepad state
                        let combined = self.kb_buttons | self.gp_buttons;
                        let all_btns: &[u8] = &[
                            Emulator::BTN_RIGHT, Emulator::BTN_LEFT, Emulator::BTN_UP, Emulator::BTN_DOWN,
                            Emulator::BTN_A, Emulator::BTN_B, Emulator::BTN_SELECT, Emulator::BTN_START,
                        ];
                        for &b in all_btns {
                            emu.set_button(b, combined & b != 0);
                        }

                        if key == KeyCode::Backspace {
                            emu.rewinding = pressed;
                        }
                    }

                    if pressed {
                        match key {
                            KeyCode::Escape => {
                                if let Some(ref emu) = self.emu {
                                    emu.save();
                                }
                                event_loop.exit();
                            }
                            KeyCode::F5 => {
                                if let Some(ref mut emu) = self.emu {
                                    emu.save_state(self.current_slot);
                                    eprintln!(
                                        "State saved to slot {}",
                                        self.current_slot + 1
                                    );
                                }
                            }
                            KeyCode::F7 => {
                                if let Some(ref mut emu) = self.emu {
                                    if emu.load_state(self.current_slot) {
                                        eprintln!(
                                            "State loaded from slot {}",
                                            self.current_slot + 1
                                        );
                                    } else {
                                        eprintln!("Slot {} is empty", self.current_slot + 1);
                                    }
                                }
                            }
                            KeyCode::Digit1 => self.current_slot = 0,
                            KeyCode::Digit2 => self.current_slot = 1,
                            KeyCode::Digit3 => self.current_slot = 2,
                            KeyCode::Digit4 => self.current_slot = 3,
                            KeyCode::Digit5 => self.current_slot = 4,
                            KeyCode::Digit6 => self.current_slot = 5,
                            KeyCode::Digit7 => self.current_slot = 6,
                            KeyCode::Digit8 => self.current_slot = 7,
                            KeyCode::Digit9 => self.current_slot = 8,
                            _ => {}
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Process menu events
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            self.handle_menu_event(event.id().0.as_str());
        }

        // ── Gamepad polling ─────────────────────────────────────────────
        if let Some(ref mut gilrs) = self.gilrs {
            while let Some(ev) = gilrs.next_event() {
                match ev.event {
                    gilrs::EventType::Connected => {
                        if self.active_gamepad.is_none() {
                            self.active_gamepad = Some(ev.id);
                            let gp = gilrs.gamepad(ev.id);
                            eprintln!("Gamepad connected: {}", gp.name());
                        }
                    }
                    gilrs::EventType::Disconnected => {
                        if self.active_gamepad == Some(ev.id) {
                            eprintln!("Gamepad disconnected");
                            self.active_gamepad = None;
                        }
                    }
                    _ => {}
                }
            }

            // Find first connected gamepad if none active
            if self.active_gamepad.is_none() {
                for (id, gp) in gilrs.gamepads() {
                    if gp.is_connected() {
                        self.active_gamepad = Some(id);
                        eprintln!("Gamepad connected: {}", gp.name());
                        break;
                    }
                }
            }

            if let Some(gp_id) = self.active_gamepad {
                let gp = gilrs.gamepad(gp_id);
                const DEADZONE: f32 = 0.3;

                let gp_map: &[(GilButton, u8)] = &[
                    (GilButton::East,      Emulator::BTN_A),
                    (GilButton::South,     Emulator::BTN_B),
                    (GilButton::Start,     Emulator::BTN_START),
                    (GilButton::Select,    Emulator::BTN_SELECT),
                    (GilButton::DPadUp,    Emulator::BTN_UP),
                    (GilButton::DPadDown,  Emulator::BTN_DOWN),
                    (GilButton::DPadLeft,  Emulator::BTN_LEFT),
                    (GilButton::DPadRight, Emulator::BTN_RIGHT),
                ];

                // Left stick
                let lx = gp.axis_data(GilAxis::LeftStickX).map_or(0.0, |a| a.value());
                let ly = gp.axis_data(GilAxis::LeftStickY).map_or(0.0, |a| a.value());

                let mut gp_bits: u8 = 0;
                for &(gb, btn) in gp_map {
                    let pressed = gp.is_pressed(gb);
                    let stick = match btn {
                        b if b == Emulator::BTN_RIGHT => lx > DEADZONE,
                        b if b == Emulator::BTN_LEFT  => lx < -DEADZONE,
                        b if b == Emulator::BTN_UP    => ly > DEADZONE,
                        b if b == Emulator::BTN_DOWN  => ly < -DEADZONE,
                        _ => false,
                    };
                    if pressed || stick {
                        gp_bits |= btn;
                    }
                }
                self.gp_buttons = gp_bits;

                // Apply combined state
                if let Some(ref mut emu) = self.emu {
                    let combined = self.kb_buttons | self.gp_buttons;
                    let all_btns: &[u8] = &[
                        Emulator::BTN_RIGHT, Emulator::BTN_LEFT, Emulator::BTN_UP, Emulator::BTN_DOWN,
                        Emulator::BTN_A, Emulator::BTN_B, Emulator::BTN_SELECT, Emulator::BTN_START,
                    ];
                    for &b in all_btns {
                        emu.set_button(b, combined & b != 0);
                    }

                    // Shoulders for rewind
                    if gp.is_pressed(GilButton::LeftTrigger) {
                        emu.rewinding = true;
                    }
                }
            } else {
                self.gp_buttons = 0;
            }
        }

        // Handle rewind
        if let Some(ref mut emu) = self.emu {
            if emu.rewinding {
                emu.rewind_one_frame();
                emu.bus.apu.drain_samples();
            }
        }

        // Step emulation
        self.step_and_render();

        // Frame rate cap
        let remaining = self.frame_dur.saturating_sub(self.frame_start.elapsed());
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(2));
        }
        while self.frame_start.elapsed() < self.frame_dur {
            std::hint::spin_loop();
        }
        self.frame_start = Instant::now();

        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let mut app = App::new(cli);

    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
