use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::emulator::Emulator;
use crate::model::GbModel;
use crate::web_printer::WebPrinter;
use crate::wgpu_vectorize::WgpuVectorizePipeline;
use crate::wgpu_scale::{WgpuScalePipeline, WgpuScaleFilter};

fn auto_detect_model(rom: &[u8]) -> GbModel {
    if rom.len() > 0x143 && (rom[0x143] & 0x80) != 0 {
        GbModel::Cgb
    } else {
        GbModel::Dmg
    }
}

const BLIT_SHADER: &str = r#"
struct VsOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOutput {
    let x = f32(vi & 1u);
    let y = f32((vi >> 1u) & 1u);
    var out: VsOutput;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, tex_sampler, in.uv);
}
"#;

/// Active rendering filter for the web frontend.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WebFilter {
    /// Nearest-neighbor (direct Canvas2D blit, no GPU)
    Nearest,
    /// Kopf-Lischinski vectorize pipeline
    Vectorize,
    /// GPU compute scaling filter
    Scale(WgpuScaleFilter),
}

#[wasm_bindgen]
pub struct WasmEmulator {
    emu: Emulator,
    rgba_buf: Vec<u8>,
    audio_buf: Vec<f32>,
    last_print_w: u32,
    last_print_h: u32,
    save_key: String,
    filter: WebFilter,
    // GPU state (initialized async after construction)
    gpu: Option<GpuState>,
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vectorize: WgpuVectorizePipeline,
    scale: WgpuScalePipeline,
}

#[wasm_bindgen]
impl WasmEmulator {
    /// Create a new emulator from ROM bytes. Call init_gpu() after to enable WebGPU rendering.
    #[wasm_bindgen(constructor)]
    pub fn new(rom: &[u8]) -> Result<WasmEmulator, JsValue> {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Warn).ok();

        if rom.len() < 0x150 {
            return Err(JsValue::from_str("ROM too small"));
        }

        let rom_vec = rom.to_vec();
        let model = auto_detect_model(&rom_vec);

        // Derive save key from ROM title (0x134..0x143) + global checksum (0x14E..0x14F)
        let title: String = rom_vec[0x134..0x143].iter()
            .take_while(|&&b| b != 0)
            .map(|&b| if b.is_ascii_alphanumeric() || b == b' ' { b as char } else { '_' })
            .collect();
        let checksum = ((rom_vec[0x14E] as u16) << 8) | rom_vec[0x14F] as u16;
        let save_key = format!("vibeboy_sav_{}_{:04X}", title.trim(), checksum);

        let emu = Emulator::new(rom_vec, None, None, model, None);
        let w = if emu.is_sgb() { 256 } else { 160 };
        let h = if emu.is_sgb() { 224 } else { 144 };

        Ok(WasmEmulator {
            emu,
            rgba_buf: vec![0u8; w * h * 4],
            audio_buf: Vec::new(),
            last_print_w: 0,
            last_print_h: 0,
            save_key,
            filter: WebFilter::Vectorize,
            gpu: None,
        })
    }

    /// Initialize WebGPU rendering on the given canvas element.
    pub async fn init_gpu(&mut self, canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
        let surface = instance.create_surface(surface_target)
            .map_err(|e| JsValue::from_str(&format!("Surface error: {e}")))?;

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }).await.ok_or_else(|| JsValue::from_str("No WebGPU adapter"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        ).await.map_err(|e| JsValue::from_str(&format!("Device error: {e}")))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let w = canvas.width();
        let h = canvas.height();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w.max(1),
            height: h.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Blit pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let vectorize = WgpuVectorizePipeline::new(&device);
        let scale = WgpuScalePipeline::new(&device);

        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            surface_config,
            blit_pipeline,
            blit_bind_group_layout,
            sampler,
            vectorize,
            scale,
        });

        log::info!("WebGPU initialized");
        Ok(())
    }

    /// Resize the GPU surface (call when canvas size changes).
    pub fn resize_gpu(&mut self, width: u32, height: u32) {
        if let Some(gpu) = &mut self.gpu {
            gpu.surface_config.width = width.max(1);
            gpu.surface_config.height = height.max(1);
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
        }
    }

    /// Step one frame of emulation.
    pub fn step_frame(&mut self) {
        self.emu.step_frame();
    }

    /// Enable or disable audio sample generation. Disabling saves CPU.
    pub fn set_audio_enabled(&mut self, enabled: bool) {
        self.emu.bus.apu.headless = !enabled;
    }

    /// Attach a Game Boy Printer to the serial port.
    pub fn attach_printer(&mut self) {
        let clock_rate = if self.emu.bus.double_speed { 8_388_608 } else { 4_194_304 };
        self.emu.bus.serial.device = Box::new(WebPrinter::new(clock_rate));
    }

    /// Check if the printer has a completed print ready for download.
    pub fn has_print(&self) -> bool {
        self.emu.bus.serial.device.as_any()
            .downcast_ref::<WebPrinter>()
            .is_some_and(|p| p.has_pending_print())
    }

    /// Take the next print as RGBA pixel data. Also stores width/height
    /// for retrieval via print_width()/print_height().
    pub fn take_print_rgba(&mut self) -> Vec<u8> {
        if let Some(printer) = self.emu.bus.serial.device.as_any_mut().downcast_mut::<WebPrinter>() {
            if let Some((rgba, w, h)) = printer.take_print() {
                self.last_print_w = w;
                self.last_print_h = h;
                return rgba;
            }
        }
        Vec::new()
    }

    pub fn print_width(&self) -> u32 { self.last_print_w }
    pub fn print_height(&self) -> u32 { self.last_print_h }

    /// Returns true if the loaded ROM has a camera sensor (Pocket Camera).
    pub fn has_camera(&self) -> bool {
        self.emu.bus.cart.has_camera()
    }

    /// Returns true if the cartridge has an accelerometer (MBC7 / Kirby Tilt 'n' Tumble).
    pub fn has_accelerometer(&self) -> bool {
        self.emu.bus.cart.has_accelerometer()
    }

    /// Feed accelerometer data. gx/gy are in g-force units (±1.0 = ±1g).
    pub fn set_accelerometer(&mut self, gx: f32, gy: f32) {
        const CENTER: f32 = 0x81D0 as f32;
        const RANGE: f32 = 0x70 as f32;
        let mbc7_x = (CENTER + gx * RANGE).clamp(0.0, 65535.0) as u16;
        let mbc7_y = (CENTER + gy * RANGE).clamp(0.0, 65535.0) as u16;
        self.emu.bus.cart.set_accelerometer(mbc7_x, mbc7_y);
    }

    /// Returns true if the cartridge has battery-backed save RAM.
    pub fn has_battery(&self) -> bool {
        self.emu.bus.cart.has_battery()
    }

    /// Get cartridge save RAM as bytes (for persisting to localStorage).
    pub fn save_data(&self) -> Vec<u8> {
        self.emu.bus.cart.save_data()
    }

    /// Load save RAM from bytes (from localStorage).
    pub fn load_save(&mut self, data: &[u8]) {
        self.emu.bus.cart.load_ram(data);
    }

    /// Get a save key derived from the ROM title + checksum.
    pub fn save_key(&self) -> String {
        self.save_key.clone()
    }

    /// Feed a 128x112 grayscale image from a webcam into the camera sensor.
    pub fn set_camera_image(&mut self, grayscale: &[u8]) {
        if grayscale.len() >= 128 * 112 {
            let mut img = [0u8; 128 * 112];
            img.copy_from_slice(&grayscale[..128 * 112]);
            self.emu.bus.cart.set_camera_image(&img);
        }
    }

    /// Set the active scaling filter by name.
    /// Valid names: "nearest", "vectorize", "epx", "eagle", "scale3x",
    /// "bicubic", "aa-nearest", "omniscale".
    /// Returns true if the filter was recognized.
    pub fn set_filter(&mut self, name: &str) -> bool {
        match name {
            "nearest" => { self.filter = WebFilter::Nearest; true }
            "vectorize" => { self.filter = WebFilter::Vectorize; true }
            _ => {
                if let Some(f) = WgpuScaleFilter::from_name(name) {
                    self.filter = WebFilter::Scale(f);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Get the name of the current filter.
    pub fn current_filter(&self) -> String {
        match self.filter {
            WebFilter::Nearest => "nearest".to_string(),
            WebFilter::Vectorize => "vectorize".to_string(),
            WebFilter::Scale(f) => format!("{:?}", f).to_lowercase(),
        }
    }

    /// Render the current frame via WebGPU.
    /// Returns true if GPU rendering was used, false if fallback needed.
    pub fn render_gpu(&mut self) -> bool {
        if self.filter == WebFilter::Nearest {
            return false; // caller should use Canvas2D path
        }

        let gpu = match &mut self.gpu {
            Some(g) => g,
            None => return false,
        };

        let is_sgb = self.emu.is_sgb();
        let src_w: u32 = if is_sgb { 256 } else { 160 };
        let src_h: u32 = if is_sgb { 224 } else { 144 };

        // Copy frame buffer to avoid borrow conflicts
        let fb: Vec<u32> = if is_sgb {
            self.emu.sgb_composited_frame().to_vec()
        } else {
            self.emu.frame_buffer().to_vec()
        };

        let out_w = gpu.surface_config.width;
        let out_h = gpu.surface_config.height;
        let scale = (out_w as f64 / src_w as f64).min(out_h as f64 / src_h as f64) as f32;

        let render_w = (src_w as f32 * scale).round() as u32;
        let render_h = (src_h as f32 * scale).round() as u32;

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("filter+blit"),
        });

        // Dispatch the appropriate pipeline
        let out_tex = match self.filter {
            WebFilter::Vectorize => {
                gpu.vectorize.encode(
                    &gpu.device, &gpu.queue, &mut encoder,
                    &fb, src_w, src_h, render_w, render_h, scale,
                )
            }
            WebFilter::Scale(filter) => {
                gpu.scale.encode(
                    &gpu.device, &gpu.queue, &mut encoder,
                    filter, &fb, src_w, src_h, render_w, render_h,
                )
            }
            WebFilter::Nearest => unreachable!(),
        };

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return false,
        };
        let fb_view = frame.texture.create_view(&Default::default());
        let tex_view = out_tex.create_view(&Default::default());

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &gpu.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&gpu.sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &fb_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&gpu.blit_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        true
    }

    /// Get a pointer to the RGBA frame buffer in wasm memory.
    /// Call frame_buffer_update() first to refresh, then use
    /// frame_buffer_ptr() + frame_buffer_len() to read directly.
    pub fn frame_buffer_update(&mut self) {
        let fb = if self.emu.is_sgb() {
            self.emu.sgb_composited_frame()
        } else {
            self.emu.frame_buffer()
        };

        let needed = fb.len() * 4;
        if self.rgba_buf.len() != needed {
            self.rgba_buf.resize(needed, 0);
        }

        for (i, &px) in fb.iter().enumerate() {
            let off = i * 4;
            self.rgba_buf[off]     = ((px >> 16) & 0xFF) as u8;
            self.rgba_buf[off + 1] = ((px >> 8) & 0xFF) as u8;
            self.rgba_buf[off + 2] = (px & 0xFF) as u8;
            self.rgba_buf[off + 3] = 0xFF;
        }
    }

    pub fn frame_buffer_ptr(&self) -> *const u8 {
        self.rgba_buf.as_ptr()
    }

    pub fn frame_buffer_len(&self) -> usize {
        self.rgba_buf.len()
    }

    pub fn width(&self) -> u32 {
        if self.emu.is_sgb() { 256 } else { 160 }
    }

    pub fn height(&self) -> u32 {
        if self.emu.is_sgb() { 224 } else { 144 }
    }

    pub fn set_button(&mut self, button: u8, pressed: bool) {
        self.emu.set_button(button, pressed);
    }

    /// Drain audio samples into internal buffer; read via audio_ptr/audio_len.
    pub fn audio_drain(&mut self) {
        self.audio_buf = self.emu.bus.apu.drain_samples();
    }

    pub fn audio_ptr(&self) -> *const f32 {
        self.audio_buf.as_ptr()
    }

    pub fn audio_len(&self) -> usize {
        self.audio_buf.len()
    }
}

/// Get the wasm linear memory for zero-copy buffer access from JS.
#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub fn btn_right() -> u8 { Emulator::BTN_RIGHT }
#[wasm_bindgen]
pub fn btn_left() -> u8 { Emulator::BTN_LEFT }
#[wasm_bindgen]
pub fn btn_up() -> u8 { Emulator::BTN_UP }
#[wasm_bindgen]
pub fn btn_down() -> u8 { Emulator::BTN_DOWN }
#[wasm_bindgen]
pub fn btn_a() -> u8 { Emulator::BTN_A }
#[wasm_bindgen]
pub fn btn_b() -> u8 { Emulator::BTN_B }
#[wasm_bindgen]
pub fn btn_select() -> u8 { Emulator::BTN_SELECT }
#[wasm_bindgen]
pub fn btn_start() -> u8 { Emulator::BTN_START }
