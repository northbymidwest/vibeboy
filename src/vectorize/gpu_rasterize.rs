//! GPU-accelerated rasterizer using wgpu compute shaders.
//!
//! Parallelizes the scanline rasterizer across all output pixels.
//! Falls back gracefully if no GPU is available.

use super::contour::ColorPath;
use super::rasterize::prepare_gpu_edges;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    out_w: u32,
    out_h: u32,
    num_paths: u32,
    bg_color: u32,
}

pub struct GpuRasterizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuRasterizer {
    /// Create a new GPU rasterizer. Returns None if no suitable GPU is available.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("GPU rasterizer: no suitable adapter found: {e}");
                return None;
            }
        };

        let (device, queue): (wgpu::Device, wgpu::Queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vectorize-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        )) {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("GPU rasterizer: device request failed: {e}");
                return None;
            }
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rasterize.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rasterize.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rasterize-bgl"),
            entries: &[
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // edges
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // paths
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rasterize-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rasterize-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self { device, queue, pipeline, bind_group_layout })
    }

    /// Rasterize vector paths on the GPU at the given scale.
    /// Returns (pixel_buffer, output_width, output_height).
    pub fn rasterize(
        &self,
        paths: &[ColorPath],
        width: usize,
        height: usize,
        bg_color: u32,
        scale: f64,
    ) -> (Vec<u32>, usize, usize) {
        let out_w = (width as f64 * scale).round() as usize;
        let out_h = (height as f64 * scale).round() as usize;
        let num_pixels = out_w * out_h;

        let (gpu_edges, gpu_paths) = prepare_gpu_edges(paths, bg_color, scale);

        // Empty scene — just return bg fill
        if gpu_paths.is_empty() {
            return (vec![bg_color; num_pixels], out_w, out_h);
        }

        let uniforms = Uniforms {
            out_w: out_w as u32,
            out_h: out_h as u32,
            num_paths: gpu_paths.len() as u32,
            bg_color,
        };

        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let edge_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edges"),
            contents: bytemuck::cast_slice(&gpu_edges),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let path_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paths"),
            contents: bytemuck::cast_slice(&gpu_paths),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (num_pixels * 4) as u64;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rasterize-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: edge_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: path_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rasterize-enc"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (out_w as u32 + 255) / 256;
            let wg_y = out_h as u32;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        (result, out_w, out_h)
    }
}
