//! Full GPU vectorize pipeline using wgpu (WebGPU-compatible).
//!
//! Six-stage pipeline matching the SDL3 GPU version:
//! 1. similarity_graph → 2. resolve_crossings → 3. cell_graph →
//! 4. optimize_energy → 5. update_tjunction → 6. cell_rasterizer
//!
//! All shaders are loaded from WGSL (cross-compiled from GLSL via naga).

use wgpu;

/// Cached wgpu compute pipelines and buffers for the vectorize pipeline.
pub struct WgpuVectorizePipeline {
    sim_graph: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    cell_graph: wgpu::ComputePipeline,
    optimizer: wgpu::ComputePipeline,
    tjunction: wgpu::ComputePipeline,
    rasterizer: wgpu::ComputePipeline,
    // Cached buffers (allocated once, reused each frame)
    bufs: Option<VecBufs>,
}

struct VecBufs {
    img_w: u32,
    img_h: u32,
    px_buf: wgpu::Buffer,
    graph_buf: wgpu::Buffer,
    graph_snapshot: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    nbr_buf: wgpu::Buffer,
    flag_buf: wgpu::Buffer,
    opt_out_buf: wgpu::Buffer,
    orig_pos_buf: wgpu::Buffer,
    // One uniform buffer per stage to avoid write conflicts
    uni_sim: wgpu::Buffer,
    uni_resolve: wgpu::Buffer,
    uni_cell: wgpu::Buffer,
    uni_opt: wgpu::Buffer,
    uni_tjunc: wgpu::Buffer,
    uni_rast: wgpu::Buffer,
    output_tex: wgpu::Texture,
    output_tex_w: u32,
    output_tex_h: u32,
    // Cached bind groups (recreated only when buffers change)
    bg_sim: [wgpu::BindGroup; 3],
    bg_resolve: [wgpu::BindGroup; 3],
    bg_cell: [wgpu::BindGroup; 3],
    bg_opt_p1: [wgpu::BindGroup; 3],
    bg_opt_p2: [wgpu::BindGroup; 3],
    bg_tjunc: [wgpu::BindGroup; 2],
    bg_rast: [wgpu::BindGroup; 3],
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    wgsl: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    // Safety: our compute shaders have no infinite loops and all buffer
    // accesses are in-bounds. Disabling runtime checks removes 65 per-access
    // bounds checks (metal::min) from the rasterizer's hot path.
    let module = unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl WgpuVectorizePipeline {
    /// Create all 6 compute pipelines from compiled WGSL shaders.
    pub fn new(device: &wgpu::Device) -> Self {
        let sim_wgsl = include_str!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.wgsl"));
        let resolve_wgsl = include_str!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.wgsl"));
        let cell_wgsl = include_str!(concat!(env!("OUT_DIR"), "/cell_graph_comp.wgsl"));
        let opt_wgsl = include_str!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.wgsl"));
        let tjunc_wgsl = include_str!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.wgsl"));
        let rast_wgsl = include_str!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.wgsl"));

        WgpuVectorizePipeline {
            sim_graph: create_compute_pipeline(device, sim_wgsl, "similarity_graph"),
            resolve: create_compute_pipeline(device, resolve_wgsl, "resolve_crossings"),
            cell_graph: create_compute_pipeline(device, cell_wgsl, "cell_graph"),
            optimizer: create_compute_pipeline(device, opt_wgsl, "optimize_energy"),
            tjunction: create_compute_pipeline(device, tjunc_wgsl, "update_tjunction"),
            rasterizer: create_compute_pipeline(device, rast_wgsl, "cell_rasterizer"),
            bufs: None,
        }
    }

    fn ensure_bufs(&mut self, device: &wgpu::Device, img_w: u32, img_h: u32, out_w: u32, out_h: u32) {
        if let Some(b) = &self.bufs {
            if b.img_w == img_w && b.img_h == img_h && b.output_tex_w == out_w && b.output_tex_h == out_h {
                return;
            }
        }
        let corners_w = img_w + 1;
        let corners_h = img_h + 1;
        let num_cps = corners_w * corners_h * 2;
        let graph_stride = 2 * img_w + 1;
        let graph_h = 2 * img_h + 1;

        let storage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let storage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let px_size = (img_w * img_h * 4) as u64;
        let graph_size = (graph_stride * graph_h * 4) as u64;
        let pos_size = (num_cps * 2 * 4) as u64;
        let nbr_size = (num_cps * 4 * 4) as u64;
        let flag_size = (num_cps * 4) as u64;
        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vectorize output"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let px_buf = mk("pixels", px_size, storage_ro);
        let graph_buf = mk("graph", graph_size, storage_rw);
        let graph_snapshot = mk("graph_snap", graph_size, storage_ro | wgpu::BufferUsages::COPY_SRC);
        let pos_buf = mk("positions", pos_size, storage_rw);
        let nbr_buf = mk("neighbors", nbr_size, storage_rw);
        let flag_buf = mk("flags", flag_size, storage_rw);
        let opt_out_buf = mk("opt_out", pos_size, storage_rw);
        let orig_pos_buf = mk("orig_pos", pos_size, storage_rw);
        let uni_sim = mk("uni_sim", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_resolve = mk("uni_resolve", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_cell = mk("uni_cell", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_opt = mk("uni_opt", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_tjunc = mk("uni_tjunc", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_rast = mk("uni_rast", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        let tex_view = output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Helper to create a bind group
        let bg = |pipeline: &wgpu::ComputePipeline, group: u32, entries: &[wgpu::BindGroupEntry]| -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(group),
                entries,
            })
        };

        // Pre-create all bind groups.
        // WGSL group mapping: group 0 = read storage (vk set 0),
        //   group 1 = RW storage/texture (vk set 1), group 2 = uniforms (vk set 2).
        // Exception: update_tjunction has uniforms in group 1 and RW+read in group 0.
        let bg_sim = [
            bg(&self.sim_graph, 0, &[wgpu::BindGroupEntry { binding: 0, resource: px_buf.as_entire_binding() }]),
            bg(&self.sim_graph, 1, &[wgpu::BindGroupEntry { binding: 0, resource: graph_buf.as_entire_binding() }]),
            bg(&self.sim_graph, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_sim.as_entire_binding() }]),
        ];
        let bg_resolve = [
            bg(&self.resolve, 0, &[wgpu::BindGroupEntry { binding: 0, resource: graph_snapshot.as_entire_binding() }]),
            bg(&self.resolve, 1, &[wgpu::BindGroupEntry { binding: 0, resource: graph_buf.as_entire_binding() }]),
            bg(&self.resolve, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_resolve.as_entire_binding() }]),
        ];
        let bg_cell = [
            bg(&self.cell_graph, 0, &[wgpu::BindGroupEntry { binding: 0, resource: graph_buf.as_entire_binding() }]),
            bg(&self.cell_graph, 1, &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.cell_graph, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_cell.as_entire_binding() }]),
        ];
        let bg_opt_p1 = [
            bg(&self.optimizer, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.optimizer, 1, &[wgpu::BindGroupEntry { binding: 0, resource: opt_out_buf.as_entire_binding() }]),
            bg(&self.optimizer, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_opt.as_entire_binding() }]),
        ];
        let bg_opt_p2 = [
            bg(&self.optimizer, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.optimizer, 1, &[wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() }]),
            bg(&self.optimizer, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_opt.as_entire_binding() }]),
        ];
        // update_tjunction: uniforms at vk set 1 (group 1), all buffers at vk set 0 (group 0)
        let bg_tjunc = [
            bg(&self.tjunction, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.tjunction, 1, &[wgpu::BindGroupEntry { binding: 0, resource: uni_tjunc.as_entire_binding() }]),
        ];
        let bg_rast = [
            bg(&self.rasterizer, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: px_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: nbr_buf.as_entire_binding() },
            ]),
            bg(&self.rasterizer, 1, &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex_view),
            }]),
            bg(&self.rasterizer, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_rast.as_entire_binding() }]),
        ];

        self.bufs = Some(VecBufs {
            img_w, img_h,
            px_buf, graph_buf, graph_snapshot, pos_buf, nbr_buf, flag_buf,
            opt_out_buf, orig_pos_buf,
            uni_sim, uni_resolve, uni_cell, uni_opt, uni_tjunc, uni_rast,
            output_tex, output_tex_w: out_w, output_tex_h: out_h,
            bg_sim, bg_resolve, bg_cell, bg_opt_p1, bg_opt_p2, bg_tjunc, bg_rast,
        });
    }

    /// Run the full vectorize pipeline. Returns the output texture for display.
    /// Encode all vectorize compute stages onto the given encoder.
    /// Does NOT submit — caller is responsible for submitting.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pixels: &[u32],
        img_w: u32, img_h: u32,
        out_w: u32, out_h: u32,
        scale: f32,
    ) -> &wgpu::Texture {
        self.ensure_bufs(device, img_w, img_h, out_w, out_h);
        let b = self.bufs.as_ref().unwrap();

        let corners_w = img_w + 1;
        let corners_h = img_h + 1;
        let num_cps = corners_w * corners_h * 2;
        let graph_stride = 2 * img_w + 1;

        // Upload pixel data
        let px_bytes = unsafe {
            std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
        };
        queue.write_buffer(&b.px_buf, 0, px_bytes);

        // Helper: write uniform data to a specific buffer
        let write_uniform = |queue: &wgpu::Queue, buf: &wgpu::Buffer, data: &[u32]| {
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            queue.write_buffer(buf, 0, bytes);
        };

        // Write all uniforms up front
        write_uniform(queue, &b.uni_sim, &[img_w, img_h, graph_stride, 0]);
        write_uniform(queue, &b.uni_resolve, &[img_w, img_h, graph_stride, 0]);
        write_uniform(queue, &b.uni_cell, &[img_w, img_h, graph_stride, corners_w]);
        let uni_opt: [u32; 4] = [num_cps, f32::to_bits(0.01), f32::to_bits(0.25), f32::to_bits(2.5)];
        write_uniform(queue, &b.uni_opt, &uni_opt);
        write_uniform(queue, &b.uni_tjunc, &[num_cps, 0, 0, 0]);
        let tiles_w = (img_w + 1) / 2;
        let tiles_h = (img_h + 1) / 2;
        let uni_rast: [u32; 8] = [
            img_w, img_h, out_w, out_h,
            f32::to_bits(scale), corners_w, tiles_w, tiles_h,
        ];
        write_uniform(queue, &b.uni_rast, &uni_rast);

        // Compute pass 1: similarity graph
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sim_graph"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.sim_graph);
            cp.set_bind_group(0, &b.bg_sim[0], &[]);
            cp.set_bind_group(1, &b.bg_sim[1], &[]);
            cp.set_bind_group(2, &b.bg_sim[2], &[]);
            cp.dispatch_workgroups((img_w + 15) / 16, (img_h + 15) / 16, 1);
        }

        // Buffer copy: graph → snapshot
        let graph_size = (graph_stride * (2 * img_h + 1) * 4) as u64;
        encoder.copy_buffer_to_buffer(&b.graph_buf, 0, &b.graph_snapshot, 0, graph_size);

        // Compute pass 2: resolve crossings + cell graph
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resolve+cell"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.resolve);
            cp.set_bind_group(0, &b.bg_resolve[0], &[]);
            cp.set_bind_group(1, &b.bg_resolve[1], &[]);
            cp.set_bind_group(2, &b.bg_resolve[2], &[]);
            cp.dispatch_workgroups((img_w.saturating_sub(1) + 15) / 16, (img_h.saturating_sub(1) + 15) / 16, 1);
            cp.set_pipeline(&self.cell_graph);
            cp.set_bind_group(0, &b.bg_cell[0], &[]);
            cp.set_bind_group(1, &b.bg_cell[1], &[]);
            cp.set_bind_group(2, &b.bg_cell[2], &[]);
            cp.dispatch_workgroups((corners_w + 15) / 16, (corners_h + 15) / 16, 1);
        }

        // Buffer copy: positions → orig_pos
        let pos_size = (num_cps * 2 * 4) as u64;
        encoder.copy_buffer_to_buffer(&b.pos_buf, 0, &b.orig_pos_buf, 0, pos_size);

        // Compute pass 3: optimizer + t-junction + rasterizer
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("opt+tjunc+rast"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.optimizer);
            cp.set_bind_group(0, &b.bg_opt_p1[0], &[]);
            cp.set_bind_group(1, &b.bg_opt_p1[1], &[]);
            cp.set_bind_group(2, &b.bg_opt_p1[2], &[]);
            cp.dispatch_workgroups((num_cps + 255) / 256, 1, 1);
            cp.set_bind_group(0, &b.bg_opt_p2[0], &[]);
            cp.set_bind_group(1, &b.bg_opt_p2[1], &[]);
            cp.set_bind_group(2, &b.bg_opt_p2[2], &[]);
            cp.dispatch_workgroups((num_cps + 255) / 256, 1, 1);
            cp.set_pipeline(&self.tjunction);
            cp.set_bind_group(0, &b.bg_tjunc[0], &[]);
            cp.set_bind_group(1, &b.bg_tjunc[1], &[]);
            cp.dispatch_workgroups((num_cps + 255) / 256, 1, 1);
            cp.set_pipeline(&self.rasterizer);
            cp.set_bind_group(0, &b.bg_rast[0], &[]);
            cp.set_bind_group(1, &b.bg_rast[1], &[]);
            cp.set_bind_group(2, &b.bg_rast[2], &[]);
            cp.dispatch_workgroups(tiles_w * tiles_h, 1, 1);
        }

        &self.bufs.as_ref().unwrap().output_tex
    }

    /// Download the output texture to CPU pixels (for screenshots/testing).
    pub fn download_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(Vec<u32>, u32, u32)> {
        let b = self.bufs.as_ref()?;
        let w = b.output_tex_w;
        let h = b.output_tex_h;
        let row_bytes = w * 4;
        let padded_row = (row_bytes + 255) & !255; // wgpu requires 256-byte row alignment

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: (padded_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &b.output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let mut result = vec![0u32; (w * h) as usize];
        for y in 0..h as usize {
            let src_offset = y * padded_row as usize;
            let dst_offset = y * w as usize;
            let row = &data[src_offset..src_offset + row_bytes as usize];
            for x in 0..w as usize {
                let r = row[x * 4] as u32;
                let g = row[x * 4 + 1] as u32;
                let b_val = row[x * 4 + 2] as u32;
                result[dst_offset + x] = (r << 16) | (g << 8) | b_val;
            }
        }
        drop(data);
        staging.unmap();

        Some((result, w, h))
    }
}

// ── Shared-chain rasterizer pipeline ──────────────────────────────────────

/// wgpu compute pipeline for the shared-chain vectorize rasterizer.
/// Dispatches `vectorize_raster.comp` (row-indexed edges with per-edge color).
///
/// Shader bind groups (matching GLSL descriptor sets):
///   group 0: binding 0 = edges, binding 1 = row_ranges, binding 2 = edge_indices (all storage)
///   group 1: binding 0 = output texture (storage image, rgba8unorm)
///   group 2: binding 0 = uniforms (uniform buffer)
pub struct WgpuSharedChainRasterizer {
    pipeline: wgpu::ComputePipeline,
    bgl_buffers: wgpu::BindGroupLayout,  // group 0: storage buffers
    bgl_texture: wgpu::BindGroupLayout,  // group 1: output texture
    bgl_uniform: wgpu::BindGroupLayout,  // group 2: uniforms
}

impl WgpuSharedChainRasterizer {
    pub fn new(device: &wgpu::Device) -> Self {
        let wgsl = include_str!(concat!(env!("OUT_DIR"), "/vectorize_raster_comp.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vectorize_raster"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Group 0: uniforms
        let bgl_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vr_bgl0"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 1: edges (b0), row_ranges (b1), edge_indices (b2)
        let bgl_buffers = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vr_bgl1"),
            entries: &[storage_entry(0), storage_entry(1), storage_entry(2)],
        });

        // Group 2: output texture
        let bgl_texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vr_bgl2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vr_pl"),
            bind_group_layouts: &[Some(&bgl_buffers), Some(&bgl_texture), Some(&bgl_uniform)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vectorize_raster"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl_buffers, bgl_texture, bgl_uniform }
    }

    /// Encode the shared-chain rasterize pass. Returns the output texture.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        edges: &[crate::vectorize::rasterize::GpuEdgeV2],
        row_ranges: &[crate::vectorize::rasterize::GpuRowRange],
        edge_indices: &[u32],
        out_w: u32,
        out_h: u32,
        bg_color: u32,
    ) -> wgpu::Texture {
        use wgpu::util::DeviceExt;

        let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vr_edges"),
            contents: bytemuck::cast_slice(edges),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let row_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vr_rows"),
            contents: bytemuck::cast_slice(row_ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let idx_data = if edge_indices.is_empty() { &[0u32][..] } else { edge_indices };
        let idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vr_indices"),
            contents: bytemuck::cast_slice(idx_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vr_output"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&Default::default());

        let uniforms = [out_w, out_h, edges.len() as u32, bg_color];
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vr_uniform"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Group 0: storage buffers (vk set 0)
        let bg_buffers = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_bg0"),
            layout: &self.bgl_buffers,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: edge_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: row_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: idx_buf.as_entire_binding() },
            ],
        });

        // Group 1: output texture (vk set 1)
        let bg_texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_bg1"),
            layout: &self.bgl_texture,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_view) },
            ],
        });

        // Group 2: uniforms (vk set 2)
        let bg_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_bg2"),
            layout: &self.bgl_uniform,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vectorize_raster"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_buffers, &[]);
            pass.set_bind_group(1, &bg_texture, &[]);
            pass.set_bind_group(2, &bg_uniform, &[]);
            pass.dispatch_workgroups(
                (out_w + 15) / 16,
                (out_h + 15) / 16,
                1,
            );
        }

        output_tex
    }
}

// ── Diffusion rasterizer pipeline ─────────────────────────────────────────

/// Gaussian diffusion rasterizer (single-pass `diffusion_raster.comp`).
///
/// Builds the similarity graph and region labels on CPU, then dispatches the
/// GPU shader for Gaussian-blended pixel output.
pub struct WgpuDiffusionRasterizer {
    pipeline: wgpu::ComputePipeline,
}

/// Gaussian blur kernel width for diffusion rasterizers.
const GAUSS_K: f32 = 2.5;
/// Gaussian blur radius for diffusion rasterizers.
const GAUSS_RADIUS: f32 = 2.0;

/// Compute the diagonal ownership state for a given corner (cx, cy) in the
/// similarity graph. Returns 0 = no diagonal, 1 = down-right, 2 = down-left.
fn corner_diag(g: &crate::vectorize::graph::SimilarityGraph, cx: usize, cy: usize) -> u8 {
    if cx == 0 || cy == 0 || cx >= g.width || cy >= g.height { return 0; }
    if g.edge(cx - 1, cy - 1).down_right { return 1; }
    if g.edge(cx, cy - 1).down_left { return 2; }
    0
}

impl WgpuDiffusionRasterizer {
    pub fn new(device: &wgpu::Device) -> Self {
        let wgsl = include_str!(concat!(env!("OUT_DIR"), "/diffusion_raster_comp.wgsl"));
        let pipeline = create_compute_pipeline(device, wgsl, "diffusion_raster");
        Self { pipeline }
    }

    /// Encode the diffusion rasterize pass. Returns the output texture.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pixels: &[u32],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        scale: u32,
    ) -> wgpu::Texture {
        use wgpu::util::DeviceExt;
        use crate::vectorize::{graph, rasterize::build_graph_regions};

        // Build similarity graph and regions on CPU
        let g = graph::build(pixels, src_w as usize, src_h as usize);
        let regions = build_graph_regions(src_w as usize, src_h as usize, &g);

        // Compute packed diagonal states per pixel
        let mut diags = vec![0u32; (src_w * src_h) as usize];
        for py in 0..src_h as usize {
            for px in 0..src_w as usize {
                let tl = corner_diag(&g, px, py) as u32;
                let tr = corner_diag(&g, px + 1, py) as u32;
                let br = corner_diag(&g, px + 1, py + 1) as u32;
                let bl = corner_diag(&g, px, py + 1) as u32;
                diags[py * src_w as usize + px] = tl | (tr << 2) | (br << 4) | (bl << 6);
            }
        }

        // Upload buffers
        let px_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff_pixels"),
            contents: bytemuck::cast_slice(pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let reg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff_regions"),
            contents: bytemuck::cast_slice(&regions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let diag_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff_diags"),
            contents: bytemuck::cast_slice(&diags),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("diff_output"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&Default::default());

        let inv_scale = 1.0f32 / scale as f32;
        let uniforms: [u32; 8] = [
            out_w, out_h, src_w, src_h,
            f32::to_bits(inv_scale), f32::to_bits(GAUSS_K), f32::to_bits(GAUSS_RADIUS), 0,
        ];
        let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff_uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind groups: group 0=storage read (vk set 0), 1=texture (vk set 1), 2=uniforms (vk set 2)
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diff_bg0"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: px_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: reg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: diag_buf.as_entire_binding() },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diff_bg1"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_view) },
            ],
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diff_bg2"),
            layout: &self.pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uni_buf.as_entire_binding() },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diffusion_raster"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups((out_w + 15) / 16, (out_h + 15) / 16, 1);
        }

        output_tex
    }
}

// ── Spline-diffusion 2-pass pipeline ──────────────────────────────────────

/// Two-pass spline-diffusion pipeline:
/// 1. `vectorize_to_buf.comp` — rasterize B-spline edges into region buffer
/// 2. `spline_diffusion.comp` — Gaussian blend using region buffer + source pixels
pub struct WgpuSplineDiffusionPipeline {
    pass1: wgpu::ComputePipeline,
    pass2: wgpu::ComputePipeline,
}

impl WgpuSplineDiffusionPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let wgsl1 = include_str!(concat!(env!("OUT_DIR"), "/vectorize_to_buf_comp.wgsl"));
        let wgsl2 = include_str!(concat!(env!("OUT_DIR"), "/spline_diffusion_comp.wgsl"));
        Self {
            pass1: create_compute_pipeline(device, wgsl1, "vectorize_to_buf"),
            pass2: create_compute_pipeline(device, wgsl2, "spline_diffusion"),
        }
    }

    /// Encode both passes. Returns the output texture.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        edges: &[crate::vectorize::rasterize::GpuEdgeV2],
        row_ranges: &[crate::vectorize::rasterize::GpuRowRange],
        edge_indices: &[u32],
        pixels: &[u32],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        bg_color: u32,
        scale: u32,
    ) -> wgpu::Texture {
        use wgpu::util::DeviceExt;

        // Upload edge data buffers
        let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sd_edges"),
            contents: bytemuck::cast_slice(edges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let row_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sd_rows"),
            contents: bytemuck::cast_slice(row_ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let idx_data = if edge_indices.is_empty() { &[0u32][..] } else { edge_indices };
        let idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sd_indices"),
            contents: bytemuck::cast_slice(idx_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Intermediate region buffer (read-write storage, NOT a texture)
        let region_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sd_region_buf"),
            size: ((out_w * out_h * 4) as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Pass 1 uniforms
        let uni1 = [out_w, out_h, edges.len() as u32, bg_color];
        let uni1_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sd_uni1"),
            contents: bytemuck::cast_slice(&uni1),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Pass 1 bind groups: group 0=read storage (vk set 0), 1=RW storage (vk set 1), 2=uniforms (vk set 2)
        let bg1_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sd_p1_bg0"),
            layout: &self.pass1.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: edge_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: row_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: idx_buf.as_entire_binding() },
            ],
        });
        let bg1_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sd_p1_bg1"),
            layout: &self.pass1.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: region_buf.as_entire_binding() },
            ],
        });
        let bg1_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sd_p1_bg2"),
            layout: &self.pass1.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uni1_buf.as_entire_binding() },
            ],
        });

        // Pass 1: vectorize_to_buf
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vectorize_to_buf"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pass1);
            pass.set_bind_group(0, &bg1_0, &[]);
            pass.set_bind_group(1, &bg1_1, &[]);
            pass.set_bind_group(2, &bg1_2, &[]);
            pass.dispatch_workgroups((out_w + 15) / 16, (out_h + 15) / 16, 1);
        }

        // Output texture for pass 2
        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sd_output"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&Default::default());

        // Upload pixel data for pass 2
        let px_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sd_pixels"),
            contents: bytemuck::cast_slice(pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pass 2 uniforms
        let inv_scale = 1.0f32 / scale as f32;
        let uni2: [u32; 8] = [
            out_w, out_h, src_w, src_h,
            f32::to_bits(inv_scale), f32::to_bits(GAUSS_K), f32::to_bits(GAUSS_RADIUS), scale,
        ];
        let uni2_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sd_uni2"),
            contents: bytemuck::cast_slice(&uni2),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Pass 2 bind groups: group 0=read storage (vk set 0), 1=texture (vk set 1), 2=uniforms (vk set 2)
        let bg2_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sd_p2_bg0"),
            layout: &self.pass2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: px_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: region_buf.as_entire_binding() },
            ],
        });
        let bg2_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sd_p2_bg1"),
            layout: &self.pass2.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_view) },
            ],
        });
        let bg2_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sd_p2_bg2"),
            layout: &self.pass2.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uni2_buf.as_entire_binding() },
            ],
        });

        // Pass 2: spline_diffusion
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spline_diffusion"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pass2);
            pass.set_bind_group(0, &bg2_0, &[]);
            pass.set_bind_group(1, &bg2_1, &[]);
            pass.set_bind_group(2, &bg2_2, &[]);
            pass.dispatch_workgroups((out_w + 15) / 16, (out_h + 15) / 16, 1);
        }

        output_tex
    }
}
