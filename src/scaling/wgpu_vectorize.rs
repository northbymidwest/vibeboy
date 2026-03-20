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
    ecolor_buf: wgpu::Buffer,
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
        let ecolor_size = (num_cps * 4 * 4) as u64;

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
        let ecolor_buf = mk("edge_colors", ecolor_size, storage_rw);
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

        // Pre-create all bind groups
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
                wgpu::BindGroupEntry { binding: 3, resource: ecolor_buf.as_entire_binding() },
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
                wgpu::BindGroupEntry { binding: 5, resource: ecolor_buf.as_entire_binding() },
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
            ecolor_buf, opt_out_buf, orig_pos_buf,
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
        device.poll(wgpu::Maintain::Wait);
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
