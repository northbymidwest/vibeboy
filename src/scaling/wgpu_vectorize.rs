//! Full GPU vectorize pipeline using wgpu (WebGPU-compatible).
//!
//! Seven-stage pipeline matching the SDL3 GPU version:
//! 1. similarity_graph → 2. resolve_crossings → 3. cell_graph →
//! 4. optimize_energy → 5. update_tjunction → 5b. crossing_pack →
//! 6. cell_rasterizer
//!
//! All shaders are loaded from WGSL (cross-compiled from Slang via slangc).

use wgpu;

/// Cached wgpu compute pipelines and buffers for the vectorize pipeline.
pub struct WgpuVectorizePipeline {
    sim_graph: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    cell_graph: wgpu::ComputePipeline,
    picard: wgpu::ComputePipeline,
    grad: wgpu::ComputePipeline,
    tjunction: wgpu::ComputePipeline,
    crossing_pack: wgpu::ComputePipeline,
    rasterizer: wgpu::ComputePipeline,
    /// Number of outer iterations of the (Picard → gradient) 2-pass
    /// cycle. Matches vectorscale's optimize-energy + gradient-correction
    /// chain — IFT correction was dropped after CPU sweeps showed the
    /// clamped Picard + gradient correction alone converges to a lower
    /// final energy than the 3-pass cycle, and the IFT pass added register
    /// pressure for no convergence win. N=3 is the default — within 1.3%
    /// of the true energy minimum and visually indistinguishable from
    /// fully-converged CG on standard pixel-art sprites.
    pub outer_passes: u32,
    /// Gradient-correction step size. CPU sweep showed η=0.05 is the
    /// sweet spot — larger values diverge, smaller values undershoot.
    pub eta: f32,
    /// Per-CP gradient-correction step magnitude cap.
    pub max_step: f32,
    // Cached buffers (allocated once, reused each frame)
    bufs: Option<VecBufs>,
}

struct VecBufs {
    img_w: u32,
    img_h: u32,
    px_buf: wgpu::Buffer,
    graph_buf: wgpu::Buffer,
    graph_snapshot: wgpu::Buffer,
    /// 8-bit valence mask per pixel, populated by similarity_graph and
    /// read by resolve_crossings.
    valence_buf: wgpu::Buffer,
    /// Ping-pong slot A. Initialized by cell_graph; on iter k it is the
    /// outer-iter input if k is even and the output if k is odd.
    pos_buf: wgpu::Buffer,
    nbr_buf: wgpu::Buffer,
    flag_buf: wgpu::Buffer,
    /// Ping-pong slot B. Opposite parity of pos_buf each outer iter.
    opt_out_buf: wgpu::Buffer,
    /// Intermediate after Picard pass each outer iter. Overwritten by
    /// each iter's Pass A and read by Pass B (gradient correction).
    opt_picard: wgpu::Buffer,
    orig_pos_buf: wgpu::Buffer,
    crossing_t_buf: wgpu::Buffer,
    // One uniform buffer per stage to avoid write conflicts.
    uni_sim: wgpu::Buffer,
    uni_resolve: wgpu::Buffer,
    uni_cell: wgpu::Buffer,
    uni_opt: wgpu::Buffer,
    uni_grad: wgpu::Buffer,
    uni_tjunc: wgpu::Buffer,
    uni_xpack: wgpu::Buffer,
    uni_rast: wgpu::Buffer,
    output_tex: wgpu::Texture,
    output_tex_w: u32,
    output_tex_h: u32,
    // Cached bind groups.
    bg_sim: [wgpu::BindGroup; 3],
    bg_resolve: [wgpu::BindGroup; 3],
    bg_cell: [wgpu::BindGroup; 3],
    // Per outer iter k (3 passes). Two variants depending on which
    // ping-pong slot holds the iter's input:
    //   even k: input = pos_buf, output = opt_out_buf
    //   odd k:  input = opt_out_buf, output = pos_buf
    bg_picard_a: [wgpu::BindGroup; 3],
    bg_picard_b: [wgpu::BindGroup; 3],
    bg_grad_a: [wgpu::BindGroup; 3],
    bg_grad_b: [wgpu::BindGroup; 3],
    bg_tjunc: [wgpu::BindGroup; 3],
    bg_xpack: [wgpu::BindGroup; 3],
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
    /// Create all 8 compute pipelines from compiled WGSL shaders.
    pub fn new(device: &wgpu::Device) -> Self {
        let sim_wgsl = include_str!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.wgsl"));
        let resolve_wgsl = include_str!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.wgsl"));
        let cell_wgsl = include_str!(concat!(env!("OUT_DIR"), "/cell_graph_comp.wgsl"));
        let picard_wgsl = include_str!(concat!(env!("OUT_DIR"), "/picard_step_comp.wgsl"));
        let grad_wgsl = include_str!(concat!(env!("OUT_DIR"), "/gradient_correction_comp.wgsl"));
        let tjunc_wgsl = include_str!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.wgsl"));
        let xpack_wgsl = include_str!(concat!(env!("OUT_DIR"), "/crossing_pack_comp.wgsl"));
        let rast_wgsl = include_str!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.wgsl"));

        WgpuVectorizePipeline {
            sim_graph: create_compute_pipeline(device, sim_wgsl, "similarity_graph"),
            resolve: create_compute_pipeline(device, resolve_wgsl, "resolve_crossings"),
            cell_graph: create_compute_pipeline(device, cell_wgsl, "cell_graph"),
            picard: create_compute_pipeline(device, picard_wgsl, "picard_step"),
            grad: create_compute_pipeline(device, grad_wgsl, "gradient_correction"),
            tjunction: create_compute_pipeline(device, tjunc_wgsl, "update_tjunction"),
            crossing_pack: create_compute_pipeline(device, xpack_wgsl, "crossing_pack"),
            rasterizer: create_compute_pipeline(device, rast_wgsl, "cell_rasterizer"),
            outer_passes: 3,
            eta: 0.05,
            max_step: 0.25,
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

        let crossing_t_size = (num_cps * 4) as u64;
        let px_buf = mk("pixels", px_size, storage_ro);
        let graph_buf = mk("graph", graph_size, storage_rw);
        let graph_snapshot = mk("graph_snap", graph_size, storage_ro | wgpu::BufferUsages::COPY_SRC);
        let valence_buf = mk("valence", (img_w * img_h * 4) as u64, storage_rw);
        let pos_buf = mk("positions", pos_size, storage_rw | wgpu::BufferUsages::COPY_DST);
        let nbr_buf = mk("neighbors", nbr_size, storage_rw);
        let flag_buf = mk("flags", flag_size, storage_rw);
        let opt_out_buf = mk("opt_out", pos_size, storage_rw | wgpu::BufferUsages::COPY_SRC);
        let opt_picard = mk("opt_picard", pos_size, storage_rw);
        let orig_pos_buf = mk("orig_pos", pos_size, storage_rw);
        let crossing_t_buf = mk("crossing_t", crossing_t_size, storage_rw);
        let uni_sim = mk("uni_sim", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_resolve = mk("uni_resolve", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_cell = mk("uni_cell", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_opt = mk("uni_opt", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_grad = mk("uni_grad", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_tjunc = mk("uni_tjunc", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let uni_xpack = mk("uni_xpack", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
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
            bg(&self.sim_graph, 1, &[
                wgpu::BindGroupEntry { binding: 0, resource: graph_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: valence_buf.as_entire_binding() },
            ]),
            bg(&self.sim_graph, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_sim.as_entire_binding() }]),
        ];
        let bg_resolve = [
            bg(&self.resolve, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: graph_snapshot.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: valence_buf.as_entire_binding() },
            ]),
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
        // Picard pass bind groups. Reads pos_in + orig + nbrs + flags,
        // writes opt_picard. Two variants by ping-pong parity:
        //   bg_picard_a (even iter): pos_in = pos_buf
        //   bg_picard_b (odd  iter): pos_in = opt_out_buf
        let bg_picard_a = [
            bg(&self.picard, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.picard, 1, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_picard.as_entire_binding() },
            ]),
            bg(&self.picard, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_opt.as_entire_binding() }]),
        ];
        let bg_picard_b = [
            bg(&self.picard, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.picard, 1, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_picard.as_entire_binding() },
            ]),
            bg(&self.picard, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_opt.as_entire_binding() }]),
        ];
        // IFT pass bind groups. Reads pos_a + opt_picard + orig + nbrs + flags,
        // Gradient pass bind groups. Reads opt_picard (picard's output)
        // + orig + nbrs + flags, writes pos_output (opt_out_buf on even
        // iter, pos_buf on odd). Matches vectorscale's 2-pass
        // (Picard → gradient correction) optimizer chain.
        let bg_grad_a = [
            bg(&self.grad, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_picard.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.grad, 1, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_out_buf.as_entire_binding() },
            ]),
            bg(&self.grad, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_grad.as_entire_binding() }]),
        ];
        let bg_grad_b = [
            bg(&self.grad, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: opt_picard.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.grad, 1, &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
            ]),
            bg(&self.grad, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_grad.as_entire_binding() }]),
        ];
        // update_tjunction: 3 buffers at group 0 (positions RW + nbr/flag RO),
        // update_tjunction: RO inputs (neighbor_data, node_flags) at group 0,
        // RW positions at group 1 binding 0, uniforms at group 2.
        let bg_tjunc = [
            bg(&self.tjunction, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: flag_buf.as_entire_binding() },
            ]),
            bg(&self.tjunction, 1, &[wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() }]),
            bg(&self.tjunction, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_tjunc.as_entire_binding() }]),
        ];
        // crossing_pack: RO inputs (neighbor_data, node_flags, positions) at
        // group 0, RW crossing_t at group 1 binding 0, uniforms at group 2.
        let bg_xpack = [
            bg(&self.crossing_pack, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: nbr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: flag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pos_buf.as_entire_binding() },
            ]),
            bg(&self.crossing_pack, 1, &[wgpu::BindGroupEntry { binding: 0, resource: crossing_t_buf.as_entire_binding() }]),
            bg(&self.crossing_pack, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_xpack.as_entire_binding() }]),
        ];
        // Rasterizer: group 1 holds the output texture *and* crossing_t (now
        // declared as RWStructuredBuffer in the slang shader so wgpu emits
        // a barrier from the prior crossing_pack write — the shader only
        // reads, but wgpu's hazard tracking needs the RW binding type).
        let bg_rast = [
            bg(&self.rasterizer, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: px_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: orig_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: flag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: nbr_buf.as_entire_binding() },
            ]),
            bg(&self.rasterizer, 1, &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry { binding: 1, resource: crossing_t_buf.as_entire_binding() },
            ]),
            bg(&self.rasterizer, 2, &[wgpu::BindGroupEntry { binding: 0, resource: uni_rast.as_entire_binding() }]),
        ];

        self.bufs = Some(VecBufs {
            img_w, img_h,
            px_buf, graph_buf, graph_snapshot, valence_buf, pos_buf, nbr_buf, flag_buf,
            opt_out_buf, opt_picard, orig_pos_buf, crossing_t_buf,
            uni_sim, uni_resolve, uni_cell, uni_opt, uni_grad, uni_tjunc, uni_xpack, uni_rast,
            output_tex, output_tex_w: out_w, output_tex_h: out_h,
            bg_sim, bg_resolve, bg_cell,
            bg_picard_a, bg_picard_b, bg_grad_a, bg_grad_b,
            bg_tjunc, bg_xpack, bg_rast,
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
        // Picard + IFT uniforms: { num_nodes, _pad0, _pad1, _pad2 }.
        write_uniform(queue, &b.uni_opt, &[num_cps, 0, 0, 0]);
        // Gradient uniform: { num_nodes, eta, max_step, _pad }.
        let uni_grad_data: [u32; 4] = [
            num_cps,
            f32::to_bits(self.eta),
            f32::to_bits(self.max_step),
            0,
        ];
        write_uniform(queue, &b.uni_grad, &uni_grad_data);
        write_uniform(queue, &b.uni_tjunc, &[num_cps, 0, 0, 0]);
        write_uniform(queue, &b.uni_xpack, &[num_cps, 0, 0, 0]);
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

        // Compute pass 3: outer optimizer loop. Each iter dispatches 2
        // compute shaders in sequence (Picard → gradient), with
        // ping-pong between pos_buf and opt_out_buf as the iter
        // input/output. Picard writes the inner-Newton step to
        // opt_picard; gradient correction reads opt_picard and writes
        // the de-biased position to the iter's output slot. Matches
        // vectorscale's optimize-energy + gradient-correction chain.
        let outer_passes = self.outer_passes;
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("optimizer"),
                timestamp_writes: None,
            });
            let nworkgroups = (num_cps + 255) / 256;
            for pass in 0..outer_passes {
                let even = pass % 2 == 0;
                let (bg_picard, bg_grad) = if even {
                    (&b.bg_picard_a, &b.bg_grad_a)
                } else {
                    (&b.bg_picard_b, &b.bg_grad_b)
                };
                cp.set_pipeline(&self.picard);
                cp.set_bind_group(0, &bg_picard[0], &[]);
                cp.set_bind_group(1, &bg_picard[1], &[]);
                cp.set_bind_group(2, &bg_picard[2], &[]);
                cp.dispatch_workgroups(nworkgroups, 1, 1);
                cp.set_pipeline(&self.grad);
                cp.set_bind_group(0, &bg_grad[0], &[]);
                cp.set_bind_group(1, &bg_grad[1], &[]);
                cp.set_bind_group(2, &bg_grad[2], &[]);
                cp.dispatch_workgroups(nworkgroups, 1, 1);
            }
        }
        // Even outer_passes: final grad pass wrote to opt_out_buf.
        // Odd outer_passes: final grad pass wrote to pos_buf.
        // Downstream stages read pos_buf — copy if needed.
        if outer_passes % 2 == 0 && outer_passes > 0 {
            encoder.copy_buffer_to_buffer(&b.opt_out_buf, 0, &b.pos_buf, 0, pos_size);
        }

        // Compute pass 4: tjunction snap (3×) → crossing pack → rasterizer.
        // tjunction now writes only stem CPs (legacy IS_CROSSING branch
        // dropped); crossing_pack runs once after the snap loop converges
        // so neighbor stem positions are visible to the quartic solve.
        let _ = pos_size; // kept in scope for clarity, no longer copied
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tjunc+xpack+rast"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.tjunction);
            cp.set_bind_group(0, &b.bg_tjunc[0], &[]);
            cp.set_bind_group(1, &b.bg_tjunc[1], &[]);
            cp.set_bind_group(2, &b.bg_tjunc[2], &[]);
            for _ in 0..3 {
                cp.dispatch_workgroups((num_cps + 255) / 256, 1, 1);
            }
            cp.set_pipeline(&self.crossing_pack);
            cp.set_bind_group(0, &b.bg_xpack[0], &[]);
            cp.set_bind_group(1, &b.bg_xpack[1], &[]);
            cp.set_bind_group(2, &b.bg_xpack[2], &[]);
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

        let data = slice.get_mapped_range().ok()?;
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

// (Legacy shared-chain, diffusion, and spline-diffusion pipelines removed --
// use the full GPU vectorize pipeline via WgpuVectorizePipeline instead.)
