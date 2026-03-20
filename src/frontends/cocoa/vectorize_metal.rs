use metal::*;

/// Full GPU vectorize pipeline via Metal compute (6 stages).
pub(super) struct MetalVectorizePipeline {
    sim_graph: ComputePipelineState,
    resolve: ComputePipelineState,
    cell_graph: ComputePipelineState,
    optimizer: ComputePipelineState,
    tjunction: ComputePipelineState,
    rasterizer: ComputePipelineState,
    // Cached buffers (allocated once, reused)
    bufs: Option<MetalVecBufs>,
}

pub(super) struct MetalVecBufs {
    img_w: u32,
    img_h: u32,
    px_buf: Buffer,
    graph_buf: Buffer,
    graph_snapshot: Buffer,
    pos_buf: Buffer,
    nbr_buf: Buffer,
    flag_buf: Buffer,
    ecolor_buf: Buffer,
    opt_out_buf: Buffer,
    orig_pos_buf: Buffer,
}

impl MetalVectorizePipeline {
    pub fn new(device: &Device) -> Option<Self> {
        fn load_msl(device: &Device, msl: &[u8]) -> Option<ComputePipelineState> {
            let src = std::str::from_utf8(msl).ok()?;
            let lib = device.new_library_with_source(src, &CompileOptions::new())
                .map_err(|e| eprintln!("MSL compile error: {e}")).ok()?;
            let func = lib.get_function("main0", None)
                .map_err(|e| eprintln!("MSL function error: {e}")).ok()?;
            device.new_compute_pipeline_state_with_function(&func)
                .map_err(|e| eprintln!("Pipeline error: {e}")).ok()
        }

        // Use RAW spirv-cross MSL output (no build.rs remap).
        // We generate it at runtime to avoid the SDL3 remap that build.rs applies.
        fn load_spv_as_msl(device: &Device, spv: &[u8]) -> Option<ComputePipelineState> {
            // Use pre-compiled MSL — bind according to its buffer indices
            load_msl(device, spv)
        }

        Some(MetalVectorizePipeline {
            sim_graph: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.metal")))?,
            resolve: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.metal")))?,
            cell_graph: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.metal")))?,
            optimizer: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.metal")))?,
            tjunction: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.metal")))?,
            rasterizer: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.metal")))?,
            bufs: None,
        })
    }

    pub fn run(
        &mut self,
        device: &Device,
        queue: &CommandQueue,
        pixels: &[u32],
        img_w: u32, img_h: u32,
        out_w: u32, out_h: u32,
        scale: f32,
        out_tex: &Texture,
    ) {
        let graph_stride = 2 * img_w + 1;
        let corners_w = img_w + 1;
        let corners_h = img_h + 1;
        let num_cps = corners_w * corners_h * 2;
        let graph_elems = graph_stride * (2 * img_h + 1);

        // Allocate/reuse buffers
        if self.bufs.as_ref().map_or(true, |b| b.img_w != img_w || b.img_h != img_h) {
            let mk = |sz: u64| device.new_buffer(sz.max(4), MTLResourceOptions::StorageModeShared);
            self.bufs = Some(MetalVecBufs {
                img_w, img_h,
                px_buf: mk((img_w * img_h * 4) as u64),
                graph_buf: mk((graph_elems * 4) as u64),
                graph_snapshot: mk((graph_elems * 4) as u64),
                pos_buf: mk((num_cps * 2 * 4) as u64),
                nbr_buf: mk((num_cps * 4 * 4) as u64),
                flag_buf: mk((num_cps * 4) as u64),
                ecolor_buf: mk((num_cps * 4 * 4) as u64),
                opt_out_buf: mk((num_cps * 2 * 4) as u64),
                orig_pos_buf: mk((num_cps * 2 * 4) as u64),
            });
        }
        let b = self.bufs.as_ref().unwrap();

        // Upload pixels
        unsafe {
            std::ptr::copy_nonoverlapping(
                pixels.as_ptr() as *const u8,
                b.px_buf.contents() as *mut u8,
                (img_w * img_h * 4) as usize,
            );
        }

        let tiles_w = (img_w + 1) / 2;
        let tiles_h = (img_h + 1) / 2;

        // Zero all buffers before pipeline starts (required for correctness)
        let cmd = queue.new_command_buffer();
        {
            let enc = cmd.new_blit_command_encoder();
            for buf in [&b.graph_buf, &b.graph_snapshot, &b.pos_buf, &b.nbr_buf,
                        &b.flag_buf, &b.ecolor_buf, &b.opt_out_buf, &b.orig_pos_buf] {
                enc.fill_buffer(buf, metal::NSRange::new(0, buf.length()), 0);
            }
            enc.end_encoding();
        }

        // Helper to create uniform buffer
        let mk_uni = |data: &[u32]| -> Buffer {
            let buf = device.new_buffer((data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    data.len() * 4,
                );
            }
            buf
        };

        // Stage 1: Similarity graph
        {
            let uni = mk_uni(&[img_w, img_h, graph_stride, 0]);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.sim_graph);
            enc.set_buffer(0, Some(&uni), 0);
            enc.set_buffer(1, Some(&b.px_buf), 0);
            enc.set_buffer(2, Some(&b.graph_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(((img_w + 15) / 16) as u64, ((img_h + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        // Copy graph -> graph_snapshot
        {
            let enc = cmd.new_blit_command_encoder();
            enc.copy_from_buffer(&b.graph_buf, 0, &b.graph_snapshot, 0, (graph_elems * 4) as u64);
            enc.end_encoding();
        }

        // Stage 2: Resolve crossings
        {
            let uni = mk_uni(&[img_w, img_h, graph_stride, 0]);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.resolve);
            enc.set_buffer(0, Some(&uni), 0);
            enc.set_buffer(1, Some(&b.graph_snapshot), 0);
            enc.set_buffer(2, Some(&b.graph_buf), 0);
            let rw = img_w.saturating_sub(1);
            let rh = img_h.saturating_sub(1);
            enc.dispatch_thread_groups(
                MTLSize::new(((rw + 15) / 16) as u64, ((rh + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        // Stage 3: Cell graph
        {
            let uni = mk_uni(&[img_w, img_h, graph_stride, corners_w]);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.cell_graph);
            enc.set_buffer(0, Some(&uni), 0);
            enc.set_buffer(1, Some(&b.graph_buf), 0);
            enc.set_buffer(2, Some(&b.pos_buf), 0);
            enc.set_buffer(3, Some(&b.nbr_buf), 0);
            enc.set_buffer(4, Some(&b.flag_buf), 0);
            enc.set_buffer(5, Some(&b.ecolor_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(((corners_w + 15) / 16) as u64, ((corners_h + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        // Copy pos -> orig_pos
        {
            let enc = cmd.new_blit_command_encoder();
            enc.copy_from_buffer(&b.pos_buf, 0, &b.orig_pos_buf, 0, (num_cps * 2 * 4) as u64);
            enc.end_encoding();
        }

        // Stage 4: Optimize energy (2 iterations, ping-pong)
        let pos_size = (num_cps * 2 * 4) as u64;
        for iter in 0..2u32 {
            let (src, dst) = if iter % 2 == 0 {
                (&b.pos_buf, &b.opt_out_buf)
            } else {
                (&b.opt_out_buf, &b.pos_buf)
            };
            let uni = mk_uni(&[num_cps, f32::to_bits(0.01), f32::to_bits(0.25), f32::to_bits(2.5)]);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.optimizer);
            enc.set_buffer(0, Some(&uni), 0);
            enc.set_buffer(1, Some(src), 0);
            enc.set_buffer(2, Some(&b.orig_pos_buf), 0);
            enc.set_buffer(3, Some(&b.nbr_buf), 0);
            enc.set_buffer(4, Some(&b.flag_buf), 0);
            enc.set_buffer(5, Some(dst), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(((num_cps + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        // After 2 iterations, result is back in pos_buf

        // Stage 4b: T-junction correction
        // MSL buffer order (after remap): 0=uniforms, 1=neighbors, 2=flags, 3=positions
        {
            let uni = mk_uni(&[num_cps, 0, 0, 0]);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.tjunction);
            enc.set_buffer(0, Some(&uni), 0);
            enc.set_buffer(1, Some(&b.nbr_buf), 0);
            enc.set_buffer(2, Some(&b.flag_buf), 0);
            enc.set_buffer(3, Some(&b.pos_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(((num_cps + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }

        // Stage 5: Cell rasterizer
        // MSL buffer order (after remap): 0=uniforms, 1=pixels,
        // 2=positions, 3=orig_positions, 4=flags, 5=neighbors, 6=edge_colors
        {
            let uni = mk_uni(&[img_w, img_h, out_w, out_h,
                f32::to_bits(scale), corners_w, tiles_w, tiles_h]);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.rasterizer);
            enc.set_buffer(0, Some(&uni), 0);
            enc.set_buffer(1, Some(&b.px_buf), 0);
            enc.set_buffer(2, Some(&b.pos_buf), 0);
            enc.set_buffer(3, Some(&b.orig_pos_buf), 0);
            enc.set_buffer(4, Some(&b.flag_buf), 0);
            enc.set_buffer(5, Some(&b.nbr_buf), 0);
            enc.set_buffer(6, Some(&b.ecolor_buf), 0);
            enc.set_texture(0, Some(out_tex));
            enc.dispatch_thread_groups(
                MTLSize::new((tiles_w * tiles_h) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();
    }
}
