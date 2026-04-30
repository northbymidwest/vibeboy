use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString, ns_string};
use objc2_metal::*;

type Device = ProtocolObject<dyn MTLDevice>;
type CmdQueue = ProtocolObject<dyn MTLCommandQueue>;
type Texture = ProtocolObject<dyn MTLTexture>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ComputePipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Full GPU vectorize pipeline via Metal compute (7 stages: similarity_graph
/// → resolve_crossings → cell_graph → optimize_energy → update_tjunction →
/// crossing_pack → cell_rasterizer).
pub(super) struct MetalVectorizePipeline {
    sim_graph: ComputePipeline,
    resolve: ComputePipeline,
    cell_graph: ComputePipeline,
    optimizer: ComputePipeline,
    tjunction: ComputePipeline,
    crossing_pack: ComputePipeline,
    rasterizer: ComputePipeline,
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
    opt_out_buf: Buffer,
    orig_pos_buf: Buffer,
    crossing_t_buf: Buffer,
}

fn load_msl(device: &Device, msl: &[u8]) -> Option<ComputePipeline> {
    let src = std::str::from_utf8(msl).ok()?;
    let ns_src = NSString::from_str(src);
    let lib = device.newLibraryWithSource_options_error(&ns_src, None)
        .map_err(|e| eprintln!("MSL compile error: {e}")).ok()?;
    let func = lib.newFunctionWithName(ns_string!("main_0"))?;
    device.newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| eprintln!("Pipeline error: {e}")).ok()
}

fn mk_buf(device: &Device, sz: usize) -> Buffer {
    device.newBufferWithLength_options(sz.max(4), MTLResourceOptions::StorageModeShared)
        .expect("failed to create buffer")
}

impl MetalVectorizePipeline {
    pub fn new(device: &Device) -> Option<Self> {
        Some(MetalVectorizePipeline {
            sim_graph: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.metal")))?,
            resolve: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.metal")))?,
            cell_graph: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.metal")))?,
            optimizer: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/optimize_energy_comp.metal")))?,
            tjunction: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.metal")))?,
            crossing_pack: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/crossing_pack_comp.metal")))?,
            rasterizer: load_msl(device, include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.metal")))?,
            bufs: None,
        })
    }

    pub fn run(
        &mut self,
        device: &Device,
        queue: &CmdQueue,
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
            self.bufs = Some(MetalVecBufs {
                img_w, img_h,
                px_buf: mk_buf(device, (img_w * img_h * 4) as usize),
                graph_buf: mk_buf(device, (graph_elems * 4) as usize),
                graph_snapshot: mk_buf(device, (graph_elems * 4) as usize),
                pos_buf: mk_buf(device, (num_cps * 2 * 4) as usize),
                nbr_buf: mk_buf(device, (num_cps * 4 * 4) as usize),
                flag_buf: mk_buf(device, (num_cps * 4) as usize),
                opt_out_buf: mk_buf(device, (num_cps * 2 * 4) as usize),
                orig_pos_buf: mk_buf(device, (num_cps * 2 * 4) as usize),
                crossing_t_buf: mk_buf(device, (num_cps * 4) as usize),
            });
        }
        let b = self.bufs.as_ref().unwrap();

        // Upload pixels
        unsafe {
            std::ptr::copy_nonoverlapping(
                pixels.as_ptr() as *const u8,
                b.px_buf.contents().as_ptr() as *mut u8,
                (img_w * img_h * 4) as usize,
            );
        }

        let tiles_w = (img_w + 1) / 2;
        let tiles_h = (img_h + 1) / 2;

        // Helper to create uniform buffer
        let mk_uni = |data: &[u32]| -> Buffer {
            let buf = mk_buf(device, data.len() * 4);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    buf.contents().as_ptr() as *mut u8,
                    data.len() * 4,
                );
            }
            buf
        };

        // Zero all buffers before pipeline starts
        let cmd = queue.commandBuffer().unwrap();
        {
            let enc = cmd.blitCommandEncoder().unwrap();
            for buf in [&b.graph_buf, &b.graph_snapshot, &b.pos_buf, &b.nbr_buf,
                        &b.flag_buf, &b.opt_out_buf, &b.orig_pos_buf, &b.crossing_t_buf] {
                enc.fillBuffer_range_value(buf, NSRange::new(0, buf.length()), 0);
            }
            enc.endEncoding();
        }

        // Stage 1: Similarity graph
        {
            let uni = mk_uni(&[img_w, img_h, graph_stride, 0]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.sim_graph);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b.px_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.graph_buf), 0, 2);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((img_w + 15) / 16) as usize, height: ((img_h + 15) / 16) as usize, depth: 1 },
                    MTLSize { width: 16, height: 16, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Copy graph -> graph_snapshot
        {
            let enc = cmd.blitCommandEncoder().unwrap();
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &b.graph_buf, 0, &b.graph_snapshot, 0, (graph_elems * 4) as usize,
                );
            }
            enc.endEncoding();
        }

        // Stage 2: Resolve crossings
        {
            let uni = mk_uni(&[img_w, img_h, graph_stride, 0]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.resolve);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b.graph_snapshot), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.graph_buf), 0, 2);
            }
            let rw = img_w.saturating_sub(1);
            let rh = img_h.saturating_sub(1);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((rw + 15) / 16) as usize, height: ((rh + 15) / 16) as usize, depth: 1 },
                    MTLSize { width: 16, height: 16, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Stage 3: Cell graph
        {
            let uni = mk_uni(&[img_w, img_h, graph_stride, corners_w]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.cell_graph);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b.graph_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.pos_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b.nbr_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&b.flag_buf), 0, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((corners_w + 15) / 16) as usize, height: ((corners_h + 15) / 16) as usize, depth: 1 },
                    MTLSize { width: 16, height: 16, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Copy pos -> orig_pos
        {
            let enc = cmd.blitCommandEncoder().unwrap();
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &b.pos_buf, 0, &b.orig_pos_buf, 0, (num_cps * 2 * 4) as usize,
                );
            }
            enc.endEncoding();
        }

        // Stage 4: Optimize energy (2 iterations, ping-pong)
        for iter in 0..2u32 {
            let (src, dst) = if iter % 2 == 0 {
                (&b.pos_buf, &b.opt_out_buf)
            } else {
                (&b.opt_out_buf, &b.pos_buf)
            };
            let uni = mk_uni(&[num_cps, f32::to_bits(0.01), f32::to_bits(0.25), f32::to_bits(2.5)]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.optimizer);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(src), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.orig_pos_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b.nbr_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&b.flag_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(dst), 0, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((num_cps + 255) / 256) as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Stage 5a: T-junction stem snap (3× for convergence). The legacy
        // IS_CROSSING inverse-correction branch is gone, so the tjunction
        // shader now binds only nbr/flag (RO) + pos (RW) — no tjunc_orig.
        for _ in 0..3 {
            let uni = mk_uni(&[num_cps, 0, 0, 0]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.tjunction);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b.nbr_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.flag_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b.pos_buf), 0, 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((num_cps + 255) / 256) as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Stage 5b: Crossing intersection pack. Writes (t_NS, t_EW) to
        // crossing_t for every IS_CROSSING CP and 0.5 elsewhere. Reads
        // finalized stem positions from the prior tjunction snap loop —
        // Metal inserts the necessary inter-encoder barriers automatically.
        {
            let uni = mk_uni(&[num_cps, 0, 0, 0]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.crossing_pack);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b.nbr_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.flag_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b.pos_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&b.crossing_t_buf), 0, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: ((num_cps + 255) / 256) as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        // Stage 6: Cell rasterizer
        {
            let uni = mk_uni(&[img_w, img_h, out_w, out_h,
                f32::to_bits(scale), corners_w, tiles_w, tiles_h]);
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&self.rasterizer);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&uni), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b.px_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b.pos_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b.orig_pos_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&b.flag_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&b.nbr_buf), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&b.crossing_t_buf), 0, 6);
                enc.setTexture_atIndex(Some(out_tex), 0);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (tiles_w * tiles_h) as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
        }

        cmd.commit();
        cmd.waitUntilCompleted();
    }
}
