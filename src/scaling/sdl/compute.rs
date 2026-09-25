//! Full GPU vectorize pipeline: compute pipeline init and dispatch.

use super::common::*;
use crate::scaling::vectorize::{OPT_GRAD_ETA, OPT_GRAD_MAX_STEP, OPT_OUTER_PASSES};
use sdl3::gpu;

// ── Full GPU vectorize pipeline ─────────────────────────────────────────────
//
// Eight-stage pipeline: similarity_graph → resolve_crossings → cell_graph →
// picard_step → gradient_correction (alternating per outer iter) →
// tjunction_snap → crossing_pack → cell_rasterizer. All stages run on GPU
// with no CPU readback between stages. Buffers are cached between frames
// for zero per-frame allocation.

/// Cached GPU buffers for the cell rasterizer pipeline.
/// Allocated once and reused each frame (GB image dimensions never change).
struct CellRastBufCache {
    img_w: u32,
    img_h: u32,
    px_buf: gpu::Buffer,
    graph_buf: gpu::Buffer,
    graph_snapshot: gpu::Buffer,
    /// 8-bit valence mask per pixel, populated by similarity_graph
    /// and read by resolve_crossings. Replaces the 8-fetch valence
    /// walk in resolve_crossings's hot loops.
    valence_buf: gpu::Buffer,
    pos_buf: gpu::Buffer,
    nbr_buf: gpu::Buffer,
    flag_buf: gpu::Buffer,
    opt_out_buf: gpu::Buffer,
    orig_pos_buf: gpu::Buffer,
    crossing_t_buf: gpu::Buffer,
    /// Per-iter intermediate: Picard pass writes here, grad pass reads from
    /// here. Lives across iters as a fixed buffer (no ping-pong).
    opt_picard_buf: gpu::Buffer,
    px_xfer: gpu::TransferBuffer,
}

/// All pipelines for the full GPU vectorize pipeline.
pub struct GpuVectorizePipelines {
    sim_graph: gpu::ComputePipeline,
    resolve: gpu::ComputePipeline,
    cell_graph: gpu::ComputePipeline,
    picard: gpu::ComputePipeline,
    grad: gpu::ComputePipeline,
    tjunction: gpu::ComputePipeline,
    crossing_pack: gpu::ComputePipeline,
    rasterizer: gpu::ComputePipeline,
    buf_cache: Option<CellRastBufCache>,
}

pub fn init_full_gpu_pipeline(device: &gpu::Device) -> Option<GpuVectorizePipelines> {
    fn make(
        device: &gpu::Device,
        spirv: &[u8],
        msl: &[u8],
        dxil: &[u8],
        ro_bufs: u32,
        rw_bufs: u32,
        rw_tex: u32,
        threads: (u32, u32, u32),
        label: &str,
    ) -> Option<gpu::ComputePipeline> {
        let result = device
            .create_compute_pipeline()
            .with_code(gpu::ShaderFormat::SPIRV, spirv)
            .with_entrypoint(c"main")
            .with_uniform_buffers(1)
            .with_readonly_storage_buffers(ro_bufs)
            .with_readwrite_storage_buffers(rw_bufs)
            .with_readwrite_storage_textures(rw_tex)
            .with_thread_count(threads.0, threads.1, threads.2)
            .build()
            .or_else(|_| {
                if !dxil.is_empty() {
                    device
                        .create_compute_pipeline()
                        .with_code(gpu::ShaderFormat::DXIL, dxil)
                        .with_entrypoint(c"main")
                        .with_uniform_buffers(1)
                        .with_readonly_storage_buffers(ro_bufs)
                        .with_readwrite_storage_buffers(rw_bufs)
                        .with_readwrite_storage_textures(rw_tex)
                        .with_thread_count(threads.0, threads.1, threads.2)
                        .build()
                } else {
                    Err(sdl3::get_error())
                }
            })
            .or_else(|_| {
                device
                    .create_compute_pipeline()
                    .with_code(gpu::ShaderFormat::MSL, msl)
                    .with_entrypoint(c"main_0")
                    .with_uniform_buffers(1)
                    .with_readonly_storage_buffers(ro_bufs)
                    .with_readwrite_storage_buffers(rw_bufs)
                    .with_readwrite_storage_textures(rw_tex)
                    .with_thread_count(threads.0, threads.1, threads.2)
                    .build()
            });
        match result {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("vectorize-gpu: {label} pipeline failed: {e}");
                None
            }
        }
    }

    let sim = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/similarity_graph_comp.dxil")),
        1,
        2,
        0,
        (16, 16, 1),
        "similarity_graph",
    )?;

    let resolve = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/resolve_crossings_comp.dxil")),
        2,
        1,
        0,
        (16, 16, 1),
        "resolve_crossings",
    )?;

    let cell = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_graph_comp.dxil")),
        1,
        3,
        0,
        (16, 16, 1),
        "cell_graph",
    )?;

    let picard = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/picard_step_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/picard_step_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/picard_step_comp.dxil")),
        4,
        1,
        0,
        (256, 1, 1),
        "picard_step",
    )?;

    let grad = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/gradient_correction_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/gradient_correction_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/gradient_correction_comp.dxil")),
        4,
        1,
        0,
        (256, 1, 1),
        "gradient_correction",
    )?;

    let tjunc = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/update_tjunction_comp.dxil")),
        2,
        1,
        0,
        (256, 1, 1),
        "update_tjunction",
    )?;

    let xpack = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/crossing_pack_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/crossing_pack_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/crossing_pack_comp.dxil")),
        3,
        1,
        0,
        (256, 1, 1),
        "crossing_pack",
    )?;

    let rast = make(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.metal")),
        include_bytes!(concat!(env!("OUT_DIR"), "/cell_rasterizer_comp.dxil")),
        5,
        1,
        1,
        (256, 1, 1),
        "cell_rasterizer",
    )?;

    eprintln!("Full GPU vectorize pipeline ready (8 stages)");
    Some(GpuVectorizePipelines {
        sim_graph: sim,
        resolve,
        cell_graph: cell,
        picard,
        grad,
        tjunction: tjunc,
        crossing_pack: xpack,
        rasterizer: rast,
        buf_cache: None,
    })
}

/// Dispatch stages 1-5b (similarity graph through crossing intersection pack).
/// Shared between live pipeline and screenshot pipeline.
/// Returns the buffer containing the optimized+snapped positions. The
/// optimizer ping-pongs between `pos_buf` and `opt_out_buf`, so which one
/// holds the final result depends on the pass count; callers must bind the
/// returned buffer, not either input.
#[must_use = "the optimizer result may live in opt_out_buf, not pos_buf"]
fn dispatch_stages_1_5b(
    device: &gpu::Device,
    cmd: &gpu::CommandBuffer,
    pipelines: &GpuVectorizePipelines,
    px_buf: &gpu::Buffer,
    graph_buf: &gpu::Buffer,
    graph_snapshot: &gpu::Buffer,
    valence_buf: &gpu::Buffer,
    pos_buf: &gpu::Buffer,
    nbr_buf: &gpu::Buffer,
    flag_buf: &gpu::Buffer,
    opt_out_buf: &gpu::Buffer,
    orig_pos_buf: &gpu::Buffer,
    opt_picard_buf: &gpu::Buffer,
    crossing_t_buf: &gpu::Buffer,
    img_w: u32,
    img_h: u32,
) -> gpu::Buffer {
    let graph_stride = 2 * img_w + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;
    let graph_size = (graph_stride * (2 * img_h + 1) * 4).max(4);
    let pos_size = (num_cps * 2 * 4).max(4);

    // Stage 1: Similarity graph (also writes per-pixel valence mask)
    {
        let cp = device
            .begin_compute_pass(
                cmd,
                &[],
                &[
                    gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(graph_buf)
                        .with_cycle(false),
                    gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(valence_buf)
                        .with_cycle(false),
                ],
            )
            .expect("sim pass");
        cp.bind_compute_pipeline(&pipelines.sim_graph);
        cp.bind_compute_storage_buffers(0, std::slice::from_ref(px_buf));
        #[repr(C)]
        struct U {
            img_w: u32,
            img_h: u32,
            graph_stride: u32,
            _p: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                img_w,
                img_h,
                graph_stride,
                _p: 0,
            },
        );
        cp.dispatch(img_w.div_ceil(16), img_h.div_ceil(16), 1);
        device.end_compute_pass(cp);
    }

    // Stage 2: Resolve crossings
    {
        let cp = device.begin_copy_pass(cmd).expect("graph copy");
        unsafe {
            let src = sdl3::sys::gpu::SDL_GPUBufferLocation {
                buffer: graph_buf.raw(),
                offset: 0,
            };
            let dst = sdl3::sys::gpu::SDL_GPUBufferLocation {
                buffer: graph_snapshot.raw(),
                offset: 0,
            };
            sdl3::sys::gpu::SDL_CopyGPUBufferToBuffer(cp.raw(), &src, &dst, graph_size, false);
        }
        device.end_copy_pass(cp);
    }
    {
        let cp = device
            .begin_compute_pass(
                cmd,
                &[],
                &[gpu::StorageBufferReadWriteBinding::new()
                    .with_buffer(graph_buf)
                    .with_cycle(false)],
            )
            .expect("resolve pass");
        cp.bind_compute_pipeline(&pipelines.resolve);
        cp.bind_compute_storage_buffers(0, &[graph_snapshot.clone(), valence_buf.clone()]);
        #[repr(C)]
        struct U {
            img_w: u32,
            img_h: u32,
            graph_stride: u32,
            _p: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                img_w,
                img_h,
                graph_stride,
                _p: 0,
            },
        );
        cp.dispatch(
            img_w.saturating_sub(1).div_ceil(16),
            img_h.saturating_sub(1).div_ceil(16),
            1,
        );
        device.end_compute_pass(cp);
    }

    // Stage 3: Cell graph
    {
        let cp = device
            .begin_compute_pass(
                cmd,
                &[],
                &[
                    gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(pos_buf)
                        .with_cycle(false),
                    gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(nbr_buf)
                        .with_cycle(false),
                    gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(flag_buf)
                        .with_cycle(false),
                ],
            )
            .expect("cell pass");
        cp.bind_compute_pipeline(&pipelines.cell_graph);
        cp.bind_compute_storage_buffers(0, std::slice::from_ref(graph_buf));
        #[repr(C)]
        struct U {
            img_w: u32,
            img_h: u32,
            graph_stride: u32,
            corners_w: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                img_w,
                img_h,
                graph_stride,
                corners_w,
            },
        );
        cp.dispatch(corners_w.div_ceil(16), corners_h.div_ceil(16), 1);
        device.end_compute_pass(cp);
    }

    // Save original positions before optimization
    {
        let cp = device.begin_copy_pass(cmd).expect("orig pos copy");
        unsafe {
            let src = sdl3::sys::gpu::SDL_GPUBufferLocation {
                buffer: pos_buf.raw(),
                offset: 0,
            };
            let dst = sdl3::sys::gpu::SDL_GPUBufferLocation {
                buffer: orig_pos_buf.raw(),
                offset: 0,
            };
            sdl3::sys::gpu::SDL_CopyGPUBufferToBuffer(cp.raw(), &src, &dst, pos_size, false);
        }
        device.end_copy_pass(cp);
    }

    // Stage 4: Optimizer — N outer iters of (Picard → grad).
    //
    // Per iter, Picard reads cur_in, writes opt_picard_buf (per-CP Newton
    // step). Grad reads opt_picard_buf, writes cur_out (-η · ∇E debias).
    // After both dispatches we ping-pong cur_in/cur_out so the next iter
    // sees the latest result as input. The pair converges to ∇E = 0 (the
    // exact local minimum) — Picard alone reaches a biased Gauss-Seidel
    // fixed point; the grad pass debiases it.
    //
    // The two shaders take different uniform layouts: picard_step reads
    // { num_nodes, pad x3 }, gradient_correction reads
    // { num_nodes, eta, max_step, pad }.
    #[repr(C)]
    struct PicardU {
        num_nodes: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    #[repr(C)]
    struct GradU {
        num_nodes: u32,
        eta: f32,
        max_step: f32,
        _pad0: u32,
    }
    let picard_uni = PicardU {
        num_nodes: num_cps,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let grad_uni = GradU {
        num_nodes: num_cps,
        eta: OPT_GRAD_ETA,
        max_step: OPT_GRAD_MAX_STEP,
        _pad0: 0,
    };
    let mut cur_in = pos_buf.clone();
    let mut cur_out = opt_out_buf.clone();
    for _ in 0..OPT_OUTER_PASSES {
        // Picard pass: cur_in + orig + nbr + flag → opt_picard_buf
        {
            let cp = device
                .begin_compute_pass(
                    cmd,
                    &[],
                    &[gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(opt_picard_buf)
                        .with_cycle(false)],
                )
                .expect("picard pass");
            cp.bind_compute_pipeline(&pipelines.picard);
            cp.bind_compute_storage_buffers(
                0,
                &[
                    cur_in.clone(),
                    orig_pos_buf.clone(),
                    nbr_buf.clone(),
                    flag_buf.clone(),
                ],
            );
            cmd.push_compute_uniform_data(0, &picard_uni);
            cp.dispatch(num_cps.div_ceil(256), 1, 1);
            device.end_compute_pass(cp);
        }
        // Grad pass: opt_picard_buf + orig + nbr + flag → cur_out
        {
            let cp = device
                .begin_compute_pass(
                    cmd,
                    &[],
                    &[gpu::StorageBufferReadWriteBinding::new()
                        .with_buffer(&cur_out)
                        .with_cycle(false)],
                )
                .expect("grad pass");
            cp.bind_compute_pipeline(&pipelines.grad);
            cp.bind_compute_storage_buffers(
                0,
                &[
                    opt_picard_buf.clone(),
                    orig_pos_buf.clone(),
                    nbr_buf.clone(),
                    flag_buf.clone(),
                ],
            );
            cmd.push_compute_uniform_data(0, &grad_uni);
            cp.dispatch(num_cps.div_ceil(256), 1, 1);
            device.end_compute_pass(cp);
        }
        std::mem::swap(&mut cur_in, &mut cur_out);
    }
    let optimized_pos = cur_in;

    // Stage 5a: T-junction stem CP snap. Dispatch 3× for convergence when
    // stem CPs are also neighbors of other T-junctions.
    for _ in 0..3 {
        let cp = device
            .begin_compute_pass(
                cmd,
                &[],
                &[gpu::StorageBufferReadWriteBinding::new()
                    .with_buffer(&optimized_pos)
                    .with_cycle(false)],
            )
            .expect("tjunc pass");
        cp.bind_compute_pipeline(&pipelines.tjunction);
        cp.bind_compute_storage_buffers(0, &[nbr_buf.clone(), flag_buf.clone()]);
        #[repr(C)]
        struct U {
            num_nodes: u32,
            _p0: u32,
            _p1: u32,
            _p2: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                num_nodes: num_cps,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            },
        );
        cp.dispatch(num_cps.div_ceil(256), 1, 1);
        device.end_compute_pass(cp);
    }

    // Stage 5b: Crossing intersection pack. Writes (t_NS, t_EW) to the
    // crossing_t buffer. Reads finalized positions, so must run after the
    // tjunction snap loop. The rasterizer pass (next) declares crossing_t_buf
    // as a RW binding too — that's what triggers SDL3's automatic
    // buffer-write→buffer-read barrier; pure RO bindings don't.
    {
        let cp = device
            .begin_compute_pass(
                cmd,
                &[],
                &[gpu::StorageBufferReadWriteBinding::new()
                    .with_buffer(crossing_t_buf)
                    .with_cycle(false)],
            )
            .expect("xpack pass");
        cp.bind_compute_pipeline(&pipelines.crossing_pack);
        cp.bind_compute_storage_buffers(
            0,
            &[nbr_buf.clone(), flag_buf.clone(), optimized_pos.clone()],
        );
        #[repr(C)]
        struct U {
            num_nodes: u32,
            _p0: u32,
            _p1: u32,
            _p2: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                num_nodes: num_cps,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            },
        );
        cp.dispatch(num_cps.div_ceil(256), 1, 1);
        device.end_compute_pass(cp);
    }

    optimized_pos
}

/// Run the full GPU vectorize pipeline and blit to window.
/// Called every frame from the emulator's render loop. Uses cached GPU
/// buffers for zero per-frame allocation. No CPU computation after the
/// initial pixel upload — all stages run on GPU back-to-back.
pub fn gpu_vectorize_full_pipeline(
    device: &gpu::Device,
    window: &sdl3::video::Window,
    gpu_tex: &gpu::Texture<'static>,
    pipelines: &mut GpuVectorizePipelines,
    pixels: &[u32],
    img_w: u32,
    img_h: u32,
    out_w: u32,
    out_h: u32,
    scale: f32,
) {
    let graph_stride = 2 * img_w + 1;
    let graph_h = 2 * img_h + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;
    let px_size = img_w * img_h * 4;
    let graph_size = (graph_stride * graph_h * 4).max(4);
    let pos_size = (num_cps * 2 * 4).max(4);

    // Ensure GPU buffers are cached (allocated once, reused each frame)
    if pipelines
        .buf_cache
        .as_ref()
        .is_none_or(|c| c.img_w != img_w || c.img_h != img_h)
    {
        let rw = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ
            | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE;
        let ro = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ;
        let nbr_size = (num_cps * 4 * 4).max(4);
        let flag_size = (num_cps * 4).max(4);
        let crossing_t_size = (num_cps * 4).max(4);
        let valence_size = (img_w * img_h * 4).max(4);
        pipelines.buf_cache = Some(CellRastBufCache {
            img_w,
            img_h,
            px_buf: device
                .create_buffer()
                .with_usage(ro)
                .with_size(px_size)
                .build()
                .expect("px buf"),
            graph_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(graph_size)
                .build()
                .expect("graph buf"),
            graph_snapshot: device
                .create_buffer()
                .with_usage(ro)
                .with_size(graph_size)
                .build()
                .expect("graph snapshot"),
            valence_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(valence_size)
                .build()
                .expect("valence buf"),
            pos_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(pos_size)
                .build()
                .expect("pos buf"),
            nbr_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(nbr_size)
                .build()
                .expect("nbr buf"),
            flag_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(flag_size)
                .build()
                .expect("flag buf"),
            opt_out_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(pos_size)
                .build()
                .expect("opt out buf"),
            orig_pos_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(pos_size)
                .build()
                .expect("orig pos buf"),
            crossing_t_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(crossing_t_size)
                .build()
                .expect("crossing_t buf"),
            opt_picard_buf: device
                .create_buffer()
                .with_usage(rw)
                .with_size(pos_size)
                .build()
                .expect("opt_picard buf"),
            px_xfer: device
                .create_transfer_buffer()
                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
                .with_size(px_size)
                .build()
                .expect("px xfer"),
        });
        eprintln!(
            "Cell rasterizer buffer cache allocated for {}x{}",
            img_w, img_h
        );
    }
    let b = pipelines.buf_cache.as_ref().unwrap();

    let cmd = device.acquire_command_buffer().expect("cmd buf");

    // Upload pixel data (reuse cached transfer buffer)
    {
        let mut map = b.px_xfer.map::<u8>(device, true);
        let bytes =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        map.mem_mut()[..bytes.len()].copy_from_slice(bytes);
        map.unmap();
    }
    {
        let cp = device.begin_copy_pass(&cmd).expect("copy pass");
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&b.px_xfer),
            gpu::BufferRegion::new()
                .with_buffer(&b.px_buf)
                .with_size(px_size),
            false,
        );
        device.end_copy_pass(cp);
    }

    // Stages 1-5b: vectorize pipeline (shared with screenshot path)
    let optimized_pos = dispatch_stages_1_5b(
        device,
        &cmd,
        pipelines,
        &b.px_buf,
        &b.graph_buf,
        &b.graph_snapshot,
        &b.valence_buf,
        &b.pos_buf,
        &b.nbr_buf,
        &b.flag_buf,
        &b.opt_out_buf,
        &b.orig_pos_buf,
        &b.opt_picard_buf,
        &b.crossing_t_buf,
        img_w,
        img_h,
    );

    // Stage 6: Tile-based cell rasterizer (one workgroup per 2×2 source tile).
    //
    // SDL3 sync gotcha: crossing_t_buf was just RW-written by crossing_pack,
    // and we need the rasterizer's read of it to see those writes. SDL3 has
    // no explicit barrier primitive — the only sync point is the compute
    // pass boundary, AND the boundary only fences resources declared as
    // writeable bindings on `SDL_BeginGPUComputePass`. RO bindings via
    // `SDL_BindGPUComputeStorageBuffers` are *not* tracked for hazards, so
    // a buffer that's RW-written in pass A and only RO-read in pass B sits
    // outside the sync model and gets stale reads.
    //
    // The fix is to list `crossing_t_buf` in this pass's storage_buffer
    // _bindings (i.e., as a writeable binding) even though the shader only
    // reads it. SDL3 then sees the producer→consumer edge and emits the
    // backend barrier (Vulkan: VkBufferMemoryBarrier; Metal:
    // MTLComputeCommandEncoder dependency; D3D12: resource state transition).
    // The shader-side type still has to be `RWStructuredBuffer<float>` to
    // keep the slang→SPV/MSL/DXIL/WGSL emitters happy; we just never write.
    //
    // See SDL_gpu.h's `SDL_BeginGPUComputePass` (~line 3640): "A compute
    // pass is defined by a set of texture subresources and buffers that
    // may be written to" — and `SDL_BindGPUComputeStorageBuffers` (~line
    // 3747): "Binds storage buffers as readonly" — those two together
    // define the leaky tracking model.
    {
        let tiles_w = img_w.div_ceil(2);
        let tiles_h = img_h.div_ceil(2);
        let total_tiles = tiles_w * tiles_h;
        let cp = device
            .begin_compute_pass(
                &cmd,
                &[gpu::StorageTextureReadWriteBinding::new()
                    .with_texture(gpu_tex)
                    .with_cycle(true)],
                &[gpu::StorageBufferReadWriteBinding::new()
                    .with_buffer(&b.crossing_t_buf)
                    .with_cycle(false)],
            )
            .expect("rast pass");
        cp.bind_compute_pipeline(&pipelines.rasterizer);
        cp.bind_compute_storage_buffers(
            0,
            &[
                b.px_buf.clone(),
                optimized_pos.clone(),
                b.orig_pos_buf.clone(),
                b.flag_buf.clone(),
                b.nbr_buf.clone(),
            ],
        );
        #[repr(C)]
        struct U {
            img_w: u32,
            img_h: u32,
            out_w: u32,
            out_h: u32,
            scale: f32,
            corners_w: u32,
            tiles_w: u32,
            tiles_h: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                img_w,
                img_h,
                out_w,
                out_h,
                scale,
                corners_w,
                tiles_w,
                tiles_h,
            },
        );
        cp.dispatch(total_tiles, 1, 1);
        device.end_compute_pass(cp);
    }

    // Blit to swapchain
    let (swapchain_raw, sw_w, sw_h) = acquire_swapchain(&cmd, window);
    if !swapchain_raw.is_null() {
        let src_aspect = out_w as f32 / out_h as f32;
        let dst_aspect = sw_w as f32 / sw_h as f32;
        let (dx, dy, dw, dh) = if dst_aspect > src_aspect {
            let dh = sw_h;
            let dw = (sw_h as f32 * src_aspect) as u32;
            ((sw_w - dw) / 2, 0, dw, dh)
        } else {
            let dw = sw_w;
            let dh = (sw_w as f32 / src_aspect) as u32;
            (0, (sw_h - dh) / 2, dw, dh)
        };
        let mut blit_info = sdl3::sys::gpu::SDL_GPUBlitInfo::default();
        blit_info.source.texture = gpu_tex.raw();
        blit_info.source.w = out_w;
        blit_info.source.h = out_h;
        blit_info.destination.texture = swapchain_raw;
        blit_info.destination.x = dx;
        blit_info.destination.y = dy;
        blit_info.destination.w = dw;
        blit_info.destination.h = dh;
        blit_info.load_op = sdl3::sys::gpu::SDL_GPULoadOp::CLEAR;
        blit_info.filter = sdl3::sys::gpu::SDL_GPUFilter(gpu::Filter::Nearest as i32);
        unsafe {
            sdl3::sys::gpu::SDL_BlitGPUTexture(cmd.raw(), &blit_info);
        }
    }
    submit_and_sync(device, cmd, swapchain_raw.is_null());
}

/// Headless GPU full-pipeline screenshot (creates own device).
/// Used by the test_runner for offline vectorization. Creates a temporary
/// SDL context and GPU device, runs the full pipeline, and downloads the
/// result. Uses `dispatch_stages_1_5b()` shared with the live pipeline.
pub fn gpu_full_pipeline_screenshot(
    src: &[u32],
    src_w: usize,
    src_h: usize,
    scale: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    let img_w = src_w as u32;
    let img_h = src_h as u32;
    let out_w = (src_w * scale) as u32;
    let out_h = (src_h * scale) as u32;
    if out_w == 0 || out_h == 0 {
        return None;
    }

    let sdl = sdl3::init().ok()?;
    let video = sdl.video().ok()?;
    let window = video.window("gpu_full", 1, 1).hidden().build().ok()?;

    let all_formats = gpu::ShaderFormat::PRIVATE
        | gpu::ShaderFormat::SPIRV
        | gpu::ShaderFormat::MSL
        | gpu::ShaderFormat::DXBC
        | gpu::ShaderFormat::DXIL;
    let device = gpu::Device::new(all_formats, false)
        .ok()?
        .with_window(&window)
        .ok()?;

    let pipelines = init_full_gpu_pipeline(&device)?;

    let out_tex = device
        .create_texture(
            gpu::TextureCreateInfo::new()
                .with_type(gpu::TextureType::_2D)
                .with_format(gpu::TextureFormat::B8g8r8a8Unorm)
                .with_usage(gpu::TextureUsage::SAMPLER | gpu::TextureUsage::COMPUTE_STORAGE_WRITE)
                .with_width(out_w)
                .with_height(out_h)
                .with_layer_count_or_depth(1)
                .with_num_levels(1),
        )
        .ok()?;

    // Run the full pipeline into the offscreen texture, then download it.
    let cmd = device.acquire_command_buffer().ok()?;

    let graph_stride = 2 * img_w + 1;
    let graph_h = 2 * img_h + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;
    let rw =
        gpu::BufferUsageFlags::COMPUTE_STORAGE_READ | gpu::BufferUsageFlags::COMPUTE_STORAGE_WRITE;
    let ro = gpu::BufferUsageFlags::COMPUTE_STORAGE_READ;

    let px_size = img_w * img_h * 4;
    let graph_size = graph_stride * graph_h * 4;
    let pos_size = num_cps * 2 * 4;
    let nbr_size = num_cps * 4 * 4;
    let flag_size = num_cps * 4;

    // Upload pixels
    let px_xfer = device
        .create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::UPLOAD)
        .with_size(px_size)
        .build()
        .ok()?;
    {
        let mut map = px_xfer.map::<u8>(&device, true);
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        map.mem_mut()[..bytes.len()].copy_from_slice(bytes);
        map.unmap();
    }
    let px_buf = device
        .create_buffer()
        .with_usage(ro)
        .with_size(px_size)
        .build()
        .ok()?;
    let graph_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(graph_size.max(4))
        .build()
        .ok()?;
    let pos_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(pos_size.max(4))
        .build()
        .ok()?;
    let nbr_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(nbr_size.max(4))
        .build()
        .ok()?;
    let flag_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(flag_size.max(4))
        .build()
        .ok()?;
    let opt_out_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(pos_size.max(4))
        .build()
        .ok()?;

    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        cp.upload_to_gpu_buffer(
            gpu::TransferBufferLocation::new().with_transfer_buffer(&px_xfer),
            gpu::BufferRegion::new()
                .with_buffer(&px_buf)
                .with_size(px_size),
            false,
        );
        device.end_copy_pass(cp);
    }

    let graph_snapshot = device
        .create_buffer()
        .with_usage(ro)
        .with_size(graph_size.max(4))
        .build()
        .ok()?;
    let valence_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size((img_w * img_h * 4).max(4))
        .build()
        .ok()?;
    let orig_pos_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(pos_size.max(4))
        .build()
        .ok()?;
    let crossing_t_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size((num_cps * 4).max(4))
        .build()
        .ok()?;
    let opt_picard_buf = device
        .create_buffer()
        .with_usage(rw)
        .with_size(pos_size.max(4))
        .build()
        .ok()?;

    // Stages 1-5b: shared vectorize pipeline dispatch
    let optimized_pos = dispatch_stages_1_5b(
        &device,
        &cmd,
        &pipelines,
        &px_buf,
        &graph_buf,
        &graph_snapshot,
        &valence_buf,
        &pos_buf,
        &nbr_buf,
        &flag_buf,
        &opt_out_buf,
        &orig_pos_buf,
        &opt_picard_buf,
        &crossing_t_buf,
        img_w,
        img_h,
    );

    // Tile-based rasterizer: one workgroup per 2×2 source tile
    {
        let tiles_w = img_w.div_ceil(2);
        let tiles_h = img_h.div_ceil(2);
        let total_tiles = tiles_w * tiles_h;
        let cp = device
            .begin_compute_pass(
                &cmd,
                &[gpu::StorageTextureReadWriteBinding::new()
                    .with_texture(&out_tex)
                    .with_cycle(true)],
                &[gpu::StorageBufferReadWriteBinding::new()
                    .with_buffer(&crossing_t_buf)
                    .with_cycle(false)],
            )
            .ok()?;
        cp.bind_compute_pipeline(&pipelines.rasterizer);
        cp.bind_compute_storage_buffers(
            0,
            &[
                px_buf.clone(),
                optimized_pos.clone(),
                orig_pos_buf.clone(),
                flag_buf.clone(),
                nbr_buf.clone(),
            ],
        );
        #[repr(C)]
        struct U {
            iw: u32,
            ih: u32,
            ow: u32,
            oh: u32,
            s: f32,
            cw: u32,
            tw: u32,
            th: u32,
        }
        cmd.push_compute_uniform_data(
            0,
            &U {
                iw: img_w,
                ih: img_h,
                ow: out_w,
                oh: out_h,
                s: scale as f32,
                cw: corners_w,
                tw: tiles_w,
                th: tiles_h,
            },
        );
        cp.dispatch(total_tiles, 1, 1);
        device.end_compute_pass(cp);
    }

    // Download from texture
    let dl_buf = device
        .create_transfer_buffer()
        .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
        .with_size(out_w * out_h * 4)
        .build()
        .ok()?;
    {
        let cp = device.begin_copy_pass(&cmd).ok()?;
        unsafe {
            let src_r = sdl3::sys::gpu::SDL_GPUTextureRegion {
                texture: out_tex.raw(),
                w: out_w,
                h: out_h,
                d: 1,
                ..Default::default()
            };
            let dst = sdl3::sys::gpu::SDL_GPUTextureTransferInfo {
                transfer_buffer: dl_buf.raw(),
                ..Default::default()
            };
            sdl3::sys::gpu::SDL_DownloadFromGPUTexture(cp.raw(), &src_r, &dst);
        }
        device.end_copy_pass(cp);
    }

    let fence = cmd.submit_and_acquire_fence(&device).ok()?;
    device.wait_fences(true, &[fence]).ok()?;

    let map = dl_buf.map::<u8>(&device, false);
    let bytes = map.mem();
    let mut pixels = vec![0u32; (out_w * out_h) as usize];
    for i in 0..pixels.len() {
        let off = i * 4;
        let b = bytes[off] as u32;
        let g = bytes[off + 1] as u32;
        let r = bytes[off + 2] as u32;
        pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
    }
    map.unmap();
    Some((pixels, out_w, out_h))
}
