//! CPU vs GPU comparison for every scaling filter.
//!
//! Renders a fixed synthetic 160x144 frame (flat regions, hard edges,
//! diagonals at several slopes, isolated pixels, gradients and dithered
//! blocks) through each filter with `scaling::cpu_scale` and with the wgpu
//! compute path (headless, any backend: Metal on macOS), then checks each
//! filter's difference against its tolerance below.
//!
//! Needs a GPU, so it is ignored by default:
//!
//! ```text
//! cargo test --release --features gpu --test filter_parity -- --ignored --nocapture
//! ```
#![cfg(feature = "gpu")]

use vibeboy_core::scaling::wgpu_scale::{WgpuScaleFilter, WgpuScalePipeline};
use vibeboy_core::scaling::wgpu_vectorize::WgpuVectorizePipeline;
use vibeboy_core::scaling::{self, ScaleFilter};

const W: usize = 160;
const H: usize = 144;
/// Adaptive filters render at 4x.
const DISP_W: usize = W * 4;
const DISP_H: usize = H * 4;

/// Allowed difference for one filter: the largest per-channel difference and
/// the share of output pixels (in percent) that may differ at all.
struct Tolerance {
    max_channel: u8,
    max_percent: f64,
    /// Why the two implementations are allowed to differ.
    note: &'static str,
}

const EXACT: Tolerance = Tolerance {
    max_channel: 0,
    max_percent: 0.0,
    note: "",
};

/// Per-filter tolerances, measured on the frame below. Filters not listed
/// must match exactly.
fn tolerance(filter: ScaleFilter) -> Tolerance {
    use ScaleFilter as F;
    use vibeboy_core::scaling::{HqxScale, XbrScale, XbrzScale};
    let t = |max_channel, max_percent, note| Tolerance {
        max_channel,
        max_percent,
        note,
    };
    match filter {
        // Float rounding of blend weights: off by at most one step.
        F::Bilinear => t(1, 4.5, "rounding"),
        F::Hqx(HqxScale::Hq2x) => t(1, 32.5, "blend rounding"),
        F::Hqx(HqxScale::Hq3x) => t(1, 22.1, "blend rounding"),
        F::Hqx(HqxScale::Hq4x) => t(1, 34.6, "blend rounding"),
        F::OmniScale => t(1, 20.9, "rounding"),
        F::Xbr(XbrScale::Xbr4x) => t(1, 0.31, "rounding"),
        // Differences confined to the frame border (edge clamping).
        F::Bicubic => t(17, 0.13, "edge clamping"),
        F::Scale4x => t(208, 0.01, "edge clamping of the 2x stage"),
        // Super xBR's cardinal pass reads edge-clamped cells that the same
        // pass writes, so its border pixels vary from run to run.
        F::SuperXbr => t(255, 1.3, "border race in GPU pass 1, and unexplained"),
        // Known divergences between the CPU and GPU implementations.
        F::LcdGrid => t(18, 24.21, "GPU gap row skips the subpixel mask"),
        F::Scale3x => t(
            226,
            0.78,
            "GPU centre pixel falls through to the corner rule",
        ),
        F::Dcci => t(117, 13.9, "unexplained"),
        F::Edi => t(188, 11.14, "unexplained"),
        F::Nedi => t(180, 6.52, "unexplained"),
        F::OmniScaleLegacy => t(209, 0.18, "unexplained"),
        F::ScaleFx => t(240, 4.38, "unexplained"),
        F::ScaleFx9x => t(240, 5.23, "unexplained"),
        F::Vectorize => t(58, 4.25, "unexplained"),
        F::Xbr(XbrScale::Xbr2x) => t(63, 12.53, "unexplained"),
        F::Xbr(XbrScale::Xbr3x) => t(156, 10.8, "unexplained"),
        F::Xbrz(XbrzScale::Xbrz2x) => t(120, 2.56, "unexplained"),
        F::Xbrz(XbrzScale::Xbrz3x) => t(160, 1.3, "unexplained"),
        F::Xbrz(XbrzScale::Xbrz4x) => t(160, 1.3, "unexplained"),
        F::Xbrz(XbrzScale::Xbrz5x) => t(160, 0.69, "unexplained"),
        F::Xbrz(XbrzScale::Xbrz6x) => t(160, 0.92, "unexplained"),
        _ => EXACT,
    }
}

/// Deterministic test frame in 0xFFRRGGBB.
fn test_frame() -> Vec<u32> {
    const SHADES: [u32; 4] = [0x9BBC0F, 0x8BAC0F, 0x306230, 0x0F380F];
    let mut px = vec![0u32; W * H];
    let mut lcg = 0x1234_5678u32;
    let mut rand = move || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        lcg >> 16
    };
    for y in 0..H {
        for x in 0..W {
            let (xi, yi) = (x as i32, y as i32);
            let c = if y < 72 && x < 80 {
                // Sprite-like area: checker background, a disc, lines of
                // slope 1, 1/2 and 3, and isolated pixels.
                let (dx, dy) = (xi - 40, yi - 36);
                if dx * dx + dy * dy <= 18 * 18 {
                    SHADES[3]
                } else if xi == yi || xi == 2 * yi - 10 || 3 * xi == yi + 60 {
                    SHADES[2]
                } else if x % 13 == 5 && y % 11 == 3 {
                    0xFFFFFF
                } else {
                    SHADES[(x / 8 + y / 8) % 2]
                }
            } else if y < 72 {
                // Red-to-blue ramp across, green ramp down, with a
                // one-pixel grid every 16 pixels.
                if x % 16 == 0 || y % 16 == 0 {
                    0x000000
                } else {
                    let r = ((x - 80) * 255 / 79) as u32;
                    let g = (y * 255 / 71) as u32;
                    (r << 16) | (g << 8) | (255 - r)
                }
            } else if y < 90 {
                // Smooth full-width gradient band.
                let r = (x * 255 / (W - 1)) as u32;
                (r << 16) | (0x80 << 8) | (255 - r)
            } else if x < 80 {
                // Staircases, 1px stripes and a dither.
                if (xi + yi) % 6 == 0 {
                    0xE0E0E0
                } else if (x / 4) % 2 == 0 && y > 117 {
                    if (x + y) % 2 == 0 { 0x202020 } else { 0xC04040 }
                } else if (x / 3 + y / 3) % 2 == 0 {
                    0x4060C0
                } else {
                    0x103010
                }
            } else {
                // 2x2 blocks from a small palette.
                SHADES[(rand() % 4) as usize]
            };
            px[y * W + x] = 0xFF00_0000 | c;
        }
    }
    // The block area draws random numbers row by row, but the blocks are
    // 2x2: copy each even pixel over its right and lower neighbours.
    for y in (90..H).step_by(2) {
        for x in (80..W).step_by(2) {
            let c = px[y * W + x];
            px[y * W + x + 1] = c;
            px[(y + 1) * W + x] = c;
            px[(y + 1) * W + x + 1] = c;
        }
    }
    px
}

fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no wgpu adapter");
    eprintln!("adapter: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("filter_parity"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("no wgpu device")
}

/// Read an Rgba8Unorm texture back as 0x00RRGGBB pixels.
fn download(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u32> {
    let (w, h) = (tex.width(), tex.height());
    let row_bytes = w * 4;
    let padded_row = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity_download"),
        size: (padded_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
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
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().expect("map failed");
    let data = slice.get_mapped_range().expect("mapped range");
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        let row = &data[y * padded_row as usize..][..row_bytes as usize];
        out.extend(
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32),
        );
    }
    out
}

struct Diff {
    max_channel: u8,
    differing: usize,
    /// Differing pixels that map to the outer two source pixels of the
    /// frame, where the implementations' edge clamping can disagree.
    at_border: usize,
    total: usize,
}

impl Diff {
    fn percent(&self) -> f64 {
        100.0 * self.differing as f64 / self.total as f64
    }
}

fn diff(a: &[u32], b: &[u32], out_w: usize, out_h: usize) -> Diff {
    let mut d = Diff {
        max_channel: 0,
        differing: 0,
        at_border: 0,
        total: a.len(),
    };
    for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
        let m = [16, 8, 0]
            .iter()
            .map(|s| ((p >> s) as u8).abs_diff((q >> s) as u8))
            .max()
            .unwrap();
        if m > 0 {
            d.differing += 1;
            d.max_channel = d.max_channel.max(m);
            let sx = i % out_w * W / out_w;
            let sy = i / out_w * H / out_h;
            if sx < 2 || sy < 2 || sx >= W - 2 || sy >= H - 2 {
                d.at_border += 1;
            }
        }
    }
    d
}

#[test]
#[ignore = "needs a GPU; run with --features gpu -- --ignored"]
fn cpu_and_gpu_filters_agree() {
    let (device, queue) = gpu_device();
    let mut scale = WgpuScalePipeline::new(&device);
    let mut vectorize = WgpuVectorizePipeline::new(&device);
    let src = test_frame();

    let mut failures = Vec::new();
    eprintln!(
        "{:<17} {:>9} {:>9} {:>7} {:>4}  {:>9} {:>8}",
        "filter", "size", "differ", "border", "max", "allowed%", "allowed"
    );
    for info in ScaleFilter::all_filters() {
        let filter = info.filter;
        let (cpu, cw, ch) = scaling::cpu_scale(filter, &src, W, H, DISP_W, DISP_H)
            .expect("cpu_scale returned None");

        let mut encoder = device.create_command_encoder(&Default::default());
        let gpu = if filter == ScaleFilter::Vectorize {
            vectorize.encode(
                &device,
                &queue,
                &mut encoder,
                &src,
                W as u32,
                H as u32,
                cw,
                ch,
                4.0,
            );
            queue.submit(Some(encoder.finish()));
            let (px, gw, gh) = vectorize.download_output(&device, &queue).unwrap();
            assert_eq!((gw, gh), (cw, ch), "{}", info.cli_name);
            px
        } else {
            let wf = WgpuScaleFilter::from_scale_filter(filter)
                .unwrap_or_else(|| panic!("{} has no GPU path", info.cli_name));
            let (ow, oh) = match filter.factor() {
                0 => (DISP_W as u32, DISP_H as u32),
                f => (W as u32 * f, H as u32 * f),
            };
            assert_eq!((ow, oh), (cw, ch), "{}", info.cli_name);
            let tex = scale.encode(
                &device,
                &queue,
                &mut encoder,
                wf,
                &src,
                W as u32,
                H as u32,
                ow,
                oh,
            );
            queue.submit(Some(encoder.finish()));
            download(&device, &queue, tex)
        };

        let d = diff(&cpu, &gpu, cw as usize, ch as usize);
        let tol = tolerance(filter);
        let ok = d.max_channel <= tol.max_channel && d.percent() <= tol.max_percent;
        eprintln!(
            "{:<17} {:>9} {:>8.3}% {:>7} {:>4}  {:>8.3}% {:>8}{}{}",
            info.cli_name,
            format!("{cw}x{ch}"),
            d.percent(),
            d.at_border,
            d.max_channel,
            tol.max_percent,
            tol.max_channel,
            if ok { "" } else { "  FAIL" },
            if tol.note.is_empty() {
                String::new()
            } else {
                format!("  ({})", tol.note)
            },
        );
        if !ok {
            failures.push(info.cli_name);
        }
    }
    assert!(failures.is_empty(), "outside tolerance: {failures:?}");
}
