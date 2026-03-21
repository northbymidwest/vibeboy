//! Shared utility functions used by multiple UI frontends (main.rs, cocoa_ui.rs, winit_ui.rs)
//! and the test runner.

use crate::model::GbModel;
use crate::scaling;
use std::time::Duration;

/// Target frame time: 70224 T-cycles / cpu_clock_rate.
/// Standard: ~16.74ms (~59.73 fps). SGB1: ~16.35ms (~61.17 fps).
pub fn frame_duration(model: GbModel) -> Duration {
    let nanos = 70_224u64 * 1_000_000_000 / model.cpu_clock_rate() as u64;
    Duration::from_nanos(nanos)
}

/// Parse a model string for clap value_parser.
pub fn parse_model(s: &str) -> Result<GbModel, String> {
    s.parse::<GbModel>()
}

/// Parse a filter string for clap value_parser.
pub fn parse_filter(s: &str) -> Result<String, String> {
    scaling::ScaleFilter::validate_name(s)
}

/// Auto-detect hardware model from ROM header CGB flag.
pub fn auto_detect_model(rom: &[u8]) -> GbModel {
    let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
    if cgb_flag == 0x80 || cgb_flag == 0xC0 {
        GbModel::Cgb
    } else {
        GbModel::Dmg
    }
}

/// Reverse stereo interleaved audio in-place.
/// Swaps stereo sample pairs so the audio plays backward.
pub fn reverse_audio(samples: &mut [f32]) {
    let n = samples.len() / 2;
    for i in 0..n / 2 {
        let j = n - 1 - i;
        samples.swap(i * 2, j * 2);
        samples.swap(i * 2 + 1, j * 2 + 1);
    }
}

/// Apply a short cosine fade-in to the start of each frame's audio to
/// suppress transient pops from cold APU filter state on snapshot restore.
/// `frame_len` is the number of stereo sample pairs per frame.
pub fn fade_frame_boundaries(samples: &mut [f32], frame_len: usize) {
    if frame_len == 0 { return; }
    // Fade the first ~64 stereo pairs of each frame
    let fade_len = 64.min(frame_len);
    let mut offset = 0;
    while offset + frame_len * 2 <= samples.len() {
        for i in 0..fade_len {
            let t = i as f32 / fade_len as f32;
            // Cosine ease-in: 0 at start, 1 at fade_len
            let gain = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            samples[offset + i * 2] *= gain;
            samples[offset + i * 2 + 1] *= gain;
        }
        offset += frame_len * 2;
    }
}

/// Downsample stereo interleaved audio by an integer factor using a
/// Blackman-windowed sinc FIR low-pass filter followed by decimation.
///
/// The filter removes frequencies above the new Nyquist (original_rate / 2*factor)
/// before decimation, preventing aliasing artifacts.
///
/// Input/output: interleaved [L, R, L, R, ...].
pub fn downsample_audio(samples: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 || samples.len() < 2 {
        return samples.to_vec();
    }

    // Precompute FIR kernel: Blackman-windowed sinc, 33 taps
    const HALF_TAPS: i32 = 16;
    const TAPS: usize = (HALF_TAPS * 2 + 1) as usize;
    let cutoff = 1.0 / (2.0 * factor as f64);
    let mut kernel = [0.0f32; TAPS];
    let mut sum = 0.0f64;
    for i in 0..TAPS {
        let n = i as f64 - HALF_TAPS as f64;
        let sinc = if n.abs() < 1e-10 {
            2.0 * std::f64::consts::PI * cutoff
        } else {
            (2.0 * std::f64::consts::PI * cutoff * n).sin() / n
        };
        let t = i as f64 / (TAPS - 1) as f64;
        let window = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()
                          + 0.08 * (4.0 * std::f64::consts::PI * t).cos();
        let v = sinc * window;
        kernel[i] = v as f32;
        sum += v;
    }
    let inv = 1.0 / sum as f32;
    for k in &mut kernel { *k *= inv; }

    let stereo_frames = samples.len() / 2;
    let out_frames = stereo_frames / factor;
    let mut out = Vec::with_capacity(out_frames * 2);

    for i in 0..out_frames {
        let center = i * factor;
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for (j, &k) in kernel.iter().enumerate() {
            let src = (center as i32 + j as i32 - HALF_TAPS)
                .clamp(0, stereo_frames as i32 - 1) as usize;
            l += samples[src * 2] * k;
            r += samples[src * 2 + 1] * k;
        }
        out.push(l);
        out.push(r);
    }

    out
}

