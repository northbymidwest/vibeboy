//! Pixel-art scaling algorithms.

pub mod aa_nearest;
pub mod bicubic;
pub mod bilinear;
pub mod dcci;
pub mod eagle;
pub mod edi;
pub mod epx;
pub mod hqx;
pub mod nedi;
pub mod omniscale;
pub mod omniscale_legacy;
pub mod sai;
pub mod scale3x;
pub mod super_xbr;
pub mod xbr;
pub mod xbrz;
#[cfg(feature = "sdl3-gpu-shaders")]
pub mod gpu;
#[cfg(feature = "sdl3-gpu-shaders")]
pub mod gpu_pipelines;
#[cfg(any(feature = "gpu", feature = "winit-ui"))]
pub mod wgpu_vectorize;

/// Sample a pixel with clamped coordinates.
#[inline(always)]
fn get(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let cx = x.clamp(0, w as isize - 1) as usize;
    let cy = y.clamp(0, h as isize - 1) as usize;
    src[cy * w + cx]
}

/// Weighted color distance in YCbCr-like space.
/// Shared by xBR, xBRZ, xBR-Hybrid, and Super xBR.
#[inline(always)]
fn color_dist(a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    let ar = ((a >> 16) & 0xFF) as f32;
    let ag = ((a >> 8) & 0xFF) as f32;
    let ab = (a & 0xFF) as f32;
    let br = ((b >> 16) & 0xFF) as f32;
    let bg = ((b >> 8) & 0xFF) as f32;
    let bb = (b & 0xFF) as f32;

    let dr = ar - br;
    let dg = ag - bg;
    let db = ab - bb;

    let dy = 0.299 * dr + 0.587 * dg + 0.114 * db;
    let dcb = -0.169 * dr - 0.331 * dg + 0.500 * db;
    let dcr = 0.500 * dr - 0.419 * dg - 0.081 * db;

    48.0 * dy * dy + 7.0 * dcb * dcb + 6.0 * dcr * dcr
}

/// Blend two ARGB colors with weight alpha (0.0 = all a, 1.0 = all b).
#[inline(always)]
fn blend_argb(a: u32, b: u32, alpha: f32) -> u32 {
    if alpha <= 0.0 { return a; }
    if alpha >= 1.0 { return b; }
    let inv = 1.0 - alpha;
    let r = (((a >> 16) & 0xFF) as f32 * inv + ((b >> 16) & 0xFF) as f32 * alpha).round() as u32;
    let g = (((a >> 8) & 0xFF) as f32 * inv + ((b >> 8) & 0xFF) as f32 * alpha).round() as u32;
    let bl = ((a & 0xFF) as f32 * inv + (b & 0xFF) as f32 * alpha).round() as u32;
    0xFF000000 | (r.min(255) << 16) | (g.min(255) << 8) | bl.min(255)
}

/// Extract RGB channels from a packed ARGB pixel as `[R, G, B]` floats.
#[inline(always)]
fn channels(c: u32) -> [f32; 3] {
    [
        ((c >> 16) & 0xFF) as f32,
        ((c >> 8) & 0xFF) as f32,
        (c & 0xFF) as f32,
    ]
}

/// Pack floating-point RGB channels into an ARGB u32 (alpha = 0xFF).
#[inline(always)]
fn pack_channels(ch: [f32; 3]) -> u32 {
    let r = ch[0].round().clamp(0.0, 255.0) as u32;
    let g = ch[1].round().clamp(0.0, 255.0) as u32;
    let b = ch[2].round().clamp(0.0, 255.0) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b
}

/// YCbCr color distance using ITU-R BT.2020 conversion.
/// Used by xBRZ; also suitable for any perceptual color comparison.
#[inline(always)]
fn color_dist_bt2020(a: u32, b: u32) -> f32 {
    if a == b { return 0.0; }
    let dr = ((a >> 16) & 0xFF) as f32 - ((b >> 16) & 0xFF) as f32;
    let dg = ((a >> 8) & 0xFF) as f32 - ((b >> 8) & 0xFF) as f32;
    let db = (a & 0xFF) as f32 - (b & 0xFF) as f32;
    const K_R: f32 = 0.2627;
    const K_G: f32 = 0.6780;
    const K_B: f32 = 0.0593;
    const S_B: f32 = 0.5 / (1.0 - K_B);
    const S_R: f32 = 0.5 / (1.0 - K_R);
    let y = K_R * dr + K_G * dg + K_B * db;
    let cb = S_B * (db - y);
    let cr = S_R * (dr - y);
    (y * y + cb * cb + cr * cr).sqrt()
}

/// HQx scaling factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HqxScale {
    Hq2x,
    Hq3x,
    Hq4x,
}

impl HqxScale {
    pub fn factor(self) -> u32 {
        match self {
            HqxScale::Hq2x => 2,
            HqxScale::Hq3x => 3,
            HqxScale::Hq4x => 4,
        }
    }
}

pub use xbr::XbrScale;
pub use xbrz::XbrzScale;

/// Scaling filter for the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleFilter {
    Nearest,
    Bilinear,
    Bicubic,
    Hqx(HqxScale),
    Epx,
    Scale2x,
    Scale3x,
    Scale4x,
    Eagle,
    Sai2x,
    Super2xSai,
    SuperEagle,
    Xbr(XbrScale),
    Xbrz(XbrzScale),
    SuperXbr,
    Nedi,
    Dcci,
    Edi,
    /// Arbitrary-resolution OmniScale (scales to display size).
    OmniScale,
    /// Arbitrary-resolution OmniScale legacy variant (scales to display size).
    OmniScaleLegacy,
    /// Anti-aliased nearest neighbor (scales to display size).
    AaNearestNeighbor,
    /// Kopf-Lischinski vectorization with full B-spline optimization (legacy scanline rasterizer).
    VectorizeLegacy,
    /// Legacy Kopf-Lischinski vectorization — adaptive fast path (skips B-spline
    /// optimization, uses straight line segments at boundary junctions).
    VectorizeLegacyAdaptive,
    /// Kopf-Lischinski vectorization with Gaussian diffusion rendering
    /// (Paper Section 3.5). Uses truncated Gaussians at cell centroids
    /// instead of scanline-filled vector paths.
    VectorizeDiffusion,
    /// Paper's full rendering: B-spline contour boundaries + Gaussian diffusion.
    VectorizeSplineDiffusion,
    /// Adaptive spline-diffusion: skips B-spline optimization on complex frames.
    VectorizeSplineDiffusionAdaptive,
    /// Shared-chain vectorization: gap-free rendering using shared boundary spans.
    Vectorize,
    /// Adaptive shared-chain: skips optimization on complex frames.
    VectorizeAdaptive,
    /// Full GPU vectorize: all pipeline stages run on GPU compute shaders.
    VectorizeGpu,
}

impl ScaleFilter {
    /// All valid filter name strings for CLI validation.
    pub const ALL_NAMES: &[&str] = &[
        "nearest", "none", "bilinear", "bicubic", "epx", "scale2x", "scale3x", "scale4x", "eagle",
        "2xsai", "super-2xsai", "super-eagle",
        "hq2x", "hq3x", "hq4x", "xbr2x", "xbr3x", "xbr4x",
        "xbrz2x", "xbrz3x", "xbrz4x", "xbrz5x", "xbrz6x",
        "super-xbr", "nedi", "dcci", "edi",
        "omniscale", "omniscale-legacy",
        "aa-nearest", "vectorize", "vectorize-adaptive",
        "vectorize-gpu",
        "vectorize-legacy", "vectorize-legacy-adaptive", "vectorize-diffusion",
        "vectorize-spline-diffusion", "vectorize-spline-diffusion-adaptive",
    ];

    /// Parse a filter name string into a ScaleFilter enum variant.
    /// Returns None for unrecognized names.
    pub fn from_name(s: &str) -> Option<ScaleFilter> {
        Some(match s {
            "nearest" | "none" => ScaleFilter::Nearest,
            "bilinear" => ScaleFilter::Bilinear,
            "bicubic" => ScaleFilter::Bicubic,
            "epx" => ScaleFilter::Epx,
            "scale2x" => ScaleFilter::Scale2x,
            "scale3x" => ScaleFilter::Scale3x,
            "scale4x" => ScaleFilter::Scale4x,
            "eagle" => ScaleFilter::Eagle,
            "2xsai" => ScaleFilter::Sai2x,
            "super-2xsai" => ScaleFilter::Super2xSai,
            "super-eagle" => ScaleFilter::SuperEagle,
            "hq2x" => ScaleFilter::Hqx(HqxScale::Hq2x),
            "hq3x" => ScaleFilter::Hqx(HqxScale::Hq3x),
            "hq4x" => ScaleFilter::Hqx(HqxScale::Hq4x),
            "xbr2x" => ScaleFilter::Xbr(XbrScale::Xbr2x),
            "xbr3x" => ScaleFilter::Xbr(XbrScale::Xbr3x),
            "xbr4x" => ScaleFilter::Xbr(XbrScale::Xbr4x),
            "xbrz2x" => ScaleFilter::Xbrz(XbrzScale::Xbrz2x),
            "xbrz3x" => ScaleFilter::Xbrz(XbrzScale::Xbrz3x),
            "xbrz4x" => ScaleFilter::Xbrz(XbrzScale::Xbrz4x),
            "xbrz5x" => ScaleFilter::Xbrz(XbrzScale::Xbrz5x),
            "xbrz6x" => ScaleFilter::Xbrz(XbrzScale::Xbrz6x),
            "super-xbr" => ScaleFilter::SuperXbr,
            "nedi" => ScaleFilter::Nedi,
            "dcci" => ScaleFilter::Dcci,
            "edi" => ScaleFilter::Edi,
            "omniscale" => ScaleFilter::OmniScale,
            "omniscale-legacy" => ScaleFilter::OmniScaleLegacy,
            "aa-nearest" => ScaleFilter::AaNearestNeighbor,
            "vectorize" => ScaleFilter::Vectorize,
            "vectorize-adaptive" => ScaleFilter::VectorizeAdaptive,
            "vectorize-gpu" => ScaleFilter::VectorizeGpu,
            "vectorize-legacy" => ScaleFilter::VectorizeLegacy,
            "vectorize-legacy-adaptive" => ScaleFilter::VectorizeLegacyAdaptive,
            "vectorize-diffusion" => ScaleFilter::VectorizeDiffusion,
            "vectorize-spline-diffusion" => ScaleFilter::VectorizeSplineDiffusion,
            "vectorize-spline-diffusion-adaptive" => ScaleFilter::VectorizeSplineDiffusionAdaptive,
            _ => return None,
        })
    }

    /// Validate a filter name string for CLI parsing.
    pub fn validate_name(s: &str) -> Result<String, String> {
        let lower = s.to_lowercase();
        if Self::ALL_NAMES.contains(&lower.as_str()) {
            Ok(lower)
        } else {
            Err(format!(
                "unknown filter '{}'\n  [possible values: {}]",
                s,
                Self::ALL_NAMES.join(", ")
            ))
        }
    }

    /// Returns the fixed scale factor for fixed-factor filters,
    /// or 1 for resolution-adaptive filters (which scale to window size).
    pub fn factor(self) -> u32 {
        match self {
            ScaleFilter::Nearest
            | ScaleFilter::Bilinear | ScaleFilter::Bicubic
            | ScaleFilter::OmniScale | ScaleFilter::OmniScaleLegacy
            | ScaleFilter::AaNearestNeighbor
            | ScaleFilter::VectorizeLegacy | ScaleFilter::VectorizeLegacyAdaptive
            | ScaleFilter::VectorizeDiffusion
            | ScaleFilter::VectorizeSplineDiffusion
            | ScaleFilter::VectorizeSplineDiffusionAdaptive
            | ScaleFilter::Vectorize
            | ScaleFilter::VectorizeAdaptive
            | ScaleFilter::VectorizeGpu => 1,
            ScaleFilter::Hqx(h) => h.factor(),
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Eagle
            | ScaleFilter::Sai2x | ScaleFilter::Super2xSai | ScaleFilter::SuperEagle => 2,
            ScaleFilter::Scale3x => 3,
            ScaleFilter::Scale4x => 4,
            ScaleFilter::Xbr(x) => x.factor(),
            ScaleFilter::Xbrz(x) => x.factor(),
            ScaleFilter::SuperXbr
            | ScaleFilter::Nedi | ScaleFilter::Dcci | ScaleFilter::Edi => 2,
        }
    }

    /// Whether the window should be freely resizable with this filter.
    pub fn is_resizable(self) -> bool {
        matches!(self,
            ScaleFilter::Nearest
            | ScaleFilter::Bilinear | ScaleFilter::Bicubic
            | ScaleFilter::OmniScale | ScaleFilter::OmniScaleLegacy
            | ScaleFilter::AaNearestNeighbor
            | ScaleFilter::VectorizeLegacy | ScaleFilter::VectorizeLegacyAdaptive
            | ScaleFilter::VectorizeDiffusion
            | ScaleFilter::VectorizeSplineDiffusion
            | ScaleFilter::VectorizeSplineDiffusionAdaptive
            | ScaleFilter::Vectorize
            | ScaleFilter::VectorizeAdaptive
            | ScaleFilter::VectorizeGpu)
    }

    /// Whether this filter produces output scaled to the display dimensions.
    /// Nearest is resizable but relies on GPU texture stretching instead.
    pub fn scales_to_display(self) -> bool {
        matches!(self,
            ScaleFilter::Bilinear | ScaleFilter::Bicubic
            | ScaleFilter::OmniScale | ScaleFilter::OmniScaleLegacy
            | ScaleFilter::AaNearestNeighbor
            | ScaleFilter::VectorizeLegacy | ScaleFilter::VectorizeLegacyAdaptive
            | ScaleFilter::VectorizeDiffusion
            | ScaleFilter::VectorizeSplineDiffusion
            | ScaleFilter::VectorizeSplineDiffusionAdaptive
            | ScaleFilter::Vectorize
            | ScaleFilter::VectorizeAdaptive
            | ScaleFilter::VectorizeGpu)
    }
}

/// Apply a CPU scaling filter to a frame buffer.
///
/// Returns `(scaled_pixels, output_width, output_height)`.
/// For filters that scale to a fixed factor, `disp_w`/`disp_h` are ignored.
/// For resolution-adaptive filters (Bilinear, Bicubic, OmniScale, etc.),
/// the output is sized to `disp_w` x `disp_h`.
///
/// Returns `None` for Nearest (which should use GPU blit instead).
pub fn cpu_scale(
    filter: ScaleFilter,
    src: &[u32], sw: usize, sh: usize,
    disp_w: usize, disp_h: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    Some(match filter {
        ScaleFilter::Hqx(mode) => {
            let s = hqx::scale(src, sw, sh, mode);
            let f = mode.factor() as u32;
            (s, sw as u32 * f, sh as u32 * f)
        }
        ScaleFilter::Epx | ScaleFilter::Scale2x => {
            let s = epx::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Scale3x => {
            let s = scale3x::scale(src, sw, sh);
            (s, sw as u32 * 3, sh as u32 * 3)
        }
        ScaleFilter::Scale4x => {
            let s = epx::scale4x(src, sw, sh);
            (s, sw as u32 * 4, sh as u32 * 4)
        }
        ScaleFilter::Eagle => {
            let s = eagle::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Sai2x => {
            let s = sai::scale_2xsai(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Super2xSai => {
            let s = sai::scale_super2xsai(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::SuperEagle => {
            let s = sai::scale_super_eagle(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Bilinear => {
            let s = bilinear::scale_to(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::Bicubic => {
            let s = bicubic::scale_to(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::Xbr(mode) => {
            let s = xbr::scale(src, sw, sh, mode);
            let f = mode.factor() as u32;
            (s, sw as u32 * f, sh as u32 * f)
        }
        ScaleFilter::Xbrz(mode) => {
            let s = xbrz::scale(src, sw, sh, mode);
            let f = mode.factor() as u32;
            (s, sw as u32 * f, sh as u32 * f)
        }
        ScaleFilter::SuperXbr => {
            let s = super_xbr::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Nedi => {
            let s = nedi::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Dcci => {
            let s = dcci::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::Edi => {
            let s = edi::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::OmniScale => {
            let s = omniscale::scale_to(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::OmniScaleLegacy => {
            let s = omniscale_legacy::scale_to(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::AaNearestNeighbor => {
            let s = aa_nearest::scale(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        // Nearest, Vectorize handled elsewhere (GPU blit / compute)
        _ => return None,
    })
}
