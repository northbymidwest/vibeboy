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
pub mod scale3x;
pub mod super_xbr;
pub mod xbr;
pub mod xbr_hybrid;
pub mod xbrz;

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

/// OmniScale factor (works at any integer scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OmniScaleFactor {
    X2,
    X3,
    X4,
    X5,
    X6,
}

impl OmniScaleFactor {
    pub fn factor(self) -> u32 {
        match self {
            OmniScaleFactor::X2 => 2,
            OmniScaleFactor::X3 => 3,
            OmniScaleFactor::X4 => 4,
            OmniScaleFactor::X5 => 5,
            OmniScaleFactor::X6 => 6,
        }
    }
}

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
    Eagle,
    Xbr(XbrScale),
    Xbrz(XbrzScale),
    XbrHybrid,
    SuperXbr,
    Nedi,
    Dcci,
    Edi,
    /// Arbitrary-resolution OmniScale (scales to display size).
    OmniScale,
    /// Fixed-factor OmniScale legacy variant.
    OmniScaleLegacy(OmniScaleFactor),
    /// Anti-aliased nearest neighbor (scales to display size).
    AaNearestNeighbor,
    /// Kopf-Lischinski vectorization (scales to display size).
    Vectorize,
}

impl ScaleFilter {
    /// Returns the fixed scale factor for fixed-factor filters,
    /// or 1 for resolution-adaptive filters (which scale to window size).
    pub fn factor(self) -> u32 {
        match self {
            ScaleFilter::Nearest
            | ScaleFilter::Bilinear | ScaleFilter::Bicubic
            | ScaleFilter::OmniScale | ScaleFilter::AaNearestNeighbor
            | ScaleFilter::Vectorize => 1,
            ScaleFilter::Hqx(h) => h.factor(),
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Eagle => 2,
            ScaleFilter::Scale3x => 3,
            ScaleFilter::Xbr(x) => x.factor(),
            ScaleFilter::Xbrz(x) => x.factor(),
            ScaleFilter::XbrHybrid | ScaleFilter::SuperXbr
            | ScaleFilter::Nedi | ScaleFilter::Dcci | ScaleFilter::Edi => 2,
            ScaleFilter::OmniScaleLegacy(f) => f.factor(),
        }
    }

    /// Whether this filter scales to arbitrary display dimensions.
    pub fn is_resizable(self) -> bool {
        matches!(self,
            ScaleFilter::Bilinear | ScaleFilter::Bicubic
            | ScaleFilter::OmniScale | ScaleFilter::AaNearestNeighbor
            | ScaleFilter::Vectorize)
    }
}
