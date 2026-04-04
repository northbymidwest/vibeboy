//! Pixel-art scaling algorithms.

pub mod nearest_aa;
pub mod bicubic;
pub mod bilinear;
pub mod dcci;
pub mod eagle;
pub mod edi;
pub mod epx;
pub mod hqx;
pub mod lcd_grid;
pub mod mmpx;
pub mod nedi;
pub mod omniscale;
pub mod omniscale_legacy;
pub mod sai;
pub mod scale3x;
pub mod super_xbr;
pub mod vectorize_gpu;
pub mod xbr;
pub mod xbrz;
pub mod scalefx;
#[cfg(feature = "sdl3-gpu-shaders")]
pub mod sdl;
#[cfg(feature = "gpu")]
pub mod wgpu_vectorize;
#[cfg(feature = "gpu")]
pub mod wgpu_scale;

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
    Mmpx,
    LcdGrid,
    /// Arbitrary-resolution OmniScale (scales to display size).
    OmniScale,
    /// Arbitrary-resolution OmniScale legacy variant (scales to display size).
    OmniScaleLegacy,
    /// Anti-aliased nearest neighbor (scales to display size).
    NearestAa,
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
    /// CPU implementation of the full GPU vectorize pipeline.
    VectorizeGpu,
    /// ScaleFX 3x edge interpolation (Sp00kyFox).
    ScaleFx,
    /// ScaleFX 9x (two ScaleFX passes chained: 3x → 3x).
    ScaleFx9x,
}

/// Filter metadata: (variant, cli_name, display_name, scale_factor).
/// Scale factor 0 = adaptive (scales to display size). This is the single
/// source of truth — ALL_NAMES, from_name(), factor(), menu entries, etc.
/// are all derived from this table.
pub struct FilterInfo {
    pub filter: ScaleFilter,
    pub cli_name: &'static str,
    pub display_name: &'static str,
    pub factor: u32,  // 0 = adaptive
}

const REGISTRY: &[FilterInfo] = &[
    FilterInfo { filter: ScaleFilter::Nearest,          cli_name: "nearest",      display_name: "Nearest",        factor: 0 },
    FilterInfo { filter: ScaleFilter::NearestAa,       cli_name: "nearest-aa",   display_name: "Nearest AA",     factor: 0 },
    FilterInfo { filter: ScaleFilter::Bilinear,         cli_name: "bilinear",     display_name: "Bilinear",      factor: 0 },
    FilterInfo { filter: ScaleFilter::Bicubic,          cli_name: "bicubic",      display_name: "Bicubic",       factor: 0 },
    FilterInfo { filter: ScaleFilter::Epx,              cli_name: "epx",          display_name: "EPX / Scale2x", factor: 2 },
    FilterInfo { filter: ScaleFilter::Scale2x,          cli_name: "scale2x",      display_name: "Scale2x",       factor: 2 },
    FilterInfo { filter: ScaleFilter::Scale3x,          cli_name: "scale3x",      display_name: "Scale3x",       factor: 3 },
    FilterInfo { filter: ScaleFilter::Scale4x,          cli_name: "scale4x",      display_name: "Scale4x",       factor: 4 },
    FilterInfo { filter: ScaleFilter::Eagle,            cli_name: "eagle",        display_name: "Eagle",          factor: 2 },
    FilterInfo { filter: ScaleFilter::Sai2x,            cli_name: "2xsai",       display_name: "2xSaI",          factor: 2 },
    FilterInfo { filter: ScaleFilter::Super2xSai,       cli_name: "super-2xsai", display_name: "Super 2xSaI",    factor: 2 },
    FilterInfo { filter: ScaleFilter::SuperEagle,       cli_name: "super-eagle",  display_name: "Super Eagle",    factor: 2 },
    FilterInfo { filter: ScaleFilter::Hqx(HqxScale::Hq2x), cli_name: "hq2x",    display_name: "HQ2x",          factor: 2 },
    FilterInfo { filter: ScaleFilter::Hqx(HqxScale::Hq3x), cli_name: "hq3x",    display_name: "HQ3x",          factor: 3 },
    FilterInfo { filter: ScaleFilter::Hqx(HqxScale::Hq4x), cli_name: "hq4x",    display_name: "HQ4x",          factor: 4 },
    FilterInfo { filter: ScaleFilter::Xbr(XbrScale::Xbr2x), cli_name: "xbr2x",  display_name: "xBR 2x",        factor: 2 },
    FilterInfo { filter: ScaleFilter::Xbr(XbrScale::Xbr3x), cli_name: "xbr3x",  display_name: "xBR 3x",        factor: 3 },
    FilterInfo { filter: ScaleFilter::Xbr(XbrScale::Xbr4x), cli_name: "xbr4x",  display_name: "xBR 4x",        factor: 4 },
    FilterInfo { filter: ScaleFilter::SuperXbr,         cli_name: "super-xbr",    display_name: "Super xBR",     factor: 2 },
    FilterInfo { filter: ScaleFilter::Xbrz(XbrzScale::Xbrz2x), cli_name: "xbrz2x", display_name: "xBRZ 2x",    factor: 2 },
    FilterInfo { filter: ScaleFilter::Xbrz(XbrzScale::Xbrz3x), cli_name: "xbrz3x", display_name: "xBRZ 3x",    factor: 3 },
    FilterInfo { filter: ScaleFilter::Xbrz(XbrzScale::Xbrz4x), cli_name: "xbrz4x", display_name: "xBRZ 4x",    factor: 4 },
    FilterInfo { filter: ScaleFilter::Xbrz(XbrzScale::Xbrz5x), cli_name: "xbrz5x", display_name: "xBRZ 5x",    factor: 5 },
    FilterInfo { filter: ScaleFilter::Xbrz(XbrzScale::Xbrz6x), cli_name: "xbrz6x", display_name: "xBRZ 6x",    factor: 6 },
    FilterInfo { filter: ScaleFilter::Nedi,             cli_name: "nedi",         display_name: "NEDI",           factor: 2 },
    FilterInfo { filter: ScaleFilter::Dcci,             cli_name: "dcci",         display_name: "DCCI",           factor: 2 },
    FilterInfo { filter: ScaleFilter::Edi,              cli_name: "edi",          display_name: "EDI",            factor: 2 },
    FilterInfo { filter: ScaleFilter::Mmpx,             cli_name: "mmpx",         display_name: "MMPX",           factor: 2 },
    FilterInfo { filter: ScaleFilter::LcdGrid,          cli_name: "lcd-grid",     display_name: "LCD Grid",       factor: 4 },
    FilterInfo { filter: ScaleFilter::OmniScale,        cli_name: "omniscale",    display_name: "OmniScale",      factor: 0 },
    FilterInfo { filter: ScaleFilter::OmniScaleLegacy,  cli_name: "omniscale-legacy", display_name: "OmniScale Legacy", factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeLegacy,  cli_name: "vectorize-legacy", display_name: "Vectorize Legacy", factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeLegacyAdaptive, cli_name: "vectorize-legacy-adaptive", display_name: "Vectorize Legacy Adaptive", factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeDiffusion, cli_name: "vectorize-diffusion", display_name: "Vectorize Diffusion", factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeSplineDiffusion, cli_name: "vectorize-spline-diffusion", display_name: "Vectorize Spline Diffusion", factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeSplineDiffusionAdaptive, cli_name: "vectorize-spline-diffusion-adaptive", display_name: "Vectorize Spline Diff Adaptive", factor: 0 },
    FilterInfo { filter: ScaleFilter::Vectorize,        cli_name: "vectorize",    display_name: "Vectorize",      factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeAdaptive, cli_name: "vectorize-adaptive", display_name: "Vectorize Adaptive", factor: 0 },
    FilterInfo { filter: ScaleFilter::VectorizeGpu,     cli_name: "vectorize-gpu", display_name: "Vectorize GPU", factor: 0 },
    FilterInfo { filter: ScaleFilter::ScaleFx,          cli_name: "scalefx",      display_name: "ScaleFX",       factor: 3 },
    FilterInfo { filter: ScaleFilter::ScaleFx9x,        cli_name: "scalefx-9x",   display_name: "ScaleFX 9x",    factor: 9 },
];

impl ScaleFilter {
    fn info(self) -> &'static FilterInfo {
        REGISTRY.iter().find(|e| e.filter == self).expect("filter not in registry")
    }

    /// Create a new VectorizeCache appropriate for this filter, or None if not needed.
    pub fn new_vectorize_cache(self) -> Option<crate::vectorize::VectorizeCache> {
        match self {
            Self::Vectorize | Self::VectorizeSplineDiffusion
                => Some(crate::vectorize::VectorizeCache::new(false)),
            Self::VectorizeAdaptive | Self::VectorizeSplineDiffusionAdaptive
                => Some(crate::vectorize::VectorizeCache::new(true)),
            Self::VectorizeLegacy | Self::VectorizeDiffusion
                => Some(crate::vectorize::VectorizeCache::new_legacy(false)),
            Self::VectorizeLegacyAdaptive
                => Some(crate::vectorize::VectorizeCache::new_legacy(true)),
            _ => None,
        }
    }

    /// All valid CLI name strings (includes "none" as alias for "nearest").
    pub fn all_names() -> Vec<&'static str> {
        let mut names: Vec<&str> = REGISTRY.iter().map(|e| e.cli_name).collect();
        names.push("none"); // alias
        names
    }

    /// Parse a CLI name into a ScaleFilter. Returns None for unrecognized names.
    pub fn from_name(s: &str) -> Option<ScaleFilter> {
        if s == "none" { return Some(ScaleFilter::Nearest); }
        REGISTRY.iter().find(|e| e.cli_name == s).map(|e| e.filter)
    }

    /// Validate a filter name string for CLI parsing.
    pub fn validate_name(s: &str) -> Result<String, String> {
        let lower = s.to_lowercase();
        if Self::from_name(&lower).is_some() {
            Ok(lower)
        } else {
            let names = Self::all_names();
            Err(format!(
                "unknown filter '{}'\n  [possible values: {}]",
                s,
                names.join(", ")
            ))
        }
    }

    /// Human-readable display name for menus.
    pub fn display_name(self) -> &'static str {
        self.info().display_name
    }

    /// CLI name string.
    pub fn cli_name(self) -> &'static str {
        self.info().cli_name
    }

    /// Scale factor: 0 for adaptive filters that scale to display size,
    /// or the fixed integer multiplier (2, 3, 4, etc.).
    pub fn factor(self) -> u32 {
        self.info().factor
    }

    /// Whether the window should be freely resizable with this filter.
    pub fn is_resizable(self) -> bool {
        self.factor() == 0
    }

    /// Whether this filter produces output scaled to the display dimensions.
    /// Nearest is resizable but relies on GPU texture stretching instead.
    pub fn scales_to_display(self) -> bool {
        self.factor() == 0 && self != ScaleFilter::Nearest
    }

    /// All registered filters in display order (for building menus).
    pub fn all_filters() -> &'static [FilterInfo] {
        REGISTRY
    }

    /// Iterator over (display_name, ScaleFilter) pairs for menu building.
    /// Excludes Scale2x (alias for EPX).
    pub fn menu_entries() -> impl Iterator<Item = (&'static str, ScaleFilter)> {
        REGISTRY.iter()
            .filter(|e| e.filter != ScaleFilter::Scale2x)
            .map(|e| (e.display_name, e.filter))
    }

    /// Which submenu group a filter belongs to, for building grouped menus.
    pub fn menu_group(self) -> FilterMenuGroup {
        match self {
            ScaleFilter::Hqx(_) => FilterMenuGroup::Hqx,
            ScaleFilter::Xbr(_) | ScaleFilter::SuperXbr => FilterMenuGroup::Xbr,
            ScaleFilter::Xbrz(_) => FilterMenuGroup::Xbrz,
            ScaleFilter::Nedi | ScaleFilter::Dcci | ScaleFilter::Edi => FilterMenuGroup::EdgeDetect,
            _ => FilterMenuGroup::Main,
        }
    }
}

/// Submenu grouping for filter menus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMenuGroup {
    Main,
    Hqx,
    Xbr,
    Xbrz,
    EdgeDetect,
}

impl FilterMenuGroup {
    /// Display name for the submenu.
    pub fn label(self) -> &'static str {
        match self {
            Self::Main => "Filter",
            Self::Hqx => "HQx",
            Self::Xbr => "xBR",
            Self::Xbrz => "xBRZ",
            Self::EdgeDetect => "Edge Detect",
        }
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
        ScaleFilter::Mmpx => {
            let s = mmpx::scale(src, sw, sh);
            (s, sw as u32 * 2, sh as u32 * 2)
        }
        ScaleFilter::LcdGrid => {
            let s = lcd_grid::scale(src, sw, sh, 4);
            (s, sw as u32 * 4, sh as u32 * 4)
        }
        ScaleFilter::OmniScale => {
            let s = omniscale::scale_to(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::OmniScaleLegacy => {
            let s = omniscale_legacy::scale_to(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::NearestAa => {
            let s = nearest_aa::scale(src, sw, sh, disp_w, disp_h);
            (s, disp_w as u32, disp_h as u32)
        }
        ScaleFilter::VectorizeGpu => {
            let scale_f = (disp_w as f32 / sw as f32).min(disp_h as f32 / sh as f32);
            let ow = (sw as f32 * scale_f).ceil() as usize;
            let oh = (sh as f32 * scale_f).ceil() as usize;
            let s = vectorize_gpu::scale(src, sw, sh, scale_f);
            (s, ow as u32, oh as u32)
        }
        ScaleFilter::Vectorize | ScaleFilter::VectorizeAdaptive => {
            let scale = (disp_w / sw).max(disp_h / sh).max(1);
            let (s, w, h) = crate::vectorize::vectorize_to_raster_shared(src, sw, sh, scale);
            (s, w as u32, h as u32)
        }
        ScaleFilter::VectorizeLegacy | ScaleFilter::VectorizeLegacyAdaptive => {
            let scale = (disp_w / sw).max(disp_h / sh).max(1);
            let (s, w, h) = crate::vectorize::vectorize_to_raster(src, sw, sh, scale);
            (s, w as u32, h as u32)
        }
        ScaleFilter::VectorizeDiffusion => {
            let scale = (disp_w / sw).max(disp_h / sh).max(1);
            let (s, w, h) = crate::vectorize::rasterize::rasterize_diffusion(src, sw, sh, scale);
            (s, w as u32, h as u32)
        }
        ScaleFilter::VectorizeSplineDiffusion | ScaleFilter::VectorizeSplineDiffusionAdaptive => {
            let scale = (disp_w / sw).max(disp_h / sh).max(1);
            let (paths, bg_color) = crate::vectorize::vectorize_core(src, sw, sh);
            let (s, w, h) = crate::vectorize::rasterize::rasterize_spline_diffusion(
                &paths, src, sw, sh, bg_color, scale,
            );
            (s, w as u32, h as u32)
        }
        ScaleFilter::ScaleFx => {
            let s = scalefx::scale(src, sw, sh);
            (s, sw as u32 * 3, sh as u32 * 3)
        }
        ScaleFilter::ScaleFx9x => {
            let s = scalefx::scale9x(src, sw, sh);
            (s, sw as u32 * 9, sh as u32 * 9)
        }
        ScaleFilter::Nearest => {
            let mut out = vec![0u32; disp_w * disp_h];
            for y in 0..disp_h {
                let sy = y * sh / disp_h;
                for x in 0..disp_w {
                    let sx = x * sw / disp_w;
                    out[y * disp_w + x] = src[sy * sw + sx];
                }
            }
            (out, disp_w as u32, disp_h as u32)
        }
        _ => return None,
    })
}
