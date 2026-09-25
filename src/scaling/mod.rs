//! Pixel-art scaling algorithms.

pub mod bicubic;
pub mod bilinear;
pub mod dcci;
pub mod eagle;
pub mod edi;
pub mod epx;
pub mod hqx;
pub mod lcd_grid;
pub mod mmpx;
pub mod nearest_aa;
pub mod nedi;
pub mod omniscale;
pub mod omniscale_legacy;
pub mod sai;
pub mod scale3x;
pub mod scalefx;
#[cfg(feature = "sdl3-gpu-shaders")]
pub mod sdl;
pub mod super_xbr;
pub mod vectorize;
#[cfg(feature = "gpu")]
pub mod wgpu_scale;
#[cfg(feature = "gpu")]
pub mod wgpu_vectorize;
pub mod xbr;
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
    if alpha <= 0.0 {
        return a;
    }
    if alpha >= 1.0 {
        return b;
    }
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
    if a == b {
        return 0.0;
    }
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
    /// Full GPU vectorize: all pipeline stages run on GPU compute shaders.
    /// CPU implementation of the full GPU vectorize pipeline.
    Vectorize,
    /// ScaleFX 3x edge interpolation (Sp00kyFox).
    ScaleFx,
    /// ScaleFX 9x (two ScaleFX passes chained: 3x → 3x).
    ScaleFx9x,
}

/// Invokes `$callback!` with every scaling compute shader as
/// `Variant => "module"` pairs. The module name is the shader's base name:
/// `src/shaders/{module}.slang`, compiled by `build.rs` to
/// `{OUT_DIR}/{module}_comp.{spv,metal,dxil,wgsl}`.
///
/// This is the one list of scaling shaders. `ScaleShader` is generated from
/// it, and each GPU backend expands it into its own bytecode table, so a new
/// shader is added here (and to `build.rs`) and nowhere else.
#[macro_export]
macro_rules! scale_shader_list {
    ($callback:ident) => {
        $callback! {
            Nearest => "nearest",
            NearestAa => "nearest_aa",
            Bilinear => "bilinear",
            Bicubic => "bicubic",
            Eagle => "eagle",
            Epx => "epx",
            Scale3x => "scale3x",
            LcdGrid => "lcd_grid",
            Mmpx => "mmpx",
            Sai2x => "sai2x",
            SuperSai2x => "super_sai2x",
            SuperEagle => "super_eagle",
            Dcci => "dcci",
            Nedi => "nedi",
            Edi => "edi",
            Hqx => "hqx",
            Xbr => "xbr",
            Xbrz => "xbrz",
            OmniScale => "omniscale",
            OmniScaleLegacy => "omniscale_legacy",
            SuperXbr => "super_xbr",
            ScaleFx => "scalefx",
        }
    };
}

macro_rules! define_scale_shaders {
    ($($variant:ident => $module:literal),* $(,)?) => {
        /// A scaling compute shader. Several filters can share one shader
        /// (EPX serves Scale2x and Scale4x; ScaleFX serves the 3x and 9x
        /// filters), so GPU backends cache pipelines per shader.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ScaleShader {
            $($variant),*
        }

        impl ScaleShader {
            /// Every scaling shader, in `scale_shader_list!` order.
            pub const ALL: &'static [ScaleShader] = &[$(ScaleShader::$variant),*];
            /// Number of scaling shaders (size of per-shader pipeline caches).
            pub const COUNT: usize = Self::ALL.len();

            /// Shader base name (`src/shaders/{module}.slang`).
            pub fn module(self) -> &'static str {
                match self {
                    $(ScaleShader::$variant => $module),*
                }
            }

            /// Dense index for per-shader pipeline caches.
            pub fn index(self) -> usize {
                self as usize
            }
        }
    };
}

scale_shader_list!(define_scale_shaders);

/// How a filter's shader is dispatched. The kind also fixes the shader's
/// resource layout: every kind reads one pixel storage buffer and writes one
/// storage texture; the multi-pass kinds add read-write storage buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuPass {
    /// One dispatch over the output, uniforms `[src_w, src_h, out_w, out_h,
    /// extra, 0, 0, 0]`.
    Single,
    /// Super xBR: three dispatches over the output (diagonal, cardinal,
    /// polish) sharing one `out_w * out_h` u32 intermediate buffer. The fifth
    /// uniform word is the pass index.
    SuperXbr,
    /// ScaleFX: five dispatches (four over the source into float4
    /// intermediates, one over the 3x output that also packs pixels into a
    /// buffer). `chained` runs the five passes a second time on the packed
    /// 3x result for 9x output. The fifth uniform word is the pass index.
    ScaleFx { chained: bool },
}

impl GpuPass {
    /// Number of read-write storage buffers the shader binds.
    pub fn rw_storage_buffers(self) -> u32 {
        match self {
            GpuPass::Single => 0,
            GpuPass::SuperXbr => 1,
            GpuPass::ScaleFx { .. } => 5,
        }
    }
}

/// What a single-pass shader reads from the fifth uniform word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformExtra {
    /// Unused; always 0.
    None,
    /// Integer scale factor, `out_w / src_w`.
    IntScale,
    /// OmniScale's source pixel size in output space as f32 bits,
    /// `sqrt((src_w/out_w)^2 + (src_h/out_h)^2)`.
    PixelSize,
}

/// GPU compute description of a filter, shared by every GPU backend
/// (SDL3 GPU, wgpu, Metal).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuShader {
    pub shader: ScaleShader,
    pub pass: GpuPass,
    pub extra: UniformExtra,
}

impl GpuShader {
    /// Uniform block for a single-pass dispatch.
    pub fn uniforms(&self, src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> [u32; 8] {
        let extra = match self.extra {
            UniformExtra::None => 0,
            UniformExtra::IntScale => out_w.checked_div(src_w).unwrap_or(1),
            UniformExtra::PixelSize => {
                let sx = src_w as f32 / out_w as f32;
                let sy = src_h as f32 / out_h as f32;
                f32::to_bits((sx * sx + sy * sy).sqrt())
            }
        };
        [src_w, src_h, out_w, out_h, extra, 0, 0, 0]
    }
}

/// Uniform block for pass `pass` of a multi-pass shader (Super xBR, ScaleFX).
pub fn multipass_uniforms(src_w: u32, src_h: u32, out_w: u32, out_h: u32, pass: u32) -> [u32; 8] {
    [src_w, src_h, out_w, out_h, pass, 0, 0, 0]
}

/// Filter metadata: variant, CLI name, display name, scale factor and GPU
/// shader. Scale factor 0 = adaptive (scales to display size). This is the
/// single source of truth: ALL_NAMES, from_name(), factor(), menu entries and
/// every GPU backend's shader mapping are derived from this table.
pub struct FilterInfo {
    pub filter: ScaleFilter,
    pub cli_name: &'static str,
    pub display_name: &'static str,
    pub factor: u32, // 0 = adaptive
    /// Compute shader description. `None` for Vectorize, which runs its own
    /// multi-stage pipeline in each backend.
    pub gpu: Option<GpuShader>,
}

const REGISTRY: &[FilterInfo] = &[
    FilterInfo {
        filter: ScaleFilter::Sai2x,
        cli_name: "2xsai",
        display_name: "2xSaI",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Sai2x,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Bicubic,
        cli_name: "bicubic",
        display_name: "Bicubic",
        factor: 0,
        gpu: Some(GpuShader {
            shader: ScaleShader::Bicubic,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Bilinear,
        cli_name: "bilinear",
        display_name: "Bilinear",
        factor: 0,
        gpu: Some(GpuShader {
            shader: ScaleShader::Bilinear,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Dcci,
        cli_name: "dcci",
        display_name: "DCCI",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Dcci,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Eagle,
        cli_name: "eagle",
        display_name: "Eagle",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Eagle,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Edi,
        cli_name: "edi",
        display_name: "EDI",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Edi,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Epx,
        cli_name: "epx",
        display_name: "EPX / Scale2x",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Epx,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Hqx(HqxScale::Hq2x),
        cli_name: "hq2x",
        display_name: "HQ2x",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Hqx,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Hqx(HqxScale::Hq3x),
        cli_name: "hq3x",
        display_name: "HQ3x",
        factor: 3,
        gpu: Some(GpuShader {
            shader: ScaleShader::Hqx,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Hqx(HqxScale::Hq4x),
        cli_name: "hq4x",
        display_name: "HQ4x",
        factor: 4,
        gpu: Some(GpuShader {
            shader: ScaleShader::Hqx,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::LcdGrid,
        cli_name: "lcd-grid",
        display_name: "LCD Grid",
        factor: 4,
        gpu: Some(GpuShader {
            shader: ScaleShader::LcdGrid,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Mmpx,
        cli_name: "mmpx",
        display_name: "MMPX",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Mmpx,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Nearest,
        cli_name: "nearest",
        display_name: "Nearest",
        factor: 0,
        gpu: Some(GpuShader {
            shader: ScaleShader::Nearest,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::NearestAa,
        cli_name: "nearest-aa",
        display_name: "Nearest AA",
        factor: 0,
        gpu: Some(GpuShader {
            shader: ScaleShader::NearestAa,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Nedi,
        cli_name: "nedi",
        display_name: "NEDI",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Nedi,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::OmniScale,
        cli_name: "omniscale",
        display_name: "OmniScale",
        factor: 0,
        gpu: Some(GpuShader {
            shader: ScaleShader::OmniScale,
            pass: GpuPass::Single,
            extra: UniformExtra::PixelSize,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::OmniScaleLegacy,
        cli_name: "omniscale-legacy",
        display_name: "OmniScale Legacy",
        factor: 0,
        gpu: Some(GpuShader {
            shader: ScaleShader::OmniScaleLegacy,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Scale2x,
        cli_name: "scale2x",
        display_name: "Scale2x",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Epx,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Scale3x,
        cli_name: "scale3x",
        display_name: "Scale3x",
        factor: 3,
        gpu: Some(GpuShader {
            shader: ScaleShader::Scale3x,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Scale4x,
        cli_name: "scale4x",
        display_name: "Scale4x",
        factor: 4,
        gpu: Some(GpuShader {
            shader: ScaleShader::Epx,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::ScaleFx,
        cli_name: "scalefx",
        display_name: "ScaleFX",
        factor: 3,
        gpu: Some(GpuShader {
            shader: ScaleShader::ScaleFx,
            pass: GpuPass::ScaleFx { chained: false },
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::ScaleFx9x,
        cli_name: "scalefx-9x",
        display_name: "ScaleFX 9x",
        factor: 9,
        gpu: Some(GpuShader {
            shader: ScaleShader::ScaleFx,
            pass: GpuPass::ScaleFx { chained: true },
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Super2xSai,
        cli_name: "super-2xsai",
        display_name: "Super 2xSaI",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::SuperSai2x,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::SuperEagle,
        cli_name: "super-eagle",
        display_name: "Super Eagle",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::SuperEagle,
            pass: GpuPass::Single,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::SuperXbr,
        cli_name: "super-xbr",
        display_name: "Super xBR",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::SuperXbr,
            pass: GpuPass::SuperXbr,
            extra: UniformExtra::None,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Vectorize,
        cli_name: "vectorize",
        display_name: "Vectorize",
        factor: 0,
        gpu: None,
    },
    FilterInfo {
        filter: ScaleFilter::Xbr(XbrScale::Xbr2x),
        cli_name: "xbr2x",
        display_name: "xBR 2x",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbr,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbr(XbrScale::Xbr3x),
        cli_name: "xbr3x",
        display_name: "xBR 3x",
        factor: 3,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbr,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbr(XbrScale::Xbr4x),
        cli_name: "xbr4x",
        display_name: "xBR 4x",
        factor: 4,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbr,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbrz(XbrzScale::Xbrz2x),
        cli_name: "xbrz2x",
        display_name: "xBRZ 2x",
        factor: 2,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbrz,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbrz(XbrzScale::Xbrz3x),
        cli_name: "xbrz3x",
        display_name: "xBRZ 3x",
        factor: 3,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbrz,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbrz(XbrzScale::Xbrz4x),
        cli_name: "xbrz4x",
        display_name: "xBRZ 4x",
        factor: 4,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbrz,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbrz(XbrzScale::Xbrz5x),
        cli_name: "xbrz5x",
        display_name: "xBRZ 5x",
        factor: 5,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbrz,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
    FilterInfo {
        filter: ScaleFilter::Xbrz(XbrzScale::Xbrz6x),
        cli_name: "xbrz6x",
        display_name: "xBRZ 6x",
        factor: 6,
        gpu: Some(GpuShader {
            shader: ScaleShader::Xbrz,
            pass: GpuPass::Single,
            extra: UniformExtra::IntScale,
        }),
    },
];

impl ScaleFilter {
    fn info(self) -> &'static FilterInfo {
        REGISTRY
            .iter()
            .find(|e| e.filter == self)
            .expect("filter not in registry")
    }

    pub fn all_names() -> Vec<&'static str> {
        let mut names: Vec<&str> = REGISTRY.iter().map(|e| e.cli_name).collect();
        names.push("none"); // alias
        names
    }

    /// Parse a CLI name into a ScaleFilter. Returns None for unrecognized names.
    pub fn from_name(s: &str) -> Option<ScaleFilter> {
        if s == "none" {
            return Some(ScaleFilter::Nearest);
        }
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

    /// GPU compute shader description, or `None` for filters without a
    /// scaling compute shader (Vectorize has its own pipeline).
    pub fn gpu(self) -> Option<&'static GpuShader> {
        self.info().gpu.as_ref()
    }

    /// Output size of a GPU or CPU scaling pass: the fixed integer multiple
    /// of the source, or `adaptive` for filters that scale to the display.
    pub fn output_size(self, src_w: u32, src_h: u32, adaptive: (u32, u32)) -> (u32, u32) {
        match self.factor() {
            0 => adaptive,
            f => (src_w * f, src_h * f),
        }
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
        REGISTRY
            .iter()
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
    src: &[u32],
    sw: usize,
    sh: usize,
    disp_w: usize,
    disp_h: usize,
) -> Option<(Vec<u32>, u32, u32)> {
    Some(match filter {
        ScaleFilter::Hqx(mode) => {
            let s = hqx::scale(src, sw, sh, mode);
            let f = mode.factor();
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
            let f = mode.factor();
            (s, sw as u32 * f, sh as u32 * f)
        }
        ScaleFilter::Xbrz(mode) => {
            let s = xbrz::scale(src, sw, sh, mode);
            let f = mode.factor();
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
        ScaleFilter::Vectorize => {
            let scale_f = (disp_w as f32 / sw as f32).min(disp_h as f32 / sh as f32);
            let ow = (sw as f32 * scale_f).ceil() as usize;
            let oh = (sh as f32 * scale_f).ceil() as usize;
            let s = vectorize::scale(src, sw, sh, scale_f);
            (s, ow as u32, oh as u32)
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_filter_but_vectorize_has_a_gpu_shader() {
        for e in REGISTRY {
            assert_eq!(
                e.gpu.is_none(),
                e.filter == ScaleFilter::Vectorize,
                "{}",
                e.cli_name
            );
        }
    }

    #[test]
    fn filters_sharing_a_shader_agree_on_its_layout() {
        for a in REGISTRY.iter().filter_map(|e| e.gpu) {
            for b in REGISTRY.iter().filter_map(|e| e.gpu) {
                if a.shader == b.shader {
                    assert_eq!(
                        a.pass.rw_storage_buffers(),
                        b.pass.rw_storage_buffers(),
                        "{:?}",
                        a.shader
                    );
                }
            }
        }
    }

    #[test]
    fn every_shader_is_used() {
        for &s in ScaleShader::ALL {
            assert!(
                REGISTRY
                    .iter()
                    .any(|e| e.gpu.is_some_and(|g| g.shader == s)),
                "{s:?} has no filter"
            );
        }
    }

    #[test]
    fn multipass_filters_have_fixed_factors() {
        for e in REGISTRY {
            if let Some(g) = e.gpu {
                match g.pass {
                    GpuPass::Single => {}
                    GpuPass::SuperXbr => assert_eq!(e.factor, 2),
                    GpuPass::ScaleFx { chained } => {
                        assert_eq!(e.factor, if chained { 9 } else { 3 })
                    }
                }
            }
        }
    }
}
