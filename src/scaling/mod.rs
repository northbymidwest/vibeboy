//! Pixel-art scaling algorithms.

pub mod eagle;
pub mod epx;
pub mod hqx;
pub mod scale3x;

/// Sample a pixel with clamped coordinates.
#[inline(always)]
fn get(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let cx = x.clamp(0, w as isize - 1) as usize;
    let cy = y.clamp(0, h as isize - 1) as usize;
    src[cy * w + cx]
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

/// Scaling filter for the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleFilter {
    Nearest,
    Hqx(HqxScale),
    Epx,
    Scale2x,
    Scale3x,
    Eagle,
}

impl ScaleFilter {
    pub fn factor(self) -> u32 {
        match self {
            ScaleFilter::Nearest => 1,
            ScaleFilter::Hqx(h) => h.factor(),
            ScaleFilter::Epx | ScaleFilter::Scale2x | ScaleFilter::Eagle => 2,
            ScaleFilter::Scale3x => 3,
        }
    }
}
