use super::*;

/// Compute aspect-correct display dimensions for the current window.
pub(super) fn display_size(window: &sdl3::video::Window, src_w: u32, src_h: u32) -> (usize, usize) {
    let (ww, wh) = window.size();
    let src_aspect = src_w as f64 / src_h as f64;
    let win_aspect = ww as f64 / wh as f64;
    if win_aspect > src_aspect {
        ((wh as f64 * src_aspect) as usize, wh as usize)
    } else {
        (ww as usize, (ww as f64 / src_aspect) as usize)
    }
}

/// Compute rounded integer scale factor from display size and source size.
pub(super) fn compute_integer_scale(disp_w: usize, disp_h: usize, sw: usize, sh: usize) -> usize {
    let scale_f = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
    scale_f.round().max(1.0) as usize
}

/// Run a CPU scaling filter and return (pixels, width, height).
pub(super) fn cpu_scale_frame(
    filter: &scaling::ScaleFilter,
    src: &[u32], sw: usize, sh: usize,
    disp_w: usize, disp_h: usize,
    vec_cache: &mut Option<crate::vectorize::VectorizeCache>,
) -> (Vec<u32>, u32, u32) {
    // Vectorize filters that use cache-based path rendering
    if matches!(filter,
        scaling::ScaleFilter::VectorizeLegacy | scaling::ScaleFilter::VectorizeLegacyAdaptive
        | scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive)
    {
        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let cache = vec_cache.as_mut().unwrap();
        let (raster, w, h) = cache.rasterize(src, sw, sh, scale);
        return (raster.to_vec(), w as u32, h as u32);
    }
    // Diffusion rasterizer works directly from pixels (no vector paths)
    if matches!(filter, scaling::ScaleFilter::VectorizeDiffusion) {
        let scale_f = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let scale = scale_f.round().max(1.0) as usize;
        let (raster, w, h) = crate::vectorize::rasterize::rasterize_diffusion(src, sw, sh, scale);
        return (raster, w as u32, h as u32);
    }
    // Spline-diffusion: vectorize for paths, then Gaussian diffusion with spline boundaries
    if matches!(filter, scaling::ScaleFilter::VectorizeSplineDiffusion | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive) {
        let scale_f = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let scale = scale_f.round().max(1.0) as usize;
        let cache = vec_cache.as_mut().unwrap();
        let (paths, bg_color) = cache.get_paths(src, sw, sh);
        let (raster, w, h) = crate::vectorize::rasterize::rasterize_spline_diffusion(
            paths, src, sw, sh, bg_color, scale,
        );
        return (raster, w as u32, h as u32);
    }
    // All other CPU filters use the shared dispatcher
    scaling::cpu_scale(*filter, src, sw, sh, disp_w, disp_h)
        .unwrap_or_else(|| (src.to_vec(), sw as u32, sh as u32))
}
