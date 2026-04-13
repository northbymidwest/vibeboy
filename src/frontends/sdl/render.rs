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

/// Run a CPU scaling filter and return (pixels, width, height).
pub(super) fn cpu_scale_frame(
    filter: &scaling::ScaleFilter,
    src: &[u32], sw: usize, sh: usize,
    disp_w: usize, disp_h: usize,
) -> (Vec<u32>, u32, u32) {
    scaling::cpu_scale(*filter, src, sw, sh, disp_w, disp_h)
        .unwrap_or_else(|| (src.to_vec(), sw as u32, sh as u32))
}
