//! SDL3 GPU pipeline management for scaling filters.
//!
//! Contains all GPU texture, pipeline, and render functions used by the
//! SDL3 frontend. Separated from main.rs so the code can be shared
//! and eventually adapted for other backends (Metal, wgpu).

mod common;
mod graphics;
mod compute;
mod screenshot;

// Re-export all public items so callers can use `super::gpu::function_name()`
// exactly as before the split.
pub use common::create_texture;
pub use graphics::{
    upload_and_blit,
    init_omniscale_pipeline, render_omniscale,
    init_hqx_pipeline, render_hqx,
    init_bicubic_pipeline, render_bicubic,
    init_omniscale_legacy_pipeline, render_omniscale_legacy,
    init_scale3x_pipeline, render_scale3x,
    init_eagle_pipeline, render_eagle,
    init_aa_nearest_pipeline, render_aa_nearest,
    init_epx_pipeline, render_epx,
    init_xbr_pipeline, render_xbr,
    init_xbrz_pipeline, render_xbrz,
    init_super_xbr_pipeline, render_super_xbr,
};
pub use compute::{
    init_vectorize_compute_pipeline, vectorize_and_blit,
    GpuVectorizePipelines, init_full_gpu_pipeline, gpu_vectorize_full_pipeline,
    gpu_full_pipeline_screenshot,
    init_diffusion_compute_pipeline, prepare_diffusion_data, diffusion_and_blit,
    init_spline_diffusion_pipelines, spline_diffusion_and_blit,
};
pub use screenshot::{
    gpu_screenshot,
    gpu_vectorize_screenshot,
    gpu_vectorize_shared_screenshot,
    gpu_spline_diffusion_screenshot,
};
