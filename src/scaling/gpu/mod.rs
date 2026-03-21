//! SDL3 GPU pipeline management for scaling filters.
//!
//! Contains all GPU texture, pipeline, and render functions used by the
//! SDL3 frontend. Separated from main.rs so the code can be shared
//! and eventually adapted for other backends (Metal, wgpu).

mod common;
mod compute;
mod screenshot;

pub use common::{create_texture, upload_and_blit};
pub use compute::{
    init_vectorize_compute_pipeline, vectorize_and_blit,
    GpuVectorizePipelines, init_full_gpu_pipeline, gpu_vectorize_full_pipeline,
    gpu_full_pipeline_screenshot,
    init_diffusion_compute_pipeline, prepare_diffusion_data, diffusion_and_blit,
    init_spline_diffusion_pipelines, spline_diffusion_and_blit,
    init_omniscale_compute_pipeline, scale_compute_and_blit, super_xbr_compute_and_blit,
    init_epx_compute_pipeline, init_eagle_compute_pipeline,
    init_scale3x_compute_pipeline, init_bicubic_compute_pipeline,
    init_aa_nearest_compute_pipeline, init_hqx_compute_pipeline,
    init_xbr_compute_pipeline, init_xbrz_compute_pipeline,
    init_super_xbr_compute_pipeline, init_omniscale_legacy_compute_pipeline,
    init_edi_compute_pipeline, init_nedi_compute_pipeline, init_dcci_compute_pipeline,
};
pub use screenshot::{
    gpu_screenshot,
    gpu_vectorize_screenshot,
    gpu_vectorize_shared_screenshot,
    gpu_spline_diffusion_screenshot,
};
