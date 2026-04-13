//! SDL3 GPU pipeline management for scaling filters.
//!
//! Contains all GPU texture, pipeline, and render functions used by the
//! SDL3 frontend. Separated from main.rs so the code can be shared
//! and eventually adapted for other backends (Metal, wgpu).

mod common;
mod compute;
pub mod pipelines;
mod screenshot;

pub use common::{create_texture, upload_and_blit};
pub use compute::{
    GpuVectorizePipelines, init_full_gpu_pipeline, gpu_vectorize_full_pipeline,
    gpu_full_pipeline_screenshot,
    init_omniscale_compute_pipeline, scale_compute_and_blit, super_xbr_compute_and_blit,
    init_epx_compute_pipeline, init_eagle_compute_pipeline,
    init_scale3x_compute_pipeline, init_bicubic_compute_pipeline,
    init_nearest_aa_compute_pipeline, init_hqx_compute_pipeline,
    init_xbr_compute_pipeline, init_xbrz_compute_pipeline,
    init_super_xbr_compute_pipeline, init_omniscale_legacy_compute_pipeline,
    init_edi_compute_pipeline, init_nedi_compute_pipeline, init_dcci_compute_pipeline,
    init_mmpx_compute_pipeline,
    init_lcd_grid_compute_pipeline,
    init_nearest_compute_pipeline,
    init_bilinear_compute_pipeline,
    init_sai2x_compute_pipeline,
    init_super_sai2x_compute_pipeline,
    init_super_eagle_compute_pipeline,
    init_scalefx_compute_pipeline, scalefx_compute_and_blit,
};
pub use screenshot::gpu_screenshot;
