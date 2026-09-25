//! SDL3 GPU pipeline management for scaling filters.
//!
//! Contains all GPU texture, pipeline, and render functions used by the
//! SDL3 frontend. Separated from main.rs so the code can be shared
//! and eventually adapted for other backends (Metal, wgpu).

mod common;
mod compute;
pub mod pipelines;
mod scale;
mod screenshot;

pub use common::{create_texture, upload_and_blit};
pub use compute::{
    GpuVectorizePipelines, gpu_full_pipeline_screenshot, gpu_vectorize_full_pipeline,
    init_full_gpu_pipeline,
};
pub use scale::{ScaleBufCache, init_scale_pipeline, scale_compute_and_blit};
pub use screenshot::gpu_screenshot;
