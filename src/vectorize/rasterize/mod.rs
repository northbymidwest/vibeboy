//! Rasterize vector paths (ColorPath) to pixel buffers.
//!
//! Three rasterizers:
//! - **Scanline** (`rasterize`/`rasterize_scaled`): 2x2 supersampling, nonzero winding.
//! - **Voronoi diffusion** (`rasterize_diffusion`): Gaussian blending with graph-based regions.
//! - **Spline diffusion** (`rasterize_spline_diffusion`): B-spline boundaries + Gaussian blending.

mod scanline;
mod diffusion;
mod spline_diffusion;
mod gpu;

// Re-export public API
pub use scanline::{rasterize, rasterize_scaled};
pub use diffusion::{rasterize_diffusion, build_graph_regions, build_voronoi_ownership};
pub use spline_diffusion::rasterize_spline_diffusion;
pub use gpu::{
    GpuEdge, GpuPathMeta, prepare_gpu_edges,
    GpuEdgeV2, GpuRowRange, prepare_gpu_edges_v2,
};
