//! Shared utility functions used by multiple UI frontends (main.rs, cocoa_ui.rs, winit_ui.rs)
//! and the test runner.

use crate::model::GbModel;
use crate::scaling;
use std::time::Duration;

/// Target frame time: 70224 T-cycles / cpu_clock_rate.
/// Standard: ~16.74ms (~59.73 fps). SGB1: ~16.35ms (~61.17 fps).
pub fn frame_duration(model: GbModel) -> Duration {
    let nanos = 70_224u64 * 1_000_000_000 / model.cpu_clock_rate() as u64;
    Duration::from_nanos(nanos)
}

/// Parse a model string for clap value_parser.
pub fn parse_model(s: &str) -> Result<GbModel, String> {
    s.parse::<GbModel>()
}

/// Parse a filter string for clap value_parser.
pub fn parse_filter(s: &str) -> Result<String, String> {
    scaling::ScaleFilter::validate_name(s)
}

/// Auto-detect hardware model from ROM header CGB flag.
pub fn auto_detect_model(rom: &[u8]) -> GbModel {
    let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
    if cgb_flag == 0x80 || cgb_flag == 0xC0 {
        GbModel::Cgb
    } else {
        GbModel::Dmg
    }
}
