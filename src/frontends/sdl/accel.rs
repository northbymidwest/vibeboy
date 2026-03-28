use super::*;

/// Try to open an accelerometer: prefer macOS native IOKit on Apple Silicon,
/// fall back to SDL3 sensor API.
pub(super) fn init_accel(sdl: &sdl3::Sdl) -> AccelSource {
    // Try macOS native accelerometer first
    #[cfg(target_os = "macos")]
    {
        if macos_accel::init() {
            eprintln!("Accelerometer: macOS native (Apple Silicon)");
            return AccelSource::MacosNative;
        }
        log::info!("macOS native accelerometer not available, trying SDL3");
    }

    // Fall back to SDL3 sensor
    let sensor_sys = match sdl.sensor() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("SDL sensor subsystem init failed: {} — accelerometer disabled", e);
            return AccelSource::None;
        }
    };
    let ids = match sensor_sys.num_sensors() {
        Ok(ids) => ids,
        Err(e) => {
            log::info!("No sensors found: {} — accelerometer disabled", e);
            return AccelSource::None;
        }
    };
    for id in ids {
        if let Ok(sensor) = sensor_sys.open(id) {
            if sensor.sensor_type() == SensorType::Accelerometer {
                eprintln!("Accelerometer: SDL3 ({})", sensor.name());
                return AccelSource::Sdl(sensor);
            }
        }
    }
    log::info!("No accelerometer found — MBC7 will use center values");
    AccelSource::None
}

/// Open a gamepad and enable its accelerometer if present. Logs the result.
pub(super) fn enable_gamepad_sensors(gp: &sdl3::gamepad::Gamepad) {
    if unsafe { gp.has_sensor(SensorType::Accelerometer) } {
        if gp.sensor_set_enabled(SensorType::Accelerometer, true).is_ok() {
            eprintln!("  Accelerometer enabled");
        }
    }
}
