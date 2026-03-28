use objc2::rc::Retained;
use objc2_core_motion::CMMotionManager;

use super::AccelSource;

pub(super) fn init_accel() -> AccelSource {
    // Try IOKit HID first (Apple Silicon native)
    #[cfg(target_os = "macos")]
    {
        if super::macos_accel::init() {
            eprintln!("Accelerometer: macOS native (Apple Silicon)");
            return AccelSource::IoKit;
        }
    }

    // Fallback: CMMotionManager
    unsafe {
        let manager = CMMotionManager::new();
        if !manager.isAccelerometerAvailable() {
            log::info!("CMMotionManager: no accelerometer available");
            return AccelSource::None;
        }
        // Set update interval (~60 Hz)
        manager.setAccelerometerUpdateInterval(1.0 / 60.0);
        manager.startAccelerometerUpdates();
        eprintln!("Accelerometer: CoreMotion (CMMotionManager)");
        AccelSource::CoreMotion(manager)
    }
}

pub(super) fn poll_accel(source: &AccelSource) -> Option<(f32, f32, f32)> {
    match source {
        AccelSource::None => None,
        AccelSource::IoKit => super::macos_accel::poll(),
        AccelSource::CoreMotion(manager) => unsafe {
            let data = manager.accelerometerData()?;
            let accel = data.acceleration();
            Some((accel.x as f32, accel.y as f32, accel.z as f32))
        },
    }
}

pub(super) fn close_accel(source: &AccelSource) {
    match source {
        AccelSource::None => {}
        AccelSource::IoKit => {
            #[cfg(target_os = "macos")]
            super::macos_accel::close();
        }
        AccelSource::CoreMotion(manager) => unsafe {
            manager.stopAccelerometerUpdates();
        },
    }
}
