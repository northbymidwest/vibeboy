use cocoa::base::id;
use objc::{class, msg_send, sel, sel_impl};

use super::AccelSource;

#[link(name = "CoreMotion", kind = "framework")]
unsafe extern "C" {}

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
        let cm_class = class!(CMMotionManager);
        let manager: id = msg_send![cm_class, alloc];
        let manager: id = msg_send![manager, init];
        if manager.is_null() {
            log::info!("CMMotionManager init failed — accelerometer disabled");
            return AccelSource::None;
        }
        let available: bool = msg_send![manager, isAccelerometerAvailable];
        if !available {
            let () = msg_send![manager, release];
            log::info!("CMMotionManager: no accelerometer available");
            return AccelSource::None;
        }
        // Set update interval (~60 Hz)
        let interval: f64 = 1.0 / 60.0;
        let () = msg_send![manager, setAccelerometerUpdateInterval: interval];
        let () = msg_send![manager, startAccelerometerUpdates];
        eprintln!("Accelerometer: CoreMotion (CMMotionManager)");
        AccelSource::CoreMotion(manager)
    }
}

pub(super) fn poll_accel(source: &AccelSource) -> Option<(f32, f32, f32)> {
    match source {
        AccelSource::None => None,
        AccelSource::IoKit => super::macos_accel::poll(),
        AccelSource::CoreMotion(manager) => unsafe {
            let data: id = msg_send![*manager, accelerometerData];
            if data.is_null() {
                return None;
            }
            // CMAcceleration is a struct { x: f64, y: f64, z: f64 }
            // -[CMAccelerometerData acceleration] returns it by value
            #[repr(C)]
            struct CMAcceleration {
                x: f64,
                y: f64,
                z: f64,
            }
            let accel: CMAcceleration = msg_send![data, acceleration];
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
            let () = msg_send![*manager, stopAccelerometerUpdates];
            let () = msg_send![*manager, release];
        },
    }
}
