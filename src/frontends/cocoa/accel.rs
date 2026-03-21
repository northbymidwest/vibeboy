use objc2::{class, msg_send, sel};
use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::runtime::AnyObject;

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
        let manager: *mut AnyObject = msg_send![cm_class, alloc];
        let manager: *mut AnyObject = msg_send![manager, init];
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
            let data: *mut AnyObject = msg_send![*manager, accelerometerData];
            if data.is_null() {
                return None;
            }
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct CMAcceleration {
                x: f64,
                y: f64,
                z: f64,
            }
            unsafe impl Encode for CMAcceleration {
                const ENCODING: Encoding = Encoding::Struct(
                    "CMAcceleration",
                    &[Encoding::Double, Encoding::Double, Encoding::Double],
                );
            }
            unsafe impl RefEncode for CMAcceleration {
                const ENCODING_REF: Encoding = Encoding::Pointer(&CMAcceleration::ENCODING);
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
