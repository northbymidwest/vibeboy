use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub(super) struct CameraThread {
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CameraThread {
    pub fn start() -> Option<Self> {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};

        // Check if any camera is available
        let devices = nokhwa::query(nokhwa::utils::ApiBackend::Auto).ok()?;
        if devices.is_empty() {
            log::info!("No cameras found — using noise generator for Pocket Camera");
            return None;
        }

        let buffer: Arc<Mutex<[u8; 128 * 112]>> = Arc::new(Mutex::new([0u8; 128 * 112]));
        let has_new_frame = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let buf_clone = Arc::clone(&buffer);
        let frame_clone = Arc::clone(&has_new_frame);
        let stop_clone = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                camera_thread_main(buf_clone, frame_clone, stop_clone);
            })
            .expect("failed to spawn camera thread");

        log::info!("Camera capture thread started");
        Some(CameraThread {
            buffer,
            has_new_frame,
            stop,
            handle: Some(handle),
        })
    }

    pub fn read_frame(&self, buf: &mut [u8; 128 * 112]) -> bool {
        if self.has_new_frame.swap(false, Ordering::Acquire) {
            let lock = self.buffer.lock().unwrap();
            buf.copy_from_slice(&*lock);
            true
        } else {
            false
        }
    }
}

impl Drop for CameraThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn camera_thread_main(
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
    use nokhwa::Camera;

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestResolution(
        Resolution::new(640, 480),
    ));
    let mut camera = match Camera::new(CameraIndex::Index(0), format) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("Camera thread: failed to open camera: {}", e);
            return;
        }
    };

    if let Err(e) = camera.open_stream() {
        log::warn!("Camera thread: failed to start stream: {}", e);
        return;
    }

    log::info!("Camera thread: webcam opened");

    while !stop.load(Ordering::Acquire) {
        match camera.frame() {
            Ok(frame) => {
                let decoded = match frame.decode_image::<RgbFormat>() {
                    Ok(img) => img,
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                };

                let src_w = decoded.width();
                let src_h = decoded.height();

                // Convert RGB to RGBA for the image crate
                let rgba = image::RgbaImage::from_fn(src_w, src_h, |x, y| {
                    let p = decoded.get_pixel(x, y);
                    image::Rgba([p[0], p[1], p[2], 255])
                });

                // Crop to 8:7 aspect ratio (128:112)
                let target_ratio = 128.0 / 112.0;
                let src_ratio = src_w as f64 / src_h as f64;
                let (crop_w, crop_h) = if src_ratio > target_ratio {
                    let cw = (src_h as f64 * target_ratio) as u32;
                    (cw, src_h)
                } else {
                    let ch = (src_w as f64 / target_ratio) as u32;
                    (src_w, ch)
                };
                let crop_x = (src_w - crop_w) / 2;
                let crop_y = (src_h - crop_h) / 2;
                let cropped =
                    image::imageops::crop_imm(&rgba, crop_x, crop_y, crop_w, crop_h).to_image();

                // Convert to grayscale and resize
                let gray = image::imageops::grayscale(&cropped);
                let resized = image::imageops::resize(
                    &gray,
                    128,
                    112,
                    image::imageops::FilterType::Lanczos3,
                );

                {
                    let mut lock = buffer.lock().unwrap();
                    lock.copy_from_slice(resized.as_raw());
                }
                has_new_frame.store(true, Ordering::Release);
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    let _ = camera.stop_stream();
    log::info!("Camera thread: shut down");
}
