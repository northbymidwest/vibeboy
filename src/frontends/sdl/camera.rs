use super::*;

/// Handle to a background camera capture thread.
pub(super) struct CameraThread {
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    _camera_subsystem: sdl3::CameraSubsystem,
}

impl CameraThread {
    /// Enumerate cameras on the main thread, spawn a background thread to capture frames.
    /// Returns None if no camera is available (cart falls back to noise generator).
    pub(super) fn start(sdl: &sdl3::Sdl) -> Option<Self> {
        let cam_sys = match sdl.camera() {
            Ok(cs) => cs,
            Err(e) => {
                log::warn!("SDL camera subsystem init failed: {} — webcam disabled", e);
                return None;
            }
        };

        let device_id = unsafe {
            let mut count: std::ffi::c_int = 0;
            let ids = SDL_GetCameras(&mut count);
            if ids.is_null() || count <= 0 {
                log::info!("No cameras found — using noise generator for Pocket Camera");
                if !ids.is_null() {
                    SDL_free(ids as *mut _);
                }
                return None;
            }
            let first_id = *ids;
            SDL_free(ids as *mut _);
            first_id
        };

        let buffer: Arc<Mutex<[u8; 128 * 112]>> = Arc::new(Mutex::new([0u8; 128 * 112]));
        let has_new_frame = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let buf_clone = Arc::clone(&buffer);
        let new_frame_clone = Arc::clone(&has_new_frame);
        let stop_clone = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                camera_thread_main(device_id.0, buf_clone, new_frame_clone, stop_clone);
            })
            .expect("failed to spawn camera thread");

        log::info!("Camera capture thread started");
        Some(CameraThread {
            buffer,
            has_new_frame,
            stop,
            handle: Some(handle),
            _camera_subsystem: cam_sys,
        })
    }

    /// Read the latest frame into `buf`. Returns true if a new frame was available.
    pub(super) fn read_frame(&self, buf: &mut [u8; 128 * 112]) -> bool {
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

/// Background thread: opens the camera, captures and processes frames in a loop.
fn camera_thread_main(
    device_id: u32,
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) {
    use sdl3::sys::camera::SDL_CameraID;

    // Open the camera on this thread (SDL_Camera* is not Send)
    let camera = unsafe {
        let spec = SDL_CameraSpec {
            format: SysPixelFormat::RGBA32,
            colorspace: SDL_Colorspace::SRGB,
            width: 640,
            height: 480,
            framerate_numerator: 30,
            framerate_denominator: 1,
        };
        let cam = SDL_OpenCamera(SDL_CameraID(device_id), &spec);
        if cam.is_null() {
            log::warn!("Camera thread: failed to open camera");
            return;
        }
        cam
    };

    log::info!("Camera thread: webcam opened (640x480 requested)");

    while !stop.load(Ordering::Acquire) {
        let got_frame = unsafe {
            let mut ts: u64 = 0;
            let surface = SDL_AcquireCameraFrame(camera, &mut ts);
            if surface.is_null() || ts == 0 {
                false
            } else {
                process_camera_frame(surface, &buffer, &has_new_frame);
                SDL_ReleaseCameraFrame(camera, surface);
                true
            }
        };

        if !got_frame {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    unsafe {
        SDL_CloseCamera(camera);
    }
    log::info!("Camera thread: shut down");
}

/// Process a raw SDL surface into 128×112 grayscale and write to shared buffer.
unsafe fn process_camera_frame(
    surface: *mut SDL_Surface,
    buffer: &Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: &Arc<AtomicBool>,
) {
    let surf: &SDL_Surface = unsafe { &*surface };
    let src_w = surf.w as usize;
    let src_h = surf.h as usize;
    let pitch = surf.pitch as usize;
    let pixels = surf.pixels as *const u8;

    if pixels.is_null() || src_w == 0 || src_h == 0 {
        return;
    }

    // Build RGBA image from SDL surface (may have row padding)
    let mut rgba = vec![0u8; src_w * src_h * 4];
    for y in 0..src_h {
        let src_row = unsafe { pixels.add(y * pitch) };
        let dst_off = y * src_w * 4;
        unsafe { std::ptr::copy_nonoverlapping(src_row, rgba.as_mut_ptr().add(dst_off), src_w * 4) };
    }

    let img = image::RgbaImage::from_raw(src_w as u32, src_h as u32, rgba)
        .expect("camera frame size mismatch");

    // Crop to 8:7 aspect ratio (128:112) before resizing
    let target_ratio = 128.0 / 112.0;
    let src_ratio = src_w as f64 / src_h as f64;
    let (crop_w, crop_h) = if src_ratio > target_ratio {
        let cw = (src_h as f64 * target_ratio) as u32;
        (cw, src_h as u32)
    } else {
        let ch = (src_w as f64 / target_ratio) as u32;
        (src_w as u32, ch)
    };
    let crop_x = (src_w as u32 - crop_w) / 2;
    let crop_y = (src_h as u32 - crop_h) / 2;
    let cropped =
        image::imageops::crop_imm(&img, crop_x, crop_y, crop_w, crop_h).to_image();

    // Convert to grayscale and resize with Lanczos3
    let gray = image::imageops::grayscale(&cropped);
    let resized =
        image::imageops::resize(&gray, 128, 112, image::imageops::FilterType::Lanczos3);

    // Write to shared buffer (lock held only for memcpy)
    {
        let mut lock = buffer.lock().unwrap();
        lock.copy_from_slice(resized.as_raw());
    }
    has_new_frame.store(true, Ordering::Release);
}
