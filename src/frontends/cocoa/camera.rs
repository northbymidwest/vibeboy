use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cocoa::base::{id, nil, YES};
use cocoa::foundation::NSString;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

pub(super) mod avf_camera {
    use super::*;
    use objc::declare::ClassDecl;
    use objc::runtime::{Class, Object, Sel};
    use std::os::raw::c_void;

    // CoreMedia / CoreVideo FFI
    type CMSampleBufferRef = *mut c_void;
    type CVImageBufferRef = *mut c_void;
    type CVPixelBufferRef = CVImageBufferRef;
    type CVReturn = i32;

    #[link(name = "CoreMedia", kind = "framework")]
    unsafe extern "C" {
        fn CMSampleBufferGetImageBuffer(sbuf: CMSampleBufferRef) -> CVImageBufferRef;
    }

    #[link(name = "CoreVideo", kind = "framework")]
    unsafe extern "C" {
        fn CVPixelBufferLockBaseAddress(buf: CVPixelBufferRef, flags: u64) -> CVReturn;
        fn CVPixelBufferUnlockBaseAddress(buf: CVPixelBufferRef, flags: u64) -> CVReturn;
        fn CVPixelBufferGetBaseAddress(buf: CVPixelBufferRef) -> *mut u8;
        fn CVPixelBufferGetWidth(buf: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetHeight(buf: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetBytesPerRow(buf: CVPixelBufferRef) -> usize;
        static kCVPixelBufferPixelFormatTypeKey: id;
    }

    unsafe fn get_cv_pixel_format_key() -> id {
        unsafe { kCVPixelBufferPixelFormatTypeKey }
    }

    // Shared state passed to the delegate callback via an ivar
    struct DelegateContext {
        buffer: Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: Arc<AtomicBool>,
    }

    // Register a runtime ObjC class that implements the sample buffer delegate
    fn register_delegate_class() -> &'static Class {
        use std::sync::Once;
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            let superclass = Class::get("NSObject").unwrap();
            let mut decl = ClassDecl::new("VBCameraDelegate", superclass).unwrap();

            decl.add_ivar::<*mut c_void>("_context");

            unsafe {
                // captureOutput:didOutputSampleBuffer:fromConnection:
                decl.add_method(
                    sel!(captureOutput:didOutputSampleBuffer:fromConnection:),
                    capture_output as extern "C" fn(&Object, Sel, id, id, id),
                );
            }

            decl.register();
        });
        Class::get("VBCameraDelegate").unwrap()
    }

    extern "C" fn capture_output(
        this: &Object,
        _sel: Sel,
        _output: id,
        sample_buffer: id,
        _connection: id,
    ) {
        unsafe {
            let ctx_ptr: *mut c_void = *this.get_ivar("_context");
            if ctx_ptr.is_null() {
                return;
            }
            let ctx = &*(ctx_ptr as *const DelegateContext);

            let pixel_buf = CMSampleBufferGetImageBuffer(sample_buffer as *mut c_void);
            if pixel_buf.is_null() {
                return;
            }

            CVPixelBufferLockBaseAddress(pixel_buf, 1); // kCVPixelBufferLock_ReadOnly = 1
            let base = CVPixelBufferGetBaseAddress(pixel_buf);
            let w = CVPixelBufferGetWidth(pixel_buf);
            let h = CVPixelBufferGetHeight(pixel_buf);
            let stride = CVPixelBufferGetBytesPerRow(pixel_buf);

            if !base.is_null() && w > 0 && h > 0 {
                process_bgra_frame(base, w, h, stride, &ctx.buffer, &ctx.has_new_frame);
            }

            CVPixelBufferUnlockBaseAddress(pixel_buf, 1);
        }
    }

    fn process_bgra_frame(
        base: *const u8,
        w: usize,
        h: usize,
        stride: usize,
        buffer: &Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: &Arc<AtomicBool>,
    ) {
        // Build RGBA image from BGRA pixel buffer
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let src_off = y * stride + x * 4;
                let dst_off = (y * w + x) * 4;
                unsafe {
                    let b = *base.add(src_off);
                    let g = *base.add(src_off + 1);
                    let r = *base.add(src_off + 2);
                    let a = *base.add(src_off + 3);
                    rgba[dst_off] = r;
                    rgba[dst_off + 1] = g;
                    rgba[dst_off + 2] = b;
                    rgba[dst_off + 3] = a;
                }
            }
        }

        let img = match image::RgbaImage::from_raw(w as u32, h as u32, rgba) {
            Some(i) => i,
            None => return,
        };

        // Crop to 8:7 aspect ratio (128:112)
        let target_ratio = 128.0 / 112.0;
        let src_ratio = w as f64 / h as f64;
        let (crop_w, crop_h) = if src_ratio > target_ratio {
            let cw = (h as f64 * target_ratio) as u32;
            (cw, h as u32)
        } else {
            let ch = (w as f64 / target_ratio) as u32;
            (w as u32, ch)
        };
        let crop_x = (w as u32 - crop_w) / 2;
        let crop_y = (h as u32 - crop_h) / 2;
        let cropped =
            image::imageops::crop_imm(&img, crop_x, crop_y, crop_w, crop_h).to_image();

        // Convert to grayscale and resize with Lanczos3
        let gray = image::imageops::grayscale(&cropped);
        let resized =
            image::imageops::resize(&gray, 128, 112, image::imageops::FilterType::Lanczos3);

        // Write to shared buffer
        if let Ok(mut lock) = buffer.lock() {
            lock.copy_from_slice(resized.as_raw());
        }
        has_new_frame.store(true, Ordering::Release);
    }

    pub struct CameraCapture {
        buffer: Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: Arc<AtomicBool>,
        session: id,          // AVCaptureSession (retained)
        _delegate: id,        // VBCameraDelegate (retained)
        _context: *mut DelegateContext, // leaked; freed on drop
    }

    unsafe impl Send for CameraCapture {}

    impl CameraCapture {
        pub fn start() -> Option<Self> {
            unsafe {
                // Get default video capture device
                let av_capture_device = Class::get("AVCaptureDevice")?;
                let device: id = msg_send![av_capture_device,
                    defaultDeviceWithMediaType: NSString::alloc(nil).init_str("vide")];
                if device == nil {
                    log::info!("No camera found — using noise generator for Pocket Camera");
                    return None;
                }

                // Create input
                let av_capture_input = Class::get("AVCaptureDeviceInput")?;
                let mut error: id = nil;
                let input: id = msg_send![av_capture_input,
                    deviceInputWithDevice: device error: &mut error];
                if input == nil || error != nil {
                    log::warn!("Failed to create camera input");
                    return None;
                }

                // Create session
                let session: id = msg_send![Class::get("AVCaptureSession")?, new];
                // AVCaptureSessionPreset640x480 = "AVCaptureSessionPreset640x480"
                // The preset string literal IS the constant value on macOS
                let _: () = msg_send![session, setSessionPreset:
                    NSString::alloc(nil).init_str("AVCaptureSessionPreset640x480")];

                let can_add_input: bool = msg_send![session, canAddInput: input];
                if !can_add_input {
                    log::warn!("Cannot add camera input to session");
                    let _: () = msg_send![session, release];
                    return None;
                }
                let _: () = msg_send![session, addInput: input];

                // Create video data output
                let output: id = msg_send![Class::get("AVCaptureVideoDataOutput")?, new];

                // Request BGRA pixel format
                // kCVPixelBufferPixelFormatTypeKey = "PixelFormatType" (CoreVideo constant)
                let pixel_format_key: id = get_cv_pixel_format_key();
                let bgra_value: id = msg_send![Class::get("NSNumber")?,
                    numberWithUnsignedInt: 0x42475241u32]; // kCVPixelFormatType_32BGRA = 'BGRA'
                let settings: id = msg_send![Class::get("NSDictionary")?,
                    dictionaryWithObject: bgra_value forKey: pixel_format_key];
                let _: () = msg_send![output, setVideoSettings: settings];
                let _: () = msg_send![output, setAlwaysDiscardsLateVideoFrames: YES];

                // Create delegate
                let delegate_class = register_delegate_class();
                let delegate: id = msg_send![delegate_class, new];

                let buffer = Arc::new(Mutex::new([0u8; 128 * 112]));
                let has_new_frame = Arc::new(AtomicBool::new(false));

                let context = Box::into_raw(Box::new(DelegateContext {
                    buffer: Arc::clone(&buffer),
                    has_new_frame: Arc::clone(&has_new_frame),
                }));
                (*delegate).set_ivar("_context", context as *mut c_void);

                // Create a dispatch queue for callbacks
                let queue_label = b"com.vibeboy.camera\0".as_ptr() as *const i8;
                let queue: id = dispatch_queue_create(queue_label, std::ptr::null());

                let _: () = msg_send![output, setSampleBufferDelegate: delegate queue: queue];

                let can_add_output: bool = msg_send![session, canAddOutput: output];
                if !can_add_output {
                    log::warn!("Cannot add video output to session");
                    let _: () = msg_send![session, release];
                    let _: () = msg_send![delegate, release];
                    let _: () = msg_send![output, release];
                    drop(Box::from_raw(context));
                    return None;
                }
                let _: () = msg_send![session, addOutput: output];

                // Start capture
                let _: () = msg_send![session, startRunning];

                log::info!("AVFoundation camera capture started");

                Some(CameraCapture {
                    buffer,
                    has_new_frame,
                    session,
                    _delegate: delegate,
                    _context: context,
                })
            }
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

    impl Drop for CameraCapture {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.session, stopRunning];
                let _: () = msg_send![self.session, release];
                let _: () = msg_send![self._delegate, release];
                drop(Box::from_raw(self._context));
            }
            log::info!("AVFoundation camera capture stopped");
        }
    }

    // Force-link AVFoundation so the ObjC runtime finds its classes
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "System", kind = "dylib")]
    unsafe extern "C" {
        fn dispatch_queue_create(label: *const i8, attr: *const c_void) -> id;
    }
}

pub(super) use avf_camera::CameraCapture;
