use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub(super) mod avf_camera {
    use super::*;
    use std::os::raw::c_void;

    use dispatch2::DispatchQueue;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, AnyObject, ClassBuilder, ProtocolObject, Sel};
    use objc2::{msg_send, sel, ClassType};
    use objc2_av_foundation::{
        AVCaptureDevice, AVCaptureDeviceInput, AVCaptureInput, AVCaptureOutput,
        AVCaptureSession, AVCaptureVideoDataOutput,
        AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureSessionPreset640x480,
        AVMediaTypeVideo,
    };
    use objc2_foundation::{NSNumber, NSString};

    // CoreMedia / CoreVideo FFI -- no objc2 typed wrappers exist for these C functions
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
        static kCVPixelBufferPixelFormatTypeKey: *mut AnyObject;
    }

    unsafe fn get_cv_pixel_format_key() -> *mut AnyObject {
        unsafe { kCVPixelBufferPixelFormatTypeKey }
    }

    // Shared state passed to the delegate callback via an ivar
    struct DelegateContext {
        buffer: Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: Arc<AtomicBool>,
    }

    // Register a runtime ObjC class that implements the sample buffer delegate.
    // ClassBuilder is required because we need a custom ObjC class with an ivar
    // to hold our Rust context pointer; the delegate callback is invoked by
    // AVFoundation's Objective-C runtime.
    fn register_delegate_class() -> &'static AnyClass {
        use std::sync::Once;
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            let superclass = AnyClass::get(c"NSObject").unwrap();
            let mut builder = ClassBuilder::new(c"VBCameraDelegate", superclass).unwrap();

            // Add protocol conformance so the ProtocolObject cast below is valid
            if let Some(proto) = objc2::runtime::AnyProtocol::get(
                c"AVCaptureVideoDataOutputSampleBufferDelegate",
            ) {
                builder.add_protocol(proto);
            }

            builder.add_ivar::<*mut c_void>(c"_context");

            unsafe {
                builder.add_method(
                    sel!(captureOutput:didOutputSampleBuffer:fromConnection:),
                    capture_output
                        as unsafe extern "C" fn(
                            *mut AnyObject,
                            Sel,
                            *mut AnyObject,
                            *mut AnyObject,
                            *mut AnyObject,
                        ),
                );
            }

            let _ = builder.register();
        });
        AnyClass::get(c"VBCameraDelegate").unwrap()
    }

    unsafe extern "C" fn capture_output(
        this: *mut AnyObject,
        _sel: Sel,
        _output: *mut AnyObject,
        sample_buffer: *mut AnyObject,
        _connection: *mut AnyObject,
    ) {
        unsafe {
            let class = AnyClass::get(c"VBCameraDelegate").unwrap();
            let ivar = class.instance_variable(c"_context").unwrap();
            let ctx_ptr: *mut c_void = *ivar.load_ptr::<*mut c_void>(&*this);
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
        session: Retained<AVCaptureSession>,
        _delegate: Retained<AnyObject>,
        _context: *mut DelegateContext, // leaked; freed on drop
    }

    unsafe impl Send for CameraCapture {}

    impl CameraCapture {
        pub fn start() -> Option<Self> {
            unsafe {
                // Get default video capture device
                let media_type = AVMediaTypeVideo?;
                let device = AVCaptureDevice::defaultDeviceWithMediaType(media_type)?;

                // Create input
                let input = match AVCaptureDeviceInput::deviceInputWithDevice_error(&device) {
                    Ok(input) => input,
                    Err(err) => {
                        log::warn!(
                            "Failed to create camera input: {}",
                            err.localizedDescription()
                        );
                        return None;
                    }
                };

                // Create session
                let session = AVCaptureSession::new();
                let preset = AVCaptureSessionPreset640x480;
                session.setSessionPreset(preset);

                // canAddInput/addInput take &AVCaptureInput; upcast via as_super()
                let input_as_capture_input: &AVCaptureInput = input.as_super();
                if !session.canAddInput(input_as_capture_input) {
                    log::warn!("Cannot add camera input to session");
                    return None;
                }
                session.addInput(input_as_capture_input);

                // Create video data output
                let output = AVCaptureVideoDataOutput::new();

                // Request BGRA pixel format via NSDictionary
                let pixel_format_key = get_cv_pixel_format_key();
                let pixel_format_key_ns: &NSString = &*(pixel_format_key as *const NSString);
                let bgra_value =
                    NSNumber::numberWithUnsignedInt(0x42475241); // kCVPixelFormatType_32BGRA
                // Use msg_send! for NSDictionary creation because the typed
                // dictionaryWithObject_forKey requires &ProtocolObject<dyn NSCopying>
                // and the key is a raw CoreVideo extern NSString pointer.
                let settings: *mut AnyObject = msg_send![
                    AnyClass::get(c"NSDictionary").unwrap(),
                    dictionaryWithObject: &*bgra_value,
                    forKey: pixel_format_key_ns
                ];
                let settings_ref =
                    &*(settings as *const objc2_foundation::NSDictionary<NSString, AnyObject>);
                output.setVideoSettings(Some(settings_ref));
                output.setAlwaysDiscardsLateVideoFrames(true);

                // Create delegate (runtime ObjC class with ivar for Rust context).
                // msg_send![new] is needed because the class is built at runtime
                // and has no Rust type for Retained<T>.
                let delegate_class = register_delegate_class();
                let delegate: Retained<AnyObject> = msg_send![delegate_class, new];

                let buffer = Arc::new(Mutex::new([0u8; 128 * 112]));
                let has_new_frame = Arc::new(AtomicBool::new(false));

                let context = Box::into_raw(Box::new(DelegateContext {
                    buffer: Arc::clone(&buffer),
                    has_new_frame: Arc::clone(&has_new_frame),
                }));
                let ivar = delegate_class.instance_variable(c"_context").unwrap();
                let delegate_ptr = Retained::as_ptr(&delegate) as *mut AnyObject;
                let ptr = ivar.load_ptr::<*mut c_void>(&*delegate_ptr);
                *ptr = context as *mut c_void;

                // Create a serial dispatch queue for callbacks
                let queue = DispatchQueue::new("com.vibeboy.camera", None);

                // setSampleBufferDelegate_queue requires
                // &ProtocolObject<dyn AVCaptureVideoDataOutputSampleBufferDelegate>.
                // Our runtime class conforms via builder.add_protocol(), so the
                // pointer cast is sound.
                let delegate_proto: &ProtocolObject<
                    dyn AVCaptureVideoDataOutputSampleBufferDelegate,
                > = &*(delegate_ptr
                    as *const ProtocolObject<
                        dyn AVCaptureVideoDataOutputSampleBufferDelegate,
                    >);
                output.setSampleBufferDelegate_queue(Some(delegate_proto), Some(&queue));

                // canAddOutput/addOutput take &AVCaptureOutput; upcast via as_super()
                let output_as_capture_output: &AVCaptureOutput = output.as_super();
                if !session.canAddOutput(output_as_capture_output) {
                    log::warn!("Cannot add video output to session");
                    drop(Box::from_raw(context));
                    return None;
                }
                session.addOutput(output_as_capture_output);

                // Start capture
                session.startRunning();

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
                self.session.stopRunning();
                drop(Box::from_raw(self._context));
            }
            log::info!("AVFoundation camera capture stopped");
        }
    }

    // Force-link AVFoundation so the ObjC runtime finds its classes
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {}
}

pub(super) use avf_camera::CameraCapture;
