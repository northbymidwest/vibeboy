//! macOS Apple Silicon accelerometer via IOKit HID.
//! Reads the AppleSPUHIDDevice accelerometer and exposes a simple poll API.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

// Opaque CoreFoundation / IOKit types
type CFAllocatorRef = *const std::ffi::c_void;
type CFStringRef = *const std::ffi::c_void;
type CFNumberRef = *const std::ffi::c_void;
type CFTypeRef = *const std::ffi::c_void;
type CFMutableDictionaryRef = *mut std::ffi::c_void;
type CFRunLoopRef = *const std::ffi::c_void;
type CFStringEncoding = u32;
type CFNumberType = i64;
type CFIndex = isize;
type IOHIDDeviceRef = *const std::ffi::c_void;
type IOHIDReportType = u32;
type IOReturn = i32;

type IOOptionBits = u32;

// mach_port_t is u32 on macOS
type MachPort = u32;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFAllocatorDefault: CFAllocatorRef;
    static kCFRunLoopDefaultMode: CFStringRef;

    fn CFStringCreateWithCString(
        alloc: CFAllocatorRef,
        c_str: *const u8,
        encoding: CFStringEncoding,
    ) -> CFStringRef;
    fn CFNumberCreate(
        alloc: CFAllocatorRef,
        the_type: CFNumberType,
        value_ptr: *const i32,
    ) -> CFNumberRef;
    fn CFNumberGetValue(number: CFNumberRef, the_type: CFNumberType, value_ptr: *mut i32) -> bool;
    fn CFRelease(cf: *const std::ffi::c_void);
    fn CFRunLoopGetCurrent() -> CFRunLoopRef;
    fn CFRunLoopRunInMode(mode: CFStringRef, seconds: f64, return_after_source: bool) -> i32;
    fn CFRunLoopStop(rl: CFRunLoopRef);
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    static kIOMainPortDefault: MachPort;

    fn IOServiceMatching(name: *const u8) -> CFMutableDictionaryRef;
    fn IOServiceGetMatchingServices(
        main_port: MachPort,
        matching: CFMutableDictionaryRef,
        existing: *mut u32,
    ) -> i32;
    fn IOIteratorNext(iterator: u32) -> u32;
    fn IOObjectRelease(object: u32) -> i32;
    fn IORegistryEntrySetCFProperty(
        entry: u32,
        property_name: CFStringRef,
        property: CFTypeRef,
    ) -> i32;
    fn IORegistryEntryCreateCFProperty(
        entry: u32,
        key: CFStringRef,
        allocator: CFAllocatorRef,
        options: IOOptionBits,
    ) -> CFTypeRef;
    fn IOHIDDeviceCreate(allocator: CFAllocatorRef, service: u32) -> IOHIDDeviceRef;
    fn IOHIDDeviceOpen(device: IOHIDDeviceRef, options: IOOptionBits) -> IOReturn;
    fn IOHIDDeviceClose(device: IOHIDDeviceRef, options: IOOptionBits) -> IOReturn;
    fn IOHIDDeviceRegisterInputReportCallback(
        device: IOHIDDeviceRef,
        report: *mut u8,
        report_length: CFIndex,
        callback: unsafe extern "C" fn(
            context: *mut std::ffi::c_void,
            result: IOReturn,
            sender: *mut std::ffi::c_void,
            report_type: IOHIDReportType,
            report_id: u32,
            report: *mut u8,
            report_length: CFIndex,
        ),
        context: *mut std::ffi::c_void,
    );
    fn IOHIDDeviceScheduleWithRunLoop(
        device: IOHIDDeviceRef,
        run_loop: CFRunLoopRef,
        run_loop_mode: CFStringRef,
    );
}

const KERN_SUCCESS: i32 = 0;
const IO_OBJECT_NULL: u32 = 0;
const K_IO_RETURN_SUCCESS: IOReturn = 0;
const K_IO_HID_OPTIONS_TYPE_NONE: IOOptionBits = 0;
const K_CF_STRING_ENCODING_UTF8: CFStringEncoding = 0x0800_0100;
const K_CF_NUMBER_SINT32_TYPE: CFNumberType = 3;

const ACCEL_SCALE: f32 = 65536.0;
const IMU_REPORT_LEN: isize = 22;
const IMU_DATA_OFF: usize = 6;

// Global state — atomics replace C volatile
static HAS_DATA: AtomicBool = AtomicBool::new(false);
static RUNNING: AtomicBool = AtomicBool::new(false);
// Store f32 as u32 bits for atomic access
static ACCEL_X: AtomicU32 = AtomicU32::new(0);
static ACCEL_Y: AtomicU32 = AtomicU32::new(0);
static ACCEL_Z: AtomicU32 = AtomicU32::new(0);

use std::sync::Mutex;
static DEVICE: Mutex<usize> = Mutex::new(0); // stores IOHIDDeviceRef as usize
static RUN_LOOP: Mutex<usize> = Mutex::new(0); // stores CFRunLoopRef as usize
static THREAD: Mutex<Option<thread::JoinHandle<()>>> = Mutex::new(None);

fn cfstr(s: &[u8]) -> CFStringRef {
    unsafe { CFStringCreateWithCString(kCFAllocatorDefault, s.as_ptr(), K_CF_STRING_ENCODING_UTF8) }
}

fn cfnum32(v: i32) -> CFNumberRef {
    unsafe { CFNumberCreate(kCFAllocatorDefault, K_CF_NUMBER_SINT32_TYPE, &v) }
}

/// HID input report callback — called from the CFRunLoop thread.
unsafe extern "C" fn report_callback(
    _context: *mut std::ffi::c_void,
    _result: IOReturn,
    _sender: *mut std::ffi::c_void,
    _report_type: IOHIDReportType,
    _report_id: u32,
    report: *mut u8,
    report_length: CFIndex,
) {
    unsafe {
        if report_length != IMU_REPORT_LEN {
            return;
        }

        let mut raw_x: i32 = 0;
        let mut raw_y: i32 = 0;
        let mut raw_z: i32 = 0;
        std::ptr::copy_nonoverlapping(
            report.add(IMU_DATA_OFF) as *const u8,
            &mut raw_x as *mut i32 as *mut u8,
            4,
        );
        std::ptr::copy_nonoverlapping(
            report.add(IMU_DATA_OFF + 4) as *const u8,
            &mut raw_y as *mut i32 as *mut u8,
            4,
        );
        std::ptr::copy_nonoverlapping(
            report.add(IMU_DATA_OFF + 8) as *const u8,
            &mut raw_z as *mut i32 as *mut u8,
            4,
        );

        ACCEL_X.store(f32::to_bits(raw_x as f32 / ACCEL_SCALE), Ordering::Relaxed);
        ACCEL_Y.store(f32::to_bits(raw_y as f32 / ACCEL_SCALE), Ordering::Relaxed);
        ACCEL_Z.store(f32::to_bits(raw_z as f32 / ACCEL_SCALE), Ordering::Relaxed);
        HAS_DATA.store(true, Ordering::Release);
    }
}

/// Wake SPU drivers so they start producing reports.
unsafe fn wake_spu_drivers() -> bool {
    unsafe {
        let match_dict = IOServiceMatching(b"AppleSPUHIDDriver\0".as_ptr());
        if match_dict.is_null() {
            return false;
        }

        let mut iter: u32 = 0;
        let kr = IOServiceGetMatchingServices(kIOMainPortDefault, match_dict, &mut iter);
        if kr != KERN_SUCCESS {
            return false;
        }

        let mut woke = 0;
        loop {
            let svc = IOIteratorNext(iter);
            if svc == IO_OBJECT_NULL {
                break;
            }

            let k1 = cfstr(b"SensorPropertyReportingState\0");
            let k2 = cfstr(b"SensorPropertyPowerState\0");
            let k3 = cfstr(b"ReportInterval\0");
            let v1 = cfnum32(1);
            let v2 = cfnum32(1);
            let v3 = cfnum32(1000);

            IORegistryEntrySetCFProperty(svc, k1, v1);
            IORegistryEntrySetCFProperty(svc, k2, v2);
            IORegistryEntrySetCFProperty(svc, k3, v3);

            CFRelease(k1);
            CFRelease(k2);
            CFRelease(k3);
            CFRelease(v1);
            CFRelease(v2);
            CFRelease(v3);
            IOObjectRelease(svc);
            woke += 1;
        }
        IOObjectRelease(iter);
        woke > 0
    }
}

/// Read an integer property from an IOService.
unsafe fn prop_int(svc: u32, key: &[u8]) -> i32 {
    unsafe {
        let cfkey = cfstr(key);
        let r = IORegistryEntryCreateCFProperty(svc, cfkey, kCFAllocatorDefault, 0);
        CFRelease(cfkey);
        if r.is_null() {
            return 0;
        }
        let mut val: i32 = 0;
        CFNumberGetValue(r, K_CF_NUMBER_SINT32_TYPE, &mut val);
        CFRelease(r);
        val
    }
}

/// Initialize the macOS native accelerometer. Returns true on success.
pub fn init() -> bool {
    unsafe {
        wake_spu_drivers();

        let match_dict = IOServiceMatching(b"AppleSPUHIDDevice\0".as_ptr());
        if match_dict.is_null() {
            return false;
        }

        let mut iter: u32 = 0;
        let kr = IOServiceGetMatchingServices(kIOMainPortDefault, match_dict, &mut iter);
        if kr != KERN_SUCCESS {
            return false;
        }

        let mut found: IOHIDDeviceRef = std::ptr::null();
        loop {
            let service = IOIteratorNext(iter);
            if service == IO_OBJECT_NULL {
                break;
            }

            let usage_page = prop_int(service, b"PrimaryUsagePage\0");
            let usage = prop_int(service, b"PrimaryUsage\0");

            if usage_page == 0xFF00 && usage == 3 {
                found = IOHIDDeviceCreate(kCFAllocatorDefault, service);
                IOObjectRelease(service);
                break;
            }
            IOObjectRelease(service);
        }
        IOObjectRelease(iter);

        if found.is_null() {
            return false;
        }

        let ret = IOHIDDeviceOpen(found, K_IO_HID_OPTIONS_TYPE_NONE);
        if ret != K_IO_RETURN_SUCCESS {
            CFRelease(found);
            return false;
        }

        *DEVICE.lock().unwrap() = found as usize;
        RUNNING.store(true, Ordering::SeqCst);

        let handle = thread::spawn(run_loop_thread);
        *THREAD.lock().unwrap() = Some(handle);

        true
    }
}

/// Background thread: register callback and run CFRunLoop.
fn run_loop_thread() {
    unsafe {
        let rl = CFRunLoopGetCurrent();
        *RUN_LOOP.lock().unwrap() = rl as usize;

        let device = *DEVICE.lock().unwrap() as IOHIDDeviceRef;

        // Report buffer — lives on this thread's stack for the lifetime of the run loop
        let mut report_buf = [0u8; 4096];

        IOHIDDeviceRegisterInputReportCallback(
            device,
            report_buf.as_mut_ptr(),
            report_buf.len() as CFIndex,
            report_callback,
            std::ptr::null_mut(),
        );
        IOHIDDeviceScheduleWithRunLoop(device, rl, kCFRunLoopDefaultMode);

        while RUNNING.load(Ordering::SeqCst) {
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, false);
        }
    }
}

/// Stop the accelerometer and clean up.
pub fn close() {
    if !RUNNING.load(Ordering::SeqCst) {
        return;
    }
    RUNNING.store(false, Ordering::SeqCst);

    {
        let rl = *RUN_LOOP.lock().unwrap() as CFRunLoopRef;
        if !rl.is_null() {
            unsafe {
                CFRunLoopStop(rl);
            }
        }
    }

    if let Some(handle) = THREAD.lock().unwrap().take() {
        let _ = handle.join();
    }

    {
        let dev = *DEVICE.lock().unwrap() as IOHIDDeviceRef;
        if !dev.is_null() {
            unsafe {
                IOHIDDeviceClose(dev, K_IO_HID_OPTIONS_TYPE_NONE);
                CFRelease(dev);
            }
        }
        *DEVICE.lock().unwrap() = 0;
    }
    *RUN_LOOP.lock().unwrap() = 0;
}

/// Poll the latest accelerometer reading. Returns Some((x, y, z)) if data is available.
pub fn poll() -> Option<(f32, f32, f32)> {
    if !HAS_DATA.load(Ordering::Acquire) {
        return None;
    }
    let x = f32::from_bits(ACCEL_X.load(Ordering::Relaxed));
    let y = f32::from_bits(ACCEL_Y.load(Ordering::Relaxed));
    let z = f32::from_bits(ACCEL_Z.load(Ordering::Relaxed));
    Some((x, y, z))
}
