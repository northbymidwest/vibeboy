// macOS Apple Silicon accelerometer via IOKit HID
// Reads the AppleSPUHIDDevice accelerometer and exposes a simple poll API.

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDManager.h>
#include <CoreFoundation/CoreFoundation.h>

// Atomic storage for latest accelerometer reading (g-forces)
static volatile float g_accel_x = 0.0f;
static volatile float g_accel_y = 0.0f;
static volatile float g_accel_z = 0.0f;
static volatile bool  g_has_data = false;

static IOHIDDeviceRef g_device = NULL;
static CFRunLoopRef   g_run_loop = NULL;
static pthread_t      g_thread;
static volatile bool  g_running = false;

// Report buffer — Python reference uses 4096
static uint8_t g_report_buf[4096];

// Q16 raw → g-forces
#define ACCEL_SCALE 65536.0f
// Report layout: 22 bytes, xyz int32 LE at offsets 6, 10, 14
#define IMU_REPORT_LEN 22
#define IMU_DATA_OFF   6

// HID input report callback
static void report_callback(void *context, IOReturn result, void *sender,
                            IOHIDReportType type, uint32_t reportID,
                            uint8_t *report, CFIndex reportLength) {
    (void)context; (void)result; (void)sender; (void)type; (void)reportID;

    if (reportLength != IMU_REPORT_LEN) return;

    int32_t raw_x, raw_y, raw_z;
    memcpy(&raw_x, report + IMU_DATA_OFF,     4);
    memcpy(&raw_y, report + IMU_DATA_OFF + 4,  4);
    memcpy(&raw_z, report + IMU_DATA_OFF + 8,  4);

    g_accel_x = (float)raw_x / ACCEL_SCALE;
    g_accel_y = (float)raw_y / ACCEL_SCALE;
    g_accel_z = (float)raw_z / ACCEL_SCALE;
    g_has_data = true;
}

// Background thread: schedule device on run loop
static void *run_loop_thread(void *arg) {
    (void)arg;
    g_run_loop = CFRunLoopGetCurrent();

    IOHIDDeviceRegisterInputReportCallback(
        g_device, g_report_buf, sizeof(g_report_buf),
        report_callback, NULL);
    IOHIDDeviceScheduleWithRunLoop(g_device, g_run_loop, kCFRunLoopDefaultMode);

    // Run until stopped
    while (g_running) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, false);
    }
    return NULL;
}

// Helper: create a CFNumber from an int32
static CFNumberRef cfnum32(int32_t v) {
    return CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &v);
}

// Helper: create a CFString from a C string
static CFStringRef cfstr(const char *s) {
    return CFStringCreateWithCString(kCFAllocatorDefault, s, kCFStringEncodingUTF8);
}

// Wake the SPU drivers so they start producing reports.
// This sets SensorPropertyReportingState=1, SensorPropertyPowerState=1,
// and ReportInterval=1000us on every AppleSPUHIDDriver service.
static bool wake_spu_drivers(void) {
    CFMutableDictionaryRef match = IOServiceMatching("AppleSPUHIDDriver");
    if (!match) return false;

    io_iterator_t iter;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, match, &iter);
    if (kr != KERN_SUCCESS) return false;

    int woke = 0;
    io_service_t svc;
    while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        CFStringRef k1 = cfstr("SensorPropertyReportingState");
        CFStringRef k2 = cfstr("SensorPropertyPowerState");
        CFStringRef k3 = cfstr("ReportInterval");
        CFNumberRef v1 = cfnum32(1);
        CFNumberRef v2 = cfnum32(1);
        CFNumberRef v3 = cfnum32(1000); // microseconds

        IORegistryEntrySetCFProperty(svc, k1, v1);
        IORegistryEntrySetCFProperty(svc, k2, v2);
        IORegistryEntrySetCFProperty(svc, k3, v3);

        CFRelease(k1); CFRelease(k2); CFRelease(k3);
        CFRelease(v1); CFRelease(v2); CFRelease(v3);
        IOObjectRelease(svc);
        woke++;
    }
    IOObjectRelease(iter);
    return woke > 0;
}

static int32_t prop_int(io_service_t svc, const char *key) {
    CFStringRef cfkey = cfstr(key);
    CFTypeRef ref = IORegistryEntryCreateCFProperty(svc, cfkey, kCFAllocatorDefault, 0);
    CFRelease(cfkey);
    if (!ref) return 0;
    int32_t val = 0;
    CFNumberGetValue(ref, kCFNumberSInt32Type, &val);
    CFRelease(ref);
    return val;
}

bool macos_accel_init(void) {
    // Try to wake SPU drivers (may succeed without root on some systems)
    wake_spu_drivers();

    // Find AppleSPUHIDDevice with vendor usage page 0xFF00, usage 3
    CFMutableDictionaryRef match = IOServiceMatching("AppleSPUHIDDevice");
    if (!match) return false;

    io_iterator_t iter;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, match, &iter);
    if (kr != KERN_SUCCESS) return false;

    io_service_t service;
    IOHIDDeviceRef found = NULL;

    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        int32_t usage_page = prop_int(service, "PrimaryUsagePage");
        int32_t usage      = prop_int(service, "PrimaryUsage");

        if (usage_page == 0xFF00 && usage == 3) {
            found = IOHIDDeviceCreate(kCFAllocatorDefault, service);
            IOObjectRelease(service);
            break;
        }
        IOObjectRelease(service);
    }
    IOObjectRelease(iter);

    if (!found) return false;

    // Open the device
    IOReturn ret = IOHIDDeviceOpen(found, kIOHIDOptionsTypeNone);
    if (ret != kIOReturnSuccess) {
        CFRelease(found);
        return false;
    }

    g_device = found;
    g_running = true;

    // Start background thread for CFRunLoop
    if (pthread_create(&g_thread, NULL, run_loop_thread, NULL) != 0) {
        IOHIDDeviceClose(g_device, kIOHIDOptionsTypeNone);
        CFRelease(g_device);
        g_device = NULL;
        g_running = false;
        return false;
    }

    return true;
}

void macos_accel_close(void) {
    if (!g_running) return;
    g_running = false;

    if (g_run_loop) {
        CFRunLoopStop(g_run_loop);
    }
    pthread_join(g_thread, NULL);

    if (g_device) {
        IOHIDDeviceClose(g_device, kIOHIDOptionsTypeNone);
        CFRelease(g_device);
        g_device = NULL;
    }
    g_run_loop = NULL;
}

bool macos_accel_poll(float *x, float *y, float *z) {
    if (!g_has_data) return false;
    *x = g_accel_x;
    *y = g_accel_y;
    *z = g_accel_z;
    return true;
}
