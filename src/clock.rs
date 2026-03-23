//! Clock abstraction for RTC cartridges.
//!
//! Provides wall-clock time to cartridges that have real-time clocks (MBC3, HuC3, TAMA5).
//! Frontends supply a concrete implementation; the core emulator never reads the system clock.

/// Provides wall-clock time for RTC-equipped cartridges.
pub trait Clock: Send + Sync {
    /// Current time in seconds since an arbitrary epoch (used for elapsed-time calculation).
    fn now_secs(&self) -> u64;
    /// Unix timestamp in seconds (used for save-file RTC persistence).
    fn unix_timestamp_secs(&self) -> u64;
}

/// System clock using `std::time`. Used by all native frontends.
#[cfg(not(target_arch = "wasm32"))]
pub struct SystemClock;

#[cfg(not(target_arch = "wasm32"))]
impl Clock for SystemClock {
    fn now_secs(&self) -> u64 {
        // Use a monotonic-ish counter based on UNIX_EPOCH for simplicity.
        // std::time::Instant can't be converted to seconds-since-epoch,
        // so we use SystemTime which is close enough for RTC purposes.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn unix_timestamp_secs(&self) -> u64 {
        self.now_secs()
    }
}

/// JavaScript clock for WebAssembly builds.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub struct JsClock;

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl Clock for JsClock {
    fn now_secs(&self) -> u64 {
        (js_sys::Date::now() / 1000.0) as u64
    }

    fn unix_timestamp_secs(&self) -> u64 {
        self.now_secs()
    }
}

/// Default clock for the current platform.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_clock() -> std::sync::Arc<dyn Clock> {
    std::sync::Arc::new(SystemClock)
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub fn default_clock() -> std::sync::Arc<dyn Clock> {
    std::sync::Arc::new(JsClock)
}
