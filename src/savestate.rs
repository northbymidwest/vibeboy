//! Save state serialization — converts snapshots to/from portable byte streams.
//!
//! Uses serde + bincode for safe serialization of all emulator state.
//! Frontend-agnostic: produces/consumes `Vec<u8>` which frontends can
//! write to disk, localStorage, IndexedDB, etc.

use crate::snapshot::Snapshot;
use std::io;

const MAGIC: &[u8; 8] = b"VIBEBOY\0";

/// Save state format version, written after the magic. Bump it whenever the
/// encoding of `Snapshot` changes: a field added, removed, reordered or
/// retyped anywhere in the snapshot, including the cartridge `CartState`.
/// States with another version are rejected instead of misread. The
/// `encoding_is_pinned_to_format_version` test fails when the encoding
/// changes, as a reminder.
///
/// The encoding is bincode's standard configuration, which writes integers
/// independently of the platform's pointer width (serde encodes `usize` as
/// `u64`), so native and WebAssembly builds read each other's states.
const FORMAT_VERSION: u32 = 8;

/// Serialize a Snapshot to bytes.
pub fn serialize(snap: &Snapshot) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512 * 1024);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

    let encoded = bincode::serde::encode_to_vec(snap, bincode::config::standard())
        .expect("snapshot serialization failed");
    buf.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    buf.extend_from_slice(&encoded);
    buf
}

/// Deserialize a Snapshot from bytes.
pub fn deserialize(data: &[u8]) -> io::Result<Snapshot> {
    let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    if data.len() < 16 {
        return Err(invalid("too short".into()));
    }
    if &data[0..8] != MAGIC {
        return Err(invalid("not a VibeBoy save state".into()));
    }
    let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    if version != FORMAT_VERSION {
        return Err(invalid(format!(
            "incompatible save state version (expected {FORMAT_VERSION}, got {version})"
        )));
    }
    let payload_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let Some(payload) = 16usize
        .checked_add(payload_len)
        .and_then(|end| data.get(16..end))
    else {
        return Err(invalid("truncated save state".into()));
    };
    let (snap, used) = bincode::serde::decode_from_slice(payload, bincode::config::standard())
        .map_err(|e| invalid(format!("deserialize failed: {e}")))?;
    if used != payload.len() {
        return Err(invalid("trailing bytes after save state".into()));
    }
    Ok(snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::emulator::Emulator;
    use crate::model::GbModel;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Wall clock the tests set by hand: one second per emulated frame, so
    /// RTC mappers tick deterministically.
    struct TestClock(AtomicU64);

    impl Clock for TestClock {
        fn now_secs(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
        fn unix_timestamp_secs(&self) -> u64 {
            self.now_secs()
        }
    }

    const START_SECS: u64 = 1_750_000_000;

    /// A cartridge header configuration and the model to run it on.
    struct Cart {
        name: &'static str,
        cart_type: u8,
        ram_size: u8,
        model: GbModel,
    }

    const fn cart(name: &'static str, cart_type: u8, ram_size: u8, model: GbModel) -> Cart {
        Cart {
            name,
            cart_type,
            ram_size,
            model,
        }
    }

    /// One cartridge per mapper type, spread over DMG, SGB and CGB so the
    /// model-specific parts of the snapshot are populated too.
    const CARTS: [Cart; 13] = [
        cart("ROM only", 0x00, 0x00, GbModel::Sgb),
        cart("ROM+RAM", 0x09, 0x02, GbModel::Dmg),
        cart("MBC1", 0x03, 0x03, GbModel::Dmg),
        cart("MBC2", 0x06, 0x00, GbModel::Dmg),
        cart("MBC3+RTC", 0x10, 0x03, GbModel::Dmg),
        cart("MBC5+rumble", 0x1E, 0x03, GbModel::Cgb),
        cart("MBC6", 0x20, 0x02, GbModel::Dmg),
        cart("MBC7", 0x22, 0x00, GbModel::Dmg),
        cart("MMM01", 0x0D, 0x03, GbModel::Dmg),
        cart("Pocket Camera", 0xFC, 0x00, GbModel::Dmg),
        cart("TAMA5", 0xFD, 0x00, GbModel::Dmg),
        cart("HuC1", 0xFF, 0x02, GbModel::Dmg),
        cart("HuC3", 0xFE, 0x02, GbModel::Dmg),
    ];

    /// Program at $0150: select RAM and ROM banks, start a square wave, then
    /// loop writing a counter to cart RAM, SCY and the CH2 frequency, copying
    /// banked ROM and the joypad to WRAM, and latching the MBC3 RTC.
    const PROGRAM: &[u8] = &[
        0x3E, 0x0A, 0xEA, 0x00, 0x00, // ld a,$0A; ld ($0000),a
        0x3E, 0x02, 0xEA, 0x00, 0x20, // ld a,$02; ld ($2000),a
        0x3E, 0x01, 0xEA, 0x00, 0x40, // ld a,$01; ld ($4000),a
        0x3E, 0xF0, 0xE0, 0x17, // ld a,$F0; ldh (NR22),a
        0x3E, 0x87, 0xE0, 0x19, // ld a,$87; ldh (NR24),a
        // loop ($0167):
        0x04, 0x78, // inc b; ld a,b
        0xEA, 0x00, 0xA0, // ld ($A000),a
        0xE0, 0x42, // ldh (SCY),a
        0xE0, 0x18, // ldh (NR23),a
        0xFA, 0x00, 0x40, 0xEA, 0x00, 0xC0, // ld a,($4000); ld ($C000),a
        0xF0, 0x00, 0xEA, 0x01, 0xC0, // ldh a,(P1); ld ($C001),a
        0x3E, 0x00, 0xEA, 0x00, 0x60, // ld a,$00; ld ($6000),a
        0x3E, 0x01, 0xEA, 0x00, 0x60, // ld a,$01; ld ($6000),a
        0xC3, 0x67, 0x01, // jp loop
    ];

    /// 64 KiB ROM filled with a bank-dependent pattern. The header and program
    /// are in both 32 KiB halves, since MMM01 boots from the last 32 KiB.
    fn test_rom(c: &Cart) -> Vec<u8> {
        let mut rom: Vec<u8> = (0..0x10000usize)
            .map(|i| (i.wrapping_mul(7) >> 3) as u8 ^ (i >> 14) as u8)
            .collect();
        for half in [0, 0x8000] {
            let h = &mut rom[half..half + 0x8000];
            h[0x100..0x200].fill(0);
            h[0x100..0x104].copy_from_slice(&[0x00, 0xC3, 0x50, 0x01]); // nop; jp $0150
            h[0x146] = 0x03; // SGB support
            h[0x147] = c.cart_type;
            h[0x149] = c.ram_size;
            h[0x14B] = 0x33;
            h[0x150..0x150 + PROGRAM.len()].copy_from_slice(PROGRAM);
        }
        rom
    }

    /// An emulator plus the clock and frame counter that drive its input.
    struct Rig {
        emu: Emulator,
        clock: Arc<TestClock>,
        frame: u64,
    }

    impl Rig {
        fn new(c: &Cart) -> Self {
            let clock = Arc::new(TestClock(AtomicU64::new(START_SECS)));
            let emu = Emulator::new(test_rom(c), None, c.model, None, clock.clone(), 48_000);
            Rig {
                emu,
                clock,
                frame: 0,
            }
        }

        /// Emulate one frame with input and clock derived from the frame number.
        fn step(&mut self) {
            let f = self.frame;
            self.clock.0.store(START_SECS + f, Ordering::Relaxed);
            self.emu.set_button(Emulator::BTN_A, f % 4 < 2);
            self.emu.set_button(Emulator::BTN_RIGHT, f % 6 < 3);
            self.emu
                .set_button(Emulator::BTN_START, f.is_multiple_of(10));
            self.emu.step_frame();
            self.frame += 1;
        }

        fn run(&mut self, frames: u64) {
            for _ in 0..frames {
                self.step();
            }
        }
    }

    /// Frames emulated before a state is saved.
    const SAVE_AT: u64 = 12;

    /// Save state after `SAVE_AT` frames of a fresh run.
    fn saved(c: &Cart) -> (Rig, Vec<u8>) {
        let mut rig = Rig::new(c);
        rig.run(SAVE_AT);
        let bytes = serialize(&rig.emu.save_snapshot());
        (rig, bytes)
    }

    /// FNV-1a over all bytes, 64-bit.
    fn fnv1a(hash: u64, data: &[u8]) -> u64 {
        data.iter().fold(hash, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3)
        })
    }

    /// Hash of the encoded states of every `CARTS` entry after `SAVE_AT`
    /// frames, for `FORMAT_VERSION`.
    const ENCODING_HASH: u64 = 0x654E_E44E_3368_A764;

    #[test]
    fn encoding_is_pinned_to_format_version() {
        let hash = CARTS.iter().fold(0xCBF2_9CE4_8422_2325, |h, c| {
            let (_, bytes) = saved(c);
            if let Err(e) = deserialize(&bytes) {
                panic!("{}: saved state does not load: {e}", c.name);
            }
            fnv1a(h, &bytes)
        });
        assert!(
            hash == ENCODING_HASH,
            "The save state encoding changed (hash {hash:#018X}). If that is intended, \
             bump FORMAT_VERSION in src/savestate.rs so old states are rejected, and set \
             ENCODING_HASH to the new hash."
        );
    }
}
