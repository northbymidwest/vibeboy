//! Save state serialization — converts snapshots to/from portable byte streams.
//!
//! Uses serde + bincode for safe serialization of all emulator state.
//! Frontend-agnostic: produces/consumes `Vec<u8>` which frontends can
//! write to disk, localStorage, IndexedDB, etc.

use crate::snapshot::Snapshot;
use std::io;

const MAGIC: &[u8; 8] = b"VIBEBOY\0";

/// Version hash derived from struct sizes and field layout description.
/// Changes automatically when any struct's layout or field order changes.
const fn fnv1a(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    let mut i = 0;
    while i < data.len() {
        hash ^= data[i] as u32;
        hash = hash.wrapping_mul(0x01000193);
        i += 1;
    }
    hash
}

const fn layout_hash() -> u32 {
    let desc = concat!(
        "Cpu:regs.a,f,b,c,d,e,h,l,sp,pc,ime,ime_pending,halted,halt_bug,speed_switch;",
        "Ppu:fifo,fetcher,vram,oam,regs,frame_buffer,shade_buffer,scanline_sprites;",
        "Apu:ch1-4,frame_seq,master,nr50-52,blip;",
        "Timer:div,tima,tma,tac,internal;",
        "Joypad:select,state;",
        "Serial:sb,sc,interrupt,count,master,mask,cgb;",
        "OamDma:active,source,progress,delay,blocking;",
        "Hdma:src,dst,blocks,mode,active;",
        "Bus:ppu,timer,joypad,apu,wram[8],wram_bank,hram,if,ie,key1,double_speed,",
        "boot_rom_active,model,ff72-75,dmg_compat,serial,oam_dma,hdma,sgb,cart_state;",
        "Sgb:palettes,attrs,border,mask,packets;",
        "Snes:cpu65816,wram,ppu_regs,dma,icd2,io_regs;",
        "Snapshot:cpu,bus,snes,frame_count;",
    );
    let sizes: [usize; 12] = [
        std::mem::size_of::<crate::cpu::Cpu>(),
        std::mem::size_of::<crate::ppu::Ppu>(),
        std::mem::size_of::<crate::apu::Apu>(),
        std::mem::size_of::<crate::timer::Timer>(),
        std::mem::size_of::<crate::joypad::Joypad>(),
        std::mem::size_of::<crate::serial::SerialSnapshot>(),
        std::mem::size_of::<crate::bus::OamDma>(),
        std::mem::size_of::<crate::bus::Hdma>(),
        std::mem::size_of::<crate::sgb::Sgb>(),
        std::mem::size_of::<crate::snapshot::BusSnapshot>(),
        std::mem::size_of::<crate::snapshot::SnesSnapshot>(),
        std::mem::size_of::<crate::snapshot::Snapshot>(),
    ];
    let mut hash = fnv1a(desc.as_bytes());
    let mut i = 0;
    while i < sizes.len() {
        let bytes = (sizes[i] as u32).to_le_bytes();
        let mut j = 0;
        while j < 4 {
            hash ^= bytes[j] as u32;
            hash = hash.wrapping_mul(0x01000193);
            j += 1;
        }
        i += 1;
    }
    hash
}

const VERSION: u32 = layout_hash();

/// Serialize a Snapshot to bytes.
pub fn serialize(snap: &Snapshot) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512 * 1024);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());

    let encoded = bincode::serde::encode_to_vec(snap, bincode::config::standard())
        .expect("snapshot serialization failed");
    buf.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    buf.extend_from_slice(&encoded);
    buf
}

/// Deserialize a Snapshot from bytes.
pub fn deserialize(data: &[u8]) -> io::Result<Snapshot> {
    if data.len() < 16 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "too short"));
    }
    if &data[0..8] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a VibeBoy save state"));
    }
    let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("incompatible save state version (expected {:08X}, got {:08X})", VERSION, version)));
    }
    let payload_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if data.len() < 16 + payload_len {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated save state"));
    }
    bincode::serde::decode_from_slice(&data[16..16 + payload_len], bincode::config::standard())
        .map(|(snap, _)| snap)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("deserialize failed: {e}")))
}
