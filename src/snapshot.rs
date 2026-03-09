/// Snapshot infrastructure for rewind and save states.
///
/// A `Snapshot` captures the entire emulator state at one point in time.
/// It can be restored to rewind gameplay or load a save state.

use crate::apu::Apu;
use crate::bus::{Hdma, OamDma};
use crate::cpu::Cpu;
use crate::joypad::Joypad;
use crate::model::GbModel;
use crate::ppu::Ppu;
use crate::serial::SerialSnapshot;
use crate::sgb::Sgb;
use crate::timer::Timer;

/// Full snapshot of the Bus (everything except ROM and the serial device).
#[derive(Clone)]
pub struct BusSnapshot {
    pub ppu: Ppu,
    pub timer: Timer,
    pub joypad: Joypad,
    pub apu: Apu,
    pub wram: [[u8; 0x1000]; 8],
    pub wram_bank: usize,
    pub hram: [u8; 0x7F],
    pub if_: u8,
    pub ie: u8,
    pub serial: SerialSnapshot,
    pub key1: u8,
    pub double_speed: bool,
    pub oam_dma: OamDma,
    pub hdma: Hdma,
    pub boot_rom_active: bool,
    pub model: GbModel,
    pub sgb: Option<Sgb>,
    pub cart_state: Vec<u8>,
    pub ff72: u8,
    pub ff73: u8,
    pub ff74: u8,
    pub ff75: u8,
    pub dmg_compat: bool,
}

/// Complete emulator snapshot.
#[derive(Clone)]
pub struct Snapshot {
    pub cpu: Cpu,
    pub bus: BusSnapshot,
    pub snes: Option<SnesSnapshot>,
    pub frame_count: u64,
}

/// SNES subsystem snapshot (everything except the ROM, which is read-only).
#[derive(Clone)]
pub struct SnesSnapshot {
    pub cpu: crate::snes::cpu::Cpu65816,
    pub bus_wram: Vec<u8>,
    pub bus_ppu: crate::snes::ppu_regs::SnesPpuRegs,
    pub bus_dma: crate::snes::dma::DmaController,
    pub bus_icd2: crate::snes::icd2::Icd2,
    // CPU I/O registers
    pub nmitimen: u8,
    pub rdnmi: u8,
    pub timeup: u8,
    pub hvbjoy: u8,
    pub joy1: u16,
    pub joy2: u16,
    pub wrmpya: u8,
    pub wrmpyb: u8,
    pub wrdiv: u16,
    pub wrdivb: u8,
    pub rddiv: u16,
    pub rdmpy: u16,
    pub wrio: u8,
    pub htime: u16,
    pub vtime: u16,
    pub mdmaen: u8,
    pub hdmaen: u8,
    pub memsel: u8,
    pub apu_out: [u8; 4],
    pub apu_in: [u8; 4],
    pub apu_state: u8,
    pub apu_last_counter: u8,
    pub apu_port1_val: u8,
    pub apu_echo_pending: bool,
    pub nmi_fire_count: u64,
    pub wmadd: u32,
    pub active_display_start: u64,
    pub current_cpu_cycles: u64,
    pub in_vblank: bool,
    pub irq_ack: bool,
    pub frame_count: u64,
}
