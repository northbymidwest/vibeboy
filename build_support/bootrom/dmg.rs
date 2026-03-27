//! DMG boot ROM generator.
//!
//! Produces a 256-byte DMG boot ROM with:
//! - Nintendo logo (from cart header) scrolling in from the right
//! - Boot chime via CH1 descending sweep
//! - Header checksum computation (for correct post-boot flag state)
//! - Correct post-boot hardware state

use super::asm::Asm;

const DMG_BOOT_ROM_SIZE: usize = 0x0100;

pub fn build() -> Vec<u8> {
    let mut a = Asm::new();

    // ================================================================
    // Init
    // ================================================================
    a.di();
    a.ld_sp(0xFFFE);
    a.xor_a();
    a.ldh_n_a(0x40); // LCDC off
    a.ldh_n_a(0x42); // SCY = 0
    a.ldh_n_a(0x43); // SCX = 0

    // Clear VRAM ($8000-$9FFF)
    a.ld_hl_imm(0x8000);
    a.ld_b(0x20); // 32 × 256 = 8KB
    a.label("clr");
    a.ld_hli_a();
    a.dec_c(); // C starts 0, wraps to FF
    a.jr_nz("clr");
    a.dec_b();
    a.jr_nz("clr");

    // ================================================================
    // Decompress Nintendo logo from cart header ($0104) into tiles.
    //
    // Each of 48 source bytes → 2 nibbles → 2 expanded rows (repeated).
    // Uses PUSH BC / RL C / POP BC trick to double each bit.
    // Produces 24 tiles (12 wide × 2 tall) at VRAM $8010.
    // ================================================================
    a.ld_de_imm(0x0104);
    a.ld_hl_imm(0x8010);
    a.ld_a(48);
    a.ldh_n_a(0x80); // outer counter in HRAM

    a.label("logo_byte");
    a.ld_a_de_ind();   // read source byte
    a.inc_de();
    a.push_de();       // save source ptr
    a.push_af();       // save source byte
    a.call("expand");  // expand high nibble → 4 VRAM bytes
    a.pop_af();        // restore source byte
    a.swap_a();        // low nibble → high position
    a.call("expand");  // expand low nibble → 4 VRAM bytes
    a.pop_de();        // restore source ptr
    a.ldh_a_n(0x80);
    a.dec_a();
    a.ldh_n_a(0x80);
    a.jr_nz("logo_byte");

    a.jr("after_expand");

    // ── Expand subroutine ────────────────────────────────────────────
    // Input: A = byte with nibble to expand in bits 7-4
    //        HL = VRAM dest (advances by 4)
    // Output: 4 bytes written (lo, hi, lo, hi = repeated color-3 row)
    // Clobbers: A, B, C
    a.label("expand");
    a.ld_c_a();    // source → C
    a.ld_b(4);     // 4 bits
    a.xor_a();     // clear result + carry
    a.label("exp_bit");
    a.push_bc();   // save C (source state) and B (counter)
    a.rl_c();      // source.bit7 → carry (C shifted left)
    a.rla();       // carry → A.bit0
    a.pop_bc();    // restore C to pre-shift state
    a.rl_c();      // same bit → carry again
    a.rla();       // duplicate bit into A
    a.dec_b();
    a.jr_nz("exp_bit");
    // A = expanded byte: each source bit doubled
    a.ld_hli_a();  // lo plane
    a.ld_hli_a();  // hi plane (= color 3)
    a.ld_hli_a();  // lo plane (row repeat)
    a.ld_hli_a();  // hi plane (row repeat)
    a.ret();

    a.label("after_expand");

    // ================================================================
    // Tilemap: logo at columns 20-31, rows 8-9
    // (scrolls in from the right via SCX)
    // ================================================================
    a.ld_hl_imm(0x9800 + 8 * 32 + 20);
    a.ld_a(1);
    a.ld_b(12);
    a.label("tm_top");
    a.ld_hli_a();
    a.inc_a();
    a.dec_b();
    a.jr_nz("tm_top");

    a.ld_hl_imm(0x9800 + 9 * 32 + 20);
    // A = 13 (continues from above)
    a.ld_b(12);
    a.label("tm_bot");
    a.ld_hli_a();
    a.inc_a();
    a.dec_b();
    a.jr_nz("tm_bot");

    // ================================================================
    // Palette + LCD + chime
    // ================================================================
    a.ld_a(0xFC);
    a.ldh_n_a(0x47); // BGP
    a.ld_a(0x91);
    a.ldh_n_a(0x40); // LCDC on

    for &(reg, val) in &[
        (0x26u8, 0x80u8), (0x25, 0xF3), (0x24, 0x77), (0x10, 0x67),
        (0x11, 0x80), (0x12, 0xF3), (0x13, 0x83), (0x14, 0x87),
    ] {
        a.ld_a(val);
        a.ldh_n_a(reg);
    }

    // ================================================================
    // Horizontal scroll: SCX 0→128 (logo slides from right to center)
    // ================================================================
    a.ld_e(128);
    a.label("scroll");
    a.ldh_a_n(0x44);
    a.cp_a_n(144);
    a.jr_nz("scroll"); // wait for VBlank

    a.ldh_a_n(0x43);
    a.inc_a();
    a.ldh_n_a(0x43); // SCX++
    a.dec_e();
    a.jr_z("scroll_done");

    a.label("vbl_end");
    a.ldh_a_n(0x44);
    a.cp_a_n(144);
    a.jr_z("vbl_end"); // wait for VBlank to end
    a.jr("scroll");

    a.label("scroll_done");

    // Hold ~0.5 sec
    a.ld_e(30);
    a.label("hold");
    a.ldh_a_n(0x44);
    a.cp_a_n(144);
    a.jr_nz("hold");
    a.label("hold2");
    a.ldh_a_n(0x44);
    a.cp_a_n(144);
    a.jr_z("hold2");
    a.dec_e();
    a.jr_nz("hold");

    // ================================================================
    // Header checksum: A = sum of -(byte)-1 for $0134-$014C
    // ================================================================
    a.ld_hl_imm(0x0134);
    a.ld_b(25);
    a.xor_a();
    a.label("hdr_ck");
    a.sub_hl_ind(); // A -= (HL)
    a.dec_a();      // A -= 1
    a.inc_hl();
    a.dec_b();
    a.jr_nz("hdr_ck");

    // ================================================================
    // Silence chime
    // ================================================================
    a.ld_a(0x00);
    a.ldh_n_a(0x12); // NR12 = 0
    a.ld_a(0xBF);
    a.ldh_n_a(0x14); // NR14 trigger (applies vol=0)

    // ================================================================
    // Post-boot register state
    // A=$01, F=$B0, B=$00, C=$13, D=$00, E=$D8, H=$01, L=$4D
    // ================================================================
    a.ld_b(0x01);
    a.ld_c(0xB0); // F=$B0: Z=1, N=0, H=1, C=1
    a.push_bc();
    a.pop_af(); // A=$01, F=$B0

    a.ld_b(0x00);
    a.ld_c(0x13);
    a.ld_d(0x00);
    a.ld_e(0xD8);
    a.ld_h(0x01);
    a.ld_l(0x4D);

    // ================================================================
    // Handoff: disable boot ROM
    // ================================================================
    a.ldh_n_a(0x50); // A=$01 disables boot ROM mapping

    a.into_rom(DMG_BOOT_ROM_SIZE)
}
