//! CGB boot ROM generator.
//!
//! Produces a 2304-byte CGB boot ROM with:
//! - Rainbow "VIBEBOY" logo using per-letter CGB BG palette attributes
//! - Boot chime via CH1 descending sweep
//! - DMG game palette selection matching the official algorithm
//! - Button combos for manual palette override with real-time swatch preview
//! - Correct post-boot hardware state

use super::asm::Asm;
use super::data::*;

/// CGB boot ROM total size: 0x0000-0x00FF + 0x0200-0x08FF = 2304 bytes.
const CGB_BOOT_ROM_SIZE: usize = 0x0900;

/// Generate the complete CGB boot ROM. If `agb` is true, sets B=0x01
/// in the post-boot register state (GBA running in GBC mode).
pub fn build(agb: bool) -> Vec<u8> {
    let mut a = Asm::new();

    // ================================================================
    // Region 1: 0x0000-0x00FF (DMG-compatible boot region)
    // ================================================================
    a.di();
    a.ld_sp(0xFFFE);
    a.xor_a();
    a.ldh_n_a(0x40); // LCDC off
    a.ldh_n_a(0x0F); // IF clear
    a.jp("cgb_main");

    a.pad_to(0x00FC);
    a.label("handoff");
    a.ld_a(0x11);
    a.ldh_n_a(0x50); // disable boot ROM

    // ================================================================
    // Region 2: 0x0200-0x08FF (CGB-extended boot region)
    // ================================================================
    a.pad_to(0x0200);
    a.label("cgb_main");

    // --- Clear VRAM ---
    a.call("sub_clear_vram");

    // --- Tile data ---
    a.jp("after_tiles");
    a.label("tile_data");
    let tiles = font_tiles();
    a.db(&tiles);
    a.label("after_tiles");

    // Copy tile data to VRAM at $8010 (tiles 1-10)
    a.ld_hl_imm(0x8010);
    a.ld_de_label("tile_data");
    a.ld_b((tiles.len() & 0xFF) as u8);
    a.label("cpy");
    a.ld_a_de_ind();
    a.ld_hli_a();
    a.inc_de();
    a.dec_b();
    a.jr_nz("cpy");

    // --- Tilemap + attributes ---
    let tmap = 0x9800 + 8 * 32 + 6; // "VIBEBOY" at row 8, col 6
    let swatch = 0x9800 + 10 * 32 + 8; // swatch bar at row 10, col 8

    // Bank 0: tile indices
    a.ld_hl_imm(tmap);
    a.ld_a(1);
    for i in 0..7 {
        a.ld_hli_a();
        if i < 6 { a.inc_a(); }
    }
    // Swatch: tiles 0 (color0), 8 (color1), 9 (color2), 10 (color3)
    a.ld_hl_imm(swatch);
    a.xor_a(); a.ld_hli_a();
    a.ld_a(8); a.ld_hli_a();
    a.ld_a(9); a.ld_hli_a();
    a.ld_a(10); a.ld_hli_a();

    // Bank 1: palette attributes
    a.ld_a(1);
    a.ldh_n_a(0x4F); // VBK = 1
    a.ld_hl_imm(tmap);
    for p in 0..7u8 {
        a.ld_a(p);
        a.ld_hli_a();
    }
    // Swatch tiles: all use palette 7
    a.ld_hl_imm(swatch);
    a.ld_a(7);
    a.ld_hli_a(); a.ld_hli_a(); a.ld_hli_a(); a.ld_hli_a();
    a.xor_a();
    a.ldh_n_a(0x4F); // VBK = 0

    // --- Rainbow palettes (BG palettes 0-6) ---
    a.ld_a(0x80);
    a.ldh_n_a(0x68); // BCPS: auto-inc
    for &color in &LOGO_COLORS {
        let r = color & 0x1F;
        let g = (color >> 5) & 0x1F;
        let b = (color >> 10) & 0x1F;
        let light = ((r + 31) / 2) | (((g + 31) / 2) << 5) | (((b + 31) / 2) << 10);
        for c in &[0x7FFF, light, color, color] {
            a.ld_a(*c as u8); a.ldh_n_a(0x69);
            a.ld_a((*c >> 8) as u8); a.ldh_n_a(0x69);
        }
    }

    // --- Palette 7: initially white (swatch invisible until combo held) ---
    a.ld_a(0x80 | 56); // BGPI: auto-inc, palette 7
    a.ldh_n_a(0x68);
    for _ in 0..4 {
        a.ld_a(0xFF); a.ldh_n_a(0x69);
        a.ld_a(0x7F); a.ldh_n_a(0x69);
    }

    // --- LCD on, boot chime ---
    a.ld_a(0x91); a.ldh_n_a(0x40); // LCDC
    for &(reg, val) in &[
        (0x26u8, 0x80u8), (0x24, 0x77), (0x25, 0xF3), (0x10, 0x67),
        (0x11, 0x80), (0x12, 0xF3), (0x13, 0x83), (0x14, 0x87),
    ] {
        a.ld_a(val); a.ldh_n_a(reg);
    }

    // --- Wait ~2 sec, polling joypad once per frame ---
    // HRAM $86 = saved combo, HRAM $87 = previous joypad state
    a.xor_a();
    a.ldh_n_a(0x86);
    a.ldh_n_a(0x87);

    a.ld_e(120);
    a.label("w1");
    // Wait for VBlank start (LY == 144)
    a.ldh_a_n(0x44); a.cp_a_n(144); a.jr_nz("w1");

    // Poll joypad
    a.ld_a(0x20); a.ldh_n_a(0x00); // select D-pad
    a.ldh_a_n(0x00); // debounce
    a.ldh_a_n(0x00); // stable read
    a.and_n(0x0F);
    a.cpl();
    a.and_n(0x0F);
    a.swap_a(); // D-pad in high nibble
    a.ld_b_a();
    a.ld_a(0x10); a.ldh_n_a(0x00); // select buttons
    a.ldh_a_n(0x00);
    a.ldh_a_n(0x00);
    a.and_n(0x03);
    a.cpl();
    a.and_n(0x03);
    a.or_b(); // combine
    a.jr_z("w_no_btn");
    a.ldh_n_a(0x86); // save combo
    // Check if combo changed
    a.ld_c_a();
    a.ldh_a_n(0x87); // previous
    a.cp_c();
    a.ld_a_c();
    a.ldh_n_a(0x87); // update previous
    a.jr_z("w_same");
    // New combo → reset timer + update swatch
    a.ld_e(120);
    a.label("w_same");
    a.push_de();
    a.call("sub_update_swatch");
    a.pop_de();
    a.label("w_no_btn");
    a.ld_a(0x30); a.ldh_n_a(0x00); // reset joypad

    // Wait for VBlank to end
    a.label("w2");
    a.ldh_a_n(0x44); a.cp_a_n(144); a.jr_z("w2");
    a.dec_e(); a.jr_nz("w1");

    // --- Final audio ---
    a.ld_a(0xBF); a.ldh_n_a(0x11);
    a.ld_a(0xF3); a.ldh_n_a(0x12);
    a.ld_a(0xBF); a.ldh_n_a(0x14);

    // --- LCD off ---
    a.label("off_vbl");
    a.ldh_a_n(0x44); a.cp_a_n(144); a.jr_nz("off_vbl");
    a.xor_a(); a.ldh_n_a(0x40);

    // --- Clear VRAM for game ---
    a.call("sub_clear_vram");

    // --- Hardware cleanup ---
    // Clear OAM
    a.ld_hl_imm(0xFE00);
    a.ld_b(160);
    a.xor_a();
    a.label("clr_oam");
    a.ld_hli_a();
    a.dec_b();
    a.jr_nz("clr_oam");

    // Clear HRAM scratch ($FF80-$FF86)
    a.xor_a();
    for i in 0x80..0x87u8 {
        a.ldh_n_a(i);
    }

    // CH1: zero volume, set post-boot register values
    a.ld_a(0x80); a.ldh_n_a(0x10); // NR10
    a.xor_a(); a.ldh_n_a(0x12); // NR12 = 0 → vol=0 on trigger
    a.ld_a(0xBF); a.ldh_n_a(0x14); // trigger
    a.ld_a(0xBF); a.ldh_n_a(0x11); // NR11
    a.ld_a(0xF3); a.ldh_n_a(0x12); // NR12

    // Joypad + IF
    a.ld_a(0x30); a.ldh_n_a(0x00); // P1
    a.ld_a(0x01); a.ldh_n_a(0x0F); // IF

    // --- CGB vs DMG? ---
    a.ld_a_nn(0x0143);
    a.and_n(0x80);
    a.jp_nz("is_cgb");

    // ============================================================
    // DMG game: palette selection
    // ============================================================
    a.ld_a(0xFC); a.ldh_n_a(0x47); // BGP

    // Check saved joypad combo
    a.ldh_a_n(0x86);
    a.ld_d_a();
    a.cp_a_n(0);
    a.jr_z("no_combo");
    a.ld_hl_label("dat_combos");
    a.ld_b(12);
    a.ld_c(0);
    a.label("cb_lp");
    a.ld_a_hli();
    a.cp_d();
    a.jr_z("cb_hit");
    a.inc_c();
    a.dec_b();
    a.jr_nz("cb_lp");
    a.jr("no_combo");

    a.label("cb_hit");
    a.ld_hl_label("dat_combo_pal");
    a.ld_b(0);
    a.add_hl_bc();
    a.ld_a_hl_ind();
    a.ld_b_a();
    a.jp("pal_from_flags");
    a.label("no_combo");

    // Title-based palette selection
    a.ld_hl_imm(0x0134);
    a.ld_b(16);
    a.xor_a();
    a.label("tsum");
    a.add_a_hl();
    a.inc_hl();
    a.dec_b();
    a.jr_nz("tsum");
    a.ld_d_a(); // D = title checksum

    a.ld_hl_label("dat_title_ck");
    a.ld_b(79);
    a.ld_c(0);
    a.label("tck_lp");
    a.ld_a_hli();
    a.cp_d();
    a.jr_z("tck_found");
    a.inc_c();
    a.dec_b();
    a.jr_nz("tck_lp");
    a.ld_c(0);
    a.jp("pal_resolve");

    a.label("tck_found");
    a.ld_a_c();
    a.cp_a_n(65);
    a.jr_c("pal_resolve");

    // Disambiguation
    a.sub_n(65);
    a.ld_e_a();
    a.ld_a_nn(0x0137);
    a.ld_d_a();

    // Row 0
    a.ld_hl_label("dat_4th_r0");
    a.push_de();
    a.ld_d(0);
    a.add_hl_de();
    a.pop_de();
    a.ld_a_d();
    a.cp_hl_ind();
    a.jr_z("row0_hit");

    // Row 1
    a.ld_hl_label("dat_4th_r1");
    a.push_de();
    a.ld_d(0);
    a.add_hl_de();
    a.pop_de();
    a.ld_a_d();
    a.cp_hl_ind();
    a.jr_z("row1_hit");

    // Row 2 (only column 0)
    a.ld_a_e();
    a.cp_a_n(0);
    a.jr_nz("no_4th_match");
    a.ld_a_d();
    a.cp_a_n(0x52); // 'R'
    a.jr_z("row2_hit");

    a.label("no_4th_match");
    a.ld_c(0);
    a.jr("pal_resolve");

    a.label("row0_hit");
    a.jr("pal_resolve");

    a.label("row1_hit");
    a.ld_a_c();
    a.add_a_n(14);
    a.ld_c_a();
    a.jr("pal_resolve");

    a.label("row2_hit");
    a.ld_a_c();
    a.add_a_n(28);
    a.ld_c_a();

    // ============================================================
    // Resolve palette_id → program BCPD/OCPD
    // ============================================================
    a.label("pal_resolve");
    a.ld_hl_label("dat_pal_flags");
    a.ld_b(0);
    a.add_hl_bc();
    a.ld_a_hl_ind();
    a.ld_b_a();

    a.label("pal_from_flags");
    a.ld_a_b();
    a.and_n(0x1F);
    a.ld_l_a();
    a.add_a_a();
    a.add_a_l();
    a.ld_e_a();
    a.ld_d(0);
    a.ld_hl_label("dat_triplets");
    a.add_hl_de();

    // Read 3 offsets
    a.ld_a_hli(); a.ldh_n_a(0x80);
    a.ld_a_hli(); a.ldh_n_a(0x81);
    a.ld_a_hl_ind(); a.ldh_n_a(0x82);

    // BG = offset2
    a.ldh_a_n(0x82); a.ldh_n_a(0x83);

    // OBJ0: offset0 if bit5, else offset2
    a.ld_a_b();
    a.and_n(0x20);
    a.jr_z("obj0_def");
    a.ldh_a_n(0x80);
    a.jr("obj0_sv");
    a.label("obj0_def");
    a.ldh_a_n(0x82);
    a.label("obj0_sv");
    a.ldh_n_a(0x84);

    // OBJ1: offset1 if bit7, offset0 if bit6, else offset2
    a.ld_a_b();
    a.and_n(0x80);
    a.jr_nz("obj1_e");
    a.ld_a_b();
    a.and_n(0x40);
    a.jr_nz("obj1_d");
    a.ldh_a_n(0x82);
    a.jr("obj1_sv");
    a.label("obj1_e");
    a.ldh_a_n(0x81);
    a.jr("obj1_sv");
    a.label("obj1_d");
    a.ldh_a_n(0x80);
    a.label("obj1_sv");
    a.ldh_n_a(0x85);

    // Program BG palette 0
    a.ld_a(0x80); a.ldh_n_a(0x68);
    a.ldh_a_n(0x83);
    a.ld_c(0x69);
    a.call("sub_copy_pal");

    // Program OBJ palette 0
    a.ld_a(0x80); a.ldh_n_a(0x6A);
    a.ldh_a_n(0x84);
    a.ld_c(0x6B);
    a.call("sub_copy_pal");

    // Program OBJ palette 1
    a.ldh_a_n(0x85);
    a.call("sub_copy_pal");

    // DMG compat register values
    a.ld_d(0x00); a.ld_e(0x08); a.ld_h(0x00); a.ld_l(0x7C);
    a.jr("final");

    // ============================================================
    // CGB native game
    // ============================================================
    a.label("is_cgb");
    a.ld_d(0xFF); a.ld_e(0x56); a.ld_h(0x00); a.ld_l(0x0D);

    // ============================================================
    // Final: set remaining IO and CPU registers, hand off
    // ============================================================
    a.label("final");

    a.ld_a(0x88); a.ldh_n_a(0x68); // BCPS
    a.ld_a(0x90); a.ldh_n_a(0x6A); // OCPS
    a.ld_a(0x91); a.ldh_n_a(0x40); // LCDC

    a.ld_b(if agb { 0x01 } else { 0x00 });
    a.ld_c(0x00);
    a.xor_a(); // F = $80 (Z flag set)
    a.ld_sp(0xFFFE);
    a.jp("handoff");

    // ============================================================
    // Subroutine: clear both VRAM banks
    // ============================================================
    a.label("sub_clear_vram");
    a.xor_a(); a.ldh_n_a(0x4F);
    a.ld_hl_imm(0x8000); a.ld_b(0x20); a.ld_c(0);
    a.label("sv0");
    a.ld_hli_a(); a.dec_c(); a.jr_nz("sv0");
    a.dec_b(); a.jr_nz("sv0");
    a.ld_a(1); a.ldh_n_a(0x4F);
    a.xor_a(); a.ld_hl_imm(0x8000); a.ld_b(0x20); a.ld_c(0);
    a.label("sv1");
    a.ld_hli_a(); a.dec_c(); a.jr_nz("sv1");
    a.dec_b(); a.jr_nz("sv1");
    a.xor_a(); a.ldh_n_a(0x4F);
    a.ret();

    // ============================================================
    // Subroutine: copy 8 bytes from PALETTE_RGB[A] to IO port C
    // ============================================================
    a.label("sub_copy_pal");
    a.ld_e_a();
    a.ld_d(0);
    a.ld_hl_label("dat_pal_rgb");
    a.add_hl_de();
    a.ld_b(8);
    a.label("scp_lp");
    a.ld_a_hli();
    a.ld_ffc_a();
    a.dec_b();
    a.jr_nz("scp_lp");
    a.ret();

    // ============================================================
    // Subroutine: update swatch palette 7 from current combo
    // ============================================================
    a.label("sub_update_swatch");
    a.ldh_a_n(0x86);
    a.ld_d_a();
    a.ld_hl_label("dat_combos");
    a.ld_b(12);
    a.ld_c(0);
    a.label("sw_lp");
    a.ld_a_hli();
    a.cp_d();
    a.jr_z("sw_hit");
    a.inc_c();
    a.dec_b();
    a.jr_nz("sw_lp");
    a.ret();

    a.label("sw_hit");
    a.ld_hl_label("dat_combo_pal");
    a.ld_b(0);
    a.add_hl_bc();
    a.ld_a_hl_ind();
    a.and_n(0x1F);
    a.ld_l_a();
    a.add_a_a();
    a.add_a_l();
    a.ld_e_a();
    a.ld_d(0);
    a.ld_hl_label("dat_triplets");
    a.add_hl_de();
    a.inc_hl();
    a.inc_hl();
    a.ld_a_hl_ind();
    a.ld_e_a();
    a.ld_d(0);
    a.ld_hl_label("dat_pal_rgb");
    a.add_hl_de();
    a.ld_a(0x80 | 56);
    a.ldh_n_a(0x68);
    a.ld_b(8);
    a.label("sw_cp");
    a.ld_a_hli();
    a.ldh_n_a(0x69);
    a.dec_b();
    a.jr_nz("sw_cp");
    a.ret();

    // ============================================================
    // Data tables
    // ============================================================
    a.label("dat_combos");
    a.db(COMBOS);

    a.label("dat_combo_pal");
    a.db(COMBO_PAL);

    a.label("dat_title_ck");
    a.db(TITLE_CHECKSUMS);

    a.label("dat_4th_r0");
    a.db(FOURTH_LETTERS_R0);

    a.label("dat_4th_r1");
    a.db(FOURTH_LETTERS_R1);

    a.label("dat_pal_flags");
    a.db(PAL_IDX_FLAGS);

    a.label("dat_triplets");
    a.db(PAL_TRIPLETS);

    a.label("dat_pal_rgb");
    a.db(&palette_rgb());

    // ============================================================
    // Resolve and pad
    // ============================================================
    a.into_rom(CGB_BOOT_ROM_SIZE)
}
