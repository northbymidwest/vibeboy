#!/usr/bin/env python3
"""Generate a custom CGB boot ROM for vibeboy.

CGB boot ROM layout:
  0x0000-0x00FF  DMG-compatible boot region (256 bytes)
  0x0200-0x08FF  CGB-extended boot region (1792 bytes)
Total file: 2304 bytes (0x900).

Features:
  - Rainbow "VIBEBOY" logo with per-letter CGB palettes
  - Boot chime (CH1 descending sweep)
  - DMG game palette selection matching the official CGB boot ROM algorithm:
    title checksum → lookup table → palette preset with disambiguation
  - Correct post-boot register state for CGB and DMG games
"""

import sys

# --- Mini SM83 assembler ---

class Asm:
    def __init__(self):
        self.code = bytearray()
        self.labels = {}
        self.fixups = []

    @property
    def pc(self): return len(self.code)

    def label(self, name): self.labels[name] = self.pc

    def _emit(self, *args):
        for b in args: self.code.append(b & 0xFF)

    # Basic
    def nop(self):    self._emit(0x00)
    def di(self):     self._emit(0xF3)
    def ret(self):    self._emit(0xC9)
    def xor_a(self):  self._emit(0xAF)
    def cp_a(self, n): self._emit(0xFE, n)
    def and_n(self, n): self._emit(0xE6, n)
    def add_a_n(self, n): self._emit(0xC6, n)
    def sub_n(self, n): self._emit(0xD6, n)
    def cpl(self):    self._emit(0x2F)
    def swap_a(self): self._emit(0xCB, 0x37)

    # CP r (compare A with register)
    def cp_b(self): self._emit(0xB8)
    def cp_c(self): self._emit(0xB9)
    def cp_d(self): self._emit(0xBA)
    def cp_e(self): self._emit(0xBB)
    def cp_hl(self): self._emit(0xBE)  # CP (HL)

    # ADD A,r
    def add_a_a(self): self._emit(0x87)
    def add_a_hl(self): self._emit(0x86)
    def add_a_l(self): self._emit(0x85)
    def add_a_e(self): self._emit(0x83)

    # 8-bit loads
    def ld_a(self, n): self._emit(0x3E, n)
    def ld_b(self, n): self._emit(0x06, n)
    def ld_c(self, n): self._emit(0x0E, n)
    def ld_d(self, n): self._emit(0x16, n)
    def ld_e(self, n): self._emit(0x1E, n)
    def ld_h(self, n): self._emit(0x26, n)
    def ld_l(self, n): self._emit(0x2E, n)

    # Register-to-register
    def ld_a_b(self): self._emit(0x78)
    def ld_a_c(self): self._emit(0x79)
    def ld_a_d(self): self._emit(0x7A)
    def ld_a_e(self): self._emit(0x7B)
    def ld_a_h(self): self._emit(0x7C)
    def ld_a_l(self): self._emit(0x7D)
    def ld_a_hl(self): self._emit(0x7E)
    def ld_a_de(self): self._emit(0x1A)
    def ld_b_a(self): self._emit(0x47)
    def ld_c_a(self): self._emit(0x4F)
    def ld_d_a(self): self._emit(0x57)
    def ld_e_a(self): self._emit(0x5F)
    def ld_h_a(self): self._emit(0x67)
    def ld_l_a(self): self._emit(0x6F)
    def ld_hl_a(self): self._emit(0x77)
    def ld_b_c(self): self._emit(0x41)
    def ld_l_e(self): self._emit(0x6B)

    # 16-bit loads
    def ld_bc(self, nn): self._emit(0x01, nn & 0xFF, nn >> 8)
    def ld_de(self, nn):
        if isinstance(nn, str):
            self.fixups.append((self.pc + 1, nn, 'jp'))
            self._emit(0x11, 0, 0)
        else:
            self._emit(0x11, nn & 0xFF, nn >> 8)
    def ld_hl(self, nn):
        if isinstance(nn, str):
            self.fixups.append((self.pc + 1, nn, 'jp'))
            self._emit(0x21, 0, 0)
        else:
            self._emit(0x21, nn & 0xFF, nn >> 8)
    def ld_sp(self, nn): self._emit(0x31, nn & 0xFF, nn >> 8)

    # 16-bit arithmetic
    def add_hl_bc(self): self._emit(0x09)
    def add_hl_de(self): self._emit(0x19)

    # Memory
    def ldh_n_a(self, n): self._emit(0xE0, n)
    def ldh_a_n(self, n): self._emit(0xF0, n)
    def ld_hli_a(self): self._emit(0x22)
    def ld_a_hli(self): self._emit(0x2A)
    def ld_a_nn(self, nn): self._emit(0xFA, nn & 0xFF, nn >> 8)
    def ld_ffc_a(self): self._emit(0xE2)  # LD (FF00+C), A

    # Inc/Dec
    def inc_a(self): self._emit(0x3C)
    def inc_c(self): self._emit(0x0C)
    def inc_hl(self): self._emit(0x23)
    def inc_de(self): self._emit(0x13)
    def dec_a(self): self._emit(0x3D)
    def dec_b(self): self._emit(0x05)
    def dec_c(self): self._emit(0x0D)
    def dec_e(self): self._emit(0x1D)

    # Stack
    def push_af(self): self._emit(0xF5)
    def push_de(self): self._emit(0xD5)
    def push_hl(self): self._emit(0xE5)
    def pop_af(self): self._emit(0xF1)
    def pop_de(self): self._emit(0xD1)
    def pop_hl(self): self._emit(0xE1)

    # Jumps
    def jp(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jp'))
            self._emit(0xC3, 0, 0)
        else:
            self._emit(0xC3, t & 0xFF, t >> 8)
    def jp_nz(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jp'))
            self._emit(0xC2, 0, 0)
        else:
            self._emit(0xC2, t & 0xFF, t >> 8)
    def jr(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jr'))
            self._emit(0x18, 0)
        else:
            self._emit(0x18, t & 0xFF)
    def jr_z(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jr'))
            self._emit(0x28, 0)
        else:
            self._emit(0x28, t & 0xFF)
    def jr_nz(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jr'))
            self._emit(0x20, 0)
        else:
            self._emit(0x20, t & 0xFF)
    def jr_c(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jr'))
            self._emit(0x38, 0)
        else:
            self._emit(0x38, t & 0xFF)
    def call(self, t):
        if isinstance(t, str):
            self.fixups.append((self.pc + 1, t, 'jp'))
            self._emit(0xCD, 0, 0)
        else:
            self._emit(0xCD, t & 0xFF, t >> 8)

    def db(self, *args):
        for b in args: self._emit(b)
    def dw(self, *args):
        for w in args: self._emit(w & 0xFF, (w >> 8) & 0xFF)

    def pad_to(self, addr):
        while self.pc < addr: self._emit(0x00)

    def resolve(self):
        for offset, label, kind in self.fixups:
            if label not in self.labels:
                raise ValueError(f"Undefined label: {label}")
            target = self.labels[label]
            if kind == 'jr':
                rel = target - (offset + 1)
                if rel < -128 or rel > 127:
                    raise ValueError(f"JR to {label} out of range: {rel}")
                self.code[offset] = rel & 0xFF
            else:
                self.code[offset] = target & 0xFF
                self.code[offset + 1] = (target >> 8) & 0xFF


# --- Graphics ---

def make_tile(rows):
    data = []
    for row in rows:
        data.append(row); data.append(row)
    return data

FONT = {
    'V': make_tile([0xC6,0xC6,0xC6,0xC6,0x6C,0x6C,0x38,0x10]),
    'I': make_tile([0x7C,0x10,0x10,0x10,0x10,0x10,0x10,0x7C]),
    'B': make_tile([0xFC,0xC6,0xC6,0xFC,0xC6,0xC6,0xC6,0xFC]),
    'E': make_tile([0xFE,0xC0,0xC0,0xFC,0xC0,0xC0,0xC0,0xFE]),
    'O': make_tile([0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C]),
    'Y': make_tile([0xC6,0xC6,0x6C,0x38,0x10,0x10,0x10,0x10]),
}

# RGB555: R=bits 0-4, G=bits 5-9, B=bits 10-14
LOGO_COLORS = [0x001F, 0x02DF, 0x03FF, 0x03A0, 0x7E00, 0x7E0F, 0x7C1F]


# --- Official CGB boot ROM palette data ---
# From publicly documented reverse engineering (ISSOtm/gb-bootroms, Pan Docs)

TITLE_CHECKSUMS = bytes.fromhex(
    "00 88 16 36 D1 DB F2 3C 8C 92 3D 5C 58 C9 3E 70"
    "1D 59 69 19 35 A8 14 AA 75 95 99 34 6F 15 FF 97"
    "4B 90 17 10 39 F7 F6 A2 49 4E 43 68 E0 8B F0 CE"
    "0C 29 E8 B7 86 9A 52 01 9D 71 9C BD 5D 6D 67 3F"
    "6B B3 46 28 A5 C6 D3 27 61 18 66 6A BF 0D F4"
)

FOURTH_LETTERS_R0 = bytes.fromhex("42 45 46 41 41 52 42 45 4B 45 4B 20 52 2D")
FOURTH_LETTERS_R1 = bytes.fromhex("55 52 41 52 20 49 4E 41 49 4C 49 43 45 20")
FOURTH_LETTERS_R2 = bytes.fromhex("52")

PAL_IDX_FLAGS = bytes.fromhex(
    "7C 08 12 A3 A2 07 87 4B 20 12 65 A8 16 A9 86 B1"
    "68 A0 87 66 12 A1 30 3C 12 85 12 64 1B 07 06 6F"
    "6E 6E AE AF 6F B2 AF B2 A8 AB 6F AF 86 AE A2 A2"
    "12 AF 13 12 A1 6E AF AF AD 06 4C 6E AF AF 12 7C"
    "AC A8 6A 6E 13 A0 2D A8 2B AC 64 AC 6D 87 BC 60"
    "B4 13 72 7C B5 AE AE 7C 7C 65 A2 6C 64 85"
)

PAL_TRIPLETS = bytes.fromhex(
    "80 B0 40 88 20 68 DE 00 70 DE 20 78 20 20 38 20"
    "B0 90 20 B0 A0 E0 B0 C0 98 B6 48 80 E0 50 1E 1E"
    "58 20 B8 E0 88 B0 10 20 00 10 20 E0 18 E0 18 00"
    "18 E0 20 A8 E0 20 18 E0 00 20 18 D8 C8 18 E0 00"
    "E0 40 28 28 28 18 E0 60 20 18 E0 00 00 08 E0 18"
    "30 D0 D0 D0 20 E0 E8"
)

PALETTE_RGB = bytes.fromhex(
    "FF7F 8C3C 0000 0000  CF78 4831 0008 0100"
    "FF7F 5838 1D19 1400  FF7F 061F 0008 0000"
    "FF7F 101F 8707 0000  FF7F 1414 050A 0000"
    "FF7F 001F 090F 0000  FF7F 001F 0E16 0000"
    "FF7F 1015 0F0E 0800  1F5C 001F 000C 0000"
    "737F 9D0C 8610 0B00  1316 121F 0A08 0000"
    "FF7F 141F 1212 1F00  131F 1216 0E12 0700"
    "001F 1F1F 090A 0000  001B 101F 001F 1F00"
    "FF7F 001F 0A09 0000  0818 1A1F 0012 0000"
    "FF7F 001F 0800 0000  FF7F 001F 1006 0000"
    "FF7F 0F0B 1F00 001F  FF7F 0F1F 0010 1F00"
    "FF7F 001F 001F 0000  FF7F 001F 0000 0000"
    "FF7F 131F 0C00 0000  FF7F 191F 0C08 0000"
    "FF7F 081F 1B0A 0900  1F0C 0000 001F 1F1F"
    "FF7F 1F0F 1F00 1F00  1F0F 0000 001F 1F1F"
).replace(b' ', b'')
# Remove spaces from hex — actually bytes.fromhex ignores them with spaces
# Let me fix this:

# Original palette RGB data designed for vibeboy.
# 30 palettes x 8 bytes = 240 bytes. Each palette: 4 RGB555 colors (LE).
# The triplet table references these by byte offset; three composite
# palettes (0x1E, 0xB6, 0xDE) span entry boundaries, so adjacent
# entries must produce sensible cross-boundary palettes.

def _rgb555(r, g, b):
    v = (r & 0x1F) | ((g & 0x1F) << 5) | ((b & 0x1F) << 10)
    return [v & 0xFF, (v >> 8) & 0xFF]

def _pal(c0, c1, c2, c3):
    return _rgb555(*c0) + _rgb555(*c1) + _rgb555(*c2) + _rgb555(*c3)

W = (31,31,31)  # white
K = (0,0,0)     # black

PALETTE_RGB = bytes(
    #  0: Warm cream/peach (many default BGs)
    _pal((31,30,26), (24,18,10), (14,8,4),   K) +
    #  1: Earthy terracotta
    _pal((28,24,18), (22,14,8),  (14,6,2),   (4,2,0)) +
    #  2: Steel blue
    _pal(W,          (14,20,28), (6,10,18),   K) +
    #  3: Green/lime — color3 is transparent-black for composite 0x1E
    _pal(W,          (10,28,4),  (2,16,0),    K) +
    #  4: Rose/red — colors 0-2 join with entry3.color3 for composite 0x1E
    #     Composite 0x1E = [black, white, rose, dark-rose] (good OBJ palette)
    _pal(W,          (28,10,12), (18,2,4),    K) +
    #  5: Classic gray
    _pal(W,          (20,20,20), (10,10,10),  K) +
    #  6: Gold/amber
    _pal(W,          (30,26,4),  (20,14,0),   K) +
    #  7: Warm yellow
    _pal(W,          (30,24,6),  (22,14,2),   K) +
    #  8: Salmon/peach
    _pal(W,          (26,18,16), (20,8,6),    K) +
    #  9: Teal/sea
    _pal((28,31,28), (4,26,22),  (0,14,12),   K) +
    # 10: Dusty violet
    _pal((30,28,31), (22,12,24), (14,4,16),   (4,0,6)) +
    # 11: Warm brown (Super Mario Land BG)
    _pal((30,28,24), (22,16,8),  (12,8,2),    K) +
    # 12: Pastel pink
    _pal((31,24,26), (26,14,20), (18,6,14),   K) +
    # 13: Olive/army
    _pal((26,30,22), (16,22,8),  (8,14,2),    (2,4,0)) +
    # 14: Inverted dark base
    _pal((8,6,4),    (18,16,14), (26,24,22),  K) +
    # 15: Inverted blue accent
    _pal((6,8,18),   (12,16,24), (22,26,31),  W) +
    # 16: Sky blue
    _pal(W,          (14,22,31), (4,12,22),   K) +
    # 17: Deep forest
    _pal((14,22,10), (8,16,4),   (2,10,0),    (0,4,0)) +
    # 18: Tangerine
    _pal(W,          (30,18,4),  (22,8,0),    K) +
    # 19: Dark moss
    _pal((16,20,10), (10,14,4),  (4,8,0),     K) +
    # 20: Grass green
    _pal(W,          (10,26,6),  (2,18,0),    K) +
    # 21: Harvest gold
    _pal(W,          (28,22,6),  (18,12,2),   (4,2,0)) +
    # 22: Aqua/cyan — color3 must work as composite 0xB6 shade0
    _pal(W,          (6,28,26),  (0,16,16),   K) +
    # 23: Bright pastel — colors 0-2 join entry22.color3 for composite 0xB6
    #     Composite 0xB6 = [black, white, mint, light-green]
    _pal(W,          (24,31,24), (16,24,16),  (6,10,6)) +
    # 24: Lime/chartreuse
    _pal(W,          (22,30,4),  (12,20,0),   K) +
    # 25: Dark hunter
    _pal((22,26,10), (14,18,4),  (6,10,0),    K) +
    # 26: Warm neutral
    _pal(W,          (22,20,16), (12,10,6),   K) +
    # 27: Inverted base — color3 must work as composite 0xDE shade0
    _pal(K,          (10,12,14), (20,22,24),  W) +
    # 28: Vivid blue — colors 0-2 join entry27.color3 for composite 0xDE
    #     Composite 0xDE = [white, white, blue, dark-blue]
    _pal(W,          (12,18,30), (2,8,24),    K) +
    # 29: DMG green tint
    _pal(W,          (14,28,6),  (2,18,0),    K)
)
assert len(PALETTE_RGB) == 240


# --- Build ---

def build():
    a = Asm()

    # ================================================================
    # Region 1: 0x0000-0x00FF
    # ================================================================
    a.di()
    a.ld_sp(0xFFFE)
    a.xor_a()
    a.ldh_n_a(0x40)
    a.ldh_n_a(0x0F)
    a.jp('cgb_main')

    a.pad_to(0x00FC)
    a.label('handoff')
    a.ld_a(0x11)
    a.ldh_n_a(0x50)

    # ================================================================
    # Region 2: 0x0200-0x08FF
    # ================================================================
    a.pad_to(0x0200)
    a.label('cgb_main')

    # --- Clear VRAM ---
    a.call('sub_clear_vram')

    # --- Copy tile data ---
    a.jp('after_tiles')
    a.label('tile_data')
    for ch in "VIBEBOY":
        for byte in FONT[ch]:
            a.db(byte)
    # Swatch tiles: solid color 1, 2, 3 (tile 0 = color 0 already exists)
    # Tile 8: solid color 1 (lo=FF, hi=00 per row)
    a.label('swatch_tile1')
    for _ in range(8): a.db(0xFF, 0x00)
    # Tile 9: solid color 2 (lo=00, hi=FF per row)
    for _ in range(8): a.db(0x00, 0xFF)
    # Tile 10: solid color 3 (lo=FF, hi=FF per row)
    for _ in range(8): a.db(0xFF, 0xFF)
    a.label('after_tiles')

    a.ld_hl(0x8010)
    a.ld_de('tile_data')
    a.ld_b(10 * 16)          # 7 font tiles + 3 swatch tiles = 160 bytes
    a.label('cpy')
    a.ld_a_de()
    a.ld_hli_a()
    a.inc_de()
    a.dec_b()
    a.jr_nz('cpy')

    # --- Tilemap + attributes ---
    tmap = 0x9800 + 8 * 32 + 6   # "VIBEBOY" at row 8, col 6
    swatch = 0x9800 + 10 * 32 + 8 # swatch bar at row 10, col 8

    # Bank 0: tile indices
    a.ld_hl(tmap)
    a.ld_a(1)
    for i in range(7):
        a.ld_hli_a()
        if i < 6: a.inc_a()
    # Swatch: tiles 0 (color0), 8 (color1), 9 (color2), 10 (color3)
    a.ld_hl(swatch)
    a.xor_a(); a.ld_hli_a()       # tile 0
    a.ld_a(8); a.ld_hli_a()       # tile 8
    a.ld_a(9); a.ld_hli_a()       # tile 9
    a.ld_a(10); a.ld_hli_a()      # tile 10

    # Bank 1: palette attributes
    a.ld_a(1)
    a.ldh_n_a(0x4F)
    a.ld_hl(tmap)
    for p in range(7):
        a.ld_a(p)
        a.ld_hli_a()
    # Swatch tiles: all use palette 7
    a.ld_hl(swatch)
    a.ld_a(7)
    a.ld_hli_a(); a.ld_hli_a(); a.ld_hli_a(); a.ld_hli_a()
    a.xor_a()
    a.ldh_n_a(0x4F)

    # --- Rainbow palettes ---
    a.ld_a(0x80)
    a.ldh_n_a(0x68)
    for color in LOGO_COLORS:
        r, g, b = color & 0x1F, (color >> 5) & 0x1F, (color >> 10) & 0x1F
        light = ((r+31)//2) | (((g+31)//2) << 5) | (((b+31)//2) << 10)
        for c in [0x7FFF, light, color, color]:
            a.ld_a(c & 0xFF); a.ldh_n_a(0x69)
            a.ld_a((c >> 8) & 0xFF); a.ldh_n_a(0x69)

    # --- Palette 7: initially white (swatch invisible until combo held) ---
    # BGPI already auto-incremented past palettes 0-6 (56 bytes)
    # Palette 7 starts at BGPI index 56 = 0x38, but let's set explicitly
    a.ld_a(0x80 | 56)       # BGPI: auto-inc, palette 7
    a.ldh_n_a(0x68)
    for _ in range(4):       # 4 white colors
        a.ld_a(0xFF); a.ldh_n_a(0x69)
        a.ld_a(0x7F); a.ldh_n_a(0x69)

    # --- LCD on, chime ---
    a.ld_a(0x91); a.ldh_n_a(0x40)
    for reg, val in [(0x26,0x80),(0x24,0x77),(0x25,0xF3),(0x10,0x67),
                     (0x11,0x80),(0x12,0xF3),(0x13,0x83),(0x14,0x87)]:
        a.ld_a(val); a.ldh_n_a(reg)

    # --- Wait ~2 sec, polling joypad once per frame ---
    # HRAM $86 = saved combo, HRAM $87 = previous frame's joypad state
    a.xor_a()
    a.ldh_n_a(0x86)          # clear saved combo
    a.ldh_n_a(0x87)          # clear previous joypad state

    a.ld_e(120)
    a.label('w1')
    # Wait for VBlank start (LY == 144)
    a.ldh_a_n(0x44); a.cp_a(144); a.jr_nz('w1')

    # Poll joypad once per frame (during VBlank)
    # Write 0x20: bit 4 clear → D-pad selected (P14 active low)
    # Write 0x10: bit 5 clear → Buttons selected (P15 active low)
    a.ld_a(0x20)             # select D-pad
    a.ldh_n_a(0x00)
    a.ldh_a_n(0x00)          # debounce read
    a.ldh_a_n(0x00)          # stable read
    a.and_n(0x0F)
    a.cpl()
    a.and_n(0x0F)
    a.swap_a()               # D-pad in high nibble
    a.ld_b_a()
    a.ld_a(0x10)             # select buttons
    a.ldh_n_a(0x00)
    a.ldh_a_n(0x00)
    a.ldh_a_n(0x00)
    a.and_n(0x03)
    a.cpl()
    a.and_n(0x03)
    a.db(0xB0)               # OR B
    a.jr_z('w_no_btn')       # no buttons → skip
    a.ldh_n_a(0x86)          # save combo to HRAM
    # Check if combo changed from previous frame (new press or switch)
    a.ld_c_a()               # C = current combo
    a.ldh_a_n(0x87)          # A = previous combo
    a.cp_c()                 # compare
    a.ld_a_c()               # A = current (restore for swatch)
    a.ldh_n_a(0x87)          # save as previous
    a.jr_z('w_same')         # same combo as last frame → don't reset timer
    # New/changed combo → reset idle timer + update swatch
    a.ld_e(120)
    a.label('w_same')
    a.push_de()
    a.call('sub_update_swatch')
    a.pop_de()
    a.label('w_no_btn')
    a.ld_a(0x30)             # reset joypad
    a.ldh_n_a(0x00)

    # Wait for VBlank to end
    a.label('w2')
    a.ldh_a_n(0x44); a.cp_a(144); a.jr_z('w2')
    a.dec_e(); a.jr_nz('w1')

    # --- Final audio ---
    a.ld_a(0xBF); a.ldh_n_a(0x11)
    a.ld_a(0xF3); a.ldh_n_a(0x12)
    a.ld_a(0xBF); a.ldh_n_a(0x14)

    # --- LCD off ---
    a.label('off_vbl')
    a.ldh_a_n(0x44); a.cp_a(144); a.jr_nz('off_vbl')
    a.xor_a(); a.ldh_n_a(0x40)

    # --- Clear VRAM for game ---
    a.call('sub_clear_vram')

    # --- Hardware cleanup (before CGB/DMG branch, avoids clobbering regs) ---

    # Clear OAM (160 bytes at $FE00-$FE9F)
    a.ld_hl(0xFE00)
    a.ld_b(160)
    a.xor_a()
    a.label('clr_oam')
    a.ld_hli_a()
    a.dec_b()
    a.jr_nz('clr_oam')

    # Clear HRAM scratch ($FF80-$FF86)
    a.xor_a()
    for i in range(0x80, 0x87):
        a.ldh_n_a(i)

    # CH1: zero internal volume, set post-boot register values
    a.ld_a(0x80)
    a.ldh_n_a(0x10)          # NR10 = $80
    a.xor_a()
    a.ldh_n_a(0x12)          # NR12 = 0 → vol=0 on trigger
    a.ld_a(0xBF)
    a.ldh_n_a(0x14)          # NR14 = trigger → applies vol=0
    a.ld_a(0xBF)
    a.ldh_n_a(0x11)          # NR11 = $BF
    a.ld_a(0xF3)
    a.ldh_n_a(0x12)          # NR12 = $F3 (readable value, vol stays 0)

    # Joypad
    a.ld_a(0x30)
    a.ldh_n_a(0x00)          # P1 = $30

    # Interrupt flags
    a.ld_a(0x01)
    a.ldh_n_a(0x0F)          # IF = $01 (reads as $E1)

    # --- CGB vs DMG? ---
    a.ld_a_nn(0x0143)
    a.and_n(0x80)
    a.jp_nz('is_cgb')        # CGB game (bit 7 set) → skip palette selection

    # ============================================================
    # DMG game: palette selection
    # ============================================================
    a.ld_a(0xFC); a.ldh_n_a(0x47)  # BGP

    # --- Check saved joypad combo from boot animation ---
    a.ldh_a_n(0x86)          # load saved combo from HRAM
    a.ld_d_a()
    a.cp_a(0)
    a.jr_z('no_combo')       # no buttons were held
    a.ld_hl('dat_combos')
    a.ld_b(12)
    a.ld_c(0)
    a.label('cb_lp')
    a.ld_a_hli()
    a.cp_d()                 # compare with joypad state
    a.jr_z('cb_hit')
    a.inc_c()
    a.dec_b()
    a.jr_nz('cb_lp')
    a.jr('no_combo')         # no match

    a.label('cb_hit')
    # C = combo index. Look up combo palette flags byte.
    a.ld_hl('dat_combo_pal')
    a.ld_b(0)
    a.add_hl_bc()
    a.ld_a_hl()
    a.ld_b_a()               # B = flags+triplet byte
    a.jp('pal_from_flags')   # skip title-based selection
    a.label('no_combo')

    # --- Title-based palette selection ---
    # Compute title checksum: sum bytes 0x0134-0x0143
    a.ld_hl(0x0134)
    a.ld_b(16)
    a.xor_a()
    a.label('tsum')
    a.add_a_hl()            # A += (HL)
    a.inc_hl()
    a.dec_b()
    a.jr_nz('tsum')
    # A = title checksum
    a.ld_d_a()              # D = title checksum

    # Search TitleChecksums table
    a.ld_hl('dat_title_ck')
    a.ld_b(79)
    a.ld_c(0)               # C = index
    a.label('tck_lp')
    a.ld_a_hli()            # A = table entry
    a.cp_d()                # compare with D (title checksum)
    a.jr_z('tck_found')
    a.inc_c()
    a.dec_b()
    a.jr_nz('tck_lp')
    # Not found → palette_id = 0
    a.ld_c(0)
    a.jp('pal_resolve')

    a.label('tck_found')
    # C = index (0-78). If < 65, palette_id = C.
    a.ld_a_c()
    a.cp_a(65)
    a.jr_c('pal_resolve')   # C < 65

    # Disambiguation: check 4th title char against FourthLetters rows
    a.sub_n(65)             # A = column (0-13)
    a.ld_e_a()              # E = column
    a.ld_a_nn(0x0137)       # A = 4th title char
    a.ld_d_a()              # D = 4th char

    # Check row 0
    a.ld_hl('dat_4th_r0')
    a.ld_b(0)
    a.ld_b_c()              # save original C
    # Wait, I need to add E to HL. LD BC with B=0,C=E won't work easily.
    # Use: push, set up BC=0:E, ADD HL,BC, pop
    a.push_de()
    a.ld_d(0)               # DE = column (E already set)
    a.add_hl_de()            # HL = row0 + column
    a.pop_de()               # restore D=4th char, E=column
    a.ld_a_d()               # A = 4th char
    a.cp_hl()                # compare with (HL)
    a.jr_z('row0_hit')

    # Check row 1
    a.ld_hl('dat_4th_r1')
    a.push_de()
    a.ld_d(0)
    a.add_hl_de()
    a.pop_de()
    a.ld_a_d()
    a.cp_hl()
    a.jr_z('row1_hit')

    # Check row 2 (only column 0)
    a.ld_a_e()
    a.cp_a(0)
    a.jr_nz('no_4th_match')
    a.ld_a_d()
    a.cp_a(0x52)            # 'R'
    a.jr_z('row2_hit')

    a.label('no_4th_match')
    a.ld_c(0)               # default palette
    a.jr('pal_resolve')

    a.label('row0_hit')
    # palette_id = C (original index, already correct)
    a.jr('pal_resolve')

    a.label('row1_hit')
    a.ld_a_c()
    a.add_a_n(14)
    a.ld_c_a()
    a.jr('pal_resolve')

    a.label('row2_hit')
    a.ld_a_c()
    a.add_a_n(28)
    a.ld_c_a()

    # ============================================================
    # Resolve palette_id (in C) → program BCPD/OCPD
    # ============================================================
    a.label('pal_resolve')
    # Look up PaletteIndexesAndFlags[C]
    a.ld_hl('dat_pal_flags')
    a.ld_b(0)
    a.add_hl_bc()            # HL += C
    a.ld_a_hl()              # A = flags_and_triplet
    a.ld_b_a()               # B = save flags byte

    # --- Entry point for direct flags byte (from button combo) ---
    a.label('pal_from_flags')
    # B = flags+triplet byte (from PaletteIndexesAndFlags or combo table)
    a.ld_a_b()

    # Triplet index = A & 0x1F
    a.and_n(0x1F)
    # Multiply by 3: A + A + A
    a.ld_l_a()
    a.add_a_a()              # A*2
    a.add_a_l()              # A*3
    a.ld_e_a()
    a.ld_d(0)
    a.ld_hl('dat_triplets')
    a.add_hl_de()

    # Read 3 offsets from triplet
    a.ld_a_hli()
    a.ldh_n_a(0x80)          # HRAM $80 = offset0
    a.ld_a_hli()
    a.ldh_n_a(0x81)          # HRAM $81 = offset1
    a.ld_a_hl()
    a.ldh_n_a(0x82)          # HRAM $82 = offset2

    # BG always uses offset2
    a.ldh_a_n(0x82)
    a.ldh_n_a(0x83)          # HRAM $83 = BG offset

    # OBJ0: offset0 if flag bit 0 (byte bit 5) set, else offset2
    a.ld_a_b()
    a.and_n(0x20)            # bit 5
    a.jr_z('obj0_def')
    a.ldh_a_n(0x80)
    a.jr('obj0_sv')
    a.label('obj0_def')
    a.ldh_a_n(0x82)
    a.label('obj0_sv')
    a.ldh_n_a(0x84)          # HRAM $84 = OBJ0 offset

    # OBJ1: offset1 if flag bit 2 (byte bit 7), offset0 if flag bit 1 (byte bit 6), else offset2
    a.ld_a_b()
    a.and_n(0x80)            # bit 7
    a.jr_nz('obj1_e')
    a.ld_a_b()
    a.and_n(0x40)            # bit 6
    a.jr_nz('obj1_d')
    a.ldh_a_n(0x82)
    a.jr('obj1_sv')
    a.label('obj1_e')
    a.ldh_a_n(0x81)
    a.jr('obj1_sv')
    a.label('obj1_d')
    a.ldh_a_n(0x80)
    a.label('obj1_sv')
    a.ldh_n_a(0x85)          # HRAM $85 = OBJ1 offset

    # Program BG palette 0
    a.ld_a(0x80); a.ldh_n_a(0x68)   # BGPI auto-inc
    a.ldh_a_n(0x83)
    a.ld_c(0x69)             # BGPD port
    a.call('sub_copy_pal')

    # Program OBJ palette 0
    a.ld_a(0x80); a.ldh_n_a(0x6A)   # OBPI auto-inc
    a.ldh_a_n(0x84)
    a.ld_c(0x6B)             # OBPD port
    a.call('sub_copy_pal')

    # Program OBJ palette 1 (OBPI auto-continues)
    a.ldh_a_n(0x85)
    a.call('sub_copy_pal')

    # DMG compat register values
    a.ld_d(0x00); a.ld_e(0x08); a.ld_h(0x00); a.ld_l(0x7C)
    a.jr('final')

    # ============================================================
    # CGB native game
    # ============================================================
    a.label('is_cgb')
    a.ld_d(0xFF); a.ld_e(0x56); a.ld_h(0x00); a.ld_l(0x0D)

    # ============================================================
    # Final: set remaining IO and CPU registers, hand off
    # (OAM, HRAM, audio, P1, IF already cleaned up above)
    # ============================================================
    a.label('final')

    # Palette index registers (post-boot state)
    a.ld_a(0x88)
    a.ldh_n_a(0x68)          # BCPS = auto-inc + index 8
    a.ld_a(0x90)
    a.ldh_n_a(0x6A)          # OCPS = auto-inc + index 16

    # LCD on
    a.ld_a(0x91)
    a.ldh_n_a(0x40)          # LCDC = $91

    # CPU registers (D, E, H, L already set by CGB/DMG branch)
    a.ld_b(0x00)
    a.ld_c(0x00)
    a.xor_a()                # F = $80 (Z flag set)
    a.ld_sp(0xFFFE)
    a.jp('handoff')           # sets A=$11, disables bootrom

    # ============================================================
    # Subroutine: clear both VRAM banks
    # ============================================================
    a.label('sub_clear_vram')
    a.xor_a(); a.ldh_n_a(0x4F)
    a.ld_hl(0x8000); a.ld_b(0x20); a.ld_c(0)
    a.label('sv0')
    a.ld_hli_a(); a.dec_c(); a.jr_nz('sv0')
    a.dec_b(); a.jr_nz('sv0')
    a.ld_a(1); a.ldh_n_a(0x4F)
    a.xor_a(); a.ld_hl(0x8000); a.ld_b(0x20); a.ld_c(0)
    a.label('sv1')
    a.ld_hli_a(); a.dec_c(); a.jr_nz('sv1')
    a.dec_b(); a.jr_nz('sv1')
    a.xor_a(); a.ldh_n_a(0x4F)
    a.ret()

    # ============================================================
    # Subroutine: copy 8 bytes from PALETTE_RGB[A] to IO port C
    # Input: A = offset, C = $69 (BGPD) or $6B (OBPD)
    # ============================================================
    a.label('sub_copy_pal')
    a.ld_e_a()
    a.ld_d(0)
    a.ld_hl('dat_pal_rgb')
    a.add_hl_de()
    a.ld_b(8)
    a.label('scp_lp')
    a.ld_a_hli()
    a.ld_ffc_a()             # LD (FF00+C), A
    a.dec_b()
    a.jr_nz('scp_lp')
    a.ret()

    # ============================================================
    # Subroutine: update swatch palette 7 from current combo
    # Reads HRAM $86 (joypad combo), searches combo table,
    # resolves BG palette offset, copies 8 bytes to palette 7.
    # ============================================================
    a.label('sub_update_swatch')
    a.ldh_a_n(0x86)          # A = saved combo
    a.ld_d_a()
    # Search combo table for match
    a.ld_hl('dat_combos')
    a.ld_b(12)
    a.ld_c(0)
    a.label('sw_lp')
    a.ld_a_hli()
    a.cp_d()
    a.jr_z('sw_hit')
    a.inc_c()
    a.dec_b()
    a.jr_nz('sw_lp')
    a.ret()                  # no match → don't update

    a.label('sw_hit')
    # C = combo index → look up flags byte
    a.ld_hl('dat_combo_pal')
    a.ld_b(0)
    a.add_hl_bc()
    a.ld_a_hl()              # A = flags+triplet byte
    # Extract BG offset (offset2 from triplet)
    a.and_n(0x1F)            # triplet index
    a.ld_l_a()
    a.add_a_a()              # *2
    a.add_a_l()              # *3
    a.ld_e_a()
    a.ld_d(0)
    a.ld_hl('dat_triplets')
    a.add_hl_de()
    a.inc_hl()               # skip offset0
    a.inc_hl()               # skip offset1
    a.ld_a_hl()              # A = offset2 (BG offset)
    # Copy 8 bytes from PALETTE_RGB[A] to BG palette 7
    a.ld_e_a()
    a.ld_d(0)
    a.ld_hl('dat_pal_rgb')
    a.add_hl_de()
    a.ld_a(0x80 | 56)        # BGPI: auto-inc, palette 7 (index 56)
    a.ldh_n_a(0x68)
    a.ld_b(8)
    a.label('sw_cp')
    a.ld_a_hli()
    a.ldh_n_a(0x69)           # BGPD
    a.dec_b()
    a.jr_nz('sw_cp')
    a.ret()

    # ============================================================
    # Data tables
    # ============================================================
    # Joypad combo table: combined joypad state bytes
    # High nibble = D-pad (Down=$80,Up=$40,Left=$20,Right=$10)
    # Low nibble = buttons (B=$02, A=$01)
    a.label('dat_combos')
    a.db(0x40, 0x41, 0x42,   # Up, Up+A, Up+B
         0x20, 0x21, 0x22,   # Left, Left+A, Left+B
         0x80, 0x81, 0x82,   # Down, Down+A, Down+B
         0x10, 0x11, 0x12)   # Right, Right+A, Right+B

    # Palette flags bytes for each combo (same format as PaletteIndexesAndFlags)
    a.label('dat_combo_pal')
    a.db(0x12, 0xB0, 0x79,   # Up, Up+A, Up+B
         0xB8, 0xAD, 0x16,   # Left, Left+A, Left+B
         0x17, 0x07, 0xBA,   # Down, Down+A, Down+B
         0x05, 0x7C, 0x13)   # Right, Right+A, Right+B

    a.label('dat_title_ck')
    for b in TITLE_CHECKSUMS: a.db(b)

    a.label('dat_4th_r0')
    for b in FOURTH_LETTERS_R0: a.db(b)

    a.label('dat_4th_r1')
    for b in FOURTH_LETTERS_R1: a.db(b)

    a.label('dat_pal_flags')
    for b in PAL_IDX_FLAGS: a.db(b)

    a.label('dat_triplets')
    for b in PAL_TRIPLETS: a.db(b)

    a.label('dat_pal_rgb')
    for b in PALETTE_RGB: a.db(b)

    # ============================================================
    # Resolve and pad
    # ============================================================
    a.resolve()

    used = sum(1 for b in a.code[0x200:] if b != 0)
    print(f"CGB region: {a.pc - 0x200} bytes used of 1792 ({used} non-zero)", file=sys.stderr)

    if a.pc > 0x0900:
        print(f"ERROR: code is {a.pc:#x} bytes, exceeds 0x900!", file=sys.stderr)
        sys.exit(1)

    while len(a.code) < 0x0900:
        a.code.append(0x00)

    return bytes(a.code)


def main():
    rom = build()
    out = "bootroms/vibeboy_cgb_boot.bin"
    if len(sys.argv) > 1: out = sys.argv[1]
    with open(out, 'wb') as f: f.write(rom)
    print(f"Generated CGB boot ROM: {out} ({len(rom)} bytes)")


if __name__ == '__main__':
    main()
