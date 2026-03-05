#!/usr/bin/env python3
"""WDC 65C816 disassembler for SGB BIOS analysis.

Automatically tracks M/X flag state through REP/SEP instructions to
correctly size 8-bit vs 16-bit immediate operands.
"""

import sys

MNEMONICS = {
    0x00:'BRK',0x01:'ORA (dp,X)',0x02:'COP',0x03:'ORA sr,S',0x04:'TSB dp',0x05:'ORA dp',0x06:'ASL dp',0x07:'ORA [dp]',
    0x08:'PHP',0x09:'ORA imm',0x0A:'ASL A',0x0B:'PHD',0x0C:'TSB abs',0x0D:'ORA abs',0x0E:'ASL abs',0x0F:'ORA long',
    0x10:'BPL',0x11:'ORA (dp),Y',0x12:'ORA (dp)',0x13:'ORA (sr,S),Y',0x14:'TRB dp',0x15:'ORA dp,X',0x16:'ASL dp,X',0x17:'ORA [dp],Y',
    0x18:'CLC',0x19:'ORA abs,Y',0x1A:'INC A',0x1B:'TCS',0x1C:'TRB abs',0x1D:'ORA abs,X',0x1E:'ASL abs,X',0x1F:'ORA long,X',
    0x20:'JSR abs',0x21:'AND (dp,X)',0x22:'JSL long',0x23:'AND sr,S',0x24:'BIT dp',0x25:'AND dp',0x26:'ROL dp',0x27:'AND [dp]',
    0x28:'PLP',0x29:'AND imm',0x2A:'ROL A',0x2B:'PLD',0x2C:'BIT abs',0x2D:'AND abs',0x2E:'ROL abs',0x2F:'AND long',
    0x30:'BMI',0x31:'AND (dp),Y',0x32:'AND (dp)',0x33:'AND (sr,S),Y',0x34:'BIT dp,X',0x35:'AND dp,X',0x36:'ROL dp,X',0x37:'AND [dp],Y',
    0x38:'SEC',0x39:'AND abs,Y',0x3A:'DEC A',0x3B:'TSC',0x3C:'BIT abs,X',0x3D:'AND abs,X',0x3E:'ROL abs,X',0x3F:'AND long,X',
    0x40:'RTI',0x41:'EOR (dp,X)',0x42:'WDM',0x43:'EOR sr,S',0x44:'MVP',0x45:'EOR dp',0x46:'LSR dp',0x47:'EOR [dp]',
    0x48:'PHA',0x49:'EOR imm',0x4A:'LSR A',0x4B:'PHK',0x4C:'JMP abs',0x4D:'EOR abs',0x4E:'LSR abs',0x4F:'EOR long',
    0x50:'BVC',0x51:'EOR (dp),Y',0x52:'EOR (dp)',0x53:'EOR (sr,S),Y',0x54:'MVN',0x55:'EOR dp,X',0x56:'LSR dp,X',0x57:'EOR [dp],Y',
    0x58:'CLI',0x59:'EOR abs,Y',0x5A:'PHY',0x5B:'TCD',0x5C:'JML long',0x5D:'EOR abs,X',0x5E:'LSR abs,X',0x5F:'EOR long,X',
    0x60:'RTS',0x61:'ADC (dp,X)',0x62:'PER',0x63:'ADC sr,S',0x64:'STZ dp',0x65:'ADC dp',0x66:'ROR dp',0x67:'ADC [dp]',
    0x68:'PLA',0x69:'ADC imm',0x6A:'ROR A',0x6B:'RTL',0x6C:'JMP (abs)',0x6D:'ADC abs',0x6E:'ROR abs',0x6F:'ADC long',
    0x70:'BVS',0x71:'ADC (dp),Y',0x72:'ADC (dp)',0x73:'ADC (sr,S),Y',0x74:'STZ dp,X',0x75:'ADC dp,X',0x76:'ROR dp,X',0x77:'ADC [dp],Y',
    0x78:'SEI',0x79:'ADC abs,Y',0x7A:'PLY',0x7B:'TDC',0x7C:'JMP (abs,X)',0x7D:'ADC abs,X',0x7E:'ROR abs,X',0x7F:'ADC long,X',
    0x80:'BRA',0x81:'STA (dp,X)',0x82:'BRL',0x83:'STA sr,S',0x84:'STY dp',0x85:'STA dp',0x86:'STX dp',0x87:'STA [dp]',
    0x88:'DEY',0x89:'BIT imm',0x8A:'TXA',0x8B:'PHB',0x8C:'STY abs',0x8D:'STA abs',0x8E:'STX abs',0x8F:'STA long',
    0x90:'BCC',0x91:'STA (dp),Y',0x92:'STA (dp)',0x93:'STA (sr,S),Y',0x94:'STY dp,X',0x95:'STA dp,X',0x96:'STX dp,Y',0x97:'STA [dp],Y',
    0x98:'TYA',0x99:'STA abs,Y',0x9A:'TXS',0x9B:'TXY',0x9C:'STZ abs',0x9D:'STA abs,X',0x9E:'STZ abs,X',0x9F:'STA long,X',
    0xA0:'LDY imm',0xA1:'LDA (dp,X)',0xA2:'LDX imm',0xA3:'LDA sr,S',0xA4:'LDY dp',0xA5:'LDA dp',0xA6:'LDX dp',0xA7:'LDA [dp]',
    0xA8:'TAY',0xA9:'LDA imm',0xAA:'TAX',0xAB:'PLB',0xAC:'LDY abs',0xAD:'LDA abs',0xAE:'LDX abs',0xAF:'LDA long',
    0xB0:'BCS',0xB1:'LDA (dp),Y',0xB2:'LDA (dp)',0xB3:'LDA (sr,S),Y',0xB4:'LDY dp,X',0xB5:'LDA dp,X',0xB6:'LDX dp,Y',0xB7:'LDA [dp],Y',
    0xB8:'CLV',0xB9:'LDA abs,Y',0xBA:'TSX',0xBB:'TYX',0xBC:'LDY abs,X',0xBD:'LDA abs,X',0xBE:'LDX abs,Y',0xBF:'LDA long,X',
    0xC0:'CPY imm',0xC1:'CMP (dp,X)',0xC2:'REP',0xC3:'CMP sr,S',0xC4:'CPY dp',0xC5:'CMP dp',0xC6:'DEC dp',0xC7:'CMP [dp]',
    0xC8:'INY',0xC9:'CMP imm',0xCA:'DEX',0xCB:'WAI',0xCC:'CPY abs',0xCD:'CMP abs',0xCE:'DEC abs',0xCF:'CMP long',
    0xD0:'BNE',0xD1:'CMP (dp),Y',0xD2:'CMP (dp)',0xD3:'CMP (sr,S),Y',0xD4:'PEI',0xD5:'CMP dp,X',0xD6:'DEC dp,X',0xD7:'CMP [dp],Y',
    0xD8:'CLD',0xD9:'CMP abs,Y',0xDA:'PHX',0xDB:'STP',0xDC:'JML [abs]',0xDD:'CMP abs,X',0xDE:'DEC abs,X',0xDF:'CMP long,X',
    0xE0:'CPX imm',0xE1:'SBC (dp,X)',0xE2:'SEP',0xE3:'SBC sr,S',0xE4:'CPX dp',0xE5:'SBC dp',0xE6:'INC dp',0xE7:'SBC [dp]',
    0xE8:'INX',0xE9:'SBC imm',0xEA:'NOP',0xEB:'XBA',0xEC:'CPX abs',0xED:'SBC abs',0xEE:'INC abs',0xEF:'SBC long',
    0xF0:'BEQ',0xF1:'SBC (dp),Y',0xF2:'SBC (dp)',0xF3:'SBC (sr,S),Y',0xF4:'PEA',0xF5:'SBC dp,X',0xF6:'INC dp,X',0xF7:'SBC [dp],Y',
    0xF8:'SED',0xF9:'SBC abs,Y',0xFA:'PLX',0xFB:'XCE',0xFC:'JSR (abs,X)',0xFD:'SBC abs,X',0xFE:'INC abs,X',0xFF:'SBC long,X',
}

SIMPLE_MN = {op: mn.split()[0] for op, mn in MNEMONICS.items()}

# M-flag dependent immediates (accumulator width)
IMM_M = {0x09,0x29,0x49,0x69,0x89,0xA9,0xC9,0xE9}
# X-flag dependent immediates (index width)
IMM_X = {0xA0,0xA2,0xC0,0xE0}

# Implied/accumulator (1-byte) opcodes
IMPLIED = {
    0x08,0x0A,0x0B,0x18,0x1A,0x1B,0x28,0x2A,0x2B,0x38,0x3A,0x3B,
    0x40,0x48,0x4A,0x4B,0x58,0x5A,0x5B,0x60,0x68,0x6A,0x6B,0x78,0x7A,0x7B,
    0x88,0x8A,0x8B,0x98,0x9A,0x9B,0xA8,0xAA,0xAB,0xB8,0xBA,0xBB,
    0xC8,0xCA,0xCB,0xD8,0xDA,0xDB,0xE8,0xEA,0xEB,0xF8,0xFA,0xFB,
}

# dp addressing (2-byte) opcodes
DP_OPS = {
    0x01,0x03,0x05,0x06,0x07,0x11,0x12,0x13,0x15,0x16,0x17,
    0x21,0x23,0x25,0x26,0x27,0x31,0x32,0x33,0x35,0x36,0x37,
    0x41,0x43,0x45,0x46,0x47,0x51,0x52,0x53,0x55,0x56,0x57,
    0x61,0x63,0x65,0x66,0x67,0x71,0x72,0x73,0x75,0x76,0x77,
    0x81,0x83,0x85,0x86,0x87,0x91,0x92,0x93,0x95,0x96,0x97,
    0xA1,0xA3,0xA5,0xA6,0xA7,0xB1,0xB2,0xB3,0xB5,0xB6,0xB7,
    0xC1,0xC3,0xC5,0xC6,0xC7,0xD1,0xD2,0xD3,0xD5,0xD6,0xD7,
    0xE1,0xE3,0xE5,0xE6,0xE7,0xF1,0xF2,0xF3,0xF5,0xF6,0xF7,
    0x04,0x14,0x24,0x34,0x64,0x74,0x84,0x94,0xA4,0xB4,0xC4,0xE4,
}

# abs addressing (3-byte) opcodes
ABS_OPS = {
    0x0C,0x0D,0x0E,0x1C,0x1D,0x1E,
    0x2C,0x2D,0x2E,0x3C,0x3D,0x3E,
    0x4C,0x4D,0x4E,0x5D,0x5E,
    0x6C,0x6D,0x6E,0x7C,0x7D,0x7E,
    0x8C,0x8D,0x8E,0x9C,0x9D,0x9E,
    0xAC,0xAD,0xAE,0xBC,0xBD,0xBE,
    0xCC,0xCD,0xCE,0xDC,0xDD,0xDE,
    0xEC,0xED,0xEE,0xFC,0xFD,0xFE,
    0x19,0x39,0x59,0x79,0x99,0xB9,0xD9,0xF9,
    0x20,
}

# long addressing (4-byte) opcodes
LONG_OPS = {
    0x0F,0x1F,0x2F,0x3F,0x4F,0x5F,0x6F,0x7F,
    0x8F,0x9F,0xAF,0xBF,0xCF,0xDF,0xEF,0xFF,
    0x22,0x5C,
}


def instr_size(op, m8=True, x8=True):
    if op in IMPLIED:
        return 1
    if op in (0x00, 0x02):  # BRK, COP
        return 2
    if op in (0x10,0x30,0x50,0x70,0x80,0x90,0xB0,0xD0,0xF0):
        return 2
    if op == 0x82:  # BRL
        return 3
    if op in (0xC2, 0xE2):  # REP/SEP
        return 2
    if op in IMM_M:
        return 2 if m8 else 3
    if op in IMM_X:
        return 2 if x8 else 3
    if op in (0x44, 0x54):  # Block moves
        return 3
    if op in (0xF4, 0x62):  # PEA, PER
        return 3
    if op == 0xD4:  # PEI
        return 2
    if op in DP_OPS:
        return 2
    if op in ABS_OPS:
        return 3
    if op in LONG_OPS:
        return 4
    if op == 0x42:  # WDM
        return 2
    return 1


def format_operand(rom, off, op, pc, m8, x8):
    size = instr_size(op, m8, x8)
    mn_full = MNEMONICS.get(op, '???')
    mn = SIMPLE_MN.get(op, '???')

    if size == 1:
        return mn_full

    # Branches
    if op in (0x10,0x30,0x50,0x70,0x80,0x90,0xB0,0xD0,0xF0):
        b1 = rom[off+1]
        rel = b1 if b1 < 0x80 else b1 - 256
        return f'{mn} ${pc+2+rel:04X}'
    if op == 0x82:  # BRL
        w = rom[off+1] | (rom[off+2] << 8)
        rel = w if w < 0x8000 else w - 0x10000
        return f'{mn} ${pc+3+rel:04X}'

    # REP/SEP
    if op in (0xC2, 0xE2):
        return f'{mn} #${rom[off+1]:02X}'

    # Block moves
    if op in (0x44, 0x54):
        return f'{mn} ${rom[off+1]:02X},${rom[off+2]:02X}'

    # Immediate
    if op in IMM_M or op in IMM_X:
        if size == 2:
            return f'{mn} #${rom[off+1]:02X}'
        else:
            w = rom[off+1] | (rom[off+2] << 8)
            return f'{mn} #${w:04X}'

    # Use the full mnemonic which has the addressing mode
    if size == 2:
        b1 = rom[off+1]
        parts = mn_full.split()
        if len(parts) > 1:
            mode = parts[1]
            mode = mode.replace('dp', f'${b1:02X}')
            mode = mode.replace('sr', f'${b1:02X}')
            return f'{parts[0]} {mode}'
        return f'{mn} ${b1:02X}'

    if size == 3:
        w = rom[off+1] | (rom[off+2] << 8)
        parts = mn_full.split()
        if len(parts) > 1:
            mode = parts[1]
            mode = mode.replace('abs', f'${w:04X}')
            return f'{parts[0]} {mode}'
        return f'{mn} ${w:04X}'

    if size == 4:
        l = rom[off+1] | (rom[off+2] << 8) | (rom[off+3] << 16)
        parts = mn_full.split()
        if len(parts) > 1:
            mode = parts[1]
            mode = mode.replace('long', f'${l:06X}')
            return f'{parts[0]} {mode}'
        return f'{mn} ${l:06X}'

    return mn_full


def rom_offset(pc):
    """Convert a PC address to ROM file offset (LoROM mapping).
    Bank 00-7D: offset = (bank * 0x8000) + (addr - 0x8000)
    For simplicity with single-bank ROMs, uses pc - 0x8000."""
    bank = (pc >> 16) & 0x7F
    addr = pc & 0xFFFF
    if addr >= 0x8000:
        return bank * 0x8000 + (addr - 0x8000)
    return pc - 0x8000  # fallback


def disassemble(rom, pc_start, count, m8=True, x8=True):
    """Disassemble `count` instructions starting at PC address `pc_start`.
    Automatically tracks REP/SEP to update M/X flag state."""
    lines = []
    pc = pc_start
    for _ in range(count):
        off = rom_offset(pc)
        if off < 0 or off >= len(rom):
            lines.append(f'${pc:04X}: (out of range)')
            break
        op = rom[off]
        size = instr_size(op, m8, x8)
        if off + size > len(rom):
            lines.append(f'${pc:04X}: (truncated)')
            break
        raw = ' '.join(f'{rom[off+i]:02X}' for i in range(size))
        text = format_operand(rom, off, op, pc, m8, x8)

        # Show mode indicator when it changes
        mode_note = ''
        if op == 0xC2 and off + 1 < len(rom):  # REP
            val = rom[off+1]
            parts = []
            if val & 0x20: parts.append('M=16')
            if val & 0x10: parts.append('X=16')
            if parts:
                mode_note = f'  ; {", ".join(parts)}'
        elif op == 0xE2 and off + 1 < len(rom):  # SEP
            val = rom[off+1]
            parts = []
            if val & 0x20: parts.append('M=8')
            if val & 0x10: parts.append('X=8')
            if parts:
                mode_note = f'  ; {", ".join(parts)}'

        lines.append(f'${pc:04X}: {raw:14s} {text}{mode_note}')

        # Update M/X flags based on REP/SEP
        if op == 0xC2 and off + 1 < len(rom):  # REP — clear bits (16-bit)
            val = rom[off+1]
            if val & 0x20:
                m8 = False
            if val & 0x10:
                x8 = False
        elif op == 0xE2 and off + 1 < len(rom):  # SEP — set bits (8-bit)
            val = rom[off+1]
            if val & 0x20:
                m8 = True
            if val & 0x10:
                x8 = True
        # XCE switches emulation mode; conservatively assume 8-bit after
        elif op == 0xFB:
            m8 = True
            x8 = True

        pc += size
    return '\n'.join(lines)


def search_bytes(rom, pattern, label=""):
    """Search ROM for a byte pattern, return list of (rom_offset, pc) tuples."""
    results = []
    for i in range(len(rom) - len(pattern) + 1):
        if rom[i:i+len(pattern)] == pattern:
            results.append((i, i + 0x8000))
    return results


def show_vectors(rom):
    """Show 65816 interrupt vectors."""
    def vec(off):
        return rom[off] | (rom[off+1] << 8)
    print("=== Interrupt Vectors ===")
    print(f"  Native COP:   ${vec(0x7FE4):04X}")
    print(f"  Native BRK:   ${vec(0x7FE6):04X}")
    print(f"  Native ABORT: ${vec(0x7FE8):04X}")
    print(f"  Native NMI:   ${vec(0x7FEA):04X}")
    print(f"  Native IRQ:   ${vec(0x7FEE):04X}")
    print(f"  Emu COP:      ${vec(0x7FF4):04X}")
    print(f"  Emu ABORT:    ${vec(0x7FF8):04X}")
    print(f"  Emu NMI:      ${vec(0x7FFA):04X}")
    print(f"  Emu RESET:    ${vec(0x7FFC):04X}")
    print(f"  Emu IRQ:      ${vec(0x7FFE):04X}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='65C816 disassembler for SGB BIOS')
    parser.add_argument('rom', help='ROM file path')
    parser.add_argument('--pc', type=lambda x: int(x, 16), default=None,
                        help='Start PC address (hex, e.g. BF4A)')
    parser.add_argument('-n', '--count', type=int, default=30,
                        help='Number of instructions to disassemble')
    parser.add_argument('--m16', action='store_true', help='Start in 16-bit accumulator mode')
    parser.add_argument('--x16', action='store_true', help='Start in 16-bit index mode')
    parser.add_argument('--vectors', action='store_true', help='Show interrupt vectors')
    parser.add_argument('--search', type=lambda x: bytes.fromhex(x), default=None,
                        help='Search for hex byte pattern (e.g. 8D0042)')
    parser.add_argument('--context', type=int, default=5,
                        help='Instructions of context around search results')
    args = parser.parse_args()

    with open(args.rom, 'rb') as f:
        rom = f.read()

    if args.vectors:
        show_vectors(rom)
        print()

    if args.search is not None:
        results = search_bytes(rom, args.search)
        print(f"Found {len(results)} match(es) for {args.search.hex().upper()}:")
        for rom_off, pc in results:
            print(f"\n  ROM offset ${rom_off:04X} (PC=${pc:04X}):")
            start_pc = max(0x8000, pc - args.context * 2)
            print(disassemble(rom, start_pc, args.context * 2 + 5,
                              m8=not args.m16, x8=not args.x16))
        print()

    if args.pc is not None:
        print(f"=== Disassembly at ${args.pc:04X} ({args.count} instructions) ===")
        print(disassemble(rom, args.pc, args.count,
                          m8=not args.m16, x8=not args.x16))

    if args.pc is None and args.search is None and not args.vectors:
        show_vectors(rom)
        reset = rom[0x7FFC] | (rom[0x7FFD] << 8)
        print(f"\n=== Reset handler at ${reset:04X} ===")
        print(disassemble(rom, reset, 40))

if __name__ == '__main__':
    main()
