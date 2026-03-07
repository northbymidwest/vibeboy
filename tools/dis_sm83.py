#!/usr/bin/env python3
"""SM83 (Game Boy CPU) disassembler for test ROM analysis.

Supports all SM83 opcodes including CB-prefixed bit operations.
Can disassemble at arbitrary ROM offsets or GB addresses.
"""

import sys
import argparse

# Main opcodes: mnemonic and operand byte count (excluding opcode itself)
# Format: (mnemonic_template, extra_bytes)
OPCODES = {
    0x00: ('NOP', 0), 0x01: ('LD BC,$%04X', 2), 0x02: ('LD (BC),A', 0), 0x03: ('INC BC', 0),
    0x04: ('INC B', 0), 0x05: ('DEC B', 0), 0x06: ('LD B,$%02X', 1), 0x07: ('RLCA', 0),
    0x08: ('LD ($%04X),SP', 2), 0x09: ('ADD HL,BC', 0), 0x0A: ('LD A,(BC)', 0), 0x0B: ('DEC BC', 0),
    0x0C: ('INC C', 0), 0x0D: ('DEC C', 0), 0x0E: ('LD C,$%02X', 1), 0x0F: ('RRCA', 0),

    0x10: ('STOP', 0), 0x11: ('LD DE,$%04X', 2), 0x12: ('LD (DE),A', 0), 0x13: ('INC DE', 0),
    0x14: ('INC D', 0), 0x15: ('DEC D', 0), 0x16: ('LD D,$%02X', 1), 0x17: ('RLA', 0),
    0x18: ('JR $%04X', -1), 0x19: ('ADD HL,DE', 0), 0x1A: ('LD A,(DE)', 0), 0x1B: ('DEC DE', 0),
    0x1C: ('INC E', 0), 0x1D: ('DEC E', 0), 0x1E: ('LD E,$%02X', 1), 0x1F: ('RRA', 0),

    0x20: ('JR NZ,$%04X', -1), 0x21: ('LD HL,$%04X', 2), 0x22: ('LD (HL+),A', 0), 0x23: ('INC HL', 0),
    0x24: ('INC H', 0), 0x25: ('DEC H', 0), 0x26: ('LD H,$%02X', 1), 0x27: ('DAA', 0),
    0x28: ('JR Z,$%04X', -1), 0x29: ('ADD HL,HL', 0), 0x2A: ('LD A,(HL+)', 0), 0x2B: ('DEC HL', 0),
    0x2C: ('INC L', 0), 0x2D: ('DEC L', 0), 0x2E: ('LD L,$%02X', 1), 0x2F: ('CPL', 0),

    0x30: ('JR NC,$%04X', -1), 0x31: ('LD SP,$%04X', 2), 0x32: ('LD (HL-),A', 0), 0x33: ('INC SP', 0),
    0x34: ('INC (HL)', 0), 0x35: ('DEC (HL)', 0), 0x36: ('LD (HL),$%02X', 1), 0x37: ('SCF', 0),
    0x38: ('JR C,$%04X', -1), 0x39: ('ADD HL,SP', 0), 0x3A: ('LD A,(HL-)', 0), 0x3B: ('DEC SP', 0),
    0x3C: ('INC A', 0), 0x3D: ('DEC A', 0), 0x3E: ('LD A,$%02X', 1), 0x3F: ('CCF', 0),

    0x40: ('LD B,B', 0), 0x41: ('LD B,C', 0), 0x42: ('LD B,D', 0), 0x43: ('LD B,E', 0),
    0x44: ('LD B,H', 0), 0x45: ('LD B,L', 0), 0x46: ('LD B,(HL)', 0), 0x47: ('LD B,A', 0),
    0x48: ('LD C,B', 0), 0x49: ('LD C,C', 0), 0x4A: ('LD C,D', 0), 0x4B: ('LD C,E', 0),
    0x4C: ('LD C,H', 0), 0x4D: ('LD C,L', 0), 0x4E: ('LD C,(HL)', 0), 0x4F: ('LD C,A', 0),

    0x50: ('LD D,B', 0), 0x51: ('LD D,C', 0), 0x52: ('LD D,D', 0), 0x53: ('LD D,E', 0),
    0x54: ('LD D,H', 0), 0x55: ('LD D,L', 0), 0x56: ('LD D,(HL)', 0), 0x57: ('LD D,A', 0),
    0x58: ('LD E,B', 0), 0x59: ('LD E,C', 0), 0x5A: ('LD E,D', 0), 0x5B: ('LD E,E', 0),
    0x5C: ('LD E,H', 0), 0x5D: ('LD E,L', 0), 0x5E: ('LD E,(HL)', 0), 0x5F: ('LD E,A', 0),

    0x60: ('LD H,B', 0), 0x61: ('LD H,C', 0), 0x62: ('LD H,D', 0), 0x63: ('LD H,E', 0),
    0x64: ('LD H,H', 0), 0x65: ('LD H,L', 0), 0x66: ('LD H,(HL)', 0), 0x67: ('LD H,A', 0),
    0x68: ('LD L,B', 0), 0x69: ('LD L,C', 0), 0x6A: ('LD L,D', 0), 0x6B: ('LD L,E', 0),
    0x6C: ('LD L,H', 0), 0x6D: ('LD L,L', 0), 0x6E: ('LD L,(HL)', 0), 0x6F: ('LD L,A', 0),

    0x70: ('LD (HL),B', 0), 0x71: ('LD (HL),C', 0), 0x72: ('LD (HL),D', 0), 0x73: ('LD (HL),E', 0),
    0x74: ('LD (HL),H', 0), 0x75: ('LD (HL),L', 0), 0x76: ('HALT', 0), 0x77: ('LD (HL),A', 0),
    0x78: ('LD A,B', 0), 0x79: ('LD A,C', 0), 0x7A: ('LD A,D', 0), 0x7B: ('LD A,E', 0),
    0x7C: ('LD A,H', 0), 0x7D: ('LD A,L', 0), 0x7E: ('LD A,(HL)', 0), 0x7F: ('LD A,A', 0),

    0x80: ('ADD A,B', 0), 0x81: ('ADD A,C', 0), 0x82: ('ADD A,D', 0), 0x83: ('ADD A,E', 0),
    0x84: ('ADD A,H', 0), 0x85: ('ADD A,L', 0), 0x86: ('ADD A,(HL)', 0), 0x87: ('ADD A,A', 0),
    0x88: ('ADC A,B', 0), 0x89: ('ADC A,C', 0), 0x8A: ('ADC A,D', 0), 0x8B: ('ADC A,E', 0),
    0x8C: ('ADC A,H', 0), 0x8D: ('ADC A,L', 0), 0x8E: ('ADC A,(HL)', 0), 0x8F: ('ADC A,A', 0),

    0x90: ('SUB B', 0), 0x91: ('SUB C', 0), 0x92: ('SUB D', 0), 0x93: ('SUB E', 0),
    0x94: ('SUB H', 0), 0x95: ('SUB L', 0), 0x96: ('SUB (HL)', 0), 0x97: ('SUB A', 0),
    0x98: ('SBC A,B', 0), 0x99: ('SBC A,C', 0), 0x9A: ('SBC A,D', 0), 0x9B: ('SBC A,E', 0),
    0x9C: ('SBC A,H', 0), 0x9D: ('SBC A,L', 0), 0x9E: ('SBC A,(HL)', 0), 0x9F: ('SBC A,A', 0),

    0xA0: ('AND B', 0), 0xA1: ('AND C', 0), 0xA2: ('AND D', 0), 0xA3: ('AND E', 0),
    0xA4: ('AND H', 0), 0xA5: ('AND L', 0), 0xA6: ('AND (HL)', 0), 0xA7: ('AND A', 0),
    0xA8: ('XOR B', 0), 0xA9: ('XOR C', 0), 0xAA: ('XOR D', 0), 0xAB: ('XOR E', 0),
    0xAC: ('XOR H', 0), 0xAD: ('XOR L', 0), 0xAE: ('XOR (HL)', 0), 0xAF: ('XOR A', 0),

    0xB0: ('OR B', 0), 0xB1: ('OR C', 0), 0xB2: ('OR D', 0), 0xB3: ('OR E', 0),
    0xB4: ('OR H', 0), 0xB5: ('OR L', 0), 0xB6: ('OR (HL)', 0), 0xB7: ('OR A', 0),
    0xB8: ('CP B', 0), 0xB9: ('CP C', 0), 0xBA: ('CP D', 0), 0xBB: ('CP E', 0),
    0xBC: ('CP H', 0), 0xBD: ('CP L', 0), 0xBE: ('CP (HL)', 0), 0xBF: ('CP A', 0),

    0xC0: ('RET NZ', 0), 0xC1: ('POP BC', 0), 0xC2: ('JP NZ,$%04X', 2), 0xC3: ('JP $%04X', 2),
    0xC4: ('CALL NZ,$%04X', 2), 0xC5: ('PUSH BC', 0), 0xC6: ('ADD A,$%02X', 1), 0xC7: ('RST $00', 0),
    0xC8: ('RET Z', 0), 0xC9: ('RET', 0), 0xCA: ('JP Z,$%04X', 2), 0xCB: ('PREFIX CB', 0),
    0xCC: ('CALL Z,$%04X', 2), 0xCD: ('CALL $%04X', 2), 0xCE: ('ADC A,$%02X', 1), 0xCF: ('RST $08', 0),

    0xD0: ('RET NC', 0), 0xD1: ('POP DE', 0), 0xD2: ('JP NC,$%04X', 2), 0xD3: ('???', 0),
    0xD4: ('CALL NC,$%04X', 2), 0xD5: ('PUSH DE', 0), 0xD6: ('SUB $%02X', 1), 0xD7: ('RST $10', 0),
    0xD8: ('RET C', 0), 0xD9: ('RETI', 0), 0xDA: ('JP C,$%04X', 2), 0xDB: ('???', 0),
    0xDC: ('CALL C,$%04X', 2), 0xDD: ('???', 0), 0xDE: ('SBC A,$%02X', 1), 0xDF: ('RST $18', 0),

    0xE0: ('LDH ($FF%02X),A', 1), 0xE1: ('POP HL', 0), 0xE2: ('LD ($FF00+C),A', 0), 0xE3: ('???', 0),
    0xE4: ('???', 0), 0xE5: ('PUSH HL', 0), 0xE6: ('AND $%02X', 1), 0xE7: ('RST $20', 0),
    0xE8: ('ADD SP,$%02X', 1), 0xE9: ('JP HL', 0), 0xEA: ('LD ($%04X),A', 2), 0xEB: ('???', 0),
    0xEC: ('???', 0), 0xED: ('???', 0), 0xEE: ('XOR $%02X', 1), 0xEF: ('RST $28', 0),

    0xF0: ('LDH A,($FF%02X)', 1), 0xF1: ('POP AF', 0), 0xF2: ('LD A,($FF00+C)', 0), 0xF3: ('DI', 0),
    0xF4: ('???', 0), 0xF5: ('PUSH AF', 0), 0xF6: ('OR $%02X', 1), 0xF7: ('RST $30', 0),
    0xF8: ('LD HL,SP+$%02X', 1), 0xF9: ('LD SP,HL', 0), 0xFA: ('LD A,($%04X)', 2), 0xFB: ('EI', 0),
    0xFC: ('???', 0), 0xFD: ('???', 0), 0xFE: ('CP $%02X', 1), 0xFF: ('RST $38', 0),
}

# CB-prefixed opcodes
CB_REGS = ['B', 'C', 'D', 'E', 'H', 'L', '(HL)', 'A']
CB_OPS = ['RLC', 'RRC', 'RL', 'RR', 'SLA', 'SRA', 'SWAP', 'SRL']

def cb_mnemonic(op):
    reg = CB_REGS[op & 7]
    hi = op >> 3
    if hi < 8:
        return f'{CB_OPS[hi]} {reg}'
    elif hi < 16:
        return f'BIT {hi - 8},{reg}'
    elif hi < 24:
        return f'RES {hi - 16},{reg}'
    else:
        return f'SET {hi - 24},{reg}'


# JR opcodes (relative branches) — use -1 as sentinel for extra_bytes
JR_OPCODES = {0x18, 0x20, 0x28, 0x30, 0x38}


def disassemble_one(rom, offset, pc):
    """Disassemble one instruction. Returns (text, raw_bytes, size)."""
    if offset >= len(rom):
        return ('(out of range)', '', 0)

    op = rom[offset]

    # CB prefix
    if op == 0xCB:
        if offset + 1 >= len(rom):
            return ('CB ???', 'CB', 1)
        cb_op = rom[offset + 1]
        mn = cb_mnemonic(cb_op)
        raw = f'{op:02X} {cb_op:02X}'
        return (mn, raw, 2)

    template, extra = OPCODES.get(op, ('???', 0))

    if extra == 0:
        return (template, f'{op:02X}', 1)
    elif extra == -1:
        # Relative branch
        if offset + 1 >= len(rom):
            return (template, f'{op:02X}', 1)
        b1 = rom[offset + 1]
        rel = b1 if b1 < 0x80 else b1 - 256
        target = (pc + 2 + rel) & 0xFFFF
        mn = template % target
        raw = f'{op:02X} {b1:02X}'
        return (mn, raw, 2)
    elif extra == 1:
        if offset + 1 >= len(rom):
            return (template, f'{op:02X}', 1)
        b1 = rom[offset + 1]
        mn = template % b1
        raw = f'{op:02X} {b1:02X}'
        return (mn, raw, 2)
    elif extra == 2:
        if offset + 2 >= len(rom):
            return (template, f'{op:02X}', 1)
        lo = rom[offset + 1]
        hi = rom[offset + 2]
        val = lo | (hi << 8)
        mn = template % val
        raw = f'{op:02X} {lo:02X} {hi:02X}'
        return (mn, raw, 3)

    return ('???', f'{op:02X}', 1)


def disassemble(rom, offset, pc, count):
    """Disassemble `count` instructions from ROM at `offset`, starting at GB address `pc`."""
    lines = []
    for _ in range(count):
        if offset >= len(rom):
            lines.append(f'${pc:04X}: (out of range)')
            break
        mn, raw, size = disassemble_one(rom, offset, pc)
        if size == 0:
            break
        lines.append(f'${pc:04X}: {raw:12s} {mn}')
        offset += size
        pc = (pc + size) & 0xFFFF
    return '\n'.join(lines)


def hex_dump(rom, offset, length, pc):
    """Hex dump of ROM bytes."""
    lines = []
    for i in range(0, length, 16):
        addr = pc + i
        chunk = rom[offset + i:offset + i + 16]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f'${addr:04X}: {hex_str:<48s} {ascii_str}')
    return '\n'.join(lines)


def search_bytes(rom, pattern):
    """Search ROM for a byte pattern, return list of offsets."""
    results = []
    for i in range(len(rom) - len(pattern) + 1):
        if rom[i:i + len(pattern)] == pattern:
            results.append(i)
    return results


def main():
    parser = argparse.ArgumentParser(description='SM83 (Game Boy) disassembler')
    parser.add_argument('rom', help='ROM file path')
    parser.add_argument('--offset', type=lambda x: int(x, 16), default=None,
                        help='ROM file offset in hex (e.g. 4037)')
    parser.add_argument('--pc', type=lambda x: int(x, 16), default=None,
                        help='GB address for display (hex, e.g. C037). If --offset not given, '
                             'uses standard ROM mapping: bank0=$0000-$3FFF at offset 0, '
                             'bank1=$4000-$7FFF at offset $4000')
    parser.add_argument('-n', '--count', type=int, default=30,
                        help='Number of instructions to disassemble')
    parser.add_argument('--hex', action='store_true',
                        help='Hex dump instead of disassembly')
    parser.add_argument('--hex-len', type=int, default=256,
                        help='Length of hex dump in bytes')
    parser.add_argument('--search', type=lambda x: bytes.fromhex(x), default=None,
                        help='Search for hex byte pattern (e.g. FE0C)')
    parser.add_argument('--context', type=int, default=5,
                        help='Instructions of context around search results')
    parser.add_argument('--header', action='store_true',
                        help='Show cartridge header info')
    args = parser.parse_args()

    with open(args.rom, 'rb') as f:
        rom = f.read()

    if args.header:
        title = rom[0x134:0x144].rstrip(b'\x00').decode('ascii', errors='replace')
        cgb = rom[0x143] if len(rom) > 0x143 else 0
        sgb = rom[0x146] if len(rom) > 0x146 else 0
        cart_type = rom[0x147] if len(rom) > 0x147 else 0
        rom_size = rom[0x148] if len(rom) > 0x148 else 0
        ram_size = rom[0x149] if len(rom) > 0x149 else 0
        print(f'=== Cartridge Header ===')
        print(f'  Title:     {title}')
        print(f'  CGB flag:  ${cgb:02X}')
        print(f'  SGB flag:  ${sgb:02X}')
        print(f'  Cart type: ${cart_type:02X}')
        print(f'  ROM size:  ${rom_size:02X} ({32 << rom_size} KB)')
        print(f'  RAM size:  ${ram_size:02X}')
        print(f'  ROM file:  {len(rom)} bytes ({len(rom)//1024} KB)')
        print()

    if args.search is not None:
        results = search_bytes(rom, args.search)
        print(f'Found {len(results)} match(es) for {args.search.hex().upper()}:')
        for off in results:
            # Guess PC from offset
            if off < 0x4000:
                pc = off
            elif off < 0x8000:
                pc = off  # bank 1
            else:
                pc = off  # raw offset as PC
            print(f'\n  ROM offset ${off:04X} (PC ~${pc:04X}):')
            ctx_off = max(0, off - args.context * 2)
            ctx_pc = pc - (off - ctx_off)
            print(disassemble(rom, ctx_off, ctx_pc, args.context * 2 + 5))
        print()

    # Determine offset and PC
    if args.offset is not None or args.pc is not None:
        if args.offset is not None:
            offset = args.offset
            pc = args.pc if args.pc is not None else offset
        else:
            # Map PC to ROM offset
            pc = args.pc
            if pc < 0x4000:
                offset = pc
            elif pc < 0x8000:
                offset = pc  # bank 1 at file offset 0x4000
            else:
                offset = pc  # direct mapping for WRAM-loaded code

        if args.hex:
            print(f'=== Hex dump at offset ${offset:04X} (PC ${pc:04X}), {args.hex_len} bytes ===')
            print(hex_dump(rom, offset, min(args.hex_len, len(rom) - offset), pc))
        else:
            print(f'=== Disassembly at offset ${offset:04X} (PC ${pc:04X}), {args.count} instructions ===')
            print(disassemble(rom, offset, pc, args.count))

    if args.offset is None and args.pc is None and args.search is None and not args.header:
        # Default: show header + entry point
        args.header = True
        print(f'=== Entry point ($0100) ===')
        print(disassemble(rom, 0x100, 0x100, 20))


if __name__ == '__main__':
    main()
