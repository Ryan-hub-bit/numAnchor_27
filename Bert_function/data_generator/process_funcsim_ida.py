#!/usr/bin/env python3
"""
IDA Pro script to extract functions for the funcsim dataset.

Uses EXACTLY the same instruction formatting as func_gen.py / build_function_inline():
  - opcode(hex_addr:fp:bp:ip) operand1 operand2 ...
  - internal code refs  -> address(0xADDR:fp:bp:ip)
  - external data refs  -> daddr(0xADDR:sec:sec_off:0.0)
  - symbol calls/jumps  -> .symbol_name  (smart-merged against top_symbols list)
  - stack variables     -> var(0xOFFSET)
  - immediates          -> imm

Output: one JSON file per binary  [{func_id, binary, function, opt_level, num_instructions, num_tokens, funcins}, ...]
        opt_level derived from the binary's parent directory name (e.g. bind_O1/arpaname -> O1)
        funcins = TAB-joined instruction strings

Usage:
  idat -A -S"process_funcsim_ida.py OUTPUT_DIR SYMBOLS_FILE ENABLE_SMART_MERGE" binary
"""

import sys
import os
import re
import json
from pathlib import Path

import idaapi
import idautils
import idc
import ida_auto
import ida_funcs
import ida_segment
import ida_bytes
import ida_name

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


# ── symbol helpers ────────────────────────────────────────────────────────────

def load_top_symbols(symbols_file):
    """Load top symbols from file (same format as combine_cfg_2inline.py)."""
    try:
        with open(symbols_file, 'r') as f:
            top_symbols = set(
                line.split('\t')[0].lstrip('.') for line in f if line.strip()
            )
        print(f"[INFO] Loaded {len(top_symbols)} top symbols from {symbols_file}")
        return top_symbols
    except Exception as e:
        print(f"[ERROR] Failed to load symbols: {e}")
        return None


def smart_merge_symbol(symbol_name, top_symbols):
    """Replace low-frequency symbols with .plt."""
    if top_symbols is None:
        return symbol_name
    name = symbol_name.lstrip('.')
    return symbol_name if name in top_symbols else '.plt'


# ── raw disassembly (identical to func_gen.py clean_ida_disasm) ───────────────

def clean_ida_disasm(ea):
    mnem = idc.print_insn_mnem(ea)
    if not mnem:
        return None

    insn = idaapi.insn_t()
    idaapi.decode_insn(insn, ea)

    operands = []
    for i in range(6):
        op_type = idc.get_operand_type(ea, i)
        if op_type == idc.o_void:
            break

        op_value = idc.get_operand_value(ea, i)

        if op_type == idc.o_reg:
            reg_name = idaapi.get_reg_name(op_value, 8) or idc.print_operand(ea, i)
            operands.append(reg_name)

        elif op_type == idc.o_imm:
            operands.append("imm")

        elif op_type == idc.o_displ:
            op = insn.ops[i]
            base = idaapi.get_reg_name(op.reg, 8)
            off  = op.addr
            if base in ['rbp', 'ebp']:
                operands.append(f"[{base}+var_{off:X}h]")
            else:
                flags = idc.get_full_flags(ea)
                if idaapi.is_stkvar(flags, i):
                    operands.append(f"[{base}+var_{off:X}h]")
                else:
                    operands.append(f"[{base}+disp]")

        elif op_type == idc.o_phrase:
            op = insn.ops[i]
            base = idaapi.get_reg_name(op.reg, 8)
            if hasattr(op, 'specflag1') and op.specflag1:
                operands.append(f"[{base}+disp]")
            else:
                operands.append(f"[{base}]")

        elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
            if op_value != idaapi.BADADDR and op_value != 0:
                operands.append(hex(op_value))
            else:
                operands.append(idc.print_operand(ea, i))

        else:
            op_str = idc.print_operand(ea, i)
            operands.append(op_str if op_str else "unknown")

    return f"{mnem} {', '.join(operands)}" if operands else mnem


# ── normalize_and_mask (identical to func_gen.py) ─────────────────────────────

def normalize_and_mask(ins_raw: str, symbol_map: dict, string_map: dict):
    ins   = re.sub(r"\s+", ", ", ins_raw, 1)
    parts = ins.split(", ")
    opcode   = parts[0]
    operands = parts[1:] if len(parts) > 1 else []

    out_tokens  = [opcode]
    mask_tokens = ["0"]

    for op in operands:
        pieces = re.split(r"(0x[0-9A-Fa-f]+|[A-Za-z0-9_]+|\[|\]|,|:|\(|\))", op)
        for tok in pieces:
            if not tok or tok.isspace():
                continue
            if tok.startswith("0x") and bool(HEX_RE.fullmatch(tok)):
                out_tokens.append(tok)
                mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    return " ".join(out_tokens), " ".join(mask_tokens)


# ── build_function_inline (same logic as func_gen.py) ────────────────────────

def build_function_inline(func_instructions, ctx: dict, top_symbols=None):
    """
    Format all instructions using func_gen.py's hierarchical encoding.
    Returns TAB-joined instruction strings.
    """
    addr_positions = ctx['addr_positions']
    bb_range_map   = ctx['bb_range_map']
    min_addr       = ctx['min_addr']
    max_addr       = ctx['max_addr']
    sections       = ctx['sections']
    symbol_map     = ctx['symbol_map']

    def get_section_for_addr(addr):
        for s_start, s_end, s_name in sections:
            if s_start <= addr < s_end:
                return s_start, s_end, s_name
        return None

    def format_data_address_positions(addr):
        info = get_section_for_addr(addr)
        if info is None:
            return "2.00000000:0.00000000:0.00000000"
        sec_start, sec_end, sec_name = info
        sname = sec_name.lower()
        is_code = ".text" in sname or ".plt" in sname or sname in ("text", "plt")
        is_data = ".data" in sname or ".rodata" in sname or ".bss" in sname or ".got" in sname
        if not is_code and not is_data:
            return "2.00000000:0.00000000:0.00000000"
        sec_in_bin   = max(0.0, min(1.0, (sec_start - min_addr) / float(max_addr - min_addr))) if max_addr > min_addr else 0.0
        addr_in_sec  = max(0.0, min(1.0, (addr - sec_start) / float(sec_end - sec_start))) if sec_end > sec_start else 0.0
        return f"{sec_in_bin:.8f}:{addr_in_sec:.8f}:0.00000000"

    def format_hier_pos(entry):
        _, addr, _, bb_start, func_start, func_end = entry
        fp = max(0.0, min(1.0, (func_start - min_addr) / float(max_addr - min_addr))) if max_addr > min_addr else 0.0
        bp = max(0.0, min(1.0, (bb_start - func_start) / float(func_end - func_start))) \
             if bb_start is not None and func_end > func_start else 0.0
        if bb_start is not None and bb_start in bb_range_map:
            bb_s, bb_e = bb_range_map[bb_start]
            ip = max(0.0, min(1.0, (addr - bb_s) / float(bb_e - bb_s))) if bb_e > bb_s else 0.0
        else:
            ip = 0.0
        return f"{fp:.8f}:{bp:.8f}:{ip:.8f}"

    out_instrs = []
    for addr, norm_line, mask_line in func_instructions:
        tokens        = norm_line.strip().split()
        masks         = mask_line.strip().split()
        if not tokens:
            continue
        opcode        = tokens[0]
        operands      = tokens[1:]
        operand_masks = masks[1:]

        is_cf = opcode.lower() in {
            'call', 'jmp', 'jmpq', 'callq',
            'ja', 'jb', 'jc', 'je', 'jg', 'jl',
            'jn', 'jo', 'jp', 'js', 'jz',
        }

        formatted_ops = []
        for i, tok in enumerate(operands):
            mk     = operand_masks[i] if i < len(operand_masks) else "0"
            mk_hex = None

            if isinstance(mk, str) and mk.startswith('0x'):
                mk_hex = mk
            elif isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
                mk_hex = tok
            elif isinstance(tok, str) and re.match(r'^[0-9A-Fa-f]+h$', tok, re.IGNORECASE):
                mk_hex = '0x' + tok[:-1]

            if mk_hex is not None:
                try:
                    tgt = int(mk_hex, 16)
                except Exception:
                    tgt = None

                addr_threshold = max(min_addr, 0x1000)
                if tgt is not None and tgt >= addr_threshold:
                    if tgt in addr_positions:
                        # Internal code reference → address(...)
                        pos = format_hier_pos(addr_positions[tgt])
                        formatted_ops.append(f"address({mk_hex}:{pos})")
                    else:
                        # External reference
                        sym = symbol_map.get(tgt)
                        if sym and top_symbols is not None:
                            sym = smart_merge_symbol(sym, top_symbols)

                        info = get_section_for_addr(tgt)
                        if info:
                            sec_start, sec_end, sec_name = info
                            sn = sec_name.lower()
                            is_code = ".plt" in sn or sn in ("plt",) or ".text" in sn or \
                                      sn in ("text",) or ".extern" in sn
                            is_data = ".data" in sn or ".rodata" in sn or ".bss" in sn or ".got" in sn

                            if is_code and sym:
                                formatted_ops.append(sym if sym.startswith('.') else f'.{sym}')
                            elif is_data:
                                if is_cf and sym:
                                    formatted_ops.append(sym if sym.startswith('.') else f'.{sym}')
                                elif is_cf:
                                    formatted_ops.append(mk_hex)
                                else:
                                    formatted_ops.append(f"daddr({mk_hex}:{format_data_address_positions(tgt)})")
                            elif sym:
                                formatted_ops.append(sym if sym.startswith('.') else f'.{sym}')
                            else:
                                formatted_ops.append(f"daddr({mk_hex}:{format_data_address_positions(tgt)})")
                        else:
                            if sym:
                                formatted_ops.append(sym if sym.startswith('.') else f'.{sym}')
                            else:
                                formatted_ops.append(f"daddr({mk_hex}:2.00000000:0.00000000:0.00000000)")
                else:
                    formatted_ops.append("imm")
            else:
                if tok.startswith("var_"):
                    off = tok[4:].rstrip('hH').lstrip('sS')
                    formatted_ops.append(f"var(0x{off})")
                elif tok.startswith("arg_"):
                    off = tok[4:].lstrip('sS')
                    formatted_ops.append(f"var(0x{off})")
                elif re.match(r'^0x[0-9a-fA-F]+$', tok):
                    formatted_ops.append("imm")
                else:
                    formatted_ops.append(tok)

        addr_hdr = f"{hex(addr)}:{format_hier_pos(addr_positions[addr])}" \
                   if addr in addr_positions else hex(addr)

        ops_join = ' '.join(formatted_ops)
        if ops_join.count('[') > ops_join.count(']'):
            ops_join += ' ]'

        out_instrs.append(f"{opcode}({addr_hdr}) {ops_join}" if ops_join.strip()
                          else f"{opcode}({addr_hdr})")

    return "\t".join(out_instrs)


# ── main IDA extraction ───────────────────────────────────────────────────────

def process_binary(output_dir, symbols_file=None, enable_smart_merge=False):
    print("[INFO] Waiting for auto-analysis…")
    ida_auto.auto_wait()

    binary_path = idaapi.get_input_file_path()
    binary_name = Path(binary_path).name

    # Derive opt_level from the parent directory name.
    # Supports both:
    #   bind_O1/arpaname
    #   coreutils-7.6-O1/ls
    parent_dir = Path(binary_path).parent.name
    opt_level = "unknown"
    for _o in ['O0', 'O1', 'O2', 'O3']:
        if parent_dir.endswith('_' + _o) or parent_dir.endswith('-' + _o) or parent_dir == _o:
            opt_level = _o
            break
    print(f"[INFO] Processing {binary_name}  opt={opt_level} (from dir: {parent_dir})")

    # top symbols for smart merge
    top_symbols = None
    if enable_smart_merge and symbols_file and os.path.exists(symbols_file):
        top_symbols = load_top_symbols(symbols_file)

    # symbol map
    symbol_map = {addr: name for addr, name in idautils.Names()}
    print(f"[INFO] {len(symbol_map)} symbols")

    string_map = {}

    # segments → address range
    sections = []
    for seg in idautils.Segments():
        sections.append((idc.get_segm_start(seg), idc.get_segm_end(seg), idc.get_segm_name(seg)))
    min_addr = min(s[0] for s in sections) if sections else 0
    max_addr = max(s[1] for s in sections) if sections else 0
    print(f"[INFO] Range {hex(min_addr)}–{hex(max_addr)}")

    addr_positions = {}
    bb_range_map   = {}
    bin_counter    = 0
    func_data      = []

    for func_ea in idautils.Functions():
        func = ida_funcs.get_func(func_ea)
        if not func:
            continue
        if idc.get_segm_name(func_ea) in ['.plt', 'extern', '.init', '.fini']:
            continue

        func_name  = idc.get_func_name(func_ea)
        func_start = func.start_ea
        func_end   = func.end_ea
        flowchart  = idaapi.FlowChart(func)

        for bb in flowchart:
            bb_s = bb.start_ea
            bb_e = bb.end_ea
            bb_range_map[bb_s] = (bb_s, bb_e)
            ea = bb_s
            while ea < bb_e:
                addr_positions[ea] = (bin_counter, ea, func_name, bb_s, func_start, func_end)
                bin_counter += 1
                ea = idc.next_head(ea, bb_e)

        func_instrs = []
        for bb in flowchart:
            ea = bb.start_ea
            while ea < bb.end_ea:
                raw = clean_ida_disasm(ea)
                if raw:
                    norm, mask = normalize_and_mask(raw, symbol_map, string_map)
                    func_instrs.append((ea, norm, mask))
                ea = idc.next_head(ea, bb.end_ea)
                if ea == idaapi.BADADDR or ea >= bb.end_ea:
                    break

        if func_instrs:
            func_data.append((func_name, func_start, func_end, func_instrs))

    print(f"[INFO] {len(func_data)} functions collected")

    ctx = {
        'addr_positions': addr_positions,
        'bb_range_map':   bb_range_map,
        'min_addr':       min_addr,
        'max_addr':       max_addr,
        'sections':       sections,
        'symbol_map':     symbol_map,
    }

    MIN_INSTRUCTIONS = 5
    results  = []
    func_id  = 0

    for func_name, func_start, func_end, func_instrs in func_data:
        if len(func_instrs) < MIN_INSTRUCTIONS:
            continue
        funcins = build_function_inline(func_instrs, ctx, top_symbols=top_symbols)
        if not funcins:
            continue

        # Count instructions and tokens
        instr_list = [s for s in funcins.split('\t') if s.strip()]
        num_instructions = len(instr_list)
        num_tokens = sum(len(s.split()) for s in instr_list)

        results.append({
            'func_id':         func_id,
            'binary':          binary_name,
            'function':        func_name,
            'opt_level':       opt_level,
            'num_instructions': num_instructions,
            'num_tokens':       num_tokens,
            'funcins':         funcins,
        })
        func_id += 1

    print(f"[INFO] {len(results)} functions encoded (>= {MIN_INSTRUCTIONS} instrs)")

    # Include opt_level in filename to avoid overwriting across optimization levels
    out = Path(output_dir) / f"{binary_name}_{opt_level}_functions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f)
    print(f"[INFO] Wrote {out}")
    return len(results)


def main():
    if len(idc.ARGV) < 2:
        print("[ERROR] Usage: idat -A -S\"process_funcsim_ida.py OUTPUT_DIR SYMBOLS_FILE ENABLE_SMART_MERGE\" binary")
        idc.qexit(1)

    output_dir         = idc.ARGV[1]
    symbols_file       = idc.ARGV[2] if len(idc.ARGV) > 2 else None
    enable_smart_merge = (idc.ARGV[3].lower() == "true") if len(idc.ARGV) > 3 else False

    try:
        n = process_binary(output_dir, symbols_file, enable_smart_merge)
        print(f"[SUCCESS] {n} functions")
        idc.qexit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        idc.qexit(1)


if __name__ == "__main__":
    main()
