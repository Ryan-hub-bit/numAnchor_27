#!/usr/bin/env python3
"""
IDA Pro script for exporting SymLM-style function-name-prediction data in the
address-aware inline format used by this repository's pretraining pipeline.

Compared with the upstream SymLM scripts:
- uses IDA Pro instead of Ghidra
- exports function bodies as PalmTree-style inline instructions with address info
- preserves caller / callee context metadata for later SymLM-style use

Output: one JSON per binary
  [
    {
      "func_id": 0,
      "binary": "binary_name",
      "function": "semantic_name",
      "opt_level": "O2",
      "num_instructions": 42,
      "num_tokens": 133,
      "funcins": "mov(0x...:...) ...\\tcall(0x...:...) address(...)",
      "callers": ["foo", "bar"],
      "internal_callees": ["baz"],
      "external_callees": ["printf"]
    },
    ...
  ]

Usage:
  idat -A -S"process_address_aware_ida.py OUTPUT_DIR SYMBOLS_FILE ENABLE_SMART_MERGE" binary
"""

import json
import os
import re
import importlib.util
from pathlib import Path

import idaapi
import ida_auto
import ida_funcs
import idautils
import idc


OPT_LEVELS = ("O0", "O1", "O2", "O3")
ARCH_NAMES = {"x86", "x64", "arm", "mips"}


def _get_script_dir():
    script_path = globals().get("__file__")
    if not script_path and len(idc.ARGV) > 0:
        script_path = idc.ARGV[0]
    if not script_path:
        raise RuntimeError("Unable to resolve current script path in IDA.")
    return Path(script_path).resolve().parent


SCRIPT_DIR = _get_script_dir()
# dataset_generation -> SymLM -> FunctionNamePrediction -> PalmTree
PALMTREE_ROOT = SCRIPT_DIR.parents[2]
HELPER_PATH = PALMTREE_ROOT / "data_generator" / "process_funcsim_ida.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("palmtree_process_funcsim_ida", HELPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


helper = _load_helper()

REGISTER_RE = re.compile(
    r"^(r\d+[bwd]?|[re]?(ax|bx|cx|dx|si|di|sp|bp)|[abcd][lh]|"
    r"[cdefgs]s|xmm\d+|ymm\d+|zmm\d+)$",
    re.IGNORECASE,
)


def sanitize_component(name):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def infer_opt_level(binary_path):
    for candidate in [binary_path.parent] + list(binary_path.parents):
        name = candidate.name
        if name in OPT_LEVELS:
            return name
        for opt in OPT_LEVELS:
            if name.endswith("_" + opt):
                return opt
    return "unknown"


def build_source_tag(binary_path, opt_level):
    arch = None
    for parent in binary_path.parents:
        if parent.name in ARCH_NAMES:
            arch = parent.name
            break

    parent_name = binary_path.parent.name
    grandparent_name = binary_path.parent.parent.name if binary_path.parent.parent != binary_path.parent else ""

    parts = []
    if arch:
        parts.append(arch)
    if opt_level != "unknown":
        parts.append(opt_level)
    if grandparent_name and grandparent_name not in ARCH_NAMES and grandparent_name not in OPT_LEVELS:
        parts.append(grandparent_name)
    if parent_name and parent_name not in ARCH_NAMES and parent_name not in OPT_LEVELS:
        parts.append(parent_name)
    parts.append(binary_path.name)

    sanitized = [sanitize_component(part) for part in parts if part]
    deduped = []
    for part in sanitized:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    return "_".join(deduped)


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def in_filtered_segment(ea):
    return idc.get_segm_name(ea) in [".plt", "extern", ".init", ".fini"]


def normalize_external_name(name, top_symbols=None):
    if not name:
        return None

    name = name.strip()
    if not name:
        return None

    if top_symbols is not None:
        name = helper.smart_merge_symbol(name, top_symbols)

    name = name.strip()
    if name.startswith("."):
        name = name[1:]
    if "@plt" in name:
        name = name.split("@plt", 1)[0]
    if "::" in name:
        name = name.split("::", 1)[-1]

    if not name or name.startswith("0x") or REGISTER_RE.match(name):
        return None
    if "[" in name or "]" in name:
        return None
    return name


def collect_calling_context(func_start_ea, symbol_map, top_symbols=None):
    callers = []
    internal_callees = []
    external_callees = []

    for ref_ea in idautils.CodeRefsTo(func_start_ea, False):
        caller_func = ida_funcs.get_func(ref_ea)
        if not caller_func or caller_func.start_ea == func_start_ea:
            continue
        if in_filtered_segment(caller_func.start_ea):
            continue
        callers.append(idc.get_func_name(caller_func.start_ea))

    call_mnems = {"call", "callq", "bl", "blx", "jal", "jalr"}
    for ea in idautils.FuncItems(func_start_ea):
        mnem = idc.print_insn_mnem(ea).lower()
        if mnem not in call_mnems:
            continue

        refs = list(idautils.CodeRefsFrom(ea, False))
        found_internal = False
        for tgt in refs:
            callee_func = ida_funcs.get_func(tgt)
            if callee_func and not in_filtered_segment(callee_func.start_ea):
                callee_name = idc.get_func_name(callee_func.start_ea)
                if callee_name and callee_name != idc.get_func_name(func_start_ea):
                    internal_callees.append(callee_name)
                    found_internal = True

        if found_internal:
            continue

        operand = idc.print_operand(ea, 0).strip()
        ext_name = None

        for tgt in refs:
            sym = symbol_map.get(tgt)
            if sym:
                ext_name = normalize_external_name(sym, top_symbols=top_symbols)
                if ext_name:
                    break

        if ext_name is None:
            ext_name = normalize_external_name(operand, top_symbols=top_symbols)

        if ext_name:
            external_callees.append(ext_name)

    return {
        "callers": dedupe_keep_order(callers),
        "internal_callees": dedupe_keep_order(internal_callees),
        "external_callees": dedupe_keep_order(external_callees),
    }


def process_binary(output_dir, symbols_file=None, enable_smart_merge=False):
    print("[INFO] Waiting for auto-analysis...")
    ida_auto.auto_wait()

    binary_path = Path(idaapi.get_input_file_path())
    binary_name = binary_path.name
    opt_level = infer_opt_level(binary_path)
    source_tag = build_source_tag(binary_path, opt_level)
    print(f"[INFO] Processing {binary_name} opt={opt_level}")

    top_symbols = None
    if enable_smart_merge and symbols_file and os.path.exists(symbols_file):
        top_symbols = helper.load_top_symbols(symbols_file)

    symbol_map = {addr: name for addr, name in idautils.Names()}

    sections = []
    for seg in idautils.Segments():
        sections.append((idc.get_segm_start(seg), idc.get_segm_end(seg), idc.get_segm_name(seg)))
    min_addr = min(s[0] for s in sections) if sections else 0
    max_addr = max(s[1] for s in sections) if sections else 0

    addr_positions = {}
    bb_range_map = {}
    bin_counter = 0
    func_data = []

    for func_ea in idautils.Functions():
        func = ida_funcs.get_func(func_ea)
        if not func or in_filtered_segment(func_ea):
            continue

        func_name = idc.get_func_name(func_ea)
        flowchart = idaapi.FlowChart(func)

        for bb in flowchart:
            bb_s = bb.start_ea
            bb_e = bb.end_ea
            bb_range_map[bb_s] = (bb_s, bb_e)
            ea = bb_s
            while ea < bb_e:
                addr_positions[ea] = (bin_counter, ea, func_name, bb_s, func.start_ea, func.end_ea)
                bin_counter += 1
                ea = idc.next_head(ea, bb_e)

        func_instrs = []
        for bb in flowchart:
            ea = bb.start_ea
            while ea < bb.end_ea:
                raw = helper.clean_ida_disasm(ea)
                if raw:
                    norm, mask = helper.normalize_and_mask(raw, symbol_map, {})
                    func_instrs.append((ea, norm, mask))
                ea = idc.next_head(ea, bb.end_ea)
                if ea == idaapi.BADADDR or ea >= bb.end_ea:
                    break

        if func_instrs:
            func_data.append((func_name, func.start_ea, func.end_ea, func_instrs))

    ctx = {
        "addr_positions": addr_positions,
        "bb_range_map": bb_range_map,
        "min_addr": min_addr,
        "max_addr": max_addr,
        "sections": sections,
        "symbol_map": symbol_map,
    }

    results = []
    func_id = 0
    min_instructions = 5

    for func_name, func_start, _func_end, func_instrs in func_data:
        if len(func_instrs) < min_instructions:
            continue

        funcins = helper.build_function_inline(func_instrs, ctx, top_symbols=top_symbols)
        if not funcins:
            continue

        instrs = [s for s in funcins.split("\t") if s.strip()]
        context = collect_calling_context(func_start, symbol_map, top_symbols=top_symbols)
        results.append({
            "func_id": func_id,
            "binary": binary_name,
            "source_tag": source_tag,
            "source_path": str(binary_path),
            "function": func_name,
            "opt_level": opt_level,
            "num_instructions": len(instrs),
            "num_tokens": sum(len(s.split()) for s in instrs),
            "funcins": funcins,
            "callers": context["callers"],
            "internal_callees": context["internal_callees"],
            "external_callees": context["external_callees"],
        })
        func_id += 1

    out_path = Path(output_dir) / f"{source_tag}_addressaware_symlm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"[INFO] Wrote {len(results)} address-aware SymLM records to {out_path}")
    return len(results)


def main():
    if len(idc.ARGV) < 2:
        print(
            "[ERROR] Usage: idat -A -S\"process_address_aware_ida.py OUTPUT_DIR SYMBOLS_FILE ENABLE_SMART_MERGE\" binary"
        )
        idc.qexit(1)

    output_dir = idc.ARGV[1]
    symbols_file = idc.ARGV[2] if len(idc.ARGV) > 2 else None
    enable_smart_merge = (idc.ARGV[3].lower() == "true") if len(idc.ARGV) > 3 else False

    try:
        n = process_binary(output_dir, symbols_file, enable_smart_merge)
        print(f"[SUCCESS] {n} functions")
        idc.qexit(0)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        idc.qexit(1)


if __name__ == "__main__":
    main()
