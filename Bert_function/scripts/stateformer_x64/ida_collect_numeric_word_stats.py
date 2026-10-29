#!/usr/bin/env python3
"""
IDA batch helper for whole-binary numeric-word statistics.

Usage:
  idat -A -S"ida_collect_numeric_word_stats.py OUTPUT_JSON" binary

Counting rule:
  - every opcode counts as one word
  - every operand counts as one word
  - if an operand contains a numeric value, the whole operand counts as one
    numeric word (for example: "[rax+4]" is one numeric word)
"""

import json
import re

import ida_auto
import ida_bytes
import ida_segment
import idaapi
import idautils
import idc


O_VOID = getattr(idc, "o_void", 0)
O_REG = getattr(idc, "o_reg", 1)
O_MEM = getattr(idc, "o_mem", 2)
O_PHRASE = getattr(idc, "o_phrase", 3)
O_DISPL = getattr(idc, "o_displ", 4)
O_IMM = getattr(idc, "o_imm", 5)
O_FAR = getattr(idc, "o_far", 6)
O_NEAR = getattr(idc, "o_near", 7)

STACK_REGS = {"rsp", "esp", "sp", "rbp", "ebp", "bp"}
HEX_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9_])-?0x[0-9A-Fa-f]+(?![A-Za-z0-9_])")
HEX_SUFFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_])-?[0-9][0-9A-Fa-f]*h(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])")
STACK_BASE_RE = re.compile(r"\[\s*(?:[A-Za-z]{2}:\s*)?([A-Za-z0-9]+)", re.IGNORECASE)
DATA_ADDR_HINT_RE = re.compile(r"\b(?:offset|rva|ptr)\b", re.IGNORECASE)

CODE_REFS_CACHE = {}
DATA_REFS_CACHE = {}
SECTIONS = []
MIN_ADDR_LIKE_VALUE = 0x1000


def safe_div(num, den):
    return 0.0 if not den else float(num) / float(den)


def collect_sections():
    sections = []
    for seg_ea in idautils.Segments():
        seg = ida_segment.getseg(seg_ea)
        if seg is None:
            continue
        sections.append(
            {
                "start": seg.start_ea,
                "end": seg.end_ea,
                "name": idc.get_segm_name(seg_ea) or "",
                "is_exec": bool(seg.perm & ida_segment.SEGPERM_EXEC),
            }
        )
    return sections


def section_for_addr(addr):
    for sec in SECTIONS:
        if sec["start"] <= addr < sec["end"]:
            return sec
    return None


def is_address_like_value(addr):
    return addr is not None and addr >= MIN_ADDR_LIKE_VALUE


def has_numeric_literal(text):
    if not text:
        return False
    return bool(
        HEX_PREFIX_RE.search(text)
        or HEX_SUFFIX_RE.search(text)
        or DECIMAL_RE.search(text)
    )


def is_control_flow_mnem(mnem):
    mnem = (mnem or "").lower()
    if not mnem:
        return False
    if mnem.startswith("j"):
        return True
    return mnem in {
        "call",
        "callq",
        "jmp",
        "jmpq",
        "loop",
        "loope",
        "loopne",
        "loopnz",
        "loopz",
        "jecxz",
        "jrcxz",
    }


def get_operand(ea, op_idx):
    insn = idaapi.insn_t()
    if idaapi.decode_insn(insn, ea) == 0:
        return None
    if op_idx >= len(insn.ops):
        return None
    return insn.ops[op_idx]


def get_code_refs(ea):
    refs = CODE_REFS_CACHE.get(ea)
    if refs is None:
        refs = tuple(idautils.CodeRefsFrom(ea, False))
        CODE_REFS_CACHE[ea] = refs
    return refs


def get_data_refs(ea):
    refs = DATA_REFS_CACHE.get(ea)
    if refs is None:
        refs = tuple(idautils.DataRefsFrom(ea))
        DATA_REFS_CACHE[ea] = refs
    return refs


def get_base_reg_name(ea, op_idx, op_text):
    op = get_operand(ea, op_idx)
    if op is not None:
        reg_id = getattr(op, "reg", -1)
        if isinstance(reg_id, int) and reg_id >= 0:
            reg_name = idaapi.get_reg_name(reg_id, 8)
            if reg_name:
                return reg_name.lower()
    match = STACK_BASE_RE.search(op_text or "")
    if match:
        return match.group(1).lower()
    return None


def is_stack_operand(ea, op_idx, op_type, op_text):
    flags = idc.get_full_flags(ea)
    if idaapi.is_stkvar(flags, op_idx):
        return True

    lowered = (op_text or "").lower()
    if "var_" in lowered or "arg_" in lowered:
        return True

    if op_type in {O_DISPL, O_PHRASE}:
        base_reg = get_base_reg_name(ea, op_idx, op_text)
        if base_reg in STACK_REGS:
            return True

    return False


def operand_has_numeric(ea, op_idx, op_type, op_text):
    lowered = (op_text or "").lower()

    if is_stack_operand(ea, op_idx, op_type, op_text):
        return True

    if op_type in {O_MEM, O_NEAR, O_FAR, O_IMM}:
        return True

    if op_type == O_DISPL:
        op = get_operand(ea, op_idx)
        if op is not None and getattr(op, "addr", 0) != 0:
            return True
        return has_numeric_literal(lowered)

    if op_type == O_PHRASE:
        return has_numeric_literal(lowered)

    return has_numeric_literal(lowered)


def get_operand_target_addr(ea, op_idx, op_type, op_text):
    if op_type in {O_MEM, O_NEAR, O_FAR, O_IMM}:
        op_value = idc.get_operand_value(ea, op_idx)
        if op_value not in {None, 0, idaapi.BADADDR}:
            return int(op_value)

    code_refs = get_code_refs(ea)
    if len(code_refs) == 1:
        return int(code_refs[0])

    data_refs = get_data_refs(ea)
    if len(data_refs) == 1:
        return int(data_refs[0])

    lowered = (op_text or "").lower()
    if DATA_ADDR_HINT_RE.search(lowered):
        for ref in data_refs:
            return int(ref)

    return None


def classify_numeric_operand(ea, op_idx, mnem, op_type, op_text):
    if not operand_has_numeric(ea, op_idx, op_type, op_text):
        return None

    lowered = (op_text or "").lower()
    control_flow = is_control_flow_mnem(mnem)
    data_refs = get_data_refs(ea)
    target_addr = get_operand_target_addr(ea, op_idx, op_type, op_text)
    target_sec = section_for_addr(target_addr) if target_addr is not None else None

    if is_stack_operand(ea, op_idx, op_type, op_text):
        return "stack"

    if control_flow:
        if op_type in {O_DISPL, O_PHRASE} and not get_code_refs(ea) and not data_refs:
            return "unknown"
        return "code_address"

    if op_type in {O_MEM, O_NEAR, O_FAR}:
        return "data_address"

    if op_type in {O_DISPL, O_PHRASE} and ("[" in lowered or "]" in lowered):
        if data_refs:
            return "data_address"
        return "unknown"

    if op_type == O_IMM:
        if DATA_ADDR_HINT_RE.search(lowered):
            return "data_address"
        if data_refs:
            return "data_address"
        if target_sec is not None and is_address_like_value(target_addr):
            return "unknown"
        return "imm"

    if any(ch in lowered for ch in "[]:"):
        if data_refs:
            return "data_address"
        return "unknown"

    return "unknown"


def add_example(bucket, category, text, limit=5):
    values = bucket.setdefault(category, [])
    if len(values) < limit and text not in values:
        values.append(text)


def main():
    if len(idc.ARGV) < 2:
        print("[ERROR] Usage: idat -A -S\"ida_collect_numeric_word_stats.py OUTPUT_JSON\" binary")
        idc.qexit(1)

    output_json = idc.ARGV[1]
    ida_auto.auto_wait()
    global SECTIONS, MIN_ADDR_LIKE_VALUE
    SECTIONS = collect_sections()
    non_zero_starts = [sec["start"] for sec in SECTIONS if sec["start"] > 0]
    if non_zero_starts:
        MIN_ADDR_LIKE_VALUE = max(0x1000, min(non_zero_starts))

    counts = {
        "total_instructions": 0,
        "total_words": 0,
        "opcode_words": 0,
        "operand_words": 0,
        "numeric_words": 0,
        "non_numeric_words": 0,
        "code_address": 0,
        "data_address": 0,
        "stack": 0,
        "imm": 0,
        "unknown": 0,
    }
    examples = {}

    for seg_ea in idautils.Segments():
        seg = ida_segment.getseg(seg_ea)
        if seg is None or not (seg.perm & ida_segment.SEGPERM_EXEC):
            continue

        ea = seg.start_ea
        while ea < seg.end_ea:
            flags = ida_bytes.get_flags(ea)
            if ida_bytes.is_code(flags):
                mnem = idc.print_insn_mnem(ea) or ""
                counts["total_instructions"] += 1
                counts["total_words"] += 1
                counts["opcode_words"] += 1
                counts["non_numeric_words"] += 1

                for op_idx in range(8):
                    op_type = idc.get_operand_type(ea, op_idx)
                    if op_type == O_VOID:
                        break

                    op_text = (idc.print_operand(ea, op_idx) or "").strip()
                    if not op_text:
                        continue

                    counts["total_words"] += 1
                    counts["operand_words"] += 1

                    category = classify_numeric_operand(ea, op_idx, mnem, op_type, op_text)
                    if category is None:
                        counts["non_numeric_words"] += 1
                        continue

                    counts["numeric_words"] += 1
                    counts[category] += 1
                    add_example(examples, category, f"{hex(ea)}: {mnem} {op_text}")

                add_example(examples, "opcode", f"{hex(ea)}: {mnem}")

            next_ea = idc.next_head(ea, seg.end_ea)
            if next_ea <= ea:
                break
            ea = next_ea

    numeric_breakdown = (
        counts["code_address"]
        + counts["data_address"]
        + counts["stack"]
        + counts["imm"]
        + counts["unknown"]
    )
    operand_non_numeric_words = counts["operand_words"] - counts["numeric_words"]

    result = {
        "binary_path": idc.get_input_file_path(),
        "counts": counts,
        "derived_counts": {
            "operand_numeric_words": counts["numeric_words"],
            "operand_non_numeric_words": operand_non_numeric_words,
        },
        "ratios": {
            "numeric_over_total_words": safe_div(counts["numeric_words"], counts["total_words"]),
            "non_numeric_over_total_words": safe_div(counts["non_numeric_words"], counts["total_words"]),
            "code_address_over_total_words": safe_div(counts["code_address"], counts["total_words"]),
            "data_address_over_total_words": safe_div(counts["data_address"], counts["total_words"]),
            "stack_over_total_words": safe_div(counts["stack"], counts["total_words"]),
            "imm_over_total_words": safe_div(counts["imm"], counts["total_words"]),
            "unknown_over_total_words": safe_div(counts["unknown"], counts["total_words"]),
            "numeric_breakdown_sum_over_total_words": safe_div(numeric_breakdown, counts["total_words"]),
            "numeric_over_operand_words": safe_div(counts["numeric_words"], counts["operand_words"]),
            "non_numeric_over_operand_words": safe_div(operand_non_numeric_words, counts["operand_words"]),
            "code_address_over_operand_words": safe_div(counts["code_address"], counts["operand_words"]),
            "data_address_over_operand_words": safe_div(counts["data_address"], counts["operand_words"]),
            "stack_over_operand_words": safe_div(counts["stack"], counts["operand_words"]),
            "imm_over_operand_words": safe_div(counts["imm"], counts["operand_words"]),
            "unknown_over_operand_words": safe_div(counts["unknown"], counts["operand_words"]),
        },
        "examples": examples,
    }

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2, sort_keys=True)

    print(f"[INFO] Wrote {output_json}")
    idc.qexit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        idc.qexit(1)
