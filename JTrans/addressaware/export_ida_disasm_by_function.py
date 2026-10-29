import os
import re

import ida_auto
import ida_bytes
import ida_funcs
import ida_ua
import ida_segment
import idautils
import idc


CODE_DATA_OP_TYPES = {
    getattr(idc, "o_mem", 2),
    getattr(idc, "o_far", 6),
    getattr(idc, "o_near", 7),
}

STACK_BASE_RE = re.compile(r"\[([a-zA-Z0-9]+)")


def to_signed(value, bits=64):
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def format_stack_displacement(base_reg, disp):
    if disp == 0:
        return f"[{base_reg}]"
    if disp > 0:
        return f"[{base_reg}+0x{disp:x}]"
    return f"[{base_reg}-0x{abs(disp):x}]"


def normalize_operand(ea, op_idx):
    op_text = idc.print_operand(ea, op_idx) or ""
    op_type = idc.get_operand_type(ea, op_idx)
    op_value = idc.get_operand_value(ea, op_idx)
    insn = idautils.DecodeInstruction(ea)
    op = insn.ops[op_idx] if insn is not None else None

    if op_type in CODE_DATA_OP_TYPES:
        return f"{op_value:#x}"

    if op_type == getattr(idc, "o_imm", 5) and op_text.startswith("offset "):
        return f"{op_value:#x}"

    if op is not None and op.type == ida_ua.o_displ:
        m = STACK_BASE_RE.search(op_text)
        if m:
            base_reg = m.group(1).lower()
            if base_reg in {"rbp", "rsp", "ebp", "esp"}:
                disp = to_signed(op.addr, 64)
                return format_stack_displacement(base_reg, disp)

    if op_text.startswith("cs:"):
        return op_text[3:]

    return op_text


def format_instruction(ea):
    mnem = idc.print_insn_mnem(ea) or ""
    operands = []
    for idx in range(8):
        op = idc.print_operand(ea, idx)
        if not op:
            break
        operands.append(normalize_operand(ea, idx))
    if operands:
        return f"{mnem} {', '.join(operands)}"
    return mnem


def is_exec_function(func_ea):
    seg = ida_segment.getseg(func_ea)
    return seg is not None and (seg.perm & ida_segment.SEGPERM_EXEC)


def main():
    out_path = os.environ.get("IDA_DISASM_OUT")
    if not out_path:
        raise RuntimeError("IDA_DISASM_OUT is not set")

    ida_auto.auto_wait()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        first = True
        for func_ea in idautils.Functions():
            if not is_exec_function(func_ea):
                continue
            func = ida_funcs.get_func(func_ea)
            if func is None:
                continue

            if not first:
                fout.write("\n")
            first = False

            fout.write("=" * 80 + "\n")
            fout.write(f"Function @ {func_ea:#x}\n")
            fout.write("=" * 80 + "\n")

            for ea in idautils.FuncItems(func.start_ea):
                flags = ida_bytes.get_flags(ea)
                if ida_bytes.is_code(flags):
                    fout.write(f"{ea:#x}: {format_instruction(ea)}\n")

    idc.qexit(0)


if __name__ == "__main__":
    main()
