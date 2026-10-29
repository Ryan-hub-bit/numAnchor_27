import os

import ida_auto
import ida_bytes
import ida_idaapi
import ida_lines
import ida_segment
import idautils
import idc


def format_instruction(ea):
    mnem = idc.print_insn_mnem(ea) or ""
    operands = []
    for idx in range(8):
        op = idc.print_operand(ea, idx)
        if not op:
            break
        operands.append(op)
    if operands:
        return f"{mnem} {', '.join(operands)}"
    return mnem


def main():
    out_path = os.environ.get("IDA_DISASM_OUT")
    if not out_path:
        raise RuntimeError("IDA_DISASM_OUT is not set")

    ida_auto.auto_wait()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for seg_ea in idautils.Segments():
            seg = ida_segment.getseg(seg_ea)
            if seg is None or not (seg.perm & ida_segment.SEGPERM_EXEC):
                continue
            ea = seg.start_ea
            while ea < seg.end_ea:
                flags = ida_bytes.get_flags(ea)
                if ida_bytes.is_code(flags):
                    fout.write(f"{ea:#x}: {format_instruction(ea)}\n")
                ea = idc.next_head(ea, seg.end_ea)

    idc.qexit(0)


if __name__ == "__main__":
    main()
