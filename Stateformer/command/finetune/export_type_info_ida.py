import json
import os
import re

import ida_hexrays
import idautils
import idc


INTEGER_ARG_REGS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
EXPLICIT_PARAM_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)@<(?P<reg>[^>]+)>")


def safe_call(obj, name, default=None):
    try:
        attr = getattr(obj, name)
    except Exception:
        return default
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return default


def normalize_reg_name(name):
    if not name:
        return None
    return str(name).strip().lower()


def format_tinfo(tif):
    try:
        return str(tif)
    except Exception:
        return None


def parse_decl_regs(decl):
    if not decl or "(" not in decl or ")" not in decl:
        return {}
    inside = decl[decl.find("(") + 1: decl.rfind(")")]
    result = {}
    for match in EXPLICIT_PARAM_RE.finditer(inside):
        result[match.group("name")] = normalize_reg_name(match.group("reg"))
    return result


def collect_function_type_info():
    if not ida_hexrays.init_hexrays_plugin():
        raise RuntimeError("Hex-Rays decompiler is unavailable")

    output = []
    for func_ea in idautils.Functions():
        func_name = idc.get_func_name(func_ea)
        if not func_name:
            continue
        decl = idc.get_type(func_ea)
        explicit_regs = parse_decl_regs(decl)

        try:
            cfunc = ida_hexrays.decompile(func_ea)
        except Exception:
            continue
        if cfunc is None:
            continue

        args = []
        locals_ = []
        reg_index = 0
        for lvar in cfunc.get_lvars():
            raw_type = format_tinfo(safe_call(lvar, "type"))
            entry = {
                "name": safe_call(lvar, "name") or "",
                "type": raw_type,
                "is_arg_var": bool(safe_call(lvar, "is_arg_var", False)),
                "is_stk_var": bool(safe_call(lvar, "is_stk_var", False)),
                "is_reg_var": bool(safe_call(lvar, "is_reg_var", False)),
                "stkoff": safe_call(lvar, "get_stkoff", -1),
                "width": safe_call(lvar, "width", 0),
            }
            if entry["is_arg_var"]:
                assigned_reg = explicit_regs.get(entry["name"])
                if assigned_reg is None and entry["is_reg_var"] and reg_index < len(INTEGER_ARG_REGS):
                    assigned_reg = INTEGER_ARG_REGS[reg_index]
                    reg_index += 1
                entry["assigned_reg"] = assigned_reg
                entry["arg_index"] = len(args)
                args.append(entry)
            else:
                locals_.append(entry)

        output.append(
            {
                "function": func_name,
                "start": func_ea,
                "decl": decl,
                "args": args,
                "locals": locals_,
            }
        )
    return output


def main():
    output_dir = idc.ARGV[1]
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(str(idc.get_input_file_path()))
    output_path = os.path.join(output_dir, f"{filename}_typeinfo.json")

    payload = collect_function_type_info()
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(output_path)
    idc.qexit(0)


if __name__ == "__main__":
    main()
