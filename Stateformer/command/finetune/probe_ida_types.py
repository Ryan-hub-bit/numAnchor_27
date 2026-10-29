import json
import os

import ida_funcs
import ida_frame
import ida_hexrays
import ida_typeinf
import idautils
import idc


def safe_call(obj, name, default=None):
    try:
        attr = getattr(obj, name)
    except Exception:
        return default
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return default


def format_tinfo(tif):
    try:
        return ida_typeinf.print_tinfo(
            "",
            0,
            0,
            ida_typeinf.PRTYPE_1LINE,
            tif,
            "",
            "",
        )
    except Exception:
        return None


def probe_frame_members(func_ea):
    output = []
    try:
        func = ida_funcs.get_func(func_ea)
        frame = ida_frame.get_frame(func)
        if frame is None:
            return output
        for member_offset, member_name, member_size in idautils.StructMembers(frame.id):
            entry = {
                "offset": member_offset,
                "name": member_name,
                "size": member_size,
            }
            output.append(entry)
    except Exception as exc:
        output.append({"error": repr(exc)})
    return output


def probe_hexrays_locals(func_ea):
    output = []
    try:
        if not ida_hexrays.init_hexrays_plugin():
            return [{"error": "hexrays unavailable"}]
        cfunc = ida_hexrays.decompile(func_ea)
        for lvar in cfunc.get_lvars():
            tif = safe_call(lvar, "type")
            entry = {
                "name": safe_call(lvar, "name"),
                "type": format_tinfo(tif) if tif is not None else None,
                "is_arg_var": safe_call(lvar, "is_arg_var"),
                "is_stk_var": safe_call(lvar, "is_stk_var"),
                "is_reg_var": safe_call(lvar, "is_reg_var"),
                "stkoff": safe_call(lvar, "get_stkoff"),
                "width": safe_call(lvar, "width"),
                "location": str(safe_call(lvar, "location")),
            }
            output.append(entry)
    except Exception as exc:
        output.append({"error": repr(exc)})
    return output


def main():
    output_dir = idc.ARGV[1]
    os.makedirs(output_dir, exist_ok=True)
    target_name = idc.ARGV[2] if len(idc.ARGV) > 2 else None

    result = []
    for func_ea in idautils.Functions():
        name = idc.get_func_name(func_ea)
        if target_name and name != target_name:
            continue
        result.append(
            {
                "ea": hex(func_ea),
                "name": name,
                "decl": idc.get_type(func_ea),
                "frame_members": probe_frame_members(func_ea),
                "hexrays_locals": probe_hexrays_locals(func_ea),
            }
        )
        if len(result) >= 3:
            break

    output_path = os.path.join(output_dir, "ida_type_probe.json")
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(output_path)
    idc.qexit(0)


if __name__ == "__main__":
    main()
