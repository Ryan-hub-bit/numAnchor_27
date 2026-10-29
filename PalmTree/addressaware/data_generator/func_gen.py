"""
CFG Function Export with Hierarchical Position Encoding using IDA Pro

This script exports ENTIRE FUNCTIONS (no sequence chunking) with hierarchical address-aware encoding.
It should be run from within IDA Pro using -A -S flags.
IDA Pro installation: /path/to/ida-pro-9.0
"""

# Fix Python path to include conda environment's site-packages
import sys
import os

# Add conda environment's site-packages to path
conda_env_path = '/path/to/conda/envs/palmtree/lib/python3.11/site-packages'
if os.path.exists(conda_env_path) and conda_env_path not in sys.path:
    sys.path.insert(0, conda_env_path)

import idautils
import idaapi
import idc
import ida_auto
import ida_segment
import ida_funcs
import ida_bytes
import ida_ua
import ida_name
import ida_nalt

import networkx as nx
import re
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from section_utils import (
    classify_data_section,
    default_daddr_operand,
    format_daddr_operand,
)

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def normalize_and_mask(ins_raw: str, symbol_map: dict, string_map: dict):
    ins = re.sub(r"\s+", ", ", ins_raw, 1)
    parts = ins.split(", ")
    opcode = parts[0]
    operands = parts[1:] if len(parts) > 1 else []

    out_tokens = [opcode]
    mask_tokens = ["0"]

    for op in operands:
        pieces = re.split(r"(0x[0-9A-Fa-f]+|[A-Za-z0-9_]+|\[|\]|,|:|\(|\))", op)
        for tok in pieces:
            if not tok or tok.isspace():  # Skip empty and whitespace-only tokens
                continue
            if tok.startswith("0x") and bool(HEX_RE.fullmatch(tok)):
                out_tokens.append(tok)
                mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    return " ".join(out_tokens), " ".join(mask_tokens)


def build_function_inline(func_instructions, ctx: dict):
    """
    Build entire function with HIERARCHICAL position encoding (address-based):
    
    For CODE addresses (instructions):
    - Position 1: Function's position in binary = (func_start - min_addr) / (max_addr - min_addr)
    - Position 2: Basic block's position in function = (bb_start - func_start) / (func_end - func_start)
    - Position 3: Instruction's position in basic block = (inst_addr - bb_start) / (bb_end - bb_start)
    
    For DATA addresses:
    - Position 1: Section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
    - Position 2: Address position inside section = (addr - section_start) / (section_end - section_start)
    - Position 3: Section kind ID (1=.rodata, 2=.bss, 3=.data, 4=other-data)
    
    For EXTERNAL/PLT calls:
    - Use symbol name instead of address (e.g., .plt.printf, strcmp, etc.)
    """
    addr_positions = ctx.get('addr_positions', {})
    bb_range_map = ctx.get('bb_range_map', {})
    min_addr = ctx.get('min_addr', 0)
    max_addr = ctx.get('max_addr', min_addr)
    sections = ctx.get('sections', [])
    symbol_map = ctx.get('symbol_map', {})

    def get_section_for_addr(addr):
        """Find which section an address belongs to."""
        for sec_start, sec_end, sec_name in sections:
            if sec_start <= addr < sec_end:
                return (sec_start, sec_end, sec_name)
        return None
    
    def format_data_address_positions(addr):
        """
        Format data references as section-specific tokens:
          daddr_rodata(0xADDR:section_pos:offset_in_section)
        """
        section_info = get_section_for_addr(addr)
        if section_info is None:
            return default_daddr_operand(hex(addr))

        sec_start, sec_end, sec_name = section_info
        is_data_like, section_kind = classify_data_section(sec_name)
        if not is_data_like:
            return default_daddr_operand(hex(addr))

        if max_addr > min_addr:
            sec_in_binary = (sec_start - min_addr) / float(max_addr - min_addr)
            sec_in_binary = max(0.0, min(1.0, sec_in_binary))
        else:
            sec_in_binary = 0.0
        
        if sec_end > sec_start:
            addr_in_section = (addr - sec_start) / float(sec_end - sec_start)
            addr_in_section = max(0.0, min(1.0, addr_in_section))
        else:
            addr_in_section = 0.0

        return format_daddr_operand(hex(addr), sec_in_binary, addr_in_section, section_kind)

    def format_hierarchical_positions(entry):
        """
        Returns: func_in_binary:bb_in_function:inst_in_bb
        
        Position 1: (func_start - min_addr) / (max_addr - min_addr)
        Position 2: (bb_start - func_start) / (func_end - func_start)
        Position 3: (inst_addr - bb_start) / (bb_end - bb_start)
        """
        bin_counter, addr, func_name, bb_start, func_start, func_end = entry
        
        # Position 1: Function's position in binary (address-based)
        if max_addr > min_addr:
            func_binary_norm = (func_start - min_addr) / float(max_addr - min_addr)
            func_binary_norm = max(0.0, min(1.0, func_binary_norm))
        else:
            func_binary_norm = 0.0
        
        # Position 2: Basic block's position in function (address-based)
        # Handle case where bb_start is None (instruction not matched to a BB)
        if bb_start is not None and func_end > func_start:
            bb_function_norm = (bb_start - func_start) / float(func_end - func_start)
            bb_function_norm = max(0.0, min(1.0, bb_function_norm))
        else:
            bb_function_norm = 0.0
        
        # Position 3: Instruction's position in basic block (address-based)
        if bb_start is not None and bb_start in bb_range_map:
            bb_start_addr, bb_end_addr = bb_range_map[bb_start]
            if bb_end_addr > bb_start_addr:
                inst_bb_norm = (addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                inst_bb_norm = max(0.0, min(1.0, inst_bb_norm))
            else:
                inst_bb_norm = 0.0
        else:
            inst_bb_norm = 0.0
        
        return f"{func_binary_norm:.8f}:{bb_function_norm:.8f}:{inst_bb_norm:.8f}"

    out_instrs = []
    for addr, norm_line, mask_line in func_instructions:
        tokens = norm_line.strip().split()
        masks = mask_line.strip().split()
        if not tokens:
            continue
        opcode = tokens[0]
        operands = tokens[1:]
        operand_masks = masks[1:]
        
        # Check if this is a control flow instruction (call/jmp)
        is_control_flow = opcode.lower() in ['call', 'jmp', 'jmpq', 'callq', 'ja', 'jb', 'jc', 'je', 'jg', 'jl', 'jn', 'jo', 'jp', 'js', 'jz']

        formatted_ops = []
        for i, tok in enumerate(operands):
            mk = operand_masks[i] if i < len(operand_masks) else "0"
            mk_hex = None
            
            # Check mask first (from normalize_and_mask)
            if isinstance(mk, str) and mk.startswith('0x'):
                mk_hex = mk
            # Check token for 0x prefix
            elif isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
                mk_hex = tok
            # Check token for Intel hex suffix (e.g., 1234ABCDh)
            elif isinstance(tok, str) and re.match(r'^[0-9A-Fa-f]+h$', tok, re.IGNORECASE):
                # Convert Intel hex to 0x format
                mk_hex = '0x' + tok[:-1]

            if mk_hex is not None:
                try:
                    tgt = int(mk_hex, 16)
                except Exception:
                    tgt = None

                # Filter out immediate values
                addr_threshold = max(min_addr, 0x1000)
                if tgt is not None and tgt >= addr_threshold:
                    if tgt in addr_positions:
                        # Internal function call: use hierarchical positions (func, bb, inst)
                        entry = addr_positions[tgt]
                        pos = format_hierarchical_positions(entry)
                        formatted_ops.append(f"address({mk_hex}:{pos})")
                    else:
                        # External address - check section type
                        section_info = get_section_for_addr(tgt)
                        
                        # Check if there's a symbol name for this address
                        symbol_name = None
                        if tgt in symbol_map:
                            symbol_name = symbol_map[tgt]
                        
                        if section_info:
                            sec_start, sec_end, sec_name = section_info
                            sname = sec_name.lower()
                            
                            # Check if it's a code section (PLT/text) - external function
                            is_code_section = (
                                ".plt" in sname or
                                "plt" == sname or
                                ".text" in sname or
                                "text" == sname or
                                ".extern" in sname
                            )
                            
                            # Check if it's a data section
                            is_data_section, _ = classify_data_section(sec_name)
                            
                            # For code sections (PLT/external functions), use symbol name
                            if is_code_section and symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            # For data sections
                            elif is_data_section:
                                # If it's a control flow instruction (call/jmp), don't emit a data token.
                                # This handles function pointers in data sections
                                if is_control_flow and symbol_name:
                                    # Function pointer - use symbol name
                                    sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                    formatted_ops.append(sym)
                                elif is_control_flow:
                                    # Function pointer without symbol - use plain address
                                    formatted_ops.append(mk_hex)
                                else:
                                    # Regular data reference - use section-specific daddr token
                                    formatted_ops.append(format_data_address_positions(tgt))
                            # Other sections with symbols (external but not clearly code/data)
                            elif symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            else:
                                # No symbol, use section-specific daddr token
                                formatted_ops.append(format_data_address_positions(tgt))
                        else:
                            # No section info - use symbol if available
                            if symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            else:
                                # No symbol, no section - use fallback data token
                                formatted_ops.append(default_daddr_operand(mk_hex))
                else:
                    # It's an immediate value or stack offset
                    formatted_ops.append("imm")
            else:
                if tok.startswith("var_"):
                    var_offset = tok[4:]  # Get the part after 'var_'
                    # Remove trailing 'h' if present (Intel hex format)
                    var_offset = var_offset.rstrip('hH')
                    # Remove leading 's' if present (IDA signed offset prefix)
                    var_offset = var_offset.lstrip('sS')
                    formatted_ops.append(f"var(0x{var_offset})")
                elif tok.startswith("arg_"):
                    # IDA auto-generated argument names - treat as stack variables
                    arg_offset = tok[4:]  # Get the part after 'arg_'
                    # Remove leading 's' if present
                    arg_offset = arg_offset.lstrip('sS')
                    formatted_ops.append(f"var(0x{arg_offset})")
                elif re.match(r'^0x[0-9a-fA-F]+$', tok):
                    # Raw hex address that slipped through - should not happen, filter it out
                    formatted_ops.append("imm")
                else:
                        formatted_ops.append(tok)

        if addr in addr_positions:
            addr_hdr = f"{hex(addr)}:{format_hierarchical_positions(addr_positions[addr])}"
        else:
            addr_hdr = hex(addr)

        ops_join = ' '.join(formatted_ops)
        if ops_join.count('[') > ops_join.count(']'):
            ops_join = ops_join + ' ]'

        if ops_join.strip():
            out_instrs.append(f"{opcode}({addr_hdr}) {ops_join}")
        else:
            out_instrs.append(f"{opcode}({addr_hdr})")

    return "\t".join(out_instrs)


def clean_ida_disasm(ea):
    """
    Get clean raw assembly text WITHOUT IDA's symbol inference.
    Build operands from numeric values only - no struct fields, no variable names, no debug symbols.
    
    Simple output:
    - o_imm: -> "imm"
    - o_reg: -> register name (rax, rbx, etc.)
    - o_displ: -> [base_reg+var_XXh] or [base_reg+disp]
    - o_phrase: -> [base_reg] or [base_reg+disp]
    - o_mem/o_near/o_far: -> hex address
    """
    # Get mnemonic
    mnem = idc.print_insn_mnem(ea)
    if not mnem:
        return None
    
    # Decode instruction
    insn = idaapi.insn_t()
    idaapi.decode_insn(insn, ea)
    
    # Get operands
    operands = []
    for i in range(6):
        op_type = idc.get_operand_type(ea, i)
        if op_type == idc.o_void:
            break
            
        op_value = idc.get_operand_value(ea, i)
        op = None
        
        if op_type == idc.o_reg:
            # Register - get name
            reg_id = op_value
            reg_name = idaapi.get_reg_name(reg_id, 8)
            if not reg_name:
                reg_name = idc.print_operand(ea, i)
            op = reg_name
            
        elif op_type == idc.o_imm:
            # Immediate
            op = "imm"
            
        elif op_type == idc.o_displ:
            # Displacement: [base+offset]
            if i < len(insn.ops):
                operand = insn.ops[i]
                base_reg_id = operand.reg
                base_reg_name = idaapi.get_reg_name(base_reg_id, 8)
                offset = operand.addr
                
                # rbp (frame pointer) always accesses stack variables
                if base_reg_name in ['rbp', 'ebp']:
                    # Stack variable - format as var_XXh
                    # IDA uses unsigned offsets for both local vars and parameters
                    op = f"[{base_reg_name}+var_{offset:X}h]"
                else:
                    # For other registers (including rsp), check if it's actually a stack variable
                    flags = idc.get_full_flags(ea)
                    is_stack_var = idaapi.is_stkvar(flags, i)
                    
                    if is_stack_var:
                        # IDA confirmed this is a stack variable
                        op = f"[{base_reg_name}+var_{offset:X}h]"
                    else:
                        # Not a stack variable - generic displacement
                        op = f"[{base_reg_name}+disp]"
            else:
                op = "[unknown+disp]"
                
        elif op_type == idc.o_phrase:
            # Memory phrase: [base] or [base+index*scale]
            if i < len(insn.ops):
                operand = insn.ops[i]
                base_reg_id = operand.reg
                base_reg_name = idaapi.get_reg_name(base_reg_id, 8)
                
                # Check for index register (SIB)
                if hasattr(operand, 'specflag1') and operand.specflag1:
                    op = f"[{base_reg_name}+disp]"
                else:
                    op = f"[{base_reg_name}]"
            else:
                op = "[unknown]"
                
        elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
            # Address reference
            if op_value != idaapi.BADADDR and op_value != 0:
                op = hex(op_value)
            else:
                # Invalid address - use IDA's original output as fallback
                op = idc.print_operand(ea, i)
                
        else:
            # Fallback
            op_str = idc.print_operand(ea, i)
            op = op_str if op_str else "unknown"
        
        if op:
            operands.append(op)
    
    # Build result
    result = f"{mnem} {', '.join(operands)}" if operands else mnem
    return result


def get_basic_blocks_ida(func_ea, flowchart=None):
    """
    Get basic blocks for a function in IDA Pro.
    Returns list of (bb_start, bb_end) tuples.
    If flowchart is provided, reuse it instead of creating a new one.
    """
    func = ida_funcs.get_func(func_ea)
    if not func:
        return []
    
    if flowchart is None:
        flowchart = idaapi.FlowChart(func)
    
    blocks = [(block.start_ea, block.end_ea) for block in flowchart]
    return blocks


def process_file_ida(fpath: str, out_dir: str):
    """Process a binary file using IDA Pro and export entire functions."""
    print(f"[INFO] Processing: {fpath}")
    
    try:
        # Wait for auto-analysis
        ida_auto.auto_wait()
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        binary_name = Path(fpath).name
        out_file = out_dir / f"{binary_name}_functions.txt"
        
        print(f"[INFO] Output file: {out_file}")

        # Build symbol and string maps
        symbol_map = {}
        for ea_tuple in idautils.Names():
            addr, name = ea_tuple
            symbol_map[addr] = name
        print(f"[INFO] Found {len(symbol_map)} symbols")
        
        string_map = {}
        for s in idautils.Strings():
            string_map[s.ea] = str(s)
        print(f"[INFO] Found {len(string_map)} strings")
    
    except Exception as e:
        print(f"[ERROR] Initial setup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        addr_positions = {}
        bb_range_map = {}    # bb_start -> (bb_start, bb_end)
        bin_counter = 0
        
        # Calculate min/max addresses from segments
        print("[INFO] Calculating binary address range from segments...")
        sections = []
        for n in range(ida_segment.get_segm_qty()):
            seg = ida_segment.getnseg(n)
            if seg:
                start = seg.start_ea
                end = seg.end_ea
                name = ida_segment.get_segm_name(seg)
                sections.append((start, end, name))
                print(f"[INFO]   Segment: {name} [{hex(start)} - {hex(end)}]")
        
        # Get min/max from all segments
        if sections:
            min_addr = min(s[0] for s in sections)
            max_addr = max(s[1] for s in sections)
        else:
            min_addr = 0
            max_addr = 0
        
        print(f"[INFO] Binary address range: {hex(min_addr)} - {hex(max_addr)}")
        
    except Exception as e:
        print(f"[ERROR] Address range calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Collect all functions
    print("[INFO] Collecting function information...")
    all_functions = list(idautils.Functions())
    print(f"[INFO] Found {len(all_functions)} functions")
    
    func_data = []  # List of (func_name, func_start, func_end, instructions)
    skipped_segments = 0  # Counter for skipped segments
    
    for idx, func_ea in enumerate(all_functions):
        if idx % 100 == 0:
            print(f"[INFO] Progress: {idx}/{len(all_functions)} functions")
        
        func = ida_funcs.get_func(func_ea)
        if not func:
            continue
        
        # Filter out functions in .plt, extern, .init, .fini segments (same as baseline)
        segm_name = idc.get_segm_name(func_ea)
        if segm_name in ['.plt', 'extern', '.init', '.fini']:
            skipped_segments += 1
            continue
        
        func_start = func.start_ea
        func_end = func.end_ea
        func_name = ida_funcs.get_func_name(func_ea)
        
        # Create flowchart once and reuse
        flowchart = idaapi.FlowChart(func)
        func_bbs = get_basic_blocks_ida(func_ea, flowchart)
        
        for bb_start, bb_end in func_bbs:
            bb_range_map[bb_start] = (bb_start, bb_end)
        
        # Collect all instructions in the function
        func_instructions = []
        
        for block in flowchart:
            bb_start = block.start_ea
            bb_end = block.end_ea
            
            curr = bb_start
            while curr < bb_end:
                # Get clean disassembly
                disasm_raw = clean_ida_disasm(curr)
                if disasm_raw:
                    # Normalize and mask the disassembly
                    norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                    
                    # Store instruction data
                    func_instructions.append((curr, norm_text, mask_line))
                    
                    # Store position info
                    addr_positions[curr] = (bin_counter, curr, func_name, bb_start, func_start, func_end)
                    bin_counter += 1
                
                curr = idc.next_head(curr, bb_end)
                if curr == idaapi.BADADDR or curr >= bb_end:
                    break
        
        # Store function data if it has instructions
        if func_instructions:
            func_data.append((func_name, func_start, func_end, func_instructions))
    
    print(f"[INFO] Collected {len(func_data)} functions with {bin_counter} total instructions")
    
    # Build context for hierarchical encoding
    ctx = {
        'addr_positions': addr_positions,
        'total_bin': bin_counter,
        'bb_range_map': bb_range_map,
        'min_addr': min_addr,
        'max_addr': max_addr,
        'sections': sections,
        'symbol_map': symbol_map,
    }
    
    # Write entire functions to output file
    print("[INFO] Writing functions to output file...")
    written = 0
    skipped_empty = 0
    skipped_small = 0
    MIN_INSTRUCTIONS = 5
    
    with open(out_file, 'w', encoding='utf-8') as w:
        for func_name, func_start, func_end, func_instructions in func_data:
            # Skip empty functions
            if not func_instructions:
                skipped_empty += 1
                continue
            
            # Skip functions with fewer than MIN_INSTRUCTIONS (same as baseline)
            if len(func_instructions) < MIN_INSTRUCTIONS:
                skipped_small += 1
                continue
            
            # Build the entire function as a single line
            packed = build_function_inline(func_instructions, ctx)
            if packed:
                w.write(packed + "\n")
                written += 1
    
    print(f"[DONE] {out_file} (wrote {written} functions)")
    print(f"[INFO] Skipped: {skipped_empty} empty, {skipped_small} < {MIN_INSTRUCTIONS} instructions, {skipped_segments} in filtered segments (.plt, .init, .fini, extern)")
    print(f"[INFO] Position encoding:")
    print(f"  CODE: func_in_binary:bb_in_function:inst_in_bb")
    print(f"  DATA: section_in_binary:addr_in_section:0.0")


def main():
    """Main entry point when run from IDA Pro."""
    # Get output directory from environment or use default
    # Use IDA_OUTPUT_DIR to avoid conflicts with other scripts
    out_dir = os.getenv('IDA_OUTPUT_DIR', os.getenv('OUTPUT_DIR', '/path/to/numAnchor/JTrans/datautils_addraware/output'))
    
    # Setup logging to file
    log_file = os.path.join(out_dir, 'ida_processing.log')
    os.makedirs(out_dir, exist_ok=True)
    
    # Redirect prints to log file
    import sys
    log_f = open(log_file, 'a')
    sys.stdout = log_f
    sys.stderr = log_f
    
    print("\n" + "="*70)
    print("[INFO] Function Export Script started in IDA Pro")
    print("="*70)
    
    # Wait for IDA's auto-analysis to complete
    print("[INFO] Waiting for IDA auto-analysis to complete...")
    ida_auto.auto_wait()
    print("[INFO] Auto-analysis complete")
    
    # Get the input file path
    input_file = idc.get_input_file_path()
    print(f"[INFO] Processing: {input_file}")
    
    print(f"[CONFIG] OUTPUT={out_dir}")
    print(f"[INFO] Exporting ENTIRE FUNCTIONS (no sequence chunking)")
    print(f"[INFO] Using HIERARCHICAL position encoding (address-based):")
    print(f"  CODE addresses:")
    print(f"    - Position 1: (func_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (bb_start - func_start) / (func_end - func_start)")
    print(f"    - Position 3: (inst_addr - bb_start) / (bb_end - bb_start)")
    print(f"  DATA addresses:")
    print(f"    - Position 1: (section_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (addr - section_start) / (section_end - section_start)")
    print(f"    - Position 3: 0.0 (no BB context)")
    
    # Process the file
    try:
        process_file_ida(input_file, out_dir)
        print("[SUCCESS] Processing complete")
        log_f.close()
        idc.qexit(0)  # Exit IDA successfully
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        log_f.close()
        idc.qexit(1)  # Exit IDA with error


# Run when script is executed by IDA
if __name__ == '__main__':
    main()
