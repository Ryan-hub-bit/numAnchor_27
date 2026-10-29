"""
CFG Generation with Hierarchical Position Encoding using IDA Pro

This script should be run from within IDA Pro using -A -S flags.
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
import random
import re
from pathlib import Path

SEG_LEN = 2  # Default value, can be overridden by command-line argument

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


def random_walk(g: nx.DiGraph, length: int, max_sequences: int = 5000):
    """
    Perform random walks on the inter-procedural CFG (OPTIMIZED).
    
    Args:
        g: The global ICFG (includes call/return edges)
        length: Maximum steps per walk
        max_sequences: Maximum number of sequences to generate
    """
    sequences = []
    nodes_with_data = [n for n in g if 'text' in g.nodes[n] and 'mask' in g.nodes[n]]
    
    print(f"[INFO] Starting random walks from {len(nodes_with_data)} valid nodes...")
    
    # Cache successors to avoid repeated lookups
    successors_cache = {node: list(g.successors(node)) for node in nodes_with_data}
    
    for idx, start_node in enumerate(nodes_with_data):
        if len(sequences) >= max_sequences:
            break
        
        if idx % 5000 == 0:
            print(f"[INFO] Random walk progress: {idx}/{len(nodes_with_data)}, {len(sequences)} sequences generated")
            
        s = []
        steps = 0
        s.append((start_node, g.nodes[start_node]['text'], g.nodes[start_node]['mask']))
        cur = start_node
        
        while steps < length:
            # Use cached successors
            if cur not in successors_cache:
                successors_cache[cur] = list(g.successors(cur))
            succ = successors_cache[cur]
            
            if not succ:
                break
            
            cur = random.choice(succ)
            if 'text' in g.nodes[cur] and 'mask' in g.nodes[cur]:
                s.append((cur, g.nodes[cur]['text'], g.nodes[cur]['mask']))
                steps += 1
            else:
                break
        
        if len(s) >= 2:  # Only keep sequences with at least 2 instructions
            sequences.append(s)
    
    return sequences[:max_sequences]


def build_chunk_inline(seq, start_idx: int, k: int, ctx: dict):
    """
    Build a chunk with HIERARCHICAL position encoding (address-based):
    
    For CODE addresses (instructions):
    - Position 1: Function's position in binary = (func_start - min_addr) / (max_addr - min_addr)
    - Position 2: Basic block's position in function = (bb_start - func_start) / (func_end - func_start)
    - Position 3: Instruction's position in basic block = (inst_addr - bb_start) / (bb_end - bb_start)
    
    For DATA addresses (not in text section):
    - Position 1: Section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
    - Position 2: Address position inside section = (addr - section_start) / (section_end - section_start)
    - Position 3: BB position = 0.0 (no BB context for data)
    """
    end_idx = start_idx + k
    if start_idx < 0 or end_idx > len(seq):
        return None
    chunk = seq[start_idx:end_idx]

    addr_positions = ctx.get('addr_positions', {})
    bb_range_map = ctx.get('bb_range_map', {})
    min_addr = ctx.get('min_addr', 0)
    max_addr = ctx.get('max_addr', min_addr)
    sections = ctx.get('sections', [])

    def get_section_for_addr(addr):
        """Find which section an address belongs to."""
        for sec_start, sec_end, sec_name in sections:
            if sec_start <= addr < sec_end:
                return (sec_start, sec_end, sec_name)
        return None
    
    def format_data_address_positions(addr):
        """
        Format hierarchical positions for data/non-code addresses.
        
        Only apply section normalization for data-like sections (.data, .rodata, .bss*):
          Position 1: section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
          Position 2: address position inside section = (addr - section_start) / (section_end - section_start)
          Position 3: 0.0 (no BB context for data)
        
        For other sections or no section at all:
          return sentinel "2.00000000:0.00000000:0.00000000"
        """
        section_info = get_section_for_addr(addr)
        
        # No section at all → external/garbage/special
        if section_info is None:
            return "2.00000000:0.00000000:0.00000000"
        
        sec_start, sec_end, sec_name = section_info
        sname = sec_name.lower()
        
        # Only treat true data-like segments as meaningful
        is_data_like = (
            ".data" in sname or
            ".rodata" in sname or
            ".bss" in sname
        )
        
        if not is_data_like:
            # PLT/GOT/import/debug/etc. → special category
            return "2.00000000:0.00000000:0.00000000"
        
        # ---- Real data section: compute normalized positions ----
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
        
        bb_pos = 0.0
        return f"{sec_in_binary:.8f}:{addr_in_section:.8f}:{bb_pos:.8f}"

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
    for addr, norm_line, mask_line in chunk:
        tokens = norm_line.strip().split()
        masks = mask_line.strip().split()
        if not tokens:
            continue
        opcode = tokens[0]
        operands = tokens[1:]
        operand_masks = masks[1:]
        
        # Check if this is a control flow instruction (call/jmp)
        is_control_flow = opcode.lower() in ['call', 'jmp', 'jmpq', 'callq', 'ja', 'jb', 'jc', 'je', 'jg', 'jl', 'jn', 'jo', 'jp', 'js', 'jz']
        
        # Get symbol_map from context if available
        symbol_map = ctx.get('symbol_map', {})

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
                            is_data_section = (
                                ".data" in sname or
                                ".rodata" in sname or
                                ".bss" in sname or
                                ".got" in sname
                            )
                            
                            # For code sections (PLT/external functions), use symbol name
                            if is_code_section and symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            # For data sections
                            elif is_data_section:
                                # If it's a control flow instruction (call/jmp), don't use daddr()
                                # This handles function pointers in data sections
                                if is_control_flow and symbol_name:
                                    # Function pointer - use symbol name
                                    sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                    formatted_ops.append(sym)
                                elif is_control_flow:
                                    # Function pointer without symbol - use plain address
                                    formatted_ops.append(mk_hex)
                                else:
                                    # Regular data reference - use daddr()
                                    pos = format_data_address_positions(tgt)
                                    formatted_ops.append(f"daddr({mk_hex}:{pos})")
                            # Other sections with symbols (external but not clearly code/data)
                            elif symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            else:
                                # No symbol, use daddr with positions
                                pos = format_data_address_positions(tgt)
                                formatted_ops.append(f"daddr({mk_hex}:{pos})")
                        else:
                            # No section info - use symbol if available
                            if symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            else:
                                # No symbol, no section - use daddr with sentinel positions
                                formatted_ops.append(f"daddr({mk_hex}:2.00000000:0.00000000:0.00000000)")
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
    """Process a binary file using IDA Pro."""
    print(f"[INFO] Processing: {fpath}")
    
    try:
        # Wait for auto-analysis (already done in main, but ensure it's complete)
        ida_auto.auto_wait()
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        binary_name = Path(fpath).name
        out_inline = out_dir / f"{binary_name}_cfg_{SEG_LEN}_inline.txt"
        
        print(f"[INFO] Output file: {out_inline}")

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

    # Build a GLOBAL inter-procedural CFG (ICFG)
    try:
        G = nx.DiGraph()
        
        addr_positions = {}
        bb_range_map = {}    # bb_start -> (bb_start, bb_end)
        func_entry_map = {}  # func_start_addr -> func_ea
        call_sites = []      # List of (call_addr, call_inst, next_addr)
        bin_counter = 0
        
        # For hierarchical positions
        func_info = []  # List of (func_name, func_start, func_end, bbs)
        bb_info = {}    # func_name -> [(bb_start, bb_end), ...]

        print("[INFO] Building inter-procedural CFG with direct call/return edges...")
        
        # Calculate min/max addresses FIRST (from segments)
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
        print(f"[ERROR] Graph initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # First pass: Collect function and BB information (OPTIMIZED)
    print("[INFO] First pass: Collecting function and BB information...")
    all_functions = list(idautils.Functions())
    print(f"[INFO] Found {len(all_functions)} functions")
    
    for idx, func_ea in enumerate(all_functions):
        if idx % 100 == 0:
            print(f"[INFO] Progress: {idx}/{len(all_functions)} functions")
        
        func = ida_funcs.get_func(func_ea)
        if not func:
            continue
        
        # Filter out functions in .plt, extern, .init, .fini segments
        segm_name = idc.get_segm_name(func_ea)
        if segm_name in ['.plt', 'extern', '.init', '.fini']:
            continue
        
        func_start = func.start_ea
        func_end = func.end_ea
        func_name = ida_funcs.get_func_name(func_ea)
        func_entry_map[func_start] = func_ea
        
        # Create flowchart once and reuse
        flowchart = idaapi.FlowChart(func)
        func_bbs = get_basic_blocks_ida(func_ea, flowchart)
        
        for bb_start, bb_end in func_bbs:
            bb_range_map[bb_start] = (bb_start, bb_end)
        
        func_info.append((func_name, func_start, func_end, func_bbs, flowchart))
        bb_info[func_name] = func_bbs
    
    # Second pass: Build nodes and intra-procedural edges (OPTIMIZED)
    print("[INFO] Second pass: Building nodes and edges...")
    
    for idx, (func_name, func_start, func_end, func_bbs, flowchart) in enumerate(func_info):
        if idx % 100 == 0:
            print(f"[INFO] Progress: {idx}/{len(func_info)} functions, {bin_counter} instructions")
        
        # Build BB successor map for faster edge lookup
        bb_successors = {}
        for block in flowchart:
            bb_successors[block.start_ea] = [succ.start_ea for succ in block.succs()]
        
        for block in flowchart:
            bb_start = block.start_ea
            bb_end = block.end_ea
            
            curr = bb_start
            predecessor = None
            bb_instructions = []
            
            # Collect all instructions in BB first
            while curr < bb_end:
                bb_instructions.append(curr)
                curr = idc.next_head(curr, bb_end)
                if curr == idaapi.BADADDR or curr >= bb_end:
                    break
            
            # Process instructions
            for inst_addr in bb_instructions:
                # Get clean disassembly
                disasm_raw = clean_ida_disasm(inst_addr)
                if not disasm_raw:
                    continue
                
                # Normalize and mask the disassembly
                norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                G.add_node(inst_addr, text=norm_text, mask=mask_line)
                
                # Store position info
                addr_positions[inst_addr] = (bin_counter, inst_addr, func_name, bb_start, func_start, func_end)
                bin_counter += 1
                
                # Sequential edge within BB
                if predecessor is not None:
                    G.add_edge(predecessor, inst_addr)
                
                # Check if this is a DIRECT call instruction
                mnem = idc.print_insn_mnem(inst_addr)
                if mnem and mnem.lower() == 'call':
                    hex_match = re.search(r'0x[0-9a-fA-F]+', disasm_raw)
                    if hex_match:
                        inst_len = idc.get_item_size(inst_addr)
                        next_addr = inst_addr + inst_len
                        call_sites.append((inst_addr, disasm_raw, next_addr))
                
                predecessor = inst_addr
            
            # Add edges to successor BBs (using cached successor map)
            if predecessor is not None and bb_start in bb_successors:
                for succ_bb_start in bb_successors[bb_start]:
                    if succ_bb_start != idaapi.BADADDR:
                        G.add_edge(predecessor, succ_bb_start)

    total_bin = bin_counter
    
    # Third pass: Add inter-procedural DIRECT call and return edges (OPTIMIZED)
    print(f"[INFO] Third pass: Processing {len(call_sites)} call sites...")
    
    # Pre-compute all return instructions for each function
    func_returns = {}  # func_start -> [return_addresses]
    for func_name, func_start, func_end, func_bbs, flowchart in func_info:
        returns = []
        for block in flowchart:
            # Find the last instruction in this block
            curr = block.start_ea
            last_inst_addr = None
            while curr < block.end_ea:
                last_inst_addr = curr
                curr = idc.next_head(curr, block.end_ea)
                if curr == idaapi.BADADDR or curr >= block.end_ea:
                    break
            
            if last_inst_addr is not None:
                mnem = idc.print_insn_mnem(last_inst_addr)
                if mnem and mnem.lower().startswith('ret'):
                    returns.append(last_inst_addr)
        
        func_returns[func_start] = returns
    
    print(f"[INFO] Found return instructions in {len(func_returns)} functions")
    
    call_edges_added = 0
    return_edges_added = 0
    
    for idx, (call_addr, call_disasm, next_addr) in enumerate(call_sites):
        if idx % 1000 == 0:
            print(f"[INFO] Call sites progress: {idx}/{len(call_sites)}")
        
        try:
            # Extract hex address from call instruction
            hex_match = re.search(r'0x[0-9a-fA-F]+', call_disasm)
            if not hex_match:
                continue
            
            try:
                call_target = int(hex_match.group(), 16)
            except ValueError:
                continue
            
            if call_target in func_entry_map:
                # Add call edge
                G.add_edge(call_addr, call_target)
                call_edges_added += 1
                
                # Add return edges using cached return instructions
                if call_target in func_returns and next_addr in addr_positions:
                    for ret_addr in func_returns[call_target]:
                        G.add_edge(ret_addr, next_addr)
                        return_edges_added += 1
        
        except Exception:
            continue
    
    print(f"[INFO] Added {call_edges_added} DIRECT call edges and {return_edges_added} return edges")
    print(f"[INFO] Global ICFG has {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Collect sections
    sections = []
    for n in range(ida_segment.get_segm_qty()):
        seg = ida_segment.getnseg(n)
        if seg:
            start = seg.start_ea
            end = seg.end_ea
            name = ida_segment.get_segm_name(seg)
            sections.append((start, end, name))
    
    # Calculate min/max to cover ALL addresses
    all_addrs = list(addr_positions.keys()) if addr_positions else []
    for sec_start, sec_end, _ in sections:
        all_addrs.append(sec_start)
        all_addrs.append(sec_end)
    
    if all_addrs:
        min_addr = min(all_addrs)
        max_addr = max(all_addrs)
    else:
        min_addr = 0
        max_addr = 0

    ctx = {
        'addr_positions': addr_positions,
        'total_bin': total_bin,
        'bb_range_map': bb_range_map,
        'min_addr': min_addr,
        'max_addr': max_addr,
        'sections': sections,
        'symbol_map': symbol_map,
        'string_map': string_map,
    }

    # Perform random walks on the GLOBAL ICFG
    print("[INFO] Performing random walks on ICFG...")
    walks = random_walk(G, length=40, max_sequences=1000)
    print(f"[INFO] Generated {len(walks)} random walk sequences")
    
    written = 0
    with open(out_inline, 'w', encoding='utf-8') as w:
        for s in walks:
            if len(s) < SEG_LEN:
                continue
            for start in range(0, len(s) - SEG_LEN + 1):
                packed = build_chunk_inline(s, start, SEG_LEN, ctx)
                if not packed:
                    continue
                w.write(packed + "\n")
                written += 1

    print(f"[DONE] {out_inline} (wrote {written} sequences)")
    print(f"[INFO] Position encoding:")
    print(f"  CODE: func_in_binary:bb_in_function:inst_in_bb")
    print(f"  DATA: section_in_binary:addr_in_section:0.0")


# This script should be run through IDA Pro's batch mode
# When IDA loads, it will execute this script automatically

def main():
    """Main entry point when run from IDA Pro."""
    # Determine output directory priority:
    # (1) command-line argument (sys.argv[1])
    # (2) IDA_OUTPUT_DIR env var
    # (3) OUTPUT_DIR env var
    # (4) default hardcoded path
    default_out = '/path/to/numAnchor/PalmTree/test/'
    out_dir = None

    # Try command-line arg
    try:
        if len(sys.argv) > 1:
            out_dir = sys.argv[1]
    except Exception:
        out_dir = None

    # Fallback to env vars
    if not out_dir:
        out_dir = os.getenv('IDA_OUTPUT_DIR', os.getenv('OUTPUT_DIR', default_out))

    # Setup logging to file
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, 'ida_processing.log')
    
    # Redirect prints to log file
    import sys
    log_f = open(log_file, 'a')
    sys.stdout = log_f
    sys.stderr = log_f
    
    print("\n" + "="*70)
    print("[INFO] Script started in IDA Pro")
    print("="*70)
    
    # Wait for IDA's auto-analysis to complete
    print("[INFO] Waiting for IDA auto-analysis to complete...")
    ida_auto.auto_wait()
    print("[INFO] Auto-analysis complete")
    
    # Get the input file path
    input_file = idc.get_input_file_path()
    print(f"[INFO] Processing: {input_file}")
    
    # Get SEG_LEN from environment or use default
    global SEG_LEN
    SEG_LEN = int(os.getenv('SEG_LEN', '2'))
    
    print(f"[CONFIG] SEG_LEN={SEG_LEN}, OUTPUT={out_dir}")
    print(f"[INFO] Using HIERARCHICAL position encoding (address-based):")
    print(f"  CODE addresses:")
    print(f"    - Position 1: (func_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (bb_start - func_start) / (func_end - func_start)")
    print(f"    - Position 3: (inst_addr - bb_start) / (bb_end - bb_start)")
    print(f"  DATA addresses:")
    print(f"    - Position 1: (section_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (addr - section_start) / (section_end - section_start)")
    print(f"    - Position 3: 0.0 (no BB context)")
    print(f"[INFO] Including DIRECT call and return edges in ICFG")
    
    # Process the file
    try:
        # Fallback to environment variables if no CLI arg provided
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
