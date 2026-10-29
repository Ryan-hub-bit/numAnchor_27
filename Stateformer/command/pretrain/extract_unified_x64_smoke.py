#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86 import X86_GRP_CALL, X86_GRP_JUMP, X86_OP_IMM, X86_OP_MEM
from elftools.elf.elffile import ELFFile
from unicorn import Uc, UcError, UC_ARCH_X86, UC_HOOK_CODE, UC_HOOK_MEM_FETCH_UNMAPPED
from unicorn import UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED, UC_MODE_64
from unicorn import UC_MEM_FETCH_UNMAPPED
from unicorn import UC_PROT_EXEC, UC_PROT_READ, UC_PROT_WRITE
from unicorn.x86_const import (
    UC_X86_REG_AH,
    UC_X86_REG_AL,
    UC_X86_REG_BH,
    UC_X86_REG_BL,
    UC_X86_REG_BP,
    UC_X86_REG_BPL,
    UC_X86_REG_BX,
    UC_X86_REG_CH,
    UC_X86_REG_CL,
    UC_X86_REG_CX,
    UC_X86_REG_DH,
    UC_X86_REG_DI,
    UC_X86_REG_DIL,
    UC_X86_REG_DL,
    UC_X86_REG_DX,
    UC_X86_REG_EAX,
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_ECX,
    UC_X86_REG_EDI,
    UC_X86_REG_EDX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
    UC_X86_REG_FS_BASE,
    UC_X86_REG_R10,
    UC_X86_REG_R10B,
    UC_X86_REG_R10D,
    UC_X86_REG_R10W,
    UC_X86_REG_R11,
    UC_X86_REG_R11B,
    UC_X86_REG_R11D,
    UC_X86_REG_R11W,
    UC_X86_REG_R12,
    UC_X86_REG_R12B,
    UC_X86_REG_R12D,
    UC_X86_REG_R12W,
    UC_X86_REG_R13,
    UC_X86_REG_R13B,
    UC_X86_REG_R13D,
    UC_X86_REG_R13W,
    UC_X86_REG_R14,
    UC_X86_REG_R14B,
    UC_X86_REG_R14D,
    UC_X86_REG_R14W,
    UC_X86_REG_R15,
    UC_X86_REG_R15B,
    UC_X86_REG_R15D,
    UC_X86_REG_R15W,
    UC_X86_REG_R8,
    UC_X86_REG_R8B,
    UC_X86_REG_R8D,
    UC_X86_REG_R8W,
    UC_X86_REG_R9,
    UC_X86_REG_R9B,
    UC_X86_REG_R9D,
    UC_X86_REG_R9W,
    UC_X86_REG_RAX,
    UC_X86_REG_RBP,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDI,
    UC_X86_REG_RDX,
    UC_X86_REG_RIP,
    UC_X86_REG_RSI,
    UC_X86_REG_RSP,
    UC_X86_REG_SI,
    UC_X86_REG_SIL,
    UC_X86_REG_SP,
    UC_X86_REG_SPL,
)


PAGE_SIZE = 0x1000
STACK_BASE = 0x700000000000
STACK_SIZE = 0x200000
ARGV_BASE = 0x600000000000
FS_BASE = 0x500000000000
HEAP_BASE = 0x610000000000
NON_ADDR_SENTINEL = -1.0
NON_VAR_SENTINEL = -999999
SPECIAL_DADDR_POS = 2.0
IDA_HEADER_RE = re.compile(
    r"^(?P<opcode>[^\s(]+)\((?P<addr>0x[0-9a-fA-F]+):(?P<bin>[0-9.]+):(?P<func>[0-9.]+):(?P<bb>[0-9.]+)\)"
    r"(?:\s+(?P<ops>.*))?$"
)
IDA_ADDRESS_RE = re.compile(r"address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
IDA_DADDR_RE = re.compile(r"daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
IDA_VAR_RE = re.compile(r"var\((0x[0-9A-Fa-f]+)\)")

ROLE_NONE = "none"
ROLE_CODE = "code"
ROLE_ADDRESS = "address"
ROLE_DADDR = "daddr"
ROLE_VAR = "var"

ROLE_TO_ID = {
    ROLE_NONE: 0,
    ROLE_CODE: 1,
    ROLE_ADDRESS: 2,
    ROLE_DADDR: 3,
    ROLE_VAR: 4,
}

REGISTER_MAP = {
    "al": UC_X86_REG_AL,
    "ah": UC_X86_REG_AH,
    "ax": UC_X86_REG_RAX,
    "eax": UC_X86_REG_EAX,
    "rax": UC_X86_REG_RAX,
    "bl": UC_X86_REG_BL,
    "bh": UC_X86_REG_BH,
    "bx": UC_X86_REG_RBX,
    "ebx": UC_X86_REG_EBX,
    "rbx": UC_X86_REG_RBX,
    "cl": UC_X86_REG_CL,
    "ch": UC_X86_REG_CH,
    "cx": UC_X86_REG_RCX,
    "ecx": UC_X86_REG_ECX,
    "rcx": UC_X86_REG_RCX,
    "dl": UC_X86_REG_DL,
    "dh": UC_X86_REG_DH,
    "dx": UC_X86_REG_RDX,
    "edx": UC_X86_REG_EDX,
    "rdx": UC_X86_REG_RDX,
    "si": UC_X86_REG_SI,
    "sil": UC_X86_REG_SIL,
    "esi": UC_X86_REG_ESI,
    "rsi": UC_X86_REG_RSI,
    "di": UC_X86_REG_DI,
    "dil": UC_X86_REG_DIL,
    "edi": UC_X86_REG_EDI,
    "rdi": UC_X86_REG_RDI,
    "bp": UC_X86_REG_BP,
    "bpl": UC_X86_REG_BPL,
    "ebp": UC_X86_REG_EBP,
    "rbp": UC_X86_REG_RBP,
    "sp": UC_X86_REG_SP,
    "spl": UC_X86_REG_SPL,
    "esp": UC_X86_REG_ESP,
    "rsp": UC_X86_REG_RSP,
    "r8b": UC_X86_REG_R8B,
    "r8w": UC_X86_REG_R8W,
    "r8d": UC_X86_REG_R8D,
    "r8": UC_X86_REG_R8,
    "r9b": UC_X86_REG_R9B,
    "r9w": UC_X86_REG_R9W,
    "r9d": UC_X86_REG_R9D,
    "r9": UC_X86_REG_R9,
    "r10b": UC_X86_REG_R10B,
    "r10w": UC_X86_REG_R10W,
    "r10d": UC_X86_REG_R10D,
    "r10": UC_X86_REG_R10,
    "r11b": UC_X86_REG_R11B,
    "r11w": UC_X86_REG_R11W,
    "r11d": UC_X86_REG_R11D,
    "r11": UC_X86_REG_R11,
    "r12b": UC_X86_REG_R12B,
    "r12w": UC_X86_REG_R12W,
    "r12d": UC_X86_REG_R12D,
    "r12": UC_X86_REG_R12,
    "r13b": UC_X86_REG_R13B,
    "r13w": UC_X86_REG_R13W,
    "r13d": UC_X86_REG_R13D,
    "r13": UC_X86_REG_R13,
    "r14b": UC_X86_REG_R14B,
    "r14w": UC_X86_REG_R14W,
    "r14d": UC_X86_REG_R14D,
    "r14": UC_X86_REG_R14,
    "r15b": UC_X86_REG_R15B,
    "r15w": UC_X86_REG_R15W,
    "r15d": UC_X86_REG_R15D,
    "r15": UC_X86_REG_R15,
    "rip": UC_X86_REG_RIP,
    "eip": UC_X86_REG_EIP,
}


def tokenize(text):
    text = text.replace(",", " , ")
    text = text.replace("[", " [ ")
    text = text.replace("]", " ] ")
    text = text.replace(":", " : ")
    text = text.replace("*", " * ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("{", " { ")
    text = text.replace("}", " } ")
    text = text.replace("#", "")
    text = text.replace("$", "")
    text = text.replace("!", " ! ")
    text = re.sub(r"-(0[xX][0-9a-fA-F]+)", r"- \1", text)
    text = re.sub(r"-([0-9a-fA-F]+)", r"- \1", text)
    return text.split()


def normalize_static_token(token):
    lower = token.lower()
    if "0x" in lower:
        return "hexvar"
    if lower.isdigit():
        return "num"
    return token


def is_numeric_token(token):
    lower = token.lower()
    if lower.startswith("0x"):
        return True
    return lower.isdigit()


def parse_numeric_token(token):
    lower = token.lower()
    if lower.startswith("0x"):
        return int(lower, 16)
    return int(lower)


def value_to_bytes(value):
    value &= 0xFFFFFFFF
    return [f"{(value >> shift) & 0xFF:02x}" for shift in (0, 8, 16, 24)]


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def page_down(value):
    return value & ~(PAGE_SIZE - 1)


def page_up(value):
    return (value + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def section_lookup(sections, address):
    for section in sections:
        if section["start"] <= address < section["end"]:
            return section
    return None


def is_exec_address(sections, address):
    section = section_lookup(sections, address)
    return bool(section and section["is_exec"])


def is_data_address(sections, address):
    section = section_lookup(sections, address)
    return bool(section and not section["is_exec"])


def split_operands(op_str):
    if not op_str.strip():
        return []
    return [operand.strip() for operand in op_str.split(",")]


def locate_operand_spans(tokens, operand_texts):
    spans = []
    cursor = 1
    for operand_text in operand_texts:
        operand_tokens = tokenize(operand_text)
        while cursor < len(tokens) and tokens[cursor] == ",":
            cursor += 1
        matched = False
        for start in range(cursor, len(tokens)):
            end = start + len(operand_tokens)
            if tokens[start:end] == operand_tokens:
                spans.append((start, end))
                cursor = end
                matched = True
                break
        if not matched:
            spans.append((cursor, cursor))
    return spans


def collect_sections(elffile):
    sections = []
    for section in elffile.iter_sections():
        start = int(section["sh_addr"])
        size = int(section["sh_size"])
        if not start or not size:
            continue
        flags = int(section["sh_flags"])
        sections.append(
            {
                "name": section.name,
                "start": start,
                "end": start + size,
                "is_exec": bool(flags & 0x4),
            }
        )
    return sections


def collect_segments(elffile):
    segments = []
    for segment in elffile.iter_segments():
        if segment["p_type"] != "PT_LOAD":
            continue
        segments.append(
            {
                "vaddr": int(segment["p_vaddr"]),
                "memsz": int(segment["p_memsz"]),
                "filesz": int(segment["p_filesz"]),
                "flags": int(segment["p_flags"]),
                "data": segment.data(),
            }
        )
    return segments


def compute_binary_bounds(sections):
    if not sections:
        return 0, 1
    return min(section["start"] for section in sections), max(section["end"] for section in sections)


def parse_signed_hex(raw):
    value = int(raw, 16)
    if value >= (1 << 63):
        value -= (1 << 64)
    return value


def collect_function_symbols(elffile):
    symbols = []
    for section_name in (".symtab", ".dynsym"):
        section = elffile.get_section_by_name(section_name)
        if section is None:
            continue
        for symbol in section.iter_symbols():
            info = symbol["st_info"]
            if info["type"] != "STT_FUNC":
                continue
            if symbol["st_shndx"] == "SHN_UNDEF":
                continue
            start = int(symbol["st_value"])
            if start == 0:
                continue
            symbols.append(
                {
                    "name": symbol.name,
                    "start": start,
                    "size": int(symbol["st_size"]),
                    "section_index": symbol["st_shndx"],
                }
            )
    dedup = {}
    for symbol in symbols:
        key = (symbol["name"], symbol["start"])
        if key not in dedup or dedup[key]["size"] < symbol["size"]:
            dedup[key] = symbol
    ordered = sorted(dedup.values(), key=lambda item: item["start"])
    return ordered


def load_ida_function_metadata(ida_json_path, function_name, function_start):
    if ida_json_path is None:
        return None

    with Path(ida_json_path).open("r") as handle:
        exported = json.load(handle)

    target = None
    for item in exported:
        if item.get("function") != function_name:
            continue
        instructions = [line for line in item.get("funcins", "").split("\t") if line.strip()]
        if not instructions:
            continue
        match = IDA_HEADER_RE.match(instructions[0].strip())
        if not match:
            continue
        if int(match.group("addr"), 16) == function_start:
            target = item
            break

    if target is None:
        raise ValueError(
            f"unable to find IDA metadata for function '{function_name}' at 0x{function_start:x} in {ida_json_path}"
        )

    metadata = {}
    for line in [chunk for chunk in target.get("funcins", "").split("\t") if chunk.strip()]:
        match = IDA_HEADER_RE.match(line.strip())
        if not match:
            continue
        address = int(match.group("addr"), 16)
        operands = match.group("ops") or ""
        entry = {
            "current_pos": {
                "binary_pos": NON_ADDR_SENTINEL,
                "function_pos": float(match.group("func")),
                "bb_pos": float(match.group("bb")),
            },
            "address_targets": {},
            "daddr_targets": {},
            "var_offsets": set(),
            "line": line,
        }
        for nested in IDA_ADDRESS_RE.finditer(operands):
            entry["address_targets"][int(nested.group(1), 16)] = {
                "binary_pos": NON_ADDR_SENTINEL,
                "function_pos": float(nested.group(3)),
                "bb_pos": float(nested.group(4)),
            }
        for nested in IDA_DADDR_RE.finditer(operands):
            entry["daddr_targets"][int(nested.group(1), 16)] = {
                "binary_pos": float(nested.group(2)),
                "function_pos": float(nested.group(3)),
                "bb_pos": float(nested.group(4)),
            }
        for nested in IDA_VAR_RE.finditer(operands):
            entry["var_offsets"].add(parse_signed_hex(nested.group(1)))
        metadata[address] = entry

    return metadata


def resolve_function(symbols, sections, name=None, address=None):
    target = None
    if name is not None:
        for symbol in symbols:
            if symbol["name"] == name:
                target = dict(symbol)
                break
        if target is None:
            raise ValueError(f"function '{name}' not found")
    elif address is not None:
        for symbol in symbols:
            start = symbol["start"]
            end = start + max(symbol["size"], 1)
            if start <= address < end or start == address:
                target = dict(symbol)
                break
        if target is None:
            raise ValueError(f"no function covering address 0x{address:x}")
    else:
        raise ValueError("either function name or address is required")

    if target["size"] == 0:
        section = section_lookup(sections, target["start"])
        if section is None:
            raise ValueError("unable to derive function end without a section")
        next_start = section["end"]
        for symbol in symbols:
            if symbol["start"] > target["start"] and symbol["start"] < next_start:
                next_start = symbol["start"]
        target["size"] = next_start - target["start"]

    return target


def disassemble_function(elffile, function_info):
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    section = section_lookup(collect_sections(elffile), function_info["start"])
    if section is None:
        raise ValueError("function address is not in a mapped section")
    text_section = None
    for candidate in elffile.iter_sections():
        start = int(candidate["sh_addr"])
        end = start + int(candidate["sh_size"])
        if start <= function_info["start"] < end:
            text_section = candidate
            break
    if text_section is None:
        raise ValueError("unable to locate containing section bytes")
    start = function_info["start"]
    end = start + function_info["size"]
    offset = start - int(text_section["sh_addr"])
    code = text_section.data()[offset : offset + function_info["size"]]
    instructions = []
    for insn in md.disasm(code, start):
        if insn.address >= end:
            break
        instructions.append(insn)
    return instructions


def build_basic_blocks(instructions, function_info):
    starts = {function_info["start"]}
    addresses = [insn.address for insn in instructions]
    address_set = set(addresses)
    for index, insn in enumerate(instructions):
        next_addr = instructions[index + 1].address if index + 1 < len(instructions) else None
        if insn.group(X86_GRP_JUMP):
            if insn.operands and insn.operands[0].type == X86_OP_IMM:
                target = int(insn.operands[0].imm)
                if target in address_set:
                    starts.add(target)
            if next_addr is not None and not insn.mnemonic.startswith("jmp"):
                starts.add(next_addr)
    result = {}
    ordered_starts = [insn.address for insn in instructions if insn.address in starts]
    addr_to_index = {insn.address: idx for idx, insn in enumerate(instructions)}
    for bb_index, bb_start in enumerate(ordered_starts):
        start_idx = addr_to_index[bb_start]
        if bb_index + 1 < len(ordered_starts):
            end_idx = addr_to_index[ordered_starts[bb_index + 1]] - 1
        else:
            end_idx = len(instructions) - 1
        bb_end = instructions[end_idx].address
        for local_idx in range(start_idx, end_idx + 1):
            insn = instructions[local_idx]
            result[insn.address] = {
                "bb_index": bb_index,
                "bb_pos": local_idx - start_idx,
                "bb_start": bb_start,
                "bb_end": bb_end,
            }
    return result


def build_instruction_position_map(instructions, function_info, bb_info, binary_min, binary_max):
    func_start = instructions[0].address if instructions else function_info["start"]
    func_end = instructions[-1].address if instructions else function_info["start"]
    func_span = max(1, func_end - func_start)
    binary_span = max(1, binary_max - binary_min)
    func_binary_pos = clamp01((function_info["start"] - binary_min) / float(binary_span))

    positions = {}
    for insn in instructions:
        bb_start = bb_info[insn.address]["bb_start"]
        bb_end = bb_info[insn.address]["bb_end"]
        bb_span = max(1, bb_end - bb_start)
        positions[insn.address] = {
            "binary_pos": func_binary_pos,
            "function_pos": clamp01((bb_start - func_start) / float(func_span)),
            "bb_pos": clamp01((insn.address - bb_start) / float(bb_span)),
        }
    return positions


def build_function_entry_position_map(symbols, binary_min, binary_max):
    binary_span = max(1, binary_max - binary_min)
    mapping = {}
    for symbol in symbols:
        mapping[symbol["start"]] = {
            "binary_pos": clamp01((symbol["start"] - binary_min) / float(binary_span)),
            "function_pos": 0.0,
            "bb_pos": 0.0,
        }
    return mapping


def data_position_for_target(address, sections, binary_min, binary_max):
    section = section_lookup(sections, address)
    if section is None:
        return {"binary_pos": SPECIAL_DADDR_POS, "function_pos": 0.0, "bb_pos": 0.0}

    name = section["name"].lower()
    is_data_like = ".data" in name or ".rodata" in name or ".bss" in name
    if not is_data_like:
        return {"binary_pos": SPECIAL_DADDR_POS, "function_pos": 0.0, "bb_pos": 0.0}

    binary_span = max(1, binary_max - binary_min)
    section_span = max(1, section["end"] - section["start"])
    return {
        "binary_pos": clamp01((section["start"] - binary_min) / float(binary_span)),
        "function_pos": clamp01((address - section["start"]) / float(section_span)),
        "bb_pos": 0.0,
    }


def exec_position_for_target(address, instruction_pos_map, function_entry_pos_map, binary_min, binary_max):
    if address in instruction_pos_map:
        return dict(instruction_pos_map[address])
    if address in function_entry_pos_map:
        return dict(function_entry_pos_map[address])
    binary_span = max(1, binary_max - binary_min)
    return {
        "binary_pos": clamp01((address - binary_min) / float(binary_span)),
        "function_pos": 0.0,
        "bb_pos": 0.0,
    }


def operand_role(insn, operand, sections):
    if operand.type == X86_OP_MEM:
        base = insn.reg_name(operand.mem.base) if operand.mem.base else ""
        disp = int(operand.mem.disp)
        if base in {"rbp", "rsp", "ebp", "esp"}:
            return ROLE_VAR, disp
        if base == "rip":
            target = insn.address + insn.size + disp
            if is_exec_address(sections, target):
                return ROLE_ADDRESS, target
            if is_data_address(sections, target):
                return ROLE_DADDR, target
        return ROLE_CODE, None
    if operand.type == X86_OP_IMM:
        imm = int(operand.imm)
        if insn.group(X86_GRP_JUMP) or insn.group(X86_GRP_CALL):
            if is_exec_address(sections, imm):
                return ROLE_ADDRESS, imm
            if is_data_address(sections, imm):
                return ROLE_DADDR, imm
        return ROLE_CODE, imm
    return ROLE_CODE, None


def resolve_ida_positions_for_role(role, semantic_value, ida_entry):
    if ida_entry is None:
        return None
    if role == ROLE_VAR:
        if semantic_value in ida_entry["var_offsets"]:
            return {
                "binary_pos": NON_ADDR_SENTINEL,
                "function_pos": NON_ADDR_SENTINEL,
                "bb_pos": NON_ADDR_SENTINEL,
                "var_offset": int(semantic_value),
            }
        return None
    if role == ROLE_ADDRESS:
        return ida_entry["address_targets"].get(int(semantic_value))
    if role == ROLE_DADDR:
        return ida_entry["daddr_targets"].get(int(semantic_value))
    return None


def annotate_instruction(
    insn,
    function_info,
    inst_index,
    bb_info,
    arch,
    sections,
    current_pos_map,
    function_entry_pos_map,
    binary_min,
    binary_max,
    ida_metadata=None,
):
    asm = f"{insn.mnemonic} {insn.op_str}".strip()
    raw_tokens = tokenize(asm)
    operand_texts = split_operands(insn.op_str)
    operand_spans = locate_operand_spans(raw_tokens, operand_texts)
    current_pos = current_pos_map[insn.address]
    ida_entry = None if ida_metadata is None else ida_metadata.get(insn.address)
    opcode_positions = {
        "binary_pos": NON_ADDR_SENTINEL,
        "function_pos": current_pos["function_pos"],
        "bb_pos": current_pos["bb_pos"],
    }
    if ida_entry is not None:
        opcode_positions = {
            "binary_pos": NON_ADDR_SENTINEL,
            "function_pos": ida_entry["current_pos"]["function_pos"],
            "bb_pos": ida_entry["current_pos"]["bb_pos"],
        }

    token_entries = []
    for op_pos, token in enumerate(raw_tokens):
        dynamic_source = None
        if token.lower() in REGISTER_MAP:
            dynamic_source = {"kind": "register", "reg": token.lower()}
        elif is_numeric_token(token):
            dynamic_source = {"kind": "value", "value": parse_numeric_token(token)}
        token_entries.append(
            {
                "raw_token": token,
                "static_token": normalize_static_token(token),
                "aa_role": ROLE_CODE,
                "aa_role_id": ROLE_TO_ID[ROLE_CODE],
                "var_offset": NON_VAR_SENTINEL,
                "binary_pos": NON_ADDR_SENTINEL,
                "function_pos": NON_ADDR_SENTINEL,
                "bb_pos": NON_ADDR_SENTINEL,
                "bb_index": bb_info[insn.address]["bb_index"],
                "inst_pos_emb": inst_index,
                "op_pos_emb": op_pos,
                "arch_emb": arch,
                "dynamic_source": dynamic_source,
            }
        )
    if token_entries:
        token_entries[0]["function_pos"] = opcode_positions["function_pos"]
        token_entries[0]["bb_pos"] = opcode_positions["bb_pos"]

    for operand_index, operand in enumerate(insn.operands):
        if operand_index >= len(operand_spans):
            break
        start, end = operand_spans[operand_index]
        role, semantic_value = operand_role(insn, operand, sections)
        if role == ROLE_CODE and semantic_value is None:
            continue
        selected_positions = []
        for token_index in range(start, end):
            if is_numeric_token(raw_tokens[token_index]):
                selected_positions.append(token_index)
        if not selected_positions and role == ROLE_VAR:
            for token_index in range(start, end):
                if raw_tokens[token_index].lower() in {"rbp", "rsp", "ebp", "esp"}:
                    selected_positions.append(token_index)
                    break
        if not selected_positions and role in {ROLE_ADDRESS, ROLE_DADDR}:
            for token_index in range(start, end):
                if raw_tokens[token_index].lower() == "rip":
                    selected_positions.append(token_index)
                    break
        resolved_positions = None
        if ida_entry is not None and semantic_value is not None:
            resolved_positions = resolve_ida_positions_for_role(role, semantic_value, ida_entry)
            if resolved_positions is None:
                continue
        for token_index in selected_positions:
            token_entries[token_index]["aa_role"] = role
            token_entries[token_index]["aa_role_id"] = ROLE_TO_ID[role]
            if role == ROLE_VAR:
                token_entries[token_index]["var_offset"] = (
                    int(resolved_positions["var_offset"]) if resolved_positions is not None else int(semantic_value or 0)
                )
                token_entries[token_index]["binary_pos"] = NON_ADDR_SENTINEL
                token_entries[token_index]["function_pos"] = NON_ADDR_SENTINEL
                token_entries[token_index]["bb_pos"] = NON_ADDR_SENTINEL
            elif role == ROLE_ADDRESS:
                target_pos = resolved_positions
                if target_pos is None:
                    target_pos = exec_position_for_target(
                        int(semantic_value),
                        current_pos_map,
                        function_entry_pos_map,
                        binary_min,
                        binary_max,
                    )
                token_entries[token_index]["binary_pos"] = target_pos["binary_pos"]
                token_entries[token_index]["function_pos"] = target_pos["function_pos"]
                token_entries[token_index]["bb_pos"] = target_pos["bb_pos"]
            elif role == ROLE_DADDR:
                target_pos = resolved_positions
                if target_pos is None:
                    target_pos = data_position_for_target(int(semantic_value), sections, binary_min, binary_max)
                token_entries[token_index]["binary_pos"] = target_pos["binary_pos"]
                token_entries[token_index]["function_pos"] = target_pos["function_pos"]
                token_entries[token_index]["bb_pos"] = target_pos["bb_pos"]
            if semantic_value is not None:
                token_entries[token_index]["dynamic_source"] = {
                    "kind": "value",
                    "value": int(semantic_value),
                }

    return {
        "address": insn.address,
        "size": insn.size,
        "mnemonic": insn.mnemonic,
        "op_str": insn.op_str,
        "bb_index": bb_info[insn.address]["bb_index"],
        "bb_pos": bb_info[insn.address]["bb_pos"],
        "ida_line": None if ida_entry is None else ida_entry["line"],
        "tokens": token_entries,
    }


class FunctionEmulator:
    def __init__(self, segments, instructions, function_info, call_policy, max_instructions):
        self.segments = segments
        self.instructions = {item["address"]: item for item in instructions}
        self.function_info = function_info
        self.call_policy = call_policy
        self.max_instructions = max_instructions
        self.visited = []
        self.snapshots = {}
        self.visit_counts = {}
        self.stop_reason = "completed"
        self.instruction_count = 0
        self.heap_cursor = HEAP_BASE
        self.uc = Uc(UC_ARCH_X86, UC_MODE_64)

    def _map_segments(self):
        for segment in self.segments:
            start = page_down(segment["vaddr"])
            end = page_up(segment["vaddr"] + segment["memsz"])
            perms = 0
            if segment["flags"] & 4:
                perms |= UC_PROT_READ
            if segment["flags"] & 2:
                perms |= UC_PROT_WRITE
            if segment["flags"] & 1:
                perms |= UC_PROT_EXEC
            self.uc.mem_map(start, end - start, perms)
            self.uc.mem_write(segment["vaddr"], segment["data"])
            if segment["memsz"] > segment["filesz"]:
                pad = segment["memsz"] - segment["filesz"]
                self.uc.mem_write(segment["vaddr"] + segment["filesz"], b"\x00" * pad)

    def _map_runtime_regions(self):
        self.uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        stack_top = STACK_BASE + STACK_SIZE - 0x1000
        self.uc.reg_write(UC_X86_REG_RSP, stack_top)
        self.uc.reg_write(UC_X86_REG_RBP, stack_top)

        self.uc.mem_map(ARGV_BASE, 0x2000, UC_PROT_READ | UC_PROT_WRITE)
        arg0 = ARGV_BASE + 0x100
        self.uc.mem_write(arg0, b"stateformer-smoke\x00")
        self.uc.mem_write(ARGV_BASE, (arg0).to_bytes(8, "little") + (0).to_bytes(8, "little"))
        self.uc.reg_write(UC_X86_REG_RDI, 1)
        self.uc.reg_write(UC_X86_REG_RSI, ARGV_BASE)

        self.uc.mem_map(FS_BASE, 0x1000, UC_PROT_READ | UC_PROT_WRITE)
        self.uc.mem_write(FS_BASE + 0x28, (0x123456789ABCDEF0).to_bytes(8, "little"))
        self.uc.reg_write(UC_X86_REG_FS_BASE, FS_BASE)

    def _alloc_stub_pointer(self):
        pointer = self.heap_cursor
        page = page_down(pointer)
        self.uc.mem_map(page, PAGE_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        self.heap_cursor += PAGE_SIZE
        return pointer

    def _snapshot(self):
        snapshot = {}
        for name, register in REGISTER_MAP.items():
            if len(name) >= 2:
                try:
                    snapshot[name] = self.uc.reg_read(register)
                except UcError:
                    continue
        return snapshot

    def _hook_mem_invalid(self, uc, access, address, size, value, _user_data):
        if access == UC_MEM_FETCH_UNMAPPED:
            self.stop_reason = f"unmapped execute at 0x{address:x}"
            return False
        page = page_down(address)
        try:
            uc.mem_map(page, PAGE_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        except UcError:
            pass
        return True

    def _hook_code(self, uc, address, size, _user_data):
        if self.instruction_count >= self.max_instructions:
            self.stop_reason = f"max instruction limit {self.max_instructions}"
            uc.emu_stop()
            return

        if address not in self.instructions:
            self.stop_reason = f"left function at 0x{address:x}"
            uc.emu_stop()
            return

        instruction = self.instructions[address]
        self.visited.append(address)
        self.visit_counts[address] = self.visit_counts.get(address, 0) + 1
        if address not in self.snapshots:
            self.snapshots[address] = self._snapshot()
        self.instruction_count += 1

        if instruction["mnemonic"].startswith("ret"):
            self.stop_reason = "return"
            uc.emu_stop()
            return

        capstone_insn = instruction["capstone"]
        if capstone_insn.group(X86_GRP_CALL):
            if self.call_policy == "stop":
                self.stop_reason = f"stopped before call at 0x{address:x}"
                uc.emu_stop()
                return
            if self.call_policy == "skip":
                pointer = self._alloc_stub_pointer()
                uc.reg_write(UC_X86_REG_RAX, pointer)
                uc.reg_write(UC_X86_REG_RDX, pointer + 0x40)
                uc.reg_write(UC_X86_REG_RIP, address + size)
                return

    def run(self):
        self._map_segments()
        self._map_runtime_regions()
        self.uc.hook_add(
            UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED | UC_HOOK_MEM_FETCH_UNMAPPED,
            self._hook_mem_invalid,
        )
        self.uc.hook_add(UC_HOOK_CODE, self._hook_code)
        start = self.function_info["start"]
        end = self.function_info["start"] + self.function_info["size"]
        try:
            self.uc.emu_start(start, end)
        except UcError as exc:
            self.stop_reason = f"{type(exc).__name__}: {exc}"
        return {
            "visited": self.visited,
            "visit_counts": self.visit_counts,
            "snapshots": self.snapshots,
            "stop_reason": self.stop_reason,
            "instruction_count": self.instruction_count,
        }


def flatten_output(instructions, emu_result):
    executed = set(emu_result["visited"])
    snapshots = emu_result["snapshots"]
    fields = {
        "static": [],
        "inst_pos_emb": [],
        "op_pos_emb": [],
        "arch_emb": [],
        "byte1": [],
        "byte2": [],
        "byte3": [],
        "byte4": [],
        "cover": [],
        "aa_role": [],
        "aa_role_ids": [],
        "binary_pos": [],
        "function_pos": [],
        "bb_pos": [],
        "var_offset": [],
    }

    for instruction in instructions:
        address = instruction["address"]
        executed_now = address in executed
        snapshot = snapshots.get(address, {})
        for token in instruction["tokens"]:
            fields["static"].append(token["static_token"])
            fields["inst_pos_emb"].append(token["inst_pos_emb"])
            fields["op_pos_emb"].append(token["op_pos_emb"])
            fields["arch_emb"].append(token["arch_emb"])
            fields["cover"].append(1 if executed_now else 0)
            fields["aa_role"].append(token["aa_role"])
            fields["aa_role_ids"].append(token["aa_role_id"])
            fields["binary_pos"].append(token["binary_pos"])
            fields["function_pos"].append(token["function_pos"])
            fields["bb_pos"].append(token["bb_pos"])
            fields["var_offset"].append(token["var_offset"])

            if not executed_now or token["dynamic_source"] is None:
                encoded = ["##", "##", "##", "##"]
            elif token["dynamic_source"]["kind"] == "register":
                value = snapshot.get(token["dynamic_source"]["reg"], 0)
                encoded = value_to_bytes(value)
            else:
                value = token["dynamic_source"]["value"]
                encoded = value_to_bytes(value)
            fields["byte1"].append(encoded[0])
            fields["byte2"].append(encoded[1])
            fields["byte3"].append(encoded[2])
            fields["byte4"].append(encoded[3])

    return fields


def build_output(binary_path, function_info, instructions, emu_result, arch, addressaware_source):
    flat_fields = flatten_output(instructions, emu_result)
    instruction_payload = []
    for instruction in instructions:
        instruction_payload.append(
            {
                "address": instruction["address"],
                "size": instruction["size"],
                "mnemonic": instruction["mnemonic"],
                "op_str": instruction["op_str"],
                "bb_index": instruction["bb_index"],
                "bb_pos": instruction["bb_pos"],
                "ida_line": instruction.get("ida_line"),
                "executed": instruction["address"] in set(emu_result["visited"]),
                "visit_count": emu_result["visit_counts"].get(instruction["address"], 0),
                "tokens": [
                    {
                        "raw_token": token["raw_token"],
                        "static_token": token["static_token"],
                        "aa_role": token["aa_role"],
                        "aa_role_id": token["aa_role_id"],
                        "var_offset": token["var_offset"],
                        "binary_pos": token["binary_pos"],
                        "function_pos": token["function_pos"],
                        "bb_pos": token["bb_pos"],
                        "inst_pos_emb": token["inst_pos_emb"],
                        "op_pos_emb": token["op_pos_emb"],
                    }
                    for token in instruction["tokens"]
                ],
            }
        )

    token_count = len(flat_fields["static"])
    return {
        "binary": str(binary_path),
        "function": {
            "name": function_info["name"],
            "start": function_info["start"],
            "size": function_info["size"],
        },
        "arch": arch,
        "addressaware_source": addressaware_source,
        "summary": {
            "instruction_count": len(instructions),
            "executed_instruction_count": len(set(emu_result["visited"])),
            "token_count": token_count,
            "stop_reason": emu_result["stop_reason"],
            "cover_ones": sum(flat_fields["cover"]),
            "var_tokens": sum(1 for role in flat_fields["aa_role"] if role == ROLE_VAR),
            "address_tokens": sum(1 for role in flat_fields["aa_role"] if role == ROLE_ADDRESS),
            "daddr_tokens": sum(1 for role in flat_fields["aa_role"] if role == ROLE_DADDR),
        },
        "flat_fields": flat_fields,
        "instructions": instruction_payload,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a minimal unified x86-64 StateFormer + address-aware sample")
    parser.add_argument("--binary", required=True, help="path to the x86-64 ELF binary")
    parser.add_argument("--function-name", help="function name to extract")
    parser.add_argument("--function-address", help="function address to extract, for example 0x3c6d2")
    parser.add_argument("--ida-functions-json", help="PalmTree/IDA exported functions JSON for address-aware metadata")
    parser.add_argument("--arch", default="x86-64", help="arch token to write into arch_emb")
    parser.add_argument("--call-policy", choices=("stop", "skip"), default="skip")
    parser.add_argument("--max-emu-instructions", type=int, default=256)
    parser.add_argument("--output", required=True, help="output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    binary_path = Path(args.binary).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    function_address = int(args.function_address, 16) if args.function_address else None

    with binary_path.open("rb") as handle:
        elffile = ELFFile(handle)
        sections = collect_sections(elffile)
        binary_min, binary_max = compute_binary_bounds(sections)
        segments = collect_segments(elffile)
        symbols = collect_function_symbols(elffile)
        function_info = resolve_function(symbols, sections, name=args.function_name, address=function_address)
        capstone_instructions = disassemble_function(elffile, function_info)

    bb_info = build_basic_blocks(capstone_instructions, function_info)
    current_pos_map = build_instruction_position_map(capstone_instructions, function_info, bb_info, binary_min, binary_max)
    function_entry_pos_map = build_function_entry_position_map(symbols, binary_min, binary_max)
    ida_metadata = load_ida_function_metadata(
        args.ida_functions_json,
        function_info["name"],
        function_info["start"],
    )
    instructions = []
    for inst_index, insn in enumerate(capstone_instructions):
        annotated = annotate_instruction(
            insn,
            function_info,
            inst_index,
            bb_info,
            args.arch,
            sections,
            current_pos_map,
            function_entry_pos_map,
            binary_min,
            binary_max,
            ida_metadata=ida_metadata,
        )
        annotated["capstone"] = insn
        instructions.append(annotated)

    emulator = FunctionEmulator(
        segments=segments,
        instructions=instructions,
        function_info=function_info,
        call_policy=args.call_policy,
        max_instructions=args.max_emu_instructions,
    )
    emu_result = emulator.run()

    for instruction in instructions:
        instruction.pop("capstone", None)

    addressaware_source = "ida_pro" if ida_metadata is not None else "capstone_heuristic"
    output = build_output(binary_path, function_info, instructions, emu_result, args.arch, addressaware_source)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output["summary"], indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
