import argparse
import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from paths import ensure_parent_under_data_root, ensure_under_data_root


FILE_RE = re.compile(r"^(?P<binary>.+)-(?P<opt>O0|O1|O2|O3|Os)-(?P<hash>[0-9a-f]+)_extract\.pkl$")
HEX_SUFFIX_RE = re.compile(r"^(?:locret_|loc_|sub_)([0-9A-Fa-f]+)$")
VAR_RE = re.compile(r"\bvar_(?:s)?([0-9A-Fa-f]+)h?\b")
ARG_RE = re.compile(r"\barg_(?:s)?([0-9A-Fa-f]+)h?\b")
IMM_RE = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9A-Fa-f]+h|\d+)\b")
MEM_LABEL_RE = re.compile(r"^(?:unk_|off_|byte_|word_|dword_|qword_|xmmword_|ymmword_|zmmword_)[0-9A-Fa-f]+$")
LIB_OPERAND_RE = re.compile(r"^[._a-z][a-z0-9_$.@?]+$")
REGISTER_RE = re.compile(
    r"^(?:"
    r"(?:r(?:[0-9]|1[0-5])(?:[dwb])?)|"
    r"(?:[re]?(?:ax|bx|cx|dx|si|di|sp|bp|ip))|"
    r"(?:[abcd][lh])|(?:sil|dil|bpl|spl)|"
    r"(?:xmm(?:[0-9]|1[0-5]))|(?:ymm(?:[0-9]|1[0-5]))|(?:zmm(?:[0-9]|1[0-5]))|"
    r"(?:mm[0-7])|(?:k[0-7])|(?:cr[0-9]+)|(?:dr[0-9]+)|"
    r"(?:st(?:\([0-7]\))?)|(?:[cdefgs]s)|(?:rip|eip)"
    r")$"
)
KEEP_OPERAND_TOKENS = {
    "byte", "word", "dword", "qword", "xmmword", "ymmword", "zmmword",
    "ptr", "near", "far", "short",
    "cs", "ds", "es", "fs", "gs", "ss",
}
PLT_TOKEN_PREFIX = ".plt."
PLT_UNK_TOKEN = ".plt"
INTERNAL_FUNCTION_TOKEN = "<function>"


def fmt_pos(value: float) -> str:
    return f"{value:.8f}"


def to_u32_hex(value: int) -> str:
    if value < 0:
        value = (1 << 32) + value
    return hex(value & 0xFFFFFFFF)


def split_opcode_operands(asm: str) -> Tuple[str, str]:
    asm = asm.split(";", 1)[0].strip()
    if not asm:
        return "nop", ""
    if " " not in asm:
        return asm.lower(), ""
    opcode, operands = asm.split(None, 1)
    return opcode.lower(), operands.strip()


def parse_target_addr(token: str) -> Optional[int]:
    token = token.strip().rstrip(",")
    token = re.sub(r"\b(?:short|near|far|ptr|offset)\b", " ", token)
    token = token.replace(":", " ")
    for piece in reversed([part for part in re.split(r"[\s\[\]\(\)\+\-\*\,]+", token) if part]):
        match = HEX_SUFFIX_RE.match(piece)
        if match:
            return int(match.group(1), 16)
        if piece.startswith("0x"):
            try:
                return int(piece, 16)
            except ValueError:
                continue
    return None


def normalize_plain_token(token: str) -> str:
    if not token:
        return token
    if token == "imm":
        return token
    if token in KEEP_OPERAND_TOKENS:
        return token
    if REGISTER_RE.match(token):
        return token
    if MEM_LABEL_RE.match(token):
        return "hexvar"
    if HEX_SUFFIX_RE.match(token):
        return "hexvar"
    if token.startswith((".", "_")):
        return "hexvar"
    if "_" in token:
        return "hexvar"
    if len(token) > 24:
        return "hexvar"
    if re.match(r"^[a-z][a-z0-9@$?.]*$", token):
        return "hexvar"
    return token


def extract_library_symbol(operand: str) -> Optional[str]:
    lowered = operand.lower().strip().rstrip(",")
    lowered = re.sub(r"\b(?:short|near|far|ptr|offset)\b", " ", lowered)
    pieces = [part for part in re.split(r"[\s\[\]\(\)\+\-\*\,:]+", lowered) if part]
    candidates = []
    for piece in pieces:
        if piece in KEEP_OPERAND_TOKENS or piece == "imm":
            continue
        if REGISTER_RE.match(piece):
            continue
        if parse_target_addr(piece) is not None:
            continue
        if stack_var_token(piece) is not None:
            continue
        if MEM_LABEL_RE.match(piece):
            continue
        if LIB_OPERAND_RE.match(piece):
            candidates.append(piece)

    if not candidates:
        return None

    name = candidates[-1]
    for suffix in ("@plt.sec", "@plt", "@got.plt", "@got", "_ptr", "__ptr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("sym.imp."):
        name = name[len("sym.imp.") :]
    name = name.lstrip(".")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if not name:
        return None
    return f"{PLT_TOKEN_PREFIX}{name}"


def tokenize_plain_operand(operand: str) -> List[str]:
    operand = operand.strip()
    if not operand:
        return []
    operand = operand.lower()
    operand = operand.replace("ptr ", "")
    operand = operand.replace("offset ", "")
    operand = operand.replace("%", "")
    operand = operand.replace(":", " ")
    operand = operand.replace("*", " ")
    operand = IMM_RE.sub("imm", operand)
    chunks = re.split(r"[\s,\[\]\(\)\+\-]+", operand)
    normalized = [normalize_plain_token(chunk) for chunk in chunks if chunk]
    return normalized or ["imm"]


def stack_var_token(operand: str) -> Optional[str]:
    operand = operand.lower()
    var_match = VAR_RE.search(operand)
    if var_match:
        raw = int(var_match.group(1), 16)
        if "var_s" in operand:
            raw = -raw
        return f"var({to_u32_hex(raw)})"
    arg_match = ARG_RE.search(operand)
    if arg_match:
        return f"var({to_u32_hex(int(arg_match.group(1), 16))})"
    return None


def is_probable_data_ref(operand: str) -> bool:
    lowered = operand.lower().strip()
    prefixes = ("unk_", "off_", "byte_", "word_", "dword_", "qword_", "xmmword_", "ymmword_", "zmmword_")
    if lowered.startswith(prefixes) or " offset " in f" {lowered} ":
        return True
    if any(piece.startswith(prefixes) for piece in re.split(r"[\s,\[\]\(\)\+\-\*]+", lowered) if piece):
        return True
    return False


def resolve_special_operand(
    operand: str,
    binary_min_addr: int,
    binary_span: int,
    target_pos_map: Dict[int, Dict[str, float]],
    file_function_entries: Dict[int, str],
) -> Optional[str]:
    target_addr = parse_target_addr(operand)
    if target_addr is not None:
        if target_addr in target_pos_map:
            tp = target_pos_map[target_addr]
            return (
                f"address({hex(target_addr)}:{fmt_pos(tp['binary_pos'])}:"
                f"{fmt_pos(tp['function_pos'])}:{fmt_pos(tp['bb_pos'])})"
            )
        if target_addr in file_function_entries:
            target_binary_pos = (target_addr - binary_min_addr) / float(binary_span)
            return f"address({hex(target_addr)}:{fmt_pos(target_binary_pos)}:0.00000000:0.00000000)"
        return "daddr(0x0:2.00000000:0.00000000:0.00000000)"

    tokenized = tokenize_plain_operand(operand)
    if any(tok == "hexvar" for tok in tokenized):
        return "hexvar"
    return None


def make_daddr_token(target_addr: Optional[int], binary_min_addr: int, binary_span: int) -> str:
    if target_addr is None:
        return "daddr(0x0:2.00000000:0.00000000:0.00000000)"
    binary_pos = (target_addr - binary_min_addr) / float(binary_span)
    return f"daddr({hex(target_addr)}:{fmt_pos(binary_pos)}:0.00000000:0.00000000)"


def resolve_control_flow_operand(
    operand: str,
    opcode: str,
    binary_min_addr: int,
    binary_span: int,
    target_pos_map: Dict[int, Dict[str, float]],
    file_function_entries: Dict[int, str],
) -> Optional[str]:
    lib_tok = extract_library_symbol(operand)
    if lib_tok is not None:
        return lib_tok

    target_addr = parse_target_addr(operand)
    if target_addr is None:
        return None

    # Keep address(...) exclusively for intra-function control-flow targets.
    if target_addr in target_pos_map:
        tp = target_pos_map[target_addr]
        return (
            f"address({hex(target_addr)}:{fmt_pos(tp['binary_pos'])}:"
            f"{fmt_pos(tp['function_pos'])}:{fmt_pos(tp['bb_pos'])})"
        )

    # For address-aware modeling, direct control-flow to another internal
    # function should still preserve target address information instead of
    # collapsing to a surface-only <function> token.
    if target_addr in file_function_entries:
        target_binary_pos = (target_addr - binary_min_addr) / float(binary_span)
        return f"address({hex(target_addr)}:{fmt_pos(target_binary_pos)}:0.00000000:0.00000000)"

    # Unresolved direct control-flow targets behave more like calls to
    # unknown code than data references, so keep them in the function bucket.
    if opcode == "call" or opcode.startswith("j"):
        return INTERNAL_FUNCTION_TOKEN

    return None


def resolve_non_control_flow_operand(
    operand: str,
    binary_min_addr: int,
    binary_span: int,
) -> Optional[str]:
    target_addr = parse_target_addr(operand)
    if target_addr is not None:
        return make_daddr_token(target_addr, binary_min_addr, binary_span)

    tokenized = tokenize_plain_operand(operand)
    if any(tok == "hexvar" for tok in tokenized):
        return make_daddr_token(None, binary_min_addr, binary_span)
    return None


def replace_rare_library_tokens(
    function_blocks: Dict[str, List[str]],
    top_k: int,
) -> Tuple[Counter, set]:
    lib_counter = Counter()
    for lines in function_blocks.values():
        for line in lines:
            for tok in line.split():
                if tok.startswith(PLT_TOKEN_PREFIX):
                    lib_counter[tok] += 1

    keep = {tok for tok, _freq in lib_counter.most_common(top_k)}
    if not lib_counter:
        return lib_counter, keep

    for func_id, lines in function_blocks.items():
        rewritten = []
        for line in lines:
            toks = line.split()
            toks = [tok if not tok.startswith(PLT_TOKEN_PREFIX) or tok in keep else PLT_UNK_TOKEN for tok in toks]
            rewritten.append(" ".join(toks))
        function_blocks[func_id] = rewritten

    return lib_counter, keep


def iter_project_dirs(root: str) -> Iterable[str]:
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            yield full


def load_project_pickles(project_dir: str):
    per_opt = {}
    for entry in sorted(os.listdir(project_dir)):
        match = FILE_RE.match(entry)
        if not match:
            continue
        opt = match.group("opt")
        path = os.path.join(project_dir, entry)
        with open(path, "rb") as f:
            per_opt[opt] = {
                "binary": match.group("binary"),
                "path": path,
                "functions": pickle.load(f),
            }
    return per_opt


def annotate_function(
    project: str,
    opt: str,
    binary_name: str,
    func_name: str,
    func_record,
    binary_min_addr: int,
    binary_max_addr: int,
    file_function_entries: Dict[int, str],
):
    func_addr, asm_list, _rawbytes, cfg, _bai = func_record
    if cfg is None or cfg.number_of_nodes() == 0:
        return None

    block_starts = sorted(cfg.nodes())
    block_index = {start: idx for idx, start in enumerate(block_starts)}
    block_count = max(1, len(block_starts) - 1)
    func_span = max(1, (block_starts[-1] - block_starts[0]) if len(block_starts) > 1 else 1)
    binary_span = max(1, binary_max_addr - binary_min_addr)
    binary_pos = (func_addr - binary_min_addr) / float(binary_span)

    target_pos_map = {}
    for start in block_starts:
        f_pos = (start - block_starts[0]) / float(func_span)
        target_pos_map[start] = {
            "binary_pos": binary_pos,
            "function_pos": f_pos,
            "bb_pos": 0.0,
        }

    annotated = []
    for block_start in block_starts:
        asm_lines = list(cfg.nodes[block_start].get("asm", []))
        if not asm_lines:
            continue
        f_pos = (block_start - block_starts[0]) / float(func_span)
        instr_denom = max(1, len(asm_lines) - 1)
        for inst_idx, asm in enumerate(asm_lines):
            opcode, operands = split_opcode_operands(asm)
            pseudo_addr = block_start + inst_idx
            bb_pos = inst_idx / float(instr_denom)
            parts = [f"{opcode}({hex(pseudo_addr)}:{fmt_pos(binary_pos)}:{fmt_pos(f_pos)}:{fmt_pos(bb_pos)})"]

            raw_operands = [piece.strip() for piece in operands.split(",") if piece.strip()]
            used_special = False
            if raw_operands and (opcode.startswith("j") or opcode.startswith("call")):
                special = resolve_control_flow_operand(
                    raw_operands[-1],
                    opcode=opcode,
                    binary_min_addr=binary_min_addr,
                    binary_span=binary_span,
                    target_pos_map=target_pos_map,
                    file_function_entries=file_function_entries,
                )
                if special is not None:
                    parts.append(special)
                    used_special = True

            if not used_special:
                for operand in raw_operands:
                    var_tok = stack_var_token(operand)
                    if var_tok is not None:
                        parts.append(var_tok)
                        continue
                    special = resolve_non_control_flow_operand(
                        operand,
                        binary_min_addr=binary_min_addr,
                        binary_span=binary_span,
                    )
                    if special is not None:
                        parts.append(special)
                        continue
                    if is_probable_data_ref(operand):
                        parts.append(make_daddr_token(None, binary_min_addr, binary_span))
                        continue
                    parts.extend(tokenize_plain_operand(operand))

            annotated.append(" ".join(parts))

    if not annotated:
        return None

    func_id = f"{binary_name}::{opt}::{func_name}"
    group_key = f"{project}::{func_name}"
    return func_id, group_key, annotated


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(description="Build address-aware corpora from official jTrans extracted pickles")
    parser.add_argument("--extract_root", required=True, help="Directory such as .../small_train or .../small_test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--write_corpus", action="store_true", help="Write pretrain_corpus.txt")
    parser.add_argument("--max_projects", type=int, default=None)
    parser.add_argument("--max_functions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.extract_root = ensure_under_data_root(args.extract_root, "--extract_root", must_exist=True)
    args.output_dir = ensure_parent_under_data_root(args.output_dir, "--output_dir")

    rng = random.Random(args.seed)
    project_dirs = list(iter_project_dirs(args.extract_root))
    project_dirs.sort()
    if args.max_projects is not None:
        project_dirs = project_dirs[: args.max_projects]

    function_blocks = {}
    group_to_ids = defaultdict(list)
    opt_counter = Counter()
    project_counter = 0
    function_counter = 0

    corpus_path = os.path.join(args.output_dir, "pretrain_corpus.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    corpus_fp = open(corpus_path, "w", encoding="utf-8") if args.write_corpus else None

    try:
        for project_dir in project_dirs:
            project = os.path.basename(project_dir)
            per_opt = load_project_pickles(project_dir)
            if not per_opt:
                continue
            project_counter += 1

            for opt, payload in per_opt.items():
                functions = payload["functions"]
                file_function_entries = {
                    func_data[0]: func_name
                    for func_name, func_data in functions.items()
                    if isinstance(func_data[0], int)
                }
                if not file_function_entries:
                    continue
                binary_min_addr = min(file_function_entries)
                binary_max_addr = max(file_function_entries)

                for func_name, func_record in functions.items():
                    annotated = annotate_function(
                        project=project,
                        opt=opt,
                        binary_name=payload["binary"],
                        func_name=func_name,
                        func_record=func_record,
                        binary_min_addr=binary_min_addr,
                        binary_max_addr=binary_max_addr,
                        file_function_entries=file_function_entries,
                    )
                    if annotated is None:
                        continue
                    func_id, group_key, lines = annotated
                    function_blocks[func_id] = lines
                    group_to_ids[group_key].append(func_id)
                    opt_counter[opt] += 1
                    function_counter += 1
                    if corpus_fp is not None:
                        corpus_fp.write("\t".join(lines) + "\n")
                    if args.max_functions is not None and function_counter >= args.max_functions:
                        break
                if args.max_functions is not None and function_counter >= args.max_functions:
                    break
            if args.max_functions is not None and function_counter >= args.max_functions:
                break
    finally:
        if corpus_fp is not None:
            corpus_fp.close()

    funcsim_pairs = {}
    for ids in group_to_ids.values():
        if len(ids) < 2:
            continue
        for func_id in ids:
            gt = [other for other in ids if other != func_id]
            if gt:
                funcsim_pairs[func_id] = {"ground_truth": gt}

    lib_counter, kept_libs = replace_rare_library_tokens(function_blocks, top_k=30)

    metadata = {
        "extract_root": args.extract_root,
        "projects": project_counter,
        "functions": function_counter,
        "query_functions": len(funcsim_pairs),
        "opt_counts": dict(opt_counter),
        "library_tokens_total": len(lib_counter),
        "library_tokens_kept": len(kept_libs),
        "library_top30": lib_counter.most_common(30),
        "library_replacement_token": PLT_UNK_TOKEN,
    }

    save_json(function_blocks, os.path.join(args.output_dir, "funcsim_function_blocks.json"))
    save_json(funcsim_pairs, os.path.join(args.output_dir, "funcsim_pairs.json"))
    save_json(metadata, os.path.join(args.output_dir, "metadata.json"))

    if corpus_fp is not None:
        with open(corpus_path, "w", encoding="utf-8") as f:
            for lines in function_blocks.values():
                f.write("\t".join(lines) + "\n")

    print(json.dumps(metadata, indent=2))
    if corpus_fp is not None:
        print(f"Corpus written to {corpus_path}")


if __name__ == "__main__":
    main()
