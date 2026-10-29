#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


INTEGER_REG_FAMILIES = {
    "rdi": {"rdi", "edi", "di", "dil"},
    "rsi": {"rsi", "esi", "si", "sil"},
    "rdx": {"rdx", "edx", "dx", "dl", "dh"},
    "rcx": {"rcx", "ecx", "cx", "cl", "ch"},
    "r8": {"r8", "r8d", "r8w", "r8b"},
    "r9": {"r9", "r9d", "r9w", "r9b"},
}
XMM_REG_FAMILIES = {f"xmm{i}": {f"xmm{i}"} for i in range(8)}
REGISTER_CANDIDATES = set().union(*INTEGER_REG_FAMILIES.values(), *XMM_REG_FAMILIES.values())
NON_VAR_SENTINEL = -999999
NO_ACCESS = "no-access"
PAD_LABEL_ID = -100


def normalize_type(raw_type):
    if raw_type is None:
        return None
    text = str(raw_type).strip()
    if not text:
        return None

    pointer_count = text.count("*")
    lowered = text.lower()
    lowered = lowered.replace("const ", " ").replace("volatile ", " ")
    lowered = lowered.replace("__cdecl", " ").replace("__fastcall", " ")
    lowered = lowered.replace("__usercall", " ").replace("__noreturn", " ")
    lowered = lowered.replace("__ptr64", " ").replace("__restrict", " ")
    lowered = re.sub(r"\s+", " ", lowered.replace("*", " ")).strip()

    if "[" in lowered and "]" in lowered:
        base = "array"
    elif "enum" in lowered:
        base = "enum"
    elif "union" in lowered:
        base = "union"
    elif "struct" in lowered:
        base = "struct"
    elif any(token in lowered for token in ["void"]):
        base = "void"
    elif "long double" in lowered:
        base = "long_double"
    elif "double" in lowered:
        base = "double"
    elif "float" in lowered:
        base = "float"
    elif any(token in lowered for token in ["bool", "_bool"]):
        base = "unsigned_char"
    elif "_unknown" in lowered or "unknown" in lowered:
        base = "undefined"
    elif any(token in lowered for token in ["_byte", "uint8_t", "__int8"]):
        base = "unsigned_char" if "unsigned" in lowered or "_byte" in lowered else "signed_char"
    elif any(token in lowered for token in ["char", "uchar"]):
        base = "unsigned_char" if "unsigned" in lowered or "uchar" in lowered else "signed_char"
    elif any(token in lowered for token in ["_word", "uint16_t", "__int16"]):
        base = "unsigned_short" if "unsigned" in lowered or "_word" in lowered else "signed_short"
    elif "short" in lowered:
        base = "unsigned_short" if "unsigned" in lowered else "signed_short"
    elif any(token in lowered for token in ["_dword", "uint32_t", "__int32"]):
        base = "unsigned_int" if "unsigned" in lowered or "_dword" in lowered else "signed_int"
    elif "intmax_t" in lowered:
        base = "signed_long_long"
    elif "uintmax_t" in lowered:
        base = "unsigned_long_long"
    elif any(token in lowered for token in ["size_t", "uintptr_t"]):
        base = "unsigned_long"
    elif any(token in lowered for token in ["ssize_t", "intptr_t", "ptrdiff_t", "off_t"]):
        base = "signed_long"
    elif any(token in lowered for token in ["_qword", "uint64_t", "__int64"]):
        base = "unsigned_long_long" if "unsigned" in lowered or "_qword" in lowered else "signed_long_long"
    elif "long long" in lowered:
        base = "unsigned_long_long" if "unsigned" in lowered else "signed_long_long"
    elif "long" in lowered:
        base = "unsigned_long" if "unsigned" in lowered else "signed_long"
    elif "int" in lowered:
        base = "unsigned_int" if "unsigned" in lowered else "signed_int"
    else:
        token = lowered.replace(" ", "")
        if token and re.fullmatch(r"[a-z_][a-z0-9_]*", token):
            base = "struct"
        else:
            base = "undefined"

    if pointer_count > 0 and base not in {"array", "undefined"}:
        return f"{base}*"
    return base


def load_type_info(type_json_path):
    with Path(type_json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    mapping = {}
    for item in payload:
        arg_reg_types = {}
        for arg in item.get("args", []):
            normalized = normalize_type(arg.get("type"))
            assigned_reg = (arg.get("assigned_reg") or "").lower()
            if not normalized or not assigned_reg:
                continue
            family = INTEGER_REG_FAMILIES.get(assigned_reg) or XMM_REG_FAMILIES.get(assigned_reg)
            if family is None:
                family = {assigned_reg}
            for token in family:
                arg_reg_types[token] = normalized
        mapping[(item["function"], int(item["start"]))] = {
            "decl": item.get("decl"),
            "arg_reg_types": arg_reg_types,
            "raw_args": item.get("args", []),
        }
    return mapping


def sanitize_path_name(path):
    return "_".join(Path(path).parts[-3:]).replace(".", "_")


def run_ida_type_export(binary_path, output_dir, idat_path, ida_script):
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(output_dir.glob("*_typeinfo.json"))
    if candidates:
        return candidates[-1]
    command = [
        str(idat_path),
        "-A",
        f'-S"{ida_script} {output_dir}"',
        str(binary_path),
    ]
    subprocess.run(" ".join(command), shell=True, check=True)
    candidates = sorted(output_dir.glob("*_typeinfo.json"))
    if not candidates:
        raise FileNotFoundError(f"IDA type export did not produce a typeinfo json for {binary_path}")
    return candidates[-1]


def collect_instruction_token_labels(instruction, offset_to_type, arg_reg_types, supervision_mode):
    raw_tokens = [token["raw_token"].lower() for token in instruction["tokens"]]
    var_offsets = [
        int(token["var_offset"])
        for token in instruction["tokens"]
        if int(token["var_offset"]) != NON_VAR_SENTINEL
    ]

    reg_types = []
    for raw in raw_tokens:
        if raw in arg_reg_types:
            reg_types.append(arg_reg_types[raw])
    unique_reg_types = []
    for label in reg_types:
        if label not in unique_reg_types:
            unique_reg_types.append(label)

    if instruction["mnemonic"].lower() == "mov" and var_offsets and len(unique_reg_types) == 1:
        mapped_label = unique_reg_types[0]
        for offset in var_offsets:
            offset_to_type[offset] = mapped_label
    elif len(unique_reg_types) == 1:
        mapped_label = unique_reg_types[0]
    else:
        mapped_label = None

    labels = []
    positive_count = 0
    for token in instruction["tokens"]:
        raw = token["raw_token"].lower()
        var_offset = int(token["var_offset"])

        if supervision_mode == "dense":
            if var_offset in offset_to_type:
                label = offset_to_type[var_offset]
            elif mapped_label is not None:
                label = mapped_label
            else:
                label = NO_ACCESS
        else:
            label = None
            if var_offset != NON_VAR_SENTINEL:
                label = offset_to_type.get(var_offset, NO_ACCESS)
            elif raw in REGISTER_CANDIDATES:
                label = arg_reg_types.get(raw, NO_ACCESS)

        labels.append(label)
        if label not in {None, NO_ACCESS}:
            positive_count += 1

    return labels, positive_count


def build_label_sequence(unified_payload, function_type_info, supervision_mode):
    arg_reg_types = function_type_info.get("arg_reg_types", {})
    labels = []
    offset_to_type = {}
    positive_count = 0
    supervised_count = 0

    for instruction in unified_payload["instructions"]:
        instruction_labels, instruction_positive_count = collect_instruction_token_labels(
            instruction,
            offset_to_type,
            arg_reg_types,
            supervision_mode,
        )
        positive_count += instruction_positive_count
        supervised_count += sum(label is not None for label in instruction_labels)
        labels.extend(instruction_labels)
    return labels, positive_count, supervised_count


def build_dataset_sample(unified_payload, labels):
    flat = unified_payload["flat_fields"]
    assert len(labels) == len(flat["static"])
    return {
        "sample_id": f"{Path(unified_payload['binary']).name}_{unified_payload['function']['name']}_{unified_payload['function']['start']:x}",
        "binary": unified_payload["binary"],
        "function": unified_payload["function"],
        "arch": unified_payload["arch"],
        "token_count": len(flat["static"]),
        "static": flat["static"],
        "inst_pos_emb": flat["inst_pos_emb"],
        "op_pos_emb": flat["op_pos_emb"],
        "arch_emb": flat["arch_emb"],
        "byte1": flat["byte1"],
        "byte2": flat["byte2"],
        "byte3": flat["byte3"],
        "byte4": flat["byte4"],
        "aa_role_ids": flat["aa_role_ids"],
        "binary_pos": flat["binary_pos"],
        "function_pos": flat["function_pos"],
        "bb_pos": flat["bb_pos"],
        "var_offset": flat["var_offset"],
        "labels": labels,
        "summary": unified_payload.get("summary", {}),
    }


def split_samples(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[Path(sample["binary"]).name].append(sample)

    train_samples = []
    valid_samples = []
    for binary_name in sorted(grouped):
        group = sorted(
            grouped[binary_name],
            key=lambda item: hashlib.md5(item["sample_id"].encode("utf-8")).hexdigest(),
        )
        valid_count = round(len(group) * 0.2)
        if len(group) >= 4:
            valid_count = max(1, valid_count)
        valid_count = min(valid_count, max(0, len(group) - 1))
        valid_samples.extend(group[:valid_count])
        train_samples.extend(group[valid_count:])

    if not valid_samples and train_samples:
        valid_samples.append(train_samples.pop(0))

    def collect_labels(payloads):
        labels = set()
        for sample in payloads:
            labels.update(label for label in sample["labels"] if label is not None)
        return labels

    train_labels = collect_labels(train_samples)
    missing_labels = sorted(collect_labels(valid_samples) - train_labels)
    if missing_labels:
        rescued = []
        for label in missing_labels:
            for index, sample in enumerate(valid_samples):
                if label in sample["labels"]:
                    train_samples.append(valid_samples.pop(index))
                    rescued.append(label)
                    break
        if rescued:
            replacement_candidates = sorted(
                train_samples,
                key=lambda item: hashlib.md5((item["sample_id"] + "_replacement").encode("utf-8")).hexdigest(),
            )
            desired_valid = max(4, round(len(samples) * 0.2))
            desired_valid = min(desired_valid, len(samples) - 1)
            while len(valid_samples) < desired_valid and replacement_candidates:
                candidate = replacement_candidates.pop(0)
                train_samples.remove(candidate)
                valid_samples.append(candidate)

    train_samples = sorted(train_samples, key=lambda item: item["sample_id"])
    valid_samples = sorted(valid_samples, key=lambda item: item["sample_id"])
    return train_samples, valid_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Build a token-level type prediction smoke dataset from unified AA samples")
    parser.add_argument("--unified-json-dir", required=True)
    parser.add_argument("--source-dict-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--idat-path", default="/path/to/ida-pro-9.0/idat")
    parser.add_argument("--ida-script", default="/path/to/numAnchor/Stateformer/command/finetune/export_type_info_ida.py")
    parser.add_argument("--min-positive-instructions", type=int, default=2)
    parser.add_argument("--supervision-mode", choices=["dense", "sparse_register_var"], default="dense")
    parser.add_argument("--min-tokens", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    unified_json_dir = Path(args.unified_json_dir).resolve()
    source_dict_dir = Path(args.source_dict_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    type_dir = output_dir / "ida_typeinfo"
    dict_dir = output_dir / "dicts"
    output_dir.mkdir(parents=True, exist_ok=True)
    type_dir.mkdir(parents=True, exist_ok=True)
    dict_dir.mkdir(parents=True, exist_ok=True)

    binary_to_type_info = {}
    accepted_samples = []
    label_counter = Counter()
    supervised_counter = Counter()
    skipped_binaries = {}
    skipped_samples = Counter()

    for unified_path in sorted(unified_json_dir.glob("*.json")):
        with unified_path.open("r", encoding="utf-8") as handle:
            unified_payload = json.load(handle)

        token_count = int(unified_payload.get("summary", {}).get("token_count", len(unified_payload["flat_fields"]["static"])))
        if token_count < args.min_tokens or token_count > args.max_tokens:
            continue

        binary_path = Path(unified_payload["binary"]).resolve()
        if binary_path not in binary_to_type_info:
            try:
                type_export_path = run_ida_type_export(
                    binary_path=binary_path,
                    output_dir=type_dir / sanitize_path_name(binary_path),
                    idat_path=Path(args.idat_path),
                    ida_script=Path(args.ida_script),
                )
                binary_to_type_info[binary_path] = load_type_info(type_export_path)
            except Exception as exc:
                skipped_binaries[str(binary_path)] = str(exc)
                binary_to_type_info[binary_path] = None
                print(f"[warn] skipping binary due to IDA type export failure: {binary_path} :: {exc}")

        if binary_to_type_info[binary_path] is None:
            skipped_samples["ida_export_failure"] += 1
            continue

        key = (unified_payload["function"]["name"], int(unified_payload["function"]["start"]))
        function_type_info = binary_to_type_info[binary_path].get(key)
        if function_type_info is None:
            skipped_samples["missing_function_type"] += 1
            continue

        labels, positive_count, supervised_count = build_label_sequence(
            unified_payload,
            function_type_info,
            args.supervision_mode,
        )
        if positive_count < args.min_positive_instructions:
            skipped_samples["too_few_positive_tokens"] += 1
            continue
        if supervised_count == 0:
            skipped_samples["no_supervised_tokens"] += 1
            continue

        sample = build_dataset_sample(unified_payload, labels)
        accepted_samples.append(sample)
        label_counter.update(label for label in labels if label is not None)
        supervised_counter["supervised_tokens"] += supervised_count
        supervised_counter["positive_tokens"] += positive_count

    accepted_samples.sort(key=lambda item: item["sample_id"])
    train_samples, valid_samples = split_samples(accepted_samples)

    for name in params_files():
        shutil.copy2(source_dict_dir / f"{name}.txt", dict_dir / f"{name}.txt")

    label_vocab = [NO_ACCESS] + sorted(label for label in label_counter if label != NO_ACCESS)
    label_to_id = {label: index for index, label in enumerate(label_vocab)}

    for split_name, split_payloads in (("train", train_samples), ("valid", valid_samples)):
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in split_payloads:
                payload = dict(sample)
                payload["label_ids"] = [
                    label_to_id[label] if label is not None else PAD_LABEL_ID
                    for label in sample["labels"]
                ]
                handle.write(json.dumps(payload) + "\n")

    with (output_dir / "label_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(label_vocab, handle, indent=2)

    manifest = {
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "total_samples": len(accepted_samples),
        "label_vocab_size": len(label_vocab),
        "label_counts": dict(label_counter),
        "supervision_mode": args.supervision_mode,
        "supervised_tokens": supervised_counter["supervised_tokens"],
        "positive_tokens": supervised_counter["positive_tokens"],
        "train_binary_counts": dict(Counter(Path(sample["binary"]).name for sample in train_samples)),
        "valid_binary_counts": dict(Counter(Path(sample["binary"]).name for sample in valid_samples)),
        "train_label_set": sorted(
            {label for sample in train_samples for label in sample["labels"] if label is not None}
        ),
        "valid_label_set": sorted(
            {label for sample in valid_samples for label in sample["labels"] if label is not None}
        ),
        "skipped_samples": dict(skipped_samples),
        "skipped_binaries": skipped_binaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


def params_files():
    return [
        "static",
        "inst_pos_emb",
        "op_pos_emb",
        "arch_emb",
        "byte1",
        "byte2",
        "byte3",
        "byte4",
        "cover",
    ]


if __name__ == "__main__":
    main()
