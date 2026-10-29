#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from elftools.elf.elffile import ELFFile

from fairseq.data import Dictionary
from command import params
from command.pretrain.extract_unified_x64_smoke import (
    IDA_ADDRESS_RE,
    IDA_DADDR_RE,
    IDA_HEADER_RE,
    IDA_VAR_RE,
    NON_ADDR_SENTINEL,
    annotate_instruction,
    build_basic_blocks,
    build_instruction_position_map,
    build_output,
    build_function_entry_position_map,
    collect_function_symbols,
    collect_sections,
    collect_segments,
    compute_binary_bounds,
    disassemble_function,
    parse_signed_hex,
    resolve_function,
    FunctionEmulator,
)


def is_elf(path):
    try:
        with Path(path).open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def sanitize_path_name(path):
    return "_".join(Path(path).parts[-3:]).replace(".", "_")


def run_ida_export(binary_path, output_dir, idat_path, ida_script, dummy_symbols):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(idat_path),
        "-A",
        f'-S"{ida_script} {output_dir} {dummy_symbols} false"',
        str(binary_path),
    ]
    subprocess.run(" ".join(command), shell=True, check=True)
    candidates = sorted(output_dir.glob("*_functions.json"))
    if not candidates:
        raise FileNotFoundError(f"IDA export did not produce a functions json for {binary_path}")
    return candidates[-1]


def parse_ida_export(ida_json_path):
    with Path(ida_json_path).open("r") as handle:
        exported = json.load(handle)

    mapping = {}
    for item in exported:
        instructions = [line for line in item.get("funcins", "").split("\t") if line.strip()]
        if not instructions:
            continue
        first_match = IDA_HEADER_RE.match(instructions[0].strip())
        if not first_match:
            continue
        function_start = int(first_match.group("addr"), 16)
        metadata = {}
        for line in instructions:
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
        mapping[(item.get("function"), function_start)] = metadata
    return mapping


def build_train_sample(unified_output):
    flat = unified_output["flat_fields"]
    return {
        "sample_id": f"{Path(unified_output['binary']).name}_{unified_output['function']['name']}_{unified_output['function']['start']:x}",
        "binary": unified_output["binary"],
        "function": unified_output["function"],
        "arch": unified_output["arch"],
        "token_count": len(flat["static"]),
        "static": flat["static"],
        "inst_pos_emb": flat["inst_pos_emb"],
        "op_pos_emb": flat["op_pos_emb"],
        "arch_emb": flat["arch_emb"],
        "byte1": flat["byte1"],
        "byte2": flat["byte2"],
        "byte3": flat["byte3"],
        "byte4": flat["byte4"],
        "cover": flat["cover"],
        "aa_role_ids": flat["aa_role_ids"],
        "binary_pos": flat["binary_pos"],
        "function_pos": flat["function_pos"],
        "bb_pos": flat["bb_pos"],
        "var_offset": flat["var_offset"],
        "summary": unified_output["summary"],
    }


def build_byte_dictionary():
    dictionary = Dictionary()
    dictionary.add_symbol("##")
    for value in range(256):
        dictionary.add_symbol(f"{value:02x}")
    dictionary.finalize(threshold=-1, nwords=-1, padding_factor=1)
    return dictionary


def build_dictionary_from_values(values):
    dictionary = Dictionary()
    for value in values:
        dictionary.add_symbol(str(value))
    dictionary.finalize(threshold=-1, nwords=-1, padding_factor=1)
    return dictionary


def export_dataset(samples, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_dir = output_dir / "dicts"
    dict_dir.mkdir(parents=True, exist_ok=True)

    static_values = []
    inst_values = []
    op_values = []
    arch_values = []
    cover_values = set()
    for sample in samples:
        static_values.extend(sample["static"])
        inst_values.extend(str(value) for value in sample["inst_pos_emb"])
        op_values.extend(str(value) for value in sample["op_pos_emb"])
        arch_values.extend(sample["arch_emb"])
        cover_values.update(str(value) for value in sample["cover"])

    dictionaries = {
        "static": build_dictionary_from_values(static_values),
        "inst_pos_emb": build_dictionary_from_values(inst_values),
        "op_pos_emb": build_dictionary_from_values(op_values),
        "arch_emb": build_dictionary_from_values(arch_values),
        "byte1": build_byte_dictionary(),
        "byte2": build_byte_dictionary(),
        "byte3": build_byte_dictionary(),
        "byte4": build_byte_dictionary(),
        "cover": build_dictionary_from_values(sorted(cover_values)),
    }
    for name, dictionary in dictionaries.items():
        dictionary.save(str(dict_dir / f"{name}.txt"))

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
        valid_count = round(len(group) * 0.1)
        if len(group) >= 6:
            valid_count = max(1, valid_count)
        valid_count = min(valid_count, max(0, len(group) - 1))
        valid_samples.extend(group[:valid_count])
        train_samples.extend(group[valid_count:])

    if not valid_samples and train_samples:
        valid_samples.append(train_samples.pop(0))

    train_samples = sorted(train_samples, key=lambda item: item["sample_id"])
    valid_samples = sorted(valid_samples, key=lambda item: item["sample_id"])

    for split_name, split_samples in (("train", train_samples), ("valid", valid_samples)):
        with (output_dir / f"{split_name}.jsonl").open("w") as handle:
            for sample in split_samples:
                handle.write(json.dumps(sample) + "\n")

    manifest = {
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "total_samples": len(samples),
        "fields": params.fields,
        "cover_field": params.field_cf,
        "train_binary_counts": dict(Counter(Path(sample["binary"]).name for sample in train_samples)),
        "valid_binary_counts": dict(Counter(Path(sample["binary"]).name for sample in valid_samples)),
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build a larger pilot unified AA+GSM dataset from a subset of binaries")
    parser.add_argument("--binary-root", required=True, help="root directory containing x86-64 binaries")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--max-binaries", type=int, default=2, help="maximum number of ELF binaries to process")
    parser.add_argument("--functions-per-binary", type=int, default=8, help="maximum accepted functions per binary")
    parser.add_argument("--min-tokens", type=int, default=30, help="minimum token count for accepted samples")
    parser.add_argument("--max-tokens", type=int, default=2048, help="maximum token count for accepted samples")
    parser.add_argument("--max-emu-instructions", type=int, default=256, help="maximum instructions for Unicorn smoke execution")
    parser.add_argument("--arch", default="x86-64")
    parser.add_argument("--idat-path", default="/path/to/ida-pro-9.0/idat")
    parser.add_argument("--ida-script", default="/path/to/numAnchor/PalmTree/addressaware/data_generator/process_funcsim_ida.py")
    parser.add_argument("--dummy-symbols", default="/path/to/numAnchor/PalmTree/addressaware/data_generator/dummy_symbols.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    binary_root = Path(args.binary_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ida_dir = output_dir / "ida_exports"
    unified_dir = output_dir / "unified_json"
    dataset_dir = output_dir / "dataset"
    ida_dir.mkdir(parents=True, exist_ok=True)
    unified_dir.mkdir(parents=True, exist_ok=True)

    binaries = []
    for path in sorted(binary_root.rglob("*")):
        if not path.is_file():
            continue
        if is_elf(path):
            binaries.append(path)
        if len(binaries) >= args.max_binaries:
            break

    collected_samples = []
    summary = []

    for binary_path in binaries:
        sanitized_name = sanitize_path_name(binary_path)
        binary_ida_dir = ida_dir / sanitized_name
        accepted = 0
        ida_json_path = None
        binary_error = None
        try:
            ida_json_path = run_ida_export(
                binary_path=binary_path,
                output_dir=binary_ida_dir,
                idat_path=Path(args.idat_path),
                ida_script=args.ida_script,
                dummy_symbols=args.dummy_symbols,
            )
            ida_metadata_map = parse_ida_export(ida_json_path)

            with binary_path.open("rb") as handle:
                elffile = ELFFile(handle)
                sections = collect_sections(elffile)
                binary_min, binary_max = compute_binary_bounds(sections)
                segments = collect_segments(elffile)
                symbols = collect_function_symbols(elffile)

                for symbol in symbols:
                    if accepted >= args.functions_per_binary:
                        break
                    key = (symbol["name"], symbol["start"])
                    if key not in ida_metadata_map:
                        continue
                    function_info = resolve_function(symbols, sections, address=symbol["start"])
                    try:
                        capstone_instructions = disassemble_function(elffile, function_info)
                    except Exception:
                        continue
                    if not capstone_instructions:
                        continue

                    bb_info = build_basic_blocks(capstone_instructions, function_info)
                    current_pos_map = build_instruction_position_map(
                        capstone_instructions,
                        function_info,
                        bb_info,
                        binary_min,
                        binary_max,
                    )
                    function_entry_pos_map = build_function_entry_position_map(symbols, binary_min, binary_max)

                    instructions = []
                    for inst_index, insn in enumerate(capstone_instructions):
                        annotated = annotate_instruction(
                            insn=insn,
                            function_info=function_info,
                            inst_index=inst_index,
                            bb_info=bb_info,
                            arch=args.arch,
                            sections=sections,
                            current_pos_map=current_pos_map,
                            function_entry_pos_map=function_entry_pos_map,
                            binary_min=binary_min,
                            binary_max=binary_max,
                            ida_metadata=ida_metadata_map[key],
                        )
                        annotated["capstone"] = insn
                        instructions.append(annotated)

                    emulator = FunctionEmulator(
                        segments=segments,
                        instructions=instructions,
                        function_info=function_info,
                        call_policy="skip",
                        max_instructions=args.max_emu_instructions,
                    )
                    emu_result = emulator.run()
                    for instruction in instructions:
                        instruction.pop("capstone", None)

                    unified_output = build_output(
                        binary_path=binary_path,
                        function_info=function_info,
                        instructions=instructions,
                        emu_result=emu_result,
                        arch=args.arch,
                        addressaware_source="ida_pro",
                    )
                    token_count = unified_output["summary"]["token_count"]
                    if token_count < args.min_tokens or token_count > args.max_tokens:
                        continue

                    sample = build_train_sample(unified_output)
                    collected_samples.append(sample)
                    accepted += 1

                    output_path = unified_dir / f"{sanitized_name}_{function_info['name']}_{function_info['start']:x}.json"
                    with output_path.open("w") as handle:
                        json.dump(unified_output, handle, indent=2)
        except Exception as exc:
            binary_error = str(exc)
            print(f"[warn] skipping binary after export/build failure: {binary_path} :: {exc}", file=sys.stderr)

        summary.append(
            {
                "binary": str(binary_path),
                "ida_json": str(ida_json_path) if ida_json_path else None,
                "accepted_functions": accepted,
                "error": binary_error,
            }
        )

    manifest = export_dataset(collected_samples, dataset_dir)
    result = {
        "binaries_processed": len(binaries),
        "samples_collected": len(collected_samples),
        "dataset_manifest": manifest,
        "per_binary": summary,
    }
    with (output_dir / "pilot_summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
