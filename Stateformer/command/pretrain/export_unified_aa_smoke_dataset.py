#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from fairseq.data import Dictionary
from command import params


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


def load_sample(path):
    with Path(path).open("r") as handle:
        payload = json.load(handle)
    flat = payload["flat_fields"]
    token_count = len(flat["static"])
    sample = {
        "sample_id": f"{Path(payload['binary']).name}_{payload['function']['name']}_{payload['function']['start']:x}",
        "binary": payload["binary"],
        "function": payload["function"],
        "arch": payload["arch"],
        "token_count": token_count,
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
        "summary": payload["summary"],
    }
    return sample


def parse_args():
    parser = argparse.ArgumentParser(description="Export unified smoke JSON samples into a fairseq-readable smoke dataset")
    parser.add_argument("--input-json", nargs="+", required=True, help="unified JSON sample(s)")
    parser.add_argument("--output-dir", required=True, help="output dataset directory")
    parser.add_argument("--valid-count", type=int, default=1, help="number of samples to place in valid split")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    dict_dir = output_dir / "dicts"
    dict_dir.mkdir(parents=True, exist_ok=True)

    samples = [load_sample(path) for path in args.input_json]
    valid_count = min(args.valid_count, len(samples))
    if len(samples) == 1:
        train_samples = samples
        valid_samples = samples
    else:
        valid_samples = samples[:valid_count]
        train_samples = samples[valid_count:] or samples

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

    for split_name, split_samples in (("train", train_samples), ("valid", valid_samples)):
        with (output_dir / f"{split_name}.jsonl").open("w") as handle:
            for sample in split_samples:
                handle.write(json.dumps(sample) + "\n")

    manifest = {
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "tokens_per_sample": [sample["token_count"] for sample in samples],
        "source_json": [str(Path(path).resolve()) for path in args.input_json],
        "fields": params.fields,
        "cover_field": params.field_cf,
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
