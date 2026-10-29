#!/usr/bin/env python3
"""
Prepare an address-aware SymLM dataset from the JSON exported by
`process_address_aware_ida.py`.

This script keeps SymLM's high-level supervision structure:
- self function body
- caller contexts
- internal callee contexts
- external callee labels
- function-name labels

But unlike upstream SymLM, the function-body input is stored as a single inline
sequence in the same address-aware format used by this repository's pretraining.
"""

import argparse
import json
import os
import re
from pathlib import Path

import nltk
import sentencepiece as spm
from nltk.corpus import wordnet
from nltk.corpus.reader import CorpusReader
from nltk.stem.wordnet import WordNetLemmatizer


SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENTATION_MODEL = SCRIPT_DIR / "segmentation_model" / "segmentation.model"
DUMMY_INLINE = "nop(0x0:0.00000000:0.00000000:0.00000000)"

sp = spm.SentencePieceProcessor()
sp.load(str(SEGMENTATION_MODEL))
lem = WordNetLemmatizer()


def ensure_nltk_resource(resource_name, lookup_path):
    try:
        nltk.data.find(lookup_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)


def is_placeholder_name(name):
    lowered = name.lower()
    prefixes = (
        "sub_",
        "nullsub_",
        "loc_",
        "j_",
        "unknown_",
        "__x86.",
    )
    exact = {
        "_init",
        "_fini",
        "frame_dummy",
        "register_tm_clones",
        "deregister_tm_clones",
        "__do_global_dtors_aux",
    }
    return lowered.startswith(prefixes) or lowered in exact


def func_name_segmentation(word):
    pieces = sp.encode_as_pieces(word)
    if pieces:
        pieces[0] = pieces[0][1:]
    return pieces


def get_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def normalize_function_name(func_name):
    if not func_name or is_placeholder_name(func_name):
        return None

    func_name = func_name.replace("_", " ")
    tmp = []
    current = ""
    for ch in func_name:
        if ch.isalpha():
            if ch.isupper() and current:
                tmp.append(current)
                current = ch
            else:
                current += ch
        else:
            if current:
                tmp.append(current)
            current = ""
    if current:
        tmp.append(current)

    merged = []
    i = 0
    while i < len(tmp):
        if len(tmp[i]) == 1 and tmp[i].isalpha():
            cap = ""
            while i < len(tmp) and len(tmp[i]) == 1 and tmp[i].isalpha():
                cap += tmp[i]
                i += 1
            merged.append(cap)
        else:
            merged.append(tmp[i])
            i += 1

    words = [w for w in merged if w]
    if not words:
        return None

    tags = nltk.pos_tag(words)
    lemmatized = []
    for word, tag in tags:
        wntag = get_pos(tag)
        if wntag is None:
            lemmatized.append(lem.lemmatize(word))
        else:
            lemmatized.append(lem.lemmatize(word, pos=wntag))

    final_words = []
    for word in lemmatized:
        for piece in func_name_segmentation(word):
            if piece:
                final_words.append(piece)

    if not final_words:
        return None
    return " ".join(final_words).lower()


def normalize_external_name(name):
    if not name:
        return "##"
    name = name.strip()
    if name.startswith("."):
        name = name[1:]
    if "@plt" in name:
        name = name.split("@plt", 1)[0]
    if "::" in name:
        name = name.split("::", 1)[-1]
    return name or "##"


def open_output_handles(binary_output_dir, topk):
    paths = [binary_output_dir / "self"]
    for i in range(topk):
        paths.extend(
            [
                binary_output_dir / f"caller{i + 1}",
                binary_output_dir / f"internal_callee{i + 1}",
                binary_output_dir / f"external_callee{i + 1}",
            ]
        )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    handles = {
        "self_inline": open(binary_output_dir / "self" / "input.inline", "w", encoding="utf-8"),
        "self_label": open(binary_output_dir / "self" / "input.label", "w", encoding="utf-8"),
        "caller_inline": [],
        "callee_inline": [],
        "external_label": [],
    }
    for i in range(topk):
        handles["caller_inline"].append(
            open(binary_output_dir / f"caller{i + 1}" / "input.inline", "w", encoding="utf-8")
        )
        handles["callee_inline"].append(
            open(binary_output_dir / f"internal_callee{i + 1}" / "input.inline", "w", encoding="utf-8")
        )
        handles["external_label"].append(
            open(binary_output_dir / f"external_callee{i + 1}" / "input.label", "w", encoding="utf-8")
        )
    return handles


def close_output_handles(handles):
    handles["self_inline"].close()
    handles["self_label"].close()
    for stream in handles["caller_inline"] + handles["callee_inline"] + handles["external_label"]:
        stream.close()


def choose_context_sequences(record, field_name, sequence_by_name, topk):
    selected = []
    for name in record.get(field_name, []):
        seq = sequence_by_name.get(name)
        if seq:
            selected.append(seq)
        if len(selected) >= topk:
            break
    while len(selected) < topk:
        selected.append(DUMMY_INLINE)
    return selected


def choose_external_labels(record, topk):
    selected = []
    for name in record.get("external_callees", []):
        selected.append(normalize_external_name(name))
        if len(selected) >= topk:
            break
    while len(selected) < topk:
        selected.append("##")
    return selected


def process_json_file(json_path, output_dir, topk):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return None

    binary_stem = json_path.stem
    binary_output_dir = Path(output_dir) / binary_stem
    handles = open_output_handles(binary_output_dir, topk)

    sequence_by_name = {}
    for record in records:
        sequence_by_name.setdefault(record["function"], record["funcins"])

    written = 0
    for record in records:
        label = normalize_function_name(record["function"])
        if label is None:
            continue

        handles["self_inline"].write(record["funcins"] + "\n")
        handles["self_label"].write(label + "\n")

        caller_sequences = choose_context_sequences(record, "callers", sequence_by_name, topk)
        callee_sequences = choose_context_sequences(record, "internal_callees", sequence_by_name, topk)
        external_labels = choose_external_labels(record, topk)

        for idx in range(topk):
            handles["caller_inline"][idx].write(caller_sequences[idx] + "\n")
            handles["callee_inline"][idx].write(callee_sequences[idx] + "\n")
            handles["external_label"][idx].write(external_labels[idx] + "\n")

        written += 1

    close_output_handles(handles)

    manifest = {
        "source_json": str(json_path),
        "num_records": len(records),
        "num_labeled_records": written,
        "topk": topk,
        "format": "address-aware-inline",
    }
    with open(binary_output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def gather_json_files(input_json=None, input_dir=None):
    files = []
    if input_json:
        files.append(Path(input_json))
    if input_dir:
        files.extend(sorted(Path(input_dir).glob("*_addressaware_symlm.json")))
    return files


def main():
    ensure_nltk_resource("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")
    ensure_nltk_resource("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")
    ensure_nltk_resource("wordnet", "corpora/wordnet")
    ensure_nltk_resource("omw-1.4", "corpora/omw-1.4")

    parser = argparse.ArgumentParser(description="Prepare address-aware SymLM dataset from IDA-exported JSON.")
    parser.add_argument("--input-json", type=str, help="Single *_addressaware_symlm.json file to convert.")
    parser.add_argument("--input-dir", type=str, help="Directory containing *_addressaware_symlm.json files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for prepared dataset.")
    parser.add_argument("--topK", type=int, default=2, help="Number of caller/callee contexts to keep.")
    args = parser.parse_args()

    if not args.input_json and not args.input_dir:
        raise ValueError("Provide either --input-json or --input-dir.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    for json_file in gather_json_files(args.input_json, args.input_dir):
        manifest = process_json_file(json_file, out_dir, args.topK)
        if manifest is not None:
            manifests.append(manifest)

    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)

    print(f"[INFO] Wrote {len(manifests)} prepared address-aware SymLM dataset shards to {out_dir}")


if __name__ == "__main__":
    main()
