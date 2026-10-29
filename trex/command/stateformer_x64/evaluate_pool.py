#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from prepare_stateformer_trex import FIELD_ORDER, convert_inline_sequence, convert_inline_sequence_with_aa
except ModuleNotFoundError:
    from command.stateformer_x64.prepare_stateformer_trex import FIELD_ORDER, convert_inline_sequence, convert_inline_sequence_with_aa

BYTE_BLANK = "##"
NATIVE_INST_RE = re.compile(r"^([^(]+)\([^)]*\)\s*(.*)$")
TOKEN_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+")


def load_checkpoint_model(checkpoint_path: Path, data_dir: Path):
    from fairseq import checkpoint_utils

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint_path)],
        arg_overrides={"data": str(data_dir)},
    )
    model = models[0]
    model.eval()
    return model, task


def native_tokenize(text: str) -> List[str]:
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


def hex2trace_word(value: str) -> str:
    number = value.lower().replace("0x", "")[-8:]
    return number.rjust(8, "0")


def values_to_four_byte_lists(values: Sequence[str]) -> Dict[str, List[str]]:
    chunks = [[value[i : i + 2] for i in range(0, len(value), 2)] for value in values]
    return {
        "byte1": [chunk[0] for chunk in chunks],
        "byte2": [chunk[1] for chunk in chunks],
        "byte3": [chunk[2] for chunk in chunks],
        "byte4": [chunk[3] for chunk in chunks],
    }


def convert_native_function(function_instructions: Sequence[str]) -> Dict[str, List[str]]:
    fields = {name: [] for name in FIELD_ORDER}
    value_words: List[str] = []

    for inst_index, instruction in enumerate(function_instructions):
        match = NATIVE_INST_RE.match(instruction.strip())
        if match:
            disasm_text = f"{match.group(1)} {match.group(2)}".strip()
        else:
            disasm_text = instruction.strip()
        raw_tokens = native_tokenize(disasm_text)

        for token_index, token in enumerate(raw_tokens):
            hex_match = TOKEN_HEX_RE.search(token)
            if hex_match:
                static_token = "hexvar"
                value_word = hex2trace_word(hex_match.group(0))
            elif token.lower().isdigit():
                static_token = "num"
                value_word = hex2trace_word(hex(int(token)))
            else:
                static_token = token.lower()
                value_word = BYTE_BLANK * 4

            fields["static"].append(static_token)
            fields["inst_pos_emb"].append(str(inst_index))
            fields["op_pos_emb"].append(str(token_index))
            fields["arch_emb"].append("x64")
            value_words.append(value_word)

    byte_fields = values_to_four_byte_lists(value_words)
    for name, values in byte_fields.items():
        fields[name] = values
    return fields


def collate_aa_batch(samples: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    valid_samples = [sample for sample in samples if sample is not None]
    if not valid_samples:
        return None

    batch = {}
    for field in ["aa_role_ids", "binary_pos", "function_pos", "bb_pos", "var_offset"]:
        tensors = [sample[field] for sample in valid_samples]
        if tensors[0].dtype.is_floating_point:
            pad_value = -1.0
            result = tensors[0].new_full((len(tensors), max(t.size(0) for t in tensors)), pad_value)
            for index, tensor in enumerate(tensors):
                result[index, : tensor.size(0)] = tensor
            batch[field] = result
        else:
            pad_value = -999999 if field == "var_offset" else 0
            batch[field] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
    return batch


def encode_function(
    function_instructions: Sequence[str],
    dictionaries,
    mode: str,
    max_positions: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    if mode == "native":
        converted = convert_native_function(function_instructions)
        aa_inputs = None
    elif mode == "addressaware_aa":
        inline_text = "\t".join(function_instructions)
        converted, aa_fields = convert_inline_sequence_with_aa(inline_text, mode=mode)
        aa_inputs = {}
        for field, dtype in (
            ("aa_role_ids", torch.long),
            ("binary_pos", torch.float32),
            ("function_pos", torch.float32),
            ("bb_pos", torch.float32),
            ("var_offset", torch.long),
        ):
            values = aa_fields[field][: max_positions - 1]
            if dtype == torch.float32:
                tensor = torch.tensor([-1.0] + [float(value) for value in values], dtype=dtype)
            elif field == "var_offset":
                tensor = torch.tensor([-999999] + [int(value) for value in values], dtype=dtype)
            else:
                tensor = torch.tensor([0] + [int(value) for value in values], dtype=dtype)
            aa_inputs[field] = tensor
    else:
        inline_text = "\t".join(function_instructions)
        converted = convert_inline_sequence(inline_text, mode=mode)
        aa_inputs = None
    src_tokens: Dict[str, torch.Tensor] = {}

    for field in FIELD_ORDER:
        dictionary = dictionaries[field]
        ids = [dictionary.bos()]
        ids.extend(dictionary.index(token) for token in converted[field][: max_positions - 1])
        src_tokens[field] = torch.tensor(ids, dtype=torch.long)
    return src_tokens, aa_inputs


def collate_batch(samples: List[Dict[str, torch.Tensor]], dictionaries) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for field in FIELD_ORDER:
        pad = dictionaries[field].pad()
        seqs = [sample[field] for sample in samples]
        batch[field] = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad)
    return batch


@torch.no_grad()
def embed_functions(model, dictionaries, function_blocks: Dict[str, List[str]], func_ids: Sequence[str], mode: str, max_positions: int, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}
    for start in range(0, len(func_ids), batch_size):
        batch_ids = list(func_ids[start : start + batch_size])
        encoded_samples = [
            encode_function(function_blocks[func_id], dictionaries=dictionaries, mode=mode, max_positions=max_positions)
            for func_id in batch_ids
        ]
        batch_samples = [sample[0] for sample in encoded_samples]
        batch_aa = collate_aa_batch([sample[1] for sample in encoded_samples])
        batch_src = collate_batch(batch_samples, dictionaries=dictionaries)
        batch_src = {field: tensor.to(device) for field, tensor in batch_src.items()}
        if batch_aa is not None:
            batch_aa = {field: tensor.to(device) for field, tensor in batch_aa.items()}
        features = model(src_tokens=batch_src, aa_inputs=batch_aa, features_only=True)[0]["features"]
        mask = batch_src["static"].ne(dictionaries["static"].pad()).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, dim=-1)
        for func_id, emb in zip(batch_ids, pooled.cpu()):
            embeddings[func_id] = emb
    return embeddings


def compute_metrics(embeddings: Dict[str, torch.Tensor], funcsim_pairs: Dict[str, Dict[str, List[str]]], query_ids: Sequence[str], pool_ids: Sequence[str]) -> Dict[str, float]:
    pool = [func_id for func_id in pool_ids if func_id in embeddings]
    pool_matrix = torch.stack([embeddings[func_id] for func_id in pool], dim=0)
    recalls = {1: [], 5: [], 10: [], 20: []}
    reciprocal_ranks: List[float] = []

    for query_id in query_ids:
        if query_id not in embeddings or query_id not in funcsim_pairs:
            continue
        query = embeddings[query_id].unsqueeze(0)
        sims = (query @ pool_matrix.T).squeeze(0)
        if query_id in pool:
            sims[pool.index(query_id)] = -2.0
        ranking = torch.argsort(sims, descending=True).tolist()
        ranked_ids = [pool[index] for index in ranking]
        ground_truth = set(funcsim_pairs[query_id]["ground_truth"])
        if not ground_truth:
            continue
        for k in recalls:
            recalls[k].append(float(bool(set(ranked_ids[:k]) & ground_truth)))
        rr = 0.0
        for rank, candidate_id in enumerate(ranked_ids, start=1):
            if candidate_id in ground_truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    result = {
        f"recall@{k}": float(np.mean(values)) if values else 0.0
        for k, values in recalls.items()
    }
    result["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    result["num_queries"] = len(reciprocal_ranks)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained Trex checkpoints on the Stateformer 10k pool")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Binarized Trex data dir used to load dictionaries")
    parser.add_argument("--function-blocks", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json"))
    parser.add_argument("--funcsim-pairs", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json"))
    parser.add_argument("--pool-file", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json"))
    parser.add_argument("--mode", choices=["addressaware", "baseline", "native", "addressaware_aa"], required=True)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model, task = load_checkpoint_model(args.checkpoint, args.data_dir)
    model.to(device)

    with args.function_blocks.open("r", encoding="utf-8") as handle:
        function_blocks = json.load(handle)
    with args.funcsim_pairs.open("r", encoding="utf-8") as handle:
        funcsim_pairs = json.load(handle)
    with args.pool_file.open("r", encoding="utf-8") as handle:
        pool_spec = json.load(handle)

    query_ids = pool_spec["query_ids"]
    pool_ids = pool_spec["pool_function_ids"]
    all_ids = list(dict.fromkeys(pool_ids + query_ids))

    embeddings = embed_functions(
        model=model,
        dictionaries=task.source_dictionary,
        function_blocks=function_blocks,
        func_ids=all_ids,
        mode=args.mode,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        device=device,
    )
    metrics = compute_metrics(embeddings, funcsim_pairs, query_ids, pool_ids)
    payload = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "data_dir": str(args.data_dir),
        "pool_file": str(args.pool_file),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
