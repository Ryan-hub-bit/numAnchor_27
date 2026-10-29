from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from command.stateformer_x64.prepare_stateformer_trex import (
    FIELD_ORDER,
    convert_inline_sequence,
    convert_inline_sequence_with_aa,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_equivalence_groups(
    function_blocks: Dict[str, List[str]],
    funcsim_pairs: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        root = parent.get(node, node)
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for query_id, pair_data in funcsim_pairs.items():
        if query_id not in function_blocks:
            continue
        for gt_id in pair_data.get("ground_truth", []):
            gt_id = str(gt_id)
            if gt_id in function_blocks:
                union(query_id, gt_id)

    root_to_group: Dict[str, int] = {}
    func_to_group: Dict[str, int] = {}
    group_to_funcs: Dict[int, List[str]] = defaultdict(list)
    for func_id in function_blocks:
        root = find(func_id)
        group_id = root_to_group.setdefault(root, len(root_to_group))
        func_to_group[func_id] = group_id
        group_to_funcs[group_id].append(func_id)
    return func_to_group, dict(group_to_funcs)


def build_hash_groups(function_blocks: Dict[str, List[str]]) -> Dict[str, int]:
    hash_to_group: Dict[str, int] = {}
    func_to_hash_group: Dict[str, int] = {}
    for func_id, instructions in function_blocks.items():
        payload = json.dumps(instructions, separators=(",", ":"), ensure_ascii=False)
        content_hash = __import__("hashlib").sha1(payload.encode("utf-8")).hexdigest()
        hash_group = hash_to_group.setdefault(content_hash, len(hash_to_group))
        func_to_hash_group[func_id] = hash_group
    return func_to_hash_group


def build_positive_pairs(
    function_blocks: Dict[str, List[str]],
    funcsim_pairs: Dict[str, Dict[str, List[str]]],
) -> List[Tuple[str, str]]:
    positive_pairs: List[Tuple[str, str]] = []
    for query_id, pair_data in funcsim_pairs.items():
        if query_id not in function_blocks:
            continue
        for gt_id in pair_data.get("ground_truth", []):
            gt_id = str(gt_id)
            if gt_id in function_blocks:
                positive_pairs.append((query_id, gt_id))
    return positive_pairs


def split_positive_pairs(
    function_blocks: Dict[str, List[str]],
    funcsim_pairs: Dict[str, Dict[str, List[str]]],
    exclude_query_ids: Sequence[str],
    valid_fraction: float,
    seed: int,
) -> Dict[str, object]:
    func_to_group, group_to_funcs = build_equivalence_groups(function_blocks, funcsim_pairs)
    positive_pairs = build_positive_pairs(function_blocks, funcsim_pairs)

    eligible_groups = sorted({func_to_group[query_id] for query_id, _ in positive_pairs})
    excluded_groups = {
        func_to_group[str(query_id)]
        for query_id in exclude_query_ids
        if str(query_id) in func_to_group
    }

    train_valid_groups = [group_id for group_id in eligible_groups if group_id not in excluded_groups]
    rng = random.Random(seed)
    rng.shuffle(train_valid_groups)

    valid_group_count = int(round(len(train_valid_groups) * valid_fraction))
    if valid_group_count <= 0 and len(train_valid_groups) > 1:
        valid_group_count = 1
    if valid_group_count >= len(train_valid_groups) and len(train_valid_groups) > 1:
        valid_group_count = len(train_valid_groups) - 1

    valid_groups = set(train_valid_groups[:valid_group_count])
    train_groups = set(train_valid_groups[valid_group_count:])

    train_pairs = [
        pair for pair in positive_pairs
        if func_to_group[pair[0]] in train_groups
    ]
    valid_pairs = [
        pair for pair in positive_pairs
        if func_to_group[pair[0]] in valid_groups
    ]

    return {
        "func_to_group": func_to_group,
        "group_to_funcs": group_to_funcs,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "excluded_groups": excluded_groups,
        "train_pairs": train_pairs,
        "valid_pairs": valid_pairs,
    }


def sample_positive_pairs(
    pairs: Sequence[Tuple[str, str]],
    sample_count: int,
    seed: int,
) -> List[Tuple[str, str]]:
    if not pairs:
        return []
    rng = random.Random(seed)
    pair_list = list(pairs)
    if sample_count <= 0 or sample_count == len(pair_list):
        rng.shuffle(pair_list)
        return pair_list
    if sample_count < len(pair_list):
        return rng.sample(pair_list, sample_count)
    sampled = list(pair_list)
    while len(sampled) < sample_count:
        sampled.append(rng.choice(pair_list))
    rng.shuffle(sampled)
    return sampled


def encode_function(
    function_instructions: Sequence[str],
    dictionaries,
    mode: str,
    max_positions: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    if mode == "addressaware_aa":
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


def collate_src_batch(samples: List[Dict[str, torch.Tensor]], dictionaries) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for field in FIELD_ORDER:
        pad_value = dictionaries[field].pad()
        seqs = [sample[field] for sample in samples]
        batch[field] = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    return batch


def collate_aa_batch(samples: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    valid_samples = [sample for sample in samples if sample is not None]
    if not valid_samples:
        return None

    batch: Dict[str, torch.Tensor] = {}
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


def materialize_encodings(
    function_blocks: Dict[str, List[str]],
    func_ids: Iterable[str],
    dictionaries,
    mode: str,
    max_positions: int,
) -> Dict[str, Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]]:
    return {
        func_id: encode_function(function_blocks[func_id], dictionaries=dictionaries, mode=mode, max_positions=max_positions)
        for func_id in sorted(set(func_ids))
    }


class TrexPositivePairDataset(Dataset):
    def __init__(
        self,
        function_blocks: Dict[str, List[str]],
        pairs: Sequence[Tuple[str, str]],
        dictionaries,
        func_to_group: Dict[str, int],
        func_to_hash_group: Dict[str, int],
        mode: str,
        max_positions: int,
    ) -> None:
        self.pairs = list(pairs)
        self.func_to_group = func_to_group
        self.func_to_hash_group = func_to_hash_group
        required_func_ids = [func_id for pair in self.pairs for func_id in pair]
        self.encoded = materialize_encodings(
            function_blocks=function_blocks,
            func_ids=required_func_ids,
            dictionaries=dictionaries,
            mode=mode,
            max_positions=max_positions,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        func0_id, func1_id = self.pairs[index]
        src0, aa0 = self.encoded[func0_id]
        src1, aa1 = self.encoded[func1_id]
        return {
            "func0_src": src0,
            "func0_aa": aa0,
            "func1_src": src1,
            "func1_aa": aa1,
            "group_id": torch.tensor(self.func_to_group[func0_id], dtype=torch.long),
            "func0_hash_group_id": torch.tensor(self.func_to_hash_group[func0_id], dtype=torch.long),
            "func1_hash_group_id": torch.tensor(self.func_to_hash_group[func1_id], dtype=torch.long),
        }


def build_pair_collate_fn(dictionaries):
    def collate(batch):
        return {
            "func0_src": collate_src_batch([item["func0_src"] for item in batch], dictionaries=dictionaries),
            "func0_aa": collate_aa_batch([item["func0_aa"] for item in batch]),
            "func1_src": collate_src_batch([item["func1_src"] for item in batch], dictionaries=dictionaries),
            "func1_aa": collate_aa_batch([item["func1_aa"] for item in batch]),
            "group_id": torch.stack([item["group_id"] for item in batch]),
            "func0_hash_group_id": torch.stack([item["func0_hash_group_id"] for item in batch]),
            "func1_hash_group_id": torch.stack([item["func1_hash_group_id"] for item in batch]),
        }

    return collate
