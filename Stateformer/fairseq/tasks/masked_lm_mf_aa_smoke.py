# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os

import numpy as np
import torch

from fairseq.data import Dictionary, FairseqDataset, data_utils
from fairseq.tasks import LegacyFairseqTask, register_task
from command import params


logger = logging.getLogger(__name__)

NON_ADDR_SENTINEL = -1.0
NON_VAR_SENTINEL = -999999
BYTE_PAD_VALUE = -1.0


def byte_token_to_value(token):
    if token == "##":
        return 1.0
    return int(token, 16) / 256.0


def collate_float_tensors(values, pad_value):
    size = max(v.size(0) for v in values)
    result = values[0].new_full((len(values), size), pad_value)
    for index, value in enumerate(values):
        result[index, : value.size(0)] = value
    return result


def collate_bool_tensors(values):
    size = max(v.size(0) for v in values)
    result = torch.zeros((len(values), size), dtype=torch.bool)
    for index, value in enumerate(values):
        result[index, : value.size(0)] = value
    return result


class UnifiedAASmokeDataset(FairseqDataset):
    def __init__(self, path, dictionary_dict, dictionary_cf, mask_idx_dict, mask_prob, seed):
        super().__init__()
        self.path = path
        self.dictionary_dict = dictionary_dict
        self.dictionary_cf = dictionary_cf
        self.mask_idx_dict = mask_idx_dict
        self.mask_prob = mask_prob
        self.seed = seed
        self.epoch = 1
        self.byte_fields = params.fields[params.byte_start_pos:]
        self.samples = []

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                self.samples.append(self._encode_sample(payload))

        self.sizes = np.array([sample["length"] for sample in self.samples], dtype=np.int64)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _encode_field(self, field, values):
        dictionary = self.dictionary_dict[field]
        return torch.tensor([dictionary.index(str(value)) for value in values], dtype=torch.long)

    def _encode_sample(self, payload):
        src_tokens = {
            "static": self._encode_field("static", payload["static"]),
            "inst_pos_emb": self._encode_field("inst_pos_emb", payload["inst_pos_emb"]),
            "op_pos_emb": self._encode_field("op_pos_emb", payload["op_pos_emb"]),
            "arch_emb": self._encode_field("arch_emb", payload["arch_emb"]),
            "byte1": self._encode_field("byte1", payload["byte1"]),
            "byte2": self._encode_field("byte2", payload["byte2"]),
            "byte3": self._encode_field("byte3", payload["byte3"]),
            "byte4": self._encode_field("byte4", payload["byte4"]),
        }
        byte_values = {
            field: torch.tensor([byte_token_to_value(token) for token in payload[field]], dtype=torch.float16)
            for field in self.byte_fields
        }
        aa_inputs = {
            "aa_role_ids": torch.tensor(payload["aa_role_ids"], dtype=torch.long),
            "binary_pos": torch.tensor(payload["binary_pos"], dtype=torch.float32),
            "function_pos": torch.tensor(payload["function_pos"], dtype=torch.float32),
            "bb_pos": torch.tensor(payload["bb_pos"], dtype=torch.float32),
            "var_offset": torch.tensor(payload["var_offset"], dtype=torch.long),
        }
        target_cf = torch.tensor(
            [self.dictionary_cf.index(str(value)) for value in payload["cover"]],
            dtype=torch.long,
        )
        return {
            "id": payload["sample_id"],
            "metadata": {
                "binary": payload["binary"],
                "function": payload["function"],
                "summary": payload.get("summary", {}),
            },
            "length": len(payload["static"]),
            "src_tokens": src_tokens,
            "byte_values": byte_values,
            "target_cf": target_cf,
            "aa_inputs": aa_inputs,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        length = sample["length"]
        rng = np.random.RandomState(self.seed + self.epoch * 100003 + index)

        num_to_mask = max(1, int(round(length * self.mask_prob)))
        candidate_positions = np.arange(length, dtype=np.int64)
        mask_positions = rng.choice(candidate_positions, size=min(num_to_mask, len(candidate_positions)), replace=False)
        mask_positions = np.sort(mask_positions)
        mask_tensor = torch.tensor(mask_positions, dtype=torch.long)
        target_mask = torch.zeros(length, dtype=torch.bool)
        target_mask[mask_tensor] = True

        src_tokens = {}
        for field in params.fields:
            src_tokens[field] = sample["src_tokens"][field].clone()
            if field in self.byte_fields:
                src_tokens[field][mask_tensor] = self.mask_idx_dict[field]

        target = {}
        for field in self.byte_fields:
            target[field] = torch.full(
                (length,),
                BYTE_PAD_VALUE,
                dtype=torch.float16,
            )
            target[field][mask_tensor] = sample["byte_values"][field][mask_tensor]

        return {
            "id": index,
            "src_tokens": src_tokens,
            "target": target,
            "target_mask": target_mask,
            "target_cf_value": sample["target_cf"],
            "aa_inputs": {
                key: value.clone()
                for key, value in sample["aa_inputs"].items()
            },
            "length": length,
            "metadata": sample["metadata"],
        }

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        samples = sorted(samples, key=lambda item: item["length"], reverse=True)
        src_tokens = {}
        for field in params.fields:
            src_tokens[field] = data_utils.collate_tokens(
                [sample["src_tokens"][field] for sample in samples],
                pad_idx=self.dictionary_dict[field].pad(),
                left_pad=False,
            )

        target = {}
        for field in self.byte_fields:
            target[field] = collate_float_tensors(
                [sample["target"][field] for sample in samples],
                BYTE_PAD_VALUE,
            )

        target_mask = collate_bool_tensors([sample["target_mask"] for sample in samples])

        aa_inputs = {
            "aa_role_ids": data_utils.collate_tokens(
                [sample["aa_inputs"]["aa_role_ids"] for sample in samples],
                pad_idx=0,
                left_pad=False,
            ),
            "binary_pos": collate_float_tensors(
                [sample["aa_inputs"]["binary_pos"] for sample in samples],
                NON_ADDR_SENTINEL,
            ),
            "function_pos": collate_float_tensors(
                [sample["aa_inputs"]["function_pos"] for sample in samples],
                NON_ADDR_SENTINEL,
            ),
            "bb_pos": collate_float_tensors(
                [sample["aa_inputs"]["bb_pos"] for sample in samples],
                NON_ADDR_SENTINEL,
            ),
            "var_offset": data_utils.collate_tokens(
                [sample["aa_inputs"]["var_offset"] for sample in samples],
                pad_idx=NON_VAR_SENTINEL,
                left_pad=False,
            ),
        }

        target_cf_value = data_utils.collate_tokens(
            [sample["target_cf_value"] for sample in samples],
            pad_idx=self.dictionary_cf.pad(),
            left_pad=False,
        )

        src_lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)
        ntokens = int(src_lengths.sum().item())

        return {
            "id": torch.tensor([sample["id"] for sample in samples], dtype=torch.long),
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "aa_inputs": aa_inputs,
            },
            "target": target,
            "target_mask": target_mask,
            "target_cf_value": target_cf_value,
            "nsentences": len(samples),
            "ntokens": ntokens,
        }

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        return np.argsort(self.sizes)


@register_task("masked_lm_mf_aa_smoke")
class MaskedLMTaskMFAASmoke(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="dataset directory containing train.jsonl, valid.jsonl, and dicts/")
        parser.add_argument("--tokens-per-sample", default=2048, type=int, help="maximum supported sample length")
        parser.add_argument("--mask-prob", default=0.15, type=float, help="probability of masking byte tokens")

    def __init__(self, args, dictionary_dict, dictionary_cf):
        super().__init__(args)
        self.dictionary_dict = dictionary_dict
        self.dictionary_cf = dictionary_cf
        self.seed = args.seed
        self.mask_idx_dict = {}
        for field, dictionary in dictionary_dict.items():
            self.mask_idx_dict[field] = dictionary.add_symbol("<mask>")
        self.fields = params.fields

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_dir = os.path.join(args.data, "dicts")
        dictionary_dict = {}
        for field in params.fields:
            dictionary_dict[field] = Dictionary.load(os.path.join(dict_dir, f"{field}.txt"))
            logger.info("%s dictionary: %s types", field, len(dictionary_dict[field]))
        dictionary_cf = Dictionary.load(os.path.join(dict_dir, "cover.txt"))
        logger.info("%s dictionary: %s types", params.field_cf, len(dictionary_cf))
        return cls(args, dictionary_dict, dictionary_cf)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, f"{split}.jsonl")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"dataset split not found: {split_path}")
        dataset = UnifiedAASmokeDataset(
            path=split_path,
            dictionary_dict=self.dictionary_dict,
            dictionary_cf=self.dictionary_cf,
            mask_idx_dict=self.mask_idx_dict,
            mask_prob=self.args.mask_prob,
            seed=self.seed,
        )
        dataset.set_epoch(epoch)
        self.datasets[split] = dataset

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict

    @property
    def target_cf_dictionary(self):
        return self.dictionary_cf
