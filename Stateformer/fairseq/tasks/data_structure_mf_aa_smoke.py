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
PAD_LABEL_ID = -100


def collate_float_tensors(values, pad_value):
    size = max(v.size(0) for v in values)
    result = values[0].new_full((len(values), size), pad_value)
    for index, value in enumerate(values):
        result[index, : value.size(0)] = value
    return result


class UnifiedAATypeDataset(FairseqDataset):
    def __init__(self, path, dictionary_dict):
        super().__init__()
        self.dictionary_dict = dictionary_dict
        self.samples = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                self.samples.append(self._encode_sample(payload))
        self.sizes = np.array([sample["length"] for sample in self.samples], dtype=np.int64)

    def _encode_field(self, field, values):
        dictionary = self.dictionary_dict[field]
        return torch.tensor([dictionary.index(str(value)) for value in values], dtype=torch.long)

    def _encode_sample(self, payload):
        src_tokens = {
            field: self._encode_field(field, payload[field])
            for field in params.fields
        }
        aa_inputs = {
            "aa_role_ids": torch.tensor(payload["aa_role_ids"], dtype=torch.long),
            "binary_pos": torch.tensor(payload["binary_pos"], dtype=torch.float32),
            "function_pos": torch.tensor(payload["function_pos"], dtype=torch.float32),
            "bb_pos": torch.tensor(payload["bb_pos"], dtype=torch.float32),
            "var_offset": torch.tensor(payload["var_offset"], dtype=torch.long),
        }
        target = torch.tensor(payload["label_ids"], dtype=torch.long)
        return {
            "id": payload["sample_id"],
            "length": len(payload["static"]),
            "src_tokens": src_tokens,
            "aa_inputs": aa_inputs,
            "target": target,
            "metadata": {
                "binary": payload["binary"],
                "function": payload["function"],
            },
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "id": index,
            "src_tokens": {field: value.clone() for field, value in sample["src_tokens"].items()},
            "aa_inputs": {key: value.clone() for key, value in sample["aa_inputs"].items()},
            "target": sample["target"].clone(),
            "length": sample["length"],
            "metadata": sample["metadata"],
        }

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        samples = sorted(samples, key=lambda item: item["length"], reverse=True)
        src_tokens = {
            field: data_utils.collate_tokens(
                [sample["src_tokens"][field] for sample in samples],
                pad_idx=self.dictionary_dict[field].pad(),
                left_pad=False,
            )
            for field in params.fields
        }

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
        target = collate_float_tensors(
            [sample["target"] for sample in samples],
            PAD_LABEL_ID,
        ).long()

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
            "nsentences": len(samples),
            "ntokens": ntokens,
        }

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        return np.argsort(self.sizes)


@register_task("data_structure_mf_aa_smoke")
class DataStructureMFAASmoke(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="dataset directory containing train.jsonl, valid.jsonl, dicts/, and label_vocab.json")
        parser.add_argument("--num-classes", type=int, default=-1, help="number of label classes")
        parser.add_argument(
            "--class-weighting",
            choices=["none", "sqrt_inverse", "inverse"],
            default="none",
            help="optional class reweighting for downstream token classification",
        )
        parser.add_argument(
            "--class-weight-max",
            type=float,
            default=10.0,
            help="clip class weights to this value; set <= 0 to disable clipping",
        )

    def __init__(self, args, dictionary_dict, dictionary_cf, label_vocab, class_weights=None):
        super().__init__(args)
        self.dictionary_dict = dictionary_dict
        self.dictionary_cf = dictionary_cf
        self.label_vocab = label_vocab
        self.class_weights = class_weights
        self.label_pad_idx = PAD_LABEL_ID
        self.no_access_idx = 0
        self._max_positions = args.max_positions
        args.tokens_per_sample = args.max_positions

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_dir = os.path.join(args.data, "dicts")
        dictionary_dict = {}
        for field in params.fields:
            dictionary_dict[field] = Dictionary.load(os.path.join(dict_dir, f"{field}.txt"))
            dictionary_dict[field].add_symbol("<mask>")
            logger.info("%s dictionary: %s types", field, len(dictionary_dict[field]))
        dictionary_cf = Dictionary.load(os.path.join(dict_dir, "cover.txt"))
        with open(os.path.join(args.data, "label_vocab.json"), "r", encoding="utf-8") as handle:
            label_vocab = json.load(handle)
        class_weights = None
        manifest_path = os.path.join(args.data, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            label_counts = manifest.get("label_counts", {})
            if args.class_weighting != "none":
                counts = torch.tensor(
                    [max(int(label_counts.get(label, 0)), 1) for label in label_vocab],
                    dtype=torch.float32,
                )
                if args.class_weighting == "inverse":
                    class_weights = 1.0 / counts
                else:
                    class_weights = 1.0 / torch.sqrt(counts)
                class_weights = class_weights / class_weights.mean()
                if args.class_weight_max > 0:
                    class_weights = torch.clamp(class_weights, max=args.class_weight_max)
                logger.info(
                    "Using %s class weighting with weights=%s",
                    args.class_weighting,
                    [round(float(value), 4) for value in class_weights.tolist()],
                )
        if args.num_classes <= 0:
            args.num_classes = len(label_vocab)
        return cls(args, dictionary_dict, dictionary_cf, label_vocab, class_weights=class_weights)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        path = os.path.join(self.args.data, f"{split}.jsonl")
        self.datasets[split] = UnifiedAATypeDataset(path, self.dictionary_dict)
        logger.info("Loaded %s with #samples: %s", split, len(self.datasets[split]))

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)
        model.register_classification_list(
            getattr(args, "classification_head_name", "data_structure_head"),
            num_classes=self.args.num_classes,
        )
        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict

    @property
    def target_cf_dictionary(self):
        return self.dictionary_cf
