import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils

NON_ADDR_SENTINEL = -1.0
NON_VAR_SENTINEL = -999999


def collate_float_tensors(values, pad_value):
    size = max(v.size(0) for v in values)
    result = values[0].new_full((len(values), size), pad_value)
    for index, value in enumerate(values):
        result[index, : value.size(0)] = value
    return result


class AASidecarDataset(FairseqDataset):
    FIELD_DTYPES = {
        "aa_role_ids": torch.long,
        "binary_pos": torch.float32,
        "function_pos": torch.float32,
        "bb_pos": torch.float32,
        "var_offset": torch.long,
    }
    PREPEND_VALUES = {
        "aa_role_ids": 0,
        "binary_pos": NON_ADDR_SENTINEL,
        "function_pos": NON_ADDR_SENTINEL,
        "bb_pos": NON_ADDR_SENTINEL,
        "var_offset": NON_VAR_SENTINEL,
    }
    APPEND_VALUES = PREPEND_VALUES
    PAD_VALUES = PREPEND_VALUES

    def __init__(self, split_prefix, add_boundary_tokens=True):
        super().__init__()
        self.add_boundary_tokens = add_boundary_tokens
        self.field_names = list(self.FIELD_DTYPES.keys())
        self.lines = {}
        for field in self.field_names:
            path = f"{split_prefix}.{field}"
            with open(path, "r", encoding="utf-8") as handle:
                self.lines[field] = [line.rstrip("\n") for line in handle]

        lengths = {len(lines) for lines in self.lines.values()}
        if len(lengths) != 1:
            raise ValueError(f"AA sidecar fields have mismatched line counts: {lengths}")
        self._len = next(iter(lengths))
        offset = 2 if self.add_boundary_tokens else 0
        self.sizes = np.array(
            [offset + len(self.lines["aa_role_ids"][index].split()) for index in range(self._len)],
            dtype=np.int64,
        )

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def __len__(self):
        return self._len

    def _parse_line(self, field, index):
        raw_values = self.lines[field][index].split()
        values = []
        if self.add_boundary_tokens:
            values.append(self.PREPEND_VALUES[field])
        dtype = self.FIELD_DTYPES[field]
        if dtype == torch.float32:
            values.extend(float(value) for value in raw_values)
        else:
            values.extend(int(value) for value in raw_values)
        if self.add_boundary_tokens:
            values.append(self.APPEND_VALUES[field])
        return torch.tensor(values, dtype=dtype)

    def __getitem__(self, index):
        return {field: self._parse_line(field, index) for field in self.field_names}

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        return {
            "aa_role_ids": data_utils.collate_tokens(
                [sample["aa_role_ids"] for sample in samples],
                pad_idx=self.PAD_VALUES["aa_role_ids"],
                left_pad=False,
            ),
            "binary_pos": collate_float_tensors(
                [sample["binary_pos"] for sample in samples],
                self.PAD_VALUES["binary_pos"],
            ),
            "function_pos": collate_float_tensors(
                [sample["function_pos"] for sample in samples],
                self.PAD_VALUES["function_pos"],
            ),
            "bb_pos": collate_float_tensors(
                [sample["bb_pos"] for sample in samples],
                self.PAD_VALUES["bb_pos"],
            ),
            "var_offset": data_utils.collate_tokens(
                [sample["var_offset"] for sample in samples],
                pad_idx=self.PAD_VALUES["var_offset"],
                left_pad=False,
            ),
        }
