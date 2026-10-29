import torch

try:
    from .funcsim_base_dataloader import FunctionSimilarityDataset as _BaseFunctionSimilarityDataset
except ImportError:
    from funcsim_base_dataloader import FunctionSimilarityDataset as _BaseFunctionSimilarityDataset


class FunctionSimilarityDataset(_BaseFunctionSimilarityDataset):
    """
    Local CLAP-style funcsim dataset.

    Compared with local Jtrans_AAE, the only semantic difference is that
    segment labels carry instruction ids instead of all-zero sequence ids.
    Address-aware tokens and targets remain unchanged.
    """

    def encode_function(self, func_id: str) -> dict:
        instructions = self.function_blocks[func_id]
        instructions = instructions[: self.max_instructions]

        raw_ids = [self.sos_idx]
        raw_bp = [-1.0]
        raw_fp = [-1.0]
        raw_bb = [-1.0]
        raw_var = [-999999]
        raw_section_kind = [0]
        raw_type = [0]
        raw_segment = [1]

        boundaries = []
        last_segment = 1

        for inst_idx, inst in enumerate(instructions):
            boundaries.append(len(raw_ids))
            toks, bp, fp, bb, vo, sk, at = self._parse_instruction(inst)
            inst_segment = min(inst_idx + 1, self.seq_len)
            last_segment = inst_segment
            for t in toks:
                raw_ids.append(self.vocab.stoi.get(t, self.unk_idx))
            raw_bp.extend(bp)
            raw_fp.extend(fp)
            raw_bb.extend(bb)
            raw_var.extend(vo)
            raw_section_kind.extend(sk)
            raw_type.extend(at)
            raw_segment.extend([inst_segment] * len(toks))

        raw_ids.append(self.eos_idx)
        raw_bp.append(-1.0)
        raw_fp.append(-1.0)
        raw_bb.append(-1.0)
        raw_var.append(-999999)
        raw_section_kind.append(0)
        raw_type.append(0)
        raw_segment.append(last_segment)

        if len(raw_ids) > self.seq_len:
            raw_ids = raw_ids[: self.seq_len]
            raw_bp = raw_bp[: self.seq_len]
            raw_fp = raw_fp[: self.seq_len]
            raw_bb = raw_bb[: self.seq_len]
            raw_var = raw_var[: self.seq_len]
            raw_section_kind = raw_section_kind[: self.seq_len]
            raw_type = raw_type[: self.seq_len]
            raw_segment = raw_segment[: self.seq_len]

        boundaries = [b for b in boundaries if b < len(raw_ids) - 1]

        pad = self.seq_len - len(raw_ids)
        raw_ids += [self.pad_idx] * pad
        raw_bp += [-1.0] * pad
        raw_fp += [-1.0] * pad
        raw_bb += [-1.0] * pad
        raw_var += [-999999] * pad
        raw_section_kind += [0] * pad
        raw_type += [0] * pad
        raw_segment += [0] * pad

        if not self.use_address_var:
            raw_bp = [-1.0] * self.seq_len
            raw_fp = [-1.0] * self.seq_len
            raw_bb = [-1.0] * self.seq_len
            raw_var = [-999999] * self.seq_len
            raw_section_kind = [0] * self.seq_len
            raw_type = [0] * self.seq_len

        if not boundaries:
            boundaries = [1]

        return {
            "input_ids": torch.tensor(raw_ids, dtype=torch.long),
            "segment": torch.tensor(raw_segment, dtype=torch.long),
            "binary_pos": torch.tensor(raw_bp, dtype=torch.float),
            "function_pos": torch.tensor(raw_fp, dtype=torch.float),
            "bb_pos": torch.tensor(raw_bb, dtype=torch.float),
            "var_offsets": torch.tensor(raw_var, dtype=torch.long),
            "section_kind_ids": torch.tensor(raw_section_kind, dtype=torch.long),
            "addr_type_ids": torch.tensor(raw_type, dtype=torch.long),
            "num_instructions": len(boundaries),
            "instruction_boundaries": boundaries,
        }
