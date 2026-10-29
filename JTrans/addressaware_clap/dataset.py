import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


NON_VAR_SENTINEL = -999999
NON_ADDR_SENTINEL = -1.0


class AddressAwareFunctionDataset(Dataset):
    """
    Function-level address-aware dataset for MLM + JTP pretraining.

    Each line in the corpus is one function represented as tab-separated
    address-aware instructions, e.g.:

        mov(0x401000:0.1:0.2:0.3) rax address(0x401020:...)
        \t
        call(0x401010:...) address(0x401080:...)
    """

    def __init__(
        self,
        corpus_path: str,
        vocab,
        seq_len: int = 256,
        token_mask_prob: float = 0.15,
        max_functions: Optional[int] = None,
        seed: int = 42,
    ):
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob
        self.rand = random.Random(seed)

        self.pad_idx = vocab.stoi.get("<pad>", 0)
        self.unk_idx = vocab.stoi.get("<unk>", 1)
        self.eos_idx = vocab.stoi.get("<eos>", 2)
        self.sos_idx = vocab.stoi.get("<sos>", 3)
        self.mask_idx = vocab.stoi.get("<mask>", 4)
        self.loc_idx = vocab.stoi.get("<loc>", self.mask_idx)

        self.addr_pattern = re.compile(
            r"(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)"
        )
        self.nested_addr_pattern = re.compile(
            r"address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)"
        )
        self.daddr_pattern = re.compile(
            r"daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)"
        )
        self.var_pattern = re.compile(r"var\((0x[0-9a-fA-F]+)\)")

        self.lines: List[str] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
                    if max_functions is not None and len(self.lines) >= max_functions:
                        break

    def __len__(self):
        return len(self.lines)

    def _parse_instruction(self, inst_text: str):
        tokens: List[str] = []
        binary_pos: List[float] = []
        function_pos: List[float] = []
        bb_pos: List[float] = []
        var_offsets: List[int] = []
        token_meta: List[dict] = []
        addr_type_ids: List[int] = []

        match = self.addr_pattern.match(inst_text)
        if not match:
            return (
                [inst_text],
                [NON_ADDR_SENTINEL],
                [NON_ADDR_SENTINEL],
                [NON_ADDR_SENTINEL],
                [NON_VAR_SENTINEL],
                [0],
                [{"kind": "plain"}],
                None,
            )

        opcode = match.group(1)
        opcode_addr = match.group(2)
        f_pos = float(match.group(4))
        bb_p = float(match.group(5))
        operands_text = inst_text[match.end():].strip()

        tokens.append(opcode)
        binary_pos.append(NON_ADDR_SENTINEL)
        function_pos.append(f_pos)
        bb_pos.append(bb_p)
        var_offsets.append(NON_VAR_SENTINEL)
        token_meta.append({"kind": "opcode", "addr": opcode_addr})
        addr_type_ids.append(1 if f_pos >= 0 or bb_p >= 0 else 0)

        pos = 0
        while pos < len(operands_text):
            chunk = operands_text[pos:]

            nested = self.nested_addr_pattern.match(chunk)
            if nested:
                tokens.append("address")
                binary_pos.append(NON_ADDR_SENTINEL)
                function_pos.append(float(nested.group(3)))
                bb_pos.append(float(nested.group(4)))
                var_offsets.append(NON_VAR_SENTINEL)
                token_meta.append({"kind": "address", "addr": nested.group(1)})
                addr_type_ids.append(1)
                pos += nested.end()
                continue

            daddr = self.daddr_pattern.match(chunk)
            if daddr:
                tokens.append("daddr")
                binary_pos.append(float(daddr.group(2)))
                function_pos.append(float(daddr.group(3)))
                bb_pos.append(float(daddr.group(4)))
                var_offsets.append(NON_VAR_SENTINEL)
                token_meta.append({"kind": "daddr", "addr": daddr.group(1)})
                addr_type_ids.append(2)
                pos += daddr.end()
                continue

            var_match = self.var_pattern.match(chunk)
            if var_match:
                raw = int(var_match.group(1), 16)
                if raw >= (1 << 31):
                    raw -= (1 << 32)
                raw = max(-(1 << 20), min((1 << 20), raw))
                tokens.append("var")
                binary_pos.append(NON_ADDR_SENTINEL)
                function_pos.append(NON_ADDR_SENTINEL)
                bb_pos.append(NON_ADDR_SENTINEL)
                var_offsets.append(raw)
                token_meta.append({"kind": "var", "offset": raw})
                addr_type_ids.append(3)
                pos += var_match.end()
                continue

            ch = operands_text[pos]
            if ch.isspace():
                pos += 1
                continue
            if ch in "[]+-*,":
                tok = ch
                pos += 1
            else:
                end = pos
                while end < len(operands_text) and operands_text[end] not in " []+-*,":
                    end += 1
                tok = operands_text[pos:end]
                pos = end
            tokens.append(tok)
            binary_pos.append(NON_ADDR_SENTINEL)
            function_pos.append(NON_ADDR_SENTINEL)
            bb_pos.append(NON_ADDR_SENTINEL)
            var_offsets.append(NON_VAR_SENTINEL)
            token_meta.append({"kind": "plain"})
            addr_type_ids.append(0)

        return tokens, binary_pos, function_pos, bb_pos, var_offsets, addr_type_ids, token_meta, opcode_addr

    def _encode_function(self, line: str):
        instructions = [inst.strip() for inst in line.split("\t") if inst.strip()]

        input_ids = [self.sos_idx]
        binary_pos = [NON_ADDR_SENTINEL]
        function_pos = [NON_ADDR_SENTINEL]
        bb_pos = [NON_ADDR_SENTINEL]
        var_offsets = [NON_VAR_SENTINEL]
        addr_type_ids = [0]
        # CLAP-style variant: use instruction IDs as segment labels.
        # Padding stays 0. [SOS] shares segment 1 with the first instruction.
        segments = [1]
        token_meta = [{"kind": "special"}]

        instruction_addr_to_pos: Dict[str, int] = {}
        candidate_sources: List[Tuple[int, str]] = []

        last_segment = 1
        for inst_idx, inst in enumerate(instructions):
            (
                toks,
                bp,
                fp,
                bb,
                vo,
                at,
                meta,
                opcode_addr,
            ) = self._parse_instruction(inst)

            inst_start = len(input_ids)
            if opcode_addr is not None:
                instruction_addr_to_pos[opcode_addr.lower()] = inst_start
            inst_segment = min(inst_idx + 1, self.seq_len)
            last_segment = inst_segment

            for i, tok in enumerate(toks):
                input_ids.append(self.vocab.stoi.get(tok, self.unk_idx))
                binary_pos.append(bp[i])
                function_pos.append(fp[i])
                bb_pos.append(bb[i])
                var_offsets.append(vo[i])
                addr_type_ids.append(at[i])
                segments.append(inst_segment)
                token_meta.append(meta[i])
                if meta[i].get("kind") == "address":
                    candidate_sources.append((len(input_ids) - 1, meta[i]["addr"].lower()))

        input_ids.append(self.eos_idx)
        binary_pos.append(NON_ADDR_SENTINEL)
        function_pos.append(NON_ADDR_SENTINEL)
        bb_pos.append(NON_ADDR_SENTINEL)
        var_offsets.append(NON_VAR_SENTINEL)
        addr_type_ids.append(0)
        segments.append(last_segment)
        token_meta.append({"kind": "special"})

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[: self.seq_len]
            binary_pos = binary_pos[: self.seq_len]
            function_pos = function_pos[: self.seq_len]
            bb_pos = bb_pos[: self.seq_len]
            var_offsets = var_offsets[: self.seq_len]
            addr_type_ids = addr_type_ids[: self.seq_len]
            segments = segments[: self.seq_len]
            token_meta = token_meta[: self.seq_len]

        valid_candidates = []
        for src_idx, target_addr in candidate_sources:
            target_idx = instruction_addr_to_pos.get(target_addr)
            if target_idx is None:
                continue
            if src_idx >= len(input_ids) or target_idx >= len(input_ids):
                continue
            valid_candidates.append((src_idx, target_idx))

        pad_len = self.seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_idx] * pad_len)
            binary_pos.extend([NON_ADDR_SENTINEL] * pad_len)
            function_pos.extend([NON_ADDR_SENTINEL] * pad_len)
            bb_pos.extend([NON_ADDR_SENTINEL] * pad_len)
            var_offsets.extend([NON_VAR_SENTINEL] * pad_len)
            addr_type_ids.extend([0] * pad_len)
            segments.extend([0] * pad_len)
            token_meta.extend([{"kind": "special"}] * pad_len)

        return {
            "input_ids": input_ids,
            "binary_pos": binary_pos,
            "function_pos": function_pos,
            "bb_pos": bb_pos,
            "var_offsets": var_offsets,
            "addr_type_ids": addr_type_ids,
            "segment": segments,
            "valid_candidates": valid_candidates,
        }

    def _make_mlm_sample(self, encoded):
        bert_input = list(encoded["input_ids"])
        bert_label = [-1] * self.seq_len
        var_offsets = list(encoded["var_offsets"])

        for idx in range(1, self.seq_len):
            tok = bert_input[idx]
            if tok in (self.pad_idx, self.sos_idx, self.eos_idx):
                continue
            if self.rand.random() >= self.token_mask_prob:
                continue

            bert_label[idx] = tok
            prob = self.rand.random()
            if prob < 0.8:
                bert_input[idx] = self.mask_idx
            elif prob < 0.9:
                bert_input[idx] = self.rand.randint(6, max(6, len(self.vocab) - 1))
            var_offsets[idx] = NON_VAR_SENTINEL

        return {
            "bert_input": torch.tensor(bert_input, dtype=torch.long),
            "bert_label": torch.tensor(bert_label, dtype=torch.long),
            "segment_label": torch.tensor(encoded["segment"], dtype=torch.long),
            "binary_pos": torch.tensor(encoded["binary_pos"], dtype=torch.float),
            "function_pos": torch.tensor(encoded["function_pos"], dtype=torch.float),
            "bb_pos": torch.tensor(encoded["bb_pos"], dtype=torch.float),
            "var_offsets": torch.tensor(var_offsets, dtype=torch.long),
            "addr_type_ids": torch.tensor(encoded["addr_type_ids"], dtype=torch.long),
        }

    def _make_jtp_sample(self, encoded):
        bert_input = list(encoded["input_ids"])
        binary_pos = list(encoded["binary_pos"])
        function_pos = list(encoded["function_pos"])
        bb_pos = list(encoded["bb_pos"])
        var_offsets = list(encoded["var_offsets"])

        source_idx = -1
        target_idx = -1
        if encoded["valid_candidates"]:
            source_idx, target_idx = self.rand.choice(encoded["valid_candidates"])
            bert_input[source_idx] = self.loc_idx
            binary_pos[source_idx] = NON_ADDR_SENTINEL
            function_pos[source_idx] = NON_ADDR_SENTINEL
            bb_pos[source_idx] = NON_ADDR_SENTINEL
            var_offsets[source_idx] = NON_VAR_SENTINEL

        return {
            "bert_input": torch.tensor(bert_input, dtype=torch.long),
            "segment_label": torch.tensor(encoded["segment"], dtype=torch.long),
            "binary_pos": torch.tensor(binary_pos, dtype=torch.float),
            "function_pos": torch.tensor(function_pos, dtype=torch.float),
            "bb_pos": torch.tensor(bb_pos, dtype=torch.float),
            "var_offsets": torch.tensor(var_offsets, dtype=torch.long),
            "addr_type_ids": torch.tensor(encoded["addr_type_ids"], dtype=torch.long),
            "source_idx": torch.tensor(source_idx, dtype=torch.long),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
        }

    def __getitem__(self, idx):
        encoded = self._encode_function(self.lines[idx])
        return {
            "mlm": self._make_mlm_sample(encoded),
            "jtp": self._make_jtp_sample(encoded),
        }


def load_function_blocks(function_blocks_path: str, limit: Optional[int] = None):
    with open(function_blocks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is None:
        return data
    out = {}
    for i, (k, v) in enumerate(data.items()):
        out[k] = v
        if i + 1 >= limit:
            break
    return out


def write_corpus_from_function_blocks(function_blocks_path: str, output_path: str, limit: Optional[int] = None):
    blocks = load_function_blocks(function_blocks_path, limit=limit)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for instructions in blocks.values():
            if not instructions:
                continue
            out.write("\t".join(instructions))
            out.write("\n")


def write_synthetic_funcsim_data(function_blocks_path: str, out_blocks_path: str, out_pairs_path: str, limit: int = 256):
    """
    Build a tiny downstream dataset without local hand-made labels:
    duplicate each function into an _a/_b pair so the pipeline can be verified
    end-to-end after downloading an external function-block dataset.
    """
    blocks = load_function_blocks(function_blocks_path, limit=limit)
    out_blocks = {}
    out_pairs = {}
    for idx, (func_id, instructions) in enumerate(blocks.items()):
        a_id = f"{func_id}__a"
        b_id = f"{func_id}__b"
        out_blocks[a_id] = instructions
        out_blocks[b_id] = list(instructions)
        out_pairs[a_id] = {"ground_truth": [b_id]}
        out_pairs[b_id] = {"ground_truth": [a_id]}

    os.makedirs(os.path.dirname(out_blocks_path), exist_ok=True)
    with open(out_blocks_path, "w", encoding="utf-8") as f:
        json.dump(out_blocks, f)
    with open(out_pairs_path, "w", encoding="utf-8") as f:
        json.dump(out_pairs, f)
