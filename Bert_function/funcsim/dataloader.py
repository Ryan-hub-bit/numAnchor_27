"""
DataLoader for Function Similarity Fine-tuning.

Each sample is a (anchor, other, label) pair built from function_blocks.json
and funcsim_pairs.json.

Two encoding modes are supported:
  - legacy window-aligned encoding:
      <sos> [all instruction tokens...] <eos> <pad>...
  - function-level pretrain-aligned encoding:
      <sos> inst0 <eos> inst1 <eos> ... <pad>...

The second mode matches the new Bert_function pretraining distribution:
instruction separators are explicit and segment labels carry instruction IDs.
"""

import json
import hashlib
import random
import re
import sys
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vocab import WordVocab


class FunctionSimilarityDataset(Dataset):
    """
    Dataset for contrastive function-similarity fine-tuning.

    Ground-truth file structure (funcsim_pairs.json):
        { "<func_id>": { "ground_truth": ["<pos_id_1>", ...] }, ... }

    Function blocks file (function_blocks.json):
        { "<func_id>": ["opcode(0x...:fp:bb) operands...", ...], ... }

    Each __getitem__ returns two encoded functions + a binary label
    (1 = same function at different opt-level / positive,
     0 = different function / negative).
    """

    def __init__(
        self,
        function_blocks_file,   # path to function_blocks.json  (or None)
        funcsim_pairs_file,     # path to funcsim_pairs.json    (or None)
        vocab: WordVocab,
        seq_len: int = 420,
        max_instructions: int = 50,
        negative_samples: int = 1,
        encoding: str = "utf-8",
    ):
        self.vocab            = vocab
        self.seq_len          = seq_len
        self.max_instructions = max_instructions
        self.negative_samples = negative_samples

        # Set by train.py / evaluate.py after construction to match the loaded
        # BERT checkpoint.  True -> use parsed address / var positions.
        # False -> fill every position with the -1 sentinel (zero embedding).
        self.use_address_var = True
        # Independent ablation knob: keep function/bb/var cues but erase the
        # function-in-binary coordinate everywhere.
        self.use_binary_pos = True
        # Finer-grained knobs for follow-up ablations.
        self.use_code_binary_pos = True
        self.use_daddr_binary_pos = True
        # True -> emit the new function-level layout used by Bert_function
        # pretraining: <sos> inst0 <eos> inst1 <eos> ...
        self.use_function_level_encoding = False

        # --- special token IDs ---
        self.pad_idx  = vocab.stoi.get('<pad>',  0)
        self.unk_idx  = vocab.stoi.get('<unk>',  1)
        self.eos_idx  = vocab.stoi.get('<eos>',  2)
        self.sos_idx  = vocab.stoi.get('<sos>',  3)

        # --- regex patterns (identical to pretrain dataloader) ---
        self.addr_pattern        = re.compile(
            r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(
            r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.daddr_pattern       = re.compile(
            r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.var_pattern         = re.compile(r'var\((0x[0-9a-fA-F]+)\)')

        # --- load data ---
        self.function_blocks: dict = {}
        if function_blocks_file is not None:
            print(f"[DataLoader] Loading function blocks from {function_blocks_file}")
            with open(function_blocks_file, 'r', encoding=encoding) as f:
                self.function_blocks = json.load(f)
            print(f"[DataLoader] {len(self.function_blocks):,} function blocks loaded")

        self.funcsim_pairs: dict = {}
        if funcsim_pairs_file is not None:
            print(f"[DataLoader] Loading funcsim pairs from {funcsim_pairs_file}")
            with open(funcsim_pairs_file, 'r', encoding=encoding) as f:
                self.funcsim_pairs = json.load(f)
            print(f"[DataLoader] {len(self.funcsim_pairs):,} query functions loaded")

        # Build equivalence groups (union-find) so InfoNCE can mask
        # false negatives — functions with the same source but different
        # optimisation levels must not be treated as negatives.
        self.func_to_group = self._build_equivalence_groups()
        n_groups = len(set(self.func_to_group.values()))
        print(f"[DataLoader] {n_groups:,} equivalence groups built")

        # Build exact-content hash groups so instruction-identical functions
        # that are missing from explicit ground_truth do not become negatives.
        self.func_to_hash_group = self._build_content_hash_groups()
        n_hash_groups = len(set(self.func_to_hash_group.values()))
        print(f"[DataLoader] {n_hash_groups:,} content-hash groups built")

        self.training_pairs = self._build_pairs()
        print(f"[DataLoader] {len(self.training_pairs):,} training pairs created "
              f"({sum(l for _,_,l in self.training_pairs)} positive, "
              f"{sum(1-l for _,_,l in self.training_pairs)} negative)")

    def _segment_id_for_instruction(self, inst_idx: int) -> int:
        """Instruction IDs are 1-indexed; bucket 0 stays reserved for padding."""
        return min(inst_idx + 1, self.seq_len)

    # -------------------------------------------------------------------------
    # equivalence groups  (union-find)
    # -------------------------------------------------------------------------

    def _build_equivalence_groups(self):
        """
        Build equivalence groups from funcsim_pairs using union-find.

        For each query Q with ground_truth [G1, G2, G3], all of
        {Q, G1, G2, G3} belong to the same equivalence group (same source
        function compiled at different optimisation levels).

        Returns:
            dict[str, int]  mapping function_id -> numeric group_id
        """
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for query_id, pair_data in self.funcsim_pairs.items():
            ground_truth = [str(g) for g in pair_data.get('ground_truth', [])]
            for gt_id in ground_truth:
                union(query_id, gt_id)

        # Assign each function a numeric group id
        root_to_gid = {}
        counter = 0
        func_to_group = {}
        for func_id in self.function_blocks:
            root = find(func_id)
            if root not in root_to_gid:
                root_to_gid[root] = counter
                counter += 1
            func_to_group[func_id] = root_to_gid[root]

        return func_to_group

    def _build_content_hash_groups(self):
        """
        Build exact-content groups from function_blocks.

        These groups are stricter than optimisation-equivalence groups: if two
        function IDs have byte-for-byte identical instruction lists, they share
        one hash group even if they are not listed in each other's ground_truth.
        """
        hash_to_gid = {}
        func_to_hash_group = {}
        counter = 0

        for func_id, instructions in self.function_blocks.items():
            payload = json.dumps(instructions, ensure_ascii=False, separators=(",", ":"))
            content_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            if content_hash not in hash_to_gid:
                hash_to_gid[content_hash] = counter
                counter += 1
            func_to_hash_group[func_id] = hash_to_gid[content_hash]

        return func_to_hash_group

    # -------------------------------------------------------------------------
    # pair construction
    # -------------------------------------------------------------------------

    def _build_pairs(self):
        pairs        = []
        all_func_ids = list(self.function_blocks.keys())

        for query_id, pair_data in self.funcsim_pairs.items():
            if query_id not in self.function_blocks:
                continue
            # ground_truth IDs may be integers in JSON; normalise to strings
            # to match the string keys of function_blocks.
            ground_truth = [str(g) for g in pair_data.get('ground_truth', [])]

            for gt_id in ground_truth:
                if gt_id in self.function_blocks:
                    pairs.append((query_id, gt_id, 1))

            gt_set = set(ground_truth)
            gt_set.add(query_id)
            query_hash_group = self.func_to_hash_group.get(query_id)
            for _ in range(self.negative_samples):
                neg_id  = random.choice(all_func_ids)
                attempts = 0
                while (
                    neg_id in gt_set or
                    self.func_to_hash_group.get(neg_id) == query_hash_group
                ) and attempts < 30:
                    neg_id = random.choice(all_func_ids)
                    attempts += 1
                if neg_id not in gt_set and self.func_to_hash_group.get(neg_id) != query_hash_group:
                    pairs.append((query_id, neg_id, 0))

        random.shuffle(pairs)
        return pairs

    # -------------------------------------------------------------------------
    # instruction parser
    # -------------------------------------------------------------------------

    def _parse_instruction(self, inst_text: str):
        """
        Parse one instruction from the inline-annotated format.

        Returns six parallel lists (one entry per token):
            tokens, binary_pos, func_pos, bb_pos, var_offsets, addr_type_ids
        """
        tokens      = []
        binary_pos  = []
        func_pos    = []
        bb_pos_list = []
        var_offsets = []
        addr_type_ids = []

        # opcode
        m = self.addr_pattern.match(inst_text)
        if m:
            opcode        = m.group(1)
            fp_val        = float(m.group(4))
            bb_val        = float(m.group(5))
            operands_text = inst_text[m.end():].strip()
        else:
            parts         = inst_text.split()
            opcode        = parts[0] if parts else inst_text
            fp_val = bb_val = -1.0
            operands_text = ' '.join(parts[1:]) if len(parts) > 1 else ''

        # Match the current Bert_function pretrain routing:
        # code/opcode tokens use binary + function + bb positions.
        tokens.append(opcode)
        binary_pos.append(float(m.group(3)))
        func_pos.append(fp_val)
        bb_pos_list.append(bb_val)
        var_offsets.append(-999999)
        addr_type_ids.append(1 if fp_val >= 0 or bb_val >= 0 else 0)

        # operands
        pos = 0
        while pos < len(operands_text):
            nm = self.nested_addr_pattern.match(operands_text[pos:])
            if nm:
                tokens.append('address')
                binary_pos.append(float(nm.group(2)))
                func_pos.append(float(nm.group(3)))
                bb_pos_list.append(float(nm.group(4)))
                var_offsets.append(-999999)
                addr_type_ids.append(1)
                pos += nm.end()
                continue

            dm = self.daddr_pattern.match(operands_text[pos:])
            if dm:
                tokens.append('daddr')
                binary_pos.append(float(dm.group(2)))
                func_pos.append(float(dm.group(3)))
                bb_pos_list.append(float(dm.group(4)))
                var_offsets.append(-999999)
                addr_type_ids.append(2)
                pos += dm.end()
                continue

            vm = self.var_pattern.match(operands_text[pos:])
            if vm:
                raw = int(vm.group(1), 16)
                if raw >= (1 << 31):
                    raw -= (1 << 32)
                raw = max(-(1 << 20), min((1 << 20), raw))
                tokens.append('var')
                binary_pos.append(-1.0)
                func_pos.append(-1.0)
                bb_pos_list.append(-1.0)
                var_offsets.append(raw)
                addr_type_ids.append(3)
                pos += vm.end()
                continue

            ch = operands_text[pos]
            if ch.isspace():
                pos += 1
                continue
            if ch in '[]+-*,':
                tokens.append(ch)
                binary_pos.append(-1.0)
                func_pos.append(-1.0)
                bb_pos_list.append(-1.0)
                var_offsets.append(-999999)
                addr_type_ids.append(0)
                pos += 1
                continue
            end = pos
            while end < len(operands_text) and operands_text[end] not in ' []+-*,':
                end += 1
            tokens.append(operands_text[pos:end])
            binary_pos.append(-1.0)
            func_pos.append(-1.0)
            bb_pos_list.append(-1.0)
            var_offsets.append(-999999)
            addr_type_ids.append(0)
            pos = end

        return tokens, binary_pos, func_pos, bb_pos_list, var_offsets, addr_type_ids

    # -------------------------------------------------------------------------
    # function encoder (public — also used by evaluate.py)
    # -------------------------------------------------------------------------

    def _encode_function_legacy(self, func_id: str) -> dict:
        """
        Encode a function into fixed-length tensors of shape [seq_len], plus
        per-instruction boundary metadata for the per-instruction BERT encoder.

        Format of the flat buffer:
            <sos>  [instr_0 tokens]  [instr_1 tokens]  ...  <eos>  <pad>...

        instruction_boundaries[i] is the index in the flat buffer where
        instruction i begins (i.e. its first non-<sos> token position).
        Only boundaries that fall within [1, seq_len-1) are kept so the model
        can always slice at least one token per instruction.

        Returns a dict with keys:
            input_ids, segment, binary_pos, function_pos, bb_pos, var_offsets
                — all tensors of shape [seq_len]
            num_instructions  : int
            instruction_boundaries : list[int]  (length == num_instructions)
        """
        instructions = self.function_blocks[func_id]   # list[str]
        instructions = instructions[:self.max_instructions]

        raw_ids = [self.sos_idx]
        raw_bp  = [-1.0]
        raw_fp  = [-1.0]
        raw_bb  = [-1.0]
        raw_var = [-999999]
        raw_type = [0]

        boundaries: list[int] = []   # start index of each instruction

        for inst in instructions:
            # record the position *before* appending this instruction's tokens
            boundaries.append(len(raw_ids))
            toks, bp, fp, bb, vo, at = self._parse_instruction(inst)
            for t in toks:
                raw_ids.append(self.vocab.stoi.get(t, self.unk_idx))
            raw_bp.extend(bp)
            raw_fp.extend(fp)
            raw_bb.extend(bb)
            raw_var.extend(vo)
            raw_type.extend(at)

        raw_ids.append(self.eos_idx)
        raw_bp.append(-1.0)
        raw_fp.append(-1.0)
        raw_bb.append(-1.0)
        raw_var.append(-999999)
        raw_type.append(0)

        # truncate
        if len(raw_ids) > self.seq_len:
            raw_ids = raw_ids[:self.seq_len]
            raw_bp  = raw_bp[:self.seq_len]
            raw_fp  = raw_fp[:self.seq_len]
            raw_bb  = raw_bb[:self.seq_len]
            raw_var = raw_var[:self.seq_len]
            raw_type = raw_type[:self.seq_len]

        # drop boundaries that start at or beyond the truncation point
        # (those instructions were completely cut off)
        boundaries = [b for b in boundaries if b < len(raw_ids) - 1]

        # pad
        pad = self.seq_len - len(raw_ids)
        raw_ids += [self.pad_idx] * pad
        raw_bp  += [-1.0] * pad
        raw_fp  += [-1.0] * pad
        raw_bb  += [-1.0] * pad
        raw_var += [-999999] * pad
        raw_type += [0] * pad

        # override if checkpoint has no address/var embeddings
        if not self.use_address_var:
            raw_bp  = [-1.0] * self.seq_len
            raw_fp  = [-1.0] * self.seq_len
            raw_bb  = [-1.0] * self.seq_len
            raw_var = [-999999] * self.seq_len
            raw_type = [0] * self.seq_len
        elif not self.use_binary_pos:
            raw_bp = [-1.0] * self.seq_len
        else:
            raw_bp = [
                (-1.0 if ((t == 1 and not self.use_code_binary_pos) or
                          (t == 2 and not self.use_daddr_binary_pos))
                 else bp)
                for bp, t in zip(raw_bp, raw_type)
            ]

        # guarantee at least one instruction (use position 1 = right after <sos>)
        if not boundaries:
            boundaries = [1]

        seg = [0] * self.seq_len   # single-sequence -> all zeros

        return {
            'input_ids':             torch.tensor(raw_ids, dtype=torch.long),
            'segment':               torch.tensor(seg,     dtype=torch.long),
            'binary_pos':            torch.tensor(raw_bp,  dtype=torch.float),
            'function_pos':          torch.tensor(raw_fp,  dtype=torch.float),
            'bb_pos':                torch.tensor(raw_bb,  dtype=torch.float),
            'var_offsets':           torch.tensor(raw_var, dtype=torch.long),
            'addr_type_ids':         torch.tensor(raw_type, dtype=torch.long),
            'num_instructions':      len(boundaries),
            'instruction_boundaries': boundaries,   # list[int], NOT a tensor
        }

    def _encode_function_function_level(self, func_id: str) -> dict:
        """
        Encode one function with the exact Bert_function pretrain layout:

            <sos> inst0 <eos> inst1 <eos> ...

        Segment labels are 1-indexed instruction IDs and 0 only appears on
        padding. This lets downstream fine-tuning reuse the pretrained segment
        embedding semantics directly.
        """
        instructions = self.function_blocks[func_id][:self.max_instructions]

        raw_ids = [self.sos_idx]
        raw_bp = [-1.0]
        raw_fp = [-1.0]
        raw_bb = [-1.0]
        raw_var = [-999999]
        raw_type = [0]
        raw_seg = [1]
        boundaries: list[int] = []

        remaining = self.seq_len - 1  # reserve <sos>

        for inst_idx, inst in enumerate(instructions):
            toks, bp, fp, bb, vo, at = self._parse_instruction(inst)
            if not toks:
                continue

            required = len(toks) + 1  # real tokens + trailing <eos>
            if required <= remaining:
                take = len(toks)
            elif not boundaries and remaining > 1:
                take = remaining - 1
            else:
                break

            if take <= 0:
                break

            seg_id = self._segment_id_for_instruction(inst_idx)
            boundaries.append(len(raw_ids))

            tok_ids = [self.vocab.stoi.get(t, self.unk_idx) for t in toks[:take]]
            raw_ids.extend(tok_ids)
            raw_bp.extend(bp[:take])
            raw_fp.extend(fp[:take])
            raw_bb.extend(bb[:take])
            raw_var.extend(vo[:take])
            raw_type.extend(at[:take])
            raw_seg.extend([seg_id] * take)

            raw_ids.append(self.eos_idx)
            raw_bp.append(-1.0)
            raw_fp.append(-1.0)
            raw_bb.append(-1.0)
            raw_var.append(-999999)
            raw_type.append(0)
            raw_seg.append(seg_id)

            remaining -= take + 1
            if take < len(toks):
                break

        if not boundaries:
            boundaries = [1]

        pad = self.seq_len - len(raw_ids)
        if pad > 0:
            raw_ids += [self.pad_idx] * pad
            raw_bp += [-1.0] * pad
            raw_fp += [-1.0] * pad
            raw_bb += [-1.0] * pad
            raw_var += [-999999] * pad
            raw_type += [0] * pad
            raw_seg += [0] * pad

        if not self.use_address_var:
            raw_bp = [-1.0] * self.seq_len
            raw_fp = [-1.0] * self.seq_len
            raw_bb = [-1.0] * self.seq_len
            raw_var = [-999999] * self.seq_len
            raw_type = [0] * self.seq_len
        elif not self.use_binary_pos:
            raw_bp = [-1.0] * self.seq_len
        else:
            raw_bp = [
                (-1.0 if ((t == 1 and not self.use_code_binary_pos) or
                          (t == 2 and not self.use_daddr_binary_pos))
                 else bp)
                for bp, t in zip(raw_bp, raw_type)
            ]

        return {
            'input_ids': torch.tensor(raw_ids, dtype=torch.long),
            'segment': torch.tensor(raw_seg, dtype=torch.long),
            'binary_pos': torch.tensor(raw_bp, dtype=torch.float),
            'function_pos': torch.tensor(raw_fp, dtype=torch.float),
            'bb_pos': torch.tensor(raw_bb, dtype=torch.float),
            'var_offsets': torch.tensor(raw_var, dtype=torch.long),
            'addr_type_ids': torch.tensor(raw_type, dtype=torch.long),
            'num_instructions': len(boundaries),
            'instruction_boundaries': boundaries,
        }

    def encode_function(self, func_id: str) -> dict:
        if self.use_function_level_encoding:
            return self._encode_function_function_level(func_id)
        return self._encode_function_legacy(func_id)

    # -------------------------------------------------------------------------
    # Dataset API
    # -------------------------------------------------------------------------

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        a_id, b_id, label = self.training_pairs[idx]
        a = self.encode_function(a_id)
        b = self.encode_function(b_id)
        # group_id: same for all functions in the same equivalence class.
        # Used by InfoNCE to mask false negatives in the same batch.
        group_id = self.func_to_group.get(a_id, idx)
        func1_hash_group_id = self.func_to_hash_group.get(a_id, idx)
        func2_hash_group_id = self.func_to_hash_group.get(b_id, idx)
        return {
            'func1_input':        a['input_ids'],
            'func1_segment':      a['segment'],
            'func1_binary_pos':   a['binary_pos'],
            'func1_func_pos':     a['function_pos'],
            'func1_bb_pos':       a['bb_pos'],
            'func1_var_offsets':  a['var_offsets'],
            'func1_num_instr':    a['num_instructions'],       # int
            'func1_boundaries':   a['instruction_boundaries'], # list[int]
            'func2_input':        b['input_ids'],
            'func2_segment':      b['segment'],
            'func2_binary_pos':   b['binary_pos'],
            'func2_func_pos':     b['function_pos'],
            'func2_bb_pos':       b['bb_pos'],
            'func2_var_offsets':  b['var_offsets'],
            'func2_num_instr':    b['num_instructions'],       # int
            'func2_boundaries':   b['instruction_boundaries'], # list[int]
            'label':              torch.tensor(label, dtype=torch.long),
            'group_id':           torch.tensor(group_id, dtype=torch.long),
            'func1_hash_group_id': torch.tensor(func1_hash_group_id, dtype=torch.long),
            'func2_hash_group_id': torch.tensor(func2_hash_group_id, dtype=torch.long),
        }
