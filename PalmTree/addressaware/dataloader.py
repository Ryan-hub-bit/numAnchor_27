"""
Instruction Masking DataLoader for Address-Aware BERT

Implements instruction-level masking where entire instructions are masked at once.
This is different from token-level MLM which masks individual tokens.

For a line with 8 instructions and instruction_mask_rate=0.15:
- ~1 instruction will be fully masked (all its tokens replaced with [MASK])
- The model must predict all tokens of the masked instruction
- Also supports standard token-level MLM

Tasks:
- IMC (Instruction Masking CFG): Mask entire instructions in CFG sequences
- IMD (Instruction Masking DFG): Mask entire instructions in DFG sequences  
- MLM (Masked Language Model): Mask individual tokens  
"""

import json
import random
import re

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class InstructionMaskingDataset(Dataset):
    """
    Dataset that implements instruction-level masking.
    
    Strategy:
    - IMC: Mask entire instructions in CFG at instruction_mask_rate (e.g., 15%)
    - IMD: Mask entire instructions in DFG at instruction_mask_rate (e.g., 15%)
    - MLM: Mask individual tokens at token_mask_rate (e.g., 15%)
    """
    
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        token_mask_prob=0.15,  # Standard MLM masking rate
        instruction_mask_prob=0.25,  # Instruction-level masking rate
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
        enable_imd=False,  # Enable DFG instruction masking (controlled by args)
        enable_imc=False,
        enable_mlm=True,
        enable_cfg_nsp=False,
        nsp_negative_prob=0.5,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob
        self.instruction_mask_prob = instruction_mask_prob
        self.enable_imd = enable_imd
        self.enable_imc = enable_imc
        self.enable_mlm = enable_mlm
        self.enable_cfg_nsp = enable_cfg_nsp
        self.nsp_negative_prob = nsp_negative_prob
        
        # Special token IDs
        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)
        self.mask_idx = vocab.stoi.get('<mask>', 4)
        
        # Regex patterns for parsing inline format
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # Pattern to match section-aware data-address tokens:
        #   daddr_rodata(0xHEX:section_pos:offset_in_section)
        # and legacy format:
        #   daddr(0xHEX:section_pos:offset_in_section:section_kind_id)
        self.daddr_pattern = re.compile(
            r'((?:daddr(?:_[a-z]+)?))\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+)(?::([0-9.]+))?\)'
        )
        # Pattern to match var(0xXX) tokens
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
        
        # Load CFG data if path provided
        self.cfg_lines = []
        if cfg_corpus_path:
            print(f"Loading CFG corpus from {cfg_corpus_path}")
            self.cfg_lines = self._load_corpus(cfg_corpus_path)
            print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        # Load DFG data if enabled and path provided
        self.dfg_lines = []
        if self.enable_imd and dfg_corpus_path:
            print(f"Loading DFG corpus from {dfg_corpus_path}")
            self.dfg_lines = self._load_corpus(dfg_corpus_path)
            print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage
        if data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            if self.dfg_lines:
                dfg_size = int(len(self.dfg_lines) * data_percentage)
                self.dfg_lines = self.dfg_lines[:dfg_size]
        
        # Apply train/val split
        if train_split < 1.0:
            cfg_train_size = int(len(self.cfg_lines) * train_split)
            
            if is_train:
                self.cfg_lines = self.cfg_lines[:cfg_train_size]
                if self.dfg_lines:
                    dfg_train_size = int(len(self.dfg_lines) * train_split)
                    self.dfg_lines = self.dfg_lines[:dfg_train_size]
            else:
                self.cfg_lines = self.cfg_lines[cfg_train_size:]
                if self.dfg_lines:
                    dfg_train_size = int(len(self.dfg_lines) * train_split)
                    self.dfg_lines = self.dfg_lines[dfg_train_size:]
        
        print(f"Dataset size:")
        print(f"  CFG lines (for IMC+MLM): {len(self.cfg_lines)}")
        if self.dfg_lines:
            print(f"  DFG lines (for IMD): {len(self.dfg_lines)}")
        print(f"  Instruction mask rate: {instruction_mask_prob}")
        print(f"  Token mask rate: {token_mask_prob}")
        print(f"  CFG NSP enabled: {enable_cfg_nsp}")
    
    def _load_corpus(self, path):
        """Load corpus file"""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    @staticmethod
    def _parse_section_kind(raw_value):
        try:
            parsed = int(float(raw_value))
        except (TypeError, ValueError):
            return 4
        return parsed if parsed in (1, 2, 3, 4) else 4

    @staticmethod
    def _token_to_section_kind(token_name, legacy_kind=None):
        mapping = {
            'daddr_rodata': 1,
            'daddr_bss': 2,
            'daddr_data': 3,
            'daddr_other': 4,
        }
        if token_name in mapping:
            return token_name, mapping[token_name]
        parsed = InstructionMaskingDataset._parse_section_kind(legacy_kind)
        canonical = {
            1: 'daddr_rodata',
            2: 'daddr_bss',
            3: 'daddr_data',
            4: 'daddr_other',
        }[parsed]
        return canonical, parsed

    def _create_empty_sample(self):
        return {
            'bert_input': torch.LongTensor([self.pad_idx] * self.seq_len),
            'bert_label': torch.LongTensor([-1] * self.seq_len),
            'segment_label': torch.LongTensor([0] * self.seq_len),
            'binary_pos': torch.FloatTensor([-1.0] * self.seq_len),
            'function_pos': torch.FloatTensor([-1.0] * self.seq_len),
            'bb_pos': torch.FloatTensor([-1.0] * self.seq_len),
            'var_offsets': torch.LongTensor([-999999] * self.seq_len),
            'section_kind_ids': torch.LongTensor([0] * self.seq_len),
            'addr_type_ids': torch.LongTensor([0] * self.seq_len),
        }
    
    def _parse_instruction(self, inst_text):
        """
        Parse a single instruction and extract:
          tokens, continuous positions, var offsets, daddr section kinds, addr type IDs
        """
        tokens = []
        positions = []
        var_offsets = []
        section_kind_ids = []
        addr_type_ids = []
        
        # Match opcode with address
        match = self.addr_pattern.match(inst_text)
        if not match:
            return [inst_text], [(-1.0, -1.0, -1.0)], [-999999], [0], [0]
        
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        var_offsets.append(-999999)
        section_kind_ids.append(0)
        addr_type_ids.append(1)
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = self.nested_addr_pattern.match(operand)
                daddr_match = self.daddr_pattern.match(operand)
                var_match = self.var_pattern.match(operand)
                
                if nested_match:
                    nested_binary_pos = float(nested_match.group(2))
                    nested_function_pos = float(nested_match.group(3))
                    nested_bb_pos = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                    var_offsets.append(-999999)
                    section_kind_ids.append(0)
                    addr_type_ids.append(1)
                elif daddr_match:
                    daddr_token, daddr_section_kind = self._token_to_section_kind(
                        daddr_match.group(1),
                        daddr_match.group(5),
                    )
                    daddr_section_pos = float(daddr_match.group(3))
                    daddr_offset_in_section = float(daddr_match.group(4))
                    tokens.append(daddr_token)
                    positions.append((daddr_section_pos, daddr_offset_in_section, -1.0))
                    var_offsets.append(-999999)
                    section_kind_ids.append(daddr_section_kind)
                    addr_type_ids.append(2)
                elif var_match:
                    # Extract var offset from var(0xXX)
                    var_hex = var_match.group(1)
                    var_offset_value = int(var_hex, 16)
                    # IDA encodes negative stack offsets as large unsigned hex values.
                    # e.g. var(0xFFFFFFF0) means -16 in 32-bit two's complement.
                    # Interpret as signed: if the value >= 2^31, subtract 2^32.
                    # Also clamp to [-2^20, 2^20] to guard against any runaway values.
                    if var_offset_value >= (1 << 31):
                        var_offset_value -= (1 << 32)
                    var_offset_value = max(-(1 << 20), min((1 << 20), var_offset_value))
                    tokens.append('var')
                    positions.append((-1.0, -1.0, -1.0))
                    var_offsets.append(var_offset_value)
                    section_kind_ids.append(0)
                    addr_type_ids.append(3)
                else:
                    tokens.append(operand)
                    positions.append((-1.0, -1.0, -1.0))
                    var_offsets.append(-999999)
                    section_kind_ids.append(0)
                    addr_type_ids.append(0)
        
        return tokens, positions, var_offsets, section_kind_ids, addr_type_ids
    
    def _mask_instruction(self, tokens, positions, var_offsets, section_kind_ids, addr_type_ids):
        """
        Mask entire instruction (all tokens become [MASK]).
        
        Returns:
            masked_tokens: All tokens replaced with [MASK]
            positions: Original positions preserved
            var_offsets: Zeroed out to prevent data leakage (var is masked)
            original_tokens: Ground truth labels
        """
        masked_tokens = [self.mask_idx] * len(tokens)
        original_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
        masked_var_offsets = [-999999] * len(tokens)
        
        return masked_tokens, positions, masked_var_offsets, original_tokens, section_kind_ids, addr_type_ids
    
    def _mask_tokens(self, tokens, positions, var_offsets, section_kind_ids, addr_type_ids):
        """
        Apply standard MLM token-level masking.
        
        Returns:
            masked_tokens: Tokens with some randomly masked
            masked_var_offsets: Var offsets zeroed for masked tokens (prevent data leakage)
            labels: -1 for unmasked, original token ID for masked
        """
        masked_tokens = []
        masked_var_offsets = []
        labels = []
        
        for i, (token, pos) in enumerate(zip(tokens, positions)):
            token_id = self.vocab.stoi.get(token, self.unk_idx)
            var_offset = var_offsets[i] if i < len(var_offsets) else -999999
            
            # Randomly mask this token
            if random.random() < self.token_mask_prob:
                # 80% replace with [MASK]
                # 10% replace with random token
                # 10% keep original
                prob = random.random()
                if prob < 0.8:
                    masked_tokens.append(self.mask_idx)
                elif prob < 0.9:
                    # Random token - ensure it's within valid vocab range
                    random_token_id = random.randint(5, len(self.vocab.itos) - 1)
                    masked_tokens.append(random_token_id)
                else:
                    masked_tokens.append(token_id)
                labels.append(token_id)
                # Set var_offset to -999999 for masked tokens to prevent data leakage
                masked_var_offsets.append(-999999)
            else:
                masked_tokens.append(token_id)
                labels.append(-1)
                masked_var_offsets.append(var_offset)
        
        return masked_tokens, masked_var_offsets, labels, section_kind_ids, addr_type_ids
    
    def _process_line_for_instruction_masking(self, line):
        """
        Process a full line and apply instruction-level masking.
        Format: <sos> inst1 <eos> inst2 <eos> inst3 <eos> ...
        Each instruction is followed by <eos>.
        Segment labels: <sos>+inst1+<eos> get segment 0,
                        inst2+<eos> get segment 1,
                        inst3+<eos> get segment 0, etc. (alternating 0/1)
        Padding gets segment 0.
        
        Strategy: Calculate number of instructions to mask based on instruction_mask_prob,
        then randomly select which instructions to mask.
        
        Returns:
            bert_input: Token IDs with instruction-level masking
            bert_label: Labels for masked instructions (-1 for unmasked)
            segment_label: Alternating 0/1 per instruction
            binary_pos, function_pos, bb_pos: Position embeddings
            var_offsets: Var offset values (-999999 for non-var tokens)
        """
        instructions = [inst.strip() for inst in line.split('\t') if inst.strip()]
        
        # Calculate how many instructions to mask
        num_instructions = len(instructions)
        if num_instructions == 0:
            # Empty line, return minimal valid output
            return self._create_empty_sample()
        
        num_to_mask = max(1, int(num_instructions * self.instruction_mask_prob))
        # Ensure we don't mask all instructions (if more than 1)
        if num_instructions > 1:
            num_to_mask = min(num_to_mask, num_instructions - 1)
        
        # Randomly select which instructions to mask
        instructions_to_mask = set(random.sample(range(num_instructions), num_to_mask))
        
        all_tokens = []
        all_positions = []
        all_labels = []
        all_segments = []
        all_var_offsets = []
        all_section_kind_ids = []
        all_addr_type_ids = []
        
        # Add <sos> at the beginning (segment 0 — same as first instruction)
        all_tokens.append(self.sos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-1)
        all_segments.append(0)
        all_var_offsets.append(-999999)
        all_section_kind_ids.append(0)
        all_addr_type_ids.append(0)
        
        for inst_idx, inst_text in enumerate(instructions):
            tokens, positions, var_offsets, section_kind_ids, addr_type_ids = self._parse_instruction(inst_text)
            
            # Alternating segments: inst0 -> 0, inst1 -> 1, inst2 -> 0, ...
            inst_segment = inst_idx % 2
            
            # Mask this instruction if it was selected
            if inst_idx in instructions_to_mask:
                masked_tokens, positions, var_offsets, labels, section_kind_ids, addr_type_ids = self._mask_instruction(
                    tokens, positions, var_offsets, section_kind_ids, addr_type_ids
                )
            else:
                masked_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
                labels = [-1] * len(tokens)
            
            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([inst_segment] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)
            all_section_kind_ids.extend(section_kind_ids)
            all_addr_type_ids.extend(addr_type_ids)
            
            # Add <eos> after each instruction (same segment as the instruction)
            all_tokens.append(self.eos_idx)
            all_positions.append((-1.0, -1.0, -1.0))
            all_labels.append(-1)
            all_segments.append(inst_segment)
            all_var_offsets.append(-999999)
            all_section_kind_ids.append(0)
            all_addr_type_ids.append(0)
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
            all_segments = all_segments[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
            all_section_kind_ids = all_section_kind_ids[:self.seq_len]
            all_addr_type_ids = all_addr_type_ids[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-1] * padding_len
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len
            all_var_offsets += [-999999] * padding_len
            all_section_kind_ids += [0] * padding_len
            all_addr_type_ids += [0] * padding_len
        
        # Use segment labels as instruction IDs
        segment_label = all_segments
        
        # Split positions
        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]
        
        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(segment_label),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'var_offsets': torch.LongTensor(all_var_offsets),
            'section_kind_ids': torch.LongTensor(all_section_kind_ids),
            'addr_type_ids': torch.LongTensor(all_addr_type_ids),
        }
    
    def _process_line_for_token_masking(self, line):
        """
        Process a full line and apply token-level MLM masking.
        Format: [SOS] inst1 [EOS] inst2 [EOS] inst3 [EOS] ...
        Segment labels: [SOS] gets segment 1, inst1 tokens get 1, [EOS] gets 1, 
                        inst2 tokens get 2, [EOS] gets 2, etc.
        
        Returns:
            bert_input: Token IDs with token-level masking
            bert_label: Labels for masked tokens (-1 for unmasked)
            segment_label: Instruction ID (1 for inst1, 2 for inst2, ...)
            binary_pos, function_pos, bb_pos: Position embeddings
            var_offsets: Var offset values (0 for non-var tokens)
        """
        instructions = line.split('\t')
        
        all_tokens = []
        all_positions = []
        all_labels = []
        all_segments = []
        all_var_offsets = []
        all_section_kind_ids = []
        all_addr_type_ids = []
        
        # Add [SOS] at the beginning (gets segment 1 - same as first instruction)
        all_tokens.append(self.sos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-1)
        all_segments.append(1)
        all_var_offsets.append(-999999)
        all_section_kind_ids.append(0)
        all_addr_type_ids.append(0)
        
        for inst_idx, inst_text in enumerate(instructions):
            inst_text = inst_text.strip()
            if not inst_text:
                continue
            
            tokens, positions, var_offsets, section_kind_ids, addr_type_ids = self._parse_instruction(inst_text)
            
            # Apply token-level masking
            masked_tokens, var_offsets, labels, section_kind_ids, addr_type_ids = self._mask_tokens(
                tokens, positions, var_offsets, section_kind_ids, addr_type_ids
            )
            # Segment label = instruction number (1-indexed)
            inst_segment = inst_idx + 1
            
            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([inst_segment] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)
            all_section_kind_ids.extend(section_kind_ids)
            all_addr_type_ids.extend(addr_type_ids)
            
            # Add [EOS] after each instruction (same segment as the instruction)
            all_tokens.append(self.eos_idx)
            all_positions.append((-1.0, -1.0, -1.0))
            all_labels.append(-1)
            all_segments.append(inst_segment)
            all_var_offsets.append(-999999)
            all_section_kind_ids.append(0)
            all_addr_type_ids.append(0)
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
            all_segments = all_segments[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
            all_section_kind_ids = all_section_kind_ids[:self.seq_len]
            all_addr_type_ids = all_addr_type_ids[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-1] * padding_len
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len
            all_var_offsets += [-999999] * padding_len
            all_section_kind_ids += [0] * padding_len
            all_addr_type_ids += [0] * padding_len
        
        # Use segment labels as instruction IDs
        segment_label = all_segments
        
        # Split positions
        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]
        
        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(segment_label),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'var_offsets': torch.LongTensor(all_var_offsets),
            'section_kind_ids': torch.LongTensor(all_section_kind_ids),
            'addr_type_ids': torch.LongTensor(all_addr_type_ids),
        }

    def _split_cfg_pair(self, line):
        instructions = [inst.strip() for inst in line.split('\t') if inst.strip()]
        if len(instructions) >= 2:
            return instructions[0], instructions[1]
        if len(instructions) == 1:
            return instructions[0], instructions[0]
        return "", ""

    def _get_random_cfg_second_instruction(self, exclude_index):
        if not self.cfg_lines:
            return ""

        for _ in range(16):
            rand_idx = random.randrange(len(self.cfg_lines))
            if len(self.cfg_lines) > 1 and rand_idx == exclude_index:
                continue
            _, candidate = self._split_cfg_pair(self.cfg_lines[rand_idx])
            if candidate:
                return candidate

        _, fallback = self._split_cfg_pair(self.cfg_lines[exclude_index % len(self.cfg_lines)])
        return fallback

    def _process_cfg_pair_for_mlm_nsp(self, index):
        """
        Build a BERT-style pair sample for CFG MLM + NSP:
          [SOS] inst_a [EOS] inst_b [EOS]

        Segment labels follow the original PalmTree/BERT convention:
          sentence A -> 1, sentence B -> 2.
        """
        inst_a, inst_b = self._split_cfg_pair(self.cfg_lines[index])
        is_next = 1

        if random.random() < self.nsp_negative_prob:
            inst_b = self._get_random_cfg_second_instruction(index)
            is_next = 0

        all_tokens = [self.sos_idx]
        all_positions = [(-1.0, -1.0, -1.0)]
        all_labels = [-1]
        all_segments = [1]
        all_var_offsets = [-999999]
        all_section_kind_ids = [0]
        all_addr_type_ids = [0]

        for segment_id, inst_text in ((1, inst_a), (2, inst_b)):
            if not inst_text:
                continue

            tokens, positions, var_offsets, section_kind_ids, addr_type_ids = self._parse_instruction(inst_text)
            masked_tokens, var_offsets, labels, section_kind_ids, addr_type_ids = self._mask_tokens(
                tokens, positions, var_offsets, section_kind_ids, addr_type_ids
            )

            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([segment_id] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)
            all_section_kind_ids.extend(section_kind_ids)
            all_addr_type_ids.extend(addr_type_ids)

            all_tokens.append(self.eos_idx)
            all_positions.append((-1.0, -1.0, -1.0))
            all_labels.append(-1)
            all_segments.append(segment_id)
            all_var_offsets.append(-999999)
            all_section_kind_ids.append(0)
            all_addr_type_ids.append(0)

        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
            all_segments = all_segments[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
            all_section_kind_ids = all_section_kind_ids[:self.seq_len]
            all_addr_type_ids = all_addr_type_ids[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-1] * padding_len
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len
            all_var_offsets += [-999999] * padding_len
            all_section_kind_ids += [0] * padding_len
            all_addr_type_ids += [0] * padding_len

        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]

        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(all_segments),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'var_offsets': torch.LongTensor(all_var_offsets),
            'section_kind_ids': torch.LongTensor(all_section_kind_ids),
            'addr_type_ids': torch.LongTensor(all_addr_type_ids),
            'nsp_label': torch.tensor(is_next, dtype=torch.long),
        }
    
    def __len__(self):
        """Length = number of CFG lines"""
        return len(self.cfg_lines)
    
    def __getitem__(self, index):
        """
        Get one training sample with IMC, IMD, and MLM.
        
        Returns:
            - imc_data: Dict with instruction-masked data from full CFG line
            - imd_data: Dict with instruction-masked data from full DFG line (if enabled)
            - mlm_data: Dict with token-masked data from full CFG line (different masking)
        """
        result = {}

        if self.enable_imc:
            result['imc'] = self._process_line_for_instruction_masking(self.cfg_lines[index])

        if self.enable_mlm or self.enable_cfg_nsp:
            if self.enable_cfg_nsp:
                result['mlm'] = self._process_cfg_pair_for_mlm_nsp(index)
            else:
                result['mlm'] = self._process_line_for_token_masking(self.cfg_lines[index])
        
        # IMD: Process the full DFG line with instruction-level masking (if enabled)
        if self.enable_imd and self.dfg_lines:
            dfg_index = index % len(self.dfg_lines)
            imd_data = self._process_line_for_instruction_masking(self.dfg_lines[dfg_index])
            result['imd'] = imd_data
        
        return result


class StructurePairDataset(Dataset):
    """
    Auxiliary pair-classification dataset for structural relations.

    Expected input: JSONL rows produced by build_structure_pair_corpus.py with:
      - task: same_bb | same_function
      - label: 0/1
      - left:  ["inst ...", ...]
      - right: ["inst ...", ...]
    """

    def __init__(self, pair_corpus_path, vocab, seq_len=512, max_instructions_per_side=8):
        self.vocab = vocab
        self.seq_len = seq_len
        self.max_instructions_per_side = max_instructions_per_side

        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)

        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.daddr_pattern = re.compile(
            r'((?:daddr(?:_[a-z]+)?))\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+)(?::([0-9.]+))?\)'
        )
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')

        self.samples = []
        with open(pair_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    @staticmethod
    def _parse_section_kind(raw_value):
        try:
            parsed = int(float(raw_value))
        except (TypeError, ValueError):
            return 4
        return parsed if parsed in (1, 2, 3, 4) else 4

    @staticmethod
    def _token_to_section_kind(token_name, legacy_kind=None):
        mapping = {
            'daddr_rodata': 1,
            'daddr_bss': 2,
            'daddr_data': 3,
            'daddr_other': 4,
        }
        if token_name in mapping:
            return token_name, mapping[token_name]
        parsed = StructurePairDataset._parse_section_kind(legacy_kind)
        canonical = {
            1: 'daddr_rodata',
            2: 'daddr_bss',
            3: 'daddr_data',
            4: 'daddr_other',
        }[parsed]
        return canonical, parsed

    def _parse_instruction(self, inst_text):
        match = self.addr_pattern.match(inst_text)
        if not match:
            return [inst_text], [(-1.0, -1.0, -1.0)], [-999999], [0], [0]

        tokens = []
        positions = []
        var_offsets = []
        section_kind_ids = []
        addr_type_ids = []

        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))

        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        var_offsets.append(-999999)
        section_kind_ids.append(0)
        addr_type_ids.append(1)

        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = self.nested_addr_pattern.match(operand)
                daddr_match = self.daddr_pattern.match(operand)
                var_match = self.var_pattern.match(operand)

                if nested_match:
                    tokens.append('address')
                    positions.append((
                        float(nested_match.group(2)),
                        float(nested_match.group(3)),
                        float(nested_match.group(4)),
                    ))
                    var_offsets.append(-999999)
                    section_kind_ids.append(0)
                    addr_type_ids.append(1)
                elif daddr_match:
                    daddr_token, daddr_section_kind = self._token_to_section_kind(
                        daddr_match.group(1),
                        daddr_match.group(5),
                    )
                    tokens.append(daddr_token)
                    positions.append((
                        float(daddr_match.group(3)),
                        float(daddr_match.group(4)),
                        -1.0,
                    ))
                    var_offsets.append(-999999)
                    section_kind_ids.append(daddr_section_kind)
                    addr_type_ids.append(2)
                elif var_match:
                    var_offset_value = int(var_match.group(1), 16)
                    if var_offset_value >= (1 << 31):
                        var_offset_value -= (1 << 32)
                    var_offset_value = max(-(1 << 20), min((1 << 20), var_offset_value))
                    tokens.append('var')
                    positions.append((-1.0, -1.0, -1.0))
                    var_offsets.append(var_offset_value)
                    section_kind_ids.append(0)
                    addr_type_ids.append(3)
                else:
                    tokens.append(operand)
                    positions.append((-1.0, -1.0, -1.0))
                    var_offsets.append(-999999)
                    section_kind_ids.append(0)
                    addr_type_ids.append(0)

        return tokens, positions, var_offsets, section_kind_ids, addr_type_ids

    def _build_side(self, instructions, segment_id):
        tokens = []
        positions = []
        segments = []
        var_offsets = []
        section_kind_ids = []
        addr_type_ids = []

        for inst_text in instructions[:self.max_instructions_per_side]:
            parsed_tokens, parsed_positions, parsed_var_offsets, parsed_section_kind_ids, parsed_addr_type_ids = self._parse_instruction(inst_text)
            tokens.extend([self.vocab.stoi.get(tok, self.unk_idx) for tok in parsed_tokens])
            positions.extend(parsed_positions)
            segments.extend([segment_id] * len(parsed_tokens))
            var_offsets.extend(parsed_var_offsets)
            section_kind_ids.extend(parsed_section_kind_ids)
            addr_type_ids.extend(parsed_addr_type_ids)

            tokens.append(self.eos_idx)
            positions.append((-1.0, -1.0, -1.0))
            segments.append(segment_id)
            var_offsets.append(-999999)
            section_kind_ids.append(0)
            addr_type_ids.append(0)

        return tokens, positions, segments, var_offsets, section_kind_ids, addr_type_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        left = sample.get('left', [])
        right = sample.get('right', [])

        all_tokens = [self.sos_idx]
        all_positions = [(-1.0, -1.0, -1.0)]
        all_segments = [1]
        all_var_offsets = [-999999]
        all_section_kind_ids = [0]
        all_addr_type_ids = [0]

        for segment_id, instructions in ((1, left), (2, right)):
            side = self._build_side(instructions, segment_id)
            all_tokens.extend(side[0])
            all_positions.extend(side[1])
            all_segments.extend(side[2])
            all_var_offsets.extend(side[3])
            all_section_kind_ids.extend(side[4])
            all_addr_type_ids.extend(side[5])

        all_labels = [-1] * len(all_tokens)

        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
            all_segments = all_segments[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
            all_section_kind_ids = all_section_kind_ids[:self.seq_len]
            all_addr_type_ids = all_addr_type_ids[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len
            all_var_offsets += [-999999] * padding_len
            all_section_kind_ids += [0] * padding_len
            all_addr_type_ids += [0] * padding_len
            all_labels += [-1] * padding_len

        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]

        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(all_segments),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'var_offsets': torch.LongTensor(all_var_offsets),
            'section_kind_ids': torch.LongTensor(all_section_kind_ids),
            'addr_type_ids': torch.LongTensor(all_addr_type_ids),
            'pair_label': torch.tensor(int(sample.get('label', 0)), dtype=torch.long),
            'task': sample.get('task', ''),
        }
