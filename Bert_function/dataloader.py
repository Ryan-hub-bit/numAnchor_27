"""
Function-level masking DataLoader for Address-Aware BERT.

This variant expects one FULL FUNCTION per line, with instructions separated by
tabs. It still supports:

- IMC: mask entire instructions inside the function
- IMD: same masking for DFG-style function traces when provided
- MLM: standard token-level masking

Compared with the original instruction/walk-level pipeline, this loader keeps
the function as the sampling unit and packs as many whole instructions as fit
into ``seq_len``.
"""

import torch
from torch.utils.data import Dataset
import re
import random
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
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob
        self.instruction_mask_prob = instruction_mask_prob
        self.enable_imd = enable_imd
        
        # Special token IDs
        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)
        self.mask_idx = vocab.stoi.get('<mask>', 4)
        
        # Regex patterns for parsing inline format
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # Pattern to match daddr(0xHEX:b:f:bb) tokens for data addresses
        self.daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # Pattern to match var(0xXX) tokens
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
        
        # Load CFG data if path provided. In this function-level fork, each line
        # is expected to be one complete function encoded as tab-separated
        # instructions.
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
    
    def _load_corpus(self, path):
        """Load a function-inline corpus file."""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def _split_instructions(self, line):
        """Split one function-inline line into instruction strings."""
        return [inst.strip() for inst in line.split('\t') if inst.strip()]

    def _segment_id_for_instruction(self, inst_idx):
        """
        Segment ID = instruction index + 1.

        Bucket 0 remains reserved for padding only. We clamp to seq_len as a
        defensive guard; in practice packed functions cannot exceed this range.
        """
        return min(inst_idx + 1, self.seq_len)

    def _pack_function_instructions(self, line):
        """
        Parse and pack as many WHOLE instructions as fit into seq_len.

        We reserve one token for ``<sos>`` and one ``<eos>`` per included
        instruction. If the very first instruction is longer than the remaining
        token budget, we keep a truncated prefix of that instruction so the
        sample is still usable.
        """
        packed = []
        remaining = self.seq_len - 1  # reserve <sos>

        for inst_text in self._split_instructions(line):
            tokens, positions, var_offsets = self._parse_instruction(inst_text)
            if not tokens:
                continue

            required = len(tokens) + 1  # instruction tokens + trailing <eos>
            if required <= remaining:
                packed.append((tokens, positions, var_offsets))
                remaining -= required
                continue

            # Keep at least one truncated instruction if the function begins
            # with an unusually long instruction and nothing else has fit yet.
            if not packed and remaining > 1:
                take = remaining - 1
                if take > 0:
                    packed.append((
                        tokens[:take],
                        positions[:take],
                        var_offsets[:take],
                    ))
            break

        return packed

    def _create_empty_sample(self):
        """Return a minimal all-padding sample."""
        return {
            'bert_input': torch.full((self.seq_len,), self.pad_idx, dtype=torch.long),
            'bert_label': torch.full((self.seq_len,), -1, dtype=torch.long),
            'segment_label': torch.zeros(self.seq_len, dtype=torch.long),
            'binary_pos': torch.full((self.seq_len,), -1.0, dtype=torch.float),
            'function_pos': torch.full((self.seq_len,), -1.0, dtype=torch.float),
            'bb_pos': torch.full((self.seq_len,), -1.0, dtype=torch.float),
            'var_offsets': torch.full((self.seq_len,), -999999, dtype=torch.long),
        }

    def _build_output(self, all_tokens, all_labels, all_positions, all_segments, all_var_offsets):
        """Pad to seq_len and convert packed lists into tensors."""
        padding_len = self.seq_len - len(all_tokens)
        if padding_len > 0:
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-1] * padding_len
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len
            all_var_offsets += [-999999] * padding_len

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
        }
    
    def _parse_instruction(self, inst_text):
        """Parse a single instruction and extract tokens + positions + var offsets"""
        tokens = []
        positions = []
        var_offsets = []  # Store varaddr_pattern offset for each token (-999999 for non-var tokens, actual offset for var tokens)
        
        # Match opcode with address
        match = self.addr_pattern.match(inst_text)
        if not match:
            return [inst_text], [(-1.0, -1.0, -1.0)], [-999999]
        
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        # Code/opcode tokens now use the full three-level address encoding:
        # binary_pos + function_pos + bb_pos.
        positions.append((binary_pos, function_pos, bb_pos))
        var_offsets.append(-999999)  # Opcode is not a var, use -999999 as sentinel
        
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
                    # Code addresses (jump/call targets) also use the full
                    # three-level address encoding.
                    positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                    var_offsets.append(-999999)  # address is not a var, use -999999 as sentinel
                elif daddr_match:
                    # Data address - memory operand address
                    daddr_binary_pos = float(daddr_match.group(2))
                    daddr_function_pos = float(daddr_match.group(3))
                    daddr_bb_pos = float(daddr_match.group(4))
                    tokens.append('daddr')
                    positions.append((daddr_binary_pos, daddr_function_pos, daddr_bb_pos))
                    var_offsets.append(-999999)  # daddr is not a var, use -999999 as sentinel
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
                    tokens.append('var')  # Token is just 'var'
                    positions.append((-1.0, -1.0, -1.0))  # var has no address position, use -1 as sentinel
                    var_offsets.append(var_offset_value)  # Store the actual offset (can be 0)
                else:
                    tokens.append(operand)
                    positions.append((-1.0, -1.0, -1.0))  # non-address operand, use -1 as sentinel
                    var_offsets.append(-999999)  # Not a var, use -999999 as sentinel
        
        return tokens, positions, var_offsets
    
    def _mask_instruction(self, tokens, positions, var_offsets):
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
        # Set var_offsets to -999999 to prevent data leakage - if model sees real offset,
        # it would trivially know the token is 'var'
        masked_var_offsets = [-999999] * len(tokens)
        
        return masked_tokens, positions, masked_var_offsets, original_tokens
    
    def _mask_tokens(self, tokens, positions, var_offsets):
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
                labels.append(-1)  # Not masked
                # Keep var_offset for non-masked tokens
                masked_var_offsets.append(var_offset)
        
        return masked_tokens, masked_var_offsets, labels
    
    def _process_line_for_instruction_masking(self, line):
        """
        Process one FUNCTION line and apply instruction-level masking.

        Output format stays:
        ``<sos> inst1 <eos> inst2 <eos> ...``.
        Real instructions use their 1-indexed instruction IDs as segment labels.
        """
        instructions = self._pack_function_instructions(line)
        num_instructions = len(instructions)
        if num_instructions == 0:
            return self._create_empty_sample()

        num_to_mask = max(1, int(num_instructions * self.instruction_mask_prob))
        if num_instructions > 1:
            num_to_mask = min(num_to_mask, num_instructions - 1)

        instructions_to_mask = set(random.sample(range(num_instructions), num_to_mask))
        all_tokens = []
        all_positions = []
        all_labels = []
        all_segments = []
        all_var_offsets = []

        all_tokens.append(self.sos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-1)
        all_segments.append(1)
        all_var_offsets.append(-999999)

        for inst_idx, (tokens, positions, var_offsets) in enumerate(instructions):
            inst_segment = self._segment_id_for_instruction(inst_idx)

            if inst_idx in instructions_to_mask:
                masked_tokens, positions, var_offsets, labels = self._mask_instruction(tokens, positions, var_offsets)
            else:
                masked_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
                labels = [-1] * len(tokens)

            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([inst_segment] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)

            all_tokens.append(self.eos_idx)
            all_positions.append((-1.0, -1.0, -1.0))
            all_labels.append(-1)
            all_segments.append(inst_segment)
            all_var_offsets.append(-999999)

        return self._build_output(
            all_tokens,
            all_labels,
            all_positions,
            all_segments,
            all_var_offsets,
        )
    
    def _process_line_for_token_masking(self, line):
        """
        Process one FUNCTION line and apply token-level MLM masking.

        Segment labels also use 1-indexed instruction IDs here.
        """
        instructions = self._pack_function_instructions(line)
        if not instructions:
            return self._create_empty_sample()

        all_tokens = []
        all_positions = []
        all_labels = []
        all_segments = []
        all_var_offsets = []

        all_tokens.append(self.sos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-1)
        all_segments.append(1)
        all_var_offsets.append(-999999)

        for inst_idx, (tokens, positions, var_offsets) in enumerate(instructions):
            masked_tokens, var_offsets, labels = self._mask_tokens(tokens, positions, var_offsets)
            inst_segment = self._segment_id_for_instruction(inst_idx)

            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([inst_segment] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)

            all_tokens.append(self.eos_idx)
            all_positions.append((-1.0, -1.0, -1.0))
            all_labels.append(-1)
            all_segments.append(inst_segment)
            all_var_offsets.append(-999999)

        return self._build_output(
            all_tokens,
            all_labels,
            all_positions,
            all_segments,
            all_var_offsets,
        )
    
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
        # IMC: Process the full CFG line with instruction-level masking
        imc_data = self._process_line_for_instruction_masking(self.cfg_lines[index])
        
        # MLM: Process the full CFG line with token-level masking (independent from IMC)
        mlm_data = self._process_line_for_token_masking(self.cfg_lines[index])
        
        result = {
            'imc': imc_data,
            'mlm': mlm_data,
        }
        
        # IMD: Process the full DFG line with instruction-level masking (if enabled)
        if self.enable_imd and self.dfg_lines:
            dfg_index = index % len(self.dfg_lines)
            imd_data = self._process_line_for_instruction_masking(self.dfg_lines[dfg_index])
            result['imd'] = imd_data
        
        return result
