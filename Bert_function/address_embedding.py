"""
Address-Aware Positional Embedding

Uses sin/cos encoding on three levels of address normalization:
1. Binary-level position (bnorm): Position within entire binary
2. Function-level position (fnorm): Position within function
3. Basic block-level position (bbnorm): Position within basic block

These are added to create a rich positional encoding that captures
hierarchical structure of binary code.
"""

import torch
import torch.nn as nn
import math


class SequencePositionalEmbedding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence order (like PalmTree).
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] or [batch_size, seq_len]
        Returns:
            Positional encoding: [batch_size, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


class AddressPositionalEmbedding(nn.Module):
    """
    Address-aware positional embedding using sin/cos encoding + separate MLP projections.
    
    Uses THREE separate MLPs for different token types:
    1. address: For instruction addresses, strings, and other address-like tokens
       - Uses: binary_pos + function_pos + bb_pos
    2. daddr: For data addresses in memory operands (e.g., mov rax, [0x12345])
       - Uses: binary_pos + function_pos + bb_pos
    3. var: Stack variable offsets
       - Handled separately by VarPositionalEmbedding (has its own encoding)
    
    Each type gets sin/cos encoding on its relevant levels, then projected
    via its own dedicated MLP.
    """
    
    def __init__(
        self,
        d_model,
        max_len=512,
        intermediate_size=128,
        dropout=0.1,
        address_scale=10000.0,
        learnable_address_scale=False,
    ):
        """
        Args:
            d_model: Embedding dimension (output size)
            max_len: Maximum sequence length
            intermediate_size: Size of encoding per address level before concatenation
            dropout: Dropout rate for MLP projection
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_per_level = intermediate_size  # Dimensions per address level
        self.learnable_address_scale = learnable_address_scale
        self.use_binary_pos = True
        self.use_code_binary_pos = True
        self.use_daddr_binary_pos = True

        # THREE SEPARATE MLPs for different token types
        # address: instruction addresses, strings, etc. uses
        #          binary_pos + function_pos + bb_pos
        # daddr: data addresses (memory operands) uses
        #        binary_pos + function_pos + bb_pos
        # var: stack variables handled by separate VarPositionalEmbedding
        address_concat_size = 3 * intermediate_size
        daddr_concat_size = 3 * intermediate_size
        
        # MLP for general addresses (instructions, strings, etc.)
        self.address_projection = nn.Sequential(
            nn.Linear(address_concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # MLP for data addresses (daddr - addresses in memory operands)
        self.daddr_projection = nn.Sequential(
            nn.Linear(daddr_concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable weights to balance the position levels within each address type
        self.address_level_weights = nn.Parameter(torch.ones(3))
        self.daddr_level_weights = nn.Parameter(torch.ones(3))
        scale_tensor = torch.tensor(float(address_scale), dtype=torch.float32)
        if learnable_address_scale:
            self.address_scale = nn.Parameter(scale_tensor.clone())
            self.daddr_scale = nn.Parameter(scale_tensor.clone())
        else:
            self.register_buffer("address_scale", scale_tensor.clone())
            self.register_buffer("daddr_scale", scale_tensor.clone())
        
    def _sinusoidal_encoding(self, positions, d_model, scale):
        """
        Create sinusoidal encoding for a batch of positions.
        
        Args:
            positions: [batch_size, seq_len] normalized positions in [0, 1]
            d_model: Embedding dimension for this level
            
        Returns:
            encoding: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = positions.shape
        
        # Create dimension indices
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device) 
                            * -(math.log(10000.0) / d_model))
        
        # Scale positions to [0, 10000] range for better frequency distribution
        positions_scaled = positions.unsqueeze(-1) * torch.clamp(scale, min=1.0)  # [batch, seq, 1]
        
        # Create encoding
        encoding = torch.zeros(batch_size, seq_len, d_model, device=positions.device)
        
        # Apply sin to even indices
        encoding[:, :, 0::2] = torch.sin(positions_scaled * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term[:-1])
        
        return encoding

    def set_position_ablation(
        self,
        use_binary_pos: bool = True,
        use_code_binary_pos: bool = True,
        use_daddr_binary_pos: bool = True,
    ):
        """Toggle binary-level coordinate channels on or off."""
        self.use_binary_pos = use_binary_pos
        self.use_code_binary_pos = use_code_binary_pos
        self.use_daddr_binary_pos = use_daddr_binary_pos
    
    def forward(self, token_ids, binary_pos, function_pos, bb_pos, daddr_token_id, addr_type_ids=None):
        """
        Forward pass with address positions (automatically routes based on token IDs and positions).
        
        Routing logic:
        1. token_id == daddr_token_id → daddr MLP (binary_pos + function_pos + bb_pos)
        2. Any other token with valid positions (at least one >= 0) →
           address MLP (binary_pos + function_pos + bb_pos)
        3. Tokens without valid positions → no address encoding (zero)
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            binary_pos: [batch_size, seq_len] binary-level normalized positions (-1 for non-address tokens)
            function_pos: [batch_size, seq_len] function-level normalized positions (-1 for non-address tokens)
            bb_pos: [batch_size, seq_len] basic block-level normalized positions (-1 for non-address tokens)
            daddr_token_id: vocab ID for 'daddr' token
            
        Returns:
            embedding: [batch_size, seq_len, d_model] combined positional embedding
                      (zeros for tokens without valid positions)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Create mask for tokens with valid positions. When the binary-level
        # coordinate is disabled, only function/bb coordinates decide whether a
        # token should receive address positional information.
        if addr_type_ids is not None:
            has_valid_addr = (
                (((binary_pos >= 0) & self.use_binary_pos & self.use_code_binary_pos) |
                 (function_pos >= 0) | (bb_pos >= 0))
            ).unsqueeze(-1)
            has_valid_daddr = (
                (((binary_pos >= 0) & self.use_binary_pos & self.use_daddr_binary_pos) |
                 (function_pos >= 0) | (bb_pos >= 0))
            ).unsqueeze(-1)
            is_daddr = (addr_type_ids == 2).unsqueeze(-1) & has_valid_daddr
            is_address = (addr_type_ids == 1).unsqueeze(-1) & has_valid_addr
        else:
            # Backward-compatible fallback used by older checkpoints/pipelines.
            is_daddr = (token_ids == daddr_token_id).unsqueeze(-1)  # [batch, seq, 1]
            if self.use_binary_pos:
                has_valid_pos = ((binary_pos >= 0) | (function_pos >= 0) | (bb_pos >= 0)).unsqueeze(-1)
            else:
                has_valid_pos = ((function_pos >= 0) | (bb_pos >= 0)).unsqueeze(-1)
            is_address = has_valid_pos & (~is_daddr)  # [batch, seq, 1]
        
        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Process 'address' tokens (use binary_pos + function_pos + bb_pos)
        # Always run projection so DDP sees gradients every forward pass;
        # mask zeros out the contribution for non-address positions.
        if self.use_binary_pos and self.use_code_binary_pos:
            bin_enc_addr = self._sinusoidal_encoding(binary_pos, self.d_per_level, self.address_scale)
        else:
            bin_enc_addr = torch.zeros(batch_size, seq_len, self.d_per_level, device=device)
        func_enc = self._sinusoidal_encoding(function_pos, self.d_per_level, self.address_scale)
        bb_enc = self._sinusoidal_encoding(bb_pos, self.d_per_level, self.address_scale)
        weights = torch.softmax(self.address_level_weights, dim=0)
        bin_enc_addr = bin_enc_addr * weights[0]
        func_enc = func_enc * weights[1]
        bb_enc = bb_enc * weights[2]
        concat = torch.cat([bin_enc_addr, func_enc, bb_enc], dim=-1)
        address_emb = self.address_projection(concat)
        output = output + address_emb * is_address.float()
        
        # Process 'daddr' tokens (use binary_pos + function_pos + bb_pos)
        # Always run projection so DDP sees gradients every forward pass;
        # mask zeros out the contribution for non-daddr positions.
        if self.use_binary_pos and self.use_daddr_binary_pos:
            bin_enc = self._sinusoidal_encoding(binary_pos, self.d_per_level, self.daddr_scale)
        else:
            bin_enc = torch.zeros(batch_size, seq_len, self.d_per_level, device=device)
        func_enc2 = self._sinusoidal_encoding(function_pos, self.d_per_level, self.daddr_scale)
        bb_enc2 = self._sinusoidal_encoding(bb_pos, self.d_per_level, self.daddr_scale)
        weights2 = torch.softmax(self.daddr_level_weights, dim=0)
        bin_enc = bin_enc * weights2[0]
        func_enc2 = func_enc2 * weights2[1]
        bb_enc2 = bb_enc2 * weights2[2]
        concat2 = torch.cat([bin_enc, func_enc2, bb_enc2], dim=-1)
        daddr_emb = self.daddr_projection(concat2)
        output = output + daddr_emb * is_daddr.float()
        
        return output
        


class VarPositionalEmbedding(nn.Module):
    """
    Positional embedding for var(0xXX) tokens based on their offset values.
    
    Uses hybrid strategy: Embedding table for common offsets + MLP for distance encoding.
    This handles the huge range of possible stack offsets [-2^63, 2^63] efficiently.
    
    Strategy:
    - Common offsets [-512, 512]: Direct embedding lookup (1027 buckets: 0-1024 for offsets, 1025 for >512, 1026 for <-512)
    - Rare large offsets: Additional distance encoding via MLP on log(1 + |offset|)
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate for MLP
        """
        super().__init__()
        self.d_model = d_model
        self.sentinel = -999999  # Sentinel value for non-var tokens
        
        # Embedding table for common offsets + overflow buckets
        # Buckets: 0-1024 for offsets [-512, 512], 1025 for >512, 1026 for <-512
        self.offset_embedding = nn.Embedding(1027, d_model, padding_idx=0)
        
        # MLP for distance encoding (handles large offsets)
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, var_offsets):
        """
        Forward pass.
        
        Args:
            var_offsets: [batch_size, seq_len] variable offsets
                         -999999 (sentinel) for non-var tokens
                         Any other value for var tokens (e.g., 16, 0, -16 for var(0x10), var(0x0), var(-0x10))
            
        Returns:
            encoding: [batch_size, seq_len, d_model] hybrid encoding for var offsets
                      (zeros for non-var tokens where offset = sentinel)
        """
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        
        # Create mask for var tokens (not sentinel value)
        var_mask = (var_offsets != self.sentinel).unsqueeze(-1)  # [batch, seq, 1]
        
        # Step 1: Map offsets to token IDs
        token_ids = torch.zeros_like(var_offsets, dtype=torch.long)
        
        # Common offsets [-512, 512] -> bucket 0-1024
        in_range = (var_offsets >= -512) & (var_offsets <= 512)
        token_ids[in_range] = var_offsets[in_range] + 512
        
        # Large positive offsets -> bucket 1025
        token_ids[var_offsets > 512] = 1025
        
        # Large negative offsets -> bucket 1026
        token_ids[var_offsets < -512] = 1026
        
        # Sentinel values -> bucket 0 (will be masked out anyway)
        token_ids[var_offsets == self.sentinel] = 0
        
        # Step 2: Get embedding from table (discrete buckets)
        embedding_part = self.offset_embedding(token_ids)  # [batch, seq, d_model]
        
        # Step 3: Get distance encoding (continuous, for all offsets)
        # Use log(1 + |offset|) to compress the range
        abs_offsets = torch.abs(var_offsets.float())
        log_distance = torch.log1p(abs_offsets).unsqueeze(-1)  # [batch, seq, 1]
        distance_part = self.distance_mlp(log_distance)  # [batch, seq, d_model]
        
        # Step 4: Combine embedding + distance
        encoding = embedding_part + distance_part
        
        # Zero out encoding for non-var tokens (where var_mask is False)
        encoding = encoding * var_mask.float()
        
        return encoding
class AddressAwareBERTEmbedding(nn.Module):
    """
    Address-aware BERT Embedding.
    
    Combines embeddings (ALL trained from scratch):
    1. Token embedding
    2. Sequence positional embedding (sinusoidal)
    3. Address positional embedding (sin/cos on binary_pos, function_pos, bb_pos) - OPTIONAL
    4. Segment embedding (0=padding, 1..N=instruction IDs)
    5. Var positional embedding (sin/cos on var offsets for var(0xXX) tokens) - OPTIONAL
    """
    
    def __init__(
        self,
        vocab_size,
        embed_size,
        dropout=0.1,
        max_len=512,
        use_address_embedding=True,
        use_var_embedding=True,
        address_scale=10000.0,
        learnable_address_scale=False,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        
        # Store token IDs (will be set externally after vocab is loaded)
        self.daddr_token_id = None
        
        # 1. Token embedding - trained from scratch
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 2. Sequence positional embedding - sinusoidal (standard BERT style)
        self.position_embedding = SequencePositionalEmbedding(embed_size, max_len)
        
        # 3. Address-aware positional embedding - sin/cos encoding on 3 levels (OPTIONAL)
        if self.use_address_embedding:
            self.address_position = AddressPositionalEmbedding(
                embed_size,
                max_len,
                dropout=dropout,
                address_scale=address_scale,
                learnable_address_scale=learnable_address_scale,
            )
        else:
            self.address_position = None
        
        # 4. Segment embedding.
        # Bucket 0 is reserved for padding. Real tokens use their instruction
        # ID (1-indexed). Using max_len + 1 buckets keeps the segment IDs safe
        # for any packed sample up to seq_len tokens.
        self.segment_embedding = nn.Embedding(max_len + 1, embed_size, padding_idx=0)
        
        # 5. Var positional embedding - hybrid embedding + MLP (OPTIONAL)
        if self.use_var_embedding:
            self.var_position = VarPositionalEmbedding(embed_size, dropout=dropout)
        else:
            self.var_position = None
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def set_special_token_ids(self, daddr_token_id):
        """
        Set the vocab ID for daddr token.
        Must be called after vocabulary is loaded.
        
        Args:
            daddr_token_id: vocab ID for 'daddr' token
        """
        self.daddr_token_id = daddr_token_id

    def set_position_ablation(
        self,
        use_binary_pos: bool = True,
        use_code_binary_pos: bool = True,
        use_daddr_binary_pos: bool = True,
    ):
        """Expose fine-grained coordinate ablations to downstream tasks."""
        if self.address_position is not None:
            self.address_position.set_position_ablation(
                use_binary_pos=use_binary_pos,
                use_code_binary_pos=use_code_binary_pos,
                use_daddr_binary_pos=use_daddr_binary_pos,
            )
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None, addr_type_ids=None):
        """
        Forward pass. Automatically routes address types based on token IDs.
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            segment_labels: [batch_size, seq_len] segment labels
                (0=pad, 1..N=instruction IDs)
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            var_offsets: [batch_size, seq_len] var offset values (-999999 for non-var tokens)
            
        Returns:
            embedding: [batch_size, seq_len, embed_size]
        """
        batch_size, seq_len = token_ids.size()
        
        # 1. Get token embeddings (trained from scratch)
        token_emb = self.token_embedding(token_ids)
        
        # 2. Get sequence positional embeddings (sinusoidal)
        seq_pos_emb = self.position_embedding(token_ids)
        
        # 3. Get address positional embeddings - automatically routes by token ID and positions
        if self.use_address_embedding and self.daddr_token_id is not None:
            addr_pos_emb = self.address_position(
                token_ids,
                binary_pos,
                function_pos,
                bb_pos,
                self.daddr_token_id,
                addr_type_ids=addr_type_ids,
            )
        else:
            addr_pos_emb = 0
        
        # 4. Get var positional embeddings (separate hybrid encoding) - OPTIONAL
        if self.use_var_embedding and var_offsets is not None:
            var_pos_emb = self.var_position(var_offsets)
        else:
            var_pos_emb = 0
        
        # Combine all embeddings
        embedding = token_emb + seq_pos_emb + addr_pos_emb + self.segment_embedding(segment_labels) + var_pos_emb
        
        # Apply layer norm and dropout
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding
