"""
Address-Aware BERT Model

Modified BERT architecture that incorporates address normalization
at three hierarchical levels: binary, function, and basic block.
"""

import torch
import torch.nn as nn

try:
    from .transformer_components import (
        TransformerBlock,
        MaskedLanguageModel,
        NextSentencePrediction,
    )
    from .address_embedding import AddressAwareBERTEmbedding
except ImportError:
    from transformer_components import TransformerBlock, MaskedLanguageModel, NextSentencePrediction
    from address_embedding import AddressAwareBERTEmbedding


class AddressAwareBERT(nn.Module):
    """
    Address-aware BERT for binary code understanding.
    
    Uses hierarchical address positions (binary, function, basic block)
    in addition to standard token embeddings.
    
    Trained from scratch (no pre-trained PalmTree weights).
    """
    
    def __init__(
        self,
        vocab_size,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=512,
        use_address_embedding=True,
        use_var_embedding=True,
        address_scale=10000.0,
        learnable_address_scale=False,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            hidden: Hidden size / embedding dimension
            n_layers: Number of transformer layers
            attn_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings for var(0xXX) tokens
        """
        super().__init__()
        
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        
        # Address-aware embedding layer - ALL components trained from scratch
        self.embedding = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding,
            address_scale=address_scale,
            learnable_address_scale=learnable_address_scale,
        )
        
        # Transformer blocks - trained from scratch
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])

        # Gradient checkpointing: recompute activations on backward instead of
        # storing them, trading compute for GPU memory (~40-50% reduction).
        self.gradient_checkpointing = False

    def forward(
        self,
        token_ids,
        segment_labels,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets=None,
        section_kind_ids=None,
        addr_type_ids=None,
    ):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len] (optional, for var positional embedding)
            section_kind_ids: [batch_size, seq_len] 0 for non-daddr, 1..4 for daddr
            
        Returns:
            output: [batch_size, seq_len, hidden]
        """
        # Create attention mask (mask out padding tokens)
        mask = (token_ids > 0).unsqueeze(1).repeat(1, token_ids.size(1), 1).unsqueeze(1)
        
        # Get embeddings with address and var awareness
        # Routing (address vs daddr) is handled automatically inside AddressAwareBERTEmbedding
        x = self.embedding(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            section_kind_ids=section_kind_ids,
            addr_type_ids=addr_type_ids,
        )
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint as grad_ckpt
                x = grad_ckpt(transformer.forward, x, mask, use_reentrant=False)
            else:
                x = transformer.forward(x, mask)

        return x


class AddressAwareBERTForPretraining(nn.Module):
    """
    Address-aware BERT with MLM, IMC, IMD, dual NSP, and scope prediction heads for pretraining.
    
    Following PalmTree's approach with added Instruction Masking:
    - MLM head (for token-level CFG masking)
    - IMC head (for instruction-level CFG masking - predicts entire masked instructions)
    - IMD head (for instruction-level DFG masking - predicts entire masked instructions)
    - NSP_CFG head (for CFG order coherence)
    - NSP_DFG head (for DFG trace coherence)
    - SCOPE head (for scope prediction - 3-class classification)
    
    Each task can be enabled/disabled for ablation studies.
    """
    
    def __init__(self, bert_model, vocab_size,
                 enable_mlm=True, enable_imc=False, enable_imd=False,
                 enable_nsp_cfg=True, enable_nsp_dfg=True, enable_scope=True,
                 enable_same_bb=False, enable_same_function=False):
        """
        Args:
            bert_model: AddressAwareBERT model
            vocab_size: Vocabulary size
            enable_mlm: Enable Masked Language Modeling task (token-level)
            enable_imc: Enable Instruction Masking CFG task (instruction-level for CFG)
            enable_imd: Enable Instruction Masking DFG task (instruction-level for DFG)
            enable_nsp_cfg: Enable CFG Next Sentence Prediction task
            enable_nsp_dfg: Enable DFG Next Sentence Prediction task
            enable_scope: Enable Scope Prediction task
            enable_same_bb: Enable auxiliary same-basic-block pair classification
            enable_same_function: Enable auxiliary same-function pair classification
        """
        super().__init__()
        
        self.bert = bert_model
        self.vocab_size = vocab_size
        self.hidden = bert_model.hidden
        
        # Task flags
        self.enable_mlm = enable_mlm
        self.enable_imc = enable_imc
        self.enable_imd = enable_imd
        self.enable_nsp_cfg = enable_nsp_cfg
        self.enable_nsp_dfg = enable_nsp_dfg
        self.enable_scope = enable_scope
        self.enable_same_bb = enable_same_bb
        self.enable_same_function = enable_same_function
        
        # Masked Language Model head (token-level for CFG)
        if self.enable_mlm:
            self.MLM = MaskedLanguageModel(self.hidden, vocab_size)
        else:
            self.MLM = None
        
        # Instruction Masking CFG head (instruction-level masking for CFG)
        if self.enable_imc:
            self.IMC = MaskedLanguageModel(self.hidden, vocab_size)
        else:
            self.IMC = None
        
        # Instruction Masking DFG head (instruction-level masking for DFG)
        if self.enable_imd:
            self.IMD = MaskedLanguageModel(self.hidden, vocab_size)
        else:
            self.IMD = None
        
        # CFG Next Sentence Prediction head (order coherence)
        if self.enable_nsp_cfg:
            self.CWP = NextSentencePrediction(self.hidden)
        else:
            self.CWP = None
        
        # DFG Next Sentence Prediction head (trace coherence)
        if self.enable_nsp_dfg:
            self.DUP = NextSentencePrediction(self.hidden)
        else:
            self.DUP = None
        
        # Scope prediction head (3-class classification)
        if self.enable_scope:
            self.SCOPE = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, 3),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.SCOPE = None

        if self.enable_same_bb:
            self.SAME_BB = NextSentencePrediction(self.hidden)
        else:
            self.SAME_BB = None

        if self.enable_same_function:
            self.SAME_FUNCTION = NextSentencePrediction(self.hidden)
        else:
            self.SAME_FUNCTION = None
    
    def forward(
        self,
        token_ids,
        segment_labels,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets=None,
        section_kind_ids=None,
        addr_type_ids=None,
        corpus_type='cfg',
    ):
        """
        Forward pass for pretraining.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len] (optional, for var positional embedding)
            corpus_type: 'cfg' or 'dfg' - determines which NSP head to use
            
        Returns:
            mlm_output: [batch_size, seq_len, vocab_size] - predictions for each token (None if disabled)
            nsp_output: [batch_size, 2] - binary classification for NSP (None if disabled)
        """
        # Get BERT output (address routing is automatic inside the embedding layer)
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            section_kind_ids=section_kind_ids,
            addr_type_ids=addr_type_ids,
        )
        
        # MLM prediction for all tokens (only if enabled)
        mlm_output = self.MLM(sequence_output) if self.enable_mlm else None
        
        # Use appropriate NSP head based on corpus type (only if enabled)
        nsp_output = None
        if corpus_type == 'cfg' and self.enable_nsp_cfg:
            nsp_output = self.CWP(sequence_output)  # CWP uses [CLS] token internally
        elif corpus_type == 'dfg' and self.enable_nsp_dfg:
            nsp_output = self.DUP(sequence_output)  # DUP uses [CLS] token internally
        
        return mlm_output, nsp_output
    
    def forward_im(
        self,
        token_ids,
        segment_labels,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets=None,
        section_kind_ids=None,
        addr_type_ids=None,
        corpus_type='cfg',
    ):
        """
        Forward pass for instruction masking.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len] (optional, for var positional embedding)
            corpus_type: 'cfg' or 'dfg' - determines which IM head to use
            
        Returns:
            im_output: [batch_size, seq_len, vocab_size] - predictions for instruction-masked tokens
        """
        # Get BERT output
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            section_kind_ids=section_kind_ids,
            addr_type_ids=addr_type_ids,
        )
        
        # Use appropriate IM head based on corpus type
        im_output = None
        if corpus_type == 'cfg' and self.enable_imc:
            im_output = self.IMC(sequence_output)
        elif corpus_type == 'dfg' and self.enable_imd:
            im_output = self.IMD(sequence_output)
        
        return im_output
    
    def forward_scope(
        self,
        token_ids,
        segment_labels,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets=None,
        section_kind_ids=None,
        addr_type_ids=None,
    ):
        """
        Forward pass for scope prediction.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len] (optional, for var positional embedding)
            
        Returns:
            scope_output: [batch_size, 3] - 3-class classification for scope (None if disabled)
        """
        if not self.enable_scope:
            return None
        
        # Get BERT output
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            section_kind_ids=section_kind_ids,
            addr_type_ids=addr_type_ids,
        )
        
        # Scope prediction using [CLS] token
        cls_output = sequence_output[:, 0, :]
        scope_output = self.SCOPE(cls_output)
        
        return scope_output

    def forward_pair(
        self,
        token_ids,
        segment_labels,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets=None,
        section_kind_ids=None,
        addr_type_ids=None,
        pair_type='same_bb',
    ):
        """
        Forward pass for auxiliary structural pair classification.

        pair_type:
          - same_bb
          - same_function
        """
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            section_kind_ids=section_kind_ids,
            addr_type_ids=addr_type_ids,
        )

        if pair_type == 'same_bb':
            if not self.enable_same_bb:
                return None
            return self.SAME_BB(sequence_output)

        if pair_type == 'same_function':
            if not self.enable_same_function:
                return None
            return self.SAME_FUNCTION(sequence_output)

        raise ValueError(f"Unsupported pair_type: {pair_type}")
