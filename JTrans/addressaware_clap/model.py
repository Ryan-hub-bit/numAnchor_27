import torch
import torch.nn as nn

try:
    from .address_embedding import AddressAwareBERTEmbedding
    from .transformer_components import MaskedLanguageModel, TransformerBlock
except ImportError:
    from address_embedding import AddressAwareBERTEmbedding
    from transformer_components import MaskedLanguageModel, TransformerBlock


class AddressAwareBERT(nn.Module):
    """
    Local CLAP-style address-aware backbone.

    This intentionally matches the local Jtrans_AAE backbone, except that
    segment labels are allowed to carry instruction ids rather than all-zero
    token-type ids. The architectural change is therefore only the larger
    segment embedding table wired through AddressAwareBERTEmbedding.
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
        segment_types=None,
    ):
        super().__init__()

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        self.segment_types = max(16, int(segment_types if segment_types is not None else max_len + 1))

        self.embedding = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding,
            address_scale=address_scale,
            learnable_address_scale=learnable_address_scale,
            segment_types=self.segment_types,
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )
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
        mask = (token_ids > 0).unsqueeze(1).repeat(1, token_ids.size(1), 1).unsqueeze(1)
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

        for transformer in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint as grad_ckpt

                x = grad_ckpt(transformer.forward, x, mask, use_reentrant=False)
            else:
                x = transformer.forward(x, mask)

        return x


class AddressAwareJTransPretrainer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 256,
        hidden: int = 256,
        n_layers: int = 4,
        attn_heads: int = 4,
        dropout: float = 0.1,
        use_address_embedding: bool = True,
        use_var_embedding: bool = True,
        address_scale: float = 10000.0,
        learnable_address_scale: bool = False,
        segment_types=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        self.segment_types = max(16, int(segment_types if segment_types is not None else seq_len + 1))

        self.bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=seq_len,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding,
            address_scale=address_scale,
            learnable_address_scale=learnable_address_scale,
            segment_types=self.segment_types,
        )
        self.mlm_head = MaskedLanguageModel(hidden, vocab_size)
        self.jtp_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, seq_len),
        )

    def forward_mlm(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets, addr_type_ids=None):
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            addr_type_ids=addr_type_ids,
        )
        return self.mlm_head(sequence_output)

    def forward_jtp(
        self,
        token_ids,
        segment_labels,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets,
        source_idx,
        addr_type_ids=None,
    ):
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            addr_type_ids=addr_type_ids,
        )
        safe_idx = source_idx.clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, sequence_output.size(-1))
        source_hidden = sequence_output.gather(1, safe_idx).squeeze(1)
        return self.jtp_head(source_hidden)
