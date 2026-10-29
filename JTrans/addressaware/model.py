import importlib.util
import os
import sys

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PALMTREE_ROOT = os.path.join(ROOT, "PalmTree", "addressaware")
if PALMTREE_ROOT not in sys.path:
    sys.path.append(PALMTREE_ROOT)


def _load_palmtree_module(name: str, file_name: str):
    path = os.path.join(PALMTREE_ROOT, file_name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_palmtree_model = _load_palmtree_module("palmtree_model", "model.py")
_palmtree_components = _load_palmtree_module("palmtree_components", "transformer_components.py")

AddressAwareBERT = _palmtree_model.AddressAwareBERT
MaskedLanguageModel = _palmtree_components.MaskedLanguageModel


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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding

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
