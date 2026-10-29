import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import AddressAwareBERT

from funcsim_instruction_dataset import FunctionSimilarityDataset


class AddressAwarePaperFunctionModel(nn.Module):
    """
    Paper-aligned function encoder:
      full function sequence -> AddressAwareBERT -> pooler([CLS])
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        n_layers: int,
        attn_heads: int,
        seq_len: int,
        dropout: float = 0.1,
        use_address_embedding: bool = True,
        use_var_embedding: bool = True,
        freeze_bert: bool = False,
        address_scale: float = 10000.0,
        learnable_address_scale: bool = False,
        segment_types: Optional[int] = None,
    ):
        super().__init__()
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
            segment_types=segment_types if segment_types is not None else seq_len + 1,
        )
        # Match paper-style function embedding from [CLS] via a BERT-like pooler.
        self.pooler = nn.Linear(hidden, hidden)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets, addr_type_ids=None):
        sequence_output = self.bert(
            token_ids,
            segment_labels,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets,
            addr_type_ids=addr_type_ids,
        )
        cls_output = sequence_output[:, 0, :]
        return self.pooler_activation(self.pooler(self.dropout(cls_output)))

    def load_pretrained_bert(self, checkpoint_path: str) -> Dict[str, bool]:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "bert_state_dict" in ckpt:
                sd = ckpt["bert_state_dict"]
            elif "model_state_dict" in ckpt:
                full = ckpt["model_state_dict"]
                sd = {}
                for key, value in full.items():
                    if key.startswith("bert."):
                        sd[key[5:]] = value
                    elif not key.startswith("pooler."):
                        sd[key] = value
            else:
                sd = ckpt
        else:
            sd = ckpt.state_dict()

        has_address = any("address_pos" in key or "address_position" in key for key in sd)
        has_var = any("var_offset" in key or "var_position" in key for key in sd)

        current_sd = self.bert.state_dict()
        for key in list(sd.keys()):
            if key in current_sd and sd[key].shape != current_sd[key].shape:
                sd[key] = current_sd[key]

        self.bert.load_state_dict(sd, strict=False)
        return {"has_address": has_address, "has_var": has_var}


class TripletCosineMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        return F.relu(self.margin - pos_sim + neg_sim).mean()


class AddressAwareTripletDataset(Dataset):
    """
    Build paper-style (anchor, positive, negative) samples from function_blocks
    and funcsim_pairs.
    """

    def __init__(
        self,
        function_blocks_file: str,
        funcsim_pairs_file: str,
        vocab,
        seq_len: int,
        max_instructions: int,
        use_address_var: bool = True,
        seed: int = 42,
        max_anchors: Optional[int] = None,
    ):
        self.rand = random.Random(seed)
        self.base = FunctionSimilarityDataset(
            function_blocks_file=function_blocks_file,
            funcsim_pairs_file=funcsim_pairs_file,
            vocab=vocab,
            seq_len=seq_len,
            max_instructions=max_instructions,
            negative_samples=1,
        )
        self.base.use_address_var = use_address_var

        self.group_to_funcs: Dict[int, List[str]] = defaultdict(list)
        for func_id, group_id in self.base.func_to_group.items():
            self.group_to_funcs[group_id].append(func_id)

        self.group_ids = sorted(self.group_to_funcs.keys())
        self.anchor_ids: List[str] = []
        self.positives_by_anchor: Dict[str, List[str]] = {}

        for query_id, pair_data in self.base.funcsim_pairs.items():
            if query_id not in self.base.function_blocks:
                continue

            positives = [
                str(gt)
                for gt in pair_data.get("ground_truth", [])
                if str(gt) in self.base.function_blocks
            ]
            if not positives:
                query_group = self.base.func_to_group.get(query_id)
                positives = [
                    fid
                    for fid in self.group_to_funcs.get(query_group, [])
                    if fid != query_id
                ]
            positives = sorted(set(positives))
            if positives:
                self.anchor_ids.append(query_id)
                self.positives_by_anchor[query_id] = positives

        self.anchor_ids.sort()
        if max_anchors is not None:
            self.anchor_ids = self.anchor_ids[:max_anchors]

    def __len__(self):
        return len(self.anchor_ids)

    def _sample_negative(self, anchor_group: int) -> str:
        neg_group = anchor_group
        while neg_group == anchor_group:
            neg_group = self.rand.choice(self.group_ids)
        return self.rand.choice(self.group_to_funcs[neg_group])

    def _encode(self, func_id: str):
        encoded = self.base.encode_function(func_id)
        return {
            "input_ids": encoded["input_ids"],
            "segment": encoded["segment"],
            "binary_pos": encoded["binary_pos"],
            "function_pos": encoded["function_pos"],
            "bb_pos": encoded["bb_pos"],
            "var_offsets": encoded["var_offsets"],
            "addr_type_ids": encoded["addr_type_ids"],
            "func_id": func_id,
            "group_id": self.base.func_to_group[func_id],
        }

    def __getitem__(self, idx):
        anchor_id = self.anchor_ids[idx]
        pos_id = self.rand.choice(self.positives_by_anchor[anchor_id])
        anchor_group = self.base.func_to_group[anchor_id]
        neg_id = self._sample_negative(anchor_group)
        return {
            "anchor": self._encode(anchor_id),
            "positive": self._encode(pos_id),
            "negative": self._encode(neg_id),
        }


def collate_triplets(batch):
    def stack_side(name: str):
        side = [item[name] for item in batch]
        return {
            "input_ids": torch.stack([x["input_ids"] for x in side]),
            "segment": torch.stack([x["segment"] for x in side]),
            "binary_pos": torch.stack([x["binary_pos"] for x in side]),
            "function_pos": torch.stack([x["function_pos"] for x in side]),
            "bb_pos": torch.stack([x["bb_pos"] for x in side]),
            "var_offsets": torch.stack([x["var_offsets"] for x in side]),
            "addr_type_ids": torch.stack([x["addr_type_ids"] for x in side]),
            "func_ids": [x["func_id"] for x in side],
            "group_ids": torch.tensor([x["group_id"] for x in side], dtype=torch.long),
        }

    return {
        "anchor": stack_side("anchor"),
        "positive": stack_side("positive"),
        "negative": stack_side("negative"),
    }


@torch.no_grad()
def compute_embeddings(model, dataset, func_ids, device, batch_size: int):
    model.eval()
    out = {}
    for start in range(0, len(func_ids), batch_size):
        batch_ids = func_ids[start : start + batch_size]
        encoded = [dataset.base.encode_function(fid) for fid in batch_ids]
        ids_t = torch.stack([e["input_ids"] for e in encoded]).to(device)
        seg_t = torch.stack([e["segment"] for e in encoded]).to(device)
        bp_t = torch.stack([e["binary_pos"] for e in encoded]).to(device)
        fp_t = torch.stack([e["function_pos"] for e in encoded]).to(device)
        bb_t = torch.stack([e["bb_pos"] for e in encoded]).to(device)
        vo_t = torch.stack([e["var_offsets"] for e in encoded]).to(device)
        type_t = torch.stack([e["addr_type_ids"] for e in encoded]).to(device)
        embs = model(ids_t, seg_t, bp_t, fp_t, bb_t, vo_t, type_t).cpu()
        for fid, emb in zip(batch_ids, embs):
            out[fid] = F.normalize(emb, p=2, dim=0)
    return out
