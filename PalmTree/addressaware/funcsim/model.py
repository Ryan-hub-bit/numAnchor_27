"""
Function similarity model aligned with the current address-aware pretrain.

The model groups consecutive instructions into small BERT windows, encodes each
window with the pretrained AddressAwareBERT backbone, mean-pools window
representations, and finally projects them into a retrieval embedding space.
"""

import torch
import torch.distributed as td
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from ..model import AddressAwareBERT
except ImportError:  # pragma: no cover - script mode fallback
    from model import AddressAwareBERT


class FunctionSimilarityModel(nn.Module):
    """
    Contrastive function-similarity model with per-instruction BERT encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 768,
        n_layers: int = 12,
        attn_heads: int = 12,
        dropout: float = 0.1,
        max_len: int = 32,
        window_size: int = 32,
        instrs_per_window: int = 2,
        sos_idx: int = 3,
        eos_idx: int = 2,
        pad_idx: int = 0,
        use_address_embedding: bool = True,
        use_var_embedding: bool = True,
        embedding_dim: int = 256,
        freeze_bert: bool = False,
        gradient_checkpointing: bool = False,
        window_subbatch: int = 256,
    ):
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.window_size = window_size
        self.instrs_per_window = instrs_per_window
        self.window_subbatch = window_subbatch
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding,
        )

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )

    def _bert_call(self, ids, seg, bp, fp, bb, vo, sk, at):
        return self.bert(ids, seg, bp, fp, bb, vo, sk, at)

    def _run_bert_windows(self, ids, seg, bp, fp, bb, vo, sk, at):
        """
        Run BERT over a [N, window_size] batch of windows in sub-batches.
        """
        num_windows = ids.size(0)
        chunks = []
        for start in range(0, num_windows, self.window_subbatch):
            end = min(start + self.window_subbatch, num_windows)
            s_ids = ids[start:end]
            s_seg = seg[start:end]
            s_bp = bp[start:end]
            s_fp = fp[start:end]
            s_bb = bb[start:end]
            s_vo = vo[start:end]
            s_sk = sk[start:end]
            s_at = at[start:end]
            if self.gradient_checkpointing and self.training:
                out = grad_checkpoint(
                    self._bert_call,
                    s_ids,
                    s_seg,
                    s_bp,
                    s_fp,
                    s_bb,
                    s_vo,
                    s_sk,
                    s_at,
                    use_reentrant=False,
                )
            else:
                out = self.bert(s_ids, s_seg, s_bp, s_fp, s_bb, s_vo, s_sk, s_at)
            chunks.append(out)
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        token_ids,
        segment,
        binary_pos,
        function_pos,
        bb_pos,
        var_offsets=None,
        section_kind_ids=None,
        addr_type_ids=None,
        num_instructions=None,
        instruction_boundaries=None,
    ):
        device = token_ids.device
        batch_size = token_ids.size(0)
        window_size = self.window_size
        instrs_per_window = self.instrs_per_window

        all_w_ids = []
        all_w_seg = []
        all_w_bp = []
        all_w_fp = []
        all_w_bb = []
        all_w_vo = []
        all_w_sk = []
        all_w_at = []
        win_to_func = []

        sos_id = torch.tensor([self.sos_idx], dtype=torch.long, device=device)
        eos_id = torch.tensor([self.eos_idx], dtype=torch.long, device=device)
        neg1f = torch.tensor([-1.0], dtype=torch.float, device=device)
        neg1l = torch.tensor([-999999], dtype=torch.long, device=device)
        zero1l = torch.tensor([0], dtype=torch.long, device=device)

        for batch_index in range(batch_size):
            boundaries = instruction_boundaries[batch_index]
            n_inst = num_instructions[batch_index]
            added = 0

            for win_start in range(0, n_inst, instrs_per_window):
                win_end = min(win_start + instrs_per_window, n_inst)
                n_instr_in_win = win_end - win_start
                overhead = 1 + n_instr_in_win
                max_real = window_size - overhead

                per_instr_ids = []
                per_instr_bp = []
                per_instr_fp = []
                per_instr_bb = []
                per_instr_vo = []
                per_instr_sk = []
                per_instr_at = []

                for i in range(win_start, win_end):
                    tok_s = boundaries[i]
                    tok_e = boundaries[i + 1] if i + 1 < n_inst else token_ids.size(1)
                    while tok_e > tok_s and token_ids[batch_index, tok_e - 1] == self.pad_idx:
                        tok_e -= 1

                    per_instr_ids.append(token_ids[batch_index, tok_s:tok_e])
                    per_instr_bp.append(binary_pos[batch_index, tok_s:tok_e])
                    per_instr_fp.append(function_pos[batch_index, tok_s:tok_e])
                    per_instr_bb.append(bb_pos[batch_index, tok_s:tok_e])

                    if var_offsets is not None:
                        per_instr_vo.append(var_offsets[batch_index, tok_s:tok_e])
                    else:
                        per_instr_vo.append(
                            torch.full((tok_e - tok_s,), -999999, dtype=torch.long, device=device)
                        )

                    if section_kind_ids is not None:
                        per_instr_sk.append(section_kind_ids[batch_index, tok_s:tok_e])
                    else:
                        per_instr_sk.append(
                            torch.zeros(tok_e - tok_s, dtype=torch.long, device=device)
                        )

                    if addr_type_ids is not None:
                        per_instr_at.append(addr_type_ids[batch_index, tok_s:tok_e])
                    else:
                        per_instr_at.append(
                            torch.zeros(tok_e - tok_s, dtype=torch.long, device=device)
                        )

                total_real = torch.cat(per_instr_ids)[:max_real].size(0)
                instr_lens_raw = [tokens.size(0) for tokens in per_instr_ids]
                instr_lens = []
                budget = max_real
                for length in instr_lens_raw:
                    take = min(length, budget)
                    instr_lens.append(take)
                    budget -= take

                parts_ids = [sos_id]
                parts_bp = [neg1f]
                parts_fp = [neg1f]
                parts_bb = [neg1f]
                parts_vo = [neg1l]
                parts_sk = [zero1l]
                parts_at = [zero1l]
                parts_seg = [torch.tensor([0], dtype=torch.long, device=device)]

                for local_idx in range(n_instr_in_win):
                    take = instr_lens[local_idx]
                    if take > 0:
                        parts_ids.append(per_instr_ids[local_idx][:take])
                        parts_bp.append(per_instr_bp[local_idx][:take])
                        parts_fp.append(per_instr_fp[local_idx][:take])
                        parts_bb.append(per_instr_bb[local_idx][:take])
                        parts_vo.append(per_instr_vo[local_idx][:take])
                        parts_sk.append(per_instr_sk[local_idx][:take])
                        parts_at.append(per_instr_at[local_idx][:take])

                    parts_ids.append(eos_id)
                    parts_bp.append(neg1f)
                    parts_fp.append(neg1f)
                    parts_bb.append(neg1f)
                    parts_vo.append(neg1l)
                    parts_sk.append(zero1l)
                    parts_at.append(zero1l)

                    segment_id = local_idx % 2
                    seg_len = take + 1
                    parts_seg.append(
                        torch.full((seg_len,), segment_id, dtype=torch.long, device=device)
                    )

                used = 1 + total_real + n_instr_in_win
                pad_len = window_size - used
                if pad_len > 0:
                    parts_ids.append(torch.full((pad_len,), self.pad_idx, dtype=torch.long, device=device))
                    parts_bp.append(torch.full((pad_len,), -1.0, device=device))
                    parts_fp.append(torch.full((pad_len,), -1.0, device=device))
                    parts_bb.append(torch.full((pad_len,), -1.0, device=device))
                    parts_vo.append(
                        torch.full((pad_len,), -999999, dtype=torch.long, device=device)
                    )
                    parts_sk.append(torch.zeros(pad_len, dtype=torch.long, device=device))
                    parts_at.append(torch.zeros(pad_len, dtype=torch.long, device=device))
                    parts_seg.append(torch.zeros(pad_len, dtype=torch.long, device=device))

                all_w_ids.append(torch.cat(parts_ids))
                all_w_seg.append(torch.cat(parts_seg))
                all_w_bp.append(torch.cat(parts_bp))
                all_w_fp.append(torch.cat(parts_fp))
                all_w_bb.append(torch.cat(parts_bb))
                all_w_vo.append(torch.cat(parts_vo))
                all_w_sk.append(torch.cat(parts_sk))
                all_w_at.append(torch.cat(parts_at))
                win_to_func.append(batch_index)
                added += 1

            if added == 0:
                dummy_ids = torch.full((window_size,), self.pad_idx, dtype=torch.long, device=device)
                dummy_ids[0] = self.sos_idx
                dummy_ids[1] = self.eos_idx
                all_w_ids.append(dummy_ids)
                all_w_seg.append(torch.zeros(window_size, dtype=torch.long, device=device))
                all_w_bp.append(torch.full((window_size,), -1.0, device=device))
                all_w_fp.append(torch.full((window_size,), -1.0, device=device))
                all_w_bb.append(torch.full((window_size,), -1.0, device=device))
                all_w_vo.append(torch.full((window_size,), -999999, dtype=torch.long, device=device))
                all_w_sk.append(torch.zeros(window_size, dtype=torch.long, device=device))
                all_w_at.append(torch.zeros(window_size, dtype=torch.long, device=device))
                win_to_func.append(batch_index)

        t_ids = torch.stack(all_w_ids)
        t_seg = torch.stack(all_w_seg)
        t_bp = torch.stack(all_w_bp)
        t_fp = torch.stack(all_w_fp)
        t_bb = torch.stack(all_w_bb)
        t_vo = torch.stack(all_w_vo)
        t_sk = torch.stack(all_w_sk)
        t_at = torch.stack(all_w_at)

        bert_out = self._run_bert_windows(t_ids, t_seg, t_bp, t_fp, t_bb, t_vo, t_sk, t_at)

        real_mask = (t_ids != self.pad_idx).float().unsqueeze(-1)
        win_emb = (bert_out * real_mask).sum(dim=1) / (real_mask.sum(dim=1) + 1e-9)

        func_idx = torch.tensor(win_to_func, dtype=torch.long, device=device)
        hidden = win_emb.size(-1)
        func_sum = torch.zeros(batch_size, hidden, device=device)
        func_cnt = torch.zeros(batch_size, 1, device=device)
        func_sum.scatter_add_(0, func_idx.unsqueeze(1).expand(-1, hidden), win_emb)
        func_cnt.scatter_add_(
            0,
            func_idx.unsqueeze(1),
            torch.ones(len(all_w_ids), 1, device=device),
        )
        func_emb = func_sum / (func_cnt + 1e-9)

        emb = self.projection(func_emb)
        return F.normalize(emb, p=2, dim=1)

    def load_pretrained_bert(self, checkpoint_path: str) -> dict:
        """
        Load pre-trained BERT weights from a pretrain checkpoint.
        """
        print(f"[Model] Loading pre-trained BERT from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict):
            if 'bert_state_dict' in ckpt:
                state_dict = ckpt['bert_state_dict']
            elif 'model_state_dict' in ckpt:
                full = ckpt['model_state_dict']
                state_dict = {}
                for key, value in full.items():
                    if key.startswith('bert.'):
                        state_dict[key[5:]] = value
                    elif not key.startswith('MLM.') and not key.startswith('NSP.') and not key.startswith('projection.'):
                        state_dict[key] = value
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt.state_dict()

        has_address = any('address_pos' in key or 'address_position' in key for key in state_dict)
        has_var = any('var_offset' in key or 'var_position' in key for key in state_dict)
        print(f"[Model] Checkpoint: has_address={has_address}, has_var={has_var}")

        current_sd = self.bert.state_dict()
        for key in list(state_dict.keys()):
            if key in current_sd and state_dict[key].shape != current_sd[key].shape:
                print(
                    f"[Model] Resizing '{key}': "
                    f"ckpt{list(state_dict[key].shape)} -> model{list(current_sd[key].shape)}"
                )
                state_dict[key] = current_sd[key]

        missing, unexpected = self.bert.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Model] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(
                f"[Model] Unexpected keys ({len(unexpected)}): "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        print("[Model] Pre-trained BERT loaded successfully")

        return {'has_address': has_address, 'has_var': has_var}


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels, **_ignored):
        sim = F.cosine_similarity(emb1, emb2, dim=1)
        dist = 1.0 - sim
        pos_loss = labels.float() * dist.pow(2)
        neg_loss = (1 - labels.float()) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _all_gather_with_grad(tensor):
        world_size = td.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        td.all_gather(gathered, tensor.contiguous())
        gathered[td.get_rank()] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _all_gather_no_grad(tensor):
        world_size = td.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        td.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    def forward(
        self,
        emb1,
        emb2,
        labels=None,
        group_ids=None,
        func1_hash_group_ids=None,
        func2_hash_group_ids=None,
    ):
        distributed_training = td.is_available() and td.is_initialized() and self.training

        if distributed_training:
            emb1_all = self._all_gather_with_grad(emb1)
            emb2_all = self._all_gather_with_grad(emb2)
            group_ids_all = self._all_gather_no_grad(group_ids) if group_ids is not None else None
            f1_hash_all = self._all_gather_no_grad(func1_hash_group_ids) if func1_hash_group_ids is not None else None
            f2_hash_all = self._all_gather_no_grad(func2_hash_group_ids) if func2_hash_group_ids is not None else None
        else:
            emb1_all = emb1
            emb2_all = emb2
            group_ids_all = group_ids
            f1_hash_all = func1_hash_group_ids
            f2_hash_all = func2_hash_group_ids

        logits_12 = torch.mm(emb1, emb2_all.t()) / self.temperature
        logits_21 = torch.mm(emb2, emb1_all.t()) / self.temperature

        local_batch = emb1.size(0)
        if distributed_training:
            rank = td.get_rank()
            offset = rank * local_batch
            targets = torch.arange(local_batch, device=emb1.device) + offset
        else:
            targets = torch.arange(local_batch, device=emb1.device)

        if group_ids is not None and group_ids_all is not None:
            same_group = group_ids.unsqueeze(1) == group_ids_all.unsqueeze(0)
            if distributed_training:
                global_idx = torch.arange(group_ids_all.size(0), device=group_ids.device)
                local_idx = torch.arange(local_batch, device=group_ids.device) + offset
                diag = local_idx.unsqueeze(1) == global_idx.unsqueeze(0)
            else:
                diag = torch.eye(local_batch, dtype=torch.bool, device=group_ids.device)
            logits_12 = logits_12.masked_fill(same_group & ~diag, float('-inf'))
            logits_21 = logits_21.masked_fill(same_group & ~diag, float('-inf'))

        if f1_hash_all is not None and f2_hash_all is not None:
            same_hash_12 = func1_hash_group_ids.unsqueeze(1) == f2_hash_all.unsqueeze(0)
            same_hash_21 = func2_hash_group_ids.unsqueeze(1) == f1_hash_all.unsqueeze(0)
            if distributed_training:
                global_idx = torch.arange(f2_hash_all.size(0), device=emb1.device)
                local_idx = torch.arange(local_batch, device=emb1.device) + offset
                diag_12 = local_idx.unsqueeze(1) == global_idx.unsqueeze(0)
                global_idx_21 = torch.arange(f1_hash_all.size(0), device=emb1.device)
                diag_21 = local_idx.unsqueeze(1) == global_idx_21.unsqueeze(0)
            else:
                diag_12 = torch.eye(local_batch, dtype=torch.bool, device=emb1.device)
                diag_21 = diag_12
            logits_12 = logits_12.masked_fill(same_hash_12 & ~diag_12, float('-inf'))
            logits_21 = logits_21.masked_fill(same_hash_21 & ~diag_21, float('-inf'))

        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)
        return 0.5 * (loss_12 + loss_21)
