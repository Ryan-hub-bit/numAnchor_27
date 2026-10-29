"""
Function Similarity Model

    Supports two downstream encoder modes:

      1. legacy `window` mode
         The original PalmTree-style instruction-pair window encoder used by
         earlier seq_len=20 pretraining runs.

      2. `fullseq` mode
         Directly encodes the whole function-level sequence produced by the new
         Bert_function pretraining pipeline:
             <sos> inst0 <eos> inst1 <eos> ...

    Memory optimisation:
      gradient_checkpointing=True recomputes BERT activations on backward pass.
"""

import os
import sys
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as td
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ── import AddressAwareBERT from parent PalmTree directory ───────────────────
_parent = os.path.join(os.path.dirname(__file__), '..')
if _parent not in sys.path:
    sys.path.insert(0, _parent)

_spec = importlib.util.spec_from_file_location(
    "palmtree_model", os.path.join(_parent, "model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AddressAwareBERT = _mod.AddressAwareBERT
# ─────────────────────────────────────────────────────────────────────────────


class FunctionSimilarityModel(nn.Module):
    """
    Contrastive function-similarity model with per-instruction BERT encoding.

    Forward signature:
        token_ids, segment, binary_pos, function_pos, bb_pos, var_offsets,
        num_instructions, instruction_boundaries
            -> [B, embedding_dim]  L2-normalised embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 768,
        n_layers: int = 12,
        attn_heads: int = 12,
        dropout: float = 0.1,
        max_len: int = 20,              # MUST match pretrain checkpoint (pe [1,20,768])
        window_size: int = 20,          # BERT input length = pretrain seq_len
        instrs_per_window: int = 2,     # group 2 instructions per BERT call
        sos_idx: int = 3,               # <sos> token id
        eos_idx: int = 2,               # <eos> token id
        pad_idx: int = 0,               # <pad> token id
        use_address_embedding: bool = True,
        use_var_embedding: bool = True,
        embedding_dim: int = 256,
        freeze_bert: bool = False,
        gradient_checkpointing: bool = False,
        window_subbatch: int = 256,     # max windows per BERT call (memory budget)
        encoder_mode: str = "window",
    ):
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.window_size       = window_size
        self.instrs_per_window = instrs_per_window
        self.window_subbatch   = window_subbatch
        self.sos_idx           = sos_idx
        self.eos_idx           = eos_idx
        self.pad_idx           = pad_idx
        if encoder_mode not in {"window", "fullseq"}:
            raise ValueError(f"Unsupported encoder_mode: {encoder_mode}")
        self.encoder_mode = encoder_mode

        # BERT encoder. In fullseq mode max_len should match the function-level
        # pretrain checkpoint (e.g. 512); in legacy mode it stays at 20.
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

        # Projection head: hidden -> embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bert_call(self, ids, seg, bp, fp, bb, vo):
        """Single BERT forward, used as the unit for grad_checkpoint."""
        return self.bert(ids, seg, bp, fp, bb, vo)

    def _run_bert_windows(self, ids, seg, bp, fp, bb, vo):
        """
        Run BERT over a [N, window_size] batch of windows in sub-batches.
        Returns token-level outputs: [N, window_size, hidden]
        """
        N = ids.size(0)
        chunks = []
        for start in range(0, N, self.window_subbatch):
            end   = min(start + self.window_subbatch, N)
            s_ids = ids[start:end]; s_seg = seg[start:end]
            s_bp  = bp[start:end];  s_fp  = fp[start:end]
            s_bb  = bb[start:end];  s_vo  = vo[start:end]
            if self.gradient_checkpointing and self.training:
                out = grad_checkpoint(
                    self._bert_call,
                    s_ids, s_seg, s_bp, s_fp, s_bb, s_vo,
                    use_reentrant=False,
                )
            else:
                out = self.bert(s_ids, s_seg, s_bp, s_fp, s_bb, s_vo)
            chunks.append(out)
        return torch.cat(chunks, dim=0)   # [N, window_size, hidden]

    def _run_bert_fullseq(self, ids, seg, bp, fp, bb, vo):
        """
        Run BERT over a [N, seq_len] batch of whole-function inputs in
        sub-batches so we can keep GPU memory predictable.
        """
        N = ids.size(0)
        chunks = []
        subbatch = max(1, self.window_subbatch)
        for start in range(0, N, subbatch):
            end = min(start + subbatch, N)
            s_ids = ids[start:end]; s_seg = seg[start:end]
            s_bp = bp[start:end]; s_fp = fp[start:end]
            s_bb = bb[start:end]; s_vo = vo[start:end]
            if self.gradient_checkpointing and self.training:
                out = grad_checkpoint(
                    self._bert_call,
                    s_ids, s_seg, s_bp, s_fp, s_bb, s_vo,
                    use_reentrant=False,
                )
            else:
                out = self.bert(s_ids, s_seg, s_bp, s_fp, s_bb, s_vo)
            chunks.append(out)
        return torch.cat(chunks, dim=0)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids,            # [B, seq_len]
        segment,              # [B, seq_len]  — unused, rebuilt per window
        binary_pos,           # [B, seq_len]
        function_pos,         # [B, seq_len]
        bb_pos,               # [B, seq_len]
        var_offsets=None,     # [B, seq_len]
        num_instructions=None,         # list[int] length B
        instruction_boundaries=None,   # list[list[int]] length B
    ):
        """
        Encode a batch of functions into L2-normalised embeddings.

        Instruction-pair window strategy (matches pretrain IMC format):
          Group every INSTRS_PER_WINDOW=2 consecutive instructions into one window:
            <sos>  instr0_tokens  <eos>  instr1_tokens  <eos>  <pad>...
            (≤ window_size=20)
            segment = 0 for <sos>+instr0+<eos>, 1 for instr1+<eos>, 0 for <pad>
          The per-instruction <eos> separator and alternating segments match the
          pretrained IMC format exactly.

        Returns:
            embeddings : [B, embedding_dim]
        """
        device = token_ids.device
        B      = token_ids.size(0)

        if self.encoder_mode == "fullseq":
            bert_out = self._run_bert_fullseq(
                token_ids,
                segment,
                binary_pos,
                function_pos,
                bb_pos,
                var_offsets,
            )
            real_mask = (token_ids != self.pad_idx).float().unsqueeze(-1)
            func_emb = (bert_out * real_mask).sum(dim=1) / (real_mask.sum(dim=1) + 1e-9)
            emb = self.projection(func_emb)
            return F.normalize(emb, p=2, dim=1)

        W      = self.window_size        # 20
        IPW    = self.instrs_per_window  # 2

        all_w_ids = []
        all_w_seg = []
        all_w_bp  = []
        all_w_fp  = []
        all_w_bb  = []
        all_w_vo  = []
        win_to_func = []

        sos_id  = torch.tensor([self.sos_idx], dtype=torch.long,  device=device)
        eos_id  = torch.tensor([self.eos_idx], dtype=torch.long,  device=device)
        neg1f   = torch.tensor([-1.0],         dtype=torch.float, device=device)
        neg1l   = torch.tensor([-999999],      dtype=torch.long,  device=device)

        for b in range(B):
            bounds = instruction_boundaries[b]   # list[int], start pos of each instr
            n_inst = num_instructions[b]

            added = 0
            # step through instructions in groups of IPW=2
            for win_start in range(0, n_inst, IPW):
                win_end = min(win_start + IPW, n_inst)  # exclusive
                n_instr_in_win = win_end - win_start

                # overhead: 1 <sos> + n_instr_in_win <eos> tokens
                overhead = 1 + n_instr_in_win
                max_real = W - overhead  # real tokens budget

                # collect tokens for each instruction separately
                per_instr_ids = []
                per_instr_bp  = []
                per_instr_fp  = []
                per_instr_bb  = []
                per_instr_vo  = []

                for i in range(win_start, win_end):
                    tok_s = bounds[i]
                    tok_e = bounds[i + 1] if i + 1 < n_inst else token_ids.size(1)
                    # don't go into padding
                    while tok_e > tok_s and token_ids[b, tok_e - 1] == self.pad_idx:
                        tok_e -= 1

                    per_instr_ids.append(token_ids[b, tok_s:tok_e])
                    per_instr_bp.append(binary_pos[b, tok_s:tok_e])
                    per_instr_fp.append(function_pos[b, tok_s:tok_e])
                    per_instr_bb.append(bb_pos[b, tok_s:tok_e])
                    if var_offsets is not None:
                        per_instr_vo.append(var_offsets[b, tok_s:tok_e])
                    else:
                        per_instr_vo.append(torch.full((tok_e - tok_s,), -999999,
                                                       dtype=torch.long, device=device))

                # truncate total real tokens to max_real
                all_real_ids = torch.cat(per_instr_ids)[:max_real]
                total_real = all_real_ids.size(0)

                # figure out how many tokens each instruction contributes after truncation
                instr_lens_raw = [t.size(0) for t in per_instr_ids]
                instr_lens = []
                budget = max_real
                for l in instr_lens_raw:
                    take = min(l, budget)
                    instr_lens.append(take)
                    budget -= take

                # build: <sos> instr0_tokens <eos> instr1_tokens <eos> ... <pad>
                parts_ids = [sos_id]
                parts_bp  = [neg1f]
                parts_fp  = [neg1f]
                parts_bb  = [neg1f]
                parts_vo  = [neg1l]
                parts_seg = []

                # <sos> gets segment 0
                parts_seg.append(torch.tensor([0], dtype=torch.long, device=device))

                offset = 0
                for local_idx, i in enumerate(range(win_start, win_end)):
                    take = instr_lens[local_idx]
                    if take > 0:
                        parts_ids.append(per_instr_ids[local_idx][:take])
                        parts_bp.append(per_instr_bp[local_idx][:take])
                        parts_fp.append(per_instr_fp[local_idx][:take])
                        parts_bb.append(per_instr_bb[local_idx][:take])
                        parts_vo.append(per_instr_vo[local_idx][:take])

                    # <eos> for this instruction
                    parts_ids.append(eos_id)
                    parts_bp.append(neg1f)
                    parts_fp.append(neg1f)
                    parts_bb.append(neg1f)
                    parts_vo.append(neg1l)

                    # segment: alternating 0/1 based on local_idx
                    inst_segment = local_idx % 2  # 0 for first instr, 1 for second
                    seg_len = take + 1  # instruction tokens + <eos>
                    parts_seg.append(torch.full((seg_len,), inst_segment,
                                                dtype=torch.long, device=device))

                # padding
                used = 1 + total_real + n_instr_in_win  # sos + real + eos*n_instr
                pad_len = W - used
                if pad_len > 0:
                    parts_ids.append(torch.full((pad_len,), self.pad_idx,
                                                dtype=torch.long, device=device))
                    parts_bp.append(torch.full((pad_len,), -1.0, device=device))
                    parts_fp.append(torch.full((pad_len,), -1.0, device=device))
                    parts_bb.append(torch.full((pad_len,), -1.0, device=device))
                    parts_vo.append(torch.full((pad_len,), -999999,
                                               dtype=torch.long, device=device))
                    parts_seg.append(torch.zeros(pad_len, dtype=torch.long, device=device))

                w_ids = torch.cat(parts_ids)
                w_seg = torch.cat(parts_seg)
                w_bp  = torch.cat(parts_bp)
                w_fp  = torch.cat(parts_fp)
                w_bb  = torch.cat(parts_bb)
                w_vo  = torch.cat(parts_vo)

                all_w_ids.append(w_ids)
                all_w_seg.append(w_seg)
                all_w_bp.append(w_bp)
                all_w_fp.append(w_fp)
                all_w_bb.append(w_bb)
                all_w_vo.append(w_vo)
                win_to_func.append(b)
                added += 1

            # degenerate: no instructions → one dummy window
            # Format: <sos> <eos> <pad>... — segment 0 throughout
            if added == 0:
                dummy_ids = torch.full((W,), self.pad_idx, dtype=torch.long, device=device)
                dummy_ids[0] = self.sos_idx; dummy_ids[1] = self.eos_idx
                all_w_ids.append(dummy_ids)
                all_w_seg.append(torch.zeros(W, dtype=torch.long, device=device))
                all_w_bp.append(torch.full((W,), -1.0, device=device))
                all_w_fp.append(torch.full((W,), -1.0, device=device))
                all_w_bb.append(torch.full((W,), -1.0, device=device))
                all_w_vo.append(torch.full((W,), -999999, dtype=torch.long, device=device))
                win_to_func.append(b)

        total = len(all_w_ids)

        # stack → [total, W]
        t_ids = torch.stack(all_w_ids)
        t_seg = torch.stack(all_w_seg)
        t_bp  = torch.stack(all_w_bp)
        t_fp  = torch.stack(all_w_fp)
        t_bb  = torch.stack(all_w_bb)
        t_vo  = torch.stack(all_w_vo)

        # ── BERT over all windows ─────────────────────────────────────────────
        bert_out = self._run_bert_windows(t_ids, t_seg, t_bp, t_fp, t_bb, t_vo)
        # bert_out: [total, W, hidden]

        # ── window embeddings: mean-pool non-padding tokens ───────────────────
        real_mask = (t_ids != self.pad_idx).float().unsqueeze(-1)   # [total, W, 1]
        win_emb   = (bert_out * real_mask).sum(dim=1) \
                  / (real_mask.sum(dim=1) + 1e-9)                   # [total, hidden]

        # ── function embeddings: scatter mean over windows ────────────────────
        func_idx = torch.tensor(win_to_func, dtype=torch.long, device=device)
        hidden   = win_emb.size(-1)
        func_sum = torch.zeros(B, hidden, device=device)
        func_cnt = torch.zeros(B, 1,      device=device)
        func_sum.scatter_add_(0, func_idx.unsqueeze(1).expand(-1, hidden), win_emb)
        func_cnt.scatter_add_(0, func_idx.unsqueeze(1),
                              torch.ones(total, 1, device=device))
        func_emb = func_sum / (func_cnt + 1e-9)                     # [B, hidden]

        # ── project and L2-normalise ──────────────────────────────────────────
        emb = self.projection(func_emb)                              # [B, embedding_dim]
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    # ── helpers ───────────────────────────────────────────────────────────────

    def load_pretrained_bert(self, checkpoint_path: str) -> dict:
        """
        Load pre-trained BERT weights from a training checkpoint.

        Handles three checkpoint formats:
          1. {'bert_state_dict': {...}}
          2. {'model_state_dict': {...}}   (full model, extracts bert.* keys)
          3. raw state dict

        Any key whose shape in the checkpoint differs from the current model
        is replaced by the model's own buffer (safe for the sinusoidal pe).

        Returns:
            {'has_address': bool, 'has_var': bool}
        """
        print(f"[Model] Loading pre-trained BERT from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict):
            if 'bert_state_dict' in ckpt:
                sd = ckpt['bert_state_dict']
            elif 'model_state_dict' in ckpt:
                full = ckpt['model_state_dict']
                sd = {}
                for k, v in full.items():
                    if k.startswith('bert.'):
                        sd[k[5:]] = v
                    elif not k.startswith('MLM.') and not k.startswith('NSP.') \
                            and not k.startswith('projection.'):
                        sd[k] = v
            else:
                sd = ckpt
        else:
            sd = ckpt.state_dict()

        has_address = any('address_pos' in k or 'address_position' in k
                          for k in sd)
        has_var     = any('var_offset'  in k or 'var_position'     in k
                          for k in sd)

        print(f"[Model] Checkpoint: has_address={has_address}, has_var={has_var}")

        # Replace any size-mismatched key with the model's own value.
        # The sinusoidal position buffer (pe) is deterministic given d_model, so
        # swapping it is lossless.  max_len=20 here == checkpoint max_len=20,
        # so there should be no mismatch, but we keep this as a safety net.
        current_sd = self.bert.state_dict()
        for k in list(sd.keys()):
            if k in current_sd and sd[k].shape != current_sd[k].shape:
                print(f"[Model] Resizing '{k}': "
                      f"ckpt{list(sd[k].shape)} -> model{list(current_sd[k].shape)}")
                sd[k] = current_sd[k]

        missing, unexpected = self.bert.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Model] Missing keys  ({len(missing)}): "
                  f"{missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[Model] Unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print("[Model] Pre-trained BERT loaded successfully")

        return {'has_address': has_address, 'has_var': has_var}


# ── Loss functions ─────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (cosine distance variant).

    For positive pairs (label=1): minimise cosine distance.
    For negative pairs (label=0): push cosine distance above `margin`.

    Uses cosine distance = 1 - cosine_similarity (range [0, 2]).

    NOTE: margin should be set relative to the [0,2] distance range.
    A margin of 1.0 (cosine_similarity = 0 boundary) is a good default,
    as random high-dim L2-normalised vectors already have dist ≈ 1.0,
    meaning a margin of 0.5 would give zero gradient on all negatives
    from the very first step.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels, **_ignored):
        """
        Args:
            emb1, emb2 : [B, D]  L2-normalised embeddings
            labels     : [B]     1 = positive pair, 0 = negative pair
        Returns:
            scalar loss
        """
        sim  = F.cosine_similarity(emb1, emb2, dim=1)   # [-1, 1]
        dist = 1.0 - sim                                  # [0, 2]

        pos_loss = labels.float()       * dist.pow(2)
        neg_loss = (1 - labels.float()) * F.relu(self.margin - dist).pow(2)

        return (pos_loss + neg_loss).mean()


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) loss for function similarity.

    Designed for **all-positive batches**: every (emb1[i], emb2[i]) is a
    genuine positive pair.  Other rows in the batch serve as in-batch
    negatives automatically — no explicit negative sampling needed.

    DDP support (controlled by self.training):
      During training, when torch.distributed is initialised, embeddings are
      gathered across all GPUs with all_gather so every rank sees the full set
      of in-batch negatives (standard SimCLR / CLIP approach).
      During evaluation (.eval() mode), all_gather is skipped so that
      validation can run on rank 0 alone without NCCL deadlocks.

    Args:
        temperature : logit scale (lower = sharper, 0.07 is standard)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    # ── DDP helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _all_gather_with_grad(tensor):
        """Gather tensors from all DDP ranks, preserving local gradients."""
        world_size = td.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        td.all_gather(gathered, tensor.contiguous())
        # Replace own shard with the original tensor so gradients flow back
        gathered[td.get_rank()] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _all_gather_no_grad(tensor):
        """Gather integer tensors (group_ids) across ranks — no grad needed."""
        world_size = td.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        td.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        emb1,
        emb2,
        labels=None,
        group_ids=None,
        func1_hash_group_ids=None,
        func2_hash_group_ids=None,
    ):
        """
        Args:
            emb1, emb2 : [B, D]  L2-normalised embeddings (all positive pairs)
            labels     : ignored (kept for API compat with ContrastiveLoss)
            group_ids  : [B]  equivalence-group id for each pair.  Functions
                         with the same group_id are the same source function
                         at different opt-levels → must NOT be treated as
                         negatives.  If None, no false-negative masking.
            func1_hash_group_ids : [B] exact-content hash group for emb1 rows.
            func2_hash_group_ids : [B] exact-content hash group for emb2 rows.
                         Instruction-identical functions should be masked as
                         false negatives even if they are missing from the
                         explicit ground_truth set.
        Returns:
            scalar loss
        """
        B_local = emb1.size(0)

        # Gather across GPUs during training if DDP is active
        if td.is_initialized() and self.training:
            rank = td.get_rank()
            emb1_all = self._all_gather_with_grad(emb1)   # [B_all, D]
            emb2_all = self._all_gather_with_grad(emb2)   # [B_all, D]
            targets = torch.arange(B_local, device=emb1.device) + rank * B_local
            if group_ids is not None:
                gids_all = self._all_gather_no_grad(group_ids)  # [B_all]
            else:
                gids_all = None
            if func1_hash_group_ids is not None:
                func1_hash_gids_all = self._all_gather_no_grad(func1_hash_group_ids)
            else:
                func1_hash_gids_all = None
            if func2_hash_group_ids is not None:
                func2_hash_gids_all = self._all_gather_no_grad(func2_hash_group_ids)
            else:
                func2_hash_gids_all = None
        else:
            emb1_all = emb1
            emb2_all = emb2
            targets = torch.arange(B_local, device=emb1.device)
            gids_all = group_ids
            func1_hash_gids_all = func1_hash_group_ids
            func2_hash_gids_all = func2_hash_group_ids

        # Similarity: local queries vs all keys  →  [B_local, B_all]
        logits_12 = torch.mm(emb1, emb2_all.t()) / self.temperature
        logits_21 = torch.mm(emb2, emb1_all.t()) / self.temperature

        # False-negative mask: same equivalence group, off-diagonal.
        # These are different opt-level compilations of the same source function
        # and must not be penalised as negatives.
        if (group_ids is not None and gids_all is not None) or \
           (func1_hash_group_ids is not None and func2_hash_gids_all is not None) or \
           (func2_hash_group_ids is not None and func1_hash_gids_all is not None):
            # local group_ids [B_local] vs all group_ids [B_all]
            fn_mask_12 = torch.zeros_like(logits_12, dtype=torch.bool)
            fn_mask_21 = torch.zeros_like(logits_21, dtype=torch.bool)
            if group_ids is not None and gids_all is not None:
                same_group = (group_ids.unsqueeze(1) == gids_all.unsqueeze(0))
                fn_mask_12 |= same_group
                fn_mask_21 |= same_group
            if func1_hash_group_ids is not None and func2_hash_gids_all is not None:
                fn_mask_12 |= (
                    func1_hash_group_ids.unsqueeze(1) == func2_hash_gids_all.unsqueeze(0)
                )
            if func2_hash_group_ids is not None and func1_hash_gids_all is not None:
                fn_mask_21 |= (
                    func2_hash_group_ids.unsqueeze(1) == func1_hash_gids_all.unsqueeze(0)
                )
            # Keep diagonal (true positive target) unmasked
            diag = torch.zeros_like(fn_mask_12)
            diag[torch.arange(B_local, device=emb1.device), targets] = True
            fn_mask_12 = fn_mask_12 & ~diag
            fn_mask_21 = fn_mask_21 & ~diag
            logits_12 = logits_12.masked_fill(fn_mask_12, float('-inf'))
            logits_21 = logits_21.masked_fill(fn_mask_21, float('-inf'))

        # Symmetric cross-entropy
        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)
        return (loss_12 + loss_21) * 0.5
