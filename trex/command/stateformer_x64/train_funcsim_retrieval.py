#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

TREX_ROOT = Path(__file__).resolve().parents[2]
if str(TREX_ROOT) not in sys.path:
    sys.path.insert(0, str(TREX_ROOT))

from fairseq import checkpoint_utils

from command.stateformer_x64.funcsim_retrieval_common import (
    TrexPositivePairDataset,
    build_hash_groups,
    build_pair_collate_fn,
    load_json,
    sample_positive_pairs,
    split_positive_pairs,
)


class TrexRetrievalModel(nn.Module):
    def __init__(self, backbone, static_pad_idx: int, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.static_pad_idx = static_pad_idx
        hidden_dim = int(backbone.args.encoder_embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, src_tokens, aa_inputs=None):
        features = self.backbone(src_tokens=src_tokens, aa_inputs=aa_inputs, features_only=True)[0]["features"]
        mask = src_tokens["static"].ne(self.static_pad_idx).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings = self.projection(pooled)
        return F.normalize(embeddings, dim=-1)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb0, emb1, group_ids=None, func0_hash_group_ids=None, func1_hash_group_ids=None):
        batch_size = emb0.size(0)
        targets = torch.arange(batch_size, device=emb0.device)

        logits_01 = torch.mm(emb0, emb1.t()) / self.temperature
        logits_10 = torch.mm(emb1, emb0.t()) / self.temperature

        if group_ids is not None or (func0_hash_group_ids is not None and func1_hash_group_ids is not None):
            fn_mask_01 = torch.zeros_like(logits_01, dtype=torch.bool)
            fn_mask_10 = torch.zeros_like(logits_10, dtype=torch.bool)
            if group_ids is not None:
                same_group = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
                fn_mask_01 |= same_group
                fn_mask_10 |= same_group
            if func0_hash_group_ids is not None and func1_hash_group_ids is not None:
                fn_mask_01 |= func0_hash_group_ids.unsqueeze(1) == func1_hash_group_ids.unsqueeze(0)
                fn_mask_10 |= func1_hash_group_ids.unsqueeze(1) == func0_hash_group_ids.unsqueeze(0)
            diag = torch.eye(batch_size, device=emb0.device, dtype=torch.bool)
            logits_01 = logits_01.masked_fill(fn_mask_01 & ~diag, float("-inf"))
            logits_10 = logits_10.masked_fill(fn_mask_10 & ~diag, float("-inf"))

        loss_01 = F.cross_entropy(logits_01, targets)
        loss_10 = F.cross_entropy(logits_10, targets)
        return 0.5 * (loss_01 + loss_10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trex retrieval-style function similarity finetuning")
    parser.add_argument("--mode", choices=["baseline", "addressaware", "addressaware_aa"], default="addressaware_aa")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--function-blocks", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json"))
    parser.add_argument("--funcsim-pairs", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json"))
    parser.add_argument("--exclude-query-pool", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr-backbone", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--valid-fraction", type=float, default=0.02)
    parser.add_argument("--train-positive-samples", type=int, default=50000)
    parser.add_argument("--valid-positive-samples", type=int, default=5000)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger("trex_funcsim_retrieval")
    logger.info("log_file=%s", log_path)
    return logger


def load_backbone(checkpoint_path: Path, data_dir: Path):
    models, _cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint_path)],
        arg_overrides={"data": str(data_dir)},
    )
    return models[0], task


def move_to_device(batch_src, batch_aa, device):
    batch_src = {field: tensor.to(device) for field, tensor in batch_src.items()}
    if batch_aa is not None:
        batch_aa = {field: tensor.to(device) for field, tensor in batch_aa.items()}
    return batch_src, batch_aa


def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-8, current_step / max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.01, cosine)

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, criterion, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()
    progress = tqdm(loader, desc="train" if train else "valid", leave=False)
    with context:
        for batch in progress:
            src0, aa0 = move_to_device(batch["func0_src"], batch["func0_aa"], device)
            src1, aa1 = move_to_device(batch["func1_src"], batch["func1_aa"], device)
            group_ids = batch["group_id"].to(device)
            hash0 = batch["func0_hash_group_id"].to(device)
            hash1 = batch["func1_hash_group_id"].to(device)

            emb0 = model(src0, aa0)
            emb1 = model(src1, aa1)
            loss = criterion(
                emb0,
                emb1,
                group_ids=group_ids,
                func0_hash_group_ids=hash0,
                func1_hash_group_ids=hash1,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                sim_matrix = torch.mm(emb0.detach(), emb1.detach().t())
                diag = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
                same_group = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
                same_hash = hash0.unsqueeze(1) == hash1.unsqueeze(0)
                sim_matrix = sim_matrix.masked_fill((same_group | same_hash) & ~diag, float("-inf"))
                top1 = sim_matrix.argmax(dim=1)
                targets = torch.arange(sim_matrix.size(0), device=device)
                correct += (top1 == targets).sum().item()
                total += sim_matrix.size(0)

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", r1=f"{correct / max(total, 1):.4f}")

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging(args.log_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    logger.info("loading function similarity data")
    function_blocks = load_json(args.function_blocks)
    funcsim_pairs = load_json(args.funcsim_pairs)
    pool_spec = load_json(args.exclude_query_pool)

    split_info = split_positive_pairs(
        function_blocks=function_blocks,
        funcsim_pairs=funcsim_pairs,
        exclude_query_ids=[str(query_id) for query_id in pool_spec.get("query_ids", [])],
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    func_to_hash_group = build_hash_groups(function_blocks)

    train_pairs = sample_positive_pairs(split_info["train_pairs"], args.train_positive_samples, args.seed)
    valid_pairs = sample_positive_pairs(split_info["valid_pairs"], args.valid_positive_samples, args.seed + 1)

    logger.info(
        "split stats train_pairs=%d valid_pairs=%d train_groups=%d valid_groups=%d excluded_groups=%d",
        len(train_pairs),
        len(valid_pairs),
        len(split_info["train_groups"]),
        len(split_info["valid_groups"]),
        len(split_info["excluded_groups"]),
    )

    logger.info("loading backbone from %s", args.pretrained_checkpoint)
    backbone, task = load_backbone(args.pretrained_checkpoint, args.data_dir)
    dictionaries = task.source_dictionary

    train_dataset = TrexPositivePairDataset(
        function_blocks=function_blocks,
        pairs=train_pairs,
        dictionaries=dictionaries,
        func_to_group=split_info["func_to_group"],
        func_to_hash_group=func_to_hash_group,
        mode=args.mode,
        max_positions=args.max_positions,
    )
    valid_dataset = TrexPositivePairDataset(
        function_blocks=function_blocks,
        pairs=valid_pairs,
        dictionaries=dictionaries,
        func_to_group=split_info["func_to_group"],
        func_to_hash_group=func_to_hash_group,
        mode=args.mode,
        max_positions=args.max_positions,
    )

    collate_fn = build_pair_collate_fn(dictionaries)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = TrexRetrievalModel(
        backbone=backbone,
        static_pad_idx=dictionaries["static"].pad(),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    model = model.to(device)

    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
            {"params": model.projection.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(train_loader), 1) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    logger.info(
        "device=%s batch_size=%d epochs=%d lr_backbone=%.2e lr_proj=%.2e total_steps=%d warmup_steps=%d",
        device,
        args.batch_size,
        args.epochs,
        args.lr_backbone,
        args.lr,
        total_steps,
        warmup_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_args.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "pretrained_checkpoint": str(args.pretrained_checkpoint),
                "data_dir": str(args.data_dir),
                "function_blocks": str(args.function_blocks),
                "funcsim_pairs": str(args.funcsim_pairs),
                "exclude_query_pool": str(args.exclude_query_pool),
                "train_pairs": len(train_pairs),
                "valid_pairs": len(valid_pairs),
            },
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )

    best_val_r1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_r1 = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, train=True)
        valid_loss, valid_r1 = run_epoch(model, valid_loader, optimizer, None, criterion, device, train=False)
        logger.info(
            "epoch=%d train_loss=%.4f train_r1=%.4f valid_loss=%.4f valid_r1=%.4f",
            epoch,
            train_loss,
            train_r1,
            valid_loss,
            valid_r1,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_r1": train_r1,
            "valid_loss": valid_loss,
            "valid_r1": valid_r1,
            "config": {
                "mode": args.mode,
                "embedding_dim": args.embedding_dim,
                "dropout": args.dropout,
                "max_positions": args.max_positions,
                "temperature": args.temperature,
                "pretrained_checkpoint": str(args.pretrained_checkpoint),
                "data_dir": str(args.data_dir),
                "function_blocks": str(args.function_blocks),
                "funcsim_pairs": str(args.funcsim_pairs),
                "exclude_query_pool": str(args.exclude_query_pool),
            },
        }
        torch.save(checkpoint, args.output_dir / "checkpoint_latest.pt")
        if valid_r1 > best_val_r1:
            best_val_r1 = valid_r1
            torch.save(checkpoint, args.output_dir / "best_model.pt")
            logger.info("new best model saved with valid_r1=%.4f", valid_r1)

    logger.info("training complete best_valid_r1=%.4f", best_val_r1)


if __name__ == "__main__":
    main()
