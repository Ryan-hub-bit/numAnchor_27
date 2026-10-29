import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paper_funcsim import (
    AddressAwarePaperFunctionModel,
    AddressAwareTripletDataset,
    TripletCosineMarginLoss,
    collate_triplets,
)
from paths import ensure_parent_under_data_root, ensure_under_data_root
from vocab import WordVocab


def setup_logging(log_dir: str, experiment_name: str):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{experiment_name}_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("addressaware_paper_funcsim")


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-8, current_step / max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


def _to_device(side, device):
    return {
        "input_ids": side["input_ids"].to(device),
        "segment": side["segment"].to(device),
        "binary_pos": side["binary_pos"].to(device),
        "function_pos": side["function_pos"].to(device),
        "bb_pos": side["bb_pos"].to(device),
        "var_offsets": side["var_offsets"].to(device),
        "addr_type_ids": side["addr_type_ids"].to(device),
        "group_ids": side["group_ids"].to(device),
    }


def encode_side(model, side):
    return model(
        side["input_ids"],
        side["segment"],
        side["binary_pos"],
        side["function_pos"],
        side["bb_pos"],
        side["var_offsets"],
        side["addr_type_ids"],
    )


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    bar = tqdm(loader, desc="Train" if train else "Val", leave=False)

    with ctx:
        for batch in bar:
            anchor = _to_device(batch["anchor"], device)
            positive = _to_device(batch["positive"], device)
            negative = _to_device(batch["negative"], device)

            anchor_emb = encode_side(model, anchor)
            pos_emb = encode_side(model, positive)
            neg_emb = encode_side(model, negative)
            loss = criterion(anchor_emb, pos_emb, neg_emb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                sim = torch.mm(F.normalize(anchor_emb, p=2, dim=1), F.normalize(pos_emb, p=2, dim=1).t())
                gids = anchor["group_ids"]
                same_group = gids.unsqueeze(1) == gids.unsqueeze(0)
                diag = torch.eye(sim.size(0), dtype=torch.bool, device=device)
                sim = sim.masked_fill(same_group & ~diag, float("-inf"))
                top1 = sim.argmax(dim=1)
                targets = torch.arange(sim.size(0), device=device)
                total_correct += (top1 == targets).sum().item()
                total += sim.size(0)

            total_loss += loss.item()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                r1=f"{total_correct / max(total, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return total_loss / max(len(loader), 1), total_correct / max(total, 1)


def save_checkpoint(model, optimizer, path, epoch, train_loss, val_loss, train_r1, val_r1, config):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_r1,
        "val_acc": val_r1,
        "use_addr": config["use_address_embedding"],
        "use_var": config["use_var_embedding"],
        "config": config,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned address-aware funcsim fine-tuning")
    parser.add_argument("--function_blocks", required=True)
    parser.add_argument("--funcsim_pairs", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--pretrained_bert", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--attn_heads", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_instructions", type=int, default=100000)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--dataset_fraction", type=float, default=1.0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment_name", default="paper_triplet")
    parser.add_argument("--disable_address_embedding", action="store_true")
    parser.add_argument("--disable_var_embedding", action="store_true")
    parser.add_argument("--address_scale", type=float, default=None)
    parser.add_argument("--learnable_address_scale", action="store_true")
    args = parser.parse_args()

    args.function_blocks = ensure_under_data_root(args.function_blocks, "--function_blocks", must_exist=True)
    args.funcsim_pairs = ensure_under_data_root(args.funcsim_pairs, "--funcsim_pairs", must_exist=True)
    args.vocab = ensure_under_data_root(args.vocab, "--vocab", must_exist=True)
    args.pretrained_bert = ensure_under_data_root(args.pretrained_bert, "--pretrained_bert", must_exist=True)
    args.output_dir = ensure_parent_under_data_root(args.output_dir, "--output_dir")
    args.log_dir = ensure_parent_under_data_root(args.log_dir, "--log_dir")

    logger = setup_logging(args.log_dir, args.experiment_name)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("Paper-aligned Function Similarity Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"device            : {device}")
    logger.info(f"pretrained_bert   : {args.pretrained_bert}")
    logger.info("=" * 70)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocab size: {len(vocab)}")

    pretrain_ckpt = torch.load(args.pretrained_bert, map_location="cpu", weights_only=False)
    ckpt_config = pretrain_ckpt.get("config", {})
    hidden = args.hidden or ckpt_config.get("hidden", 256)
    n_layers = args.n_layers or ckpt_config.get("n_layers", 4)
    attn_heads = args.attn_heads or ckpt_config.get("attn_heads", 4)
    seq_len = args.seq_len or 512
    use_address_embedding = ckpt_config.get("use_address_embedding", True)
    use_var_embedding = ckpt_config.get("use_var_embedding", True)
    if args.disable_address_embedding:
        use_address_embedding = False
    if args.disable_var_embedding:
        use_var_embedding = False
    use_address_var = use_address_embedding or use_var_embedding
    address_scale = args.address_scale if args.address_scale is not None else ckpt_config.get("address_scale", 10000.0)
    learnable_address_scale = args.learnable_address_scale or ckpt_config.get("learnable_address_scale", False)
    logger.info(
        "Backbone config  hidden=%d  layers=%d  heads=%d  seq_len=%d  use_address=%s  use_var=%s  address_scale=%s  learnable_scale=%s",
        hidden,
        n_layers,
        attn_heads,
        seq_len,
        use_address_embedding,
        use_var_embedding,
        address_scale,
        learnable_address_scale,
    )

    full_ds = AddressAwareTripletDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=seq_len,
        max_instructions=args.max_instructions,
        use_address_var=use_address_var,
        seed=args.seed,
    )

    if args.dataset_fraction < 1.0:
        keep = max(1, int(len(full_ds) * args.dataset_fraction))
        full_ds, _ = random_split(
            full_ds,
            [keep, len(full_ds) - keep],
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info(f"Using {args.dataset_fraction * 100:.1f}%% of anchors -> {keep}")

    test_n = int(len(full_ds) * (1 - args.train_split))
    tv_n = len(full_ds) - test_n
    tv_ds, test_ds = random_split(full_ds, [tv_n, test_n], generator=torch.Generator().manual_seed(args.seed))
    val_n = int(len(tv_ds) * args.val_split)
    train_n = len(tv_ds) - val_n
    train_ds, val_ds = random_split(tv_ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    logger.info(f"Split  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    os.makedirs(args.output_dir, exist_ok=True)
    if hasattr(test_ds, "indices"):
        test_idx_path = os.path.join(args.output_dir, "test_indices.json")
        with open(test_idx_path, "w", encoding="utf-8") as f:
            json.dump({"test_indices": test_ds.indices, "test_size": len(test_ds)}, f)
        logger.info(f"Test indices saved to {test_idx_path}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_triplets,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_triplets,
        pin_memory=True,
    )

    logger.info("Creating model …")
    model = AddressAwarePaperFunctionModel(
        vocab_size=len(vocab),
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        seq_len=seq_len,
        dropout=args.dropout,
        use_address_embedding=use_address_embedding,
        use_var_embedding=use_var_embedding,
        address_scale=address_scale,
        learnable_address_scale=learnable_address_scale,
    )
    model.bert.gradient_checkpointing = args.gradient_checkpointing
    model.load_pretrained_bert(args.pretrained_bert)
    if args.freeze_embeddings:
        for param in model.bert.embedding.parameters():
            param.requires_grad = False
    if args.freeze_layers > 0:
        for layer in model.bert.transformer_blocks[: args.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    daddr_id = vocab.stoi.get("daddr")
    if daddr_id is not None and use_address_embedding:
        model.bert.embedding.set_special_token_ids(daddr_id)
        logger.info(f"daddr token id = {daddr_id}")

    model = model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters  total={total_p:,}  trainable={trainable_p:,}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")

    criterion = TripletCosineMarginLoss(margin=args.margin)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    logger.info(f"Loss: TripletCosineMargin  margin={args.margin}")
    logger.info(f"Scheduler  warmup={warmup_steps} steps / {total_steps} total")

    config = {
        "vocab_size": len(vocab),
        "hidden": hidden,
        "n_layers": n_layers,
        "attn_heads": attn_heads,
        "seq_len": seq_len,
        "dropout": args.dropout,
        "use_address_embedding": use_address_embedding,
        "use_var_embedding": use_var_embedding,
        "address_scale": address_scale,
        "learnable_address_scale": learnable_address_scale,
        "max_instructions": args.max_instructions,
        "margin": args.margin,
        "freeze_embeddings": args.freeze_embeddings,
        "freeze_layers": args.freeze_layers,
    }

    best_val = 0.0
    logger.info("\nStarting training …")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_r1 = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        val_loss, val_r1 = run_epoch(model, val_loader, criterion, optimizer, None, device, train=False)
        logger.info(f"  train  loss={train_loss:.4f}  r@1={train_r1:.4f}")
        logger.info(f"  val    loss={val_loss:.4f}  r@1={val_r1:.4f}")
        logger.info(f"  lr={optimizer.param_groups[0]['lr']:.2e}")

        latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
        save_checkpoint(model, optimizer, latest_path, epoch, train_loss, val_loss, train_r1, val_r1, config)
        if val_r1 > best_val:
            best_val = val_r1
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, best_path, epoch, train_loss, val_loss, train_r1, val_r1, config)
            logger.info(f"  ✓ New best model  val_r@1={val_r1:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training complete.  Best val r@1 = {best_val:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
