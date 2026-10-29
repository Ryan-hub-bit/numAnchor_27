import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import AddressAwareFunctionDataset
from model import AddressAwareJTransPretrainer
from paths import ensure_parent_under_data_root, ensure_under_data_root
from vocab import WordVocab


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pretrain_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("jtrans_addressaware_pretrain")


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    mlm_criterion = nn.NLLLoss(ignore_index=-1)
    jtp_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1" or not sys.stdout.isatty()

    total_loss = 0.0
    total_mlm = 0.0
    total_jtp = 0.0
    batches = 0
    jtp_batches = 0
    jtp_examples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "val", leave=False, disable=disable_tqdm):
            mlm = batch["mlm"]
            jtp = batch["jtp"]

            mlm_input = mlm["bert_input"].to(device)
            mlm_seg = mlm["segment_label"].to(device)
            mlm_bp = mlm["binary_pos"].to(device)
            mlm_fp = mlm["function_pos"].to(device)
            mlm_bb = mlm["bb_pos"].to(device)
            mlm_vo = mlm["var_offsets"].to(device)
            mlm_type = mlm["addr_type_ids"].to(device)
            mlm_label = mlm["bert_label"].to(device)

            jtp_input = jtp["bert_input"].to(device)
            jtp_seg = jtp["segment_label"].to(device)
            jtp_bp = jtp["binary_pos"].to(device)
            jtp_fp = jtp["function_pos"].to(device)
            jtp_bb = jtp["bb_pos"].to(device)
            jtp_vo = jtp["var_offsets"].to(device)
            jtp_type = jtp["addr_type_ids"].to(device)
            jtp_source = jtp["source_idx"].to(device)
            jtp_target = jtp["target_idx"].to(device)

            mlm_out = model.forward_mlm(mlm_input, mlm_seg, mlm_bp, mlm_fp, mlm_bb, mlm_vo, mlm_type)
            mlm_loss = mlm_criterion(mlm_out.view(-1, mlm_out.size(-1)), mlm_label.view(-1))

            valid_jtp = jtp_target.ne(-1)
            if valid_jtp.any():
                jtp_out = model.forward_jtp(jtp_input, jtp_seg, jtp_bp, jtp_fp, jtp_bb, jtp_vo, jtp_source, jtp_type)
                jtp_loss = jtp_criterion(jtp_out, jtp_target)
                total_jtp += jtp_loss.item()
                jtp_batches += 1
                jtp_examples += int(valid_jtp.sum().item())
            else:
                jtp_loss = mlm_loss.new_zeros(())

            loss = mlm_loss + jtp_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_mlm += mlm_loss.item()
            batches += 1

    denom = max(1, batches)
    jtp_denom = max(1, jtp_batches)
    return {
        "loss": total_loss / denom,
        "mlm_loss": total_mlm / denom,
        "jtp_loss": total_jtp / jtp_denom,
        "jtp_batches": jtp_batches,
        "jtp_examples": jtp_examples,
    }


def save_checkpoint(model, path, vocab_size, args, epoch, metrics):
    use_address_embedding = not args.disable_address_embedding
    use_var_embedding = not args.disable_var_embedding
    ckpt = {
        "epoch": epoch,
        "metrics": metrics,
        "config": {
            "vocab_size": vocab_size,
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "attn_heads": args.attn_heads,
            "dropout": args.dropout,
            "use_address_embedding": use_address_embedding,
            "use_var_embedding": use_var_embedding,
            "address_scale": args.address_scale,
            "learnable_address_scale": args.learnable_address_scale,
            "segment_types": args.segment_types if args.segment_types is not None else args.seq_len + 1,
        },
        "bert_state_dict": model.bert.state_dict(),
        "model_state_dict": model.state_dict(),
        "use_address_embedding": use_address_embedding,
        "use_var_embedding": use_var_embedding,
        "address_scale": args.address_scale,
        "learnable_address_scale": args.learnable_address_scale,
        "segment_types": args.segment_types if args.segment_types is not None else args.seq_len + 1,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Address-aware JTrans pretraining (MLM + JTP)")
    parser.add_argument("--train_corpus", required=True)
    parser.add_argument("--val_corpus", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train_functions", type=int, default=None)
    parser.add_argument("--max_val_functions", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_address_embedding", action="store_true")
    parser.add_argument("--disable_var_embedding", action="store_true")
    parser.add_argument("--address_scale", type=float, default=10000.0)
    parser.add_argument("--learnable_address_scale", action="store_true")
    parser.add_argument("--segment_types", type=int, default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    args = parser.parse_args()

    args.train_corpus = ensure_under_data_root(args.train_corpus, "--train_corpus", must_exist=True)
    args.val_corpus = ensure_under_data_root(args.val_corpus, "--val_corpus", must_exist=True)
    args.vocab = ensure_under_data_root(args.vocab, "--vocab", must_exist=True)
    args.output_dir = ensure_parent_under_data_root(args.output_dir, "--output_dir")
    args.log_dir = ensure_parent_under_data_root(args.log_dir, "--log_dir")
    if args.resume_checkpoint:
        args.resume_checkpoint = ensure_under_data_root(args.resume_checkpoint, "--resume_checkpoint", must_exist=True)

    logger = setup_logging(args.log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    vocab = WordVocab.load_vocab(args.vocab)
    train_ds = AddressAwareFunctionDataset(args.train_corpus, vocab, seq_len=args.seq_len, max_functions=args.max_train_functions)
    val_ds = AddressAwareFunctionDataset(args.val_corpus, vocab, seq_len=args.seq_len, max_functions=args.max_val_functions)
    logger.info(f"train_functions={len(train_ds)} val_functions={len(val_ds)}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = AddressAwareJTransPretrainer(
        vocab_size=len(vocab),
        seq_len=args.seq_len,
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        use_address_embedding=not args.disable_address_embedding,
        use_var_embedding=not args.disable_var_embedding,
        address_scale=args.address_scale,
        learnable_address_scale=args.learnable_address_scale,
        segment_types=args.segment_types,
    ).to(device)
    model.bert.gradient_checkpointing = args.gradient_checkpointing
    logger.info(
        "gradient_checkpointing=%s use_address_embedding=%s use_var_embedding=%s segment_types=%s",
        args.gradient_checkpointing,
        not args.disable_address_embedding,
        not args.disable_var_embedding,
        model.bert.segment_types,
    )

    daddr_id = vocab.stoi.get("daddr")
    if daddr_id is not None and not args.disable_address_embedding:
        model.bert.embedding.set_special_token_ids(daddr_id)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    best_val = float("inf")
    history = []
    start_epoch = 1
    if args.resume_checkpoint:
        logger.info("resuming from checkpoint=%s", args.resume_checkpoint)
        resume_ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        elif "bert_state_dict" in resume_ckpt:
            current = model.state_dict()
            remap = {}
            for key, value in resume_ckpt["bert_state_dict"].items():
                remap[f"bert.{key}"] = value
            for key in list(remap.keys()):
                if key in current and remap[key].shape != current[key].shape:
                    remap[key] = current[key]
            model.load_state_dict(remap, strict=False)
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val = float(resume_ckpt.get("metrics", {}).get("val", {}).get("loss", float("inf")))
        history_path = os.path.join(args.output_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        logger.info("resume_start_epoch=%d best_val=%.4f", start_epoch, best_val)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        logger.info(
            "epoch=%d train_loss=%.4f train_mlm=%.4f train_jtp=%.4f train_jtp_batches=%d "
            "val_loss=%.4f val_mlm=%.4f val_jtp=%.4f val_jtp_batches=%d",
            epoch,
            train_metrics["loss"],
            train_metrics["mlm_loss"],
            train_metrics["jtp_loss"],
            train_metrics["jtp_batches"],
            val_metrics["loss"],
            val_metrics["mlm_loss"],
            val_metrics["jtp_loss"],
            val_metrics["jtp_batches"],
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        latest_path = os.path.join(args.output_dir, "latest_model.pt")
        save_checkpoint(model, latest_path, len(vocab), args, epoch, {"train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, best_path, len(vocab), args, epoch, {"train": train_metrics, "val": val_metrics})

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("training_complete")


if __name__ == "__main__":
    main()
