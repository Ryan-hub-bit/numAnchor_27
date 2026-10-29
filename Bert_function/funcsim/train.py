"""
Fine-tuning script for Function Similarity

Single-GPU:
    python train.py <args>

Multi-GPU (2 GPUs):
    torchrun --standalone --nproc_per_node=2 train.py <args>
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import importlib.util
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from vocab import WordVocab

_cur = os.path.dirname(os.path.abspath(__file__))

def _load_local(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_cur, fname))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dl  = _load_local("funcsim_dataloader", "dataloader.py")
_mdl = _load_local("funcsim_model",      "model.py")

FunctionSimilarityDataset = _dl.FunctionSimilarityDataset
FunctionSimilarityModel   = _mdl.FunctionSimilarityModel
ContrastiveLoss           = _mdl.ContrastiveLoss
InfoNCELoss               = _mdl.InfoNCELoss
# ─────────────────────────────────────────────────────────────────────────────


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    # Use a generous timeout so rank 1 doesn't time out while rank 0
    # is running the full single-rank validation loop.
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=4))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return (not dist.is_available()) or \
           (not dist.is_initialized()) or \
           dist.get_rank() == 0


def _state_dict_meta(sd):
    """Extract a few checkpoint-shape facts regardless of key prefixing."""
    pos_len = None
    seg_buckets = None
    for key in ('embedding.position_embedding.pe', 'bert.embedding.position_embedding.pe'):
        if key in sd:
            pos_len = int(sd[key].shape[1])
            break
    for key in ('embedding.segment_embedding.weight', 'bert.embedding.segment_embedding.weight'):
        if key in sd:
            seg_buckets = int(sd[key].shape[0])
            break
    return pos_len, seg_buckets


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int,
                                 min_lr_ratio: float = 0.01) -> LambdaLR:
    """
    Linear warmup for `warmup_steps` steps, then cosine decay to
    `min_lr_ratio * base_lr` by `total_steps`.
    Called once per optimiser step (not per epoch).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-8, current_step / max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return LambdaLR(optimizer, lr_lambda)


# ── collate function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    """Stack tensor fields; keep boundary metadata as Python lists."""
    tensor_keys = [
        'func1_input', 'func1_segment', 'func1_binary_pos',
        'func1_func_pos', 'func1_bb_pos', 'func1_var_offsets',
        'func2_input', 'func2_segment', 'func2_binary_pos',
        'func2_func_pos', 'func2_bb_pos', 'func2_var_offsets',
        'label', 'group_id', 'func1_hash_group_id', 'func2_hash_group_id',
    ]
    out = {k: torch.stack([item[k] for item in batch]) for k in tensor_keys}
    out['func1_num_instr']  = [item['func1_num_instr']  for item in batch]
    out['func1_boundaries'] = [item['func1_boundaries'] for item in batch]
    out['func2_num_instr']  = [item['func2_num_instr']  for item in batch]
    out['func2_boundaries'] = [item['func2_boundaries'] for item in batch]
    return out


# ── training utilities ────────────────────────────────────────────────────────

def setup_logging(log_dir, experiment_name):
    logger = logging.getLogger('funcsim_train')
    if not is_main():
        logger.setLevel(logging.WARNING)
        return logger
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{experiment_name}_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logger


def set_seed(seed: int):
    """Keep pair construction and random splits reproducible across reruns."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scheduler, device,
              train: bool, logger):
    """One pass over loader.  Returns (avg_loss, recall@1)."""
    model.train(train)
    criterion.train(train)   # controls InfoNCE all_gather (skip during eval)
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    bar = tqdm(loader, desc="Train" if train else "Val ", leave=False)

    with ctx:
        for batch in bar:
            ids1  = batch['func1_input'].to(device)
            seg1  = batch['func1_segment'].to(device)
            bp1   = batch['func1_binary_pos'].to(device)
            fp1   = batch['func1_func_pos'].to(device)
            bb1   = batch['func1_bb_pos'].to(device)
            vo1   = batch['func1_var_offsets'].to(device)

            ids2  = batch['func2_input'].to(device)
            seg2  = batch['func2_segment'].to(device)
            bp2   = batch['func2_binary_pos'].to(device)
            fp2   = batch['func2_func_pos'].to(device)
            bb2   = batch['func2_bb_pos'].to(device)
            vo2   = batch['func2_var_offsets'].to(device)

            labels = batch['label'].to(device)
            group_ids = batch['group_id'].to(device)
            func1_hash_group_ids = batch['func1_hash_group_id'].to(device)
            func2_hash_group_ids = batch['func2_hash_group_id'].to(device)

            emb1 = model(ids1, seg1, bp1, fp1, bb1, vo1,
                         num_instructions=batch['func1_num_instr'],
                         instruction_boundaries=batch['func1_boundaries'])
            emb2 = model(ids2, seg2, bp2, fp2, bb2, vo2,
                         num_instructions=batch['func2_num_instr'],
                         instruction_boundaries=batch['func2_boundaries'])
            loss = criterion(
                emb1,
                emb2,
                labels,
                group_ids=group_ids,
                func1_hash_group_ids=func1_hash_group_ids,
                func2_hash_group_ids=func2_hash_group_ids,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Recall@1: is the true positive the top-1 nearest neighbor?
            # Mask same-group off-diagonal entries so false negatives
            # don't inflate the metric.
            with torch.no_grad():
                sim_matrix = torch.mm(emb1.detach(), emb2.detach().t())  # [B, B]
                B = sim_matrix.size(0)
                diag = torch.eye(B, dtype=torch.bool, device=device)
                same_grp = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))  # [B,B]
                same_hash = (
                    func1_hash_group_ids.unsqueeze(1) == func2_hash_group_ids.unsqueeze(0)
                )  # [B,B]
                fn_mask = (same_grp | same_hash) & ~diag  # off-diagonal false negatives
                sim_matrix = sim_matrix.masked_fill(fn_mask, float('-inf'))
                top1 = sim_matrix.argmax(dim=1)
                targets_r1 = torch.arange(B, device=device)
                correct += (top1 == targets_r1).sum().item()
                total   += B
            total_loss += loss.item()
            bar.set_postfix(loss=f'{loss.item():.4f}',
                            r1=f'{correct/max(total,1):.4f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

    return total_loss / len(loader), correct / max(total, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Funcsim fine-tuning")

    # Data
    parser.add_argument("--function_blocks",  required=True)
    parser.add_argument("--funcsim_pairs",    required=True)
    parser.add_argument("--vocab",            required=True)

    # Pre-trained model
    parser.add_argument("--pretrained_bert",    default=None,
                        help="Pre-trained BERT checkpoint to initialise the backbone. "
                             "Ignored when --resume_checkpoint is set.")
    parser.add_argument("--resume_checkpoint",  default=None,
                        help="Path to an existing funcsim checkpoint (best_model.pt) to resume "
                             "training from. Loads full model + optimizer state and continues "
                             "for --epochs more epochs.")
    parser.add_argument("--hidden",       type=int,   default=768)
    parser.add_argument("--n_layers",     type=int,   default=12)
    parser.add_argument("--attn_heads",   type=int,   default=12)
    parser.add_argument("--seq_len",      type=int,   default=420,
                        help="Max token length per function")
    parser.add_argument("--max_instructions", type=int, default=50,
                        help="Max number of instructions per function")
    parser.add_argument("--embedding_dim",type=int,   default=256,
                        help="Output embedding dimension")
    parser.add_argument("--window_subbatch", type=int, default=256,
                        help="Max number of BERT windows processed per internal sub-batch")
    parser.add_argument("--freeze_bert",  action="store_true")
    parser.add_argument("--no_address",   action="store_true",
                        help="Force-disable address embeddings "
                             "(auto-overridden by checkpoint detection)")
    parser.add_argument("--drop_binary_pos", action="store_true",
                        help="Keep function/bb/var information but zero out the "
                             "function-in-binary coordinate everywhere.")
    parser.add_argument("--drop_code_binary_pos", action="store_true",
                        help="Zero out the binary-level coordinate only for "
                             "code-side address tokens, while keeping daddr binary positions.")

    # Training
    parser.add_argument("--batch_size",        type=int,   default=16)
    parser.add_argument("--epochs",            type=int,   default=10)
    parser.add_argument("--lr",                type=float, default=3e-5,
                        help="Peak learning rate for projection head ")
    parser.add_argument("--lr_bert",           type=float, default=None,
                        help="Peak LR for BERT backbone (default: lr/10). "
                             "Lower than --lr to avoid damaging pre-trained weights.")
    parser.add_argument("--warmup_ratio",      type=float, default=0.06,
                        help="Fraction of total steps used for linear LR warmup")
    parser.add_argument("--loss",              default="infonce",
                        choices=["infonce", "contrastive"],
                        help="Loss function: 'infonce' (default, NT-Xent) or "
                             "'contrastive' (margin-based cosine distance)")
    parser.add_argument("--temperature",       type=float, default=0.07,
                        help="InfoNCE temperature (only used when --loss=infonce)")
    parser.add_argument("--margin",            type=float, default=1.0,
                        help="Contrastive loss margin on cosine distance [0,2] "
                             "(only used when --loss=contrastive)")
    parser.add_argument("--negative_samples",  type=int,   default=3)
    parser.add_argument("--dataset_fraction",  type=float, default=1.0)
    parser.add_argument("--train_split",       type=float, default=0.8)
    parser.add_argument("--val_split",         type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute BERT activations on backward "
                             "(saves ~50%% memory, +33%% compute)")
    parser.add_argument("--seed",              type=int,   default=42)

    # Output
    parser.add_argument("--output_dir",       default="../../output/funcsim")
    parser.add_argument("--log_dir",          default="../../log/funcsim")
    parser.add_argument("--experiment_name",  default="funcsim")
    parser.add_argument("--task_name",        default="funcsim")

    # Misc
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    use_ddp = 'LOCAL_RANK' in os.environ
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device     = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    # ─────────────────────────────────────────────────────────────────────────

    logger = setup_logging(args.log_dir, args.experiment_name)
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    logger.info("=" * 70)
    logger.info("Function Similarity Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"device            : {device}")
    logger.info(f"resume_checkpoint : {args.resume_checkpoint or '(none)'}")
    logger.info(f"pretrained_bert   : {args.pretrained_bert or '(ignored — resuming)'}")
    logger.info(f"gradient_ckpt     : {args.gradient_checkpointing}")
    logger.info(f"batch_size        : {args.batch_size}")
    logger.info(f"dataset_fraction  : {args.dataset_fraction}")
    logger.info(f"seed              : {args.seed}")
    logger.info(f"seq_len           : {args.seq_len}")
    logger.info(f"max_instructions  : {args.max_instructions}")
    logger.info(f"drop_binary_pos   : {args.drop_binary_pos}")
    logger.info(f"drop_code_binary  : {args.drop_code_binary_pos}")
    logger.info(f"embedding_dim     : {args.embedding_dim}")
    logger.info(f"window_subbatch   : {args.window_subbatch}")
    logger.info("=" * 70)

    # ── vocabulary ────────────────────────────────────────────────────────────
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocab size: {len(vocab)}")

    # Keep pair generation and dataset splits stable so evaluation can
    # reconstruct the exact sampled subset later.
    set_seed(args.seed)

    # ── detect checkpoint capabilities ────────────────────────────────────────
    # Use resume_checkpoint if given, else pretrained_bert for capability detection
    _detect_path = args.resume_checkpoint if args.resume_checkpoint else args.pretrained_bert
    if _detect_path is None:
        raise ValueError("Either --pretrained_bert or --resume_checkpoint must be provided.")
    ckpt = torch.load(_detect_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        # funcsim resume checkpoint stores use_addr explicitly
        if args.resume_checkpoint and 'use_addr' in ckpt:
            use_addr    = ckpt['use_addr']
            has_address = use_addr
            has_var     = use_addr
            use_binary_pos = ckpt.get('use_binary_pos', True)
            use_code_binary_pos = ckpt.get('use_code_binary_pos', use_binary_pos)
            use_daddr_binary_pos = ckpt.get('use_daddr_binary_pos', use_binary_pos)
            sd = ckpt.get('model_state_dict', ckpt)
        else:
            sd = ckpt.get('bert_state_dict',
                 ckpt.get('model_state_dict', ckpt))
            has_address = any('address_pos' in k or 'address_position' in k for k in sd)
            has_var     = any('var_offset'  in k or 'var_position'     in k for k in sd)
            use_addr    = has_address and has_var
            use_binary_pos = not args.drop_binary_pos
            use_code_binary_pos = not (args.drop_binary_pos or args.drop_code_binary_pos)
            use_daddr_binary_pos = not args.drop_binary_pos
    else:
        sd = ckpt.state_dict()
        has_address = any('address_pos' in k or 'address_position' in k for k in sd)
        has_var     = any('var_offset'  in k or 'var_position'     in k for k in sd)
        use_addr    = has_address and has_var
        use_binary_pos = not args.drop_binary_pos
        use_code_binary_pos = not (args.drop_binary_pos or args.drop_code_binary_pos)
        use_daddr_binary_pos = not args.drop_binary_pos
    checkpoint_max_len, checkpoint_segment_buckets = _state_dict_meta(sd)
    function_level_mode = bool(
        (checkpoint_max_len is not None and checkpoint_max_len > 20) or
        (checkpoint_segment_buckets is not None and checkpoint_segment_buckets > 32)
    )
    encoder_mode = 'fullseq' if function_level_mode else 'window'
    logger.info(
        f"Checkpoint  has_address={has_address}  has_var={has_var}  "
        f"-> use_addr={use_addr}"
    )
    logger.info(f"Binary position  enabled={use_binary_pos}")
    logger.info(f"Code binary pos  enabled={use_code_binary_pos}")
    logger.info(f"Daddr binary pos enabled={use_daddr_binary_pos}")
    logger.info(
        f"Checkpoint  max_len={checkpoint_max_len}  "
        f"segment_buckets={checkpoint_segment_buckets}  "
        f"-> encoder_mode={encoder_mode}"
    )
    del ckpt, sd   # free memory before building model

    # ── dataset ───────────────────────────────────────────────────────────────
    logger.info("Building dataset …")
    full_ds = FunctionSimilarityDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=args.seq_len,
        max_instructions=args.max_instructions,
        negative_samples=args.negative_samples,
    )
    full_ds.use_address_var = use_addr
    full_ds.use_binary_pos = use_binary_pos
    full_ds.use_code_binary_pos = use_code_binary_pos
    full_ds.use_daddr_binary_pos = use_daddr_binary_pos
    full_ds.use_function_level_encoding = function_level_mode

    # optional fraction
    if args.dataset_fraction < 1.0:
        keep = int(len(full_ds) * args.dataset_fraction)
        full_ds, _ = random_split(full_ds, [keep, len(full_ds) - keep],
                                  generator=torch.Generator().manual_seed(args.seed))
        logger.info(f"Using {args.dataset_fraction*100:.1f}% → {keep} samples")

    # train / val / test split
    # If train_split=1.0 and val_split=0.0, use entire dataset for training (no val/test)
    use_val  = args.val_split   > 0.0
    use_test = args.train_split < 1.0

    if use_test:
        test_n = int(len(full_ds) * (1 - args.train_split))
        tv_n   = len(full_ds) - test_n
        tv_ds, test_ds = random_split(full_ds, [tv_n, test_n],
                                      generator=torch.Generator().manual_seed(args.seed))
    else:
        tv_ds, test_ds = full_ds, None

    if use_val:
        val_n   = int(len(tv_ds) * args.val_split)
        train_n = len(tv_ds) - val_n
        train_ds, val_ds = random_split(tv_ds, [train_n, val_n],
                                        generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = tv_ds, None

    logger.info(f"Split  train={len(train_ds)}"
                + (f"  val={len(val_ds)}" if val_ds else "  val=none")
                + (f"  test={len(test_ds)}" if test_ds else "  test=none"))

    # save test indices for evaluate.py
    if is_main() and test_ds is not None:
        test_idx_path = os.path.join(args.output_dir, 'test_indices.json')
        tidxs = test_ds.indices if hasattr(test_ds, 'indices') else list(range(len(test_ds)))
        with open(test_idx_path, 'w') as f:
            json.dump({'test_indices': tidxs, 'test_size': len(test_ds)}, f)
        logger.info(f"Test indices saved to {test_idx_path}")

    # ── samplers & loaders ────────────────────────────────────────────────────
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=(train_sampler is None),
                               sampler=train_sampler,
                               num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_fn)

    if val_ds is not None:
        # No DistributedSampler for validation: rank 0 evaluates on the full
        # val shard; other ranks wait.  Avoids NCCL collective mismatches.
        val_loader  = DataLoader(val_ds, batch_size=args.batch_size,
                                 shuffle=False, sampler=None,
                                 num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=collate_fn)
    else:
        val_loader = None

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Creating model …")
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.seq_len if function_level_mode else 20,
        window_size=args.seq_len if function_level_mode else 20,
        instrs_per_window=2,                 # 2 instructions per BERT window
        sos_idx=vocab.stoi.get('<sos>', 3),
        eos_idx=vocab.stoi.get('<eos>', 2),
        pad_idx=vocab.stoi.get('<pad>', 0),
        use_address_embedding=use_addr,
        use_var_embedding=use_addr,
        embedding_dim=args.embedding_dim,
        freeze_bert=args.freeze_bert,
        gradient_checkpointing=args.gradient_checkpointing,
        window_subbatch=args.window_subbatch,
        encoder_mode=encoder_mode,
    )

    # set daddr routing token
    daddr_id = vocab.stoi.get('daddr')
    if daddr_id is not None and use_addr:
        model.bert.embedding.set_special_token_ids(daddr_id)
        logger.info(f"daddr token id = {daddr_id}")
    raw_binary_pos_flag = use_binary_pos if 'use_binary_pos' in locals() else True
    model.bert.embedding.set_position_ablation(
        use_binary_pos=raw_binary_pos_flag,
        use_code_binary_pos=use_code_binary_pos,
        use_daddr_binary_pos=use_daddr_binary_pos,
    )

    # ── load weights ──────────────────────────────────────────────────────────
    resume_ckpt = None
    if args.resume_checkpoint:
        resume_ckpt = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(
            resume_ckpt.get('model_state_dict', resume_ckpt), strict=False)
        if missing:    logger.warning(f"Missing keys:    {missing[:5]}")
        if unexpected: logger.warning(f"Unexpected keys: {unexpected[:5]}")
        logger.info(f"Resumed from checkpoint  epoch={resume_ckpt.get('epoch','?')}  "
                    f"val_r@1={resume_ckpt.get('val_acc', 0.0):.4f}")
    else:
        model.load_pretrained_bert(args.pretrained_bert)
    model.bert.embedding.set_position_ablation(
        use_binary_pos=use_binary_pos,
        use_code_binary_pos=use_code_binary_pos,
        use_daddr_binary_pos=use_daddr_binary_pos,
    )

    model = model.to(device)

    if use_ddp:
        model     = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model

    total_p     = sum(p.numel() for p in raw_model.parameters())
    trainable_p = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    logger.info(f"Parameters  total={total_p:,}  trainable={trainable_p:,}")

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    if args.loss == 'infonce':
        criterion = InfoNCELoss(temperature=args.temperature)
        logger.info(f"Loss: InfoNCE  temperature={args.temperature}")
    else:
        criterion = ContrastiveLoss(margin=args.margin)
        logger.info(f"Loss: Contrastive  margin={args.margin}")

    # Differential learning rates:
    #   BERT backbone  → lr_bert  (default lr/10)  — preserve pre-trained weights
    #   Projection head → lr               — randomly initialised, can learn faster
    lr_bert = args.lr_bert if args.lr_bert is not None else args.lr / 10.0
    param_groups = [
        {'params': raw_model.bert.parameters(),       'lr': lr_bert},
        {'params': raw_model.projection.parameters(), 'lr': args.lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    logger.info(f"LR  bert={lr_bert:.2e}  projection={args.lr:.2e}")

    # Restore optimizer state when resuming (preserves momentum / Adam moments)
    if resume_ckpt is not None and 'optimizer_state_dict' in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        logger.info("Optimizer state restored from checkpoint")

    # Epoch counter and best metric: when resuming, continue from saved state
    # (initialize best_val_acc from checkpoint so we only overwrite if we improve)
    if resume_ckpt is not None:
        start_epoch       = resume_ckpt.get('epoch', 0) + 1
        best_val_acc_init = resume_ckpt.get('val_acc', 0.0)
    else:
        start_epoch       = 1
        best_val_acc_init = 0.0
    end_epoch = start_epoch + args.epochs   # exclusive upper bound
    del resume_ckpt  # free memory

    # Step-level warmup + cosine decay scheduler
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    logger.info(f"Scheduler  warmup={warmup_steps} steps / {total_steps} total")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_acc = best_val_acc_init
    logger.info(f"\nStarting training …  (epochs {start_epoch}–{end_epoch - 1}, "
                f"best so far={best_val_acc:.4f})")

    for epoch in range(start_epoch, end_epoch):
        logger.info(f"\nEpoch {epoch}/{end_epoch - 1}")
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, device, train=True,  logger=logger)
        logger.info(f"  train  loss={tr_loss:.4f}  r@1={tr_acc:.4f}")

        if val_loader is not None:
            # Validation: only rank 0 runs it using the unwrapped model
            # (avoids DDP NCCL collectives during no-grad forward).
            if is_main():
                va_loss, va_acc = run_epoch(raw_model, val_loader, criterion,
                                            optimizer, None, device,
                                            train=False, logger=logger)
            else:
                va_loss, va_acc = 0.0, 0.0

            # Broadcast metrics from rank 0 so all ranks have the same values
            if use_ddp:
                metrics = torch.tensor([va_loss, va_acc], device=device)
                dist.broadcast(metrics, src=0)
                va_loss, va_acc = metrics[0].item(), metrics[1].item()

            logger.info(f"  val    loss={va_loss:.4f}  r@1={va_acc:.4f}")
        else:
            va_loss, va_acc = tr_loss, tr_acc   # fall back to train metric

        logger.info(f"  lr_bert={optimizer.param_groups[0]['lr']:.2e}  "
                    f"lr_proj={optimizer.param_groups[1]['lr']:.2e}")

        if is_main():
            ckpt = {
                'epoch':              epoch,
                'model_state_dict':   raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr_loss, 'val_loss': va_loss,
                'train_acc':  tr_acc,  'val_acc':  va_acc,
                'task_name':  args.task_name,
                'use_addr':   use_addr,
                'use_binary_pos': use_binary_pos,
                'use_code_binary_pos': use_code_binary_pos,
                'use_daddr_binary_pos': use_daddr_binary_pos,
                'embedding_dim': args.embedding_dim,
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pt'))
                logger.info(f"  ✓ New best model  val_r@1={va_acc:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training complete.  Best val r@1 = {best_val_acc:.4f}")
    logger.info("=" * 70)

    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
