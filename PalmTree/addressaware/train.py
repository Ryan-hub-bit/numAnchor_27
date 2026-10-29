"""
Training script for Address-Aware BERT with Instruction Masking

Trains a BERT model with:
1. Instruction Masking CFG (IMC) - masks entire instructions in CFG
2. Instruction Masking DFG (IMD) - masks entire instructions in DFG
3. Masked Language Modeling (MLM) - masks individual tokens  

New tasks: IMC/IMD (Instruction Masking for CFG/DFG)
- Masks entire instructions at a configurable rate
- Model must predict all tokens in the masked instruction
- Different from MLM which masks individual tokens
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


try:
    from .vocab import WordVocab
    from .create_vocab import create_vocab
    from .dataloader import InstructionMaskingDataset, StructurePairDataset
    from .model import AddressAwareBERT, AddressAwareBERTForPretraining
except ImportError:
    from vocab import WordVocab
    from create_vocab import create_vocab
    from dataloader import InstructionMaskingDataset, StructurePairDataset
    from model import AddressAwareBERT, AddressAwareBERTForPretraining


def load_initial_bert_weights(bert_model, checkpoint_path, logger):
    """
    Load a BERT backbone checkpoint into the pretraining encoder.

    Supported formats:
    1. raw bert state dict
    2. {'bert_state_dict': {...}}
    3. {'model_state_dict': {...}} where BERT keys are prefixed with 'bert.'
    """
    logger.info(f"Loading initial BERT weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        if 'bert_state_dict' in ckpt:
            state_dict = ckpt['bert_state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = {}
            for key, value in ckpt['model_state_dict'].items():
                if key.startswith('module.bert.'):
                    state_dict[key[len('module.bert.'):]] = value
                elif key.startswith('bert.'):
                    state_dict[key[len('bert.'):]] = value
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()

    if any(key.startswith('module.') for key in state_dict):
        state_dict = {
            key[len('module.'):]: value
            for key, value in state_dict.items()
        }

    current_state = bert_model.state_dict()
    filtered_state = {}
    resized_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = bert_model.load_state_dict(filtered_state, strict=False)
    logger.info(f"Loaded {len(filtered_state)} matching BERT tensors from init checkpoint")
    if skipped_keys:
        logger.warning(
            f"Skipped {len(skipped_keys)} init tensors due to shape mismatch: "
            f"{skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}"
        )
    if missing:
        logger.info(
            f"Model kept {len(missing)} encoder tensors from current initialization: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        logger.info(
            f"Ignored {len(unexpected)} unexpected init tensors: "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )


def load_initial_model_weights(model, checkpoint_path, logger):
    """
    Load a pretraining checkpoint into the full model with strict=False.

    This is useful when adding new heads: matching existing tensors are loaded,
    while newly introduced heads remain randomly initialized.
    """
    logger.info(f"Loading initial full-model weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()

    if any(key.startswith('module.') for key in state_dict):
        state_dict = {
            key[len('module.'):]: value
            for key, value in state_dict.items()
        }

    current_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logger.info(f"Loaded {len(filtered_state)} matching full-model tensors from init checkpoint")
    if skipped_keys:
        logger.warning(
            f"Skipped {len(skipped_keys)} init tensors due to shape mismatch: "
            f"{skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}"
        )
    if missing:
        logger.info(
            f"Model kept {len(missing)} tensors from current initialization: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        logger.info(
            f"Ignored {len(unexpected)} unexpected init tensors: "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )


def move_embedding_inputs(sample, device):
    binary_pos = sample['binary_pos'].to(device, non_blocking=True)
    function_pos = sample['function_pos'].to(device, non_blocking=True)
    bb_pos = sample['bb_pos'].to(device, non_blocking=True)
    var_offsets = sample['var_offsets'].to(device, non_blocking=True)

    if 'section_kind_ids' in sample:
        section_kind_ids = sample['section_kind_ids'].to(device, non_blocking=True)
    else:
        section_kind_ids = torch.zeros_like(var_offsets)

    if 'addr_type_ids' in sample:
        addr_type_ids = sample['addr_type_ids'].to(device, non_blocking=True)
    else:
        addr_type_ids = torch.zeros_like(var_offsets)

    return binary_pos, function_pos, bb_pos, var_offsets, section_kind_ids, addr_type_ids


def train_epoch(model, data_loader, scope_loader, same_bb_loader, same_function_loader,
                optimizer, device, log_freq=1000, logger=None,
                enable_imc=False, enable_imd=False, enable_mlm=True, enable_nsp_cfg=False,
                enable_scope=False, enable_same_bb=False, enable_same_function=False,
                same_bb_weight=1.0, same_function_weight=1.0,
                gradient_accumulation_steps=1, use_amp=False, scaler=None):
    """
    Train for one epoch with IMC, IMD, MLM, and Scope tasks.

    Args:
        model: AddressAwareBERTForPretraining model
        data_loader: DataLoader for IMC+IMD+MLM data
        scope_loader: DataLoader for scope prediction data
        optimizer: Optimizer
        device: Device to train on
        log_freq: Log every N batches
        logger: Logger instance
        enable_imc: Enable instruction masking for CFG
        enable_imd: Enable instruction masking for DFG
        enable_mlm: Enable masked language modeling task
        enable_scope: Enable scope prediction task
        gradient_accumulation_steps: Accumulate gradients over N batches before
            stepping the optimizer (effective_batch = batch_size × steps)

    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_imc_loss = 0
    total_imd_loss = 0
    total_mlm_loss = 0
    total_nsp_cfg_loss = 0
    total_scope_loss = 0
    total_same_bb_loss = 0
    total_same_function_loss = 0
    total_loss = 0

    # Loss functions
    # MaskedLanguageModel head applies LogSoftmax internally → must use NLLLoss, not CrossEntropyLoss
    # (CrossEntropyLoss applies its own log_softmax, causing double log-softmax and wrong gradients)
    imc_criterion = nn.NLLLoss(ignore_index=-1)  # Ignore non-masked tokens
    imd_criterion = nn.NLLLoss(ignore_index=-1)
    mlm_criterion = nn.NLLLoss(ignore_index=-1)
    nsp_cfg_criterion = nn.NLLLoss()
    scope_criterion = nn.NLLLoss()
    same_bb_criterion = nn.NLLLoss()
    same_function_criterion = nn.NLLLoss()
    
    # Scope iterator
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    same_bb_iter = iter(same_bb_loader) if (enable_same_bb and same_bb_loader) else None
    same_function_iter = iter(same_function_loader) if (enable_same_function and same_function_loader) else None
    
    # Create progress bar with manual updates
    progress_bar = tqdm(total=len(data_loader), desc="Training", disable=False)
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(data_loader):
        # === Process IMC (Instruction Masking CFG) ===
        imc_loss = torch.tensor(0.0, device=device)
        if enable_imc:
            imc_batch = batch['imc']
            
            # Validate token IDs BEFORE moving to GPU
            imc_input_cpu = imc_batch['bert_input']
            imc_segment_cpu = imc_batch['segment_label']
            vocab_size = model.module.bert.embedding.token_embedding.num_embeddings if hasattr(model, 'module') else model.bert.embedding.token_embedding.num_embeddings
            segment_vocab_size = model.module.bert.embedding.segment_embedding.num_embeddings if hasattr(model, 'module') else model.bert.embedding.segment_embedding.num_embeddings
            
            # Check token IDs
            invalid_mask = (imc_input_cpu < 0) | (imc_input_cpu >= vocab_size)
            if invalid_mask.any():
                invalid_ids = imc_input_cpu[invalid_mask].unique()
                logger.error(f"Invalid token IDs found in IMC input: {invalid_ids.tolist()}")
                logger.error(f"Vocab size: {vocab_size}, Min ID: {imc_input_cpu.min()}, Max ID: {imc_input_cpu.max()}")
                # Print first few samples with invalid IDs
                for i in range(min(3, imc_input_cpu.size(0))):
                    if invalid_mask[i].any():
                        logger.error(f"Sample {i} invalid tokens: {imc_input_cpu[i][invalid_mask[i]].tolist()}")
                raise ValueError(f"Token IDs out of bounds: {invalid_ids.tolist()}")
            
            # Check segment IDs
            invalid_seg_mask = (imc_segment_cpu < 0) | (imc_segment_cpu >= segment_vocab_size)
            if invalid_seg_mask.any():
                invalid_seg_ids = imc_segment_cpu[invalid_seg_mask].unique()
                logger.error(f"Invalid segment IDs found in IMC input: {invalid_seg_ids.tolist()}")
                logger.error(f"Segment vocab size: {segment_vocab_size}, Min ID: {imc_segment_cpu.min()}, Max ID: {imc_segment_cpu.max()}")
                raise ValueError(f"Segment IDs out of bounds: {invalid_seg_ids.tolist()}")
            
            imc_input = imc_input_cpu.to(device, non_blocking=True)
            imc_labels = imc_batch['bert_label'].to(device, non_blocking=True)
            imc_segment = imc_segment_cpu.to(device, non_blocking=True)
            (
                imc_binary_pos,
                imc_function_pos,
                imc_bb_pos,
                imc_var_offsets,
                imc_section_kind_ids,
                imc_addr_type_ids,
            ) = move_embedding_inputs(imc_batch, device)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    imc_output = model.module.forward_im(
                        imc_input, imc_segment,
                        imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets,
                        section_kind_ids=imc_section_kind_ids,
                        addr_type_ids=imc_addr_type_ids,
                    )
                else:
                    imc_output = model.forward_im(
                        imc_input, imc_segment,
                        imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets,
                        section_kind_ids=imc_section_kind_ids,
                        addr_type_ids=imc_addr_type_ids,
                    )

                imc_output = imc_output.view(-1, imc_output.size(-1))
                imc_labels_flat = imc_labels.view(-1)
                imc_loss = imc_criterion(imc_output, imc_labels_flat)
        
        # === Process IMD (Instruction Masking DFG) ===
        imd_loss = torch.tensor(0.0, device=device)
        if enable_imd and 'imd' in batch:
            imd_batch = batch['imd']
            
            imd_input = imd_batch['bert_input'].to(device, non_blocking=True)
            imd_labels = imd_batch['bert_label'].to(device, non_blocking=True)
            imd_segment = imd_batch['segment_label'].to(device, non_blocking=True)
            (
                imd_binary_pos,
                imd_function_pos,
                imd_bb_pos,
                imd_var_offsets,
                imd_section_kind_ids,
                imd_addr_type_ids,
            ) = move_embedding_inputs(imd_batch, device)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    imd_output = model.module.forward_im(
                        imd_input, imd_segment,
                        imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets,
                        section_kind_ids=imd_section_kind_ids,
                        addr_type_ids=imd_addr_type_ids,
                    )
                else:
                    imd_output = model.forward_im(
                        imd_input, imd_segment,
                        imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets,
                        section_kind_ids=imd_section_kind_ids,
                        addr_type_ids=imd_addr_type_ids,
                    )

                imd_output = imd_output.view(-1, imd_output.size(-1))
                imd_labels_flat = imd_labels.view(-1)
                imd_loss = imd_criterion(imd_output, imd_labels_flat)
        
        # === Process MLM / CFG NSP (shared CFG pair path) ===
        mlm_loss = torch.tensor(0.0, device=device)
        nsp_cfg_loss = torch.tensor(0.0, device=device)
        if enable_mlm or enable_nsp_cfg:
            mlm_batch = batch['mlm']
            
            mlm_input = mlm_batch['bert_input'].to(device, non_blocking=True)
            mlm_labels = mlm_batch['bert_label'].to(device, non_blocking=True)
            mlm_segment = mlm_batch['segment_label'].to(device, non_blocking=True)
            (
                mlm_binary_pos,
                mlm_function_pos,
                mlm_bb_pos,
                mlm_var_offsets,
                mlm_section_kind_ids,
                mlm_addr_type_ids,
            ) = move_embedding_inputs(mlm_batch, device)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                mlm_output, nsp_output = model(
                    mlm_input, mlm_segment,
                    mlm_binary_pos, mlm_function_pos, mlm_bb_pos, mlm_var_offsets,
                    section_kind_ids=mlm_section_kind_ids,
                    addr_type_ids=mlm_addr_type_ids,
                    corpus_type='cfg',
                )

                if enable_mlm:
                    mlm_output = mlm_output.view(-1, mlm_output.size(-1))
                    mlm_labels_flat = mlm_labels.view(-1)
                    mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)

                if enable_nsp_cfg:
                    nsp_labels = mlm_batch['nsp_label'].to(device, non_blocking=True)
                    nsp_cfg_loss = nsp_cfg_criterion(nsp_output, nsp_labels)
        
        # === Process Scope (if available and enabled) ===
        scope_loss = torch.tensor(0.0, device=device)
        if enable_scope and scope_iter is not None:
            try:
                scope_batch = next(scope_iter)
            except StopIteration:
                scope_iter = iter(scope_loader)
                scope_batch = next(scope_iter)
            
            scope_token_ids = scope_batch['bert_input'].to(device, non_blocking=True)
            scope_segment_labels = scope_batch['segment_label'].to(device, non_blocking=True)
            (
                scope_binary_pos,
                scope_function_pos,
                scope_bb_pos,
                scope_var_offsets,
                scope_section_kind_ids,
                scope_addr_type_ids,
            ) = move_embedding_inputs(scope_batch, device)
            scope_labels = scope_batch['scope_label'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    scope_output = model.module.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets,
                        section_kind_ids=scope_section_kind_ids,
                        addr_type_ids=scope_addr_type_ids,
                    )
                else:
                    scope_output = model.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets,
                        section_kind_ids=scope_section_kind_ids,
                        addr_type_ids=scope_addr_type_ids,
                    )

                scope_loss = scope_criterion(scope_output, scope_labels)

        # === Process SAME_BB auxiliary pair classification ===
        same_bb_loss = torch.tensor(0.0, device=device)
        if enable_same_bb and same_bb_iter is not None:
            try:
                same_bb_batch = next(same_bb_iter)
            except StopIteration:
                same_bb_iter = iter(same_bb_loader)
                same_bb_batch = next(same_bb_iter)

            same_bb_input = same_bb_batch['bert_input'].to(device, non_blocking=True)
            same_bb_segment = same_bb_batch['segment_label'].to(device, non_blocking=True)
            (
                same_bb_binary_pos,
                same_bb_function_pos,
                same_bb_bb_pos,
                same_bb_var_offsets,
                same_bb_section_kind_ids,
                same_bb_addr_type_ids,
            ) = move_embedding_inputs(same_bb_batch, device)
            same_bb_labels = same_bb_batch['pair_label'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    same_bb_output = model.module.forward_pair(
                        same_bb_input, same_bb_segment,
                        same_bb_binary_pos, same_bb_function_pos, same_bb_bb_pos, same_bb_var_offsets,
                        section_kind_ids=same_bb_section_kind_ids,
                        addr_type_ids=same_bb_addr_type_ids,
                        pair_type='same_bb',
                    )
                else:
                    same_bb_output = model.forward_pair(
                        same_bb_input, same_bb_segment,
                        same_bb_binary_pos, same_bb_function_pos, same_bb_bb_pos, same_bb_var_offsets,
                        section_kind_ids=same_bb_section_kind_ids,
                        addr_type_ids=same_bb_addr_type_ids,
                        pair_type='same_bb',
                    )
                same_bb_loss = same_bb_criterion(same_bb_output, same_bb_labels)

        # === Process SAME_FUNCTION auxiliary pair classification ===
        same_function_loss = torch.tensor(0.0, device=device)
        if enable_same_function and same_function_iter is not None:
            try:
                same_function_batch = next(same_function_iter)
            except StopIteration:
                same_function_iter = iter(same_function_loader)
                same_function_batch = next(same_function_iter)

            same_function_input = same_function_batch['bert_input'].to(device, non_blocking=True)
            same_function_segment = same_function_batch['segment_label'].to(device, non_blocking=True)
            (
                same_function_binary_pos,
                same_function_function_pos,
                same_function_bb_pos,
                same_function_var_offsets,
                same_function_section_kind_ids,
                same_function_addr_type_ids,
            ) = move_embedding_inputs(same_function_batch, device)
            same_function_labels = same_function_batch['pair_label'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    same_function_output = model.module.forward_pair(
                        same_function_input, same_function_segment,
                        same_function_binary_pos, same_function_function_pos, same_function_bb_pos, same_function_var_offsets,
                        section_kind_ids=same_function_section_kind_ids,
                        addr_type_ids=same_function_addr_type_ids,
                        pair_type='same_function',
                    )
                else:
                    same_function_output = model.forward_pair(
                        same_function_input, same_function_segment,
                        same_function_binary_pos, same_function_function_pos, same_function_bb_pos, same_function_var_offsets,
                        section_kind_ids=same_function_section_kind_ids,
                        addr_type_ids=same_function_addr_type_ids,
                        pair_type='same_function',
                    )
                same_function_loss = same_function_criterion(same_function_output, same_function_labels)
        
        # Combined loss
        loss = (
            imc_loss
            + imd_loss
            + mlm_loss
            + nsp_cfg_loss
            + scope_loss
            + same_bb_weight * same_bb_loss
            + same_function_weight * same_function_loss
        )

        # Gradient accumulation: scale loss then backward; step only every N batches
        scaled_loss = loss / gradient_accumulation_steps
        if use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or \
           (batch_idx + 1) == len(data_loader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Accumulate losses
        total_imc_loss += imc_loss.item()
        total_imd_loss += imd_loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_cfg_loss += nsp_cfg_loss.item()
        total_scope_loss += scope_loss.item()
        total_same_bb_loss += same_bb_loss.item()
        total_same_function_loss += same_function_loss.item()
        total_loss += loss.item()
        
        # Update progress bar every 100 iterations
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'imc': f'{total_imc_loss/(batch_idx+1):.4f}' if enable_imc else '0',
                'imd': f'{total_imd_loss/(batch_idx+1):.4f}' if enable_imd else '0',
                'mlm': f'{total_mlm_loss/(batch_idx+1):.4f}' if enable_mlm else '0',
                'nsp': f'{total_nsp_cfg_loss/(batch_idx+1):.4f}' if enable_nsp_cfg else '0',
                'scope': f'{total_scope_loss/(batch_idx+1):.4f}' if enable_scope else '0',
                'sbb': f'{total_same_bb_loss/(batch_idx+1):.4f}' if enable_same_bb else '0',
                'sfn': f'{total_same_function_loss/(batch_idx+1):.4f}' if enable_same_function else '0',
            })
            progress_bar.update(100 if batch_idx > 0 else 1)
        
        # Log to file periodically
        if (batch_idx + 1) % log_freq == 0 and logger:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Batch {batch_idx+1}/{len(data_loader)} - "
                      f"Loss: {avg_loss:.4f} | IMC: {total_imc_loss/(batch_idx+1):.4f} | "
                      f"IMD: {total_imd_loss/(batch_idx+1):.4f} | "
                      f"MLM: {total_mlm_loss/(batch_idx+1):.4f} | "
                      f"NSP_CFG: {total_nsp_cfg_loss/(batch_idx+1):.4f} | "
                      f"SCOPE: {total_scope_loss/(batch_idx+1):.4f} | "
                      f"SAME_BB: {total_same_bb_loss/(batch_idx+1):.4f} | "
                      f"SAME_FUNCTION: {total_same_function_loss/(batch_idx+1):.4f}")
    
    # Close progress bar
    progress_bar.close()
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'imc_loss': total_imc_loss / n_batches,
        'imd_loss': total_imd_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'nsp_cfg_loss': total_nsp_cfg_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
        'same_bb_loss': total_same_bb_loss / n_batches,
        'same_function_loss': total_same_function_loss / n_batches,
    }


def validate_epoch(model, data_loader, scope_loader, same_bb_loader, same_function_loader,
                   device, logger=None, enable_imc=False, enable_imd=False, enable_mlm=True,
                   enable_nsp_cfg=False, enable_scope=False, enable_same_bb=False,
                   enable_same_function=False, same_bb_weight=1.0, same_function_weight=1.0,
                   use_amp=False):
    """Validate for one epoch"""
    model.eval()
    
    total_imc_loss = 0
    total_imd_loss = 0
    total_mlm_loss = 0
    total_nsp_cfg_loss = 0
    total_scope_loss = 0
    total_same_bb_loss = 0
    total_same_function_loss = 0
    total_loss = 0
    
    imc_criterion = nn.NLLLoss(ignore_index=-1)
    imd_criterion = nn.NLLLoss(ignore_index=-1)
    mlm_criterion = nn.NLLLoss(ignore_index=-1)
    nsp_cfg_criterion = nn.NLLLoss()
    scope_criterion = nn.NLLLoss()
    same_bb_criterion = nn.NLLLoss()
    same_function_criterion = nn.NLLLoss()
    
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    same_bb_iter = iter(same_bb_loader) if (enable_same_bb and same_bb_loader) else None
    same_function_iter = iter(same_function_loader) if (enable_same_function and same_function_loader) else None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # === Process IMC ===
            imc_loss = torch.tensor(0.0, device=device)
            if enable_imc:
                imc_batch = batch['imc']
                
                imc_input = imc_batch['bert_input'].to(device, non_blocking=True)
                imc_labels = imc_batch['bert_label'].to(device, non_blocking=True)
                imc_segment = imc_batch['segment_label'].to(device, non_blocking=True)
                (
                    imc_binary_pos,
                    imc_function_pos,
                    imc_bb_pos,
                    imc_var_offsets,
                    imc_section_kind_ids,
                    imc_addr_type_ids,
                ) = move_embedding_inputs(imc_batch, device)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        imc_output = model.module.forward_im(
                            imc_input, imc_segment,
                            imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets,
                            section_kind_ids=imc_section_kind_ids,
                            addr_type_ids=imc_addr_type_ids,
                        )
                    else:
                        imc_output = model.forward_im(
                            imc_input, imc_segment,
                            imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets,
                            section_kind_ids=imc_section_kind_ids,
                            addr_type_ids=imc_addr_type_ids,
                        )

                    imc_output = imc_output.view(-1, imc_output.size(-1))
                    imc_labels_flat = imc_labels.view(-1)
                    imc_loss = imc_criterion(imc_output, imc_labels_flat)
            
            # === Process IMD ===
            imd_loss = torch.tensor(0.0, device=device)
            if enable_imd and 'imd' in batch:
                imd_batch = batch['imd']
                
                imd_input = imd_batch['bert_input'].to(device, non_blocking=True)
                imd_labels = imd_batch['bert_label'].to(device, non_blocking=True)
                imd_segment = imd_batch['segment_label'].to(device, non_blocking=True)
                (
                    imd_binary_pos,
                    imd_function_pos,
                    imd_bb_pos,
                    imd_var_offsets,
                    imd_section_kind_ids,
                    imd_addr_type_ids,
                ) = move_embedding_inputs(imd_batch, device)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        imd_output = model.module.forward_im(
                            imd_input, imd_segment,
                            imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets,
                            section_kind_ids=imd_section_kind_ids,
                            addr_type_ids=imd_addr_type_ids,
                        )
                    else:
                        imd_output = model.forward_im(
                            imd_input, imd_segment,
                            imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets,
                            section_kind_ids=imd_section_kind_ids,
                            addr_type_ids=imd_addr_type_ids,
                        )

                    imd_output = imd_output.view(-1, imd_output.size(-1))
                    imd_labels_flat = imd_labels.view(-1)
                    imd_loss = imd_criterion(imd_output, imd_labels_flat)
            
            # === Process MLM / CFG NSP ===
            mlm_loss = torch.tensor(0.0, device=device)
            nsp_cfg_loss = torch.tensor(0.0, device=device)
            if enable_mlm or enable_nsp_cfg:
                mlm_batch = batch['mlm']
                
                mlm_input = mlm_batch['bert_input'].to(device, non_blocking=True)
                mlm_labels = mlm_batch['bert_label'].to(device, non_blocking=True)
                mlm_segment = mlm_batch['segment_label'].to(device, non_blocking=True)
                (
                    mlm_binary_pos,
                    mlm_function_pos,
                    mlm_bb_pos,
                    mlm_var_offsets,
                    mlm_section_kind_ids,
                    mlm_addr_type_ids,
                ) = move_embedding_inputs(mlm_batch, device)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    mlm_output, nsp_output = model(
                        mlm_input, mlm_segment,
                        mlm_binary_pos, mlm_function_pos, mlm_bb_pos, mlm_var_offsets,
                        section_kind_ids=mlm_section_kind_ids,
                        addr_type_ids=mlm_addr_type_ids,
                        corpus_type='cfg',
                    )

                    if enable_mlm:
                        mlm_output = mlm_output.view(-1, mlm_output.size(-1))
                        mlm_labels_flat = mlm_labels.view(-1)
                        mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)

                    if enable_nsp_cfg:
                        nsp_labels = mlm_batch['nsp_label'].to(device, non_blocking=True)
                        nsp_cfg_loss = nsp_cfg_criterion(nsp_output, nsp_labels)
            
            # Scope
            scope_loss = torch.tensor(0.0, device=device)
            if enable_scope and scope_iter is not None:
                try:
                    scope_batch = next(scope_iter)
                except StopIteration:
                    scope_iter = iter(scope_loader)
                    scope_batch = next(scope_iter)
                
                scope_token_ids = scope_batch['bert_input'].to(device, non_blocking=True)
                scope_segment_labels = scope_batch['segment_label'].to(device, non_blocking=True)
                (
                    scope_binary_pos,
                    scope_function_pos,
                    scope_bb_pos,
                    scope_var_offsets,
                    scope_section_kind_ids,
                    scope_addr_type_ids,
                ) = move_embedding_inputs(scope_batch, device)
                scope_labels = scope_batch['scope_label'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        scope_output = model.module.forward_scope(
                            scope_token_ids, scope_segment_labels,
                            scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets,
                            section_kind_ids=scope_section_kind_ids,
                            addr_type_ids=scope_addr_type_ids,
                        )
                    else:
                        scope_output = model.forward_scope(
                            scope_token_ids, scope_segment_labels,
                            scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets,
                            section_kind_ids=scope_section_kind_ids,
                            addr_type_ids=scope_addr_type_ids,
                        )

                    scope_loss = scope_criterion(scope_output, scope_labels)

            same_bb_loss = torch.tensor(0.0, device=device)
            if enable_same_bb and same_bb_iter is not None:
                try:
                    same_bb_batch = next(same_bb_iter)
                except StopIteration:
                    same_bb_iter = iter(same_bb_loader)
                    same_bb_batch = next(same_bb_iter)

                same_bb_input = same_bb_batch['bert_input'].to(device, non_blocking=True)
                same_bb_segment = same_bb_batch['segment_label'].to(device, non_blocking=True)
                (
                    same_bb_binary_pos,
                    same_bb_function_pos,
                    same_bb_bb_pos,
                    same_bb_var_offsets,
                    same_bb_section_kind_ids,
                    same_bb_addr_type_ids,
                ) = move_embedding_inputs(same_bb_batch, device)
                same_bb_labels = same_bb_batch['pair_label'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        same_bb_output = model.module.forward_pair(
                            same_bb_input, same_bb_segment,
                            same_bb_binary_pos, same_bb_function_pos, same_bb_bb_pos, same_bb_var_offsets,
                            section_kind_ids=same_bb_section_kind_ids,
                            addr_type_ids=same_bb_addr_type_ids,
                            pair_type='same_bb',
                        )
                    else:
                        same_bb_output = model.forward_pair(
                            same_bb_input, same_bb_segment,
                            same_bb_binary_pos, same_bb_function_pos, same_bb_bb_pos, same_bb_var_offsets,
                            section_kind_ids=same_bb_section_kind_ids,
                            addr_type_ids=same_bb_addr_type_ids,
                            pair_type='same_bb',
                        )
                    same_bb_loss = same_bb_criterion(same_bb_output, same_bb_labels)

            same_function_loss = torch.tensor(0.0, device=device)
            if enable_same_function and same_function_iter is not None:
                try:
                    same_function_batch = next(same_function_iter)
                except StopIteration:
                    same_function_iter = iter(same_function_loader)
                    same_function_batch = next(same_function_iter)

                same_function_input = same_function_batch['bert_input'].to(device, non_blocking=True)
                same_function_segment = same_function_batch['segment_label'].to(device, non_blocking=True)
                (
                    same_function_binary_pos,
                    same_function_function_pos,
                    same_function_bb_pos,
                    same_function_var_offsets,
                    same_function_section_kind_ids,
                    same_function_addr_type_ids,
                ) = move_embedding_inputs(same_function_batch, device)
                same_function_labels = same_function_batch['pair_label'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        same_function_output = model.module.forward_pair(
                            same_function_input, same_function_segment,
                            same_function_binary_pos, same_function_function_pos, same_function_bb_pos, same_function_var_offsets,
                            section_kind_ids=same_function_section_kind_ids,
                            addr_type_ids=same_function_addr_type_ids,
                            pair_type='same_function',
                        )
                    else:
                        same_function_output = model.forward_pair(
                            same_function_input, same_function_segment,
                            same_function_binary_pos, same_function_function_pos, same_function_bb_pos, same_function_var_offsets,
                            section_kind_ids=same_function_section_kind_ids,
                            addr_type_ids=same_function_addr_type_ids,
                            pair_type='same_function',
                        )
                    same_function_loss = same_function_criterion(same_function_output, same_function_labels)

            loss = (
                imc_loss
                + imd_loss
                + mlm_loss
                + nsp_cfg_loss
                + scope_loss
                + same_bb_weight * same_bb_loss
                + same_function_weight * same_function_loss
            )

            total_imc_loss += imc_loss.item()
            total_imd_loss += imd_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_cfg_loss += nsp_cfg_loss.item()
            total_scope_loss += scope_loss.item()
            total_same_bb_loss += same_bb_loss.item()
            total_same_function_loss += same_function_loss.item()
            total_loss += loss.item()

    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'imc_loss': total_imc_loss / n_batches,
        'imd_loss': total_imd_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'nsp_cfg_loss': total_nsp_cfg_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
        'same_bb_loss': total_same_bb_loss / n_batches,
        'same_function_loss': total_same_function_loss / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Address-Aware BERT with Instruction Masking')
    
    # Data args
    parser.add_argument("--cfg_train", type=str, required=True, help="Training CFG corpus")
    parser.add_argument("--dfg_train", type=str, default=None, help="Training DFG corpus (optional, only needed for IMD)")
    parser.add_argument("--cfg_val", type=str, help="Validation CFG corpus")
    parser.add_argument("--dfg_val", type=str, help="Validation DFG corpus")
    parser.add_argument("--scope_train", type=str, help="Training scope corpus")
    parser.add_argument("--scope_val", type=str, help="Validation scope corpus")
    parser.add_argument("--same_bb_train", type=str, help="Training same_bb pair corpus (JSONL)")
    parser.add_argument("--same_bb_val", type=str, help="Validation same_bb pair corpus (JSONL)")
    parser.add_argument("--same_function_train", type=str, help="Training same_function pair corpus (JSONL)")
    parser.add_argument("--same_function_val", type=str, help="Validation same_function pair corpus (JSONL)")
    # Test data (only used for vocab generation, not training)
    parser.add_argument("--cfg_test", type=str, help="Test CFG corpus (for vocab only)")
    parser.add_argument("--dfg_test", type=str, help="Test DFG corpus (for vocab only)")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    
    # Model args
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=60, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N batches before stepping "
                             "(effective_batch_size = batch_size × N)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute BERT activations during backward to save ~40%% GPU memory")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision training/inference on CUDA")
    
    # Masking args
    parser.add_argument("--token_mask_prob", type=float, default=0.15, help="Token-level mask probability (MLM)")
    parser.add_argument("--instruction_mask_prob", type=float, default=0.15, help="Instruction-level mask probability (IM)")
    
    # Data sampling
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of training data to use")
    parser.add_argument("--val_percentage", type=float, default=1.0, help="Percentage of validation data to use")
    
    # Task flags
    parser.add_argument("--enable_imc", action="store_true", help="Enable Instruction Masking for CFG")
    parser.add_argument("--disable_imc", action="store_true", help="Disable Instruction Masking for CFG")
    parser.add_argument("--enable_imd", action="store_true", help="Enable Instruction Masking for DFG")
    parser.add_argument("--enable_nsp_cfg", action="store_true", help="Enable CFG Next Sentence Prediction")
    parser.add_argument("--enable_mlm", action="store_true", default=True, help="Enable MLM")
    parser.add_argument("--disable_mlm", action="store_true", help="Disable MLM")
    parser.add_argument("--enable_scope", action="store_true", help="Enable Scope Prediction")
    parser.add_argument("--enable_same_bb", action="store_true", help="Enable auxiliary same_bb pair classification")
    parser.add_argument("--enable_same_function", action="store_true", help="Enable auxiliary same_function pair classification")
    parser.add_argument("--use_address_embedding", action="store_true", default=True, help="Use address embeddings")
    parser.add_argument("--disable_address_embedding", action="store_true", help="Disable address embeddings")
    parser.add_argument("--use_var_embedding", action="store_true", default=True, help="Use var offset embeddings for var(0xXX) tokens")
    parser.add_argument("--disable_var_embedding", action="store_true", help="Disable var offset embeddings")
    
    # Output args
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=1000, help="Log every N batches")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint if available")
    parser.add_argument("--init_bert", type=str, default=None,
                        help="Initial BERT checkpoint to continue pretraining from "
                             "(ignored when --resume finds checkpoint_latest.pt)")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Initial full-model checkpoint to continue pretraining from "
                             "with partial loading (ignored when --resume finds checkpoint_latest.pt)")
    
    # Device args
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pair_max_instructions", type=int, default=2,
                        help="Max instructions per side for same_function auxiliary pairs")
    parser.add_argument("--same_bb_weight", type=float, default=1.0, help="Loss weight for auxiliary same_bb task")
    parser.add_argument("--same_function_weight", type=float, default=1.0, help="Loss weight for auxiliary same_function task")
    
    args = parser.parse_args()
    
    # Handle disable flags
    if args.disable_imc:
        args.enable_imc = False
    if args.disable_mlm:
        args.enable_mlm = False
    if args.disable_address_embedding:
        args.use_address_embedding = False
    if args.disable_var_embedding:
        args.use_var_embedding = False
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Setup device — supports torchrun DDP (LOCAL_RANK set) and single-GPU/DataParallel
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_ddp = local_rank != -1
    if is_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    is_main_process = (not is_ddp) or dist.get_rank() == 0
    print(f"Using device: {device}  (DDP={is_ddp}, local_rank={local_rank})")

    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_latest.pt')
    resume_available = args.resume and os.path.exists(checkpoint_path)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Arguments saved to: {os.path.join(args.output_dir, 'args.json')}")
    
    # Load or create vocabulary
    # Include train, val, and test data for vocab generation
    # Test data is only used for vocab, not for training
    data_files = [
        args.cfg_train,
        args.dfg_train,
        args.cfg_val,
        args.dfg_val,
        args.scope_train,
        args.scope_val,
        args.cfg_test,  # Only for vocab generation
        args.dfg_test,  # Only for vocab generation
    ]
    vocab = create_vocab(data_files, args.vocab, logger=logger)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    logger.info("Creating training dataset with instruction masking...")
    
    # Only load data if at least one task is enabled
    if not (
        args.enable_imc or args.enable_imd or args.enable_mlm or args.enable_nsp_cfg
        or args.enable_same_bb or args.enable_same_function
    ):
        logger.error("At least one task must be enabled (IMC, IMD, MLM, NSP_CFG, SAME_BB, or SAME_FUNCTION)")
        return
    
    train_dataset = InstructionMaskingDataset(
        cfg_corpus_path=args.cfg_train if args.enable_imc or args.enable_mlm or args.enable_nsp_cfg else None,
        dfg_corpus_path=args.dfg_train if args.enable_imd else None,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        data_percentage=args.data_percentage,
        train_split=1.0,
        is_train=True,
        enable_imd=args.enable_imd,
        enable_imc=args.enable_imc,
        enable_mlm=args.enable_mlm,
        enable_cfg_nsp=args.enable_nsp_cfg,
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    
    # Create validation dataset
    val_loader = None
    has_cfg_val = bool(args.cfg_val)
    has_required_dfg_val = (not args.enable_imd) or bool(args.dfg_val)
    if has_cfg_val and has_required_dfg_val:
        logger.info("Creating validation dataset...")
        val_dataset = InstructionMaskingDataset(
            cfg_corpus_path=args.cfg_val if args.enable_imc or args.enable_mlm or args.enable_nsp_cfg else None,
            dfg_corpus_path=args.dfg_val if args.enable_imd else None,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            data_percentage=args.val_percentage,
            train_split=1.0,
            is_train=True,
            enable_imd=args.enable_imd,
            enable_imc=args.enable_imc,
            enable_mlm=args.enable_mlm,
            enable_cfg_nsp=args.enable_nsp_cfg,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )
    elif args.cfg_val and args.enable_imd and not args.dfg_val:
        logger.warning("cfg_val was provided but dfg_val is missing while IMD is enabled; skipping validation loader.")
    
    same_bb_train_loader = None
    same_bb_val_loader = None
    if args.enable_same_bb:
        if not args.same_bb_train:
            logger.warning("SAME_BB is enabled but --same_bb_train is missing; disabling SAME_BB for this run.")
            args.enable_same_bb = False
        else:
            logger.info("Creating SAME_BB auxiliary dataset...")
            same_bb_train_dataset = StructurePairDataset(
                args.same_bb_train,
                vocab=vocab,
                seq_len=args.seq_len,
                max_instructions_per_side=1,
            )
            same_bb_train_loader = DataLoader(
                same_bb_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(args.num_workers > 0),
            )
            if args.same_bb_val:
                same_bb_val_dataset = StructurePairDataset(
                    args.same_bb_val,
                    vocab=vocab,
                    seq_len=args.seq_len,
                    max_instructions_per_side=1,
                )
                same_bb_val_loader = DataLoader(
                    same_bb_val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == 'cuda'),
                    persistent_workers=(args.num_workers > 0),
                )

    same_function_train_loader = None
    same_function_val_loader = None
    if args.enable_same_function:
        if not args.same_function_train:
            logger.warning("SAME_FUNCTION is enabled but --same_function_train is missing; disabling SAME_FUNCTION for this run.")
            args.enable_same_function = False
        else:
            logger.info("Creating SAME_FUNCTION auxiliary dataset...")
            same_function_train_dataset = StructurePairDataset(
                args.same_function_train,
                vocab=vocab,
                seq_len=args.seq_len,
                max_instructions_per_side=args.pair_max_instructions,
            )
            same_function_train_loader = DataLoader(
                same_function_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(args.num_workers > 0),
            )
            if args.same_function_val:
                same_function_val_dataset = StructurePairDataset(
                    args.same_function_val,
                    vocab=vocab,
                    seq_len=args.seq_len,
                    max_instructions_per_side=args.pair_max_instructions,
                )
                same_function_val_loader = DataLoader(
                    same_function_val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == 'cuda'),
                    persistent_workers=(args.num_workers > 0),
                )

    # Scope datasets removed — keep variables for API compatibility
    scope_train_loader = None
    scope_val_loader = None
    
    # Create model
    logger.info("Creating model...")
    logger.info("Task configuration:")
    logger.info(f"  - IMC: {args.enable_imc}")
    logger.info(f"  - IMD: {args.enable_imd}")
    logger.info(f"  - MLM: {args.enable_mlm}")
    logger.info(f"  - NSP_CFG: {args.enable_nsp_cfg}")
    logger.info(f"  - SCOPE: {args.enable_scope}")
    logger.info(f"  - SAME_BB: {args.enable_same_bb}")
    logger.info(f"  - SAME_FUNCTION: {args.enable_same_function}")
    logger.info(f"  - Address Embedding: {args.use_address_embedding}")
    logger.info(f"  - Var Embedding: {args.use_var_embedding}")
    
    bert = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        max_len=args.seq_len,
        use_address_embedding=args.use_address_embedding,
        use_var_embedding=args.use_var_embedding
    )

    if not resume_available and args.init_bert and not args.init_checkpoint:
        load_initial_bert_weights(bert, args.init_bert, logger)
    
    model = AddressAwareBERTForPretraining(
        bert,
        vocab_size=len(vocab),
        enable_mlm=args.enable_mlm,
        enable_imc=args.enable_imc,
        enable_imd=args.enable_imd,
        enable_nsp_cfg=args.enable_nsp_cfg,
        enable_nsp_dfg=False,
        enable_scope=args.enable_scope,
        enable_same_bb=args.enable_same_bb,
        enable_same_function=args.enable_same_function,
    )

    if not resume_available and args.init_checkpoint:
        load_initial_model_weights(model, args.init_checkpoint, logger)
    
    model = model.to(device)

    # Enable gradient checkpointing if requested (saves ~40% GPU memory)
    if args.gradient_checkpointing:
        model.bert.gradient_checkpointing = True
        if is_main_process:
            logger.info("Gradient checkpointing ENABLED (saves memory, slightly slower)")

    # Wrap with DDP (torchrun launch) or DataParallel (single-node fallback)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if is_main_process:
            logger.info(f"DDP enabled across {dist.get_world_size()} GPUs")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"DataParallel: {torch.cuda.device_count()} GPUs")

    # Set section-aware daddr token IDs AFTER parallel wrapping.
    daddr_token_ids = sorted(
        token_id
        for token, token_id in vocab.stoi.items()
        if token == 'daddr' or token.startswith('daddr_')
    )
    bert_model = model.module if hasattr(model, 'module') else model
    if daddr_token_ids:
        bert_model.bert.embedding.set_special_token_ids(daddr_token_ids)
        logger.info(f"Set daddr token IDs for address routing: {daddr_token_ids}")
    else:
        logger.warning("No daddr section tokens found in vocab — daddr routing will rely on addr_type_ids only")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps} "
                f"(effective batch size = {args.batch_size * args.gradient_accumulation_steps})")
    logger.info(f"AMP enabled: {args.amp and device.type == 'cuda'}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == 'cuda'))
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    if resume_available:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        if args.resume:
            logger.info("Resume requested but no checkpoint found at " + checkpoint_path)
        if args.init_checkpoint:
            logger.info("Starting pretraining from provided full-model initialization.")
        elif args.init_bert:
            logger.info("Starting pretraining from provided BERT initialization.")
        else:
            logger.info("Starting training from scratch.")
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)  # ensures different shuffling per epoch
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, scope_train_loader, same_bb_train_loader, same_function_train_loader,
            optimizer, device, args.log_freq, logger,
            enable_imc=args.enable_imc,
            enable_imd=args.enable_imd,
            enable_mlm=args.enable_mlm,
            enable_nsp_cfg=args.enable_nsp_cfg,
            enable_scope=args.enable_scope,
            enable_same_bb=args.enable_same_bb,
            enable_same_function=args.enable_same_function,
            same_bb_weight=args.same_bb_weight,
            same_function_weight=args.same_function_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=(args.amp and device.type == 'cuda'),
            scaler=scaler,
        )
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  - IMC: {train_metrics['imc_loss']:.4f}")
        logger.info(f"  - IMD: {train_metrics['imd_loss']:.4f}")
        logger.info(f"  - MLM: {train_metrics['mlm_loss']:.4f}")
        logger.info(f"  - NSP_CFG: {train_metrics['nsp_cfg_loss']:.4f}")
        logger.info(f"  - Scope: {train_metrics['scope_loss']:.4f}")
        logger.info(f"  - SAME_BB: {train_metrics['same_bb_loss']:.4f}")
        logger.info(f"  - SAME_FUNCTION: {train_metrics['same_function_loss']:.4f}")
        
        # Validate
        if val_loader:
            val_metrics = validate_epoch(
                model, val_loader, scope_val_loader, same_bb_val_loader, same_function_val_loader, device, logger,
                enable_imc=args.enable_imc,
                enable_imd=args.enable_imd,
                enable_mlm=args.enable_mlm,
                enable_nsp_cfg=args.enable_nsp_cfg,
                enable_scope=args.enable_scope,
                enable_same_bb=args.enable_same_bb,
                enable_same_function=args.enable_same_function,
                same_bb_weight=args.same_bb_weight,
                same_function_weight=args.same_function_weight,
                use_amp=(args.amp and device.type == 'cuda')
            )
            
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  - IMC: {val_metrics['imc_loss']:.4f}")
            logger.info(f"  - IMD: {val_metrics['imd_loss']:.4f}")
            logger.info(f"  - MLM: {val_metrics['mlm_loss']:.4f}")
            logger.info(f"  - NSP_CFG: {val_metrics['nsp_cfg_loss']:.4f}")
            logger.info(f"  - Scope: {val_metrics['scope_loss']:.4f}")
            logger.info(f"  - SAME_BB: {val_metrics['same_bb_loss']:.4f}")
            logger.info(f"  - SAME_FUNCTION: {val_metrics['same_function_loss']:.4f}")
            
            # Check for improvement
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                epochs_without_improvement = 0
                
                if is_main_process:
                    # Save best BERT only
                    bert_state = model.module.bert.state_dict() if hasattr(model, 'module') else model.bert.state_dict()
                    torch.save(bert_state, os.path.join(args.output_dir, 'best_bert.pt'))
                    logger.info(f"✓ New best BERT saved! Val loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save checkpoint every epoch so SLURM job can resume after OOM kill
        if is_main_process:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'best_val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
                'train_metrics': train_metrics,
            }
            # Overwrite latest checkpoint (used for resuming)
            torch.save(checkpoint_data, checkpoint_path)
            # Also keep a permanent per-epoch full checkpoint
            epoch_ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint_data, epoch_ckpt_path)
            logger.info(f"Full checkpoint saved: checkpoint_epoch_{epoch+1}.pt")

            # Save latest BERT weights every epoch for verification
            bert_state = model.module.bert.state_dict() if hasattr(model, 'module') else model.bert.state_dict()
            torch.save(bert_state, os.path.join(args.output_dir, 'latest_bert.pt'))
            logger.info(f"Latest BERT saved (epoch {epoch+1}): latest_bert.pt")
            # Save per-epoch BERT weights so every epoch is preserved
            epoch_bert_path = os.path.join(args.output_dir, f'bert_epoch_{epoch+1}.pt')
            torch.save(bert_state, epoch_bert_path)
            logger.info(f"Per-epoch BERT saved: bert_epoch_{epoch+1}.pt")
        
        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
        
        scheduler.step()
    
    if is_main_process:
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Models saved to: {args.output_dir}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
