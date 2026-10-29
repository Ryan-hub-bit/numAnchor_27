"""
Training script for function-level Address-Aware BERT pretraining.

This fork expects one FULL FUNCTION per line in the input corpus. Inside each
function it supports:
1. IMC - mask entire instructions in CFG/function corpora
2. IMD - mask entire instructions in DFG/function corpora
3. MLM - standard token-level masking
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


# Import local modules
from vocab import WordVocab
from create_vocab import create_vocab
from dataloader import InstructionMaskingDataset
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


def align_checkpoint_state_dict(model_state_dict, target_state_dict, logger):
    """Align checkpoint key prefixes with the current model before resume."""
    target_keys = set(target_state_dict.keys())
    loaded_keys = list(model_state_dict.keys())

    if set(loaded_keys) == target_keys:
        return model_state_dict

    candidates = [("identity", model_state_dict)]

    if any(key.startswith("module.") for key in loaded_keys):
        stripped = {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in model_state_dict.items()
        }
        candidates.append(("strip_module", stripped))

    if any(not key.startswith("module.") for key in loaded_keys):
        added = {
            key if key.startswith("module.") else f"module.{key}": value
            for key, value in model_state_dict.items()
        }
        candidates.append(("add_module", added))

    best_name = "identity"
    best_state = model_state_dict
    best_overlap = len(target_keys.intersection(loaded_keys))

    for name, candidate in candidates[1:]:
        overlap = len(target_keys.intersection(candidate.keys()))
        if overlap > best_overlap:
            best_name = name
            best_state = candidate
            best_overlap = overlap

    logger.info(
        f"Resume checkpoint key alignment: {best_name} "
        f"(matched {best_overlap}/{len(target_keys)} model tensors)"
    )
    return best_state


def train_epoch(model, data_loader, scope_loader, optimizer, device, log_freq=1000, logger=None,
                enable_imc=False, enable_imd=False, enable_mlm=True, enable_scope=False,
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
    total_scope_loss = 0
    total_loss = 0

    # Loss functions
    # MaskedLanguageModel head applies LogSoftmax internally → must use NLLLoss, not CrossEntropyLoss
    # (CrossEntropyLoss applies its own log_softmax, causing double log-softmax and wrong gradients)
    imc_criterion = nn.NLLLoss(ignore_index=-1)  # Ignore non-masked tokens
    imd_criterion = nn.NLLLoss(ignore_index=-1)
    mlm_criterion = nn.NLLLoss(ignore_index=-1)
    scope_criterion = nn.NLLLoss()
    
    # Scope iterator
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    
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
            imc_binary_pos = imc_batch['binary_pos'].to(device, non_blocking=True)
            imc_function_pos = imc_batch['function_pos'].to(device, non_blocking=True)
            imc_bb_pos = imc_batch['bb_pos'].to(device, non_blocking=True)
            imc_var_offsets = imc_batch['var_offsets'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    imc_output = model.module.forward_im(
                        imc_input, imc_segment,
                        imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
                    )
                else:
                    imc_output = model.forward_im(
                        imc_input, imc_segment,
                        imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
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
            imd_binary_pos = imd_batch['binary_pos'].to(device, non_blocking=True)
            imd_function_pos = imd_batch['function_pos'].to(device, non_blocking=True)
            imd_bb_pos = imd_batch['bb_pos'].to(device, non_blocking=True)
            imd_var_offsets = imd_batch['var_offsets'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    imd_output = model.module.forward_im(
                        imd_input, imd_segment,
                        imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                    )
                else:
                    imd_output = model.forward_im(
                        imd_input, imd_segment,
                        imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                    )

                imd_output = imd_output.view(-1, imd_output.size(-1))
                imd_labels_flat = imd_labels.view(-1)
                imd_loss = imd_criterion(imd_output, imd_labels_flat)
        
        # === Process MLM (Token-level Masking) ===
        mlm_loss = torch.tensor(0.0, device=device)
        if enable_mlm:
            mlm_batch = batch['mlm']
            
            mlm_input = mlm_batch['bert_input'].to(device, non_blocking=True)
            mlm_labels = mlm_batch['bert_label'].to(device, non_blocking=True)
            mlm_segment = mlm_batch['segment_label'].to(device, non_blocking=True)
            mlm_binary_pos = mlm_batch['binary_pos'].to(device, non_blocking=True)
            mlm_function_pos = mlm_batch['function_pos'].to(device, non_blocking=True)
            mlm_bb_pos = mlm_batch['bb_pos'].to(device, non_blocking=True)
            mlm_var_offsets = mlm_batch['var_offsets'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                # Use standard forward pass for MLM
                mlm_output, _ = model(
                    mlm_input, mlm_segment,
                    mlm_binary_pos, mlm_function_pos, mlm_bb_pos, mlm_var_offsets
                )

                mlm_output = mlm_output.view(-1, mlm_output.size(-1))
                mlm_labels_flat = mlm_labels.view(-1)
                mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)
        
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
            scope_binary_pos = scope_batch['binary_pos'].to(device, non_blocking=True)
            scope_function_pos = scope_batch['function_pos'].to(device, non_blocking=True)
            scope_bb_pos = scope_batch['bb_pos'].to(device, non_blocking=True)
            scope_var_offsets = scope_batch['var_offsets'].to(device, non_blocking=True)
            scope_labels = scope_batch['scope_label'].to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                if hasattr(model, 'module'):
                    scope_output = model.module.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                    )
                else:
                    scope_output = model.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                    )

                scope_loss = scope_criterion(scope_output, scope_labels)
        
        # Combined loss
        loss = imc_loss + imd_loss + mlm_loss + scope_loss

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
        total_scope_loss += scope_loss.item()
        total_loss += loss.item()
        
        # Update progress bar every 100 iterations
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'imc': f'{total_imc_loss/(batch_idx+1):.4f}' if enable_imc else '0',
                'imd': f'{total_imd_loss/(batch_idx+1):.4f}' if enable_imd else '0',
                'mlm': f'{total_mlm_loss/(batch_idx+1):.4f}' if enable_mlm else '0',
                'scope': f'{total_scope_loss/(batch_idx+1):.4f}' if enable_scope else '0',
            })
            progress_bar.update(100 if batch_idx > 0 else 1)
        
        # Log to file periodically
        if (batch_idx + 1) % log_freq == 0 and logger:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Batch {batch_idx+1}/{len(data_loader)} - "
                      f"Loss: {avg_loss:.4f} | IMC: {total_imc_loss/(batch_idx+1):.4f} | "
                      f"IMD: {total_imd_loss/(batch_idx+1):.4f} | "
                      f"MLM: {total_mlm_loss/(batch_idx+1):.4f} | "
                      f"SCOPE: {total_scope_loss/(batch_idx+1):.4f}")
    
    # Close progress bar
    progress_bar.close()
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'imc_loss': total_imc_loss / n_batches,
        'imd_loss': total_imd_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
    }


def validate_epoch(model, data_loader, scope_loader, device, logger=None,
                   enable_imc=False, enable_imd=False, enable_mlm=True, enable_scope=False,
                   use_amp=False):
    """Validate for one epoch"""
    model.eval()
    
    total_imc_loss = 0
    total_imd_loss = 0
    total_mlm_loss = 0
    total_scope_loss = 0
    total_loss = 0
    
    imc_criterion = nn.NLLLoss(ignore_index=-1)
    imd_criterion = nn.NLLLoss(ignore_index=-1)
    mlm_criterion = nn.NLLLoss(ignore_index=-1)
    scope_criterion = nn.NLLLoss()
    
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # === Process IMC ===
            imc_loss = torch.tensor(0.0, device=device)
            if enable_imc:
                imc_batch = batch['imc']
                
                imc_input = imc_batch['bert_input'].to(device, non_blocking=True)
                imc_labels = imc_batch['bert_label'].to(device, non_blocking=True)
                imc_segment = imc_batch['segment_label'].to(device, non_blocking=True)
                imc_binary_pos = imc_batch['binary_pos'].to(device, non_blocking=True)
                imc_function_pos = imc_batch['function_pos'].to(device, non_blocking=True)
                imc_bb_pos = imc_batch['bb_pos'].to(device, non_blocking=True)
                imc_var_offsets = imc_batch['var_offsets'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        imc_output = model.module.forward_im(
                            imc_input, imc_segment,
                            imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
                        )
                    else:
                        imc_output = model.forward_im(
                            imc_input, imc_segment,
                            imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
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
                imd_binary_pos = imd_batch['binary_pos'].to(device, non_blocking=True)
                imd_function_pos = imd_batch['function_pos'].to(device, non_blocking=True)
                imd_bb_pos = imd_batch['bb_pos'].to(device, non_blocking=True)
                imd_var_offsets = imd_batch['var_offsets'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        imd_output = model.module.forward_im(
                            imd_input, imd_segment,
                            imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                        )
                    else:
                        imd_output = model.forward_im(
                            imd_input, imd_segment,
                            imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                        )

                    imd_output = imd_output.view(-1, imd_output.size(-1))
                    imd_labels_flat = imd_labels.view(-1)
                    imd_loss = imd_criterion(imd_output, imd_labels_flat)
            
            # === Process MLM ===
            mlm_loss = torch.tensor(0.0, device=device)
            if enable_mlm:
                mlm_batch = batch['mlm']
                
                mlm_input = mlm_batch['bert_input'].to(device, non_blocking=True)
                mlm_labels = mlm_batch['bert_label'].to(device, non_blocking=True)
                mlm_segment = mlm_batch['segment_label'].to(device, non_blocking=True)
                mlm_binary_pos = mlm_batch['binary_pos'].to(device, non_blocking=True)
                mlm_function_pos = mlm_batch['function_pos'].to(device, non_blocking=True)
                mlm_bb_pos = mlm_batch['bb_pos'].to(device, non_blocking=True)
                mlm_var_offsets = mlm_batch['var_offsets'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    mlm_output, _ = model(
                        mlm_input, mlm_segment,
                        mlm_binary_pos, mlm_function_pos, mlm_bb_pos, mlm_var_offsets
                    )

                    mlm_output = mlm_output.view(-1, mlm_output.size(-1))
                    mlm_labels_flat = mlm_labels.view(-1)
                    mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)
            
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
                scope_binary_pos = scope_batch['binary_pos'].to(device, non_blocking=True)
                scope_function_pos = scope_batch['function_pos'].to(device, non_blocking=True)
                scope_bb_pos = scope_batch['bb_pos'].to(device, non_blocking=True)
                scope_var_offsets = scope_batch['var_offsets'].to(device, non_blocking=True)
                scope_labels = scope_batch['scope_label'].to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) \
                    if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    if hasattr(model, 'module'):
                        scope_output = model.module.forward_scope(
                            scope_token_ids, scope_segment_labels,
                            scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                        )
                    else:
                        scope_output = model.forward_scope(
                            scope_token_ids, scope_segment_labels,
                            scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                        )

                    scope_loss = scope_criterion(scope_output, scope_labels)

            loss = imc_loss + imd_loss + mlm_loss + scope_loss

            total_imc_loss += imc_loss.item()
            total_imd_loss += imd_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_scope_loss += scope_loss.item()
            total_loss += loss.item()

    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'imc_loss': total_imc_loss / n_batches,
        'imd_loss': total_imd_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train function-level Address-Aware BERT')
    
    # Data args
    parser.add_argument("--cfg_train", type=str, required=True, help="Training function-inline CFG corpus (one function per line)")
    parser.add_argument("--dfg_train", type=str, default=None, help="Training function-inline DFG corpus (optional, only needed for IMD)")
    parser.add_argument("--cfg_val", type=str, help="Validation function-inline CFG corpus")
    parser.add_argument("--dfg_val", type=str, help="Validation function-inline DFG corpus")
    parser.add_argument("--scope_train", type=str, help="Training scope corpus")
    parser.add_argument("--scope_val", type=str, help="Validation scope corpus")
    # Test data (only used for vocab generation, not training)
    parser.add_argument("--cfg_test", type=str, help="Test CFG corpus (for vocab only)")
    parser.add_argument("--dfg_test", type=str, help="Test DFG corpus (for vocab only)")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    
    # Model args
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum tokens per function sample")
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
    parser.add_argument("--enable_mlm", action="store_true", default=True, help="Enable MLM")
    parser.add_argument("--disable_mlm", action="store_true", help="Disable MLM")
    parser.add_argument("--enable_scope", action="store_true", help="Enable Scope Prediction")
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
    
    # Device args
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
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
    logger.info("Creating function-level training dataset...")
    
    # Only load data if at least one task is enabled
    if not (args.enable_imc or args.enable_imd or args.enable_mlm):
        logger.error("At least one task must be enabled (IMC, IMD, or MLM)")
        return
    
    train_dataset = InstructionMaskingDataset(
        cfg_corpus_path=args.cfg_train if args.enable_imc or args.enable_mlm else None,
        dfg_corpus_path=args.dfg_train if args.enable_imd else None,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        data_percentage=args.data_percentage,
        train_split=1.0,
        is_train=True,
        enable_imd=args.enable_imd
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
        logger.info("Creating function-level validation dataset...")
        val_dataset = InstructionMaskingDataset(
            cfg_corpus_path=args.cfg_val if args.enable_imc or args.enable_mlm else None,
            dfg_corpus_path=args.dfg_val if args.enable_imd else None,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            data_percentage=args.val_percentage,
            train_split=1.0,
            is_train=True,
            enable_imd=args.enable_imd
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
    
    # Scope datasets removed — keep variables for API compatibility
    scope_train_loader = None
    scope_val_loader = None
    
    # Create model
    logger.info("Creating model...")
    logger.info("Task configuration:")
    logger.info(f"  - IMC: {args.enable_imc}")
    logger.info(f"  - IMD: {args.enable_imd}")
    logger.info(f"  - MLM: {args.enable_mlm}")
    logger.info(f"  - SCOPE: {args.enable_scope}")
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

    if not resume_available and args.init_bert:
        load_initial_bert_weights(bert, args.init_bert, logger)
    
    model = AddressAwareBERTForPretraining(
        bert,
        vocab_size=len(vocab),
        enable_mlm=args.enable_mlm,
        enable_imc=args.enable_imc,
        enable_imd=args.enable_imd,
        enable_nsp_cfg=False,
        enable_nsp_dfg=False,
        enable_scope=args.enable_scope
    )
    
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

    # Set daddr_token_id AFTER parallel wrapping
    daddr_token_id = vocab.stoi.get('daddr', None)
    bert_model = model.module if hasattr(model, 'module') else model
    if daddr_token_id is not None:
        bert_model.bert.embedding.set_special_token_ids(daddr_token_id)
        logger.info(f"Set daddr_token_id = {daddr_token_id} for address routing")
    else:
        logger.warning("'daddr' token not found in vocab — daddr address embedding will be zero")
    
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
        resume_state = align_checkpoint_state_dict(
            checkpoint['model_state_dict'],
            model.state_dict(),
            logger,
        )
        model.load_state_dict(resume_state)
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
        if args.init_bert:
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
            model, train_loader, scope_train_loader,
            optimizer, device, args.log_freq, logger,
            enable_imc=args.enable_imc,
            enable_imd=args.enable_imd,
            enable_mlm=args.enable_mlm,
            enable_scope=args.enable_scope,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=(args.amp and device.type == 'cuda'),
            scaler=scaler,
        )
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  - IMC: {train_metrics['imc_loss']:.4f}")
        logger.info(f"  - IMD: {train_metrics['imd_loss']:.4f}")
        logger.info(f"  - MLM: {train_metrics['mlm_loss']:.4f}")
        logger.info(f"  - Scope: {train_metrics['scope_loss']:.4f}")
        
        # Validate
        if val_loader:
            logger.info("Starting validation...")
            val_metrics = validate_epoch(
                model, val_loader, scope_val_loader, device, logger,
                enable_imc=args.enable_imc,
                enable_imd=args.enable_imd,
                enable_mlm=args.enable_mlm,
                enable_scope=args.enable_scope,
                use_amp=(args.amp and device.type == 'cuda')
            )
            logger.info("Validation complete.")
            
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  - IMC: {val_metrics['imc_loss']:.4f}")
            logger.info(f"  - IMD: {val_metrics['imd_loss']:.4f}")
            logger.info(f"  - MLM: {val_metrics['mlm_loss']:.4f}")
            logger.info(f"  - Scope: {val_metrics['scope_loss']:.4f}")
            
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
