"""
Extract the BERT encoder weights from a pretraining checkpoint.

Usage:
    python extract_bert.py \
        --checkpoint /path/to/checkpoint_latest.pt \
        --output     /path/to/bert.pt
"""

import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint_latest.pt or checkpoint_epoch_N.pt')
    parser.add_argument('--output',     required=True, help='Path to save the extracted BERT weights')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    full_state = ckpt['model_state_dict']
    print(f"  Epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"  Keys  : {len(full_state)} total parameter tensors")

    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith('module.') for k in full_state):
        full_state = {k[len('module.'):]: v for k, v in full_state.items()}
        print("  Stripped DataParallel 'module.' prefix")

    # Extract only keys that belong to the BERT encoder (prefix 'bert.')
    bert_state = {k[len('bert.'):]: v for k, v in full_state.items() if k.startswith('bert.')}
    print(f"  Extracted {len(bert_state)} BERT encoder tensors")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(bert_state, args.output)
    print(f"Saved BERT weights to: {args.output}")

if __name__ == '__main__':
    main()
