"""
Inspect vocabulary file and display token-to-id mappings.

This script loads a vocab.pkl file and displays:
1. Vocabulary statistics
2. Special tokens
3. All token-to-id mappings
4. Sample tokens with their IDs
"""

import os
import sys
import argparse

try:
    from .vocab import WordVocab
except ImportError:
    from vocab import WordVocab

def inspect_vocab(vocab_path, output_file=None, show_all=False, limit=100):
    """
    Load and inspect vocabulary file.
    
    Args:
        vocab_path: Path to vocab.pkl file
        output_file: Optional output file to save mappings
        show_all: If True, show all tokens; if False, show first `limit` tokens
        limit: Number of tokens to display if not showing all
    """
    print("=" * 80)
    print("VOCABULARY INSPECTOR")
    print("=" * 80)
    print(f"Loading vocabulary from: {vocab_path}")
    
    if not os.path.exists(vocab_path):
        print(f"ERROR: File not found: {vocab_path}")
        sys.exit(1)
    
    # Load vocabulary
    try:
        vocab = WordVocab.load_vocab(vocab_path)
        print(f"✓ Successfully loaded vocabulary")
    except Exception as e:
        print(f"ERROR: Failed to load vocabulary: {e}")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("VOCABULARY STATISTICS")
    print("=" * 80)
    print(f"Total vocabulary size: {len(vocab):,}")
    print(f"Number of tokens (itos): {len(vocab.itos):,}")
    print(f"Number of mappings (stoi): {len(vocab.stoi):,}")
    
    # Special token indices
    print()
    print("Special Token Indices:")
    print(f"  PAD index: {vocab.pad_index}")
    print(f"  UNK index: {vocab.unk_index}")
    print(f"  EOS index: {vocab.eos_index}")
    print(f"  SOS index: {vocab.sos_index}")
    print(f"  MASK index: {vocab.mask_index}")
    
    # Show special tokens
    print()
    print("=" * 80)
    print("SPECIAL TOKENS")
    print("=" * 80)
    special_tokens = ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>']
    for token in special_tokens:
        if token in vocab.stoi:
            print(f"  {token:20s} -> ID: {vocab.stoi[token]}")
    
    # Show frequency information if available
    if hasattr(vocab, 'freqs') and vocab.freqs:
        print()
        print("=" * 80)
        print("TOP 20 MOST FREQUENT TOKENS")
        print("=" * 80)
        # Sort by frequency
        sorted_freqs = sorted(vocab.freqs.items(), key=lambda x: x[1], reverse=True)
        for i, (token, freq) in enumerate(sorted_freqs[:20]):
            token_id = vocab.stoi.get(token, -1)
            print(f"  {i+1:2d}. {token:30s} -> ID: {token_id:6d}  Freq: {freq:,}")
    
    # Show all or sample tokens
    print()
    print("=" * 80)
    if show_all:
        print(f"ALL TOKEN-TO-ID MAPPINGS ({len(vocab.itos)} tokens)")
    else:
        print(f"SAMPLE TOKEN-TO-ID MAPPINGS (first {limit} tokens)")
    print("=" * 80)
    
    # Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TOKEN-TO-ID MAPPINGS")
    output_lines.append("=" * 80)
    output_lines.append(f"Vocabulary size: {len(vocab)}")
    output_lines.append("")
    output_lines.append(f"{'ID':<10} {'Token':<50} {'Frequency':<15}")
    output_lines.append("-" * 80)
    
    # Display tokens
    num_to_show = len(vocab.itos) if show_all else min(limit, len(vocab.itos))
    
    for i in range(num_to_show):
        token = vocab.itos[i]
        token_id = i
        freq = vocab.freqs.get(token, 0) if hasattr(vocab, 'freqs') else 'N/A'
        
        # Print to console
        if freq != 'N/A':
            print(f"  {token_id:<10d} {token:<50s} {freq:,}")
        else:
            print(f"  {token_id:<10d} {token:<50s} {freq}")
        
        # Add to output file
        if freq != 'N/A':
            output_lines.append(f"{token_id:<10d} {token:<50s} {freq:,}")
        else:
            output_lines.append(f"{token_id:<10d} {token:<50s} {freq}")
    
    if not show_all and len(vocab.itos) > limit:
        remaining = len(vocab.itos) - limit
        print(f"\n  ... and {remaining:,} more tokens ...")
        print(f"\n  Use --all flag to see all tokens")
        output_lines.append("")
        output_lines.append(f"... and {remaining:,} more tokens ...")
    
    # Save to file if requested
    if output_file:
        print()
        print("=" * 80)
        print(f"Saving mappings to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"✓ Saved {len(output_lines)} lines to {output_file}")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Vocabulary file: {vocab_path}")
    print(f"Total tokens: {len(vocab):,}")
    print(f"Special tokens: {len(special_tokens)}")
    if output_file:
        print(f"Output file: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect vocabulary file and display token-to-id mappings"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="./vocab_8.pkl",
        help="Path to vocabulary file (default: ./vocab.pkl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save token mappings (optional)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all tokens (default: show first 100)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of tokens to display if not showing all (default: 100)"
    )
    
    args = parser.parse_args()
    
    inspect_vocab(
        vocab_path=args.vocab,
        output_file=args.output,
        show_all=args.all,
        limit=args.limit
    )
