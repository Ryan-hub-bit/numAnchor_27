"""
Generate vocabulary from all CFG and DFG data (train, val, test)

Uses the WordVocab class for fast vocabulary creation.
"""

import os
import sys
import re

try:
    from .vocab import WordVocab
except ImportError:
    from vocab import WordVocab


LEGACY_DADDR_RE = re.compile(r'\bdaddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')


def canonicalize_legacy_daddr_tokens(line):
    """
    Convert legacy `daddr(...:section_kind_id)` tokens into section-specific
    tokens so vocab creation stays aligned with the runtime parser.
    """
    def repl(match):
        try:
            section_kind = int(float(match.group(4)))
        except (TypeError, ValueError):
            section_kind = 4
        token_name = {
            1: "daddr_rodata",
            2: "daddr_bss",
            3: "daddr_data",
            4: "daddr_other",
        }.get(section_kind, "daddr_other")
        return f"{token_name}({match.group(1)}:{match.group(2)}:{match.group(3)})"

    return LEGACY_DADDR_RE.sub(repl, line)


def preprocess_line(line):
    """
    Remove all address information in parentheses from a line.
    Also filter out standalone hex address tokens.
    
    Examples:
        mov(0x401000:0.5:0.3:0.2) eax ebx -> mov eax ebx
        address(0x123:0.5:0.3:0.2) -> address
        var(0x10028) -> var
        lea rax 0x12345 -> lea rax  (filter out hex address)
    
    Returns cleaned tokens as a list.
    """
    line = canonicalize_legacy_daddr_tokens(line)

    # Remove all patterns like (0xADDR:pos1:pos2:pos3) - address positions
    cleaned = re.sub(r'\(0x[0-9a-fA-F]+(?::[0-9.]+){2,3}\)', '', line)
    # Remove all patterns like (0xADDR) - var offsets, e.g., var(0x10028) -> var
    cleaned = re.sub(r'\(0x[0-9a-fA-F]+\)', '', cleaned)
    # Split by whitespace and tab, filter out empty strings and standalone hex addresses
    tokens = []
    for tok in cleaned.replace('\t', ' ').split():
        if tok and not re.match(r'^0x[0-9a-fA-F]+$', tok):
            tokens.append(tok)
    return tokens


class PreprocessedFile:
    """
    Wrapper that preprocesses lines from a file by removing address info.
    Acts as an iterator that yields lists of tokens.
    """
    def __init__(self, file_handle):
        self.file_handle = file_handle
    
    def __iter__(self):
        for line in self.file_handle:
            yield preprocess_line(line)


def create_vocab(data_files, vocab_path, max_size=13000, min_freq=2, logger=None):
    """
    Create vocabulary from data files if it doesn't exist.
    
    Args:
        data_files: List of data file paths to build vocabulary from
        vocab_path: Path to save vocabulary
        max_size: Maximum vocabulary size
        min_freq: Minimum token frequency
        logger: Optional logger instance
    
    Returns:
        WordVocab instance
    """
    # Check if vocab already exists
    if os.path.exists(vocab_path):
        if logger:
            logger.info(f"Loading existing vocabulary from {vocab_path}")
        else:
            print(f"Loading existing vocabulary from {vocab_path}")
        return WordVocab.load_vocab(vocab_path)
    
    # Vocab doesn't exist, create it
    if logger:
        logger.info(f"Vocabulary file not found: {vocab_path}")
        logger.info("Creating vocabulary from data files...")
    else:
        print(f"Vocabulary file not found: {vocab_path}")
        print("Creating vocabulary from data files...")
    
    # Filter existing files
    files_to_use = [f for f in data_files if f and os.path.exists(f)]
    
    if not files_to_use:
        raise FileNotFoundError("No data files found to build vocabulary!")
    
    for fpath in files_to_use:
        if logger:
            logger.info(f"  Using: {fpath}")
        else:
            print(f"  Using: {fpath}")
    
    # Open all files and build vocabulary
    file_handles = [open(f, 'r', encoding='utf-8') for f in files_to_use]
    preprocessed = [PreprocessedFile(fh) for fh in file_handles]
    
    vocab = WordVocab(preprocessed, max_size=max_size, min_freq=min_freq)
    
    # Close file handles
    for fh in file_handles:
        fh.close()
    
    # Save vocabulary
    vocab_dir = os.path.dirname(vocab_path)
    if vocab_dir:
        os.makedirs(vocab_dir, exist_ok=True)
    vocab.save_vocab(vocab_path)
    
    if logger:
        logger.info(f"Vocabulary created with {len(vocab)} tokens")
        logger.info(f"Vocabulary saved to: {vocab_path}")
    else:
        print(f"Vocabulary created with {len(vocab)} tokens")
        print(f"Vocabulary saved to: {vocab_path}")
    
    return vocab


if __name__ == "__main__":
    # Data file - currently only have train data
    train_cfg_dataset = "/path/to/numAnchor_artifact/addressaware/cfg_train/cfg_8inline_pretrain.txt"
    
    vocab_path = "./vocab_8.pkl"
    
    print("=" * 60)
    print("Creating Vocabulary from Train Data")
    print("=" * 60)
    print(f"Using WordVocab with max_size=13000, min_freq=1")
    print(f"Output: {vocab_path} (pickle format)")
    print()
    
    # Check if file exists
    if not os.path.exists(train_cfg_dataset):
        print(f"ERROR: File not found: {train_cfg_dataset}")
        sys.exit(1)
    
    print(f"Found: {train_cfg_dataset}")
    print()
    
    # Open file and build vocabulary with preprocessing
    # Wrap file handle with PreprocessedFile to remove address info
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
        
        # Wrap file with preprocessing
        preprocessed_files = [
            PreprocessedFile(f1)
        ]
        
        vocab = WordVocab(
            preprocessed_files,
            max_size=13000,
            min_freq=1
        )
    
    print()
    print("VOCAB SIZE:", len(vocab))
    print()
    
    # Save vocabulary
    vocab.save_vocab(vocab_path)
    print(f"Vocabulary saved to: {vocab_path}")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
