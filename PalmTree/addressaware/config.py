"""
Configuration reference for strupos training scripts.

NOTE: This file is for REFERENCE ONLY. 
Actual training uses command-line arguments in run_*.sh scripts.
"""

# ==================== Data Paths (Reference) ====================
# These paths are used in shell scripts, not loaded directly
cfg_train = "/path/to/numAnchor_artifact/dataset/train_cfg.txt"
dfg_train = "/path/to/numAnchor_artifact/dataset/train_dfg.txt"
cfg_val = "/path/to/numAnchor_artifact/dataset/val_cfg.txt"
dfg_val = "/path/to/numAnchor_artifact/dataset/val_dfg.txt"
cfg_test = "/path/to/numAnchor_artifact/dataset/test_cfg.txt"
dfg_test = "/path/to/numAnchor_artifact/dataset/test_dfg.txt"
vocab_path = "./vocab.txt"

# ==================== Model Architecture (Reference) ====================
hidden = 768 
layers = 12
attn_heads = 12
seq_len = 100
dropout = 0.1

# ==================== Training Hyperparameters (Reference) ====================
epochs = 10
batch_size = 1024
lr = 1e-4
warmup_steps = 10000
num_workers = 4
early_stopping_patience = 5
seed = 42  # Random seed for reproducibility

# ==================== Data Processing (Reference) ====================
mask_prob = 0.15
nsp_prob = 0.5
data_percentage = 1.0
train_split = 0.9

# To train with different configurations, modify the parameters in run_*.sh scripts

