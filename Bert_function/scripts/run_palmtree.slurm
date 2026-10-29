#!/bin/bash
#SBATCH --job-name=palmtree_ctx2
#SBATCH --output=/work/kliu14/Palmtree_mlm/log/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_GPUS: ${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-<unset>}}"
echo ""

set +u
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/palmtree
set -u

PROJECT_ROOT="/path/to/numAnchor/PalmTree"
cd "${PROJECT_ROOT}/scripts"

# Optional env overrides:
# export BATCH_SIZE=768
# export TRAIN_SAMPLE_RATIO=0.70
# export EPOCHS=10

bash ./run_palmtree.sh

echo "Done!"
