#!/bin/bash
#SBATCH --job-name=addr_pretrain_8
#SBATCH --output=/work/kliu14/Palmtree_mlm/log/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/palmtree

./run_hpc.sh

echo "Done!"