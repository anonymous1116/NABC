#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --partition=v100
#SBATCH --array=0-99
#SBATCH --output=output_log_calib/SDE/output_log_%A_%a.out
#SBATCH --error=output_log_calib/SDE/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log_calib/SDE

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

# Compute x0_ind and seed from SLURM_ARRAY_TASK_ID
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))

L=1000000000 
tol=1e-5
num_training_mean=300000
num_training_cov=600000
layer_len=64
echo "Running with x0_ind=${x0_ind}, seed=${seed}, L=${L}, tol=${tol}, num_training=${num_training_mean}"
python SDE/SDE_calibrating2.py \
    --x0_ind ${x0_ind} \
    --seed ${seed} \
    --L $L \
    --task OU \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --tol $tol \
    --layer_len $layer_len

#python SDE/SDE_calibrating.py --x0_ind 1 --seed 1 --L 10000000 --task "OU" --layer_len 16 --num_training_mean 300000 --num_training_cov 600000 --tol 1e-3

