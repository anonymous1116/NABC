#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:57:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=170G
#SBATCH --qos=normal
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --array=0-99
#SBATCH --output=output_log_calib/DP/output_log_%A_%a.out
#SBATCH --error=output_log_calib/DP/error_log_%A_%a.txt

# #SBATCH --partition=a10,a100-40gb,a100-80gb

# Create the output_log directory if it doesn't exist
mkdir -p output_log_calib/DP

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

# Compute x0_ind and seed from SLURM_ARRAY_TASK_ID
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))

L=10000000
tol=1e-3
num_training_mean=50000
num_training_cov=100000

echo "Running with x0_ind=${x0_ind}, seed=${seed}, L=${L}, tol=${tol}, num_training=${num_training_mean}"
python DP/DP_calibrating.py \
    --x0_ind ${x0_ind} \
    --seed ${seed} \
    --L $L \
    --task cont_full2\
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --tol $tol

#python DP/DP_calibrating.py \
#    --x0_ind 3 \
#    --seed 2 \
#    --L 10000000 \
#    --task cont_table_dp_transform \
#    --num_training_mean 50000 \
#    --num_training_cov 100000 \
#    --tol 1e-3


#python benchmark/benchmark_calibrating_withcov.py --x0_ind 1 --seed 1 --L 10000000 --task "slcp_summary" --num_training_mean 300000 --num_training_cov 600000 --tol 1e-3
