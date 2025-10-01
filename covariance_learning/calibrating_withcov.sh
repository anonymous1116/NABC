#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=170G
#SBATCH --qos=normal
#SBATCH --partition=v100
#SBATCH --array=0-9
#SBATCH --output=output_log_calib/slcp/output_log_%A_%a.out
#SBATCH --error=output_log_calib/slcp/error_log_%A_%a.txt

# #SBATCH --partition=a10,a100-40gb,a100-80gb

# Create the output_log directory if it doesn't exist
mkdir -p output_log_calib/slcp

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

# Compute x0_ind and seed from SLURM_ARRAY_TASK_ID
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))

L=100000000000 
tol=1e-7
num_training_mean=300000
num_training_cov=600000

echo "Running with x0_ind=${x0_ind}, seed=${seed}, L=${L}, tol=${tol}, num_training=${num_training_mean}"
python benchmark/benchmark_calibrating_withcov_exp.py \
    --x0_ind ${x0_ind} \
    --seed ${seed} \
    --L $L \
    --task slcp \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --tol $tol

#python benchmark/benchmark_calibrating_withcov_exp.py \
#    --x0_ind 1 \
#    --seed 2 \
#    --L 10000000 \
#    --task slcp \
#    --num_training_mean 300000 \
#    --num_training_cov 600000 \
#    --tol 1e-3


#tol_pilot=1e+00
#python benchmark/benchmark_calibrating_experiment_pilot.py \
#    --x0_ind ${x0_ind} \
#   --seed ${seed} \
#    --L $L \
#    --task bernoulli_glm \
#    --num_training_mean $num_training_mean \
#    --num_training_cov $num_training_cov \
#    --tol $tol \
#    --tol_pilot $tol_pilot

#python benchmark/benchmark_calibrating_withcov.py --x0_ind 1 --seed 1 --L 10000000 --task "slcp_summary" --num_training_mean 300000 --num_training_cov 600000 --tol 1e-3
