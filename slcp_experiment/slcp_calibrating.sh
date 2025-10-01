#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --account=standby
#SBATCH -w gilbreth-b[000-015],gilbreth-c[000-002],gilbreth-d[000-007],gilbreth-g[000-011],gilbreth-h[000-015],gilbreth-i[000-004],gilbreth-j[000-001],gilbreth-k[000-051]
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-99
#SBATCH --output=output_log_calib/output_log_%A_%a.out
#SBATCH --error=output_log_calib/error_log_%A_%a.txt

# #SBATCH -w gilbreth-b[000-015],gilbreth-c[000-002],gilbreth-d[000-007],gilbreth-g[000-011],gilbreth-h[000-015],gilbreth-i[000-004],gilbreth-j[000-001],gilbreth-k[000-051]
# #SBATCH -w gilbreth-h[000-015],gilbreth-i[000-004],gilbreth-j[000-001],gilbreth-k[000-051]

# Create the output_log directory if it doesn't exist
mkdir -p output_log_calib

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

# Compute x0_ind and seed from SLURM_ARRAY_TASK_ID
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))

L=10000000000
tol=1e-6
num_training_mean=300000
num_training_cov=600000

echo "Running with x0_ind=${x0_ind}, seed=${seed}, L=${L}, tol=${tol}, num_training=${num_training_mean}"
python slcp_experiment/slcp_calibrating4.py \
    --x0_ind ${x0_ind} \
    --seed ${seed} \
    --L $L \
    --task slcp_summary_transform \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --tol $tol

#python slcp_experiment/slcp_calibrating4.py --x0_ind 1 --seed 3 --L 100000000 --task "slcp_summary" --num_training_mean 300000 --num_training_cov 600000 --tol 1e-4


#tol_pilot=1e-1
#python benchmark/benchmark_calibrating_experiment_pilot.py \
#    --x0_ind ${x0_ind} \
#   --seed ${seed} \
#    --L $L \
#    --task MoG_10 \
#    --num_training_mean $num_training_mean \
#    --num_training_cov $num_training_cov \
#    --tol $tol \
#    --tol_pilot $tol_pilot

#python benchmark/benchmark_calibrating_withcov.py --x0_ind 1 --seed 1 --L 10000000 --task "slcp_summary" --num_training_mean 300000 --num_training_cov 600000 --tol 1e-3
