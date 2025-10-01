#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-99
#SBATCH --output=benchmark/output_log/output_log_%A_%a.out
#SBATCH --error=benchmark/output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p benchmark/output_log

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NDP
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
#L=100000000
L=5000000000
task="slcp_summary"
num_training=300000
tol=2e-6

# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"

python benchmark/benchmark_calibrating.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

#python benchmark/benchmark_calibrating.py --x0_ind 1 --seed 1 --L 100000000 --task "slcp" --num_training 300000 --tol 1e-4
