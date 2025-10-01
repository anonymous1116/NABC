#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=standby
#SBATCH --partition=a10,a100-80gb
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --output=vary_dim_out/output_log/output_log_%A_%a.out
#SBATCH --error=vary_dim_out/output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p vary_dim_out/output_log

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NDP
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
#seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
#L=100000000
#task="MoG"
#num_training=100000
#tol=1e-4
# Run the calibrate.py
#echo "Running with seed=$seed, L=$L"
#python vary_dim_out/calibrating.py --seed $seed --L $L
#echo "## Run completed with seed=$seed, dim_out=$L"

# Run the MCMC.py
#s_dp_ind=$((SLURM_ARRAY_TASK_ID))
#python vary_dim_out/MCMC.py --s_dp_ind $s_dp_ind 

# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))

#echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"
#python vary_dim_out/calibrating_amor_resample.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python vary_dim_out/calibrating_amor.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

#python benchmark/calib_ber.py --x0_ind 1 --seed 1 --L 100000000 --task bernoulli_glm --num_training 500000 --tol 1e-3
#python covariance_learning/calibrating_withcov.py --x0_ind 1 --seed 1 --L 100000000 --task bernoulli_glm --num_training 1000000 --tol 1e-4

# Run ABC.py
L=10000000000
task="MoG"
tol=1e-6 

#seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1))
#x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))

python ./vary_dim_out/ABC.py --x0_ind $x0_ind --seed $seed --L $L --task $task --tol $tol --dim_start 10 --dim_end 11 --x0_plot 1
#python ./vary_dim_out/ABC.py --x0_ind 1 --seed 1 --L 10000000000 --task "MoG" --tol 1e-6 --x0_plot 1
#python ./vary_dim_out/ABC_calibrating_com  p_withcov.py --x0_ind 1 --seed 1 --L 100000000 --task "MoG" --tol 1e-4 --num_training 200000
#python ./vary_dim_out/ABC_calibrating_comp_withcov.py --seed $seed --L $L --task $task --tol $tol --num_training 100000 --x0_plot 1 --dim_start 8 --dim_end 11



