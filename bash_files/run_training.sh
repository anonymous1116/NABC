#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:57:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-99%25
#SBATCH --output=output_log/output_log_%A_%a.out
#SBATCH --error=output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NDP
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
dim_out=$((SLURM_ARRAY_TASK_ID % 10 + 1))

TASK="Lapl"
N_EPOCHS=200
layer_len=128
L=100000000

# Run the Python script
echo "Running with seed=$seed, dim_out=$dim_out, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, L=$L"
python training_Lapl.py --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len --L $L --dim_out $dim_out

echo "## Run completed with seed=$seed, dim_out=$dim_out, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, L=$L"
