#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --qos=normal
#SBATCH --partition=v100
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --output=output_log_training/output_log_%A_%a.out
#SBATCH --error=output_log_training/error_log_%A_%a.txt
# #SBATCH -w gilbreth-f[000-004],gilbreth-c[000-002]

# #SBATCH -w gilbreth-h[000-015],gilbreth-i[000-004],gilbreth-j[000-001],gilbreth-k[000-051]
# ##SBATCH -w gilbreth-b[000-015],gilbreth-c[000-002],gilbreth-d[000-007],gilbreth-g[000-011],gilbreth-h[000-015],gilbreth-i[000-004],gilbreth-j[000-001],gilbreth-k[000-051]
## #SBATCH -w gilbreth-f[000-004],gilbreth-c[000-002]

# Create the output_log directory if it doesn't exist
mkdir -p output_log_training

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NDP
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3

TASK="OU" # MoG, bernoulli_glm
N_EPOCHS=200
layer_len=64
num_training=300000

# Run the Python script
#python ./SDE/SDE_training.py --num_training $num_training --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len 
num_training_mean=300000
num_training_cov=600000
python ./SDE/SDE_cov2_training.py --num_training_mean $num_training_mean --num_training_cov $num_training_cov --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len 

