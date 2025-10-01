#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:57:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=standby
#SBATCH --partition=a10,a100-80gb
#SBATCH --array=0-9
#SBATCH --output=output_log_training/output_log_%A_%a.out
#SBATCH --error=output_log_training/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
# ##SBATCH -w gilbreth-h[000-015]

mkdir -p output_log_training

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NDP
cd $SLURM_SUBMIT_DIR

TASK="cont_full3" # MoG_10, bernoulli_glm, slcp_summary_transform2
N_EPOCHS=200
layer_len=256
num_training=50000

#echo "Running with seed=$dim_out, dim_out=10, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3
python ./DP/DP_training.py --num_training $num_training --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len 
#python ./DP/benchmark_training.py --num_training 100000 --seed 1 --task "cont_table" --N_EPOCHS 1 --layer_len 256 

#seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3
#python ./DP/DP_cov_training.py \
#  --num_training_mean 50000 \
#  --num_training_cov  100000 \
#  --seed  $seed \
#  --task $TASK \
#  --N_EPOCHS 200 \
#  --layer_len 256
#python ./DP/DP_cov_training.py \
#  --num_training_mean 100000 \
#  --num_training_cov  200000 \
#  --seed  1 \
#  --task cont_table \
#  --N_EPOCHS 1 \
#  --layer_len 256
