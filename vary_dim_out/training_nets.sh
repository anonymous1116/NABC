#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=standby
#SBATCH --partition=a10,a100-80gb,a30,a100-40gb
#SBATCH --nodes=1
#SBATCH --array=10-99
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

# Calculate seed and dim_out
dim_out=$((SLURM_ARRAY_TASK_ID / 10 + 1)) # tens digit 10,20,30
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3

TASK="MoG" # MoG, bernoulli_glm
N_EPOCHS=200
layer_len=256
num_training=300000

# Run the Python script
#echo "Running with seed=$seed, dim_out=$dim_out, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"
#python ./vary_dim_out/training_nets.py --num_training $num_training --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len --dim_out $dim_out
#python ./vary_dim_out/training_nets.py --num_training 200000 --seed 1 --task "MoG" --N_EPOCHS 1 --layer_len 256 --dim_out 2
python ./vary_dim_out/training_covs.py --num_training_mean 100000 --num_training_cov 200000 --seed $seed --task "MoG" --N_EPOCHS $N_EPOCHS --layer_len $layer_len --dim_out $dim_out
#echo "## Run completed with seed=$seed, dim_out=$dim_out, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"

#echo "Running with seed=$dim_out, dim_out=10, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"
#python ./benchmark/benchmark_training.py --num_training $num_training --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len 
#python ./benchmark/benchmark_cov_training.py \
#  --num_training_mean 25000 \
#  --num_training_cov  50000 \
#  --seed  $seed \
#  --task "MoG_5" \
#  --N_EPOCHS 200 \
#  --layer_len 256
