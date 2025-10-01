#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:59
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --output=output_log_cov/output_log_%A_%a.out
#SBATCH --error=output_log_cov/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log_cov

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

# Define arrays
seeds=(1 2 3)
trainings=(500000 1000000 1500000)
layer_lens=(128 256)

# Map SLURM_ARRAY_TASK_ID to a unique combination
index=$SLURM_ARRAY_TASK_ID

seed_index=$((index % 3))
training_index=$(((index / 3) % 3))
layer_index=$((index / 9))

seed=${seeds[$seed_index]}
num_training=${trainings[$training_index]}
layer_len=${layer_lens[$layer_index]}

#echo "Running with seed=$seed, num_training=$num_training, layer_len=$layer_len"

# Run training
#python ./covariance_learning/bernoulli_cov.py \
#  --num_training ${num_training} \
#  --seed ${seed} \
#  --task "bernoulli_glm" \
#  --N_EPOCHS 200 \
#  --layer_len ${layer_len}


# Define the starting point for seed 
seed_START=1

# Get the current N_EPOCHS value based on the job array index
seeds=$((seed_START + SLURM_ARRAY_TASK_ID - 1))

#echo "Running with seed=$seed, num_training=1_000_000, layer_len=256"

#python ./covariance_learning/bernoulli_cov.py \
#  --num_training 1000000 \
#  --seed  $seeds \
#  --task "bernoulli_glm" \
#  --N_EPOCHS 200 \
#  --layer_len 256

#echo "Running with seed=$seed, num_training=500_000, layer_len=256"

#python ./covariance_learning/bernoulli_cov.py \
#  --num_training_mean 300000 \
#  --num_training_cov  1000000 \
#  --seed  $seeds \
#  --task "bernoulli_glm" \
#  --N_EPOCHS 200 \
#  --layer_len 256


python ./benchmark/benchmark_cov_training.py \
  --num_training_mean 100000 \
  --num_training_cov  200000 \
  --seed  $seeds \
  --task "MoG_5" \
  --N_EPOCHS 200 \
  --layer_len 512
  