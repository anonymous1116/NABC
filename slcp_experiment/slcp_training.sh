#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:57:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=standby
#SBATCH --partition=a10,a100-80gb,a100-40gb,v100
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

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3

TASK="slcp_summary_transform2" # MoG, bernoulli_glm
N_EPOCHS=200
layer_len=256
num_training=500000

# Run the Python script
#echo "Running with seed=$seed, dim_out=$dim_out, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"
python ./slcp_experiment/slcp_training.py --num_training $num_training --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len 
#num_training_cov=600000
#python ./slcp_experiment/slcp_cov4.py --num_training_mean $num_training --num_training_cov $num_training_cov --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len 
#python ./slcp_experiment/slcp_training.py --num_training 100000 --seed 0 --task "my_slcp3" --N_EPOCHS 1 --layer_len 256 
#python ./slcp_experiment/slcp_cov.py --num_training_mean 100000 --num_training_cov 200000 --seed 0 --task "my_slcp3" --N_EPOCHS 1 --layer_len 256 
#python ./benchmark/benchmark_cov_training.py \
#  --num_training_mean 25000 \
#  --num_training_cov  50000 \
#  --seed  $seed \x
#  --task "MoG_5" \
#  --N_EPOCHS 200 \
#  --layer_len 256
