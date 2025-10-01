#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:57:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --partition=v100
#SBATCH --array=0-39
#SBATCH --output=output_log_training/sensitivity/output_log_%A_%a.out
#SBATCH --error=output_log_training/sensitivity/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log_training

# #SBATCH --partition=v100,a10,a100-80gb,a100-40gb

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NDP
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3

TASK="my_twomoons" # MoG_5, Lapl_5 bernoulli_glm
N_EPOCHS=200
layer_num=3

NUM_TRAININGS=(100000 300000 500000 1000000)
#layer_len_list=(1792 2048 2304 2560)
#layer_len_list=(256 512 1024 1792 2048)

SEEDS_PER_VALUE=10
nt_index=$(( SLURM_ARRAY_TASK_ID / SEEDS_PER_VALUE ))

num_training=${NUM_TRAININGS[$nt_index]}
#num_training=1000000

layer_len=256

#layer_len=${layer_len_list[$nt_index]}


# Run the Python script
#echo "Running with seed=$seed, dim_out=$dim_out, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"
python ./training_sensitivity/sensitivity_training3.py --num_training $num_training --seed $seed --task $TASK --N_EPOCHS $N_EPOCHS --layer_len $layer_len --layer_num $layer_num
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
