#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:57:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=170G
#SBATCH --qos=standby
#SBATCH --partition=a10,a100-80gb,v100,a30,a100-40gb
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --output=output_log_training/DP_gibbs/output_log_%A_%a.out
#SBATCH --error=output_log_training/DP_gibbs/error_log_%A_%a.txt

# #SBATCH --partition=a10,a100-40gb,a100-80gb

# Create the output_log directory if it doesn't exist
mkdir -p output_log_training/DP_gibbs

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

# Compute x0_ind and seed from SLURM_ARRAY_TASK_ID
x0_ind=$((SLURM_ARRAY_TASK_ID % 10 + 1))

#python MCMC/my_slcp_posterior.py --i $seeds
python DP/DP_gibbs.py --i $x0_ind --p 0.6
echo "## Run completed"

