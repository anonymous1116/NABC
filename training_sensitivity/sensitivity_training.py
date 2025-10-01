import torch
import numpy as np
import argparse
import os
from sbi import utils as utils
from sbibm.metrics.c2st import c2st
from torch.distributions import MultivariateNormal
import sbibm
import sys
import subprocess
from sensitivity_nets import FL_Net_bounded_sen, FL_Net_sen
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import time
from NDP import NDP_train_l2
from benchmark.simulator import Simulators, Priors, Bounds
# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    
    Y_train = priors.sample((args.num_training,))
    X_train = simulators(Y_train)

    # Learning hyperparameters
    D_in, D_out, Hs = X_train.size(1), Y_train.size(1), args.layer_len

    # Save the models
    ## Define the output directory
    print(f"start", flush=True)
    output_dir = f"../depot_hyun/hyun/NDP/sensitivity/{args.task}/J{int(args.num_training/1_000)}K_{args.layer_len}_{args.layer_num}"
    
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    bounds = Bounds(args.task)

    if bounds is None:
        net = FL_Net_sen(D_in = D_in, D_out = D_out, H = Hs, p = 0.1, num = args.layer_num)
    else:
        net = FL_Net_bounded_sen(D_in = D_in, D_out = D_out, H = Hs, p = 0.1, bounds = bounds, num = args.layer_num)
     
    # Train Mean Function
    print(f"start training for mean function", flush=True)
    start_time = time.time()  # Start timer
    val_batch = 10_000
    early_stop_patience = 50
    tmp, val_error_plt, best_model_state = NDP_train_l2(X_train, Y_train, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch, early_stop_patience = early_stop_patience)
    end_time = time.time() 
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Mean Function Training completed in {elapsed_time/60:.2f} mins", flush=True)
    
    net.load_state_dict(tmp)
    net = net.to("cpu")

    torch.save(net.state_dict(),  output_dir + "/" + args.task + str(args.seed) +"_mean.pt")
    torch.save([val_error_plt, best_model_state] ,  output_dir + "/" + args.task + str(args.seed) +"_val_error.pt")
    torch.save(elapsed_time,  output_dir + "/" + args.task + str(args.seed) +"_time.pt")
    torch.save(torch.cuda.get_device_name(0), output_dir + "/" + args.task + str(args.seed)+ "_gpu.pt")
    
    print(f"Mean Function saved", flush=True)

    if args.cov == 1:
        create_cov_job_script(args)

    torch.set_default_device("cpu")

    
def create_cov_job_script(args):
    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --partition=a10,a100-80gb
#SBATCH --output=output_log_cov/output_log_%A_%a.out
#SBATCH --error=output_log_cov/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log_cov

# Load the required Python environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NDP

echo "Running simulation for task {args.task}, num_training: {args.num_training}, N_EPOCHS: {args.N_EPOCHS} seed={args.seed}, layer_len={args.layer_len}..."
python ./training_sensitivity/sensitivity_cov_training.py \
  --num_training_mean {args.num_training} \
  --num_training_cov  {args.num_training} \
  --seed  {args.seed} \
  --task {args.task} \
  --N_EPOCHS {args.N_EPOCHS} \
  --layer_len {args.layer_len}
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"../NDP/benchmark/cov_learning/sensitivity/{args.task}/J_{int(args.num_training/1000)}K/slurm_files"
    os.makedirs(output_dir, exist_ok=True)

    job_file_path = os.path.join(output_dir, f"cov_{args.task}_{args.num_training}_{args.seed}_{args.layer_len}_cov_{int(args.num_training*2/1000)}K.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--num_training", type=int, default=100_000, 
                        help="Number of training data (default: 100_000)")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: twomoons, MoG, MoUG, Lapl, GL_U or slcp, slcp2')
    parser.add_argument("--N_EPOCHS", type=int, default=100, 
                        help="Number of EPOCHS (default: 100)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--layer_len", type = int, default = 256,
                        help = "layer width of FL network (default: 256)")
    parser.add_argument("--layer_num", type = int, default = 3,
                        help = "layer number of FL network (default: 3)")
    parser.add_argument("--cov", type = int, default = 0, help = "1: training cov, 0: not training cov")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    
    # Use the parsed arguments
    print(f"task: {args.task}")
    print(f"Number of epochs: {args.N_EPOCHS}")
    print(f"seed: {args.seed}")