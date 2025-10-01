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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from module import FL_Net, FL_Net_bounded
import time
from NDP import NDP_train
from simulator import Simulators, Priors, Bounds
# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    if args.nets_directory is None:
        nets_directory = "./nets_NABC"
    else:
        nets_directory = args.nets_directory
        
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    
    Y = priors.sample((args.num_training,))
    X = simulators(Y)

    X_train, Y_train = X[:args.num_training, :], Y[:args.num_training,:]
    
    print(X_train.size(), Y_train.size())
    # Learning hyperparameters
    D_in, D_out, Hs = X_train.size(1), Y_train.size(1), args.layer_len

    # Save the models
    ## Define the output directory
    print(f"start", flush=True)
    output_dir = f"{nets_directory}/{args.task}/train_{int(args.num_training/1_000)}K/"
    
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    bounds = Bounds(args.task)
    
    if bounds is None:
        net = FL_Net(D_in, D_out, H = Hs, H2 = Hs, H3 = Hs).to(device)
    else:
        net = FL_Net_bounded(D_in, D_out, H=Hs, p = 0.1, bounds = bounds).to(device)
     
    # Train Mean Function
    print(f"start training for mean function", flush=True)
    start_time = time.time()  # Start timer
    val_batch = 10_000
    early_stop_patience = 50
    tmp = NDP_train(X_train, Y_train, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch, early_stop_patience = early_stop_patience)
    end_time = time.time() 
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Mean Function Training completed in {elapsed_time/60:.2f} mins", flush=True)
    
    net.load_state_dict(tmp)
    net = net.to("cpu")

    torch.save(net.state_dict(),  output_dir + "/" + args.task + str(args.seed) +"_mean.pt")
    torch.save(elapsed_time,  output_dir + "/" + args.task + str(args.seed) +"_time.pt")
    torch.save(torch.cuda.get_device_name(0), output_dir + "/" + args.task + str(args.seed)+ "_gpu.pt")


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
                        help = "layer length of FL network (default: 256)")
    parser.add_argument("--nets_directory", type = str, default = None,
                        help = "None: default")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    #main_cond(args)
    
    # Use the parsed arguments
    print(f"task: {args.task}")
    print(f"Number of epochs: {args.N_EPOCHS}")
    print(f"seed: {args.seed}")

#python benchmark/benchmark_training.py --num_training 10000 --task "my_twomoons" --N_EPOCHS 1 --seed 1 --layer_len 256 
