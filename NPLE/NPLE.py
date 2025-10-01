import sys
import torch
import numpy as np
from sbi import utils
from sbi.inference import SNPE,SNLE
import pickle
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from benchmark.simulator import Simulators, Priors


def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)

    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    simulators = Simulators(args.task)

    # Sample theta from the prior
    theta = priors.sample((args.num_training,))

    # Run the simulator
    X = simulators(theta)

    # Create inference object
    if args.method == "NPE":
        inference = SNPE(prior=priors, density_estimator=args.cond_den)
    elif args.method == "NLE":
        inference = SNLE(prior=priors, density_estimator=args.cond_den)
    else:
        raise ValueError(f"Unsupported method: {args.method}. Choose 'NPE' or 'NLE'.")
        
    inference = inference.append_simulations(theta, X)

    # Train the density estimator and build the posterior
    start_time = time.time()  # Start timer
    density_estimator = inference.train()
    end_time = time.time()  # End timer

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Training with {args.cond_den}")

    # Define the output directory
    output_dir = f"./{args.method}_nets/{args.task}/J_{int(args.num_training/1000)}K"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save the inference object using pickle in the specified directory
    # Save the inference object and elapsed time using pickle in the specified directory
    output_file_path = os.path.join(output_dir, f"{args.task}_{args.seed}_{args.cond_den}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump({'inference': inference, 'elapsed_time': elapsed_time}, f)
    
    print(f"Saved inference object and elapsed time to '{output_file_path}'.")

def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--method', type=str, default='NPE', help='NPE or NLE')
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--cond_den', type=str, default='maf', help='Conditional density estimator type: mdn, maf, nsf')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function

# python NPE/NPE_training.py --task "my_twomoons" --seed 1 --method "NPE" --cond_den "maf" --num_training 1000