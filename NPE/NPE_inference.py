import sys
import torch
import numpy as np
from sbi import utils
from sbi.inference import SNPE
import pickle
import os
import argparse
import time
import sbibm
from sbibm.metrics.c2st import c2st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from NABC_functions import SLCP_summary


def main(args):
    # import x0
    if args.task in ["slcp_summary"]:
        sbi_task = sbibm.get_task("slcp")  # See sbibm.get_available_tasks() for all tasks
    elif args.task in ["bernoulli_glm"]:    
        sbi_task = sbibm.get_task(args.task)  # See sbibm.get_available_tasks() for all tasks
    
    if args.task in ["bernoulli_glm"]:
        s_dp_tmp = sbi_task.get_observation(num_observation = args.x0_ind)
    elif args.task in ["slcp_summary"]:
        s_dp_tmp = sbi_task.get_observation(num_observation = args.x0_ind)
        s_dp_tmp = SLCP_summary(s_dp_tmp)
    elif args.task in ["MoG_2", "MoG_5", "MoG_10", "Lapl_5", "Lapl_10"]:
        tmp = torch.load(f"x0s/{args.task}_x0.pt")
        s_dp_tmp = torch.tensor(tmp.numpy().tolist()[args.x0_ind -1], dtype = torch.float32)
    elif args.task == "my_twomoons":
        tmp = torch.load("x0s/my_twomoons.pt")
        s_dp_tmp = torch.tensor(tmp.numpy().tolist()[args.x0_ind -1], dtype = torch.float32)
    
    print(s_dp_tmp)
    if s_dp_tmp.ndim == 1:
        s_dp_tmp = torch.reshape(s_dp_tmp, (1,s_dp_tmp.size(0)))
    x0 = torch.tensor(s_dp_tmp, dtype = torch.float32)

    # import trained network
    if args.task in ["bernoulli_glm", "MoG_5", "Lapl_5", "MoG_10", "Lapl_10", "MoG_2", "my_twomoons"]:    
        post_sample = torch.load(f"posterior/{args.task}/post_{args.x0_ind}.pt")
    elif args.task in ["slcp_summary"]:    
        post_sample = torch.load(f"posterior/{args.task}/benchmark_post_sample_x0_{args.x0_ind}.pt")
        if post_sample.size(0) >12000:
            burn_in = int(post_sample.size(0) * 0.2)
            sam_ind = np.random.choice(np.arange(burn_in, post_sample.size(0)), 10_000, replace = False)
            post_sample = post_sample[sam_ind,:]
    else:
        print(f"no reference posterior samples avilable for the task {args.task}")
    
    output_file_path = os.path.join(f'./{args.method}_nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{args.seed}_{args.cond_den}.pkl')
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"{args.method} results file not found: {output_file_path}")
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    inference = saved_data['inference']
    training_time = saved_data['elapsed_time']

    # inference
    if args.method == "NPE":    
        sample_post = inference.build_posterior().sample((10000,), x=x0)
    elif args.method == "NLE":
        time0 = time.time()
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                        mcmc_parameters={"num_chains": 20,
                                                        "thin": 10})
        proposal = posterior.set_default_x(x0)
        sample_post = proposal.sample((10000,), x=x0)
        time1 = time.time()
        NLE_inference_elapsed_time = time1-time0
    else:
        raise ValueError(f"Unsupported method: {args.method}. Choose 'NPE' or 'NLE'.")
    
    # evaluate and save
    dist = c2st(post_sample, sample_post)
    print("C2ST", dist)
    output_dir = f"./{args.method}_results/{args.task}/{args.cond_den}/J_{int(args.num_training/1000)}K"   
    os.makedirs(output_dir, exist_ok=True)
    torch.save(dist, f"{output_dir}/C2ST_x0_{args.x0_ind}_seed_{args.seed}.pt")
    if args.method == "NPE":
        torch.save(training_time, f"{output_dir}/elapsed_time_x0_{args.x0_ind}_seed_{args.seed}.pt")  
    else:
        torch.save([training_time, NLE_inference_elapsed_time], f"{output_dir}/elapsed_time_x0_{args.x0_ind}_seed_{args.seed}.pt")
    

def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--method', type=str, default='NPE', help='NPE or NLE')
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--x0_ind', type=int, default=1, help='x0_ind: 1-10')
    parser.add_argument('--cond_den', type=str, default='maf', help='Conditional density estimator type: mdn, maf, nsf')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function


# python NPE/NPE_inference.py --task "my_twomoons" --seed 1 --x0_ind 10 --cond_den "maf" --num_training 1000 