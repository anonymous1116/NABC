import torch
import numpy as np
import os, sys
import sys
import argparse
import sbibm
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from NDP import calibrate
from NDP_functions import ABC_rej, ABC_rej2
from module import FL_Net, FL_Net_bounded
from sbibm.metrics.c2st import c2st
from simul_funcs import bernoulli_GLM, truncated_mvn_sample
from simulator import Priors, Simulators, Bounds
from vary_dim_out.NDP_resampling import UnifSample, param_box
import time
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from pathlib import Path
from NDP_functions import SLCP_summary


def main(args):
    seed = args.seed
    torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L
    NABC_results = []
    
    # Select the correct synthetic pair generator
    
    num_training = args.num_training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.task == "slcp_summary":
        sbi_task = sbibm.get_task("slcp")  # See sbibm.get_available_tasks() for all tasks
    else:    
        sbi_task = sbibm.get_task(args.task)  # See sbibm.get_available_tasks() for all tasks
    
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)

    net_dir = f"../depot_hyun/hyun/NDP/{args.task}/train_{int(num_training/1_000)}K/{args.task}{seed}_mean.pt"
    tmp = torch.load(net_dir)

    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)


    # Learning hyperparameters
    D_in, D_out, Hs = X_cal.size(1), Y_cal.size(1), 256
    if bounds is None:
        net = FL_Net(D_in, D_out, H = Hs, H2 = Hs, H3 = Hs).to(device)
    else:
        net = FL_Net_bounded(D_in, D_out, H=Hs, p = 0.1, bounds = bounds).to(device)
        
    net.to("cpu")
        
    net2_dir = f"../depot_hyun/hyun/NDP/{args.task}/train_{int(num_training/1_000)}K/{args.task}{seed}_cMAD.pt"
    tmp2 = torch.load(net2_dir)

    net2 = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs)
    net2.load_state_dict(tmp2)
    net2 = net2.to("cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the Priors and Simulators classes and ABC_methods
    tol0 = 1e-3
    chunk_size = 5_000_000
    num_chunks = L // chunk_size
    

    start_time = time.time()
    s_dp_tmp = sbi_task.get_observation(num_observation = args.x0_ind)
    if args.task == "slcp_summary":
        s_dp_tmp = SLCP_summary(s_dp_tmp)

    chunk_size_cal = 10_000
    print("s_dp_size", s_dp_tmp.size(), flush = True)
    print("X_cal size", X_cal.size(), flush = True)
    
    with torch.no_grad():
        _, adj = calibrate(s_dp_tmp, X_cal, Y_cal, net, net2, tol=.1, device = device, case = args.task, chunk_size=chunk_size_cal, bounds = bounds)
    
    X_abc = []
    Y_abc = []
    
    if bounds is not None:
        adj = torch.clamp(adj, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])

    with torch.no_grad():
        max_vals = torch.max(adj,0).values
        min_vals = torch.min(adj,0).values

    priors_mean = torch.zeros(10)
    priors_std = torch.ones(10) * np.sqrt([2])

    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)

    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break
        if args.task == "bernoulli_glm":
            Y_chunk = truncated_mvn_sample(nums, priors_mean, priors_std, min_vals, max_vals)
        else:
            Y_chunk = param_box(UnifSample(bins = 10), adj, num=nums)
        
        X_chunk = simulators(Y_chunk)
        
        index_ABC = ABC_rej2(s_dp_tmp, X_chunk, tol0, device, args.task)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    print("X_abc size", X_abc.size())

    if args.task in ["slcp", "slcp_summary"]:
        post_sample = sbi_task.get_reference_posterior_samples(num_observation=args.x0_ind)
    elif args.task == "bernoulli_glm":    
        post_sample = torch.load(f"../depot_hyun/NeuralABC_R/{args.task}/post_{args.x0_ind}.pt")
    else:
        print(f"no reference posterior samples avilable for the task {args.task}")
    
    tol = (args.tol/tol0 + 1e-12)

    calibrate_results = calibrate(s_dp_tmp, X_abc, Y_abc, net, net2, n_samples = 10000, tol = tol, bounds = bounds)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("NABC sample size: ", calibrate_results[0].detach().cpu().size())
    results_size = min(10_000, calibrate_results[0].detach().cpu().size(0))

    tmp = c2st(post_sample[:results_size].cpu(), calibrate_results[0].detach().cpu()[:results_size] )
    print(tmp)    
    del net, net2
    NABC_results.append(tmp)
    
    sci_str = format(tol*tol0, ".0e")
    print(sci_str)  # Output: '1e-02'
    
    output_dir = f"./NABC_results/{args.task}/{int(num_training/1_000)}K_resample/amor_{int(args.L/1_000_000)}M_eta{sci_str}"
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    # Save to output_dir
    pairplot(post_sample, figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()

    pairplot(calibrate_results[0].detach().cpu()[:10000], figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    

    torch.save(NABC_results, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
    torch.save([torch.cuda.get_device_name(0), elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--x0_ind", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--L", type = int, default = 10_000_000,
                        help = "Calibration data size (default: 10M)")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: Lapl, MoG')
    parser.add_argument("--num_training", type=int, default=100_000, 
                        help="Number of training data (default: 100_000)")
    parser.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerance value for ABC (any positive float, default: 1e-4 but less than 1e-2)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"x0_ind: {args.x0_ind}")
    print(f"seed: {args.seed}")
    print(f"L: {args.L}")
    print(f"task: {args.task}")