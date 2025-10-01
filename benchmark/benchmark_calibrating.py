import torch
import numpy as np
import os, sys
import argparse
import sbibm
import time
from sbibm.metrics.c2st import c2st
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.analysis import pairplot

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from NDP import calibrate_cov
from NDP_functions import ABC_rej2
from module import FL_Net, CovarianceNet, FL_Net_bounded
from benchmark.simul_funcs import truncated_mvn_sample
from simulator import Priors, Simulators, Bounds
from NDP_functions import SLCP_summary
from simul_funcs import UnifSample, param_box

def main(args):
    if args.nets_directory is None:
        nets_directory = "./nets_NABC"
    else:
        nets_directory = args.nets_directory
    
    seed = args.seed
    torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L
    NABC_results = []
    
    # Select the correct synthetic pair generator
    
    num_training_mean = args.num_training_mean
    num_training_cov = args.num_training_cov
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.task in ["slcp_summary"]:
        sbi_task = sbibm.get_task("slcp")  # See sbibm.get_available_tasks() for all tasks
    elif args.task in ["bernoulli_glm"]:    
        sbi_task = sbibm.get_task(args.task)  # See sbibm.get_available_tasks() for all tasks
    
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
    
    net_dir = f"{nets_directory}/{args.task}/mean_{int(num_training_mean/1000)}K_cov_{int(num_training_cov/1_000)}K_layer_{args.layer_len}"

    assert os.path.exists(net_dir), f"Model directory {net_dir} does not exist"

    tmp1 = torch.load(f"{net_dir}/best_model_mean_state_{seed}.pt")
    tmp2 = torch.load(f"{net_dir}/best_model_cov_state_{seed}.pt")
    tmp3 = torch.load(f"{net_dir}/val_error_plt_{seed}.pt")

    print(f"best_val: {tmp3[1]}", flush=True)

    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)


    # Learning hyperparameters
    D_in, D_out, Hs = X_cal.size(1), Y_cal.size(1), args.layer
    if bounds is None:
        mean_net = FL_Net(D_in, D_out, H = Hs, H2 = Hs, H3 = Hs).to(device)
    else:
        mean_net = FL_Net_bounded(D_in, D_out, H=Hs, p = 0.1, bounds = bounds).to(device)
    

    mean_net.load_state_dict(tmp1)
    
    covnet = CovarianceNet(input_dim=D_in, output_dim=D_out, hidden_dim=args.layer_len)
    covnet.load_state_dict(tmp2)
        
    # Initialize the Priors and Simulators classes and ABC_me   thods
    if args.task == "my_twomoons":
        tol0 = 1e-2
    else:
        tol0 = 1e-3

    if L > 1_000_000_000 + 1:
        tol0 = 1e-5
    
    chunk_size = 50_000_000
    num_chunks = L // chunk_size

    
    start_time = time.time()
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
        

    chunk_size_cal = 10_000
    print("s_dp_size", s_dp_tmp.size(), flush = True)
    print("X_cal size", X_cal.size(), flush = True)
    
    with torch.no_grad():
        _, adj = calibrate_cov(s_dp_tmp, X_cal, Y_cal, mean_net, covnet, tol=.01, device = device, case = args.task, chunk_size=chunk_size_cal, bounds = bounds)
    
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
        print(f"{i}th iteration out of {num_chunks}", flush = True)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    print("X_abc size", X_abc.size())

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
    
    tol = (args.tol/tol0 + 1e-12)

    calibrate_results = calibrate_cov(s_dp_tmp, X_abc, Y_abc, mean_net, covnet, n_samples = 10000, tol = tol, bounds = bounds)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("NABC sample size: ", calibrate_results[0].detach().cpu().size())
    results_size = min(10_000, calibrate_results[0].detach().cpu().size(0))

    NABC_results = c2st(post_sample[:results_size].cpu(), calibrate_results[0].detach().cpu()[:results_size] )
    print("C2ST:", NABC_results)    
    
    
    sci_str = format(tol*tol0, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"results/{args.task}/mean_{int(num_training_mean/1_000)}K_cov_{int(num_training_cov/1_000)}K/calib_{int(args.L/1_000_000)}M_eta{sci_str}"
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
    
    torch.save(NABC_results, f"{output_dir}/C2ST_x0{args.x0_ind}_seed{args.seed}.pt")
    torch.save([torch.cuda.get_device_name(0), elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")
    torch.save(calibrate_results[0].detach().cpu(), f"{output_dir}/samples_x0{args.x0_ind}_seed{args.seed}.pt")

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
    parser.add_argument("--num_training_mean", type=int, default=300_000, 
                        help="Number of training data (default: 300_000)")
    parser.add_argument("--num_training_cov", type=int, default=300_000, 
                        help="Number of training data (default: 300_000)")
    parser.add_argument("--layer_len", type=int, default=256, 
                        help="layer length of covariance NN (default: 256)")
    parser.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerance value for ABC (any positive float, default: 1e-4 but less than 1e-2)")
    parser.add_argument("--nets_directory", type = str, default = None,
                        help = "None: default")
    
    return parser.parse_args()

#python benchmark/benchmark_calibrating.py --x0_ind 1 --seed 1 --L 10000000 --task my_twomoons --num_training_mean 10000 --num_training_cov 20000 --layer_len 256 --tol 1e-3


if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"x0_ind: {args.x0_ind}")
    print(f"seed: {args.seed}")
    print(f"L: {args.L}")
    print(f"task: {args.task}")
    print(f"num_training_mean: {args.num_training_mean}")
    print(f"num_training_cov: {args.num_training_cov}")
    print(f"layer_len: {args.layer_len}")
    print(f"tol: {args.tol}")