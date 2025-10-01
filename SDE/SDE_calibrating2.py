import torch
import numpy as np
import os, sys
import argparse
import sbibm
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from NDP import calibrate, calibrate_cov, Markov_sliced_wasserstein_distance, wasserstein_distance_Nto1, wasserstein_gaussian
from module import GRU_net, GRU_net_bounded, GRU_CovarianceNet2
from sbibm.metrics.c2st import c2st
from benchmark.simul_funcs import truncated_mvn_sample
from benchmark.simulator import Priors, Simulators, Bounds
from vary_dim_out.NDP_resampling import UnifSample, param_box
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.analysis import pairplot

def main(args):
    print(torch.cuda.get_device_name(0), flush=True)

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

    priors = Priors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
    
    net_dir = f"../depot_hyun/hyun/NDP/{args.task}/mean_{int(num_training_mean/1000)}K_cov2_{int(num_training_cov/1_000)}K_layer_{args.layer_len}"
    assert os.path.exists(net_dir), f"Model directory {net_dir} does not exist"

    tmp1 = torch.load(f"{net_dir}/best_model_mean_state_{seed}.pt")
    tmp2 = torch.load(f"{net_dir}/best_model_cov_state_{seed}.pt")
    tmp3 = torch.load(f"{net_dir}/val_error_plt_{seed}.pt")

    print(f"best_val: {tmp3[1]}", flush=True)

    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)

    # Learning hyperparameters
    D_in, D_out, Hs = X_cal.size(1), Y_cal.size(1), args.layer_len
    if bounds is None:
        mean_net = GRU_net(input_dim = 1, hidden_dim = Hs, output_dim = D_out)
    else:
        mean_net = GRU_net_bounded(input_dim = 1, hidden_dim = Hs, output_dim = D_out, bounds = bounds)
    

    mean_net.load_state_dict(tmp1)
    
    covnet = GRU_CovarianceNet2(input_dim=1, output_dim=D_out, hidden_dim=args.layer_len)
    covnet.load_state_dict(tmp2)
        
    # Initialize the Priors and Simulators classes and ABC_methods
    tol0 = 1e-3

    if L > 1_000_000_000 + 1:
        tol0 = 1e-5
    
    chunk_size = 2_000_000
    num_chunks = L // chunk_size
    
    start_time = time.time()
    
    file_path = "/home/hyun18/depot_hyun/NeuralABC_R/OU3/OU3_obs.pt"
    tmp = torch.load(file_path)
    
    x0_list = tmp.numpy().tolist()
    s_dp_tmp = torch.tensor(x0_list[args.x0_ind - 1])
    
    
    if s_dp_tmp.ndim == 1:
        s_dp_tmp = torch.reshape(s_dp_tmp, (1,s_dp_tmp.size(0)))
        

    chunk_size_cal = 1_000
    print("s_dp_size", s_dp_tmp.size(), flush = True)
    print("X_cal size", X_cal.size(), flush = True)
    
    with torch.no_grad():
        _, adj = calibrate_cov(s_dp_tmp, X_cal, Y_cal, mean_net, covnet, tol=.01, device = device, case = args.task, chunk_size=chunk_size_cal, bounds = bounds)
    print("adj: ", adj.size())
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
        print(X_chunk.size(), Y_chunk.size())
        dist = wasserstein_gaussian(X_chunk, s_dp_tmp, chunk_size=100_000)
        sort_dist, _ = torch.sort(dist)

        print(dist.size())
        # new index
        nacc = int(nums * tol0)
        ds = sort_dist[nacc-1]
        wt1 = (dist <= ds)

        print(wt1.shape)
        X_chunk, Y_chunk = X_chunk[wt1], Y_chunk[wt1]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)
        print(f"{i}th iteration out of {num_chunks}", flush = True)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    print("X_abc size", X_abc.size())

    # true posterior
    file_path =f"../depot_hyun/NeuralABC_R/OU3/OU3_post_{args.x0_ind}.pt"
    post_sample = torch.load(file_path)
    num_rows = post_sample.shape[0]
    indices = torch.randperm(num_rows)[:10000]

    post_sample = torch.tensor(post_sample[indices,:], dtype = torch.float32)
    
    tol = (args.tol/tol0 + 1e-12)

    calibrate_results = calibrate_cov(s_dp_tmp, X_abc, Y_abc, mean_net, covnet, n_samples = 10000, tol = tol, bounds = bounds, chunk_size=chunk_size_cal)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("NABC sample size: ", calibrate_results[0].detach().cpu().size())
    results_size = min(10_000, calibrate_results[0].detach().cpu().size(0))

    tmp = c2st(post_sample[:results_size].cpu(), calibrate_results[0].detach().cpu()[:results_size] )
    print(tmp)    
    
    NABC_results.append(tmp)
    
    sci_str = format(tol*tol0, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"./NABC_results/{args.task}_withcov/mean_{int(num_training_mean/1_000)}K_cov2_{int(num_training_cov/1_000)}K_{args.layer_len}/{int(args.L/1_000_000)}M_eta{sci_str}"
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
    parser.add_argument("--num_training_mean", type=int, default=300_000, 
                        help="Number of training data (default: 300_000)")
    parser.add_argument("--num_training_cov", type=int, default=300_000, 
                        help="Number of training data (default: 300_000)")
    parser.add_argument("--layer_len", type=int, default=256, 
                        help="layer length of covariance NN (default: 256)")
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
    print(f"num_training_mean: {args.num_training_mean}")
    print(f"num_training_cov: {args.num_training_cov}")
    print(f"layer_len: {args.layer_len}")
    print(f"tol: {args.tol}")