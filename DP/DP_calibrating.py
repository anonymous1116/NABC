import torch
import numpy as np
import os, sys
import argparse
import sbibm
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from NDP import calibrate, calibrate_cov
from NDP_functions import ABC_rej2, cont_table_transform
from module import FL_Net, CovarianceNet2, FL_Net_bounded
from sbibm.metrics.c2st import c2st
from benchmark.simulator import Priors, Simulators, Bounds
from vary_dim_out.NDP_resampling import UnifSample, param_box
import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from pathlib import Path
from DP.DP_utils import truncated_dirichlet_batch

def main(args):
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
    
    net_dir = f"../depot_hyun/hyun/NDP/{args.task}/mean_{int(num_training_mean/1000)}K_cov_{int(num_training_cov/1_000)}K_layer_{args.layer_len}"
    assert os.path.exists(net_dir), f"Model directory {net_dir} does not exist"

    tmp1 = torch.load(f"{net_dir}/best_model_mean_state_{seed}.pt")
    tmp2 = torch.load(f"{net_dir}/best_model_cov_state_{seed}.pt")
    tmp3 = torch.load(f"{net_dir}/val_error_plt_{seed}.pt")

    print(f"best_val: {tmp3[1]}", flush=True)

    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)

    X_cal = torch.clone(X_cal[:,:3])
    Y_cal = torch.clone(Y_cal[:,:3])
    

    # Learning hyperparameters
    D_in, D_out, Hs = X_cal.size(1), Y_cal.size(1), 256
    if bounds is None:
        mean_net = FL_Net(D_in, D_out, H = Hs, H2 = Hs, H3 = Hs).to(device)
    else:
        mean_net = FL_Net_bounded(D_in, D_out, H=Hs, p = 0.1, bounds = bounds).to(device)
    

    mean_net.load_state_dict(tmp1)
    
    covnet = CovarianceNet2(input_dim=D_in, output_dim=D_out, hidden_dim=args.layer_len)
    covnet.load_state_dict(tmp2)
        
    # Initialize the Priors and Simulators classes and ABC_me   thods
    tol0 = 1e-3

    if L > 1_000_000_000 + 1:
        tol0 = 1e-5
    
    chunk_size = 10_000_000
    num_chunks = L // chunk_size

    
    start_time = time.time()
    if args.task in ["cont_table", "cont_table_dp", "cont_table_dp2", "cont_full", "cont_full2",  "cont_full3"]:
        tmp = torch.load(f"/home/hyun18/depot_hyun/NeuralABC_R/{args.task}/{args.task}_x0_list.pt")
        s_dp_tmp = torch.tensor(tmp[args.x0_ind -1], dtype = torch.float32)
        s_dp_tmp = s_dp_tmp[:3]
    elif args.task in ["cont_table_dp_transform"]:
        tmp = torch.load(f"/home/hyun18/depot_hyun/NeuralABC_R/cont_table_dp/cont_table_dp_x0_list.pt")
        s_dp_tmp = torch.tensor(tmp[args.x0_ind -1], dtype = torch.float32)
        s_dp_tmp = s_dp_tmp[:3]
    
    print(s_dp_tmp)

    if s_dp_tmp.ndim == 1:
        s_dp_tmp = torch.reshape(s_dp_tmp, (1,s_dp_tmp.size(0)))

    if args.task in ["cont_table_dp_transform"]:
        s_dp_tmp = cont_table_transform(s_dp_tmp)

    chunk_size_cal = 10_000
    print("s_dp_size", s_dp_tmp.size(), flush = True)
    print("X_cal size", X_cal.size(), flush = True)
    
    with torch.no_grad():
        _, adj = calibrate_cov(s_dp_tmp, X_cal, Y_cal, mean_net, covnet, tol=.01, device = device, case = args.task, chunk_size=chunk_size_cal, bounds = bounds)
    
    X_abc = []
    Y_abc = []
    
    if bounds is not None:
        adj = torch.clamp(adj, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])

    theta4 = torch.max(torch.ones(adj.size(0)) - adj[:,0]- adj[:,1]- adj[:,2], torch.zeros(adj.size(0)))
    
    adj = torch.column_stack([adj,theta4])
    
    
    with torch.no_grad():
        max_vals = torch.max(adj,0).values.tolist()
        min_vals = torch.min(adj,0).values.tolist()

    
    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)

    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break
        
        Y_chunk = truncated_dirichlet_batch(nums, min_vals, max_vals, dtype=torch.float64).to("cpu").to(dtype=torch.float32)
        X_chunk = simulators(Y_chunk)
        
        X_chunk, Y_chunk = torch.clone(X_chunk[:,:3]), torch.clone(Y_chunk[:,:3])
        
        index_ABC = ABC_rej2(s_dp_tmp, X_chunk, tol0, device, args.task)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)
        print(f"{i}th iteration out of {num_chunks}", flush = True)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    print("X_abc size", X_abc.size())
    print("Y_abc size", Y_abc.size())
    print("X_abc dtype", X_abc.dtype)
    print("Y_abc dtype", Y_abc.dtype)
    

    if args.task in ["cont_table"]:    
        post_sample = torch.load(f"../depot_hyun/NeuralABC_R/{args.task}/post_{args.x0_ind}.pt")
        #post_sample = torch.clone(post_sample[:,:3])
    elif args.task in ["cont_table_dp", "cont_table_dp2", "cont_full", "cont_full2", "cont_full3"]:
        post_sample = torch.load(f"../depot_hyun/NeuralABC_R/{args.task}/post_{args.x0_ind}.pt")
        #post_sample = torch.clone(post_sample[:,:3])
        num_rows = post_sample.shape[0]
        indices = torch.randperm(num_rows)[:10000]
        post_sample = torch.tensor(post_sample[indices,:], dtype = torch.float32)
    elif args.task in ["cont_table_dp_transform"]:
        post_sample = torch.load(f"../depot_hyun/NeuralABC_R/cont_table_dp/post_{args.x0_ind}_80.pt")
        #post_sample = torch.clone(post_sample[:,:3])
        num_rows = post_sample.shape[0]
        indices = torch.randperm(num_rows)[:10000]
        post_sample = torch.tensor(post_sample[indices,:], dtype = torch.float32)
    else:
        print(f"no reference posterior samples avilable for the task {args.task}")
    
    tol = (args.tol/tol0 + 1e-12)

    calibrate_results = calibrate_cov(s_dp_tmp, X_abc, Y_abc, mean_net, covnet, n_samples = 10000, tol = tol, bounds = bounds)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    results = calibrate_results[0].detach().cpu()
    print("NABC sample size: ", results.size())
    results_size = min(10_000, results.size(0))
    results_3 = torch.ones(results_size) - results[:,0] - results[:,1] - results[:,2]

    results= torch.column_stack([results,results_3])

    tmp = c2st(post_sample[:results_size].cpu(), results[:results_size] )
    print(tmp)    
    
    NABC_results.append(tmp)
    
    sci_str = format(tol*tol0, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"./NABC_results/{args.task}_withcov/mean_{int(num_training_mean/1_000)}K_cov_{int(num_training_cov/1_000)}K_resample/amor_{int(args.L/1_000_000)}M_eta{sci_str}"
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save to output_dir
    pairplot(post_sample, figsize=(6,6), limits = [[0,1]]*4)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()

    pairplot(results[:10000], figsize=(6,6), limits = [[0,1]]*4)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
    torch.save(NABC_results, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
    torch.save([torch.cuda.get_device_name(0), elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")
    if args.x0_ind == 3:
        torch.save(results[:10000], f"{output_dir}/samples_seed{args.seed}.pt")
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