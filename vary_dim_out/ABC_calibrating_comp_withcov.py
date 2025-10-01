import torch
import numpy as np
import os, sys
import sys
import argparse
from sbi.utils import BoxUniform
from sbibm.metrics.c2st import c2st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from NDP import NDP_train, calibrate, calibrate_cov
from NDP_functions import synthetic_pairs_Lapl, ABC_rej, synthetic_pairs_MoG, MoG
from module import FL_Net, FL_Net_bounded, CovarianceNet
from NDP_resampling import UnifSample, param_box
from NDP_functions import ABC_rej2
from benchmark.simulator import Priors,Bounds, simulator_MoG, MoG_posterior

def main(args):
    seed = args.seed
    L = args.L
    task = args.task
    num_training_mean = args.num_training
    num_training_cov = args.num_training * 2
    x0_ind = args.x0_ind
    layer_len = 256

    for dim_out in range(args.dim_start, args.dim_end):
        # Simulator
        priors = BoxUniform(low = -10*torch.ones(dim_out), high = 10*torch.ones(dim_out))
        simulators = simulator_MoG

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.set_default_device("cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)   

        net_dir = f"../depot_hyun/hyun/NDP/{args.task}/mean_{int(num_training_mean/1000)}K_cov_{int(num_training_cov/1_000)}K_layer_{layer_len}/dim_out_{dim_out}/best_model_mean_state_{seed}.pt"
        net_dir2 = f"../depot_hyun/hyun/NDP/{args.task}/mean_{int(num_training_mean/1000)}K_cov_{int(num_training_cov/1_000)}K_layer_{layer_len}/dim_out_{dim_out}/best_model_cov_state_{seed}.pt"
        
        tmp = torch.load(net_dir)
        tmp2 = torch.load(net_dir2)

        # Learning hyperparameters
        D_in, D_out, Hs = dim_out, dim_out, layer_len
        bounds_base = [-5,5] if task == "Lapl" else [-10,10]  
        bounds = [bounds_base]*dim_out

        net = FL_Net_bounded(D_in, D_out, H=Hs, p = 0.1, bounds = bounds)
        net2 = CovarianceNet(input_dim=D_in, output_dim=D_out, hidden_dim=layer_len)
        
        net.load_state_dict(tmp)
        net2.load_state_dict(tmp2)

        net.to("cpu")
        net2.to("cpu")

        net.eval()
        net2.eval()

        torch.set_default_device("cpu")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # x0
        if args.x0_plot == 1:
            x0 = torch.zeros((1,dim_out))
        else:
            x0_list = torch.load(f"../depot_hyun/NeuralABC_R/NDP_MoG/x0_list.pt")
            tmp = torch.tensor(x0_list[int(dim_out - 1)][int(x0_ind-1)], dtype = torch.float32)
            x0 = torch.reshape(tmp, (1, tmp.size(0)))
        print("x0:", x0)

        if x0.ndim == 1:
            x0 = torch.reshape(x0, (1, x0.size(0)))
        print("x0 size:",  x0.size())

        # Initialize the Priors and Simulators classes and ABC_methods
        tol0 = 1e-2
        Y_cal = priors.sample((1_000_000,))
        X_cal = simulators(Y_cal)

        chunk_size_cal = 10_000
        with torch.no_grad():
            _, adj = calibrate_cov(x0, X_cal, Y_cal, net, net2, tol=.01, device = device, case = args.task, chunk_size=chunk_size_cal, bounds = bounds)
    
            
        X_abc = []
        Y_abc = []

        chunk_size = 50_000_000
        num_chunks = L // chunk_size
    
        for i in range(num_chunks + 1): 
            start = i * chunk_size
            end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
            nums = end-start

            if nums == 0:
                break
            Y_chunk = param_box(UnifSample(bins = 10), adj, num=nums)
            X_chunk = simulators(Y_chunk)
            
            index_ABC = ABC_rej2(x0, X_chunk, tol0, device, args.task)
            X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
            X_abc.append(X_chunk)
            Y_abc.append(Y_chunk)
            print(f"{i+1}th iteration out of {num_chunks}", flush = True)

        X_abc = torch.cat(X_abc)
        Y_abc = torch.cat(Y_abc)    

        print(X_abc.size())
        # True distributions
        n_samples = 10000
        true_samples = MoG_posterior(x0, n_samples, bounds= bounds)
        
        tol = (args.tol/tol0 + 1e-12)


        calibrate_results = calibrate_cov(x0, X_abc, Y_abc, net, net2, n_samples = 10000, tol = tol, bounds = bounds)
        
        print("NABC sample size: ", calibrate_results[0].detach().cpu().size())
        results_size = min(10_000, calibrate_results[0].detach().cpu().size(0))

        c2st_results = c2st(true_samples[:results_size].cpu(), calibrate_results[0].detach().cpu()[:results_size] )
        print(f"c2st_results {c2st_results} for dim {dim_out}")
        sci_str = format(args.tol, ".0e")
        print(sci_str)  # Output: '1e-02'
    
        output_dir = f"./NABC_results/{task}/J{int(num_training_mean/1_000)}K_withcov/amor_{int(args.L/1_000_000)}M_eta{sci_str}/{dim_out}"
        ## Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")

        if args.x0_plot == 1:
            torch.save(c2st_results, f"{output_dir}/x0_0_seed{seed}.pt")
            if args.seed == 1:
                torch.save(calibrate_results[0].detach().cpu(), f"{output_dir}/x0_0_samples.pt")
        else:
            torch.save(c2st_results, f"{output_dir}/x0{x0_ind}_seed{seed}.pt")
        
        
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
    parser.add_argument('--x0_plot', type=int, default=0, 
                        help='1: x0_plot, 0:x0_plot no')
    parser.add_argument('--dim_start', type=int, default=2, 
                        help='dim_start')
    parser.add_argument('--dim_end', type=int, default=11, 
                        help='dim_end')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"x0_ind: {args.x0_ind}")
    print(f"seed: {args.seed}")
    print(f"L: {args.L}")
    print(f"task: {args.task}")