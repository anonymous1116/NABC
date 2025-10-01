import torch
import numpy as np
import torch.distributions as D
import os, sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbi.utils import BoxUniform
import time
from NDP_functions import learning_checking, ABC_rej2
from benchmark.simulator import Priors,Bounds, simulator_MoG, MoG_posterior
from sbibm.metrics.c2st import c2st


def main(args):
    print("torch cuda:", torch.cuda.get_device_name(0))
    L = args.L
    x0_ind = args.x0_ind
    tol = args.tol
    seed = args.seed

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chunk_size = 50_000_000
    num_chunks = L // chunk_size

    # ABC_rej
    
    for dim_out in range(args.dim_start, args.dim_end):
        torch.manual_seed(seed)
        np.random.seed(seed)   

        priors = BoxUniform(low = -10*torch.ones(dim_out), high = 10*torch.ones(dim_out))
        simulators = simulator_MoG

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

        X_abc = []
        Y_abc = []
        start_time = time.time()
    
        for i in range(num_chunks + 1): 
            start = i * chunk_size
            end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
            nums = end-start

            if nums == 0:
                break
            else:
                Y_chunk = priors.sample((nums,))
            X_chunk = simulators(Y_chunk)
            
            index_ABC = ABC_rej2(x0, X_chunk, tol, device, args.task)
            X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
            X_abc.append(X_chunk)
            Y_abc.append(Y_chunk)
            print(f"{i+1}th iter out of {num_chunks}", flush = True)
            del X_chunk, Y_chunk, index_ABC
            torch.cuda.empty_cache()
    

        X_abc = torch.cat(X_abc)
        Y_abc = torch.cat(Y_abc)    
        

        # True distributions
        n_samples = 10000
        true_samples = MoG_posterior(x0, 10_000)
        if Y_abc.size(1) > n_samples:
            sam_ind_post = np.random.choice(np.arange(0, Y_abc.size()[0]), n_samples, replace = False)
            Y_abc = Y_abc[sam_ind_post,:]
        
        end_time = time.time()
    
        c2st_results = c2st(true_samples, Y_abc)
        print(f"c2st_results {c2st_results} for dim {dim_out}")
        sci_str = format(tol, ".0e")
        print(sci_str)  # Output: '1e-02'
    

        output_dir = f"./NABC_results/ABC_results/{args.task}/{int(L/1_000_000)}M_eta{sci_str}/dim_out{dim_out}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")
        
        if args.x0_plot == 1:
            torch.save(c2st_results, f"{output_dir}/x0_0_seed{seed}.pt")
            if args.seed == 1:
                torch.save(Y_abc, f"{output_dir}/x0_0_samples.pt")
        else:
            torch.save(c2st_results, f"{output_dir}/x0{x0_ind}_seed{seed}.pt")
        
        torch.save(end_time-start_time, f"{output_dir}/x0{x0_ind}_seed{seed}_time.pt")
        
        

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--L", type = int, default = 100_000_000,
                        help = "Number of calibration data (default: 1)")
    parser.add_argument("--x0_ind", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerance value for ABC (any positive float, default: 1e-4 but less than 1e-2)")
    parser.add_argument('--task', type=str, default='MoG', 
                        help='Simulation type: MoG')
    parser.add_argument('--x0_plot', type=int, default=0, 
                        help='1: x0_plot, 0:x0_plot no')
    parser.add_argument('--dim_start', type=int, default=1, 
                        help='dim_start')
    parser.add_argument('--dim_end', type=int, default=10, 
                        help='dim_end')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
