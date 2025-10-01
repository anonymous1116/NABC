import torch
import numpy as np
import os, sys
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from NDP import NDP_train, calibrate
from NDP_functions import synthetic_pairs_Lapl, ABC_rej, synthetic_pairs_MoG
from module import FL_Net, FL_Net_bounded
from sbibm.metrics.c2st import c2st


def main(args):
    seed = args.seed
    L = args.L
    NABC_results = []
    task = args.task

    # Select the correct synthetic pair generator
    if task == "Lapl":
        synthetic_fn = synthetic_pairs_Lapl
    elif task == "MoG":
        synthetic_fn = synthetic_pairs_MoG
    else:
        raise ValueError(f"Unsupported task: {task}")

    for dim_out in range(1, 11):
        num_training = args.num_training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.set_default_device("cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)   

        net_dir = f"../depot_hyun/hyun/NDP/{task}/train_{int(num_training/1_000)}K/dim_out_{dim_out}/{task}{seed}_mean.pt"
        tmp = torch.load(net_dir)

        # Learning hyperparameters
        D_in, D_out, Hs = dim_out, dim_out, 256
        bounds_base = [-5,5] if task == "Lapl" else [-10,10]  
    
        net = FL_Net_bounded(D_in, D_out, H=Hs, p = 0.1, bounds = [bounds_base]*dim_out)
        net.load_state_dict(tmp)
        net.eval()

        torch.set_default_device("cpu")

        net.to("cpu")
        net.eval()
        #net(s_dp_Lapl(dim_out))

        net2_dir = f"../depot_hyun/hyun/NDP/{task}/train_{int(num_training/1_000)}K/dim_out_{dim_out}/{task}{seed}_cMAD.pt"
        tmp2 = torch.load(net2_dir)

        net2 = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs)
        net2.load_state_dict(tmp2)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.set_default_device("cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)   

        # Initialize the Priors and Simulators classes and ABC_methods
        tol0 = 0.01
        chunk_size = 5_000_000
        num_chunks = L // chunk_size

        out_folder = "NDP" if args.task == "Lapl" else f"NDP_{task}"
        x0_list = torch.load(f"../depot_hyun/NeuralABC_R/{out_folder}/x0_list.pt")

        tmp = torch.tensor(x0_list[int(dim_out - 1)][int(args.x0_ind-1)], dtype = torch.float32)
        s_dp_tmp = torch.reshape(tmp, (1, tmp.size(0)))
        print(s_dp_tmp)
        #s_dp_tmp = s_dp_Lapl(dim_out)

        X_abc = []
        Y_abc = []

        for i in range(num_chunks + 1): 
            start = i * chunk_size
            end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
            nums = end-start
            if nums == 0:
                break
            X_chunk, Y_chunk =  synthetic_fn(nums, dim_out)
            index_ABC = ABC_rej(s_dp_tmp, X_chunk, tol0, device)
            X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
            X_abc.append(X_chunk)
            Y_abc.append(Y_chunk)

        X_abc = torch.cat(X_abc)
        Y_abc = torch.cat(Y_abc)    

        print(X_abc.size())
        
        post_sample = torch.load(f"../depot_hyun/NeuralABC_R/{out_folder}/post_dim_{dim_out}_x0_ind_{args.x0_ind}.pt")
        
        tol = (args.tol/tol0 + 1e-12)

        calibrate_results = calibrate(s_dp_tmp, X_abc, Y_abc, net, net2, n_samples = 10000, tol = tol)
        print("NABC sample size: ", calibrate_results[0].detach().cpu().size(), flush = True)
        tmp = c2st(post_sample.cpu(), calibrate_results[0].detach().cpu()[:10000] )
        print(tmp, flush = True)    
        del net, net2
        NABC_results.append(tmp)

    sci_str = format(tol*tol0, ".0e")
    print(sci_str)  # Output: '1e-02'
    
    output_dir = f"./NABC_results/{task}/{int(num_training/1_000)}K/amor_{int(args.L/1_000_000)}M_eta{sci_str}"
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    torch.save(NABC_results, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")

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