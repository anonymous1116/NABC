import torch
import numpy as np
import os, sys
import argparse
import sbibm
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from NDP import calibrate, calibrate_cov
from NDP_functions import synthetic_pairs_Lapl, ABC_rej, ABC_rej2, synthetic_pairs_MoG, MoG
from module import FL_Net, CovarianceNet
from sbibm.metrics.c2st import c2st
from benchmark.simul_funcs import get_bernoulli_prior, bernoulli_GLM, truncated_mvn_sample


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

    sbi_task = sbibm.get_task(args.task)  # See sbibm.get_available_tasks() for all tasks
    
    priors = get_bernoulli_prior()
    simulators = bernoulli_GLM
    
    net_dir = f"../depot_hyun/hyun/NDP/{args.task}/mean_{int(num_training_mean/1000)}K_cov_{int(num_training_cov/1_000)}K_layer_{args.layer_len}"
    assert os.path.exists(net_dir), f"Model directory {net_dir} does not exist"

    tmp1 = torch.load(f"{net_dir}/best_model_mean_state_{seed}.pt")
    tmp2 = torch.load(f"{net_dir}/best_model_cov_state_{seed}.pt")
    tmp3 = torch.load(f"{net_dir}/val_error_plt_{seed}.pt")

    print(f"best_val: {tmp3[1]}", flush=True)

    # Learning hyperparameters
    D_in, D_out, Hs = 10, 10, 256
    mean_net = FL_Net(D_in, D_out, Hs, Hs, Hs)
    mean_net.load_state_dict(tmp1)
    
    covnet = CovarianceNet(input_dim=10, output_dim=10, hidden_dim=args.layer_len)
    covnet.load_state_dict(tmp2)
        
    # Initialize the Priors and Simulators classes and ABC_me   thods
    tol0 = 1e-3
    chunk_size = 5_000_000
    num_chunks = L // chunk_size

    
    s_dp_tmp = sbi_task.get_observation(num_observation = args.x0_ind)
        
    Y_cal = priors.sample((1_000_000,))
    X_cal = bernoulli_GLM(Y_cal)

    chunk_size_cal = 10_000
    with torch.no_grad():
        _, adj = calibrate_cov(s_dp_tmp, X_cal, Y_cal, mean_net, covnet, tol=1, device = device, case = args.task, chunk_size=chunk_size_cal)
    
    X_abc = []
    Y_abc = []
    with torch.no_grad():
        max_vals = torch.max(adj,0).values
        min_vals = torch.min(adj,0).values

    priors_mean = torch.zeros(10)
    priors_std = torch.ones(10) * np.sqrt([2])

    print("ABC start", flush = True)
    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break
        Y_chunk = truncated_mvn_sample(nums, priors_mean, priors_std, min_vals, max_vals)
        X_chunk = bernoulli_GLM(Y_chunk)
        
        index_ABC = ABC_rej2(s_dp_tmp, X_chunk, tol0, device)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    print("ABC done X_abc size", X_abc.size())
        
    post_sample = torch.load(f"../depot_hyun/NeuralABC_R/bernoulli_glm/post_{args.x0_ind}.pt")
    
    tol = (args.tol/tol0 + 1e-12)

    calibrate_results = calibrate_cov(s_dp_tmp, X_abc, Y_abc, mean_net, covnet, n_samples = 10000, tol = tol)
    print("NABC sample size: ", calibrate_results[0].detach().cpu().size())
    
    tmp = c2st(post_sample.cpu(), calibrate_results[0].detach().cpu()[:10000] )
    print(f"c2st: {tmp}")    
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