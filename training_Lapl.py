import torch
import numpy as np
import argparse
import os
import subprocess
from sbi import utils as utils
from sbibm.metrics.c2st import c2st

from module import FL_Net
import time
from NDP import NDP_train, calibrate
from NDP_functions import ABC_rej, synthetic_pairs_Lapl, s_dp_Lapl, true_samples_Lapl
# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   

    # Initialize the Priors and Simulators classes and ABC_methods
    tol0 = 0.01/2
    chunk_size = 5_000_000
    num_chunks = args.L // chunk_size
    
    s_dp_tmp = s_dp_Lapl(args.dim_out)
    
    X_abc = []
    Y_abc = []

    for i in range(num_chunks + 1):
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < args.L else args.L
        nums = end-start
        if nums == 0:
            break
        X_chunk, Y_chunk =  synthetic_pairs_Lapl(nums, args.dim_out)
        index_ABC = ABC_rej(s_dp_tmp, X_chunk, tol0, device)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    X_train = X_abc[:100000,:]
    Y_train = Y_abc[:100000,:]

    # Learning hyperparameters
    D_in, D_out, Hs = X_train.size(1), Y_train.size(1), args.layer_len

    # Save the models
    ## Define the output directory
    print(f"start", flush=True)
    output_dir = f"../depot_hyun/hyun/NDP/{args.task}/L_{int(args.L/1_000_000)}M/dim_out_{args.dim_out}"
    
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    net = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to(device)
        
    # Train Mean Function
    print(f"start training for mean function", flush=True)
    start_time = time.time()  # Start timer
    val_batch = 10_000
    early_stop_patience = 50
    tmp = NDP_train(X_train, Y_train, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch, early_stop_patience = early_stop_patience)
    end_time = time.time() 
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Mean Function Training completed in {elapsed_time/60:.2f} mins", flush=True)
    
    net.load_state_dict(tmp)
    net = net.to("cpu")

    torch.save(net.state_dict(),  output_dir + "/" + args.task + str(args.seed) +"_mean.pt")
    torch.save(elapsed_time,  output_dir + "/" + args.task + str(args.seed) +"_time.pt")
    torch.save(torch.cuda.get_device_name(0), output_dir + "/" + args.task + str(args.seed)+ "_gpu.pt")
    
    
    print(f"Mean Function saved", flush=True)
    torch.set_default_device("cpu")

    print(f"cMAD learning start", flush=True)
    
    X_train2 = X_abc[100000:200000,:]
    Y_train2 = Y_abc[100000:200000,:]
    net.eval()
    resid = (Y_train2.detach().cpu() - net(X_train2).detach().cpu())
    
    resid = torch.max(torch.abs(resid), torch.ones(1) * 1e-30).log()
    net2 = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs)

    tmp2 = NDP_train(X_train2, resid, net2, device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch, early_stop_patience = 50)

    net2.load_state_dict(tmp2)
    net2 = net2.to("cpu")

    torch.save(net2.state_dict(),  output_dir + "/" + args.task + str(args.seed) +"_cMAD.pt")
    torch.save(elapsed_time,  output_dir + "/" + args.task + str(args.seed) +"_time_cMAD.pt")
    print("## cMAD training job script completed ##", flush=True)


    print("## Calibrate start", flush=True)
    torch.set_default_device("cpu")
    
    tol_seq = np.arange(0.001, 0.002, 0.001) * 2
    #tol_seq = np.arange(0.001, 0.002, 0.001) * 2
    #tol_seq = np.arange(0.01, 0.02, 0.01) * 2

    post_sample  = true_samples_Lapl(args.dim_out)

    NABC_results = []
    NABC_post = []
    for j in range(len(tol_seq)):
        calibrate_results = calibrate(s_dp_tmp, X_abc, Y_abc, net, net2, n_samples = 10000, tol = tol_seq[j])
        tmp = c2st(post_sample.cpu(), calibrate_results[0].detach().cpu()[:10000] )
        NABC_results.append(tmp[0].numpy().tolist())

        NABC_post.append(calibrate_results[0].detach().cpu()[:10000])
        print(f"{j}th calibration eneded", flush=True)
    torch.save(NABC_results, output_dir + "/calib_" + args.task + str(args.seed) +".pt")
    
    print("## Calibration ended", flush=True)

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: twomoons, MoG, MoUG, Lapl, GL_U or slcp, slcp2')
    parser.add_argument("--N_EPOCHS", type=int, default=100, 
                        help="Number of EPOCHS (default: 100)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--layer_len", type = int, default = 256,
                        help = "layer length of FL network (default: 256)")
    parser.add_argument("--L", type=int, default=100_000_000,
                        help="Number of calibrations for sampling (default: 10_000_000)")
    parser.add_argument("--dim_out", type = int, default = 1,
                        help = "See number (default: 1)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    #main_cond(args)
    
    # Use the parsed arguments
    print(f"task: {args.task}")
    print(f"Number of epochs: {args.N_EPOCHS}")
    print(f"seed: {args.seed}")