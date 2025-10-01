import torch
import torch.nn as nn
import numpy as np
import os, sys
import sys
import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import copy
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from module import GRU_CovarianceNet, GRU_net, GRU_net_bounded
from NDP import WeightDecayScheduler_cov
from benchmark.simulator import Simulators, Priors, Bounds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    print(torch.cuda.get_device_name(0), flush=True)

    torch.manual_seed(args.seed * 1234)
    
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    
    Y = priors.sample((args.num_training_cov,))
    X = simulators(Y)

    print(X.size(), Y.size())
    net_dir = f"../depot_hyun/hyun/NDP/{args.task}/train_{int(args.num_training_mean/1_000)}K_{int(args.layer_len)}/{args.task}{args.seed}_mean.pt"
    tmp = torch.load(net_dir)

    bounds = Bounds(args.task)

    D_in, D_out, Hs = X.size(1), Y.size(1), args.layer_len

    if bounds is None:
        mean_net = GRU_net(input_dim = 1, hidden_dim = Hs, output_dim = D_out).to(device)
    else:
        mean_net = GRU_net_bounded(input_dim = 1, hidden_dim = Hs, output_dim = D_out, bounds = bounds).to(device)
    
    
    mean_net.load_state_dict(tmp)
    
    mean_net.eval()  # We don't train this at the beginning
    mean_net = mean_net.to(device)

    torch.set_default_device(device)
        
    def gaussian_nll(y, mu, L):
        d = y.size(1)
        residual = (y - mu).unsqueeze(-1)  # (batch, d, 1)
        Linv_residual = torch.linalg.solve_triangular(L, residual, upper=False)
        mahal = (Linv_residual ** 2).sum(dim=1).squeeze()
        log_det = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        nll = 0.5 * (mahal + log_det + d * torch.log(torch.tensor(2 * torch.pi, device=y.device)))
        return nll.mean()
    
    p_train = 0.7
    val_batch = 500
    N_EPOCHS = args.N_EPOCHS
    early_stop_patience = 30

    X = X.to(device)
    Y = Y.to(device)
        
    L = Y.size(0)
    L_train = int(L * p_train)
    L_val = L - L_train

    indices = torch.randperm(L)

    # Divide Data
    X_train = X[indices[:L_train]]
    Y_train = Y[indices[:L_train]]

    X_val = X[indices[L_train:]]
    Y_val = Y[indices[L_train:]]


    # Model 
    covnet = GRU_CovarianceNet(input_dim=1, output_dim=D_out, hidden_dim=args.layer_len)
    

    # Define the batch size
    BATCH_SIZE = 64

    # Use torch.utils.data to create a DataLoader
    print(X_train.size())
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))


    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9)
    #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    #weight_decay_scheduler = WeightDecayScheduler_cov(optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10)

    train_error_plt = []
    val_error_plt = []

    best_val_loss = float('inf')
    best_model_state = None

    # Create DataLoader for entire training set (for evaluation)
    eval_train_dataset = TensorDataset(X_train, Y_train)
    eval_train_dataloader = DataLoader(eval_train_dataset, batch_size=val_batch, shuffle=True, generator=torch.Generator(device=device))

    # Create DataLoader for validation set (for evaluation)
    eval_val_dataset = TensorDataset(X_val, Y_val)
    eval_val_dataloader = DataLoader(eval_val_dataset, batch_size=val_batch, shuffle=True, generator=torch.Generator(device=device))
    epochs_no_improve = 0


    # Dataset + DataLoader
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))


    UNFREEZE_EPOCH = 20
    
    
    # --- Setup: start frozen ---
    for p in mean_net.parameters():
        p.requires_grad = False

    mean_net.eval()  # deterministic outputs while frozen
    covnet.train()

    optimizer = torch.optim.Adam([
        {'params': covnet.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9
    )
    weight_decay_scheduler = WeightDecayScheduler_cov(
        optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10
    )

    Joint_train=False
    start_time = time.time()
    def disable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.eval()   # disable dropout, but still allows grads
        
    for epoch in range(N_EPOCHS):

        # ===== Phase switch =====
        if (not Joint_train) and (epoch >= UNFREEZE_EPOCH):
            Joint_train = True
            print("🔓 Unfreezing mean_net for joint fine-tuning!")

            for p in mean_net.parameters():
                p.requires_grad = True

            # Rebuild optimizer to include mean_net (resets Adam state; acceptable here)
            optimizer = torch.optim.Adam([
                {'params': mean_net.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
                {'params': covnet.parameters(),   'lr': 1e-3, 'weight_decay': 1e-5}
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9
            )
            weight_decay_scheduler = WeightDecayScheduler_cov(
                optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10
            )
            
        # ===== Set modes for this phase =====
        if Joint_train:
            mean_net.train(True)   # IMPORTANT: train mode before forward
            mean_net.apply(disable_dropout)   # disable only Dropout layers
        else:
            mean_net.eval()        # deterministic while frozen
        covnet.train(True)

        total_loss = 0.0
        n_seen = 0
                # Unfreeze mean_net at scheduled epoch
        
            
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # ----- Forward -----
            if Joint_train:
                # train mean_net: allow grads, no eval() here
                mu_batch = mean_net(x_batch)
            else:
                # frozen mean_net: eval + no grad (no BN updates, no dropout)
                with torch.no_grad():
                    mu_batch = mean_net(x_batch)

            L_batch = covnet(x_batch)
            loss = gaussian_nll(y_batch, mu_batch, L_batch)

            # ----- Backward/step -----
            loss.backward()
            optimizer.step()

            bs = x_batch.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

            if epoch % 20 == 0 and id_batch % 300 == 0:
                print(f"train_loss: {loss.item():.6f}  "
                    f"[{(id_batch+1)*bs:>6d}/{len(dataloader.dataset):>6d}]",
                    flush=True)

        train_epoch_loss = total_loss / max(1, n_seen)
        
            
        # ===== Validation (always eval mode, no grad) =====
        mean_net.eval()
        covnet.eval()
        with torch.no_grad():
            val_total, val_seen = 0.0, 0
            for x_val, y_val in eval_val_dataloader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                mu_val = mean_net(x_val)
                L_val  = covnet(x_val)
                val_total += gaussian_nll(y_val, mu_val, L_val).item() * x_val.size(0)
                val_seen  += x_val.size(0)
        val_loss = val_total / max(1, val_seen)

        scheduler.step(val_loss)
        weight_decay_scheduler.step(epoch)  # or .step() per your API
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch + 1}\n-------------------------------", flush = True)
            print(f"train_loss {train_epoch_loss:>7f} val_loss {val_loss:>7f}", flush = True)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            best_model_mean_state = copy.deepcopy(mean_net.state_dict())
            best_model_cov_state = copy.deepcopy(covnet.state_dict())
            
            epochs_no_improve = 0  # Reset the counter for early stopping
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve}/{early_stop_patience} epochs." , flush = True)

        # Early stopping condition
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered. Restoring best model...", flush = True)
            print(f"It stops within {epoch}/{N_EPOCHS}", flush = True)
            early_stop = True
            break


    end_time = time.time() 
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    output_dir = f"../depot_hyun/hyun/NDP/{args.task}/mean_{int(args.num_training_mean/1000)}K_cov_{int(args.num_training_cov/1_000)}K_layer_{args.layer_len}"
    
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    torch.save(best_model_mean_state, f"{output_dir}/best_model_mean_state_{args.seed}.pt")
    torch.save(best_model_cov_state, f"{output_dir}/best_model_cov_state_{args.seed}.pt")
    torch.save([val_error_plt, best_val_loss] , f"{output_dir}/val_error_plt_{args.seed}.pt")
    torch.save(elapsed_time,  f"{output_dir}/{args.task}{str(args.seed)}cov_time.pt")
    torch.save(torch.cuda.get_device_name(0), f"{output_dir}/{args.task}{str(args.seed)}cov_gpu.pt")
    

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: Lapl, MoG')
    parser.add_argument("--num_training_mean", type=int, default=300_000, 
                        help="Number of training data for mean function (default: 300_000)")
    parser.add_argument("--num_training_cov", type=int, default=300_000, 
                        help="Number of training data for cov function (default: 300_000)")
    parser.add_argument("--layer_len", type=int, default=128, 
                        help="Number of layers for covnet (default: 128)")
    parser.add_argument("--N_EPOCHS", type=int, default=200, 
                        help="Number of epochs (default: 200)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    
    print(f"seed: {args.seed}")
    print(f"task: {args.task}")
    print(f"num_training_mean: {args.num_training_mean}")
    print(f"num_training_cov: {args.num_training_cov}")
    print(f"layer_len: {args.layer_len}")
    print(f"N_EPOCHS: {args.N_EPOCHS}")