import torch
import numpy as np
import os, sys
import sys
import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from benchmark.simul_funcs import bernoulli_GLM, get_bernoulli_prior
from module import CovarianceNet, FL_Net
from NDP import WeightDecayScheduler_cov


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    torch.manual_seed(args.seed * 1234)
    
    priors = get_bernoulli_prior()
    simulators = bernoulli_GLM
    
    Y = priors.sample((args.num_training_cov,))
    X = simulators(Y)

    print(X.size(), Y.size())
    net_dir = f"../depot_hyun/hyun/NDP/bernoulli_glm/train_{int(args.num_training_mean/1_000)}K/bernoulli_glm{args.seed}_mean.pt"
    tmp = torch.load(net_dir)

    D_in, D_out, Hs = 10, 10, 256
    mean_net = FL_Net(D_in, D_out, Hs, Hs, Hs)
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
    val_batch = 10_000
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


    # Model & optimizer
    covnet = CovarianceNet(input_dim=10, output_dim=10, hidden_dim=args.layer_len)
    optimizer = torch.optim.Adam(covnet.parameters(), lr=1e-3, weight_decay = 1e-5)


    # Define the batch size
    BATCH_SIZE = 64

    # Use torch.utils.data to create a DataLoader
    print(X_train.size())
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9)
    #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    weight_decay_scheduler = WeightDecayScheduler_cov(optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10)


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


    UNFREEZE_EPOCH = 10

    # Training loop
    for epoch in range(N_EPOCHS):
        mean_net.train()
        covnet.train()
        total_loss = 0.0
        #for x_batch, y_batch in loader:
        
            # Unfreeze mean_net at scheduled epoch
        if epoch == UNFREEZE_EPOCH:
            print("🔓 Unfreezing mean_net for joint fine-tuning!")
            for param in mean_net.parameters():
                param.requires_grad = True

            # Rebuild optimizer with both mean and cov parameters
            optimizer = torch.optim.Adam([
                {'params': mean_net.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
                {'params': covnet.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5}
            ])
                # Re-init schedulers to bind to new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9
            )
            weight_decay_scheduler = WeightDecayScheduler_cov(
                optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10
            )

        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
                
            mu_batch = mean_net(x_batch)  # (batch, 10)

            L_batch = covnet(x_batch)  # (batch, 10, 10)
            loss = gaussian_nll(y_batch, mu_batch, L_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch % 5 == 0 and id_batch % 300 == 0:
                loss_value = loss.item()
                current = (id_batch + 1)* len(x_batch)
                print(f"train_loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]", flush = True)

        with torch.no_grad():
            covnet.eval()
            mean_net.eval()
            # Evaluate on validation set in batches
            val_loss_accum = 0.0
            for (x_batch, y_batch) in eval_val_dataloader:
                mu_batch = mean_net(x_batch)
                L_batch = covnet(x_batch)  # (batch, 10, 10)
            
                batch_loss = gaussian_nll(y_batch, mu_batch, L_batch).item()
                val_loss_accum += batch_loss
            val_loss = val_loss_accum * val_batch/ len(eval_val_dataloader)
            val_error_plt.append(torch.tensor(val_loss))

        if epoch % 5 == 0:
            with torch.no_grad():
                covnet.eval()
                mean_net.eval()
            train_loss_accum = 0.0
            for (x_batch, y_batch) in eval_train_dataloader:
                mu_batch = mean_net(x_batch)
                L_batch = covnet(x_batch)  # (batch, 10, 10)
            
                batch_loss = gaussian_nll(y_batch, mu_batch, L_batch).item()
                train_loss_accum += batch_loss
            train_loss = train_loss_accum * val_batch/ L_train  # Normalize by total number of samples
            train_error_plt.append(torch.tensor(train_loss))  # Store as tensor for consistency

            print(f"Epoch {epoch + 1}\n-------------------------------", flush = True)
            print(f"train_loss {train_loss:>7f} val_loss {val_loss:>7f}", flush = True)
        
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

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        weight_decay_scheduler.step(epoch)

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