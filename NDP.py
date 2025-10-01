import torch
import torch.distributions as D
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from NDP_functions import compute_mad

def NDP_train(X, Y, net_str, device="cpu", p_train=0.7, N_EPOCHS=250, lr=1e-3, val_batch = 10_000, early_stop_patience = 20):
    torch.set_default_device(device)
    X = X.to(device)
    Y = Y.to(device)
    net = copy.deepcopy(net_str)
    net = net.to(device)  # ensure net is on the correct device

    L = Y.size(0)
    L_train = int(L * p_train)
    L_val = L - L_train
    
    indices = torch.randperm(L)

    # Divide Data
    X_train = X[indices[:L_train]]
    Y_train = Y[indices[:L_train]]

    X_val = X[indices[L_train:]]
    Y_val = Y[indices[L_train:]]

    del X, Y  # Free memory for the full dataset

    # Define the batch size
    BATCH_SIZE = 64

    # Use torch.utils.data to create a DataLoader
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))

    def weighted_mse_loss(input, target, weight):
        return (weight * (input - target) ** 2).mean()

    out_range = [
        torch.quantile(Y_train, .01, 0).detach().cpu().numpy(),
        torch.quantile(Y_train, .99, 0).detach().cpu().numpy()
    ]
    weight_1 = torch.tensor(1/(out_range[1] - out_range[0])**2)

    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9)
    #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    weight_decay_scheduler = WeightDecayScheduler(optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10)
    
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
    early_stop = False

    for epoch in range(N_EPOCHS):
        net.train()
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            y_batch_pred = net(x_batch)
            loss = weighted_mse_loss(y_batch_pred, y_batch, weight_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 and id_batch % 300 == 0:
                loss_value = loss.item()
                current = (id_batch + 1)* len(x_batch)
                print(f"train_loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]", flush = True)

        with torch.no_grad():
            net.eval()
            # Evaluate on validation set in batches
            val_loss_accum = 0.0
            for (x_batch, y_batch) in eval_val_dataloader:
                y_pred_batch = net(x_batch)
                batch_loss = weighted_mse_loss(y_pred_batch, y_batch, weight_1).item()
                val_loss_accum += batch_loss
            val_loss = val_loss_accum * val_batch/ L_val
            val_error_plt.append(torch.tensor(val_loss))
                
        if epoch % 10 == 0:
            with torch.no_grad():
                net.eval()
            train_loss_accum = 0.0
            for (x_batch, y_batch) in eval_train_dataloader:
                y_pred_batch = net(x_batch)
                batch_loss = weighted_mse_loss(y_pred_batch, y_batch, weight_1).item()
                train_loss_accum += batch_loss
            train_loss = train_loss_accum * val_batch/ L_train  # Normalize by total number of samples
            train_error_plt.append(torch.tensor(train_loss))  # Store as tensor for consistency
            
            print(f"Epoch {epoch + 1}\n-------------------------------", flush = True)
            print(f"train_loss {train_loss:>7f} val_loss {val_loss:>7f}", flush = True)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(net.state_dict())
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
    
    torch.cuda.empty_cache()
    del net, X_train, Y_train, X_val, Y_val, dataloader, weight_1
    print(f"============= Best validation loss: {best_val_loss} =============", flush = True)
    return best_model_state

def wasserstein_distance_Nto1(X, x0, chunk_size = 100_000, device = 'cuda'):
    num_chunks = X.size(0) // chunk_size
    dist_tmp = []
    
    x0 = x0.to(device)
    x0_sorted, _ = x0.sort(1)    
    for i in range(num_chunks + 1):
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < X.size(0) else X.size(0)

        X_chunk = X[start:end].to(device)
        
        X_sorted_chunk, _ = X_chunk.sort(1)  # Ignore the indices from sort
        
        dist_chunk = torch.sqrt(torch.mean((X_sorted_chunk - x0_sorted) ** 2, dim=1))
        dist_tmp.append(dist_chunk.cpu())  # Move back to CPU to free GPU memory
    
    del X_chunk, x0_sorted, dist_chunk, X_sorted_chunk
    # Clear cached memory
    torch.cuda.empty_cache()
    return torch.cat(dist_tmp, dim=0)

def Markov_sliced_wasserstein_distance(X, x0, k = 2, n_projections= 100, chunk_size=100_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.stack([x0[:, 1:], x0[:, :-1]], dim=2)
    
    x0 = x0.to(device)
    L = X.shape[0]  # Number of time series
    SL_wasserstein_distances = torch.empty(L, dtype=torch.float32)  # Keep results on CPU

    # Directions saved
    direction = torch.randn((k, n_projections), dtype=X.dtype, device=device)
    direction = direction/torch.norm(direction,dim=0)    

    # Process in mini-batches
    for i in range(0, L, chunk_size):
        batch = X[i : i + chunk_size].to(device)
        batch = torch.stack([batch[:, 1:], batch[:, :-1]], dim=2)
        
        swd = torch.zeros(batch.size(0))

        # Compute sliced Wasserstein-2 distance
        for j in range(n_projections):
            # Project both distributions onto the random direction
            X_proj = torch.matmul(batch, direction[:,j])
            x0_proj = torch.matmul(x0, direction[:,j])

            # Compute 1D Wasserstein distance for this projection
            swd += wasserstein_distance_Nto1(X_proj, x0_proj)

        # Move results back to CPU
        SL_wasserstein_distances[i : i + chunk_size] = swd.cpu()

        del batch, swd  # Free memory
        torch.cuda.empty_cache()  # Ensure GPU memory is released

    return SL_wasserstein_distances  # Shape: (N,)

def wasserstein_gaussian(ts_batch, ts_ref, chunk_size=100_000):
    """
    Computes the Wasserstein-2 distance between a large batch of Gaussian-distributed time series 
    and one reference time series, using GPU efficiently.

    Args:
        ts_batch: (N, T) PyTorch tensor on CPU (large batch of time series).    
        ts_ref: (T,) PyTorch tensor on CPU (single reference time series).
        batch_size: Number of samples to process per iteration.
    
    Returns:
        (N,) PyTorch tensor (on CPU) of Wasserstein distances.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Keep ts_batch on CPU and only move batches to GPU
    ts_ref = ts_ref.to(device)  # Only reference needs to be on GPU

    # Compute mean and std of reference time series
    mu_ref = ts_ref.mean()
    sigma_ref = ts_ref.std()

    N = ts_batch.shape[0]  # Number of time series
    wasserstein_distances = torch.empty(N, dtype=torch.float32)  # Keep results on CPU

    # Process in mini-batches
    for i in range(0, N, chunk_size):
        batch = ts_batch[i : i + chunk_size].to(device)  # Move only batch to GPU
        mu_batch = batch.mean(dim=1)  # Mean per time series
        sigma_batch = batch.std(dim=1)  # Std per time series

        # Compute Wasserstein-2 distance
        distances = torch.sqrt((mu_batch - mu_ref) ** 2 + (sigma_batch - sigma_ref) ** 2)

        # Move results back to CPU
        wasserstein_distances[i : i + chunk_size] = distances.cpu()

        del batch, mu_batch, sigma_batch, distances  # Free memory
        torch.cuda.empty_cache()  # Ensure GPU memory is released

    return wasserstein_distances  # Shape: (N,)

def calibrate_cov(x0, X_cal, y_cal, mean_net, covnet, n_samples= 10000, tol = .01, bounds = None, device = "cpu", case = None, chunk_size = 10_000):
    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    mad = compute_mad(X_cal)
    mad = torch.reshape(mad, (1, X_cal.size(1))).to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2, 1))
    
    X_cal, x0, dist = X_cal.cpu(), x0.cpu(), dist.cpu()
    mean_net, covnet = mean_net.to("cpu"), covnet.to("cpu")
    
    sort_dist, _ = torch.sort(dist)
    
    num = X_cal.size(0)
    dim_output = y_cal.size(1)

    # new index
    nacc = int(num * tol)
    
    ds = sort_dist[nacc-1]
    wt1 = (dist <= ds)

    # weights
    h = 1/ torch.log(torch.sum(wt1)) ** (1/dim_output)
    weights = torch.exp(-dist[wt1]/ds/h)
    
    # thetahat
    with torch.no_grad():
        mean_net.eval()
        thetahat = mean_net(x0)
        if bounds is not None:
            thetahat = torch.clamp(thetahat, torch.tensor(bounds)[:,0] ,torch.tensor(bounds)[:,1])

    # Get samples
    covnet = covnet.to(device)
    mean_net = mean_net.to(device)
    y_cal = torch.clone(y_cal[wt1,:])
    X_cal = torch.clone(X_cal[wt1,:])
    
    chunk_size = chunk_size  # Adjust this based on your GPU memory
    num_chunks = X_cal.size(0) // chunk_size

    resid_tmp = []
    sd_X = []

    with torch.no_grad():
        for i in range(num_chunks + 1):
            mean_net.eval()
            covnet.eval()
            start = i * chunk_size
            end = (i + 1) * chunk_size if (i + 1) * chunk_size < X_cal.size(0) else X_cal.size(0)
            
            X_chunk = X_cal[start:end].to(device)
            y_chunk = y_cal[start:end].to(device)
            y_chunk_predict = mean_net(X_chunk)
            if bounds is not None:
                bounds_tensor = torch.tensor(bounds).to(device)
                y_chunk_predict = torch.clamp(y_chunk_predict, bounds_tensor[:,0] ,bounds_tensor[:,1]) 
            resid_chunk = y_chunk - y_chunk_predict
            
            sd_X_chunk = covnet(X_chunk)
            
            resid_tmp.append(resid_chunk.cpu())  # Move back to CPU to free GPU memory
            sd_X.append(sd_X_chunk.cpu())
    
        sd_x0 = covnet(x0.to(device)).cpu()
    
    del X_chunk, y_chunk, resid_chunk, sd_X_chunk
    # Clear cached memory
    torch.cuda.empty_cache()
    
    # Concatenate chunks
    resid_tmp = torch.cat(resid_tmp, dim=0)
    sd_X = torch.cat(sd_X, dim=0)
    identity = torch.eye(sd_X.size(-1), device=sd_X.device)
    Sigma_inv_half = torch.linalg.solve_triangular(sd_X, identity, upper=False) # Getting inverse
    print(sd_X.size())
    with torch.no_grad():    
        # Ensure x has shape (B, 10, 1) — so it's a column vector
        resid_unsqueezed = resid_tmp.unsqueeze(-1)  # (B, 10) → (B, 10, 1)
        
        # Multiply: (B, 10, 10) x (B, 10, 1) → (B, 10, 1)
        output = torch.bmm(Sigma_inv_half, resid_unsqueezed)  # (B, 10, 1)

        # Optional: squeeze back to (B, 10)
        output = output.squeeze(-1)

        adj = torch.matmul(output, sd_x0.squeeze(0).T) + thetahat
    
    weights_tmp = np.copy(weights.detach().cpu().numpy())
    P = weights_tmp / weights_tmp.sum()

    vec = adj.numpy()
    sam_ind = np.random.choice(np.arange(0, adj.size(0)), adj.size(0), replace = False)
    sample_post_large = torch.tensor(vec[sam_ind,:])
    del P, sam_ind
    
    if bounds is not None:
        assert len(bounds) == y_cal.size(1), "The dimension of bounds is not equal to the number of parameter space"
        wt2 = []
        for j in range(len(bounds)):
            tmp = ((sample_post_large[:,j] < bounds[j][1]) & (sample_post_large[:,j] > bounds[j][0]))
            wt2.append(tmp)
        wt2 = torch.stack(wt2, 1)
        wt2 = torch.all(wt2, 1)
        sample_post_large = torch.clone(sample_post_large[wt2,:])
        del wt2
    print(sample_post_large.size()[0])
    if sample_post_large.size(0) < n_samples:
        sample_post = sample_post_large
    else:    
        sam_ind_post = np.random.choice(np.arange(0, sample_post_large.size()[0]), n_samples, replace = False)
        sample_post = sample_post_large[sam_ind_post,:]
    return sample_post, adj

def learning_checking(X, Y, net, num = 10000, name = None):
    net = net.to("cpu")
    X = X.to("cpu")
    Y = Y.to("cpu")
    _, p = Y.size()
    true_name = []
    esti_name = []
    
    for i in range(p):
        true_name.append(r'true $\theta_' + str(i) + '$')
        esti_name.append(r'$\hat{\theta}_' + str(i) + '$')
    
    indices = torch.tensor(np.random.randint(_, size=num)).to("cpu")
    X_test = X[indices,:]
    Y_test = Y[indices,:]
    
    
    with torch.no_grad():
        net.eval()
        tmp = net(X_test)
        tmp = tmp.detach().cpu().numpy()

    ## Plot for model checking
    lim_left = torch.quantile(Y_test,.0001, 0).detach().cpu().numpy()
    lim_right = torch.quantile(Y_test,.9999, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(tmp[0]), figsize=(20,3))
    fig.suptitle('Learning Checking', fontsize= 10)

    for i in range(p):
        lim0 = lim_left[i]
        lim1 = lim_right[i]

        tmp1 = tmp[:, i]
        axes[i].scatter(Y_test[:,i], tmp1, marker='o', color='b', s= 1)
        axes[i].set_xlabel(true_name[i], fontsize=15)
        axes[i].set_ylabel(esti_name[i], fontsize=15)
        axes[i].plot(np.linspace(lim0, lim1, 1000), np.linspace(lim0, lim1, 1000), color = "red", linestyle='dashed', linewidth = 2.5)
        axes[i].set_axisbelow(True)
        axes[i].grid(color='gray', linestyle='dashed')
        axes[i].set_ylim([lim0, lim1])
        axes[i].set_xlim([lim0, lim1])

class WeightDecayScheduler:
    def __init__(self, optimizer, initial_weight_decay, factor, patience):
        """
        Custom scheduler to adjust weight decay.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to adjust weight decay for.
            initial_weight_decay (float): Starting weight decay value.
            factor (float): Multiplicative factor to reduce weight decay.
            patience (int): Number of epochs to wait before reducing weight decay.
        """
        self.optimizer = optimizer
        self.initial_weight_decay = initial_weight_decay
        self.factor = factor
        self.patience = patience
        self.epochs_since_last_update = 0

    def step(self, epoch):
        if self.epochs_since_last_update >= self.patience:
            self.epochs_since_last_update = 0
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] *= self.factor
                print(f"Epoch {epoch}: Reduced weight decay to {param_group['weight_decay']:.6e}")
        else:
            self.epochs_since_last_update += 1

class WeightDecayScheduler_cov:
    def __init__(self, optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10):
        self.optimizer = optimizer
        self.initial_weight_decay = initial_weight_decay
        self.factor = factor
        self.patience = patience
        self.best_epoch = 0
        self.step_count = 0

    def step(self, epoch):
        self.step_count += 1
        if epoch - self.best_epoch >= self.patience:
            for group in self.optimizer.param_groups:
                if 'weight_decay' in group:
                    group['weight_decay'] *= self.factor
                    print(f"[Epoch {epoch}] 🔁 Weight decay adjusted to: {group['weight_decay']:.2e}")
            self.best_epoch = epoch