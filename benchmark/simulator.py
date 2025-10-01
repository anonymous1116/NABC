import torch
import sbibm
import os, sys
import torch.distributions as D
from torch.distributions import MultivariateNormal, Dirichlet, Multinomial
from sbi.utils import BoxUniform

import numpy as np
# Optional: you can use this from torch.distributions if available
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from NDP_functions import SLCP_summary, SLCP_summary_transform, SLCP_summary_transform2, cont_table_transform

def Bounds(task_name: str):
    task_name = task_name.lower()
    if task_name == "bernoulli_glm":
        return None
    elif task_name in ["slcp", "slcp_summary", "slcp2_summary", "slcp3", "slcp3_summary", "my_slcp", "my_slcp2","my_slcp4", "slcp_summary_transform", "slcp_summary_transform2"]:
        return [[-3,3]] * 5
    elif task_name in ["mog_2"]:
        return [[-10, 10]] * 2
    elif task_name in ["mog_5", "lapl_5"]:
        return [[-10, 10]] * 5
    elif task_name in ["mog_10", "lapl_10"]:
        return [[-10, 10]] * 10
    elif task_name in ["my_twomoons"]:
        return [[-5, 5]] * 2
    elif task_name in ["my_slcp3"]:
        return [[-5, 5]] * 5
    elif task_name in ["ou"]:
        return [[1, 5], [1, 2.5], [0.5, 2.0]]
    elif task_name in ["cont_table", "cont_table_dp", "cont_table_dp2", "cont_table_dp_transform", "cont_full", "cont_full2", "cont_full3"]:
        return [[0,1]] * 3
    else:
        raise ValueError(f"Unknown task name for bounds: {task_name}")

def Priors(task_name: str):
    task_name = task_name.lower()
    if task_name == "bernoulli_glm":
        dim = 10
        loc = torch.zeros(dim)
        precision_diag = 0.5 * torch.ones(dim)
        precision_matrix = torch.diag(precision_diag)
        return MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
    elif task_name in ["slcp", "slcp_summary", "slcp2_summary", "slcp3", "slcp3_summary", "my_slcp", "my_slcp2", "my_slcp4", "slcp_summary_transform", "slcp_summary_transform2"]:
        return BoxUniform(low=-3 * torch.ones(5), high=3 * torch.ones(5))
    elif task_name in ["mog_2"]:
        return BoxUniform(low = -10*torch.ones(2), high = 10*torch.ones(2))
    elif task_name in ["mog_5", "lapl_5"]:
        return BoxUniform(low = -10*torch.ones(5), high = 10*torch.ones(5))
    elif task_name in ["mog_10", "lapl_10"]:
        return BoxUniform(low = -10*torch.ones(10), high = 10*torch.ones(10))
    elif task_name in ["my_twomoons"]:
        return BoxUniform(low = -5*torch.ones(2), high = 5*torch.ones(2))
    elif task_name in ["my_slcp3"]:
        return BoxUniform(low = -5*torch.ones(5), high = 5*torch.ones(5))
    elif task_name in ["ou"]:
        return BoxUniform(low = torch.tensor([1.0, 1.0, 0.5]), high = torch.tensor([5.0, 2.5, 2.0]))
    elif task_name in ["cont_table", "cont_table_dp", "cont_table_dp2", "cont_table_dp_transform", "cont_full", "cont_full2", "cont_full3"]:
        return Dirichlet(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    else:
        raise ValueError(f"Unknown task name for prior: {task_name}")

def simulator_slcp3(theta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    theta = theta.to(device)

    n = theta.shape[0]

    mu0 = theta[:, 0].unsqueeze(1)
    mu1 = theta[:, 1].unsqueeze(1)
    sigma0 = theta[:, 2].unsqueeze(1)
    sigma1 = theta[:, 3].unsqueeze(1)
    r = torch.tanh(theta[:, 4]).unsqueeze(1)

    # Repeat for 4 blocks
    eps0 = torch.randn(n, 4, device=theta.device)
    eps1 = torch.randn(n, 4, device=theta.device)

    # Broadcast params
    mu0 = mu0.repeat(1, 4)
    mu1 = mu1.repeat(1, 4)
    sigma0 = sigma0.repeat(1, 4)
    sigma1 = sigma1.repeat(1, 4)
    r = r.repeat(1, 4)

    x0 = mu0 + sigma0**2 * eps0
    x1 = mu1 + sigma1**2 * (r * eps0 + torch.sqrt(1 - r ** 2) * eps1)

    out = torch.stack([x0, x1], dim=2).reshape(n, -1)
    return out.cpu()

def simulator_bernoulli(thetas, batch_size=100_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    design_matrix = torch.load("/home/hyun18/NDP/benchmark/design_matrix.pt").to(device)

    N = thetas.size(0)
    output = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        theta_batch = thetas[start:end].to(device)

        psi = torch.matmul(theta_batch, design_matrix.T)
        z = torch.sigmoid(psi)
        y = (torch.rand_like(z) < z).float()

        output_batch = torch.matmul(y, design_matrix).to("cpu")
        output.append(output_batch)
        del theta_batch, psi, z, y, output_batch
        torch.cuda.empty_cache()  # Optional: free memory aggressively

    return torch.cat(output, dim=0)

def simulator_MoG(thetas, batch_size=1_000_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = thetas.size(0)
    output = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        theta_batch = thetas[start:end].to(device)

        # MoG parameters
        scale = torch.tensor([1.0, 0.1], device=device)

        # Bernoulli mask
        idx = D.Bernoulli(torch.tensor(0.5, device=device)).sample(theta_batch.shape)
        idx2 = 1.0 - idx

        # Sample from two Gaussians
        tmp1 = D.Normal(theta_batch, scale[0]).sample()
        tmp2 = D.Normal(theta_batch, scale[1]).sample()

        # Mixture
        mixed = tmp1 * idx + tmp2 * idx2

        output.append(mixed.cpu())

        # Free memory
        del theta_batch, idx, idx2, tmp1, tmp2, mixed
        torch.cuda.empty_cache()

    return torch.cat(output, dim=0)

def simulator_Lapl_5(theta: torch.Tensor, batch_size: int = 1_000_000):
    """
    Draw one Laplace sample per element of `theta`.
    
    Parameters
    ----------
    theta : (N, 5) tensor
        Location parameter of the Laplace distribution.
    batch_size : int, optional
        Max rows to process at once to control memory (default 1e6).

    Returns
    -------
    Tensor of shape (N, 5) on CPU.
    """
    if theta.ndim != 2 or theta.size(1) != 5:
        raise ValueError("theta must have shape (N, 5)")

    # Decide where to run
    device = theta.device if theta.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    theta = theta.to(device)

    # Fixed scale vector, broadcastable to (N, 5)
    b = torch.tensor([0.05, 0.10, 0.25, 0.50, 1.00],
                     dtype=theta.dtype, device=device)

    out_chunks = []
    for start in range(0, theta.size(0), batch_size):
        end = min(start + batch_size, theta.size(0))
        loc = theta[start:end]                # already on device

        dist = D.Laplace(loc=loc, scale=b)    # broadcasts automatically
        out_chunks.append(dist.sample().cpu())  # move back to CPU

        # Help Python’s GC; no need for empty_cache()
        del loc, dist

    return torch.cat(out_chunks, dim=0)

def simulator_Lapl_10(theta: torch.Tensor, batch_size: int = 1_000_000):
    """
    Draw one Laplace sample per element of `theta`.
    
    Parameters
    ----------
    theta : (N, 5) tensor
        Location parameter of the Laplace distribution.
    batch_size : int, optional
        Max rows to process at once to control memory (default 1e6).

    Returns
    -------
    Tensor of shape (N, 5) on CPU.
    """
    if theta.ndim != 2 or theta.size(1) != 10:
        raise ValueError("theta must have shape (N, 10)")

    # Decide where to run
    device = theta.device if theta.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    theta = theta.to(device)

    # Fixed scale vector, broadcastable to (N, 5)
    b = torch.tensor([0.05, 0.10, 0.25, 0.50, 1.00, 0.05, 0.10, 0.25, 0.50, 1.00],
                     dtype=theta.dtype, device=device)

    out_chunks = []
    for start in range(0, theta.size(0), batch_size):
        end = min(start + batch_size, theta.size(0))
        loc = theta[start:end]                # already on device

        dist = D.Laplace(loc=loc, scale=b)    # broadcasts automatically
        out_chunks.append(dist.sample().cpu())  # move back to CPU

        # Help Python’s GC; no need for empty_cache()
        del loc, dist

    return torch.cat(out_chunks, dim=0)

def simulator_my_twomoons(theta):
    # Local parameters specific to this simulator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = theta.to(device)
    
    a_l = -np.pi/2
    a_u = np.pi/2
    r_mu = .1
    r_sig = .01        
    
    n = theta.shape[0]
    # Use GPU tensors for distribution parameters
    a_dist = D.Uniform(torch.tensor(a_l, device=device), torch.tensor(a_u, device=device))
    r_dist = D.Normal(torch.tensor(r_mu, device=device), torch.tensor(r_sig, device=device))

    # Sample all at once on GPU
    a = a_dist.sample((n,))
    r = r_dist.sample((n,))

    # Compute px and py
    px = r * torch.cos(a) + 0.25
    py = r * torch.sin(a)

    # Compute final x and y
    x = px - torch.abs(theta.sum(dim=1)) / np.sqrt(2)
    y = py + (theta[:, 1] - theta[:, 0]) / np.sqrt(2)

    return torch.stack([x, y], dim=1).to("cpu")
    
def simulator_OU(theta: torch.Tensor, n = 500, delta = 1/12, batch_size=1_000_000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    L_OU = theta.size(0)
    time_OU = torch.linspace(0, n * delta, n + 1)  # (n+1) time steps

    mu_OU, theta_OU, sigma2_OU = theta[:, 0], theta[:, 1], theta[:, 2]

    # Initialize an empty list to store CPU results
    path_OU_list = []

    # Process in batches to avoid memory overload
    for start in range(0, L_OU, batch_size):
        end = min(start + batch_size, L_OU)
        
        # Process batch
        mu_batch = mu_OU[start:end].to(device)
        theta_batch = theta_OU[start:end].to(device)
        sigma2_batch = sigma2_OU[start:end].to(device)

        # Compute standard deviation for initial state
        std_init = torch.sqrt(sigma2_batch / (2 * mu_batch))

        # Initialize batch paths (Allocate **directly on CPU**)
        path_batch = torch.empty((end - start, n + 1), dtype=torch.float32, device="cpu")

        # Initialize first value of the path
        z0 = torch.normal(theta_batch, std_init)
        path_batch[:, 0] = z0.cpu()  # Store on CPU
        
        del std_init  # Free GPU memory
        torch.cuda.empty_cache()

        # Compute time step difference once
        del_L = time_OU[1] - time_OU[0]
        exp_neg_mu_del = torch.exp(-mu_batch * del_L)
        sqrt_term = torch.sqrt(sigma2_batch / (2 * mu_batch) * (1 - exp_neg_mu_del**2))
        # Compute the rest of the path
        for l in range(1, n + 1):
            OU_mean = z0 * exp_neg_mu_del + theta_batch * (1 - exp_neg_mu_del)
            z0 = torch.normal(OU_mean, sqrt_term)  # Update recursively
            
            # Store result **directly** in preallocated CPU tensor
            path_batch[:, l] = z0.cpu()

        # Store batch results
        path_OU_list.append(path_batch)

        # Free GPU memory
        del mu_batch, theta_batch, sigma2_batch, exp_neg_mu_del, sqrt_term, z0, path_batch
        torch.cuda.empty_cache()
        
    # Concatenate all batches on CPU
    return torch.row_stack(path_OU_list)
    
def simulator_cont_table(theta: torch.Tensor, n = 400, batch_size = 1_000_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = theta.size(0)
    output = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        theta_batch = theta[start:end].to(device)

        output.append(Multinomial(total_count = n, probs = theta_batch).sample().cpu())
    return torch.cat(output, dim=0)

# ---- Channel builders ----
def channel_binary(p: float) -> torch.Tensor:
    """
    One-bit randomized response channel C(p):
      with prob p -> report truth; with prob (1-p) -> uniform random bit.
    Returns a 2x2 matrix with rows=reported, cols=true.
    """
    same = 0.5 * (1 + p)
    flip = 0.5 * (1 - p)
    C = torch.tensor([[same, flip],
                      [flip, same]], dtype=torch.double)
    return C

def channel_2x2(p1: float, p2: float) -> torch.Tensor:
    """
    Joint channel for two independent binary variables.
    If both use the same p, pass p1=p2=p.
    Returns a 4x4 matrix R with rows=reported cells, cols=true cells,
    under the fixed cell order [00, 01, 10, 11].
    """
    C1 = channel_binary(p1)  # for variable 1
    C2 = channel_binary(p2)  # for variable 2
    R = torch.kron(C1, C2)   # Kronecker product
    return R  # shape (4,4)

# ---- Simulator ----
def simulator_rr_cont_table(
    theta: torch.Tensor,
    p: float = 0.5,
    n: int = 400,
    batch_size: int = 1_000_000,
) -> torch.Tensor:
    """
    Simulate privatized counts y ~ Multinomial(n, q) with q = R(p) @ theta
    for a 2x2 contingency table under randomized response.

    Args:
        theta: (N, 4) tensor; each row sums to 1; order [00, 01, 10, 11].
        p: truthful-report probability for each variable (use p in [0,1]).
           If you want different ps per variable, change call to channel_2x2(p1,p2).
        n: total count per table draw (e.g., 400).
        batch_size: process rows of theta in chunks for memory efficiency.

    Returns:
        reported_counts: (N, 4) tensor of privatized counts.
    """
    assert theta.dim() == 2 and theta.size(1) == 4, "theta must be (N,4)."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = theta.to(device)
    theta = theta.to(torch.double)

    # Build joint channel R (4x4)
    R = channel_2x2(p, p).to(theta.device)  # use (p,p); change if p1!=p2

    N = theta.size(0)
    out = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        th = theta[start:end].to(device)  # (B,4)

        # Compute q = R @ th^T  -> but batched as th @ R^T  (shape (B,4))
        q = (th @ R.T).clamp(min=0)  # numerical safety
        q = q / q.sum(dim=1, keepdim=True)  # re-normalize exactly

        # Batched multinomial sampling: one draw per row
        m = Multinomial(total_count=n, probs=q)
        y = m.sample()  # (B,4)
        out.append(y.cpu())

    return torch.cat(out, dim=0).to(torch.float32)

def Simulators(task_name: str):
    task_name = task_name.lower()
    if task_name == "bernoulli_glm":
        return simulator_bernoulli
    
    elif task_name in ["slcp", "slcp3"]:
        return simulator_slcp3
    
    elif task_name in ["my_twomoons"]:
        return simulator_my_twomoons
    
    elif task_name in ["slcp_summary", "slcp2_summary"]:
        task = sbibm.get_task("slcp")
        simulator = task.get_simulator()
        def summary_generator(theta):
            x = simulator(theta)  # [N, 8]
            return SLCP_summary(x)  # [N, 5]
        return summary_generator
    
    elif task_name in ["slcp3_summary", "my_slcp", "my_slcp2", "my_slcp3", "my_slcp4"]:
        def summary_generator(theta):
            x = simulator_slcp3(theta)  # [N, 8]
            return SLCP_summary(x)  # [N, 5]
        return summary_generator
    
    elif task_name in ["slcp_summary_transform"]:
        def summary_generator(theta):
            x = simulator_slcp3(theta)  # [N, 8]
            return SLCP_summary_transform(x)  # [N, 5]
        return summary_generator
    
    elif task_name in ["slcp_summary_transform2"]:
        def summary_generator(theta):
            x = simulator_slcp3(theta)  # [N, 8]
            return SLCP_summary_transform2(x)  # [N, 5]
        return summary_generator
    
    elif task_name in ["mog_2", "mog_5", "mog_10"]:
        return simulator_MoG
    elif task_name in ["lapl_5"]:
        return simulator_Lapl_5
    elif task_name in ["lapl_10"]:
        return simulator_Lapl_10
    
    elif task_name in ["ou"]:
        def OU_generator(theta):
            return simulator_OU(theta, n = 2000, delta = 1/12)
        return OU_generator

    elif task_name in ["cont_table"]:
        return simulator_cont_table
    elif task_name in ["cont_table_dp"]:
        def cont_table_dp_generator(theta):
            return simulator_rr_cont_table(theta, p = 0.8)
        return cont_table_dp_generator
    elif task_name in ["cont_table_dp2"]:
        def cont_table_dp_generator(theta):
            return simulator_rr_cont_table(theta, p = 0.6)
        return cont_table_dp_generator
    
    elif task_name in ["cont_full"]:
        def cont_table_dp_generator(theta):
            return simulator_rr_cont_table(theta, p = 0.8, n = 4526, batch_size = 100_000)
        return cont_table_dp_generator
    
    elif task_name in ["cont_full2"]:
        def cont_table_dp_generator(theta):
            return simulator_rr_cont_table(theta, p = 0.6, n = 4526, batch_size = 100_000)
        return cont_table_dp_generator
    
    elif task_name in ["cont_full3"]:
        def cont_full3_generator(theta):
            return simulator_cont_table(theta, n = 4526, batch_size = 100_000)
        return cont_full3_generator

    elif task_name in ["cont_table_dp_transform"]:
        def cont_table_dp_generator(theta):
            return cont_table_transform(simulator_rr_cont_table(theta, p = 0.8))
        return cont_table_dp_generator
    

    else:
        raise ValueError(f"Unknown task name for simulator: {task_name}")
    

def MoG_posterior(obs, n_samples, bounds = None):
    obs = torch.tensor(obs)
    if obs.ndim == 1:
        obs = torch.reshape(obs, (1, obs.size(0)))
    scale = [1.0, 0.1]
    n_samples2 = n_samples * 1000

    idx =  D.Bernoulli(torch.tensor(1/2)).sample((n_samples2,obs.size(1) )) 
    idx2 = 1 - idx

    tmp1 = D.Normal(obs[0], torch.tensor(scale[0])).sample((n_samples2,))
    tmp2 = D.Normal(obs[0], torch.tensor(scale[1])).sample((n_samples2,))

    tmp = tmp1 * idx + tmp2 * idx2
    if bounds is not None:
        tmp = torch.clone(apply_bounds(tmp, bounds))
    sam_ind = np.random.choice(np.arange(0, tmp.size()[0]), n_samples, replace = True)
    return tmp[sam_ind,:]

def apply_bounds(samples, bounds):
        # Apply bounds to filter the samples
        if bounds is not None:
            index = []
            for j in range(samples.size()[1]):  # Iterate over each dimension
                ind = (samples[:, j] < bounds[j][1]) & (samples[:, j] > bounds[j][0])
                index.append(ind)
            index = torch.stack(index, 1)
            index = torch.all(index, 1)  # Check if all conditions hold per sample
            samples = samples[index]
        return samples
