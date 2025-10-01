
import torch
import numpy as np
import torch.distributions as D
import matplotlib.pyplot as plt
from torch.distributions import Normal, Laplace
from sbi.utils import BoxUniform

def Lap_mec(mu, sigma0, b):
    """
    Sample s_dp for a batch of μ values.

    Args:
        mu: Tensor of shape (N,) representing μ for each sample
        sigma0: float, std dev of Normal prior for x
        b: float, scale of Laplace noise

    Returns:
        Tensor of s_dp samples of shape (N,)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu = mu.to(device)
    sigma0 = torch.tensor(sigma0, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device)

    # Step 1: x ~ N(mu, sigma0)
    x = Normal(mu, sigma0).sample()  # shape: (N,)

    # Step 2: s_dp ~ Laplace(x, b)
    s_dp = Laplace(x, b).sample()    # shape: (N,)
    return s_dp.cpu()

def MoG(theta):
    scale = [1.0, 0.1]
    idx =  D.Bernoulli(torch.tensor(1/2)).sample(theta.size()) # p = 1/3
    idx2 = 1 - idx

    tmp1 = D.Normal(theta, torch.tensor(scale[0])).sample()
    tmp2 = D.Normal(theta, torch.tensor(scale[1])).sample()
    return tmp1 * idx + tmp2 * idx2

def ABC_rej(x0, X_cal, tol, device):
    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2, 1))

    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(dist, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (dist <= ds)
    
    # Select points within tolerance and return to CPU if needed
    return wt1.cpu()

def compute_mad(X):
    # Move the tensor to GPU if available
    if torch.cuda.is_available():
        X = X.to('cuda')

    # Compute the median for each column
    medians = torch.median(X, dim=0).values  # Shape: (num_columns,)

    # Compute the absolute deviations from the median
    abs_deviation = torch.abs(X - medians)  # Broadcasting over rows

    # Compute the MAD for each column
    mad = torch.median(abs_deviation, dim=0).values  # Shape: (num_columns,)
    torch.cuda.empty_cache()
    
    # Return the result on the CPU
    return mad.cpu()

def SLCP_summary(X):
    """
    Compute summary statistics for SLCP data:
    - Means and standard deviations for even and odd indexed dimensions
    - Average correlation between even and odd groups

    Args:
        X: Tensor of shape [N, 8]

    Returns:
        Tensor of shape [N, 5] containing m0, m1, s0, s1, and rho
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    X0 = X[:, [0, 2, 4, 6]]
    X1 = X[:, [1, 3, 5, 7]]
    m0 = X0.mean(dim=1, keepdim=True)
    m1 = X1.mean(dim=1, keepdim=True)
    s0 = X0.std(dim=1, correction=0, keepdim=True)
    s1 = X1.std(dim=1, correction=0, keepdim=True)

    # Compute correlation per sample
    cov = ((X0 - m0) * (X1 - m1)).mean(dim=1, keepdim=True)
    rho = cov / (s0 * s1 + 1e-12)
    rho = torch.clamp(rho, -1.0, 1.0)

    return torch.cat((m0, m1, s0, s1, rho), dim=1).cpu()



def fisher_z(x, eps=1e-6):
    x = torch.clamp(x, -1+eps, 1-eps)        # or: x = x*(1-eps)
    z = 0.5 * torch.log((1 + x) / (1 - x))   
    return z

def log1p(x):
    return torch.log(1+x)

def log1p2(x):
    return torch.log(.1+x)

def SLCP_summary_transform2(X):
    """
    Compute summary statistics for SLCP data:
    - Means and standard deviations for even and odd indexed dimensions
    - Average correlation between even and odd groups

    Args:
        X: Tensor of shape [N, 8]

    Returns:
        Tensor of shape [N, 5] containing m0, m1, s0, s1, and rho
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    X0 = X[:, [0, 2, 4, 6]]
    X1 = X[:, [1, 3, 5, 7]]
    m0 = X0.mean(dim=1, keepdim=True)
    m1 = X1.mean(dim=1, keepdim=True)
    s0 = X0.std(dim=1, correction=0, keepdim=True)
    s1 = X1.std(dim=1, correction=0, keepdim=True)
    
    # Compute correlation per sample
    cov = ((X0 - m0) * (X1 - m1)).mean(dim=1, keepdim=True)
    rho = cov / (s0 * s1 + 1e-12)
    rho = torch.clamp(rho, -1.0, 1.0)

    s0 = log1p2(s0)
    s1 = log1p2(s1)
    rho = fisher_z(rho)

    return torch.cat((m0, m1, s0, s1, rho), dim=1).cpu()

def cont_table_transform(X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp = torch.log(1e-6+X.to(device))
    return tmp.cpu()


def ABC_rej2(x0, X_cal, tol, device, case = None):
    # Move all tensors to the target device at once
    if case in ["slcp", "slcp3"]:
        x0 = SLCP_summary_transform2(x0)
        X_cal = SLCP_summary_transform2(X_cal)

    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    mad = compute_mad(X_cal)
    mad = torch.reshape(mad, (1, X_cal.size(1))).to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2/mad**2, 1))
    
    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(dist, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (dist <= ds)
    torch.cuda.empty_cache()
    del mad, dist
    # Select points within tolerance and return to CPU if needed
    return wt1.cpu()




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

