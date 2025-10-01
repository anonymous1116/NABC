import torch

# assumes your per-sample truncated Beta(1, beta) sampler from earlier
def rtrunc_beta1b_torch(size, beta, a=0.0, b=1.0, *, device=None, dtype=torch.float32, generator=None):
    if isinstance(size, int):
        size = (size,)
    beta_t = torch.as_tensor(beta, device=device, dtype=dtype).expand(size)
    a_t = torch.as_tensor(a, device=device, dtype=dtype).expand(size)
    b_t = torch.as_tensor(b, device=device, dtype=dtype).expand(size)

    if torch.any(beta_t <= 0):
        raise ValueError("beta must be > 0.")
    if torch.any((a_t < 0) | (a_t >= b_t) | (b_t > 1)):
        raise ValueError("Require 0 <= a < b <= 1.")

    A = torch.exp(beta_t * torch.log1p(-b_t))  # (1-b)^beta
    B = torch.exp(beta_t * torch.log1p(-a_t))  # (1-a)^beta

    U = torch.rand(size, device=device, dtype=dtype, generator=generator)
    U = A + (B - A) * U
    X = 1.0 - torch.pow(U, 1.0 / beta_t)
    return X


def truncated_dirichlet1111_stick(L, lower, upper, *, device=None, dtype=torch.float32, generator=None, eps=1e-12):
    """
    Draw L samples from Dirichlet(1,1,1,1) with component-wise truncation:
        theta_i in [lower[i], upper[i]] for i=1..4.
    No rejection is used. Returns (L, 4).

    Args:
        L (int): number of samples
        lower, upper: length-4 tensors/lists (broadcastable to (L,))
                      with 0 <= lower[i] <= upper[i] <= 1 and
                      sum(lower) <= 1 <= sum(upper).
        device, dtype, generator: passed through to torch ops.
    """
    # to tensors of shape (L,)
    lower = torch.as_tensor(lower, device=device, dtype=dtype).reshape(-1)
    upper = torch.as_tensor(upper, device=device, dtype=dtype).reshape(-1)

    if lower.numel() == 4 and L != 1:
        lower = lower.expand(4).repeat(L).view(L, 4)  # (L,4)
        upper = upper.expand(4).repeat(L).view(L, 4)
    elif lower.numel() == 4 and L == 1:
        lower = lower.view(1, 4)
        upper = upper.view(1, 4)
    elif lower.numel() == 4 * L:
        lower = lower.view(L, 4)
        upper = upper.view(L, 4)
    else:
        raise ValueError("`lower` and `upper` must be length 4 (broadcast) or shape (L,4).")

    if torch.any(lower < 0) or torch.any(upper > 1) or torch.any(lower > upper):
        raise ValueError("Require 0 <= lower[i] <= upper[i] <= 1 for all i.")
    if torch.any(lower.sum(dim=1) - 1 > 1e-10):
        raise ValueError("Infeasible: sum(lower) must be <= 1.")
    if torch.any(1 - upper.sum(dim=1) > 1e-10):
        raise ValueError("Infeasible: sum(upper) must be >= 1.")

    l1,l2,l3,l4 = [lower[:,i] for i in range(4)]
    u1,u2,u3,u4 = [upper[:,i] for i in range(4)]

    # ---- Step 1: theta1 = V1, V1 ~ Beta(1,3) with feasible truncation
    # Feasibility for remaining parts:
    #   leftover = 1 - V1 must satisfy  l2+l3+l4 <= leftover <= u2+u3+u4
    a1 = torch.maximum(l1, 1 - (u2 + u3 + u4))
    b1 = torch.minimum(u1, 1 - (l2 + l3 + l4))
    if torch.any(a1 >= b1):
        raise ValueError("Infeasible bounds for theta1 given others.")

    V1 = rtrunc_beta1b_torch((L,), beta=3.0, a=a1, b=b1, device=device, dtype=dtype, generator=generator)
    theta1 = V1
    leftover1 = (1.0 - V1).clamp_min(eps)

    # ---- Step 2: theta2 = (1 - V1) * V2, V2 ~ Beta(1,2)
    # Bounds from theta2 in [l2, u2]:
    a2_from_t2 = (l2 / leftover1).clamp(0.0, 1.0)
    b2_from_t2 = (u2 / leftover1).clamp(0.0, 1.0)
    # Feasibility for remaining two parts (theta3+theta4):
    #   leftover2 = (1 - V1)(1 - V2) must be in [l3+l4, u3+u4]
    a2_from_leftover = (1.0 - (u3 + u4) / leftover1).clamp(0.0, 1.0)
    b2_from_leftover = (1.0 - (l3 + l4) / leftover1).clamp(0.0, 1.0)
    a2 = torch.maximum(a2_from_t2, a2_from_leftover)
    b2 = torch.minimum(b2_from_t2, b2_from_leftover)
    if torch.any(a2 >= b2):
        raise ValueError("Infeasible bounds for theta2 given theta1 and others.")

    V2 = rtrunc_beta1b_torch((L,), beta=2.0, a=a2, b=b2, device=device, dtype=dtype, generator=generator)
    theta2 = leftover1 * V2
    leftover2 = (leftover1 * (1.0 - V2)).clamp_min(eps)

    # ---- Step 3: theta3 = leftover2 * V3, V3 ~ Beta(1,1) (Uniform)
    # Bounds from theta3 in [l3, u3]
    a3_from_t3 = (l3 / leftover2).clamp(0.0, 1.0)
    b3_from_t3 = (u3 / leftover2).clamp(0.0, 1.0)
    # Bounds from theta4 in [l4, u4] where theta4 = leftover2 * (1 - V3)
    # => 1 - V3 in [l4/leftover2, u4/leftover2] => V3 in [1 - u4/leftover2, 1 - l4/leftover2]
    a3_from_t4 = (1.0 - (u4 / leftover2)).clamp(0.0, 1.0)
    b3_from_t4 = (1.0 - (l4 / leftover2)).clamp(0.0, 1.0)
    a3 = torch.maximum(a3_from_t3, a3_from_t4)
    b3 = torch.minimum(b3_from_t3, b3_from_t4)
    if torch.any(a3 >= b3):
        raise ValueError("Infeasible bounds for theta3/theta4 given earlier parts.")

    V3 = rtrunc_beta1b_torch((L,), beta=1.0, a=a3, b=b3, device=device, dtype=dtype, generator=generator)
    theta3 = leftover2 * V3
    theta4 = leftover2 * (1.0 - V3)

    return torch.column_stack([theta1, theta2, theta3, theta4])

def truncated_dirichlet_batch(L, lower, upper,  *, batch_size = 1_000_000, device=None, dtype=torch.float32):
    if device is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = []

    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        num = end - start
        if num ==0:
            break
        theta_batch = truncated_dirichlet1111_stick(num, lower, upper, device=device, dtype=dtype, generator=None, eps=1e-12)
        output.append(theta_batch.to("cpu"))
    return torch.cat(output)
