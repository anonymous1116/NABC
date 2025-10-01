
import torch
class ScaleLogitTransform:
    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        """
        Initialize with lower (a) and upper (b) bounds.
        Supports vector bounds for multi-dimensional parameters.
        """
        self.a = a
        self.b = b

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Map from [a, b] to ℝ using scale + logit:
        z = log((θ - a) / (b - θ))
        """
        eps = 1e-12  # to prevent division by 0
        theta = torch.clamp(theta, self.a + eps, self.b - eps)
        scaled = (theta - self.a) / (self.b - self.a)
        return torch.log(scaled / (1 - scaled))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map from ℝ back to [a, b] using sigmoid + scaling:
        θ = (b - a) * sigmoid(z) + a
        """
        sigmoid = torch.sigmoid(z)
        return self.a + (self.b - self.a) * sigmoid