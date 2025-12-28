import torch
import math


def cosine_beta_scheduler(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)

    alphas_cumprod = torch.cos(
        ((x / T) + s) / (1 + s) * math.pi / 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionScheduler:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device

        self.betas = cosine_beta_scheduler(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.rand_like(x0)

        sqrt_ab = torch.sqrt(self.alpha_bar[t])[:, None]
        sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bar[t])[:, None]

        return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise