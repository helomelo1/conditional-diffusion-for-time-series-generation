import torch
import matplotlib.pyplot as plt

from scheduler import DiffusionScheduler
from unet import UNet


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEQ_LEN = 128
T = 1000


@torch.no_grad()
def sample(model, scheduler, n_samples=4):
    model.eval()

    x = torch.randn(n_samples, SEQ_LEN).to(DEVICE)

    for t in reversed(range(T)):
        t_tensor = torch.full((n_samples, ), t, device=DEVICE)

        pred_noise = model(x, t_tensor.float())

        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_bar[t]
        beta = scheduler.betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise

    return x


def main():
    checkpoint = torch.load("diffusion_unconditional.pt", map_location=DEVICE, weights_only=False)

    model = UNet(seq_len=SEQ_LEN).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    scheduler = DiffusionScheduler(T=T, device=DEVICE)

    samples = sample(model, scheduler=scheduler, n_samples=4)
    samples = samples.cpu().numpy()

    plt.figure(figsize=(10, 6))

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(samples[i])
        plt.title(f"Generated Sample {i}")

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()