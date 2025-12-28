import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from data import TimeSeriesDataset, load_prices_from_csv
from scheduler import DiffusionScheduler
from unet import UNet


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEQ_LEN = 128
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20
T = 1000 #diffusion steps

# Training Loop
def train():
    prices = load_prices_from_csv("/data/^GSPC.csv", price_column="Close")
    dataset = TimeSeriesDataset(prices, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(seq_len=SEQ_LEN).to(DEVICE)
    scheduler = DiffusionScheduler(T=T, device=DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0

        for x0, _ in pbar:
            x0 = x0.to(DEVICE)

            B = x0.shape[0]
            t = torch.randint(0, T, (B,), device=DEVICE)

            xt, noise = scheduler.add_noise(x0, t)

            pred_noise = model(xt, t.float())
            
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "global_mean": dataset.global_mean,
            "global_std": dataset.global_std,
        },
        "diffusion_unconditional.pt"
    )

    print("Training complete. Model saved as diffusion_unconditional.pt")


if __name__ == "__main__":
    train()