import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from data import get_dataloaders
from model import SimpleInfillingModel


def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(train_loader, desc = f"epoch {epoch}")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        loss, accuracy = model.training_step(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy.item():.3f}'})

    return total_loss / len(train_loader), total_acc / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch in tqdm(val_loader, desc='validating'):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        loss, accuracy = model.training_step(batch)
        total_loss += loss.item()
        total_acc += accuracy.item()
    
    return total_loss / len(val_loader), total_acc / len(val_loader)


def main():
    BATCH_SIZE = 32
    MAX_LENGTH = 128
    HIDDEN_SIZE = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    LR = 3e-4
    NUM_EPOCHS = 10
    SAVE_DIR = "checkpoints"

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print("Loading Data...")

    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )

    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}\n")
    
    print("Initializing Model...")

    model = SimpleInfillingModel(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS
    ).to(DEVICE)
    print()

    optimizer = optim.AdamW(model.parameters, lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)

    Path(SAVE_DIR).mkdir(exist_ok=True)

    best_val_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, DEVICE)

        print(f"\nepoch {epoch}/{NUM_EPOCHS}")
        print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.3f}")
        print(f"  val loss: {val_loss:.4f} | val acc: {val_acc:.3f}")
        print(f"  lr: {scheduler.get_last_lr()[0]:.6f}")
        
        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, f"{SAVE_DIR}/best_model.pt")
            print(f"  saved best model (val_loss: {val_loss:.4f})")
        
        print()
    
    print("training complete!")
    print(f"best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()