import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """1D convolution with causal padding (only looks at past)"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        # remove future padding
        return x[:, :, :-self.padding] if self.padding != 0 else x

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # residual
        if self.downsample:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)

class DiscreteDiffusionTCN(nn.Module):
    def __init__(self, vocab_size=30522, embed_size=256, hidden_size=256, 
                 num_layers=4, num_steps=50, mask_token_id=103, kernel_size=3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.mask_token_id = mask_token_id
        
        # embedding
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # time embedding
        self.time_embed = nn.Linear(num_steps, hidden_size)
        
        # tcn layers with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        dilations = [2**i for i in range(num_layers)]
        
        for i, dilation in enumerate(dilations):
            in_ch = embed_size if i == 0 else hidden_size
            self.tcn_blocks.append(
                TCNBlock(in_ch, hidden_size, kernel_size, dilation, dropout=0.1)
            )
        
        # output
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # noise schedule
        self.noise_schedule = torch.linspace(0.0, 0.95, num_steps)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"model params: {total_params:,}")
        print(f"diffusion steps: {num_steps}")
        print(f"architecture: TCN (Temporal Convolution)")
        print(f"receptive field: {sum(2**i * (kernel_size-1) for i in range(num_layers))}")
    
    def forward_diffusion(self, x, t):
        """gradually mask tokens"""
        x_t = x.clone()
        batch_size = x.size(0)
        
        for b in range(batch_size):
            timestep = t[b].item()
            noise_level = self.noise_schedule[timestep]
            
            mask = torch.rand(x.size(1), device=x.device) < noise_level
            mask = mask & (x[b] != 0)
            x_t[b][mask] = self.mask_token_id
        
        return x_t
    
    def forward(self, x, attention_mask, timestep):
        """
        predict clean tokens
        x: [batch, seq_len]
        timestep: [batch]
        """
        batch_size, seq_len = x.shape
        
        # embed
        embedded = self.embedding(x)  # [batch, seq_len, embed]
        
        # transpose for conv1d: [batch, channels, length]
        embedded = embedded.transpose(1, 2)
        
        # tcn layers
        out = embedded
        for block in self.tcn_blocks:
            out = block(out)
        
        # transpose back: [batch, seq_len, hidden]
        out = out.transpose(1, 2)
        
        # add time embedding
        time_emb = self.time_embed(
            F.one_hot(timestep, self.num_steps).float()
        )  # [batch, hidden]
        time_emb = time_emb.unsqueeze(1)  # [batch, 1, hidden]
        out = out + time_emb
        
        # predict
        logits = self.output_layer(out)  # [batch, seq_len, vocab]
        
        return logits
    
    def compute_loss(self, logits, labels, mask_positions):
        batch_size, seq_len, vocab_size = logits.shape
        
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        loss = loss.view(batch_size, seq_len)
        
        masked_loss = (loss * mask_positions.float()).sum()
        num_masked = mask_positions.sum()
        
        return masked_loss / num_masked if num_masked > 0 else masked_loss
    
    @torch.no_grad()
    def sample(self, shape, attention_mask, device):
        """iterative denoising"""
        batch_size, seq_len = shape
        
        x = torch.full((batch_size, seq_len), self.mask_token_id, 
                       dtype=torch.long, device=device)
        
        for t in reversed(range(self.num_steps)):
            timestep = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            logits = self.forward(x, attention_mask, timestep)
            probs = F.softmax(logits, dim=-1)
            
            mask_positions = (x == self.mask_token_id)
            
            if t > 0:
                for b in range(batch_size):
                    masked_indices = mask_positions[b].nonzero(as_tuple=True)[0]
                    if len(masked_indices) > 0:
                        sampled = torch.multinomial(
                            probs[b, masked_indices], 
                            num_samples=1
                        ).squeeze(-1)
                        
                        num_to_unmask = max(1, int(len(masked_indices) * 0.1))
                        unmask_indices = masked_indices[:num_to_unmask]
                        x[b, unmask_indices] = sampled[:num_to_unmask]
            else:
                sampled = torch.multinomial(
                    probs[mask_positions].view(-1, self.vocab_size),
                    num_samples=1
                ).squeeze(-1)
                x[mask_positions] = sampled
        
        return x
    
    def training_step(self, batch):
        clean_tokens = batch['labels']
        attention_mask = batch['attention_mask']
        mask_positions = batch['mask_positions']
        
        batch_size = clean_tokens.size(0)
        
        t = torch.randint(0, self.num_steps, (batch_size,), device=clean_tokens.device)
        noisy_tokens = self.forward_diffusion(clean_tokens, t)
        logits = self.forward(noisy_tokens, attention_mask, t)
        loss = self.compute_loss(logits, clean_tokens, mask_positions)
        
        predictions = logits.argmax(dim=-1)
        correct = (predictions == clean_tokens) & mask_positions
        accuracy = correct.sum().float() / mask_positions.sum().float()
        
        return loss, accuracy


if __name__ == "__main__":
    print("testing tcn diffusion model\n")
    
    model = DiscreteDiffusionTCN(
        vocab_size=30522,
        embed_size=256,
        hidden_size=256,
        num_layers=4,
        num_steps=50
    )
    
    batch_size, seq_len = 4, 32
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'labels': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'mask_positions': torch.rand(batch_size, seq_len) > 0.85
    }
    
    print(f"input shape: {batch['input_ids'].shape}")
    print(f"masked tokens: {batch['mask_positions'].sum().item()}\n")
    
    loss, accuracy = model.training_step(batch)
    print(f"loss: {loss.item():.4f}")
    print(f"accuracy: {accuracy.item():.4f}")
    
    device = torch.device('cpu')
    samples = model.sample((2, 32), torch.ones(2, 32), device)
    print(f"\nsampled shape: {samples.shape}")
    print("tcn test passed!")