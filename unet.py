import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.mlp(x.unsqueeze(-1))
    

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    

class UNet(nn.Module):
    def __init__(self, seq_len=128, base_ch=64):
        super().__init__()

        self.time_emb = TimeEmbedding()

        self.down1 = ConvBlock(1, base_ch)
        self.down2 = ConvBlock(base_ch, base_ch * 2)

        self.pool = nn.AvgPool1d()

        self.mid = ConvBlock(base_ch * 2, base_ch * 2)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ConvBlock(base_ch * 2 + base_ch, base_ch)

        self.out = ConvBlock(base_ch, 1, 1)

    def forward(self, x, t):
        x = x.unsqueeze(1)

        t_emb = self.time_emb(t).unsqueeze(-1)

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        m = self.mid(d2)

        u = self.up(m)
        u = torch.cat([u, d1], dim=1)
        u = self.up1(u)

        return self.out(u).squeeze(1)