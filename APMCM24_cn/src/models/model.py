import torch
import torch.nn as nn

__all__ = ['AttnRegressor']

class AttnRegressor(nn.Module):
    def __init__(self, nheads):
        super(AttnRegressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=nheads, 
            bias=True, dropout=0.1
        )   
        self.decoder = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        attended, _ = self.attn(encoded, encoded, encoded)
        o = self.decoder(attended)
        return o