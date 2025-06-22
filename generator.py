import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim, img_dim):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim + label_dim, 128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),            
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)