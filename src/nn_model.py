from __future__ import annotations

import torch
from torch import nn


class DraftNet(nn.Module):
    """Draft-only neural network with hero embeddings.

    Each side is encoded separately. Pairwise interaction features are added
    by element-wise multiplication between every Radiant and Dire hero embedding.
    """

    def __init__(self, n_heroes: int, embedding_dim: int = 32):
        super().__init__()
        self.n_heroes = n_heroes
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=n_heroes + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # 10 hero embeddings + 25 cross-team pair interaction vectors.
        input_dim = (10 + 25) * embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, hero_indices: torch.Tensor) -> torch.Tensor:
        # hero_indices: [batch, 10]
        emb = self.embedding(hero_indices)  # [B, 10, D]

        radiant = emb[:, :5, :]
        dire = emb[:, 5:, :]

        base = emb.flatten(start_dim=1)

        pairwise = []
        for i in range(5):
            for j in range(5):
                pairwise.append(radiant[:, i, :] * dire[:, j, :])

        pairwise = torch.cat(pairwise, dim=1)
        x = torch.cat([base, pairwise], dim=1)

        return self.mlp(x).squeeze(1)
