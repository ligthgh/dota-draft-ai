from __future__ import annotations

import torch
from torch import nn


class DraftNetV7(nn.Module):
    def __init__(self, n_heroes: int, hero_dim=16, pos_dim=4, pick_dim=4, rank_dim=4):
        super().__init__()

        self.hero_embedding = nn.Embedding(n_heroes + 1, hero_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(6, pos_dim, padding_idx=0)
        self.pick_embedding = nn.Embedding(11, pick_dim, padding_idx=0)
        self.rank_embedding = nn.Embedding(9, rank_dim, padding_idx=0)

        slot_dim = hero_dim + pos_dim + pick_dim
        input_dim = 10 * slot_dim + 25 * hero_dim + 1 + rank_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, 1),
        )

    def forward(self, heroes, positions, picks, avg_mmr, rank_bracket):
        h = self.hero_embedding(heroes)
        p = self.position_embedding(positions)
        k = self.pick_embedding(picks)

        base = torch.cat([h, p, k], dim=-1).flatten(start_dim=1)

        radiant = h[:, :5, :]
        dire = h[:, 5:, :]

        interactions = []
        for i in range(5):
            for j in range(5):
                interactions.append(radiant[:, i, :] * dire[:, j, :])

        interactions = torch.cat(interactions, dim=1)
        mmr = (avg_mmr.float().unsqueeze(1) / 10000.0).clamp(0.0, 1.5)
        rank = self.rank_embedding(rank_bracket)

        x = torch.cat([base, interactions, mmr, rank], dim=1)
        return self.mlp(x).squeeze(1)
