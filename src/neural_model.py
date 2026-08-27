"""Optional second-stage model.

Not wired into training by default. This is the next model to benchmark
against the linear baseline after enough data is collected.
"""

from __future__ import annotations

# Install torch separately when you are ready:
# pip install torch

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


if torch is not None:
    class DraftEmbeddingNet(nn.Module):
        def __init__(self, n_heroes: int, embedding_dim: int = 32):
            super().__init__()
            # +1 allows index 0 to be reserved for unknown hero.
            self.embedding = nn.Embedding(n_heroes + 1, embedding_dim, padding_idx=0)
            self.head = nn.Sequential(
                nn.Linear(10 * embedding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 1),
            )

        def forward(self, heroes):
            # heroes shape: [batch, 10]
            x = self.embedding(heroes)
            # Give Dire embeddings an opposite sign to expose side structure.
            side = torch.ones_like(x)
            side[:, 5:, :] = -1
            x = x * side
            return self.head(x.flatten(1)).squeeze(1)
