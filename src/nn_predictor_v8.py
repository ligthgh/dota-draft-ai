from __future__ import annotations

from pathlib import Path

import torch

from .nn_data_v7 import HeroIndexer
from .nn_model_v7 import DraftNetV7

MODEL_PATH = Path("models/draft_nn_v8.pt")
META_PATH = Path("models/draft_nn_v8_meta.pt")


class NeuralDraftPredictorV8:
    def __init__(self):
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "V8 model missing. Run: python -m src.train_nn_v8"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )
        meta = torch.load(
            META_PATH,
            map_location="cpu",
            weights_only=False,
        )

        self.indexer = HeroIndexer()
        self.indexer.hero_to_idx = {
            int(k): int(v)
            for k, v in meta["hero_to_idx"].items()
        }
        self.indexer.idx_to_hero = {
            int(k): int(v)
            for k, v in meta["idx_to_hero"].items()
        }

        self.model = DraftNetV7(
            n_heroes=int(checkpoint["n_heroes"])
        )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

    def probability(
        self,
        radiant_heroes,
        dire_heroes,
        radiant_positions,
        dire_positions,
        radiant_picks,
        dire_picks,
        avg_mmr,
        rank_bracket,
    ):
        all_heroes = radiant_heroes + dire_heroes
        all_positions = radiant_positions + dire_positions
        all_picks = radiant_picks + dire_picks

        encoded_heroes = [
            self.indexer.hero_to_idx.get(int(h), 0)
            if int(h) != 0 else 0
            for h in all_heroes
        ]

        heroes = torch.tensor(
            [encoded_heroes],
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            [all_positions],
            dtype=torch.long,
            device=self.device,
        )
        picks = torch.tensor(
            [all_picks],
            dtype=torch.long,
            device=self.device,
        )
        mmr = torch.tensor(
            [float(avg_mmr)],
            dtype=torch.float32,
            device=self.device,
        )
        rank = torch.tensor(
            [int(rank_bracket)],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            p = torch.sigmoid(
                self.model(
                    heroes,
                    positions,
                    picks,
                    mmr,
                    rank,
                )
            ).item()

        return float(p)
