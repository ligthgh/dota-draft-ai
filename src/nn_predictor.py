from __future__ import annotations

from pathlib import Path

import torch

from .nn_data import HeroIndexer
from .nn_model import DraftNet


MODEL_PATH = Path("models/draft_nn.pt")
META_PATH = Path("models/draft_nn_meta.pt")


class NeuralDraftPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        meta_path: Path = META_PATH,
    ):
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "Neural model missing. Run: python -m src.train_nn"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        meta = torch.load(
            meta_path,
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

        self.model = DraftNet(
            n_heroes=int(checkpoint["n_heroes"]),
            embedding_dim=int(checkpoint["embedding_dim"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def validate(radiant: list[int], dire: list[int]):
        if len(radiant) != 5 or len(dire) != 5:
            raise ValueError("Need exactly 5 heroes on each team.")
        all_heroes = radiant + dire
        if len(set(all_heroes)) != 10:
            raise ValueError("Draft contains duplicate heroes.")

    def radiant_win_probability(
        self,
        radiant: list[int],
        dire: list[int],
    ) -> float:
        self.validate(radiant, dire)

        arr = self.indexer.encode_draft(radiant, dire)
        x = torch.tensor(
            arr[None, :],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            p = torch.sigmoid(self.model(x)).item()

        return float(p)
