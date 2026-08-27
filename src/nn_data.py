from __future__ import annotations

import numpy as np
import pandas as pd

RADIANT_COLS = [f"radiant_{i}" for i in range(1, 6)]
DIRE_COLS = [f"dire_{i}" for i in range(1, 6)]


class HeroIndexer:
    """Maps Valve/OpenDota hero IDs to compact contiguous indices.

    Index 0 is reserved for unknown/padding.
    """

    def __init__(self):
        self.hero_to_idx: dict[int, int] = {}
        self.idx_to_hero: dict[int, int] = {}

    def fit(self, df: pd.DataFrame) -> "HeroIndexer":
        heroes = set()
        for col in RADIANT_COLS + DIRE_COLS:
            heroes.update(int(x) for x in df[col].dropna().astype(int).tolist())

        for idx, hero_id in enumerate(sorted(heroes), start=1):
            self.hero_to_idx[hero_id] = idx
            self.idx_to_hero[idx] = hero_id
        return self

    @property
    def n_heroes(self) -> int:
        return len(self.hero_to_idx)

    def transform_teams(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in df.iterrows():
            heroes = []
            for col in RADIANT_COLS + DIRE_COLS:
                hero_id = int(row[col])
                heroes.append(self.hero_to_idx.get(hero_id, 0))
            rows.append(heroes)
        return np.asarray(rows, dtype=np.int64)

    def encode_draft(self, radiant: list[int], dire: list[int]) -> np.ndarray:
        if len(radiant) != 5 or len(dire) != 5:
            raise ValueError("Need exactly 5 Radiant and 5 Dire heroes.")
        vals = [self.hero_to_idx.get(int(h), 0) for h in radiant + dire]
        return np.asarray(vals, dtype=np.int64)
