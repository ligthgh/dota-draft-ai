from __future__ import annotations

import numpy as np
import pandas as pd

HERO_COLS = [f"radiant_hero_{i}" for i in range(1, 6)] + [f"dire_hero_{i}" for i in range(1, 6)]
POS_COLS = [f"radiant_pos_{i}" for i in range(1, 6)] + [f"dire_pos_{i}" for i in range(1, 6)]
PICK_COLS = [f"radiant_pick_{i}" for i in range(1, 6)] + [f"dire_pick_{i}" for i in range(1, 6)]


class HeroIndexer:
    def __init__(self):
        self.hero_to_idx = {}
        self.idx_to_hero = {}

    def fit(self, df: pd.DataFrame):
        heroes = set()
        for col in HERO_COLS:
            heroes.update(int(x) for x in df[col].dropna().astype(int).tolist())

        for idx, hero_id in enumerate(sorted(heroes), start=1):
            self.hero_to_idx[hero_id] = idx
            self.idx_to_hero[idx] = hero_id
        return self

    @property
    def n_heroes(self):
        return len(self.hero_to_idx)

    def transform(self, df: pd.DataFrame):
        heroes = np.asarray([
            [self.hero_to_idx.get(int(row[c]), 0) for c in HERO_COLS]
            for _, row in df.iterrows()
        ], dtype=np.int64)

        positions = np.asarray([
            [max(0, min(5, int(row[c]))) for c in POS_COLS]
            for _, row in df.iterrows()
        ], dtype=np.int64)

        picks = np.asarray([
            [max(0, min(10, int(row[c]))) for c in PICK_COLS]
            for _, row in df.iterrows()
        ], dtype=np.int64)

        mmr = df["avg_mmr"].fillna(0).astype(float).to_numpy(np.float32)
        rank = df["rank_bracket"].fillna(0).astype(int).to_numpy(np.int64)

        return heroes, positions, picks, mmr, rank

    def encode_draft(self, heroes, positions, picks, avg_mmr, rank_bracket):
        return (
            np.asarray([self.hero_to_idx.get(int(h), 0) for h in heroes], dtype=np.int64),
            np.asarray(positions, dtype=np.int64),
            np.asarray(picks, dtype=np.int64),
            np.asarray([float(avg_mmr)], dtype=np.float32),
            np.asarray([int(rank_bracket)], dtype=np.int64),
        )
