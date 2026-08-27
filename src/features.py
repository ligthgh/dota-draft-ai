from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


RADIANT_COLS = [f"radiant_{i}" for i in range(1, 6)]
DIRE_COLS = [f"dire_{i}" for i in range(1, 6)]


class DraftEncoder:
    """Encodes each hero separately for Radiant and Dire.

    This preserves side information:
      radiant_AntiMage != dire_AntiMage feature.
    """

    def __init__(self):
        self.hero_ids: list[int] = []
        self._index: dict[int, int] = {}

    def fit(self, df: pd.DataFrame) -> "DraftEncoder":
        heroes = set()
        for col in RADIANT_COLS + DIRE_COLS:
            heroes.update(int(x) for x in df[col].dropna().astype(int).tolist())
        self.hero_ids = sorted(heroes)
        self._index = {hero_id: i for i, hero_id in enumerate(self.hero_ids)}
        return self

    @property
    def n_features(self) -> int:
        return len(self.hero_ids) * 2

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._index:
            raise RuntimeError("DraftEncoder must be fitted first.")

        X = np.zeros((len(df), self.n_features), dtype=np.float32)
        offset = len(self.hero_ids)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col in RADIANT_COLS:
                hero = int(row[col])
                idx = self._index.get(hero)
                if idx is not None:
                    X[row_idx, idx] = 1.0

            for col in DIRE_COLS:
                hero = int(row[col])
                idx = self._index.get(hero)
                if idx is not None:
                    X[row_idx, offset + idx] = 1.0

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def feature_names(self) -> list[str]:
        radiant = [f"radiant_hero_{h}" for h in self.hero_ids]
        dire = [f"dire_hero_{h}" for h in self.hero_ids]
        return radiant + dire
