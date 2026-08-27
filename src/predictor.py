from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/draft_model.joblib")


class DraftPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Train first: python -m src.train"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.metrics = bundle.get("metrics", {})

    @staticmethod
    def _row(radiant: list[int], dire: list[int]) -> pd.DataFrame:
        if len(radiant) != 5 or len(dire) != 5:
            raise ValueError("Exactly 5 Radiant and 5 Dire heroes are required.")
        if len(set(radiant)) != 5 or len(set(dire)) != 5:
            raise ValueError("Duplicate hero in one team.")
        if set(radiant) & set(dire):
            raise ValueError("Same hero cannot appear on both teams.")

        data = {}
        for i, h in enumerate(radiant, 1):
            data[f"radiant_{i}"] = int(h)
        for i, h in enumerate(dire, 1):
            data[f"dire_{i}"] = int(h)
        return pd.DataFrame([data])

    def predict(self, radiant: list[int], dire: list[int]) -> dict:
        df = self._row(radiant, dire)
        X = self.encoder.transform(df)
        p_radiant = float(self.model.predict_proba(X)[0, 1])

        # Linear baseline is especially easy to explain:
        # coefficient sign/magnitude gives each selected hero's contribution.
        coef = self.model.coef_[0]
        n = len(self.encoder.hero_ids)
        idx = self.encoder._index

        contributions = []
        for hero in radiant:
            i = idx.get(hero)
            contributions.append({
                "hero_id": hero,
                "side": "radiant",
                "logit_contribution": float(coef[i]) if i is not None else 0.0,
                "known_to_model": i is not None,
            })
        for hero in dire:
            i = idx.get(hero)
            contributions.append({
                "hero_id": hero,
                "side": "dire",
                "logit_contribution": float(coef[n + i]) if i is not None else 0.0,
                "known_to_model": i is not None,
            })

        contributions.sort(key=lambda x: abs(x["logit_contribution"]), reverse=True)

        return {
            "radiant_win_probability": p_radiant,
            "dire_win_probability": 1.0 - p_radiant,
            "predicted_winner": "radiant" if p_radiant >= 0.5 else "dire",
            "hero_contributions": contributions,
            "model_metrics": self.metrics,
        }
