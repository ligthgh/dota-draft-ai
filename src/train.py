from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .features import DraftEncoder, RADIANT_COLS, DIRE_COLS


DATA = Path("data/matches.csv")
MODEL = Path("models/draft_model.joblib")
METRICS = Path("models/metrics.json")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "start_time", "radiant_win", *RADIANT_COLS, *DIRE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    for col in RADIANT_COLS + DIRE_COLS:
        df[col] = df[col].astype(int)

    # Remove malformed drafts.
    mask = []
    for _, row in df.iterrows():
        r = [int(row[c]) for c in RADIANT_COLS]
        d = [int(row[c]) for c in DIRE_COLS]
        mask.append(len(set(r)) == 5 and len(set(d)) == 5 and set(r).isdisjoint(d))
    return df[np.array(mask)].sort_values(["start_time", "match_id"]).reset_index(drop=True)


def main():
    if not DATA.exists():
        raise SystemExit("Missing data/matches.csv. Run: python -m src.collect --pages 20")

    df = validate(pd.read_csv(DATA))
    if len(df) < 500:
        print("WARNING: fewer than 500 matches; metrics will be noisy.")

    split = max(1, int(len(df) * 0.80))
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    encoder = DraftEncoder()
    X_train = encoder.fit_transform(train_df)
    X_test = encoder.transform(test_df)
    y_train = train_df["radiant_win"].astype(int).to_numpy()
    y_test = test_df["radiant_win"].astype(int).to_numpy()

    model = LogisticRegression(
        max_iter=2000,
        C=0.25,
        solver="liblinear",
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    metrics = {
        "n_matches": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_heroes_seen": int(len(encoder.hero_ids)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, p)) if len(set(y_test)) > 1 else None,
        "log_loss": float(log_loss(y_test, p, labels=[0, 1])),
        "test_radiant_winrate": float(y_test.mean()),
    }

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "encoder": encoder,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {MODEL}")


if __name__ == "__main__":
    main()
