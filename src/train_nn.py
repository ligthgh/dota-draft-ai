from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .nn_data import HeroIndexer, RADIANT_COLS, DIRE_COLS
from .nn_model import DraftNet


DATA = Path("data/matches.csv")
WINDOW_META = Path("data/recent_window.json")
OUT = Path("models/draft_nn.pt")
META = Path("models/draft_nn_meta.pt")
METRICS = Path("models/nn_metrics.json")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "start_time",
        "radiant_win",
        *RADIANT_COLS,
        *DIRE_COLS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()

    good = []

    for _, row in df.iterrows():
        radiant = [int(row[c]) for c in RADIANT_COLS]
        dire = [int(row[c]) for c in DIRE_COLS]

        good.append(
            len(set(radiant)) == 5
            and len(set(dire)) == 5
            and set(radiant).isdisjoint(dire)
        )

    return (
        df[np.asarray(good)]
        .sort_values(["start_time", "match_id"])
        .reset_index(drop=True)
    )


def load_cutoff() -> tuple[int | None, int | None]:
    if not WINDOW_META.exists():
        return None, None

    try:
        data = json.loads(
            WINDOW_META.read_text(encoding="utf-8")
        )
        return (
            int(data["cutoff_time"]),
            int(data.get("days", 30)),
        )
    except Exception:
        return None, None


def batch_metrics(model, loader, device):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()
            ps.extend(p.tolist())
            ys.extend(y.numpy().tolist())

    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(ps, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if len(set(y.tolist())) > 1
            else None
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
    }


def main():
    if not DATA.exists():
        raise SystemExit(
            "Missing data/matches.csv. "
            "Run: python -m src.collect --pages 300 --sleep 2"
        )

    df = validate(pd.read_csv(DATA))

    cutoff_time, days = load_cutoff()

    if cutoff_time is not None:
        df = df[
            df["start_time"] >= cutoff_time
        ].reset_index(drop=True)

    if df.empty:
        raise SystemExit(
            "No recent matches available for training. "
            "Run src.collect again."
        )

    if days:
        print(
            f"Training on the recent {days}-day window only: "
            f"{len(df):,} matches"
        )
    else:
        print(
            f"Training on matches.csv: {len(df):,} matches"
        )

    if len(df) < 10000:
        print(
            "WARNING: very little data for a neural network. "
            "For a stronger model, widen the recent window "
            "(for example --hours 72) and collect more pages."
        )

    split = int(len(df) * 0.80)

    if split < 1 or len(df) - split < 1:
        raise SystemExit(
            "Not enough matches for train/test split."
        )

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    indexer = HeroIndexer().fit(train_df)

    X_train = indexer.transform_teams(train_df)
    X_test = indexer.transform_teams(test_df)

    y_train = (
        train_df["radiant_win"]
        .astype(np.float32)
        .to_numpy()
    )
    y_test = (
        test_df["radiant_win"]
        .astype(np.float32)
        .to_numpy()
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1024,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=2048,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = DraftNet(
        n_heroes=indexer.n_heroes,
        embedding_dim=32,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_state = None
    patience = 3
    stale = 0

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(x)
            total_n += len(x)

        train_loss = total_loss / max(total_n, 1)

        metrics = batch_metrics(
            model,
            test_loader,
            device,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} "
            f"test_logloss={metrics['log_loss']:.5f} "
            f"auc={metrics['roc_auc']} "
            f"acc={metrics['accuracy']:.4f}"
        )

        if metrics["log_loss"] < best_loss:
            best_loss = metrics["log_loss"]

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            stale = 0
        else:
            stale += 1

            if stale >= patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a model."
        )

    model.load_state_dict(best_state)

    final_metrics = batch_metrics(
        model,
        test_loader,
        device,
    )

    final_metrics.update(
        {
            "n_matches": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_heroes": int(indexer.n_heroes),
            "device": str(device),
            "recent_days": days,
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_heroes": indexer.n_heroes,
            "embedding_dim": 32,
        },
        OUT,
    )

    torch.save(
        {
            "hero_to_idx": indexer.hero_to_idx,
            "idx_to_hero": indexer.idx_to_hero,
        },
        META,
    )

    METRICS.write_text(
        json.dumps(
            final_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFinal metrics:")
    print(
        json.dumps(
            final_metrics,
            indent=2,
        )
    )
    print(f"\nSaved model -> {OUT}")


if __name__ == "__main__":
    main()
