from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .nn_data_v7 import HeroIndexer, HERO_COLS, POS_COLS, PICK_COLS
from .nn_model_v7 import DraftNetV7

DATA = Path("data/detailed_matches.csv")
OUT = Path("models/draft_nn_v7.pt")
META = Path("models/draft_nn_v7_meta.pt")
METRICS = Path("models/nn_v7_metrics.json")


def make_loader(indexer, df, batch_size, shuffle):
    heroes, positions, picks, mmr, rank = indexer.transform(df)
    y = df["radiant_win"].astype(np.float32).to_numpy()

    ds = TensorDataset(
        torch.tensor(heroes, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
        torch.tensor(picks, dtype=torch.long),
        torch.tensor(mmr, dtype=torch.float32),
        torch.tensor(rank, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for heroes, pos, picks, mmr, rank, y in loader:
            logits = model(
                heroes.to(device),
                pos.to(device),
                picks.to(device),
                mmr.to(device),
                rank.to(device),
            )
            p = torch.sigmoid(logits).cpu().numpy()
            ps.extend(p.tolist())
            ys.extend(y.numpy().tolist())

    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(ps, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def main():
    if not DATA.exists():
        raise SystemExit(
            "Run first: python -m src.collect_detailed --matches 500 --sleep 2"
        )

    df = pd.read_csv(DATA).dropna(subset=["match_id", "start_time", "radiant_win"])
    df = df.sort_values(["start_time", "match_id"]).reset_index(drop=True)

    if len(df) < 5000:
        print("WARNING: collect 10,000+ detailed matches if possible.")

    split = int(len(df) * 0.8)
    if split < 1 or len(df) - split < 1:
        raise SystemExit("Not enough matches.")

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    indexer = HeroIndexer().fit(train_df)

    train_loader = make_loader(indexer, train_df, 512, True)
    test_loader = make_loader(indexer, test_df, 1024, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DraftNetV7(indexer.n_heroes).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    best_loss = float("inf")
    best_state = None
    stale = 0
    patience = 5

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        total_n = 0

        for heroes, pos, picks, mmr, rank, y in train_loader:
            heroes = heroes.to(device)
            pos = pos.to(device)
            picks = picks.to(device)
            mmr = mmr.to(device)
            rank = rank.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(heroes, pos, picks, mmr, rank)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(y)
            total_n += len(y)

        train_loss = total_loss / max(total_n, 1)
        m = evaluate(model, test_loader, device)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} "
            f"test_logloss={m['log_loss']:.5f} "
            f"auc={m['roc_auc']} "
            f"acc={m['accuracy']:.4f}"
        )

        if m["log_loss"] < best_loss:
            best_loss = m["log_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    final = evaluate(model, test_loader, device)
    final.update({
        "n_matches": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features": [
            "hero",
            "position",
            "pick_order",
            "avg_mmr",
            "rank_bracket",
            "hero_vs_hero_interactions",
        ],
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "n_heroes": indexer.n_heroes,
    }, OUT)

    torch.save({
        "hero_to_idx": indexer.hero_to_idx,
        "idx_to_hero": indexer.idx_to_hero,
    }, META)

    METRICS.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFinal metrics:")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
