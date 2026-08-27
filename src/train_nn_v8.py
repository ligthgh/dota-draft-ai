from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .nn_data_v7 import HeroIndexer
from .nn_model_v7 import DraftNetV7

DATA = Path("data/detailed_matches.csv")
OUT = Path("models/draft_nn_v8.pt")
META = Path("models/draft_nn_v8_meta.pt")
METRICS = Path("models/nn_v8_metrics.json")


class DraftDataset(Dataset):
    """
    Training mode randomly hides one hero slot in many examples.

    Hidden slot:
      hero = 0
      position = 0
      pick_order = 0

    This teaches the network to evaluate incomplete 4v5 / 5v4 draft states.
    """
    def __init__(self, indexer, df: pd.DataFrame, training: bool):
        self.training = training

        heroes, positions, picks, mmr, rank = indexer.transform(df)

        self.heroes = heroes
        self.positions = positions
        self.picks = picks
        self.mmr = mmr
        self.rank = rank
        self.y = df["radiant_win"].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        heroes = self.heroes[idx].copy()
        positions = self.positions[idx].copy()
        picks = self.picks[idx].copy()

        if self.training:
            # 60% of training samples contain one unknown hero.
            if random.random() < 0.60:
                # Mask either Radiant or Dire with equal probability.
                if random.random() < 0.5:
                    slot = random.randint(0, 4)
                else:
                    slot = random.randint(5, 9)

                heroes[slot] = 0
                positions[slot] = 0
                picks[slot] = 0

        return (
            torch.tensor(heroes, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            torch.tensor(picks, dtype=torch.long),
            torch.tensor(self.mmr[idx], dtype=torch.float32),
            torch.tensor(self.rank[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def make_loader(indexer, df, batch_size, shuffle, training):
    ds = DraftDataset(indexer, df, training=training)
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


def partial_test_loader(indexer, df):
    """
    Deterministic test set with one Dire hero hidden.
    Lets us measure the exact scenario used in the new interface.
    """
    heroes, positions, picks, mmr, rank = indexer.transform(df)

    # Hide Dire position 5 slot (index 9) for validation.
    heroes[:, 9] = 0
    positions[:, 9] = 0
    picks[:, 9] = 0

    y = df["radiant_win"].astype(np.float32).to_numpy()

    ds = torch.utils.data.TensorDataset(
        torch.tensor(heroes, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
        torch.tensor(picks, dtype=torch.long),
        torch.tensor(mmr, dtype=torch.float32),
        torch.tensor(rank, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(ds, batch_size=1024, shuffle=False)


def main():
    if not DATA.exists():
        raise SystemExit(
            "Run first: python -m src.collect_detailed --matches 5000 --sleep 2"
        )

    df = pd.read_csv(DATA).dropna(
        subset=["match_id", "start_time", "radiant_win"]
    )
    df = df.sort_values(["start_time", "match_id"]).reset_index(drop=True)

    if len(df) < 5000:
        print(
            "WARNING: v8 works with partial drafts and benefits from more data. "
            "10,000+ detailed matches is recommended."
        )

    split = int(len(df) * 0.80)
    if split < 1 or len(df) - split < 1:
        raise SystemExit("Not enough matches.")

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    indexer = HeroIndexer().fit(train_df)

    train_loader = make_loader(
        indexer, train_df, 512, True, training=True
    )
    full_test_loader = make_loader(
        indexer, test_df, 1024, False, training=False
    )
    masked_test_loader = partial_test_loader(indexer, test_df)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = DraftNetV7(indexer.n_heroes).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3,
    )

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

            logits = model(
                heroes,
                pos,
                picks,
                mmr,
                rank,
            )

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(y)
            total_n += len(y)

        train_loss = total_loss / max(total_n, 1)

        full_metrics = evaluate(
            model,
            full_test_loader,
            device,
        )
        masked_metrics = evaluate(
            model,
            masked_test_loader,
            device,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} "
            f"full_logloss={full_metrics['log_loss']:.5f} "
            f"partial_logloss={masked_metrics['log_loss']:.5f} "
            f"partial_auc={masked_metrics['roc_auc']} "
            f"partial_acc={masked_metrics['accuracy']:.4f}"
        )

        # Optimize for the partial-draft use case.
        current_loss = masked_metrics["log_loss"]

        if current_loss < best_loss:
            best_loss = current_loss

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
        raise RuntimeError("Training did not produce a model.")

    model.load_state_dict(best_state)

    full_final = evaluate(model, full_test_loader, device)
    masked_final = evaluate(model, masked_test_loader, device)

    final = {
        "n_matches": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "full_draft": full_final,
        "partial_enemy_draft": masked_final,
        "masked_training_probability": 0.60,
        "features": [
            "hero",
            "position",
            "pick_order",
            "avg_mmr",
            "rank_bracket",
            "hero_vs_hero_interactions",
            "unknown_hero_mask",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_heroes": indexer.n_heroes,
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
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinal metrics:")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
