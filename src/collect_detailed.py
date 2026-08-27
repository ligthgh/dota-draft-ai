from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

from .collect_positions import assign_team_positions

MATCHES_CSV = Path("data/matches.csv")
CACHE_DIR = Path("data/match_details")
OUT = Path("data/detailed_matches.csv")
API_BASE = "https://api.opendota.com/api"


def fetch_match(session: requests.Session, match_id: int, sleep_s: float) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{match_id}.json"

    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    for attempt in range(8):
        r = session.get(f"{API_BASE}/matches/{match_id}", timeout=45)

        if r.status_code == 429:
            wait = min(120, 10 * (attempt + 1))
            print(f"429 for {match_id}. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code == 404:
            return None

        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(sleep_s)
        return data

    return None


def rank_bracket_from_mmr(avg_mmr) -> int:
    try:
        mmr = float(avg_mmr)
        if math.isnan(mmr):
            return 0
    except Exception:
        return 0

    if mmr < 1000: return 1
    if mmr < 2000: return 2
    if mmr < 3000: return 3
    if mmr < 4000: return 4
    if mmr < 5000: return 5
    if mmr < 6000: return 6
    if mmr < 7000: return 7
    return 8


def build_pick_order(match: dict) -> dict[int, int]:
    seq = match.get("picks_bans") or []
    picks = []

    for item in seq:
        if item.get("is_pick") is True and item.get("hero_id") is not None:
            raw_order = item.get("order")
            hero_id = int(item["hero_id"])
            order = int(raw_order) if raw_order is not None else len(picks)
            picks.append((order, hero_id))

    picks.sort(key=lambda x: x[0])
    return {hero_id: idx + 1 for idx, (_, hero_id) in enumerate(picks)}


def normalize_match(match: dict) -> dict | None:
    players = match.get("players") or []
    if len(players) != 10:
        return None

    radiant = [p for p in players if int(p.get("player_slot", 999)) < 128]
    dire = [p for p in players if int(p.get("player_slot", 0)) >= 128]

    r_pos = assign_team_positions(radiant)
    d_pos = assign_team_positions(dire)

    if len(r_pos) != 5 or len(d_pos) != 5:
        return None

    pick_order = build_pick_order(match)
    avg_mmr = float(match.get("avg_mmr") or 0)

    row = {
        "match_id": int(match["match_id"]),
        "start_time": int(match.get("start_time") or 0),
        "avg_mmr": avg_mmr,
        "rank_bracket": rank_bracket_from_mmr(avg_mmr),
        "game_mode": int(match.get("game_mode") or 0),
        "lobby_type": int(match.get("lobby_type") or 0),
        "radiant_win": int(bool(match.get("radiant_win"))),
    }

    def add_team(prefix, team, positions):
        by_position = {}
        for p in team:
            pos = int(positions[int(p["player_slot"])])
            by_position[pos] = p

        if len(by_position) != 5:
            raise ValueError("Could not infer five unique positions.")

        for pos in range(1, 6):
            p = by_position[pos]
            hero_id = int(p["hero_id"])
            row[f"{prefix}_hero_{pos}"] = hero_id
            row[f"{prefix}_pos_{pos}"] = pos
            row[f"{prefix}_pick_{pos}"] = int(pick_order.get(hero_id, 0))

    try:
        add_team("radiant", radiant, r_pos)
        add_team("dire", dire, d_pos)
    except Exception:
        return None

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if not MATCHES_CSV.exists():
        raise SystemExit("Run first: python -m src.collect --pages 300 --sleep 2")

    df = pd.read_csv(MATCHES_CSV)
    df = df.sort_values(["start_time", "match_id"], ascending=False)
    match_ids = [int(x) for x in df["match_id"].dropna().head(args.matches)]

    session = requests.Session()
    session.headers["User-Agent"] = "dota-draft-ai-v7/0.7"

    rows = []

    for i, match_id in enumerate(match_ids, 1):
        try:
            details = fetch_match(session, match_id, args.sleep)
            row = normalize_match(details) if details else None
            if row is not None:
                rows.append(row)

            print(f"[{i}/{len(match_ids)}] match={match_id} accepted={len(rows)}")
        except Exception as exc:
            print(f"[{i}/{len(match_ids)}] match={match_id} ERROR: {exc}")

    if not rows:
        raise RuntimeError("No detailed matches collected.")

    out = pd.DataFrame(rows).drop_duplicates("match_id")
    out = out.sort_values(["start_time", "match_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\nSaved {len(out):,} matches -> {OUT}")


if __name__ == "__main__":
    main()
