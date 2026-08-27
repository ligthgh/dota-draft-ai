from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

MATCHES_CSV = Path("data/matches.csv")
WINDOW_META = Path("data/recent_window.json")
CACHE_DIR = Path("data/match_details")
OUT = Path("data/hero_position_stats.json")
API_BASE = "https://api.opendota.com/api"


def api_params() -> dict:
    key = os.getenv("OPENDOTA_API_KEY")
    return {"api_key": key} if key else {}


def fetch_match(session: requests.Session, match_id: int, sleep_s: float) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{match_id}.json"

    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    for attempt in range(5):
        response = session.get(
            f"{API_BASE}/matches/{match_id}",
            params=api_params(),
            timeout=45,
        )

        if response.status_code == 429:
            wait = max(sleep_s, 5.0 * (attempt + 1))
            print(f"OpenDota rate limit. Waiting {wait:.1f}s...")
            time.sleep(wait)
            continue

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(sleep_s)
        return data

    return None


def farm_value(player: dict) -> float:
    if player.get("net_worth") is not None:
        return float(player["net_worth"])
    if player.get("gold_per_min") is not None:
        return float(player["gold_per_min"]) * 100.0
    if player.get("last_hits") is not None:
        return float(player["last_hits"])
    return 0.0


def assign_team_positions(players: list[dict]) -> dict[int, int]:
    """Infer approximate positions 1-5 from lane_role + farm priority."""
    valid = [
        p for p in players
        if p.get("hero_id") is not None and p.get("player_slot") is not None
    ]
    if len(valid) != 5:
        return {}

    result = {}
    used_positions = set()

    def assign(player: dict, position: int):
        slot = int(player["player_slot"])
        if slot not in result and position not in used_positions:
            result[slot] = position
            used_positions.add(position)

    roamers = [p for p in valid if p.get("is_roaming") is True]
    if roamers:
        assign(min(roamers, key=farm_value), 4)

    mids = [p for p in valid if int(p.get("lane_role") or 0) == 2]
    if mids:
        assign(max(mids, key=farm_value), 2)

    safe = [
        p for p in valid
        if int(p.get("lane_role") or 0) == 1
        and int(p["player_slot"]) not in result
    ]
    if len(safe) >= 2:
        safe = sorted(safe, key=farm_value, reverse=True)
        assign(safe[0], 1)
        assign(safe[-1], 5)
    elif len(safe) == 1:
        assign(safe[0], 1)

    off = [
        p for p in valid
        if int(p.get("lane_role") or 0) == 3
        and int(p["player_slot"]) not in result
    ]
    if len(off) >= 2:
        off = sorted(off, key=farm_value, reverse=True)
        assign(off[0], 3)
        assign(off[-1], 4)
    elif len(off) == 1:
        assign(off[0], 3)

    remaining_players = [
        p for p in valid if int(p["player_slot"]) not in result
    ]
    remaining_players = sorted(remaining_players, key=farm_value, reverse=True)
    remaining_positions = [
        p for p in [1, 2, 3, 4, 5] if p not in used_positions
    ]

    for player, position in zip(remaining_players, remaining_positions):
        assign(player, position)

    return result if len(result) == 5 else {}


def process_match(match: dict, counts, wins) -> bool:
    players = match.get("players") or []
    if len(players) != 10:
        return False

    radiant = [p for p in players if int(p.get("player_slot", 999)) < 128]
    dire = [p for p in players if int(p.get("player_slot", 0)) >= 128]

    r_positions = assign_team_positions(radiant)
    d_positions = assign_team_positions(dire)
    if len(r_positions) != 5 or len(d_positions) != 5:
        return False

    radiant_win = bool(match.get("radiant_win"))

    for player in radiant:
        hero = int(player["hero_id"])
        pos = r_positions[int(player["player_slot"])]
        counts[hero][pos] += 1
        wins[hero][pos] += int(radiant_win)

    for player in dire:
        hero = int(player["hero_id"])
        pos = d_positions[int(player["player_slot"])]
        counts[hero][pos] += 1
        wins[hero][pos] += int(not radiant_win)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=1.1)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--min-share", type=float, default=0.15)
    args = parser.parse_args()

    if not MATCHES_CSV.exists():
        raise SystemExit(
            "data/matches.csv not found. First run: "
            "python -m src.collect --pages 50 --sleep 1"
        )

    df = pd.read_csv(MATCHES_CSV)

    cutoff_time = None
    recent_days = None

    if WINDOW_META.exists():
        try:
            window = json.loads(
                WINDOW_META.read_text(encoding="utf-8")
            )
            cutoff_time = int(window["cutoff_time"])
            recent_days = int(window.get("days", 30))
        except Exception:
            cutoff_time = None

    if cutoff_time is not None and "start_time" in df.columns:
        df = df[df["start_time"] >= cutoff_time].copy()

    if df.empty:
        raise SystemExit(
            "No recent matches available. "
            "Run python -m src.collect first."
        )

    if recent_days:
        print(
            f"Position analysis: recent {recent_days}-day window only"
        )
    sort_cols = ["start_time", "match_id"] if "start_time" in df.columns else ["match_id"]
    df = df.sort_values(sort_cols, ascending=False)
    match_ids = [int(x) for x in df["match_id"].dropna().head(args.matches)]

    session = requests.Session()
    session.headers["User-Agent"] = "dota-draft-ai-role-miner/0.3"

    counts = defaultdict(Counter)
    wins = defaultdict(Counter)
    accepted = 0

    for i, match_id in enumerate(match_ids, start=1):
        try:
            match = fetch_match(session, match_id, args.sleep)
            if match and process_match(match, counts, wins):
                accepted += 1
            print(f"[{i}/{len(match_ids)}] match={match_id} accepted={accepted}")
        except Exception as exc:
            print(f"[{i}/{len(match_ids)}] match={match_id} ERROR: {exc}")

    output = {
        "meta": {
            "requested_matches": len(match_ids),
            "accepted_matches": accepted,
            "min_games": args.min_games,
            "min_share": args.min_share,
            "note": "Positions are inferred from recent parsed matches; inference is heuristic.",
        },
        "heroes": {},
    }

    for hero_id in sorted(counts):
        total = sum(counts[hero_id].values())
        positions = {}

        for pos in range(1, 6):
            games = int(counts[hero_id][pos])
            won = int(wins[hero_id][pos])
            positions[str(pos)] = {
                "games": games,
                "share": games / total if total else 0.0,
                "wins": won,
                "winrate": won / games if games else None,
            }

        allowed = [
            pos for pos in range(1, 6)
            if positions[str(pos)]["games"] >= args.min_games
            and positions[str(pos)]["share"] >= args.min_share
        ]

        if not allowed and total:
            allowed = [
                max(range(1, 6), key=lambda p: positions[str(p)]["games"])
            ]

        output["heroes"][str(hero_id)] = {
            "total_games": total,
            "allowed_positions": allowed,
            "positions": positions,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved -> {OUT}")
    print(f"Accepted matches: {accepted}/{len(match_ids)}")
    print(f"Heroes with role data: {len(output['heroes'])}")


if __name__ == "__main__":
    main()
