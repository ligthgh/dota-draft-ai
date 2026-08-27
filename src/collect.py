from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://api.opendota.com/api/publicMatches"
OUT = Path("data/matches.csv")
WINDOW_META = Path("data/recent_window.json")


def parse_team(value) -> list[int] | None:
    if value is None:
        return None

    if isinstance(value, list):
        heroes = [int(x) for x in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        heroes = [int(x) for x in text.split(",") if str(x).strip()]

    if len(heroes) != 5 or len(set(heroes)) != 5:
        return None

    return heroes


def normalize_match(match: dict) -> dict | None:
    radiant = parse_team(match.get("radiant_team"))
    dire = parse_team(match.get("dire_team"))

    if radiant is None or dire is None:
        return None

    radiant_win = match.get("radiant_win")
    start_time = int(match.get("start_time") or 0)

    if radiant_win is None or not start_time:
        return None

    row = {
        "match_id": int(match["match_id"]),
        "start_time": start_time,
        "duration": int(match.get("duration") or 0),
        "avg_mmr": match.get("avg_mmr"),
        "num_mmr": match.get("num_mmr"),
        "lobby_type": match.get("lobby_type"),
        "game_mode": match.get("game_mode"),
        "radiant_win": int(bool(radiant_win)),
    }

    for i, hero in enumerate(radiant, 1):
        row[f"radiant_{i}"] = hero

    for i, hero in enumerate(dire, 1):
        row[f"dire_{i}"] = hero

    return row


def fetch_page(
    session: requests.Session,
    less_than_match_id: int | None = None,
    max_retries: int = 8,
) -> list[dict]:
    params = {}

    if less_than_match_id is not None:
        params["less_than_match_id"] = less_than_match_id

    for attempt in range(max_retries):
        response = session.get(
            API,
            params=params,
            timeout=30,
        )

        if response.status_code == 429:
            wait = min(120, 10 * (attempt + 1))
            print(
                f"OpenDota rate limit (429). "
                f"Waiting {wait}s..."
            )
            time.sleep(wait)
            continue

        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected API response: {type(payload)!r}"
            )

        return payload

    raise RuntimeError(
        "OpenDota rate limit did not reset. "
        "Try again later or use fewer pages."
    )


RECENT_DAYS = 30


def collect(
    pages: int,
    sleep_s: float,
) -> tuple[pd.DataFrame, int]:
    session = requests.Session()
    session.headers["User-Agent"] = "dota-draft-ai-recent/0.5"

    now_ts = int(time.time())
    cutoff_time = int(now_ts - RECENT_DAYS * 24 * 3600)

    cutoff_iso = datetime.fromtimestamp(
        cutoff_time,
        tz=timezone.utc,
    ).isoformat()

    print(f"Recent-match window: last {RECENT_DAYS} days")
    print(f"Cutoff UTC: {cutoff_iso}")
    print("No patch lookup and no per-match API requests are used.\n")

    rows = []
    cursor = None
    seen = set()

    for page in range(1, pages + 1):
        raw = fetch_page(session, cursor)

        if not raw:
            break

        page_rows = []
        page_times = []

        for match in raw:
            start_time = int(match.get("start_time") or 0)

            if start_time:
                page_times.append(start_time)

            if start_time < cutoff_time:
                continue

            row = normalize_match(match)

            if row and row["match_id"] not in seen:
                seen.add(row["match_id"])
                rows.append(row)
                page_rows.append(row)

        ids = [
            int(x["match_id"])
            for x in raw
            if x.get("match_id") is not None
        ]

        if not ids:
            break

        cursor = min(ids)

        print(
            f"[{page}/{pages}] "
            f"api={len(raw)} "
            f"recent={len(page_rows)} "
            f"total={len(rows)} "
            f"next_cursor={cursor}"
        )

        # publicMatches pages go newest -> older.
        # If even the newest match on this page is older than our cutoff,
        # all following pages are also too old.
        if page_times and max(page_times) < cutoff_time:
            print("Reached matches older than the selected time window. Stopping.")
            break

        time.sleep(sleep_s)

    session.close()

    if not rows:
        raise RuntimeError(
            "No matches were collected inside the selected time window. "
            "Try --hours 48 or --hours 72."
        )

    df = pd.DataFrame(rows).drop_duplicates("match_id")
    df = df[df["start_time"] >= cutoff_time]
    df = df.sort_values(
        ["start_time", "match_id"]
    ).reset_index(drop=True)

    return df, cutoff_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    df, cutoff_time = collect(
        pages=args.pages,
        sleep_s=args.sleep,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    WINDOW_META.write_text(
        json.dumps(
            {
                "days": RECENT_DAYS,
                "cutoff_time": cutoff_time,
                "cutoff_iso_utc": datetime.fromtimestamp(
                    cutoff_time,
                    tz=timezone.utc,
                ).isoformat(),
                "collected_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "matches": int(len(df)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved {len(df):,} recent matches -> {args.out}")
    print(f"Window metadata -> {WINDOW_META}")


if __name__ == "__main__":
    main()
