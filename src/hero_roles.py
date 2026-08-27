from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import requests

POSITION_STATS = Path("data/hero_position_stats.json")
FALLBACK_CACHE = Path("data/hero_roles_fallback.json")
HERO_STATS_URL = "https://api.opendota.com/api/heroStats"


def fallback_positions_from_tags(hero: dict) -> list[int]:
    roles = set(hero.get("roles") or [])
    positions = set()

    if "Carry" in roles:
        positions.add(1)
    if "Nuker" in roles or ("Carry" in roles and "Support" not in roles):
        positions.add(2)
    if "Initiator" in roles or "Durable" in roles:
        positions.add(3)
    if "Support" in roles or ("Disabler" in roles and "Carry" not in roles):
        positions.add(4)
    if "Support" in roles:
        positions.add(5)

    return sorted(positions or {1, 2, 3, 4, 5})


def load_fallback() -> dict[int, list[int]]:
    try:
        response = requests.get(HERO_STATS_URL, timeout=30)
        response.raise_for_status()
        heroes = response.json()
        FALLBACK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_CACHE.write_text(
            json.dumps(heroes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        if not FALLBACK_CACHE.exists():
            return {}
        heroes = json.loads(FALLBACK_CACHE.read_text(encoding="utf-8"))

    return {
        int(hero["id"]): fallback_positions_from_tags(hero)
        for hero in heroes
        if hero.get("id") is not None
    }


@lru_cache(maxsize=1)
def empirical_stats() -> dict:
    if not POSITION_STATS.exists():
        return {}
    try:
        data = json.loads(POSITION_STATS.read_text(encoding="utf-8"))
        return data.get("heroes", {})
    except Exception:
        return {}


@lru_cache(maxsize=1)
def fallback_roles() -> dict[int, list[int]]:
    return load_fallback()


def get_hero_positions(hero_id: int) -> list[int]:
    hero_id = int(hero_id)
    item = empirical_stats().get(str(hero_id))

    if item:
        positions = [int(x) for x in item.get("allowed_positions", [])]
        if positions:
            return positions

    return fallback_roles().get(hero_id, [1, 2, 3, 4, 5])


def hero_can_play_role(hero_id: int, role: int) -> bool:
    return int(role) in get_hero_positions(int(hero_id))


def get_position_stats(hero_id: int, role: int) -> dict | None:
    item = empirical_stats().get(str(int(hero_id)))
    if not item:
        return None
    return item.get("positions", {}).get(str(int(role)))


def clear_role_cache():
    empirical_stats.cache_clear()
    fallback_roles.cache_clear()
