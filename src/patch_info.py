from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

API_BASE = "https://api.opendota.com/api"


@dataclass(frozen=True)
class PatchInfo:
    name: str
    start_time: int

    @property
    def start_iso(self) -> str:
        return datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat()


def _extract_patch_entries(payload) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        # Common forms: {"1": {"name": ..., "date": ...}} or {"patches": [...]}
        if isinstance(payload.get("patches"), list):
            return [x for x in payload["patches"] if isinstance(x, dict)]
        entries = []
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("id", key)
                entries.append(item)
        return entries

    return []


def _entry_timestamp(entry: dict) -> int | None:
    for key in ("date", "start_time", "timestamp", "time"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            value = int(float(value))
            # milliseconds -> seconds
            if value > 10_000_000_000:
                value //= 1000
            if value > 1_000_000_000:
                return value
        except (TypeError, ValueError):
            continue
    return None


def get_latest_patch(session: requests.Session | None = None) -> PatchInfo:
    """Return the newest Dota patch start time.

    Environment override is supported for reliability:
      DOTA_PATCH_START=<unix timestamp>
      DOTA_PATCH_NAME=<optional label>

    Otherwise the function asks OpenDota constants and chooses the newest
    patch entry with a timestamp. It fails loudly instead of silently mixing
    old patches into the dataset.
    """
    override = os.getenv("DOTA_PATCH_START")
    if override:
        try:
            start = int(override)
        except ValueError as exc:
            raise RuntimeError("DOTA_PATCH_START must be a Unix timestamp") from exc
        return PatchInfo(os.getenv("DOTA_PATCH_NAME", "manual-current-patch"), start)

    own_session = session is None
    session = session or requests.Session()

    errors = []
    try:
        for resource in ("patch", "patches"):
            url = f"{API_BASE}/constants/{resource}"
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                entries = _extract_patch_entries(response.json())

                dated = []
                for entry in entries:
                    ts = _entry_timestamp(entry)
                    if ts is None:
                        continue
                    name = str(
                        entry.get("name")
                        or entry.get("patch")
                        or entry.get("version")
                        or entry.get("id")
                        or "unknown"
                    )
                    dated.append((ts, name))

                if dated:
                    ts, name = max(dated, key=lambda x: x[0])
                    return PatchInfo(name=name, start_time=ts)
            except Exception as exc:
                errors.append(f"{resource}: {exc}")
    finally:
        if own_session:
            session.close()

    raise RuntimeError(
        "Could not determine the latest patch automatically. "
        "Set DOTA_PATCH_START to the Unix timestamp of the current patch. "
        f"OpenDota errors: {'; '.join(errors)}"
    )
