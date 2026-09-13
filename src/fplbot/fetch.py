"""Talk to the official Fantasy Premier League API and keep a local snapshot.

No API key, no authentication, no scraping. Every endpoint used here is the same
public JSON the FPL website itself calls.

Snapshots are written to data/snapshots/<YYYY-MM-DD>/ so the project slowly
builds its own history — that history is what Phase 4 backtesting runs on.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {
    # The API rejects requests with no user agent.
    "User-Agent": "fpl-assistant/1.0 (personal use)",
    "Accept": "application/json",
}


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers.update(HEADERS)
    return s


class FPLClient:
    def __init__(self, cfg: Config, offline: bool = False):
        self.cfg = cfg
        self.offline = offline
        self.session = _session()
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.snapshot_dir = cfg.data_dir / "snapshots" / stamp
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- plumbing
    def _get(self, path: str, cache_name: str | None = None) -> Any:
        cache_path = self.snapshot_dir / f"{cache_name}.json" if cache_name else None
        if cache_path and cache_path.exists():
            log.debug("cache hit %s", cache_path.name)
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if self.offline:
            raise RuntimeError(
                f"offline mode and no cached copy of {cache_name!r} in {self.snapshot_dir}"
            )
        url = f"{BASE}/{path.lstrip('/')}"
        log.info("GET %s", url)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if cache_path:
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    # ------------------------------------------------------------- league data
    def bootstrap(self) -> dict:
        """Every player, team, gameweek and the whole scoring configuration."""
        return self._get("bootstrap-static/", "bootstrap")

    def fixtures(self) -> list[dict]:
        """All 380 fixtures, with FPL's own difficulty rating for each side."""
        return self._get("fixtures/", "fixtures")

    def element_summary(self, element_id: int) -> dict:
        """One player's full match-by-match history plus their upcoming fixtures."""
        return self._get(f"element-summary/{element_id}/", f"element_{element_id}")

    def element_summaries(self, ids: Iterable[int], pause: float = 0.15) -> dict[int, dict]:
        """Fetch histories for a shortlist. Never call this for all ~700 players."""
        out: dict[int, dict] = {}
        for i, pid in enumerate(ids):
            out[pid] = self.element_summary(pid)
            if pause and i % 10 == 9:
                time.sleep(pause)
        return out

    # --------------------------------------------------------------- your team
    def entry(self, team_id: int | None = None) -> dict:
        tid = team_id or self.cfg.team_id
        return self._get(f"entry/{tid}/", "entry")

    def entry_history(self, team_id: int | None = None) -> dict:
        tid = team_id or self.cfg.team_id
        return self._get(f"entry/{tid}/history/", "entry_history")

    def entry_picks(self, gameweek: int, team_id: int | None = None) -> dict:
        tid = team_id or self.cfg.team_id
        return self._get(f"entry/{tid}/event/{gameweek}/picks/", f"picks_gw{gameweek}")


# ------------------------------------------------------------------ gameweeks
def current_gameweek(bootstrap: dict) -> int | None:
    for ev in bootstrap["events"]:
        if ev.get("is_current"):
            return int(ev["id"])
    return None


def next_gameweek(bootstrap: dict) -> dict | None:
    """The gameweek you are about to pick a team for, with its deadline."""
    for ev in bootstrap["events"]:
        if ev.get("is_next"):
            return ev
    # pre-season, or the season is over
    for ev in bootstrap["events"]:
        if not ev.get("finished"):
            return ev
    return None


def deadline_utc(event: dict) -> datetime:
    return datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))


def free_transfers(entry_history: dict, picks: dict | None) -> int:
    """How many free transfers you have for the upcoming gameweek.

    The API does not expose this directly on the public endpoints, so it is
    reconstructed from your transfer history: you bank one per gameweek, spend
    what you use, and the total is capped at five.
    """
    max_saved = 5
    ft = 1
    for row in entry_history.get("current", []):
        made = int(row.get("event_transfers", 0))
        paid = int(row.get("event_transfers_cost", 0)) // 4
        free_used = max(0, made - paid)
        ft = min(max_saved, max(1, ft - free_used + 1))
    return max(1, min(max_saved, ft))


def bank_and_value(entry: dict) -> tuple[float, float]:
    """Bank and squad value in millions."""
    return (
        entry.get("last_deadline_bank", 0) / 10.0,
        entry.get("last_deadline_value", 1000) / 10.0,
    )
