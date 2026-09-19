"""Talk to the official Fantasy Premier League API and keep a local snapshot.

No API key, no authentication, no scraping. Every endpoint used here is the same
public JSON the FPL website itself calls.

Snapshots are written to data/snapshots/<YYYY-MM-DD>/ so the project slowly
builds its own history — that history is what Phase 4 backtesting runs on.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HIT_COST, MAX_SAVED_TRANSFERS, Config

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
    """Fetches live FPL data, writing every payload to a dated snapshot.

    The cache exists to build history for backtesting and to let `--offline`
    work, *not* to save requests. Injury news and availability flags change
    through the day, and a stale snapshot the evening before a deadline is
    exactly the failure this whole project exists to prevent — so a cached file
    is only reused inside `cache_ttl_minutes`, or when offline.
    """

    def __init__(self, cfg: Config, offline: bool = False, force_refresh: bool = False):
        self.cfg = cfg
        self.offline = offline
        self.force_refresh = force_refresh and not offline
        self.session = _session()
        self.provenance: dict[str, dict[str, str]] = {}
        self.run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshots = cfg.data_dir / "snapshots"
        today = snapshots / stamp
        if offline and not (today / "bootstrap.json").exists():
            available = sorted(
                (path for path in snapshots.glob("*")
                 if path.is_dir() and (path / "bootstrap.json").exists()),
                reverse=True,
            )
            self.snapshot_dir = available[0] if available else today
            if available:
                log.warning("offline mode using latest snapshot %s", self.snapshot_dir.name)
        else:
            self.snapshot_dir = today
        if not offline:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(minutes=float(
            cfg.get("planning", "cache_ttl_minutes", default=90)))

    def _cache_is_fresh(self, path: Path) -> bool:
        if self.offline:
            return True
        if self.force_refresh:
            return False
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc)
        if age <= self.ttl:
            return True
        log.debug("cache stale by %s: %s", age - self.ttl, path.name)
        return False

    # ---------------------------------------------------------------- plumbing
    def _get(self, path: str, cache_name: str | None = None) -> Any:
        cache_path = self.snapshot_dir / f"{cache_name}.json" if cache_name else None
        if cache_path and cache_path.exists() and self._cache_is_fresh(cache_path):
            log.debug("cache hit %s", cache_path.name)
            fetched = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
            self.provenance[cache_name or path] = {
                "source": "offline" if self.offline else "cache",
                "fetched_at": fetched.isoformat(),
            }
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
            archive = self.snapshot_dir / "runs" / self.run_stamp / cache_path.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, archive)
            self.provenance[cache_name or path] = {
                "source": "network", "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
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

    def entry_transfers(self, team_id: int | None = None) -> list[dict]:
        """Every transfer you have made, with the price paid for each player.

        This is public, and it is the only way to recover purchase prices
        without logging in — `picks` does not carry them.
        """
        tid = team_id or self.cfg.team_id
        return self._get(f"entry/{tid}/transfers/", "entry_transfers")

    def event_live(self, event: int) -> dict:
        """Live points for a started gameweek; provisional until every match finishes."""
        return self._get(f"event/{event}/live/", f"live_gw{event}")


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


def free_transfers(entry_history: dict) -> int:
    """How many free transfers you have for the upcoming gameweek.

    The public endpoints do not expose this, so it is reconstructed from your
    transfer history: you bank one per gameweek, spend what you use, and the
    total is capped at five.

    Wildcard and Free Hit gameweeks are skipped rather than counted — transfers
    made under those chips are unlimited and free, and treating them as spent
    free transfers would reset the count to 1 for no reason.
    """
    chip_gws = {int(c["event"]): (c.get("name") or "").lower()
                for c in entry_history.get("chips", []) if c.get("event") is not None}
    unlimited = {"wildcard", "freehit", "free_hit"}

    ft = 1
    for row in entry_history.get("current", []):
        if chip_gws.get(int(row.get("event", 0)), "") in unlimited:
            # The chip covered the moves; the banked transfer still accrues.
            ft = min(MAX_SAVED_TRANSFERS, ft + 1)
            continue
        made = int(row.get("event_transfers", 0))
        paid = int(row.get("event_transfers_cost", 0)) // HIT_COST
        free_used = max(0, made - paid)
        ft = min(MAX_SAVED_TRANSFERS, max(1, ft - free_used + 1))
    return max(1, min(MAX_SAVED_TRANSFERS, ft))


def apply_event_transfer_state(squad: list[int], bank: float,
                               transfers: list[dict] | None,
                               event: int) -> tuple[list[int], float, int]:
    """Bring the last deadline's squad and bank through this GW's moves.

    The entry endpoint exposes the bank at the previous deadline. Transfers
    made since then must therefore update both the player ids and the cash:
    ``bank + sale price - purchase price``. The transfer endpoint reports both
    prices in tenths, so this does not need an estimate of the sell-on fee.
    """
    current = list(squad)
    current_bank = float(bank)
    applied = 0
    rows = [row for row in (transfers or []) if int(row.get("event") or 0) == event]
    for row in sorted(rows, key=lambda r: r.get("time") or ""):
        out_id, in_id = int(row["element_out"]), int(row["element_in"])
        if out_id not in current:
            continue
        current[current.index(out_id)] = in_id
        try:
            sale = int(row["element_out_cost"]) / 10.0
            purchase = int(row["element_in_cost"]) / 10.0
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"current-GW transfer {out_id} -> {in_id} has no usable prices; "
                "cannot reconstruct the current bank safely"
            ) from exc
        else:
            current_bank += sale - purchase
        applied += 1
    return current, round(current_bank, 1), applied


def apply_event_transfers(squad: list[int], transfers: list[dict] | None,
                          event: int) -> tuple[list[int], int]:
    """Compatibility wrapper for callers that only need the updated squad."""
    current = list(squad)
    applied = 0
    rows = [row for row in (transfers or []) if int(row.get("event") or 0) == event]
    for row in sorted(rows, key=lambda r: r.get("time") or ""):
        out_id, in_id = int(row["element_out"]), int(row["element_in"])
        if out_id not in current:
            continue
        current[current.index(out_id)] = in_id
        applied += 1
    return current, applied


def purchase_prices(squad: list[int], transfers: list[dict] | None, players) -> dict[int, float]:
    """What you paid for each player you currently own, in millions.

    Two sources, neither needing a login:

    * Players you transferred in — the public transfers endpoint records
      `element_in_cost`, the price at the moment you bought. The most recent
      purchase wins, since you can buy the same player more than once.
    * Players from your original squad — never transferred in, so you paid the
      season-start price, which is `now_cost` minus `cost_change_start`.
    """
    bought_at: dict[int, float] = {}
    for row in sorted(
        transfers or [],
        key=lambda r: (int(r.get("event") or 0), str(r.get("time") or ""),
                       int(r.get("element_in") or 0)),
    ):
        pid = int(row["element_in"])
        bought_at[pid] = int(row["element_in_cost"]) / 10.0

    out: dict[int, float] = {}
    for pid in squad:
        if pid in bought_at:
            out[pid] = bought_at[pid]
        elif pid in players.index:
            now = float(players.price.get(pid, 0.0))
            drift = float(players.cost_change_start.get(pid, 0.0)) / 10.0
            out[pid] = round(now - drift, 1)
    return out


def selling_prices(squad: list[int], transfers: list[dict] | None, players,
                   fee: float = 0.5) -> dict[int, float]:
    """What you would actually get back for each player you own, in millions.

    FPL keeps `fee` (normally half) of any price *rise* since you bought,
    rounded down to the nearest 0.1m; a price fall you absorb in full. Using
    market value instead inflates the budget and produces plans FPL refuses
    at the point of sale.
    """
    out: dict[int, float] = {}
    for pid, bought in purchase_prices(squad, transfers, players).items():
        now = float(players.price.get(pid, bought))
        if now <= bought:
            out[pid] = now
            continue
        # Tenths, rounded down, is how FPL actually computes the sell-on fee.
        out[pid] = bought + math.floor((now - bought) * (1.0 - fee) * 10 + 1e-9) / 10.0
    return out


def bank_and_value(entry: dict) -> tuple[float, float]:
    """Bank and squad value in millions, as two separate numbers.

    `last_deadline_value` from the API is the figure FPL shows as "Value", which
    already includes money in the bank. Reporting both without subtracting would
    count the bank twice.
    """
    bank = entry.get("last_deadline_bank", 0) / 10.0
    total = entry.get("last_deadline_value", 1000) / 10.0
    return bank, round(total - bank, 1)
