"""Configuration loading and shared FPL rule constants for 2026/27."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Scoring rules.
#
# These are NOT hand-maintained any more. `verify_scoring()` below checks them
# against `game_config.scoring` in the live bootstrap payload on every build, so
# a mid-season rule change surfaces as a loud warning instead of silently wrong
# expected points. Last confirmed against the API on 13 Sep 2026.
# --------------------------------------------------------------------------
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

GOAL_POINTS = {1: 10, 2: 6, 3: 5, 4: 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
DEFCON_POINTS = 2
# Defenders count tackles + clearances/blocks/interceptions.
# Midfielders and forwards also count ball recoveries, at a higher bar.
DEFCON_THRESHOLD = {1: None, 2: 10, 3: 12, 4: 12}
SAVES_PER_POINT = 3
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3

SQUAD_SIZE = 15
SQUAD_BY_POSITION = {1: 2, 2: 5, 3: 5, 4: 3}
XI_SIZE = 11
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
HIT_COST = 4
MAX_SAVED_TRANSFERS = 5

# status codes from the API
STATUS_UNAVAILABLE = {"i", "s", "u", "n"}  # injured, suspended, unavailable, not in squad

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path = ROOT / "config.yaml"

    # -- convenience accessors -------------------------------------------------
    @property
    def team_id(self) -> int:
        return int(self.raw["entry"]["team_id"])

    @property
    def horizon(self) -> int:
        return int(self.raw["planning"]["horizon"])

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def site_dir(self) -> Path:
        return ROOT / self.get("output", "site_dir", default="docs")

    @property
    def data_dir(self) -> Path:
        return ROOT / self.get("output", "data_dir", default="data")


def verify_scoring(bootstrap: dict) -> list[str]:
    """Check our hardcoded scoring against what the API says the rules are.

    FPL publishes the live scoring table at `game_config.scoring`. A wrong
    constant here is otherwise completely silent — every expected point would be
    off and nothing would fail — so this runs on every build and returns a list
    of human-readable mismatches for the caller to log and surface.
    """
    scoring = (bootstrap.get("game_config") or {}).get("scoring")
    if not isinstance(scoring, dict):
        return ["API did not publish game_config.scoring — constants unverified"]

    problems: list[str] = []

    def by_position(api_key: str, ours: dict[int, int], label: str) -> None:
        api = scoring.get(api_key)
        if not isinstance(api, dict):
            return
        for code, name in POSITIONS.items():
            if name in api and int(api[name]) != int(ours[code]):
                problems.append(f"{label} for {name}: we use {ours[code]}, API says {api[name]}")

    def scalar(api_key: str, ours: int, label: str) -> None:
        if api_key in scoring and int(scoring[api_key]) != int(ours):
            problems.append(f"{label}: we use {ours}, API says {scoring[api_key]}")

    by_position("goals_scored", GOAL_POINTS, "goal points")
    by_position("clean_sheets", CLEAN_SHEET_POINTS, "clean sheet points")
    scalar("assists", ASSIST_POINTS, "assist points")
    scalar("yellow_cards", YELLOW_CARD_POINTS, "yellow card points")
    scalar("red_cards", RED_CARD_POINTS, "red card points")
    scalar("long_play", 2, "60+ minute appearance points")
    scalar("short_play", 1, "sub-60 minute appearance points")

    defcon = scoring.get("defensive_contribution")
    if isinstance(defcon, dict):
        for code, name in POSITIONS.items():
            expected = 0 if DEFCON_THRESHOLD[code] is None else DEFCON_POINTS
            if name in defcon and int(defcon[name]) != expected:
                problems.append(
                    f"defensive contribution for {name}: we use {expected}, "
                    f"API says {defcon[name]}")

    # Squad shape and the transfer economy live in game_settings.
    settings = bootstrap.get("game_settings") or {}
    for key, ours, label in (
        ("squad_squadsize", SQUAD_SIZE, "squad size"),
        ("squad_squadplay", XI_SIZE, "XI size"),
        ("squad_team_limit", MAX_PER_CLUB, "players per club"),
    ):
        if key in settings and int(settings[key]) != int(ours):
            problems.append(f"{label}: we use {ours}, API says {settings[key]}")

    extra = settings.get("max_extra_free_transfers")
    if extra is not None and int(extra) + 1 != MAX_SAVED_TRANSFERS:
        problems.append(
            f"banked free transfers: we cap at {MAX_SAVED_TRANSFERS}, "
            f"API implies {int(extra) + 1}")
    return problems


def selling_fee(bootstrap: dict) -> float:
    """Fraction of a price *rise* FPL keeps when you sell. Normally 0.5."""
    settings = bootstrap.get("game_settings") or {}
    try:
        return float(settings.get("transfers_sell_on_fee", 0.5))
    except (TypeError, ValueError):
        return 0.5


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    # environment overrides, handy for CI
    if os.environ.get("FPL_TEAM_ID"):
        raw.setdefault("entry", {})["team_id"] = int(os.environ["FPL_TEAM_ID"])
    if os.environ.get("FPL_SITE_URL"):
        raw.setdefault("notify", {})["site_url"] = os.environ["FPL_SITE_URL"].rstrip("/")
    return Config(raw=raw, path=cfg_path)
