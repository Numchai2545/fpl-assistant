"""Configuration loading and shared FPL rule constants for 2026/27."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# 2026/27 rules. Verified against the official rules page on 13 Sep 2026.
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

    def chips_available(self) -> list[str]:
        chips = self.get("chips", default={}) or {}
        return [name for name, used in chips.items() if not used]


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    # environment overrides, handy for CI
    if os.environ.get("FPL_TEAM_ID"):
        raw.setdefault("entry", {})["team_id"] = int(os.environ["FPL_TEAM_ID"])
    return Config(raw=raw, path=cfg_path)
