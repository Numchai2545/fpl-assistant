"""Fail cloud builds that would discard an unverifiable pre-deadline forecast."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from fplbot.config import load_config


def check() -> None:
    cfg = load_config()
    summary = json.loads((cfg.site_dir / "summary.json").read_text(encoding="utf-8"))
    if datetime.fromisoformat(summary["deadline"]) <= datetime.now(timezone.utc):
        return
    paths = sorted((cfg.data_dir / "projections").glob(f"gw{summary['gw']}-*.json"))
    if not paths:
        raise RuntimeError("no frozen projection exists for the published GW")
    projection = json.loads(paths[-1].read_text(encoding="utf-8"))
    if (projection.get("model_version") != summary["model_version"]
            or not projection.get("config_hash") or not projection.get("provenance")
            or datetime.fromisoformat(projection["built_at"]) >= datetime.fromisoformat(projection["deadline"])):
        raise RuntimeError("frozen projection has missing or inconsistent metadata")


if __name__ == "__main__":
    check()
