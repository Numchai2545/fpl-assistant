"""Deadline alert.

Reads the summary the dashboard build wrote and pushes it to Telegram when the
deadline is inside the configured window. Designed to be run on a schedule that
fires more often than it sends — it decides for itself whether it is due.

Telegram needs two environment variables:
    TELEGRAM_BOT_TOKEN   from @BotFather
    TELEGRAM_CHAT_ID     from https://api.telegram.org/bot<TOKEN>/getUpdates
Neither ever goes in config.yaml.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import requests

from .config import Config

log = logging.getLogger(__name__)


def _summary(cfg: Config) -> dict | None:
    path = cfg.site_dir / "summary.json"
    if not path.exists():
        log.error("no summary.json — run `python -m fplbot build` first")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compose(summary: dict, hours_left: float) -> str:
    lines = [
        f"⚽ *Gameweek {summary['gw']}* — deadline in {hours_left:.0f}h",
        "",
        f"*{summary['headline']}*",
        summary["detail"],
        "",
        f"Projected: {summary['expected_points']:.0f} pts",
    ]
    plan = summary.get("plan") or []
    if len(plan) > 1:
        lines += ["", "_Next weeks:_"]
        for week in plan[1:4]:
            lines.append(f"GW{week['gw']}: {week['summary']}")
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
        return False
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown",
              "disable_web_page_preview": True},
        timeout=20,
    )
    if resp.status_code != 200:
        log.error("telegram rejected the message: %s", resp.text[:300])
        return False
    return True


def send_if_due(cfg: Config, force: bool = False) -> int:
    summary = _summary(cfg)
    if summary is None:
        return 1

    deadline = datetime.fromisoformat(summary["deadline"])
    hours_left = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
    window = float(cfg.get("notify", "hours_before_deadline", default=24))

    if not force:
        # Fire once, in the hours leading up to the window closing.
        if not (window - 6 <= hours_left <= window + 1):
            log.info("not due: %.1f hours to the deadline, window is %.0fh",
                     hours_left, window)
            return 0
        marker = cfg.site_dir / f".notified-gw{summary['gw']}"
        if marker.exists():
            log.info("already notified for GW%s", summary["gw"])
            return 0
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    text = compose(summary, hours_left)
    if not cfg.get("notify", "telegram", "enabled", default=False):
        log.warning("telegram disabled in config; printing instead")
        print(text)
        return 0
    return 0 if send_telegram(text) else 1
