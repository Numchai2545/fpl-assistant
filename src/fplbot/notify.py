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


def _fmt_left(hours_left: float) -> str:
    if hours_left >= 24:
        return f"อีก {int(hours_left // 24)} วัน {int(hours_left % 24)} ชม."
    if hours_left >= 1:
        return f"อีก {int(hours_left)} ชม."
    return f"อีก {max(0, int(hours_left * 60))} นาที"


def compose(summary: dict, hours_left: float, stage: float | None = None,
            urgent_only: bool = True) -> str:
    """Build the Thai reminder.

    Urgency shapes the message. Two days out you want the plan; three hours out
    you want the one thing you have not done yet, with nothing else competing
    for attention.
    """
    urgent = stage is not None and stage <= 6
    head = "🚨 ใกล้ปิดแล้ว" if urgent else "⚽ เตือนจัดตัว"

    lines = [
        f"{head} — *GW{summary['gw']}* ปิด{_fmt_left(hours_left)}",
        f"_{summary.get('deadline_local', '')}_".rstrip("_ ") or "",
        "",
    ]

    # Anything unavailable in the XI goes above the fold: this is the failure
    # that actually costs points when a deadline is missed.
    alerts = summary.get("alerts") or []
    if alerts:
        lines.append("*⚠️ ต้องแก้ก่อนปิด*")
        lines += [f"• {a}" for a in alerts[:4]]
        lines.append("")

    lines += [f"*{summary['headline']}*", summary["detail"], ""]

    moves = summary.get("moves") or []
    if moves:
        lines.append("*เปลี่ยนตัว*")
        lines += [f"• {m['out']} ➜ {m['in']}" for m in moves]
        lines.append("")

    lines.append(f"คาดการณ์ {summary['expected_points']:.0f} แต้ม")

    if not urgent:
        plan = summary.get("plan") or []
        if len(plan) > 1:
            lines += ["", "_สัปดาห์ถัดไป_"]
            for week in plan[1:4]:
                lines.append(f"GW{week['gw']}: {week['summary']}")

    url = summary.get("site_url")
    if url:
        lines += ["", f"ดูรายละเอียด: {url}"]
    return "\n".join(line for line in lines if line is not None)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
        return False

    def post(payload: dict) -> requests.Response:
        return requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json=payload, timeout=20)

    base = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    resp = post({**base, "parse_mode": "Markdown"})
    if resp.status_code == 200:
        return True

    # A player name containing an underscore or asterisk makes Telegram reject
    # the whole message. The reminder matters more than the bold text, so send
    # it unformatted rather than losing it.
    log.warning("telegram rejected the formatted message (%s); retrying as plain text",
                resp.text[:160])
    resp = post(base)
    if resp.status_code != 200:
        log.error("telegram rejected the plain message too: %s", resp.text[:300])
        return False
    return True


def _stage_for(hours_left: float, stages: list[float]) -> float | None:
    """Which reminder stage this moment belongs to, if any.

    Stages are hours before the deadline, e.g. [48, 24, 3]. A run fires the
    nearest stage at or above the current time-to-deadline, so a scheduler that
    wakes every few hours still sends each reminder exactly once.
    """
    due = [s for s in sorted(stages) if hours_left <= s]
    return due[0] if due else None


def send_if_due(cfg: Config, force: bool = False) -> int:
    summary = _summary(cfg)
    if summary is None:
        return 1

    deadline = datetime.fromisoformat(summary["deadline"])
    hours_left = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
    stages = [float(h) for h in cfg.get(
        "notify", "remind_hours_before", default=[48, 24, 3])]
    urgent_only = bool(cfg.get("notify", "final_reminder_only_if_unset", default=True))

    stage = None
    if not force:
        if hours_left < 0:
            log.info("deadline has passed")
            return 0
        stage = _stage_for(hours_left, stages)
        if stage is None:
            log.info("not due: %.1f hours to the deadline, stages are %s",
                     hours_left, stages)
            return 0
        # State lives next to the data, not in the published site: docs/ is
        # gitignored in CI and regenerated every build, so a marker there never
        # survives and every scheduled run would alert again.
        marker = _marker_path(cfg, summary["gw"], stage)
        if marker.exists():
            log.info("already sent the %.0fh reminder for GW%s", stage, summary["gw"])
            return 0

    text = compose(summary, hours_left, stage, urgent_only)
    if not cfg.get("notify", "telegram", "enabled", default=False):
        log.warning("telegram disabled in config; printing instead")
        print(text)
        _record(cfg, summary, stage, force)
        return 0
    if not send_telegram(text):
        # Deliberately do not record: a failed send must be retried on the next
        # scheduled run rather than silently swallowed.
        return 1
    _record(cfg, summary, stage, force)
    return 0


def _marker_path(cfg: Config, gw: int, stage: float):
    return cfg.data_dir / "notified" / f"gw{gw}-{int(stage)}h"


def _record(cfg: Config, summary: dict, stage: float | None, force: bool) -> None:
    """Remember that this reminder went out, but only after it actually did."""
    if force or stage is None:
        return
    marker = _marker_path(cfg, summary["gw"], stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
