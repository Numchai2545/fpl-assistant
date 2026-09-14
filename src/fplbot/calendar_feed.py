"""Publish every gameweek deadline as a subscribable calendar.

This is the one reminder channel that keeps working when everything else breaks.
Push notifications depend on a bot token, a network call at the right moment and
a build having succeeded; a calendar subscription is copied into the phone once
and then fires from the phone itself, offline, whether or not this project ran
that week.

Output is an .ics feed at docs/deadlines.ics. Subscribe to it once — iOS:
Calendar > Add Calendar > Add Subscription Calendar; Android: Google Calendar >
Other calendars > From URL — and the phone reminds you before every deadline for
the rest of the season.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Folding at 75 octets is required by RFC 5545. Thai text is multi-byte, so the
# fold has to count bytes and must not split a character in half.
_LINE_LIMIT = 73


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;")
                .replace(",", "\\,").replace("\n", "\\n"))


def _fold(line: str) -> str:
    raw = line.encode("utf-8")
    if len(raw) <= _LINE_LIMIT:
        return line
    chunks, start = [], 0
    while start < len(raw):
        end = min(start + _LINE_LIMIT, len(raw))
        # Back off until the slice ends on a character boundary.
        while end > start and end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end -= 1
        chunks.append(raw[start:end].decode("utf-8"))
        start = end
    return "\r\n ".join(chunks)


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_ics(events: list[dict], *, site_url: str = "", calendar_name: str = "FPL Deadlines",
              remind_hours: list[float] | None = None,
              duration_minutes: int = 30) -> str:
    """Render upcoming gameweek deadlines as an iCalendar feed.

    One all-important detail: the alarms are attached to each event, so the
    phone fires them locally. A subscription that only carried the dates would
    still leave you to notice them.
    """
    remind_hours = remind_hours or [24]
    now = datetime.now(timezone.utc)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//fpl-assistant//deadlines//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_escape(calendar_name)}",
        "X-WR-TIMEZONE:UTC",
        # Ask subscribers to re-poll roughly twice a day.
        "REFRESH-INTERVAL;VALUE=DURATION:PT12H",
        "X-PUBLISHED-TTL:PT12H",
    ]

    for ev in events:
        if ev.get("finished"):
            continue
        raw = ev.get("deadline_time")
        if not raw:
            continue
        deadline = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        # A deadline that has already gone is noise at best, and on a fresh
        # subscription its alarms can fire immediately.
        if deadline <= now:
            continue

        gw = int(ev["id"])
        # Stable across rebuilds so subscribers update rather than duplicate.
        uid = hashlib.sha1(f"fpl-gw{gw}-{raw}".encode()).hexdigest()[:20]

        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}@fpl-assistant",
            f"DTSTAMP:{_stamp(now)}",
            f"DTSTART:{_stamp(deadline - timedelta(minutes=duration_minutes))}",
            f"DTEND:{_stamp(deadline)}",
            _fold(f"SUMMARY:{_escape(f'FPL ปิดจัดตัว GW{gw}')}"),
            _fold("DESCRIPTION:" + _escape(
                f"เส้นตายจัดตัว Gameweek {gw}\n"
                "ตรวจ: คนติดธงในตัวจริง / กัปตัน / ลำดับตัวสำรอง"
                + (f"\n{site_url}" if site_url else ""))),
            "TRANSP:TRANSPARENT",
        ]
        if site_url:
            lines.append(_fold(f"URL:{site_url}"))
        for hours in sorted(remind_hours, reverse=True):
            mins = int(round(hours * 60))
            lines += [
                "BEGIN:VALARM",
                "ACTION:DISPLAY",
                f"TRIGGER:-PT{mins}M",
                _fold(f"DESCRIPTION:{_escape(f'FPL GW{gw} ปิดในอีก {hours:g} ชม.')}"),
                "END:VALARM",
            ]
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def write(events: list[dict], out_dir: Path, **kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "deadlines.ics"
    path.write_text(build_ics(events, **kwargs), encoding="utf-8")
    return path
