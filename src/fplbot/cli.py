"""Command line entry point.

    python -m fplbot build          fetch, model, optimise, write the dashboard
    python -m fplbot build --offline  rebuild from today's cached snapshot
    python -m fplbot notify         send the alert if the deadline is close
    python -m fplbot check          print the next deadline and exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from . import calendar_feed, features, model, optimize, report
from .config import load_config, selling_fee, verify_scoring
from .fetch import (FPLClient, bank_and_value, deadline_utc, free_transfers,
                    next_gameweek, selling_prices)

log = logging.getLogger("fplbot")


def _setup_console() -> None:
    """Make stdout able to carry Thai on a stock Windows console.

    The default code page there is cp1252, which cannot encode Thai at all, so
    printing the recommendation raises UnicodeEncodeError and takes the whole
    run down *after* the dashboard has already been written successfully.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def build(args) -> int:
    cfg = load_config(args.config)
    client = FPLClient(cfg, offline=args.offline)

    log.info("fetching league data")
    bootstrap = client.bootstrap()
    fixtures = client.fixtures()

    # A scoring rule that changed under us would silently poison every expected
    # point, and nothing downstream would fail. Check it on every build.
    for problem in verify_scoring(bootstrap):
        log.warning("SCORING RULE MISMATCH — %s", problem)

    event = next_gameweek(bootstrap)
    if event is None:
        log.error("no upcoming gameweek found — season over?")
        return 1
    gw = int(event["id"])
    log.info("planning for GW%s (deadline %s)", gw, event["deadline_time"])

    log.info("fetching your team (%s)", cfg.team_id)
    entry = client.entry()
    history = client.entry_history()
    entry["__history__"] = history
    bank, squad_value = bank_and_value(entry)
    ft = free_transfers(history)

    last_finished = gw - 1
    current_squad: list[int] = []
    picks = None
    try:
        picks = client.entry_picks(last_finished)
        current_squad = [int(p["element"]) for p in picks["picks"]]
    except Exception as exc:  # first gameweek, or a private entry
        log.warning("could not read GW%s picks (%s) — planning from scratch",
                    last_finished, exc)

    log.info("building features")
    teams = features.build_teams(bootstrap)
    players = features.build_players(bootstrap, teams, fixtures)
    horizon = cfg.horizon
    schedule = features.build_schedule(fixtures, players, gw, horizon)
    horizon_gws = sorted(schedule.gw.unique().tolist())[:horizon]
    if not horizon_gws:
        log.error("no fixtures scheduled from GW%s onward — nothing to plan", gw)
        return 1

    log.info("scoring %s players over GW%s-%s", len(players), horizon_gws[0], horizon_gws[-1])
    ep_rows = model.expected_points(players, teams, schedule, cfg)
    ep_grid = model.per_gameweek(ep_rows, horizon_gws)

    try:
        transfers = client.entry_transfers()
    except Exception as exc:
        log.warning("could not read your transfer history (%s)", exc)
        transfers = []
    sell_at = selling_prices(current_squad, transfers, players, selling_fee(bootstrap))
    if sell_at:
        gap = sum(players.price.get(p, 0.0) for p in sell_at) - sum(sell_at.values())
        log.info("selling prices reconstructed for %d players — %.1fm below market value",
                 len(sell_at), gap)
    else:
        log.warning("no selling prices available — budget uses market value, "
                    "which is optimistic if your squad has risen in price")

    locked = [int(p) for p in (cfg.get("strategy", "never_sell", default=[]) or [])]
    log.info("solving the transfer plan")
    plan = optimize.solve(ep_grid, players, current_squad, bank, ft, cfg,
                          selling_price=sell_at, locked=locked)
    log.info("solver: %s  objective %.1f  budget %.1fm",
             plan.status, plan.objective, plan.budget)

    ctx = report.build_context(
        cfg=cfg, bootstrap=bootstrap, teams=teams, players=players,
        ep_rows=ep_rows, ep_grid=ep_grid, plan=plan, entry=entry,
        current_squad=current_squad, free_transfers=ft,
        bank=bank, squad_value=squad_value, event=event,
    )
    path = report.render(ctx, cfg)
    log.info("dashboard written to %s", path)

    ics = calendar_feed.write(
        bootstrap["events"], cfg.site_dir,
        site_url=cfg.get("notify", "site_url", default="") or "",
        calendar_name=cfg.get("output", "title", default="FPL"),
        remind_hours=[float(h) for h in cfg.get(
            "notify", "remind_hours_before", default=[48, 24, 3])],
    )
    log.info("deadline calendar written to %s", ics)
    print(f"\n  {ctx['action']['headline']}\n  {ctx['action']['detail']}\n")
    print(f"  open: {path}")
    return 0


def notify(args) -> int:
    from .notify import send_if_due
    cfg = load_config(args.config)
    return send_if_due(cfg, force=args.force)


def check(args) -> int:
    cfg = load_config(args.config)
    client = FPLClient(cfg, offline=args.offline)
    event = next_gameweek(client.bootstrap())
    if not event:
        print("no upcoming gameweek")
        return 1
    dl = deadline_utc(event)
    hours = (dl - datetime.now(timezone.utc)).total_seconds() / 3600
    print(f"GW{event['id']} deadline {dl.isoformat()}  ({hours:.1f} hours away)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fplbot", description="FPL assistant")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="fetch, model and write the dashboard")
    p_build.add_argument("--offline", action="store_true",
                         help="reuse today's cached snapshot instead of calling the API")
    p_build.set_defaults(func=build)

    p_notify = sub.add_parser("notify", help="send the deadline alert if it is due")
    p_notify.add_argument("--force", action="store_true", help="send regardless of timing")
    p_notify.set_defaults(func=notify)

    p_check = sub.add_parser("check", help="print the next deadline")
    p_check.add_argument("--offline", action="store_true")
    p_check.set_defaults(func=check)

    args = parser.parse_args(argv)
    _setup_console()
    _setup_logging(args.verbose)
    pd.set_option("display.width", 160)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
