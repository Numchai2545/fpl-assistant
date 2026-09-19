"""Command line entry point.

    python -m fplbot build          fetch, model, optimise, write the dashboard
    python -m fplbot build --offline  rebuild from today's cached snapshot
    python -m fplbot notify         send the alert if the deadline is close
    python -m fplbot check          print the next deadline and exit
    python -m fplbot backtest       evaluate frozen projections after matches
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from . import backtest, calendar_feed, features, model, optimize, report, state
from .config import load_config, selling_fee, verify_scoring
from .fetch import (FPLClient, apply_event_transfer_state, bank_and_value,
                    current_gameweek, deadline_utc, free_transfers, next_gameweek,
                    selling_prices)

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
        log.error("could not read GW%s picks (%s) — keeping the previous report",
                  last_finished, exc)
        return 2

    try:
        transfers = client.entry_transfers()
    except Exception as exc:
        log.warning("could not read your transfer history (%s)", exc)
        transfers = []
    current_squad, bank, moves_already_made = apply_event_transfer_state(
        current_squad, bank, transfers, gw)
    if moves_already_made:
        ft = max(0, ft - moves_already_made)
        log.info("applied %d transfer(s) already made in GW%s; current bank %.1fm",
                 moves_already_made, gw, bank)

    log.info("building features")
    teams = features.build_teams(bootstrap)
    players = features.build_players(bootstrap, teams, fixtures)
    shortlist = features.minutes_shortlist(
        players, current_squad,
        int(cfg.get("planning", "minutes_shortlist", default=30)))
    if client.offline:
        shortlist = [pid for pid in shortlist
                     if (client.snapshot_dir / f"element_{pid}.json").exists()]
    histories: dict[int, dict] = {}
    for pid in shortlist:
        try:
            histories[pid] = client.element_summary(pid)
        except Exception as exc:
            log.debug("minutes history unavailable for element %s: %s", pid, exc)
    if histories:
        log.info("using recent match histories for %d/%d shortlisted players",
                 len(histories), len(shortlist))
        players = features.apply_recent_minutes(players, histories, cfg)
    else:
        log.warning("recent match histories unavailable — using season minutes fallback")
    players = features.apply_price_forecast(players, cfg)
    horizon = cfg.horizon
    outlook_matches = int(cfg.get("planning", "outlook_matches", default=5))
    # Pull a little beyond the optimiser horizon so a blank gameweek does not
    # leave a player's five-match buying window one fixture short.
    schedule = features.build_schedule(
        fixtures, players, gw, max(horizon, outlook_matches + 2))
    horizon_gws = sorted(schedule.gw.unique().tolist())[:horizon]
    if not horizon_gws:
        log.error("no fixtures scheduled from GW%s onward — nothing to plan", gw)
        return 1

    log.info("scoring %s players over GW%s-%s", len(players), horizon_gws[0], horizon_gws[-1])
    ep_rows = model.expected_points(players, teams, schedule, cfg)
    ep_grid = model.per_gameweek(ep_rows, horizon_gws)

    sell_at = selling_prices(current_squad, transfers, players, selling_fee(bootstrap))
    if sell_at:
        gap = sum(players.price.get(p, 0.0) for p in sell_at) - sum(sell_at.values())
        log.info("selling prices reconstructed for %d players — %.1fm below market value",
                 len(sell_at), gap)
    else:
        log.warning("no selling prices available — budget uses market value, "
                    "which is optimistic if your squad has risen in price")

    locked = [int(p) for p in (cfg.get("strategy", "never_sell", default=[]) or [])]
    advice = optimize.analyse_transfers(
        ep_rows, players, current_squad, bank, ft, cfg, selling_price=sell_at)
    log.info("solving hold and multi-transfer scenarios")
    scenarios = optimize.solve_scenarios(
        ep_grid, players, current_squad, bank, ft, cfg,
        selling_price=sell_at, locked=locked,
    )
    selected = state.load_selected(cfg.site_dir / "selected-plan.json", gw)
    if selected:
        wanted = sorted((int(m["out"]), int(m["in"])) for m in selected.get("move_ids", []))
        matched = next((s for s in scenarios
                        if sorted(s.plan.gameweeks[0].moves) == wanted), None)
        if matched:
            for scenario in scenarios:
                scenario.recommended = scenario is matched
            log.info("kept the explicitly selected GW%s scenario", gw)
        else:
            log.warning("saved GW%s scenario no longer matches a legal optimal plan", gw)
    plan = next(s.plan for s in scenarios if s.recommended)
    log.info("solver: %s  objective %.1f  budget %.1fm",
             plan.status, plan.objective, plan.budget)

    live = None
    live_gw = current_gameweek(bootstrap)
    if live_gw:
        try:
            payload = client.event_live(live_gw)
            current_event = next(
                (e for e in bootstrap["events"] if int(e["id"]) == live_gw), {})
            points = {int(row["id"]): float(row.get("stats", {}).get("total_points", 0))
                      for row in payload.get("elements", [])}
            live_points = sum(
                points.get(int(p["element"]), 0.0) * int(p.get("multiplier", 1))
                for p in (picks or {}).get("picks", []))
            live = {"gw": live_gw, "provisional": not bool(current_event.get("finished")),
                    "players_updated": len(payload.get("elements", [])),
                    "team_points": int(live_points)}
        except Exception as exc:
            log.warning("live GW%s points unavailable (%s)", live_gw, exc)

    ctx = report.build_context(
        cfg=cfg, bootstrap=bootstrap, teams=teams, players=players,
        ep_rows=ep_rows, ep_grid=ep_grid, plan=plan, entry=entry,
        current_squad=current_squad, free_transfers=ft,
        bank=bank, squad_value=squad_value, event=event,
        transfer_advice=advice, selling_price=sell_at, scenarios=scenarios,
        provenance=client.provenance, live=live,
    )
    path = report.render(ctx, cfg)
    log.info("dashboard written to %s", path)
    projection = backtest.write_projection(
        cfg, event, players, ep_rows, plan, advice,
        model_version=report.MODEL_VERSION)
    if projection:
        log.info("pre-deadline projection frozen at %s", projection)

    ics = calendar_feed.write(
        bootstrap["events"], cfg.site_dir,
        site_url=cfg.get("notify", "site_url", default="") or "",
        calendar_name=cfg.get("output", "title", default="FPL"),
        remind_hours=[float(h) for h in cfg.get(
            "notify", "remind_hours_before", default=[24])],
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


def serve(args) -> int:
    from .webapp import serve as serve_dashboard
    cfg = load_config(args.config)
    return serve_dashboard(cfg.site_dir, cfg.path.parent, port=args.port,
                           config_path=args.config, open_browser=not args.no_browser)


def run_backtest(args) -> int:
    cfg = load_config(args.config)
    client = FPLClient(cfg, offline=args.offline,
                       force_refresh=getattr(args, "fresh", False))
    result = backtest.evaluate_projections(
        cfg, client.element_summary, fixtures=client.fixtures())
    if result["gameweeks"] == 0:
        print("ยังไม่มี projection ที่ deadline ผ่านแล้วสำหรับ backtest")
        return 0
    path = backtest.write_report(cfg, result)
    print(backtest.format_summary(result))
    print(f"\n  report: {path}")
    return 0


def select_plan(args) -> int:
    cfg = load_config(args.config)
    payload = state.select_scenario(
        cfg.site_dir / "summary.json", cfg.site_dir / "selected-plan.json",
        args.scenario)
    print(f"selected GW{payload['gw']} scenario {args.scenario}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fplbot", description="FPL assistant")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="fetch, model and write the dashboard")
    p_build.add_argument("--offline", action="store_true",
                         help="reuse today's cached snapshot instead of calling the API")
    p_build.add_argument("--fresh", action="store_true",
                         help="bypass cache and fetch a new public FPL snapshot")
    p_build.set_defaults(func=build)

    p_notify = sub.add_parser("notify", help="send the deadline alert if it is due")
    p_notify.add_argument("--force", action="store_true", help="send regardless of timing")
    p_notify.set_defaults(func=notify)

    p_check = sub.add_parser("check", help="print the next deadline")
    p_check.add_argument("--offline", action="store_true")
    p_check.set_defaults(func=check)

    p_serve = sub.add_parser("serve", help="open the local dashboard in a browser")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    p_serve.set_defaults(func=serve)

    p_backtest = sub.add_parser(
        "backtest", help="score frozen pre-deadline projections against outcomes")
    p_backtest.add_argument("--offline", action="store_true",
                            help="use cached element histories only")
    p_backtest.set_defaults(func=run_backtest)

    p_select = sub.add_parser("select-plan", help="persist a reviewed dashboard scenario")
    p_select.add_argument("--scenario", type=int, required=True)
    p_select.set_defaults(func=select_plan)

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
