"""Verify decision units, numerical mixtures, persistence and bounded execution."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pulp
import pytest
import requests
from scipy.stats import poisson

from fplbot import backtest, calendar_feed, features, model, notify, optimize, report
from fplbot.config import Config
from fplbot.fetch import FPLClient
from fplbot.webapp import run_build
from test_fplbot import _cfg, _legal_squad, _synthetic_league, _advice_players, _advice_rows


def universe():
    players, ep = _synthetic_league()
    squad = _legal_squad(players)
    clubs = players.loc[squad].team.value_counts()
    incoming = next(p for p in players.index if p not in squad and
                    players.element_type[p] == 1 and clubs.get(players.team[p], 0) < 3)
    return players, ep, squad, squad[0], incoming


@pytest.mark.parametrize("gain,recommend", [(2.95, False), (3.0, True), (3.1, True)])
def test_five_fixture_gain_drives_decision_even_when_objective_disagrees(gain, recommend):
    players, _, squad, outgoing, incoming = universe()
    rows = _advice_rows({p: 2. for p in squad} | {incoming: 2 + gain / 5}, games=5)
    advice = optimize.analyse_transfers(rows, players, squad, 20, 1, _cfg())
    ep = pd.DataFrame(1., index=players.index, columns=[1])
    # A huge objective upgrade below +3 must not override the player threshold.
    ep.loc[incoming, 1] = 100. if not recommend else 1.
    scenarios = optimize.solve_scenarios(ep, players, squad, 20, 1,
                                        _cfg(scenario_alternatives=1), advice=advice)
    chosen = next(s for s in scenarios if s.recommended)
    assert chosen.transfers == int(recommend)
    if recommend:
        assert chosen.plan.gameweeks[0].moves == [(advice.out_id, incoming)]
        assert chosen.objective_gain < 3
    else:
        assert max(s.objective_gain for s in scenarios) > 3


def test_short_end_of_season_window_is_visible_but_not_a_five_match_endorsement():
    players = _advice_players()
    advice = optimize.analyse_transfers(_advice_rows({1: 0, 2: 20}, games=3),
                                        players, [1], .5, 1, _cfg())
    assert not advice.recommend
    assert advice.candidates[0].out_matches == 3
    assert len(advice.candidates[0].fixtures) == 3


def test_multiple_blanks_and_doubles_still_get_five_actual_fixtures():
    players = pd.DataFrame([{"id": 1, "team": 1}]).set_index("id")
    fixtures = [{"event": gw, "team_h": 1, "team_a": 2,
                 "team_h_difficulty": 2, "team_a_difficulty": 2,
                 "kickoff_time": f"2030-01-{i + 1:02d}T12:00:00Z"}
                for i, gw in enumerate([1, 5, 8, 8, 10, 11])]
    schedule = features.build_schedule(fixtures, players, 1, 11)
    schedule["ep"] = 1.
    totals, windows = optimize.outlook_by_player(schedule, 5)
    assert totals[1] == 5
    assert [f["gw"] for f in windows[1]] == [1, 5, 8, 8, 10]
    grid = model.per_gameweek(schedule, list(range(1, 12)))
    assert grid.loc[1, 2] == 0 and grid.loc[1, 8] == 2


def test_global_blank_retains_free_transfer_rollover():
    players, _, squad, _, _ = universe()
    ep = pd.DataFrame(1., index=players.index, columns=[1, 2, 3])
    ep[2] = 0.
    plan = optimize.solve(ep, players, squad, 2, 2, _cfg(transfer_friction=100))
    assert [w.gw for w in plan.gameweeks] == [1, 2, 3]
    assert [w.free_transfers_before for w in plan.gameweeks] == [2, 3, 4]


@pytest.mark.parametrize("availability", [0., .5, 1.])
def test_dc_averages_the_playing_state_before_weighting_availability(availability):
    actual = model.defensive_contribution_probability(
        np.array([12.]), np.array([10.]), np.array([availability]),
        np.array([availability]), np.array([90 * availability]), 14.)
    assert actual[0] == pytest.approx(availability * poisson.sf(9, 12))


def test_dc_handles_keeper_and_start_cameo_mixture():
    actual = model.defensive_contribution_probability(
        np.array([12., 12.]), np.array([10., np.nan]), np.array([.5, 1.]),
        np.array([.75, 1.]), np.array([48.5, 90.]), 14.)
    assert actual[0] == pytest.approx(.5 * poisson.sf(9, 12) + .25 * poisson.sf(9, 12 * 14 / 90))
    assert actual[1] == 0


@pytest.mark.parametrize("mode", ["balanced", "template", "differential"])
def test_legacy_ownership_modes_cannot_modify_points(mode):
    frame = pd.DataFrame({"ep": [5., 5.], "selected_by_percent": [0., 100.]})
    result = model._apply_strategy(frame.copy(), Config(raw={"strategy": {
        "mode": mode, "differential_weight": 100}}))
    assert result.ep.tolist() == [5., 5.]


def test_sold_then_repurchased_player_uses_new_cost_basis():
    players, _, squad, outgoing, incoming = universe()
    clubs = players.loc[squad].team.value_counts()
    target = next(p for p in players.index if p not in squad and p != incoming
                  and players.element_type[p] == 1 and clubs.get(players.team[p], 0) < 3)
    players.loc[outgoing, "price"] = 6.
    players.loc[incoming, "price"] = 4.
    players.loc[target, "price"] = 6.
    ep = pd.DataFrame(0., index=players.index, columns=[1, 2, 3])
    ep.loc[incoming, 1] = 100.
    ep.loc[outgoing, 2] = 100.
    ep.loc[target, 3] = 200.
    plan = optimize.solve(ep, players, squad, .5, 1,
                           _cfg(max_transfers_per_gw=1),
                           selling_price={outgoing: 5.5},
                           locked=[p for p in squad if p != outgoing],
                           forced_first_move=(outgoing, incoming))
    assert [w.moves for w in plan.gameweeks] == [[(outgoing, incoming)],
                                               [(incoming, outgoing)], [(outgoing, target)]]


def test_selected_plan_below_threshold_is_preserved_without_becoming_recommended():
    players, _, squad, outgoing, incoming = universe()
    ep = pd.DataFrame(1., index=players.index, columns=[1])
    advice = optimize.TransferAdvice(None, [], False, False, 3., "hold")
    scenarios = optimize.solve_scenarios(ep, players, squad, 20, 1, _cfg(),
                                        advice=advice, selected_moves=[(outgoing, incoming)])
    assert next(s for s in scenarios if s.recommended).transfers == 0
    selected = next(s for s in scenarios if s.selected)
    assert selected.plan.gameweeks[0].moves == [(outgoing, incoming)]
    assert not selected.recommended


def test_total_solver_budget_prevents_an_unbounded_scenario_batch(monkeypatch):
    players, ep, squad, _, _ = universe()
    clock = [0.]
    monkeypatch.setattr(optimize.time, "monotonic", lambda: clock[0])
    limits = []
    def fake_solve(*args, **kwargs):
        limits.append(kwargs["time_limit"])
        clock[0] += 10
        count = kwargs["exact_first_moves"]
        week = optimize.GameweekPlan(1, squad, squad[:11], squad[11:], squad[0], squad[1],
                                     buys=list(range(100, 100 + count)))
        return optimize.Plan([week], "Optimal", 0, 100)
    monkeypatch.setattr(optimize, "solve", fake_solve)
    scenarios = optimize.solve_scenarios(ep, players, squad, 2, 5,
                                        _cfg(scenario_time_budget=15, solver_time_limit=180))
    assert limits == [15., 5.]
    assert [s.transfers for s in scenarios] == [0, 1]
    assert scenarios[0].plan.notes


def test_feasible_incumbent_is_not_reported_as_proven_optimal(monkeypatch):
    players, ep, squad, _, _ = universe()
    real = pulp.LpProblem.solve
    def timed_out(problem, *args, **kwargs):
        result = real(problem, *args, **kwargs)
        problem.sol_status = pulp.LpSolutionIntegerFeasible
        return result
    monkeypatch.setattr(pulp.LpProblem, "solve", timed_out)
    plan = optimize.solve(ep, players, squad, 2, 1, _cfg())
    assert plan.status == "Feasible" and plan.notes


def test_refresh_deadline_covers_the_entire_subprocess(tmp_path):
    result = run_build([sys.executable, "-c", "import time; time.sleep(30)"],
                        str(tmp_path), .1)
    assert result.returncode == 124
    assert "หมดเวลารวม" in result.stderr


def test_refresh_progress_reports_child_output(tmp_path):
    lines = []
    result = run_build([sys.executable, "-c", "print('building features', flush=True)"],
                        str(tmp_path), 10, lines.append)
    assert result.returncode == 0 and "building features" in "".join(lines)


def test_oldest_decision_input_cannot_be_hidden_by_live_scores():
    timestamp, sources = report.decision_freshness({
        "bootstrap": {"fetched_at": "2026-10-16T10:00:00+00:00", "source": "cache"},
        "entry": {"fetched_at": "2026-10-17T10:00:00+00:00", "source": "network"},
        "live_gw7": {"fetched_at": "2026-10-17T11:00:00+00:00", "source": "network"},
    }, "unknown")
    assert timestamp == "2026-10-16T10:00:00+00:00"
    assert len(sources) == 2


def test_calendar_alarm_is_exactly_24_hours_before_deadline_and_keeps_uid():
    deadline = datetime.now(timezone.utc) + timedelta(days=5)
    event = {"id": 7, "deadline_time": deadline.isoformat(), "finished": False}
    def lines(ev):
        return calendar_feed.build_ics([ev]).splitlines()
    old = lines(event)
    end = datetime.strptime(next(s.split(":", 1)[1] for s in old if s.startswith("DTEND:")),
                             "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    assert "TRIGGER;RELATED=END:-PT1440M" in old
    assert abs(((end - timedelta(minutes=1440)) - (deadline - timedelta(hours=24))).total_seconds()) < 1
    event["deadline_time"] = (deadline + timedelta(hours=1)).isoformat()
    assert next(s for s in old if s.startswith("UID:")) == next(s for s in lines(event) if s.startswith("UID:"))


def test_disabled_telegram_preview_does_not_consume_delivery_marker(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg.raw["output"] = {"data_dir": str(tmp_path), "site_dir": str(tmp_path)}
    cfg.raw["notify"] = {"remind_hours_before": [24], "telegram": {"enabled": False}}
    summary = {"gw": 1, "deadline": (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()}
    monkeypatch.setattr(notify, "_summary", lambda _: summary)
    monkeypatch.setattr(notify, "compose", lambda *_: "preview")
    assert notify.send_if_due(cfg) == 0
    assert not notify._marker_path(cfg, 1, 24).exists()


def test_notification_network_failure_does_not_leak_token(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(notify.requests, "post", Mock(side_effect=requests.ConnectionError(
        "https://api.telegram.org/botprivate-test-token/sendMessage")))
    assert notify.send_telegram("test") is False
    assert "private-test-token" not in caplog.text


def test_cached_entry_cannot_be_reused_for_another_manager(tmp_path):
    cfg = Config(raw={"entry": {"team_id": 2}, "planning": {"horizon": 1},
                      "output": {"data_dir": str(tmp_path)}})
    client = FPLClient(cfg)
    (client.snapshot_dir / "entry.json").write_text('{"id": 1}', encoding="utf-8")
    response = Mock()
    response.json.return_value = {"id": 2}
    client.session.get = Mock(return_value=response)
    assert client.entry()["id"] == 2
    assert (client.snapshot_dir / "entry_entry2.json").exists()


def test_projection_contains_reproducible_metadata_and_survives_reload(tmp_path):
    cfg = _cfg()
    cfg.raw["output"] = {"data_dir": str(tmp_path), "site_dir": str(tmp_path)}
    players = _advice_players()
    plan = SimpleNamespace(gameweeks=[SimpleNamespace(squad=[1], captain=1, vice=1)])
    advice = SimpleNamespace(candidates=[])
    event = {"id": 1, "deadline_time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}
    source = {"bootstrap": {"fetched_at": datetime.now(timezone.utc).isoformat(), "source": "network"}}
    path = backtest.write_projection(cfg, event, players, _advice_rows({1: 2}), plan, advice,
                                     model_version="test", provenance=source)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provenance"] == source and len(data["config_hash"]) == 64
    assert data["season"] and data["schema_version"] == 2
    (tmp_path / "summary.json").write_text(json.dumps({"gw": 1, "deadline": event["deadline_time"],
                                                      "model_version": "test"}), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("projection_check", "scripts/check_projection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_config = lambda: cfg
    module.check()


def test_backtest_keeps_versions_and_configs_in_separate_groups(tmp_path):
    cfg = _cfg()
    cfg.raw["output"] = {"data_dir": str(tmp_path)}
    folder = tmp_path / "projections"
    folder.mkdir()
    now = datetime.now(timezone.utc)
    for version in ("one", "two"):
        payload = {"gw": 1, "built_at": (now - timedelta(days=2)).isoformat(),
                   "deadline": (now - timedelta(days=1)).isoformat(),
                   "model_version": version, "config_hash": version,
                   "players": [{"player_id": 1, "position": "MID", "model_ep": 2., "fpl_ep_next": 3.}]}
        (folder / f"gw1-{version}.json").write_text(json.dumps(payload), encoding="utf-8")
    result = backtest.evaluate_projections(cfg, lambda _: {"history": [{"round": 1, "total_points": 2}]})
    assert {g["model_version"] for g in result["groups"]} == {"one", "two"}


def test_full_ep_pipeline_and_candidate_order_ignore_ownership():
    players = _advice_players()
    values = {"minutes": 900., "expected_goals_per_90": .2, "goals_scored": 2.,
              "expected_assists_per_90": .1, "assists": 1., "defensive_contribution_per_90": 8.,
              "bps": 200., "saves_per_90": 0., "yellow_cards": 1.,
              "expected_goals_conceded_per_90": 1., "is_penalty_taker": False,
              "is_set_piece_taker": False, "exp_minutes": 90., "p_60plus": 1.,
              "p_any_minutes": 1., "position": "MID", "form": 0.}
    for key, value in values.items():
        players[key] = value
    teams = pd.DataFrame({"id": range(1, 7), "attack": 1000., "defence": 1000.,
                          "strength_defence_away": 1000., "strength_defence_home": 1000.,
                          "strength_attack_away": 1000., "strength_attack_home": 1000.}).set_index("id")
    schedule = _advice_rows({p: 0 for p in players.index}, games=5).drop(columns="ep")
    schedule["opponent"] = 1
    first = model.expected_points(players, teams, schedule, _cfg())
    cfg = _cfg()
    cfg.raw["strategy"].update(mode="differential", differential_weight=100)
    players["selected_by_percent"] = 100 - pd.to_numeric(players.selected_by_percent)
    second = model.expected_points(players, teams, schedule, cfg)
    assert first.ep.tolist() == second.ep.tolist()
    a = optimize.analyse_transfers(first, players, [1], .5, 1, cfg)
    b = optimize.analyse_transfers(second, players, [1], .5, 1, cfg)
    assert [c.in_id for c in a.candidates] == [c.in_id for c in b.candidates]


def test_every_banked_ft_count_gets_a_complete_scenario():
    players, _, squad, _, _ = universe()
    ep = pd.DataFrame(1., index=players.index, columns=[1])
    scenarios = optimize.solve_scenarios(ep, players, squad, 20, 5,
                                        _cfg(scenario_alternatives=1, scenario_time_budget=60))
    assert [s.transfers for s in scenarios] == list(range(6))
    assert [len(s.plan.gameweeks[0].moves) for s in scenarios] == list(range(6))
    assert sum(s.recommended for s in scenarios) == 1


def test_non_integral_incumbent_is_never_published(monkeypatch):
    players, ep, squad, _, _ = universe()
    real = pulp.LpProblem.solve
    def corrupt(problem, *args, **kwargs):
        result = real(problem, *args, **kwargs)
        next(v for v in problem.variables() if v.cat == "Integer").varValue = .25
        return result
    monkeypatch.setattr(pulp.LpProblem, "solve", corrupt)
    with pytest.raises(RuntimeError, match="invalid or non-integral"):
        optimize.solve(ep, players, squad, 2, 1, _cfg())


def test_telegram_http_200_without_confirmed_api_success_is_not_delivery(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    response = Mock(status_code=200, text="rejected")
    response.json.return_value = {"ok": False}
    monkeypatch.setattr(notify.requests, "post", Mock(return_value=response))
    assert notify.send_telegram("test") is False


def test_backtest_does_not_match_old_season_projection_to_current_round(tmp_path):
    cfg = _cfg()
    cfg.raw["output"] = {"data_dir": str(tmp_path)}
    folder = tmp_path / "projections"
    folder.mkdir()
    payload = {"gw": 1, "built_at": "2024-08-01T10:00:00+00:00",
               "deadline": "2024-08-02T10:00:00+00:00", "season": "2024",
               "players": [{"player_id": 1, "position": "MID", "model_ep": 2., "fpl_ep_next": 3.}]}
    (folder / "gw1.json").write_text(json.dumps(payload), encoding="utf-8")
    loader = Mock(return_value={"history": [{"round": 1, "total_points": 20}]})
    result = backtest.evaluate_projections(cfg, loader, season="2026")
    assert result["players"] == 0
    loader.assert_not_called()
