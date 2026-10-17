"""Guard the public-data boundaries before a recommendation reaches the manager."""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from unittest.mock import Mock

import pandas as pd
import pytest

from fplbot import cli, optimize, report, state
from fplbot.config import Config
from fplbot.fetch import FPLClient, permanent_transfers, reconstruct_team, selling_prices
from test_fplbot import _cfg, _legal_squad, _synthetic_league


@pytest.mark.parametrize("fresh,expected", [(True, "network"), (False, "cache")])
def test_build_fresh_flag_reaches_the_real_cache(tmp_path, monkeypatch, fresh, expected):
    cfg = Config(raw={"entry": {"team_id": 1}, "planning": {"horizon": 1},
                      "output": {"data_dir": str(tmp_path)}})
    client = FPLClient(cfg)
    (client.snapshot_dir / "bootstrap.json").write_text('{"events": []}', encoding="utf-8")
    (client.snapshot_dir / "fixtures.json").write_text('[]', encoding="utf-8")
    response = Mock()
    response.json.side_effect = [{"events": []}, []]
    session = Mock()
    session.get.return_value = response
    monkeypatch.setattr("fplbot.fetch._session", lambda: session)
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    instances = []

    def factory(*args, **kwargs):
        instance = FPLClient(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(cli, "FPLClient", factory)
    # There is deliberately no upcoming GW: no generated report is touched.
    assert cli.build(SimpleNamespace(config=None, offline=False, fresh=fresh)) == 1
    assert instances[0].provenance["bootstrap"]["source"] == expected
    assert session.get.call_count == (2 if fresh else 0)


def test_failed_hold_cannot_turn_a_transfer_into_the_baseline(monkeypatch):
    players, ep = _synthetic_league()
    solve = Mock(side_effect=RuntimeError("no proven solution"))
    monkeypatch.setattr(optimize, "solve", solve)
    with pytest.raises(RuntimeError, match="holding baseline"):
        optimize.solve_scenarios(ep, players, _legal_squad(players), 1, 2, _cfg())
    assert solve.call_count == 1


def picks(ids, bank=None, chip=None):
    return {"picks": [{"element": p} for p in ids], "active_chip": chip,
            "entry_history": {} if bank is None else {"bank": bank}}


def transfer(event, out_id, in_id, sale=50, purchase=50, time="2026-01-01T01:00:00Z"):
    return {"event": event, "element_out": out_id, "element_in": in_id,
            "element_out_cost": sale, "element_in_cost": purchase, "time": time}


def test_free_hit_restores_permanent_squad_bank_and_purchase_prices():
    original = list(range(1, 16))
    client = Mock()
    client.entry_picks.side_effect = [picks(range(101, 116), 90, "freehit"),
                                      picks(original, 10)]
    history = {"chips": [{"event": 4, "name": "freehit"}]}
    moves = [transfer(3, 40, 1, purchase=50),
             transfer(4, 1, 101), transfer(4, 101, 1, purchase=60),
             transfer(5, 2, 20, sale=55, purchase=60)]
    squad, bank, used, live = reconstruct_team(client, history, moves, 5, 9.)
    assert squad == [1, 20] + list(range(3, 16))
    assert bank == .5 and used == 1
    assert live["active_chip"] == "freehit"
    players = pd.DataFrame([{"id": 1, "price": 6.4, "cost_change_start": 14}]).set_index("id")
    assert selling_prices([1], permanent_transfers(moves, history), players)[1] == 5.7
    assert [c.args[0] for c in client.entry_picks.call_args_list] == [4, 3]


@pytest.mark.parametrize("fault", ["missing_bank", "unmatched_transfer", "duplicate", "chip_disagreement"])
def test_unverifiable_team_is_rejected(fault):
    original = list(range(1, 16))
    history = {"chips": [{"event": 4, "name": "freehit"}]}
    client = Mock()
    base = picks(original, 10)
    moves = []
    if fault == "missing_bank":
        base["entry_history"] = {}
    if fault == "unmatched_transfer":
        moves = [transfer(5, 99, 20)]
    if fault == "duplicate":
        base["picks"][0]["element"] = 2
    if fault == "chip_disagreement":
        history = {"chips": []}
    client.entry_picks.side_effect = [picks(range(101, 116), 90, "freehit"), base]
    with pytest.raises(ValueError):
        reconstruct_team(client, history, moves, 5, 9.)


def test_normal_team_applies_chained_current_transfers_in_time_order():
    client = Mock()
    client.entry_picks.return_value = picks(range(1, 16), 10)
    rows = [transfer(5, 20, 21, 50, 55, "2026-01-01T02:00:00Z"),
            transfer(5, 1, 20, 50, 50)]
    squad, bank, count, _ = reconstruct_team(client, {}, rows, 5, 1.)
    assert squad[0] == 21 and bank == .5 and count == 2


def test_active_free_hit_does_not_publish_permanent_transfer_advice():
    client = Mock()
    client.entry_picks.return_value = picks(range(1, 16), 10)
    with pytest.raises(ValueError, match="Free Hit is active"):
        reconstruct_team(client, {"chips": [{"event": 5, "name": "freehit"}]}, [], 5, 1.)


def test_transfer_endpoint_failure_preserves_the_existing_report(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg.raw["output"] = {"site_dir": str(tmp_path)}
    old = tmp_path / "summary.json"
    old.write_text('previous report', encoding="utf-8")
    client = Mock()
    client.bootstrap.return_value = {"events": [{"id": 5, "is_next": True,
                                                  "deadline_time": "2026-10-18T10:00:00Z"}]}
    client.entry.return_value = {}
    client.entry_history.return_value = {}
    client.entry_transfers.side_effect = OSError("network down")
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli, "FPLClient", lambda *a, **k: client)
    render = Mock()
    monkeypatch.setattr(report, "render", render)
    assert cli.build(SimpleNamespace(config=None, offline=False, fresh=True)) == 2
    assert old.read_text(encoding="utf-8") == "previous report"
    render.assert_not_called()


@pytest.mark.parametrize("squad,expected_budget,allowed", [([1, 4], 6., False),
                                                         ([2, 4], 6., False),
                                                         ([2, 5], 7., True)])
def test_comparison_cash_uses_the_displayed_scenario(squad, expected_budget, allowed):
    players = pd.DataFrame([
        {"id": i, "name": str(i), "element_type": pos, "team": i,
         "team_name": str(i), "price": price, "p_available": 1., "p_start": 1.,
         "selected_by_percent": 0., "form": 0.}
        for i, pos, price in [(1, 2, 5.), (2, 2, 4.), (3, 2, 6.5),
                              (4, 3, 5.), (5, 3, 4.)]]).set_index("id")
    rows = pd.DataFrame([{"player_id": i, "gw": 5, "opponent": 1,
                          "is_home": True, "fdr": 2, "ep": float(i)} for i in players.index])
    advice = optimize.TransferAdvice(1, [optimize.TransferCandidate(1, 2, 0, 10, 10, 10)],
                                      True, False, 3, "test")
    data = report.player_comparisons(_cfg(), rows, players,
                                     pd.DataFrame({"short_name": ["T"]}, index=[1]),
                                     squad, [1, 4], 1., {1: 5., 4: 5.}, advice)
    defender = str(squad[0])
    assert data[defender]["budget"] == expected_budget
    assert any(r["id"] == 3 for r in data[defender]["rows"]) == allowed


@pytest.mark.parametrize("change", ["timestamp", "moves", "gw"])
def test_selection_rejects_a_changed_report_without_overwriting(tmp_path, change):
    summary = {"gw": 5, "built_at_iso": "old", "scenarios": [
        {"transfers": 1, "move_ids": [{"out": 1, "in": 2}], "moves": []}]}
    token = state.scenario_fingerprint(5, "old", summary["scenarios"][0]["move_ids"])
    if change == "timestamp":
        summary["built_at_iso"] = "new"
    elif change == "moves":
        summary["scenarios"][0]["move_ids"][0]["in"] = 3
    else:
        summary["gw"] = 6
    source, target = tmp_path / "summary.json", tmp_path / "selected.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    target.write_text("old selection", encoding="utf-8")
    with pytest.raises(ValueError, match="report changed"):
        state.select_scenario(source, target, 0, expected_fingerprint=token)
    assert target.read_text(encoding="utf-8") == "old selection"


def test_scenario_panels_and_saved_links_use_the_same_report(tmp_path):
    players, ep = _synthetic_league()
    players["selected_by_percent"] = 0.
    players["minutes"] = 900
    players["yellow_cards"] = 0
    cfg = _cfg(scenario_alternatives=1)
    cfg.raw["output"] = {"site_dir": str(tmp_path)}
    cfg.raw["notify"] = {"plan_request_url": "https://example.org/new?template=plan.yml"}
    squad = _legal_squad(players)
    scenarios = optimize.solve_scenarios(ep, players, squad, 2., 1, cfg)
    plan = next(s.plan for s in scenarios if s.recommended)
    rows = ep.rename_axis("player_id").reset_index().melt(
        id_vars="player_id", var_name="gw", value_name="ep")
    rows["team"] = rows.player_id.map(players.team)
    rows["opponent"] = 1
    rows["is_home"] = True
    rows["fdr"] = 2
    for col in ("exp_goals", "exp_assists", "ep_bonus"):
        rows[col] = 0.
    teams = pd.DataFrame({"id": range(1, 13),
                          "short_name": [f"T{i}" for i in range(1, 13)],
                          "name": [f"Team {i}" for i in range(1, 13)]}).set_index("id")
    advice = optimize.analyse_transfers(rows, players, squad, 2., 1, cfg)
    context = report.build_context(
        cfg=cfg, bootstrap={}, teams=teams, players=players, ep_rows=rows,
        ep_grid=ep, plan=plan, entry={}, current_squad=squad, free_transfers=1,
        bank=2., squad_value=98., event={"id": 1, "deadline_time": "2030-01-01T10:00:00Z"},
        transfer_advice=advice, selling_price=players.loc[squad].price.to_dict(),
        scenarios=scenarios)
    report.render(context, cfg)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for i, scenario in enumerate(scenarios):
        week = scenario.plan.gameweeks[0]
        data = context["scenario_comparisons"][str(i)]
        assert set(data) == {str(p) for p in week.squad}
        cash = 2. + sum(players.price[p] for p in week.sells) - sum(
            players.price[p] for p in week.buys)
        for pid in week.squad:
            assert data[str(pid)]["budget"] == pytest.approx(round(cash + players.price[pid], 1))
            assert all(r["price"] <= cash + players.price[pid] + 1e-9
                       for r in data[str(pid)]["rows"])
        title = parse_qs(urlparse(context["scenario_cards"][i]["select_url"]).query)["title"][0]
        token = title.split()[-1]
        assert token == summary["scenarios"][i]["fingerprint"]
        selected = state.select_scenario(summary_path, tmp_path / "selected.json", i,
                                         expected_fingerprint=token)
        assert selected["move_ids"] == summary["scenarios"][i]["move_ids"]


def test_cli_keeps_previous_report_when_hold_solver_fails(tmp_path, monkeypatch):
    players, ep = _synthetic_league()
    squad = _legal_squad(players)
    cfg = _cfg()
    cfg.raw["output"] = {"site_dir": str(tmp_path)}
    previous = tmp_path / "index.html"
    previous.write_text("previous dashboard", encoding="utf-8")
    client = Mock()
    client.offline = False
    client.fixtures.return_value = [{"event": 5}, {"event": 7}, {"event": 11}]
    client.bootstrap.return_value = {"events": [
        {"id": 5, "is_next": True, "deadline_time": "2030-01-01T10:00:00Z"}]}
    client.entry.return_value = {"last_deadline_bank": 20}
    client.entry_history.return_value = {}
    client.entry_picks.return_value = picks(squad)
    client.entry_transfers.return_value = []
    players["cost_change_start"] = 0.
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli, "FPLClient", lambda *a, **k: client)
    monkeypatch.setattr(cli.features, "build_teams", lambda *_: pd.DataFrame())
    monkeypatch.setattr(cli.features, "build_players", lambda *_: players)
    monkeypatch.setattr(cli.features, "minutes_shortlist", lambda *_: [])
    monkeypatch.setattr(cli.features, "apply_price_forecast", lambda p, _: p)
    schedule_builder = Mock(return_value=pd.DataFrame({"gw": [5, 7, 11]}))
    monkeypatch.setattr(cli.features, "build_schedule", schedule_builder)
    monkeypatch.setattr(cli.model, "expected_points", lambda *_: pd.DataFrame())
    per_gw = Mock(return_value=ep)
    monkeypatch.setattr(cli.model, "per_gameweek", per_gw)
    monkeypatch.setattr(optimize, "analyse_transfers", lambda *_ , **kw: None)
    monkeypatch.setattr(optimize, "rank_transfer_priorities", lambda *_, **kw: [])
    monkeypatch.setattr(optimize, "solve", Mock(side_effect=RuntimeError("timeout")))
    render = Mock()
    monkeypatch.setattr(report, "render", render)
    assert cli.build(SimpleNamespace(config=None, offline=False, fresh=True)) == 2
    assert previous.read_text(encoding="utf-8") == "previous dashboard"
    assert schedule_builder.call_args.args[-1] == 7  # all remaining GWs, not horizon+2
    assert per_gw.call_args.args[1] == [5, 6, 7]  # the global blank is retained
    render.assert_not_called()
