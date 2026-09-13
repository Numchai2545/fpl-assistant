"""Tests for the parts where being wrong is silent.

The model produces numbers for every player no matter what, so an error in the
scoring constants, the shrinkage, or the transfer arithmetic does not raise —
it just makes every recommendation quietly worse. These cover the pure
functions that carry that risk, plus the optimiser constraints, which are the
only place a plan can become one FPL would refuse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplbot import features, model, notify, optimize
from fplbot.config import (CLEAN_SHEET_POINTS, GOAL_POINTS, MAX_PER_CLUB,
                           MAX_SAVED_TRANSFERS, SQUAD_BY_POSITION, SQUAD_SIZE,
                           XI_SIZE, Config, verify_scoring)
from fplbot.fetch import free_transfers, purchase_prices, selling_prices


# --------------------------------------------------------------- shrinkage
class TestShrink:
    def test_no_minutes_returns_the_prior(self):
        """A player who has not played tells us nothing about his own rate."""
        out = features.shrink(pd.Series([9.0]), pd.Series([0.0]), 0.5, prior_minutes=360)
        assert out.iloc[0] == pytest.approx(0.5)

    def test_large_sample_approaches_the_observed_rate(self):
        out = features.shrink(pd.Series([1.0]), pd.Series([36_000.0]), 0.0, prior_minutes=360)
        assert out.iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_prior_minutes_is_the_half_way_point(self):
        """At exactly K minutes the observed rate and the prior weigh equally."""
        out = features.shrink(pd.Series([1.0]), pd.Series([360.0]), 0.0, prior_minutes=360)
        assert out.iloc[0] == pytest.approx(0.5)

    def test_is_monotone_in_minutes(self):
        minutes = pd.Series([0.0, 90.0, 360.0, 900.0, 3000.0])
        out = features.shrink(pd.Series([1.0] * 5), minutes, 0.0, prior_minutes=360)
        assert list(out) == sorted(out), "more minutes must mean more weight on the observation"

    def test_nan_observation_does_not_propagate(self):
        out = features.shrink(pd.Series([np.nan]), pd.Series([900.0]), 0.4, prior_minutes=360)
        assert not np.isnan(out.iloc[0])


class TestPositionalPrior:
    def _frame(self, n_per_pos: int = 30) -> pd.DataFrame:
        rows = []
        for pos in (1, 2, 3, 4):
            for i in range(n_per_pos):
                rows.append({"element_type": pos, "minutes": 900.0, "rate": float(pos)})
        return pd.DataFrame(rows)

    def test_uses_the_median_of_the_same_position(self):
        prior = features.positional_prior(self._frame(), "rate")
        by_pos = prior.groupby(self._frame().element_type).first()
        assert by_pos.to_dict() == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}

    def test_falls_back_when_nobody_has_reached_the_minutes_bar(self):
        """Early in the season the 450-minute pool is empty; it must still return."""
        df = self._frame()
        df["minutes"] = 100.0
        prior = features.positional_prior(df, "rate")
        assert prior.notna().all()


# ------------------------------------------------------------ goals conceded
class TestConcedePenalty:
    def test_zero_expected_goals_means_no_penalty(self):
        assert model._expected_concede_penalty(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_matches_a_direct_poisson_calculation(self):
        """E[floor(k/2)] under Poisson(2.0), summed by hand."""
        from scipy.stats import poisson
        lam = 2.0
        expected = sum(poisson.pmf(k, lam) * (k // 2) for k in range(0, 40))
        got = model._expected_concede_penalty(np.array([lam]))[0]
        assert got == pytest.approx(expected, abs=1e-3)

    def test_rises_with_expected_goals_conceded(self):
        out = model._expected_concede_penalty(np.array([0.5, 1.0, 2.0, 3.5]))
        assert list(out) == sorted(out)

    def test_never_negative(self):
        out = model._expected_concede_penalty(np.linspace(0.01, 5.0, 40))
        assert (out >= 0).all()


# ------------------------------------------------------------ free transfers
class TestFreeTransfers:
    def _history(self, rows, chips=None):
        return {"current": [
            {"event": gw, "event_transfers": made, "event_transfers_cost": cost}
            for gw, made, cost in rows
        ], "chips": chips or []}

    def test_starts_at_one(self):
        assert free_transfers({"current": [], "chips": []}) == 1

    def test_banks_an_unused_transfer(self):
        assert free_transfers(self._history([(1, 0, 0)])) == 2

    def test_spending_it_resets_to_one(self):
        assert free_transfers(self._history([(1, 0, 0), (2, 2, 0)])) == 1

    def test_caps_at_the_maximum(self):
        rows = [(gw, 0, 0) for gw in range(1, 12)]
        assert free_transfers(self._history(rows)) == MAX_SAVED_TRANSFERS

    def test_never_drops_below_one(self):
        assert free_transfers(self._history([(1, 5, 16)])) == 1

    def test_paid_transfers_do_not_consume_free_ones(self):
        """Two moves with a -4 means one was free and one was paid for."""
        # GW1 banks to 2. GW2 makes 2 moves costing 4: one free, one paid.
        # So one free transfer was used, leaving 2 - 1 + 1 = 2.
        assert free_transfers(self._history([(1, 0, 0), (2, 2, 4)])) == 2

    def test_matches_the_real_entry_history(self):
        """Gnum United GW1-4: 0, 1, 1, 0 transfers and no chips -> 3 for GW5."""
        history = self._history([(1, 0, 0), (2, 1, 0), (3, 1, 0), (4, 0, 0)])
        assert free_transfers(history) == 3

    def test_wildcard_week_does_not_burn_the_bank(self):
        """Unlimited transfers under a chip must not reset the count."""
        rows = [(1, 0, 0), (2, 0, 0), (3, 14, 0)]
        chips = [{"event": 3, "name": "wildcard"}]
        with_chip = free_transfers(self._history(rows, chips))
        without = free_transfers(self._history(rows))
        assert with_chip > without
        assert with_chip == MAX_SAVED_TRANSFERS - 1


# --------------------------------------------------------------- sell prices
class TestSellingPrices:
    def _players(self, prices: dict[int, float], drift: dict[int, float] | None = None):
        drift = drift or {}
        return pd.DataFrame(
            {"price": prices,
             "cost_change_start": {p: drift.get(p, 0.0) * 10 for p in prices}}
        ).rename_axis("id")

    def test_half_of_a_rise_is_kept_by_fpl(self):
        players = self._players({1: 8.0})
        transfers = [{"element_in": 1, "element_in_cost": 74, "event": 2}]
        # Bought 7.4, now 8.0: a 0.6 rise, half of it (0.3) is yours.
        assert selling_prices([1], transfers, players)[1] == pytest.approx(7.7)

    def test_odd_tenths_round_down(self):
        players = self._players({1: 7.7})
        transfers = [{"element_in": 1, "element_in_cost": 74, "event": 2}]
        # 0.3 rise -> half is 0.15 -> FPL rounds down to 0.1.
        assert selling_prices([1], transfers, players)[1] == pytest.approx(7.5)

    def test_a_price_fall_is_absorbed_in_full(self):
        players = self._players({1: 6.9})
        transfers = [{"element_in": 1, "element_in_cost": 74, "event": 2}]
        assert selling_prices([1], transfers, players)[1] == pytest.approx(6.9)

    def test_original_squad_uses_season_start_price(self):
        """Never transferred in, so the purchase price is now minus the drift."""
        players = self._players({7: 8.4}, drift={7: 0.4})
        assert purchase_prices([7], [], players)[7] == pytest.approx(8.0)
        assert selling_prices([7], [], players)[7] == pytest.approx(8.2)

    def test_latest_purchase_wins_when_bought_twice(self):
        players = self._players({1: 9.0})
        transfers = [
            {"element_in": 1, "element_in_cost": 70, "event": 2},
            {"element_in": 1, "element_in_cost": 88, "event": 6},
        ]
        assert purchase_prices([1], transfers, players)[1] == pytest.approx(8.8)

    def test_selling_price_never_exceeds_market_price(self):
        players = self._players({1: 8.0, 2: 5.0}, drift={2: -0.3})
        transfers = [{"element_in": 1, "element_in_cost": 60, "event": 1}]
        out = selling_prices([1, 2], transfers, players)
        for pid, sell in out.items():
            assert sell <= players.price[pid] + 1e-9


# ------------------------------------------------------------- pairing moves
class TestPairByPosition:
    POS = {1: 2, 2: 2, 3: 3, 4: 3, 5: 1}  # id -> element_type

    def test_pairs_within_the_same_position(self):
        ep = {(1, 5): 2.0, (2, 5): 3.0, (3, 5): 1.0, (4, 5): 6.0}
        pairs = optimize._pair_by_position([1, 3], [2, 4], self.POS, ep, 5)
        for out_id, in_id in pairs:
            assert self.POS[out_id] == self.POS[in_id]

    def test_never_loses_a_move(self):
        ep = {(i, 5): float(i) for i in range(1, 6)}
        pairs = optimize._pair_by_position([1, 3], [2, 4], self.POS, ep, 5)
        assert len(pairs) == 2
        assert {o for o, _ in pairs} == {1, 3}
        assert {i for _, i in pairs} == {2, 4}

    def test_best_incoming_is_matched_with_weakest_outgoing(self):
        pos = {1: 3, 2: 3, 10: 3, 11: 3}
        ep = {(1, 5): 1.0, (2, 5): 4.0, (10, 5): 9.0, (11, 5): 5.0}
        pairs = dict(optimize._pair_by_position([1, 2], [10, 11], pos, ep, 5))
        assert pairs[1] == 10, "the weakest player out should be replaced by the best in"

    def test_empty_input(self):
        assert optimize._pair_by_position([], [], self.POS, {}, 5) == []


# ----------------------------------------------------------------- reminders
class TestReminderStages:
    STAGES = [48.0, 24.0, 3.0]

    @pytest.mark.parametrize("hours,expected", [
        (72.0, None),   # too early
        (47.9, 48.0),
        (30.0, 48.0),   # still inside the 48h stage, 24h not reached
        (23.5, 24.0),
        (4.0, 24.0),
        (2.9, 3.0),
        (0.2, 3.0),
    ])
    def test_picks_the_right_stage(self, hours, expected):
        assert notify._stage_for(hours, self.STAGES) == expected

    def test_each_stage_is_reachable(self):
        seen = {notify._stage_for(h, self.STAGES)
                for h in np.arange(0.1, 60.0, 0.1)}
        assert seen == {48.0, 24.0, 3.0, None}


# --------------------------------------------------------- scoring constants
class TestVerifyScoring:
    def _bootstrap(self, **overrides) -> dict:
        scoring = {
            "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
            "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
            "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
            "assists": 3, "yellow_cards": -1, "red_cards": -3,
            "long_play": 2, "short_play": 1,
        }
        scoring.update(overrides)
        return {"game_config": {"scoring": scoring},
                "game_settings": {"squad_squadsize": 15, "squad_squadplay": 11,
                                  "squad_team_limit": 3, "max_extra_free_transfers": 4}}

    def test_current_constants_match_the_live_rules(self):
        assert verify_scoring(self._bootstrap()) == []

    def test_detects_a_changed_goal_value(self):
        bs = self._bootstrap(goals_scored={"GKP": 10, "DEF": 7, "MID": 5, "FWD": 4})
        problems = verify_scoring(bs)
        assert any("goal points for DEF" in p for p in problems)

    def test_detects_a_changed_assist_value(self):
        problems = verify_scoring(self._bootstrap(assists=4))
        assert any("assist points" in p for p in problems)

    def test_reports_when_the_api_publishes_nothing(self):
        assert verify_scoring({}) != []

    def test_our_constants_are_self_consistent(self):
        assert set(GOAL_POINTS) == set(CLEAN_SHEET_POINTS) == {1, 2, 3, 4}
        assert sum(SQUAD_BY_POSITION.values()) == SQUAD_SIZE


# ----------------------------------------------------------------- optimiser
def _synthetic_league(n_per_pos: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small but legal universe: enough players per position and per club."""
    rows, pid = [], 1
    for pos, count in {1: n_per_pos, 2: n_per_pos * 2,
                       3: n_per_pos * 2, 4: n_per_pos}.items():
        for i in range(count):
            rows.append({
                "id": pid, "element_type": pos, "team": (pid % 12) + 1,
                "price": 4.0 + (i % 6) * 0.8,
                "p_available": 1.0, "p_start": 0.9,
                "name": f"P{pid}", "team_name": f"T{(pid % 12) + 1}",
            })
            pid += 1
    players = pd.DataFrame(rows).set_index("id")
    gws = [1, 2, 3]
    rng = np.random.default_rng(7)
    ep = pd.DataFrame(rng.uniform(1.0, 7.0, size=(len(players), len(gws))),
                      index=players.index, columns=gws)
    return players, ep


def _legal_squad(players: pd.DataFrame) -> list[int]:
    """A starting fifteen that already satisfies every FPL rule.

    Taking the first N of each position looks reasonable but stacks four players
    from one club, so the solver is forced into transfers just to become legal —
    which quietly invalidates any test about how many transfers it chooses.
    """
    squad: list[int] = []
    per_club: dict[int, int] = {}
    for pos, count in SQUAD_BY_POSITION.items():
        taken = 0
        for pid, row in players[players.element_type == pos].iterrows():
            if taken == count:
                break
            if per_club.get(row.team, 0) >= MAX_PER_CLUB:
                continue
            squad.append(pid)
            per_club[row.team] = per_club.get(row.team, 0) + 1
            taken += 1
        assert taken == count, f"could not fill {count} legal players at position {pos}"
    return squad


def _cfg(**planning) -> Config:
    base = {"horizon": 3, "decay": 0.9, "candidate_pool": 120,
            "solver_time_limit": 30, "max_hit_per_gw": 8, "max_total_hits": 12,
            "transfer_friction": 0.0, "fallback_budget": 100.0}
    base.update(planning)
    return Config(raw={"entry": {"team_id": 1}, "planning": base,
                       "strategy": {"bench_weight": [0.0, 0.15, 0.1, 0.05],
                                    "captain_multiplier": 2.0, "mode": "balanced"}})


@pytest.fixture(scope="module")
def solved():
    players, ep = _synthetic_league()
    squad = _legal_squad(players)
    plan = optimize.solve(ep, players, squad, bank=2.0, free_transfers=1, cfg=_cfg())
    return plan, players, ep, squad


class TestOptimiserConstraints:
    def test_finds_an_optimal_solution(self, solved):
        plan, *_ = solved
        assert plan.status == "Optimal"

    def test_squad_is_always_fifteen(self, solved):
        plan, *_ = solved
        for week in plan.gameweeks:
            assert len(week.squad) == SQUAD_SIZE

    def test_squad_has_the_right_shape(self, solved):
        plan, players, *_ = solved
        for week in plan.gameweeks:
            counts = players.loc[week.squad].element_type.value_counts().to_dict()
            assert counts == SQUAD_BY_POSITION

    def test_xi_is_eleven_with_exactly_one_keeper(self, solved):
        plan, players, *_ = solved
        for week in plan.gameweeks:
            assert len(week.xi) == XI_SIZE
            assert (players.loc[week.xi].element_type == 1).sum() == 1

    def test_respects_the_three_per_club_limit(self, solved):
        plan, players, *_ = solved
        for week in plan.gameweeks:
            worst = players.loc[week.squad].team.value_counts().max()
            assert worst <= MAX_PER_CLUB

    def test_stays_inside_the_budget(self, solved):
        plan, players, _, squad = solved
        for week in plan.gameweeks:
            assert players.loc[week.squad].price.sum() <= plan.budget + 1e-6

    def test_captain_is_in_the_starting_eleven(self, solved):
        plan, *_ = solved
        for week in plan.gameweeks:
            assert week.captain in week.xi
            assert week.vice in week.xi
            assert week.captain != week.vice

    def test_bench_is_the_squad_minus_the_xi(self, solved):
        plan, *_ = solved
        for week in plan.gameweeks:
            assert sorted(week.xi + week.bench) == sorted(week.squad)
            assert len(week.bench) == SQUAD_SIZE - XI_SIZE

    def test_reserve_keeper_is_in_his_own_bench_slot(self, solved):
        plan, players, *_ = solved
        for week in plan.gameweeks:
            assert len(week.bench_gk) == 1
            assert players.loc[week.bench_gk[0]].element_type == 1
            assert all(players.loc[p].element_type != 1 for p in week.bench_outfield)

    def test_squad_changes_only_through_transfers(self, solved):
        plan, _, _, squad = solved
        previous = set(squad)
        for week in plan.gameweeks:
            assert set(week.squad) == (previous - set(week.sells)) | set(week.buys)
            previous = set(week.squad)

    def test_every_move_is_same_position(self, solved):
        plan, players, *_ = solved
        for week in plan.gameweeks:
            for out_id, in_id in week.moves:
                assert players.loc[out_id].element_type == players.loc[in_id].element_type

    def test_first_week_respects_the_free_transfer_count(self, solved):
        plan, *_ = solved
        first = plan.gameweeks[0]
        assert len(first.moves) <= first.free_transfers_before + first.hits


class TestOptimiserBehaviour:
    def test_empty_squad_falls_back_to_a_real_budget(self):
        """A missing squad used to make budget 0 and the problem infeasible."""
        players, ep = _synthetic_league()
        plan = optimize.solve(ep, players, [], bank=0.0, free_transfers=1, cfg=_cfg())
        assert plan.status == "Optimal"
        assert plan.budget == pytest.approx(100.0)
        assert len(plan.gameweeks[0].squad) == SQUAD_SIZE

    def test_friction_reduces_the_number_of_transfers(self):
        """The whole point: do not churn the squad for hundredths of a point."""
        players, ep = _synthetic_league()
        squad = _legal_squad(players)

        loose = optimize.solve(ep, players, squad, 2.0, 1, _cfg(transfer_friction=0.0))
        strict = optimize.solve(ep, players, squad, 2.0, 1, _cfg(transfer_friction=50.0))
        loose_moves = sum(len(w.moves) for w in loose.gameweeks)
        strict_moves = sum(len(w.moves) for w in strict.gameweeks)
        assert strict_moves < loose_moves
        assert strict_moves == 0, "prohibitive friction should stop every transfer"

    def test_never_sell_keeps_a_player_for_the_whole_horizon(self):
        players, ep = _synthetic_league()
        squad = _legal_squad(players)
        # Pick the squad member the solver would most want to drop.
        worst = min(squad, key=lambda p: ep.loc[p].sum())
        plan = optimize.solve(ep, players, squad, 2.0, 5, _cfg(), locked=[worst])
        for week in plan.gameweeks:
            assert worst in week.squad

    def test_selling_price_tightens_the_budget(self):
        players, ep = _synthetic_league()
        squad = _legal_squad(players)
        discounted = {p: players.price[p] - 0.3 for p in squad}
        full = optimize.solve(ep, players, squad, 0.0, 1, _cfg())
        real = optimize.solve(ep, players, squad, 0.0, 1, _cfg(), selling_price=discounted)
        assert real.budget < full.budget
        assert real.budget == pytest.approx(full.budget - 0.3 * len(squad), abs=0.05)


# ------------------------------------------------------------------ calendar
class TestCalendarFeed:
    def _events(self, offsets_hours: list[float], finished_ids: set[int] | None = None):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        finished_ids = finished_ids or set()
        return [{
            "id": i + 1,
            "deadline_time": (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": (i + 1) in finished_ids,
        } for i, h in enumerate(offsets_hours)]

    def test_is_a_well_formed_calendar(self):
        from fplbot import calendar_feed
        ics = calendar_feed.build_ics(self._events([100.0]))
        assert ics.startswith("BEGIN:VCALENDAR\r\n")
        assert ics.rstrip().endswith("END:VCALENDAR")
        assert ics.count("BEGIN:VEVENT") == ics.count("END:VEVENT") == 1
        assert ics.count("BEGIN:VALARM") == ics.count("END:VALARM")

    def test_uses_crlf_line_endings(self):
        """RFC 5545 requires CRLF; a bare LF makes some clients reject the feed."""
        from fplbot import calendar_feed
        ics = calendar_feed.build_ics(self._events([100.0]))
        assert "\r\n" in ics
        assert not ics.replace("\r\n", "").__contains__("\n")

    def test_skips_finished_and_past_gameweeks(self):
        from fplbot import calendar_feed
        events = self._events([-50.0, -2.0, 10.0, 200.0], finished_ids={1})
        ics = calendar_feed.build_ics(events)
        assert ics.count("BEGIN:VEVENT") == 2, "only the two future deadlines belong"

    def test_one_alarm_per_reminder_stage(self):
        from fplbot import calendar_feed
        ics = calendar_feed.build_ics(self._events([100.0]), remind_hours=[48, 24, 3])
        assert "TRIGGER:-PT2880M" in ics   # 48h
        assert "TRIGGER:-PT1440M" in ics   # 24h
        assert "TRIGGER:-PT180M" in ics    # 3h
        assert ics.count("BEGIN:VALARM") == 3

    def test_uid_is_stable_across_rebuilds(self):
        """Subscribers must update the event, not accumulate duplicates."""
        from fplbot import calendar_feed
        events = self._events([100.0])
        first = [l for l in calendar_feed.build_ics(events).split("\r\n") if l.startswith("UID:")]
        second = [l for l in calendar_feed.build_ics(events).split("\r\n") if l.startswith("UID:")]
        assert first == second and first

    def test_thai_text_survives_line_folding(self):
        """Folding counts octets, so it must not cut a Thai character in half."""
        from fplbot import calendar_feed
        ics = calendar_feed.build_ics(self._events([100.0]), site_url="https://example.com/fpl")
        # Unfold the way a calendar client does: drop CRLF + one leading space.
        unfolded = ics.replace("\r\n ", "")
        assert "ปิดจัดตัว" in unfolded
        assert "คนติดธงในตัวจริง" in unfolded
        for line in ics.split("\r\n"):
            assert len(line.encode("utf-8")) <= 75

    def test_event_ends_at_the_deadline(self):
        from fplbot import calendar_feed
        ics = calendar_feed.build_ics(self._events([100.0]), duration_minutes=30)
        start = next(l for l in ics.split("\r\n") if l.startswith("DTSTART:"))
        end = next(l for l in ics.split("\r\n") if l.startswith("DTEND:"))
        from datetime import datetime
        fmt = "%Y%m%dT%H%M%SZ"
        delta = (datetime.strptime(end[7:], fmt) - datetime.strptime(start[9:], fmt))
        assert delta.total_seconds() == 30 * 60

    def test_empty_season_still_produces_a_valid_feed(self):
        from fplbot import calendar_feed
        ics = calendar_feed.build_ics([])
        assert "BEGIN:VCALENDAR" in ics and "END:VCALENDAR" in ics
        assert "BEGIN:VEVENT" not in ics


# -------------------------------------------------------------- squad alerts
class TestSquadAlerts:
    """The feature the whole project exists for: never field a broken team.

    GW4 was started with a concussed defender in the XI for 1 point. These
    encode exactly that situation.
    """

    def _players(self, rows: list[dict]) -> pd.DataFrame:
        base = {"p_available": 1.0, "p_start": 0.9, "minutes": 900.0,
                "news": "", "yellow_cards": 0}
        return pd.DataFrame([{**base, **r} for r in rows]).set_index("id")

    def test_unavailable_player_in_the_xi_raises_an_alert(self):
        from fplbot.report import squad_alerts
        players = self._players([
            {"id": 1, "name": "Mendy", "p_available": 0.5,
             "news": "Concussion - 50% chance of playing"},
        ])
        flags, alerts = squad_alerts(players, [1], xi=[1], captain=None)
        assert len(alerts) == 1
        assert "Mendy" in alerts[0]
        assert "Concussion" in alerts[0]

    def test_the_same_player_on_the_bench_does_not_interrupt_you(self):
        """Benched is already the right answer — there is nothing to act on."""
        from fplbot.report import squad_alerts
        players = self._players([
            {"id": 1, "name": "Mendy", "p_available": 0.5, "news": "Concussion"},
        ])
        flags, alerts = squad_alerts(players, [1], xi=[], captain=None)
        assert alerts == []
        assert len(flags) == 1, "still worth showing on the page"

    def test_a_doubtful_captain_always_leads(self):
        from fplbot.report import squad_alerts
        players = self._players([
            {"id": 1, "name": "Haaland", "p_available": 0.75, "news": "Knock"},
            {"id": 2, "name": "Mendy", "p_available": 0.25, "news": "Concussion"},
        ])
        _, alerts = squad_alerts(players, [1, 2], xi=[1, 2], captain=1)
        assert "กัปตัน" in alerts[0] and "Haaland" in alerts[0]

    def test_a_fit_squad_produces_nothing(self):
        from fplbot.report import squad_alerts
        players = self._players([{"id": i, "name": f"P{i}"} for i in range(1, 12)])
        flags, alerts = squad_alerts(players, list(range(1, 12)),
                                     xi=list(range(1, 12)), captain=1)
        assert alerts == []
        assert flags == []

    def test_rotation_risk_is_a_flag_but_not_an_alert(self):
        """Worth knowing, not worth a 3am notification."""
        from fplbot.report import squad_alerts
        players = self._players([{"id": 1, "name": "Cherki", "p_start": 0.4}])
        flags, alerts = squad_alerts(players, [1], xi=[1], captain=None)
        assert alerts == []
        assert flags[0]["tag"] == "หมุนเวียน"

    def test_suspension_risk_is_reported(self):
        from fplbot.report import squad_alerts
        players = self._players([{"id": 1, "name": "Xhaka", "yellow_cards": 4}])
        flags, _ = squad_alerts(players, [1], xi=[1], captain=None)
        assert any(f["tag"] == "เสี่ยงโดนแบน" for f in flags)

    def test_high_severity_and_xi_problems_sort_first(self):
        from fplbot.report import squad_alerts
        players = self._players([
            {"id": 1, "name": "Bench", "p_available": 0.9},
            {"id": 2, "name": "Starter", "p_available": 0.2},
        ])
        flags, _ = squad_alerts(players, [1, 2], xi=[2], captain=None)
        assert flags[0]["name"] == "Starter"

    def test_players_missing_from_the_dataset_are_skipped(self):
        from fplbot.report import squad_alerts
        players = self._players([{"id": 1, "name": "A"}])
        flags, alerts = squad_alerts(players, [1, 999], xi=[1, 999], captain=999)
        assert flags == [] and alerts == []
