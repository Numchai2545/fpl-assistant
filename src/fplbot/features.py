"""Turn raw API payloads into the tables the model works on.

Three outputs:
  players   one row per player, with shrunk per-90 rates and availability
  teams     one row per club, with FPL's attack/defence strength ratings
  schedule  one row per (player, gameweek, fixture) over the planning horizon,
            which is what makes double and blank gameweeks fall out naturally
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POSITIONS, STATUS_UNAVAILABLE

NUMERIC_FIELDS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "saves",
    "yellow_cards", "red_cards", "bonus", "bps", "goals_conceded",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
    "tackles", "recoveries", "clearances_blocks_interceptions",
    "now_cost", "cost_change_event", "cost_change_start",
    "transfers_in_event", "transfers_out_event", "selected_by_percent",
    "total_points", "form", "points_per_game", "ep_next",
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goals_conceded_per_90", "defensive_contribution_per_90",
    "saves_per_90", "starts_per_90", "clean_sheets_per_90",
]


STRENGTH_COLS = ("strength_attack_home", "strength_attack_away",
                 "strength_defence_home", "strength_defence_away")


def build_teams(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["teams"])
    for col in STRENGTH_COLS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # Pre-season, and in some mirrored exports, the detailed ratings are zero or
    # missing. Fall back to the coarse 1-5 team strength so nothing downstream
    # divides by nothing.
    if df[list(STRENGTH_COLS)].fillna(0).to_numpy().sum() == 0:
        coarse = pd.to_numeric(df.get("strength"), errors="coerce").fillna(3)
        base = 1000 + (coarse - 3) * 110
        df["strength_attack_home"] = base + 55
        df["strength_attack_away"] = base - 55
        # A stronger club defends better, which means a *higher* rating here.
        df["strength_defence_home"] = base + 55
        df["strength_defence_away"] = base - 55

    df["attack"] = (df.strength_attack_home + df.strength_attack_away) / 2
    df["defence"] = (df.strength_defence_home + df.strength_defence_away) / 2
    return df.set_index("id")


def build_players(bootstrap: dict, teams: pd.DataFrame,
                   fixtures: list[dict] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["elements"])
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["position"] = df.element_type.map(POSITIONS)
    df["price"] = df.now_cost / 10.0
    df["team_name"] = df.team.map(teams.short_name)
    df["name"] = df.web_name

    # ---- availability -----------------------------------------------------
    chance = pd.to_numeric(df.get("chance_of_playing_next_round"), errors="coerce")
    avail = chance / 100.0
    avail = avail.where(chance.notna(), np.where(df.status.isin(STATUS_UNAVAILABLE), 0.0, 1.0))
    df["p_available"] = avail.clip(0.0, 1.0)

    # ---- minutes model ----------------------------------------------------
    # Games the player's club has already played, so start rate is a real rate
    # and not diluted by fixtures that have not happened yet.
    df["team_games"] = df.team.map(_team_games_played(bootstrap, fixtures)).fillna(1).clip(lower=1)
    df["start_rate"] = (df.starts / df.team_games).clip(0.0, 1.0)
    df["min_per_start"] = np.where(df.starts > 0, df.minutes / df.starts.clip(lower=1), 0.0)
    df["min_per_start"] = df.min_per_start.clip(0, 90)

    df["p_start"] = (df.start_rate * df.p_available).clip(0.0, 1.0)
    # Chance a starter is still on at 60 minutes.
    p60_given_start = (df.min_per_start / 82.0).clip(0.0, 1.0)
    # Non-starters who are fit still come off the bench sometimes.
    p_cameo = ((1 - df.start_rate) * df.p_available * 0.35).clip(0.0, 1.0)
    df["p_60plus"] = df.p_start * p60_given_start
    df["p_any_minutes"] = (df.p_start + p_cameo).clip(0.0, 1.0)
    df["exp_minutes"] = (df.p_start * df.min_per_start + p_cameo * 14.0).clip(0.0, 90.0)

    # ---- set pieces -------------------------------------------------------
    pen = pd.to_numeric(df.get("penalties_order"), errors="coerce")
    corner = pd.to_numeric(df.get("corners_and_indirect_freekicks_order"), errors="coerce")
    df["is_penalty_taker"] = (pen == 1).fillna(False)
    df["is_set_piece_taker"] = (corner == 1).fillna(False)

    # ---- price momentum ---------------------------------------------------
    net_transfers = df.transfers_in_event - df.transfers_out_event
    total_managers = float(bootstrap.get("total_players", 1) or 1)
    df["transfer_pressure"] = net_transfers / max(total_managers, 1.0) * 100.0
    df["price_trend"] = np.select(
        [df.transfer_pressure > 0.45, df.transfer_pressure < -0.45],
        ["rising", "falling"], default="stable",
    )
    return df.set_index("id")


def minutes_shortlist(players: pd.DataFrame, current_squad: list[int],
                      target_count: int) -> list[int]:
    """Choose a bounded set whose match histories can change a decision.

    element-summary is one request per player, so fetching it for the entire
    league is both slow and discourteous. Current players are always included;
    the remainder is balanced by position and ranked with only public FPL
    signals available before our own expected-points model runs.
    """
    valid_owned = {int(pid) for pid in current_squad if pid in players.index}
    frame = players[players.p_available > 0.25].copy()
    price = frame.price.clip(lower=3.5)
    def numeric(name: str) -> pd.Series:
        source = frame[name] if name in frame else pd.Series(0.0, index=frame.index)
        return pd.to_numeric(source, errors="coerce").fillna(0.0)

    frame["_shortlist_score"] = (
        numeric("ep_next") * 2.0 + numeric("form") + numeric("points_per_game")
        + numeric("total_points") / price / 5.0
    )
    keep = set(valid_owned)
    shares = {1: 0.12, 2: 0.32, 3: 0.34, 4: 0.22}
    for position, share in shares.items():
        count = max(3, round(target_count * share))
        band = frame[(frame.element_type == position) & (~frame.index.isin(keep))]
        keep.update(band.nlargest(count, "_shortlist_score").index.astype(int).tolist())
    return sorted(keep)


def apply_recent_minutes(players: pd.DataFrame, histories: dict[int, dict],
                         cfg) -> pd.DataFrame:
    """Blend recent starts and minutes into the season-level availability model.

    The blend is deliberately Bayesian-looking rather than absolute: six recent
    matches can move a player strongly, but a single start cannot erase the
    season evidence. Missing or malformed histories leave the original values
    untouched, which keeps offline and pre-season builds usable.
    """
    out = players.copy()
    out["minutes_history_matches"] = 0
    out["minutes_model_source"] = "season"
    matches = int(cfg.get("model", "recent_minutes_matches", default=6))
    half_life = max(0.25, float(cfg.get(
        "model", "recent_minutes_half_life", default=3.0)))
    prior_matches = max(0.0, float(cfg.get(
        "model", "recent_minutes_prior_matches", default=3.0)))

    for pid, payload in histories.items():
        if pid not in out.index or not isinstance(payload, dict):
            continue
        rows = payload.get("history") or []
        rows = sorted((r for r in rows if r.get("round") is not None),
                      key=lambda r: int(r["round"]))[-matches:]
        if not rows:
            continue

        minutes = np.array([max(0.0, min(90.0, float(r.get("minutes") or 0.0)))
                            for r in rows], dtype=float)
        starts = np.array([
            float(r.get("starts")) if r.get("starts") is not None
            else float(m >= 60.0) for r, m in zip(rows, minutes)
        ], dtype=float)
        ages = np.arange(len(rows) - 1, -1, -1, dtype=float)
        weights = np.power(0.5, ages / half_life)
        weight_sum = float(weights.sum())
        recent_start = float(np.average(starts, weights=weights))
        recent_any = float(np.average(minutes > 0, weights=weights))
        recent_60 = float(np.average(minutes >= 60, weights=weights))
        recent_minutes = float(np.average(minutes, weights=weights))
        evidence = weight_sum / (weight_sum + prior_matches) if weight_sum else 0.0

        availability = float(out.at[pid, "p_available"])
        base_any = float(out.at[pid, "p_any_minutes"]) / max(availability, 1e-9)
        base_60 = float(out.at[pid, "p_60plus"]) / max(availability, 1e-9)
        base_minutes = float(out.at[pid, "exp_minutes"]) / max(availability, 1e-9)
        start_rate = (1.0 - evidence) * float(out.at[pid, "start_rate"]) + evidence * recent_start
        any_rate = (1.0 - evidence) * base_any + evidence * recent_any
        sixty_rate = (1.0 - evidence) * base_60 + evidence * recent_60
        minute_rate = (1.0 - evidence) * base_minutes + evidence * recent_minutes

        out.at[pid, "start_rate"] = float(np.clip(start_rate, 0.0, 1.0))
        out.at[pid, "p_start"] = float(np.clip(start_rate * availability, 0.0, 1.0))
        out.at[pid, "p_any_minutes"] = float(np.clip(any_rate * availability, 0.0, 1.0))
        out.at[pid, "p_60plus"] = float(np.clip(sixty_rate * availability, 0.0, 1.0))
        out.at[pid, "exp_minutes"] = float(np.clip(minute_rate * availability, 0.0, 90.0))
        out.at[pid, "minutes_history_matches"] = len(rows)
        out.at[pid, "minutes_model_source"] = "recent"
    return out


def apply_price_forecast(players: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add a bounded urgency signal without allowing price to alter EP ranks."""
    out = players.copy()
    rise = max(0.01, float(cfg.get("model", "price_rise_pressure", default=0.45)))
    fall = min(-0.01, float(cfg.get("model", "price_fall_pressure", default=-0.45)))
    pressure = pd.to_numeric(out.transfer_pressure, errors="coerce").fillna(0.0)
    out["price_change_signal"] = np.select(
        [pressure >= rise, pressure <= fall], ["rising", "falling"], default="stable")
    scale = np.where(pressure >= 0, rise, abs(fall))
    out["price_risk_score"] = np.clip(np.abs(pressure) / scale, 0.0, 2.0) / 2.0
    return out


def _team_games_played(bootstrap: dict, fixtures: list[dict] | None) -> pd.Series:
    """Matches each club has actually completed.

    Counting finished fixtures per club rather than finished gameweeks keeps the
    start rate honest when a club has a postponement, a blank, or a game in hand.
    """
    ids = [t["id"] for t in bootstrap["teams"]]
    if fixtures:
        counts = {tid: 0 for tid in ids}
        for fx in fixtures:
            if not fx.get("finished"):
                continue
            for side in ("team_h", "team_a"):
                tid = int(fx[side])
                counts[tid] = counts.get(tid, 0) + 1
        series = pd.Series(counts, dtype=float)
        if series.max() > 0:
            return series
    played = len([e for e in bootstrap["events"] if e.get("finished")])
    return pd.Series(max(played, 1), index=ids, dtype=float)


# ---------------------------------------------------------------- shrinkage
def shrink(observed_rate: pd.Series, minutes: pd.Series, prior: pd.Series | float,
           prior_minutes: float = 360.0) -> pd.Series:
    """Pull a small-sample per-90 rate toward a prior.

    A player with 90 minutes played has a rate we barely believe; one with
    1500 minutes has one we mostly do. The weight is minutes / (minutes + K).
    """
    w = minutes / (minutes + prior_minutes)
    return w * observed_rate.fillna(0.0) + (1 - w) * prior


def positional_prior(df: pd.DataFrame, column: str, min_minutes: float = 450.0) -> pd.Series:
    """Median per-90 rate by position among players with a real sample.

    Falls back to the overall median early in the season when nobody has
    reached the minutes bar yet.
    """
    pool = df[df.minutes >= min_minutes]
    if len(pool) < 20:
        pool = df[df.minutes >= 90]
    if len(pool) < 5:
        pool = df
    by_pos = pool.groupby("element_type")[column].median()
    overall = float(pool[column].median() or 0.0)
    return df.element_type.map(by_pos).fillna(overall)


def per90(total: pd.Series, minutes: pd.Series) -> pd.Series:
    return (total / minutes.clip(lower=1) * 90.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ----------------------------------------------------------------- schedule
def build_schedule(fixtures: list[dict], players: pd.DataFrame,
                   start_gw: int, horizon: int) -> pd.DataFrame:
    """One row per player per fixture in the horizon.

    A club with two fixtures in a gameweek produces two rows (double gameweek);
    a club with none produces no rows (blank gameweek). The optimiser reads
    expected points per gameweek by summing this table, so both cases are
    handled without any special code.
    """
    gws = list(range(start_gw, start_gw + horizon))
    rows = []
    for fx in fixtures:
        gw = fx.get("event")
        if gw is None or int(gw) not in gws:
            continue
        rows.append({
            "gw": int(gw), "team": int(fx["team_h"]), "opponent": int(fx["team_a"]),
            "is_home": True, "fdr": int(fx["team_h_difficulty"]),
            "kickoff": fx.get("kickoff_time"),
        })
        rows.append({
            "gw": int(gw), "team": int(fx["team_a"]), "opponent": int(fx["team_h"]),
            "is_home": False, "fdr": int(fx["team_a_difficulty"]),
            "kickoff": fx.get("kickoff_time"),
        })
    team_fixtures = pd.DataFrame(rows)
    if team_fixtures.empty:
        return pd.DataFrame(columns=["player_id", "gw", "team", "opponent", "is_home", "fdr"])

    squad = players.reset_index()[["id", "team"]].rename(columns={"id": "player_id"})
    sched = squad.merge(team_fixtures, on="team", how="left")
    return sched.dropna(subset=["gw"]).astype({"gw": int, "opponent": int})
