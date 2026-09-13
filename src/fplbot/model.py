"""Expected points per player per gameweek.

The whole system rests on one number. Everything downstream — captain picks,
transfer targets, the six-week plan — is arithmetic on top of it.

    EP = P(plays) x [ minutes + goals + assists + clean sheet
                      + defensive contribution + bonus + saves - cards - goals conceded ]

Each term is built from the player's own per-90 rates, shrunk toward a
positional prior so a two-game purple patch does not outrank a season of work,
then adjusted for the specific opponent and venue of that gameweek's fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import (ASSIST_POINTS, CLEAN_SHEET_POINTS, DEFCON_POINTS,
                     DEFCON_THRESHOLD, GOAL_POINTS, SAVES_PER_POINT,
                     YELLOW_CARD_POINTS, Config)
from .features import per90, positional_prior, shrink


def build_rates(players: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-90 rates we actually believe, after shrinkage."""
    k = float(cfg.get("model", "prior_minutes", default=360))
    df = players.copy()
    mins = df.minutes

    raw = {
        "xg90": df.expected_goals_per_90,
        "g90": per90(df.goals_scored, mins),
        "xa90": df.expected_assists_per_90,
        "a90": per90(df.assists, mins),
        "dc90": df.defensive_contribution_per_90,
        "bps90": per90(df.bps, mins),
        "saves90": df.saves_per_90,
        "yellow90": per90(df.yellow_cards, mins),
        "xgc90": df.expected_goals_conceded_per_90,
    }
    for name, series in raw.items():
        df[name] = shrink(series, mins, positional_prior(df.assign(**{name: series}), name), k)

    xw = float(cfg.get("model", "xg_weight", default=0.70))
    aw = float(cfg.get("model", "xa_weight", default=0.75))
    df["goal_rate"] = xw * df.xg90 + (1 - xw) * df.g90
    df["assist_rate"] = aw * df.xa90 + (1 - aw) * df.a90

    df.loc[df.is_penalty_taker, "goal_rate"] *= float(
        cfg.get("model", "penalty_taker_mult", default=1.08))
    df.loc[df.is_set_piece_taker, "assist_rate"] *= float(
        cfg.get("model", "set_piece_mult", default=1.06))
    return df


def opponent_multipliers(schedule: pd.DataFrame, teams: pd.DataFrame,
                         cfg: Config) -> pd.DataFrame:
    """How much easier or harder this specific fixture is than an average one."""
    lo, hi = cfg.get("model", "opponent_clip", default=[0.6, 1.6])
    home_m = float(cfg.get("model", "home_attack_mult", default=1.10))
    away_m = float(cfg.get("model", "away_attack_mult", default=0.92))

    mean_att = float(teams.attack.mean())
    mean_def = float(teams.defence.mean())

    s = schedule.copy()
    # Opponent defensive strength when they are at the venue this fixture puts
    # them at: if we are home, they defend away.
    opp_def = np.where(
        s.is_home,
        s.opponent.map(teams.strength_defence_away),
        s.opponent.map(teams.strength_defence_home),
    ).astype(float)
    opp_att = np.where(
        s.is_home,
        s.opponent.map(teams.strength_attack_away),
        s.opponent.map(teams.strength_attack_home),
    ).astype(float)
    own_def = np.where(
        s.is_home,
        s.team.map(teams.strength_defence_home),
        s.team.map(teams.strength_defence_away),
    ).astype(float)

    opp_def = np.nan_to_num(opp_def, nan=mean_def)
    opp_att = np.nan_to_num(opp_att, nan=mean_att)
    own_def = np.nan_to_num(own_def, nan=mean_def)

    venue = np.where(s.is_home, home_m, away_m)
    s["attack_mult"] = np.clip(mean_def / np.maximum(opp_def, 1.0) * venue, lo, hi)
    # Goals we expect to concede in this fixture.
    base = float(cfg.get("model", "league_goals_per_team", default=1.42))
    s["lambda_conceded"] = np.clip(
        base * (opp_att / max(mean_att, 1.0)) / np.maximum(own_def / max(mean_def, 1.0), 0.5)
        * np.where(s.is_home, 0.92, 1.10),
        0.15, 4.5,
    )
    return s


def expected_points(players: pd.DataFrame, teams: pd.DataFrame,
                    schedule: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return one row per (player, gameweek, fixture) with an `ep` column."""
    rates = build_rates(players, cfg)
    s = opponent_multipliers(schedule, teams, cfg)

    cols = ["element_type", "exp_minutes", "p_60plus", "p_any_minutes", "p_start",
            "goal_rate", "assist_rate", "dc90", "bps90", "saves90", "yellow90",
            "price", "name", "team_name", "selected_by_percent", "position"]
    s = s.merge(rates[cols], left_on="player_id", right_index=True, how="left")
    s = s.dropna(subset=["element_type"])
    s["element_type"] = s.element_type.astype(int)

    minutes_share = s.exp_minutes / 90.0

    # ---- minutes ---------------------------------------------------------
    p_short = (s.p_any_minutes - s.p_60plus).clip(lower=0.0)
    s["ep_minutes"] = 1.0 * p_short + 2.0 * s.p_60plus

    # ---- attacking returns ----------------------------------------------
    s["exp_goals"] = s.goal_rate * minutes_share * s.attack_mult
    s["exp_assists"] = s.assist_rate * minutes_share * s.attack_mult
    s["ep_goals"] = s.exp_goals * s.element_type.map(GOAL_POINTS)
    s["ep_assists"] = s.exp_assists * ASSIST_POINTS

    # ---- clean sheet and goals conceded ----------------------------------
    p_cs = np.exp(-s.lambda_conceded)
    s["p_clean_sheet"] = p_cs
    s["ep_clean_sheet"] = p_cs * s.p_60plus * s.element_type.map(CLEAN_SHEET_POINTS)
    # -1 per two goals conceded, for keepers and defenders only.
    concede_penalty = _expected_concede_penalty(s.lambda_conceded.to_numpy())
    s["ep_conceded"] = np.where(
        s.element_type.isin([1, 2]), -concede_penalty * s.p_60plus, 0.0)

    # ---- defensive contribution ------------------------------------------
    thresholds = s.element_type.map(DEFCON_THRESHOLD)
    lam_dc = (s.dc90 * minutes_share).clip(lower=0.0)
    p_defcon = np.where(
        thresholds.isna(), 0.0,
        1.0 - poisson.cdf(thresholds.fillna(99).to_numpy() - 1, lam_dc.to_numpy()),
    )
    s["p_defcon"] = p_defcon
    s["ep_defcon"] = p_defcon * DEFCON_POINTS

    # ---- bonus ------------------------------------------------------------
    # Bonus goes to the top three BPS scorers in a match. A player's chance of
    # landing there rises sharply with their BPS rate; this monotone mapping is
    # a stand-in until Phase 4 fits it against real match data.
    bps_match = s.bps90 * minutes_share
    s["ep_bonus"] = np.clip((bps_match - 16.0) / 11.0, 0.0, 2.2) * s.p_start

    # ---- keeper saves -----------------------------------------------------
    save_volume = s.saves90 * minutes_share * (s.lambda_conceded / 1.42).clip(0.5, 2.0)
    s["ep_saves"] = np.where(s.element_type == 1, save_volume / SAVES_PER_POINT, 0.0)

    # ---- discipline -------------------------------------------------------
    s["ep_cards"] = s.yellow90 * minutes_share * YELLOW_CARD_POINTS

    s["ep"] = (s.ep_minutes + s.ep_goals + s.ep_assists + s.ep_clean_sheet
               + s.ep_defcon + s.ep_bonus + s.ep_saves + s.ep_cards + s.ep_conceded)
    s["ep"] = s.ep.clip(lower=0.0)

    s = _apply_strategy(s, cfg)
    return s


def _expected_concede_penalty(lam: np.ndarray) -> np.ndarray:
    """E[floor(goals conceded / 2)] under a Poisson with mean lam."""
    ks = np.arange(0, 11)
    probs = poisson.pmf(ks[None, :], lam[:, None])
    return (probs * (ks // 2)[None, :]).sum(axis=1)


def _apply_strategy(s: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mode = cfg.get("strategy", "mode", default="balanced")
    weight = float(cfg.get("strategy", "differential_weight", default=0.0) or 0.0)
    s["ep_raw"] = s.ep
    if mode == "balanced" or weight == 0.0:
        return s
    own = pd.to_numeric(s.selected_by_percent, errors="coerce").fillna(0.0)
    tilt = (own - 50.0) / 50.0          # +1 when everyone owns them, -1 when nobody does
    sign = 1.0 if mode == "template" else -1.0
    s["ep"] = s.ep + sign * weight * tilt
    return s


def per_gameweek(ep_rows: pd.DataFrame, horizon_gws: list[int]) -> pd.DataFrame:
    """Collapse fixtures to a player x gameweek matrix.

    A double gameweek sums two fixtures; a blank gameweek yields zero, which is
    exactly what the optimiser should see.
    """
    grid = (ep_rows.groupby(["player_id", "gw"])["ep"].sum()
            .unstack("gw").reindex(columns=horizon_gws).fillna(0.0))
    return grid
