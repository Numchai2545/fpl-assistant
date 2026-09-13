"""Plan transfers across the whole horizon as one integer program.

Picking the best player for this gameweek and worrying about the next one later
is how you end up making a transfer you regret. This solves all six gameweeks
together: which fifteen to own each week, who starts, who captains, when to
spend a free transfer, when to bank one, and when a -4 hit actually pays.

Decision variables, for every candidate player p and gameweek g:
    squad[p,g]  own them
    start[p,g]  they are in the XI
    cap[p,g]    they wear the armband
    buy[p,g] / sell[p,g]

Money is handled with the standard simplification that a squad is affordable if
its total current market value fits the budget. Real FPL selling prices give
back only half of any rise, so the true budget is slightly tighter; the plan
reports the gap so you can sanity-check it before pulling the trigger.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import pulp

from .config import (HIT_COST, MAX_PER_CLUB, MAX_SAVED_TRANSFERS, SQUAD_BY_POSITION,
                     SQUAD_SIZE, XI_MAX, XI_MIN, XI_SIZE, Config)

log = logging.getLogger(__name__)


@dataclass
class GameweekPlan:
    gw: int
    squad: list[int]
    xi: list[int]
    bench: list[int]
    captain: int
    vice: int
    buys: list[int] = field(default_factory=list)
    sells: list[int] = field(default_factory=list)
    hits: int = 0
    free_transfers_before: int = 1
    expected_points: float = 0.0


@dataclass
class Plan:
    gameweeks: list[GameweekPlan]
    status: str
    objective: float
    budget: float
    notes: list[str] = field(default_factory=list)


def choose_candidates(ep_grid: pd.DataFrame, players: pd.DataFrame,
                      current_squad: list[int], pool_size: int) -> list[int]:
    """Shrink ~700 players to a pool the solver can chew through.

    Keeps the best expected scorers in each position band plus everyone you
    already own, so selling is always an option the solver can see.
    """
    total = ep_grid.sum(axis=1).rename("ep_total")
    frame = players.join(total, how="inner")
    frame = frame[frame.p_available > 0.05]

    keep: set[int] = set(pid for pid in current_squad if pid in players.index)
    share = {1: 0.10, 2: 0.32, 3: 0.36, 4: 0.22}
    for pos, fraction in share.items():
        n = max(8, int(pool_size * fraction))
        band = frame[frame.element_type == pos].nlargest(n, "ep_total")
        keep.update(band.index.tolist())
    # Cheap enablers matter: the optimiser needs benchable bodies to free money.
    for pos in (1, 2, 3, 4):
        cheap = frame[frame.element_type == pos].nsmallest(6, "price")
        keep.update(cheap[cheap.p_start > 0.25].index.tolist())
    return sorted(keep)


def solve(ep_grid: pd.DataFrame, players: pd.DataFrame, current_squad: list[int],
          bank: float, free_transfers: int, cfg: Config,
          locked_out: list[int] | None = None) -> Plan:
    gws = list(ep_grid.columns)
    pool_size = int(cfg.get("planning", "candidate_pool", default=200))
    cand = choose_candidates(ep_grid, players, current_squad, pool_size)
    cand = [p for p in cand if p not in set(locked_out or [])]
    for pid in current_squad:
        if pid in players.index and pid not in cand:
            cand.append(pid)

    price = players.price.to_dict()
    pos = players.element_type.to_dict()
    club = players.team.to_dict()
    ep = {(p, g): float(ep_grid.at[p, g]) if p in ep_grid.index else 0.0
          for p in cand for g in gws}

    squad_value = sum(price.get(p, 0.0) for p in current_squad)
    budget = round(squad_value + bank, 1)

    decay = float(cfg.get("planning", "decay", default=0.86))
    bench_w = list(cfg.get("strategy", "bench_weight", default=[0.0, 0.16, 0.10, 0.05]))
    bench_value = sum(bench_w) / max(len(bench_w), 1)
    cap_mult = float(cfg.get("strategy", "captain_multiplier", default=2.0))
    max_hit_gw = int(cfg.get("planning", "max_hit_per_gw", default=8))
    max_hits_total = int(cfg.get("planning", "max_total_hits", default=12))

    prob = pulp.LpProblem("fpl_multi_gw", pulp.LpMaximize)
    V = pulp.LpVariable.dicts
    squad = V("squad", (cand, gws), cat="Binary")
    start = V("start", (cand, gws), cat="Binary")
    cap = V("cap", (cand, gws), cat="Binary")
    buy = V("buy", (cand, gws), cat="Binary")
    sell = V("sell", (cand, gws), cat="Binary")
    free_used = V("free_used", gws, lowBound=0, upBound=MAX_SAVED_TRANSFERS, cat="Integer")
    hits = V("hits", gws, lowBound=0, upBound=max_hit_gw // HIT_COST, cat="Integer")
    ft = V("ft", gws, lowBound=0, upBound=MAX_SAVED_TRANSFERS, cat="Integer")

    # ---- objective --------------------------------------------------------
    prob += pulp.lpSum(
        (decay ** i) * (
            pulp.lpSum(ep[(p, g)] * start[p][g] for p in cand)
            + pulp.lpSum(ep[(p, g)] * (cap_mult - 1.0) * cap[p][g] for p in cand)
            + bench_value * pulp.lpSum(ep[(p, g)] * (squad[p][g] - start[p][g]) for p in cand)
            - HIT_COST * hits[g]
        )
        for i, g in enumerate(gws)
    )

    # ---- squad structure --------------------------------------------------
    for g in gws:
        prob += pulp.lpSum(squad[p][g] for p in cand) == SQUAD_SIZE
        for position, count in SQUAD_BY_POSITION.items():
            prob += pulp.lpSum(squad[p][g] for p in cand if pos[p] == position) == count
        for team_id in set(club[p] for p in cand):
            prob += pulp.lpSum(squad[p][g] for p in cand if club[p] == team_id) <= MAX_PER_CLUB
        prob += pulp.lpSum(price[p] * squad[p][g] for p in cand) <= budget

        prob += pulp.lpSum(start[p][g] for p in cand) == XI_SIZE
        for position in (1, 2, 3, 4):
            members = [start[p][g] for p in cand if pos[p] == position]
            prob += pulp.lpSum(members) >= XI_MIN[position]
            prob += pulp.lpSum(members) <= XI_MAX[position]
        prob += pulp.lpSum(cap[p][g] for p in cand) == 1
        for p in cand:
            prob += start[p][g] <= squad[p][g]
            prob += cap[p][g] <= start[p][g]

    # ---- transfers link the gameweeks together ---------------------------
    for i, g in enumerate(gws):
        for p in cand:
            previous = (1 if p in current_squad else 0) if i == 0 else squad[p][gws[i - 1]]
            prob += squad[p][g] == previous + buy[p][g] - sell[p][g]
            prob += buy[p][g] + sell[p][g] <= 1
        moves = pulp.lpSum(buy[p][g] for p in cand)
        prob += moves == pulp.lpSum(sell[p][g] for p in cand)
        prob += moves <= free_used[g] + hits[g]
        prob += free_used[g] <= ft[g]
        if i == 0:
            prob += ft[g] == min(free_transfers, MAX_SAVED_TRANSFERS)
        else:
            prob += ft[g] <= ft[gws[i - 1]] - free_used[gws[i - 1]] + 1
    prob += pulp.lpSum(hits[g] for g in gws) <= max_hits_total // HIT_COST

    time_limit = int(cfg.get("planning", "solver_time_limit", default=180))
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    log.info("solver finished: %s (%d candidates, %d gameweeks)", status, len(cand), len(gws))

    plans: list[GameweekPlan] = []
    for g in gws:
        chosen = [p for p in cand if squad[p][g].value() and squad[p][g].value() > 0.5]
        xi = [p for p in cand if start[p][g].value() and start[p][g].value() > 0.5]
        captain = next((p for p in cand if cap[p][g].value() and cap[p][g].value() > 0.5), None)
        bench = [p for p in chosen if p not in xi]
        bench.sort(key=lambda p: -ep[(p, g)])
        vice = max((p for p in xi if p != captain), key=lambda p: ep[(p, g)], default=captain)
        plans.append(GameweekPlan(
            gw=int(g), squad=chosen, xi=sorted(xi, key=lambda p: (pos[p], -ep[(p, g)])),
            bench=bench, captain=captain, vice=vice,
            buys=[p for p in cand if buy[p][g].value() and buy[p][g].value() > 0.5],
            sells=[p for p in cand if sell[p][g].value() and sell[p][g].value() > 0.5],
            hits=int(round(hits[g].value() or 0)),
            free_transfers_before=int(round(ft[g].value() or 0)),
            expected_points=round(
                sum(ep[(p, g)] for p in xi) + (ep[(captain, g)] if captain else 0.0)
                - HIT_COST * int(round(hits[g].value() or 0)), 2),
        ))

    notes = []
    if status != "Optimal":
        notes.append(
            f"Solver returned {status} within {time_limit}s — the plan is the best found, "
            "not a proven optimum. Raise planning.solver_time_limit or lower candidate_pool.")
    return Plan(gameweeks=plans, status=status,
                objective=float(pulp.value(prob.objective) or 0.0),
                budget=budget, notes=notes)
