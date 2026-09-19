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

Money follows the actual transfer cash flow. Players already owned can stay in
the squad without being repurchased at today's price; cash changes only when a
player is sold or bought.
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
    # (out, in) pairs, matched by position so each one is a transfer you can
    # actually make in the FPL app.
    moves: list[tuple[int, int]] = field(default_factory=list)
    bench_gk: list[int] = field(default_factory=list)
    bench_outfield: list[int] = field(default_factory=list)
    hits: int = 0
    free_transfers_before: int = 1
    free_transfers_used: int = 0
    expected_points: float = 0.0


def _pair_by_position(sells: list[int], buys: list[int], pos: dict[int, int],
                      ep: dict, gw: int) -> list[tuple[int, int]]:
    """Match each sale to a purchase in the same position.

    The solver only constrains the squad as a whole, so `sells` and `buys` come
    back as two unordered sets. Zipping them naively produces pairs like
    "Gabriel (DEF) -> Saka (MID)", which is not a transfer FPL would accept and
    reads as nonsense. Squad composition is fixed per position, so a same-position
    matching always exists; within a position, pair the best incoming player with
    the weakest outgoing one so the headline move is the meaningful one.
    """
    pairs: list[tuple[int, int]] = []
    leftover_out, leftover_in = [], []
    for position in sorted({pos.get(p) for p in sells + buys}):
        outs = sorted((p for p in sells if pos.get(p) == position), key=lambda p: ep.get((p, gw), 0.0))
        ins = sorted((p for p in buys if pos.get(p) == position), key=lambda p: -ep.get((p, gw), 0.0))
        for i in range(min(len(outs), len(ins))):
            pairs.append((outs[i], ins[i]))
        leftover_out.extend(outs[len(ins):])
        leftover_in.extend(ins[len(outs):])
    # Should not happen while squad composition is constrained per position, but
    # never silently drop a move if it does.
    pairs.extend(zip(leftover_out, leftover_in))
    return pairs


@dataclass
class Plan:
    gameweeks: list[GameweekPlan]
    status: str
    objective: float
    budget: float
    notes: list[str] = field(default_factory=list)


@dataclass
class TransferScenario:
    """A complete first-GW choice, comparable with holding the current squad."""

    transfers: int
    plan: Plan
    variant: int = 1
    objective_gain: float = 0.0
    next_gw_gain: float = 0.0
    recommended: bool = False


@dataclass
class TransferCandidate:
    """One affordable replacement for the selected weak link."""

    out_id: int
    in_id: int
    out_ep: float
    in_ep: float
    gain: float
    net_gain: float
    fixtures: list[dict] = field(default_factory=list)


@dataclass
class TransferAdvice:
    """A single decision, plus alternatives for the manager to compare."""

    out_id: int | None
    candidates: list[TransferCandidate]
    recommend: bool
    hit: bool
    threshold: float
    reason: str


def outlook_by_player(ep_rows: pd.DataFrame, matches: int) -> tuple[dict[int, float], dict[int, list[dict]]]:
    """Sum each player's next N actual fixtures, preserving blanks and doubles."""
    if ep_rows.empty:
        return {}, {}
    ordered = ep_rows.copy()
    ordered["_kickoff"] = (ordered["kickoff"].fillna("")
                           if "kickoff" in ordered else "")
    ordered = ordered.sort_values(["player_id", "gw", "_kickoff"])
    window = ordered.groupby("player_id", sort=False).head(matches)
    totals = window.groupby("player_id").ep.sum().astype(float).to_dict()
    fixtures: dict[int, list[dict]] = {}
    for pid, rows in window.groupby("player_id", sort=False):
        fixtures[int(pid)] = [
            {"gw": int(r.gw), "opponent": int(r.opponent),
             "is_home": bool(r.is_home), "fdr": int(r.fdr), "ep": float(r.ep)}
            for r in rows.itertuples()
        ]
    return totals, fixtures


def _project_xi(ep_next: dict[int, float], players: pd.DataFrame,
                squad: list[int]) -> set[int]:
    """Pick the highest-EP legal XI from the current squad."""
    valid = [p for p in squad if p in players.index]
    keepers = sorted((p for p in valid if int(players.element_type[p]) == 1),
                     key=lambda p: ep_next.get(p, 0.0), reverse=True)
    if not keepers:
        return set()
    best: tuple[float, set[int]] | None = None
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            chosen = {keepers[0]}
            legal = True
            for pos, count in ((2, defenders), (3, midfielders), (4, forwards)):
                band = sorted((p for p in valid if int(players.element_type[p]) == pos),
                              key=lambda p: ep_next.get(p, 0.0), reverse=True)[:count]
                if len(band) != count:
                    legal = False
                    break
                chosen.update(band)
            if legal:
                score = sum(ep_next.get(p, 0.0) for p in chosen)
                if best is None or score > best[0]:
                    best = score, chosen
    return best[1] if best else set()


def _has_bench_cover(out_id: int, xi: set[int], squad: list[int],
                     players: pd.DataFrame) -> bool:
    """Return whether an available bench player can replace `out_id` legally."""
    if out_id not in xi:
        return True
    remaining = xi - {out_id}
    for pid in squad:
        if pid in xi or pid not in players.index or float(players.p_available[pid]) < 0.5:
            continue
        trial = remaining | {pid}
        counts = {pos: sum(int(players.element_type[p]) == pos for p in trial)
                  for pos in (1, 2, 3, 4)}
        if len(trial) == XI_SIZE and all(XI_MIN[pos] <= counts[pos] <= XI_MAX[pos]
                                        for pos in counts):
            return True
    return False


def analyse_transfers(ep_rows: pd.DataFrame, players: pd.DataFrame,
                      current_squad: list[int], bank: float,
                      free_transfers: int, cfg: Config,
                      selling_price: dict[int, float] | None = None) -> TransferAdvice:
    """Rank legal replacements and return one evidence-based transfer decision.

    Ownership is intentionally absent. The manager wants fixture-adjusted EP,
    availability and price to drive the ranking, with ownership shown only as
    context in the report.
    """
    threshold = float(cfg.get("planning", "min_transfer_gain", default=3.0))
    count = int(cfg.get("planning", "candidate_count", default=4))
    matches = int(cfg.get("planning", "outlook_matches", default=5))
    min_start = float(cfg.get("planning", "min_candidate_start", default=0.5))
    hit_policy = cfg.get("planning", "hit_policy", default="emergency_only")
    totals, fixtures = outlook_by_player(ep_rows, matches)
    owned = [p for p in current_squad if p in players.index]
    owned_set = set(owned)
    locked = {int(p) for p in (cfg.get("strategy", "never_sell", default=[]) or [])}
    sell_at = dict(selling_price or {})
    club_counts = players.loc[owned].team.value_counts().to_dict() if owned else {}

    pairs: list[TransferCandidate] = []
    for out_id in owned:
        if out_id in locked:
            continue
        out_pos = int(players.element_type[out_id])
        out_club = int(players.team[out_id])
        funds = float(sell_at.get(out_id, players.price[out_id])) + float(bank)
        band = players[(players.element_type == out_pos)
                       & (~players.index.isin(owned_set))
                       & (players.p_available > 0.25)
                       & (players.p_start >= min_start)
                       & (players.price <= funds + 1e-9)]
        for in_id, incoming in band.iterrows():
            in_id = int(in_id)
            in_club = int(incoming.team)
            after = int(club_counts.get(in_club, 0)) + 1 - int(in_club == out_club)
            if after > MAX_PER_CLUB:
                continue
            out_ep = float(totals.get(out_id, 0.0))
            in_ep = float(totals.get(in_id, 0.0))
            pairs.append(TransferCandidate(
                out_id=out_id, in_id=in_id, out_ep=out_ep, in_ep=in_ep,
                gain=in_ep - out_ep, net_gain=in_ep - out_ep,
                fixtures=fixtures.get(in_id, []),
            ))

    if not pairs:
        return TransferAdvice(None, [], False, False, threshold,
                              "ไม่พบตัวแทนที่ถูกกติกาและอยู่ในงบ")

    pairs.sort(key=lambda p: (p.gain, p.in_ep), reverse=True)
    out_id = pairs[0].out_id
    candidates = [p for p in pairs if p.out_id == out_id][:count]
    best = candidates[0]
    if best.gain < threshold:
        return TransferAdvice(out_id, candidates, False, False, threshold,
                              f"ตัวเลือกที่ดีที่สุดเพิ่มเพียง {best.gain:.1f} แต้มใน {matches} นัด")

    if free_transfers > 0:
        return TransferAdvice(out_id, candidates, True, False, threshold,
                              f"เพิ่ม {best.gain:.1f} แต้มใน {matches} นัดและไม่เสียแต้ม")

    first_gw = int(ep_rows.gw.min()) if not ep_rows.empty else 0
    next_ep = (ep_rows[ep_rows.gw == first_gw].groupby("player_id").ep.sum()
               .astype(float).to_dict())
    xi = _project_xi(next_ep, players, owned)
    unavailable = float(players.p_available[out_id]) < 0.5
    emergency = (hit_policy == "emergency_only" and unavailable and out_id in xi
                 and not _has_bench_cover(out_id, xi, owned, players))
    net = best.gain - HIT_COST
    for candidate in candidates:
        candidate.net_gain = candidate.gain - HIT_COST
    if emergency and net >= threshold:
        return TransferAdvice(out_id, candidates, True, True, threshold,
                              f"เหตุฉุกเฉิน: เพิ่มสุทธิ {net:.1f} แต้มหลังหัก 4 แต้ม")
    return TransferAdvice(out_id, candidates, False, False, threshold,
                          "ไม่มี free transfer และยังไม่เข้าเงื่อนไขย้ายฉุกเฉิน")


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
          selling_price: dict[int, float] | None = None,
          locked: list[int] | None = None,
          locked_out: list[int] | None = None,
          forced_first_move: tuple[int, int] | None = None,
          hold_first_week: bool = False,
          exact_first_moves: int | None = None,
          forbidden_first_buys: list[set[int]] | None = None) -> Plan:
    gws = list(ep_grid.columns)
    pool_size = int(cfg.get("planning", "candidate_pool", default=200))
    cand = choose_candidates(ep_grid, players, current_squad, pool_size)
    cand = [p for p in cand if p not in set(locked_out or [])]
    for pid in current_squad:
        if pid in players.index and pid not in cand:
            cand.append(pid)
    for pid in (forced_first_move or ()):
        if pid in players.index and pid not in cand:
            cand.append(pid)

    price = players.price.to_dict()
    pos = players.element_type.to_dict()
    club = players.team.to_dict()
    ep = {(p, g): float(ep_grid.at[p, g]) if p in ep_grid.index else 0.0
          for p in cand for g in gws}
    stable_rank = {p: rank for rank, p in enumerate(sorted(cand), start=1)}

    # What you would actually receive for the players you own. FPL gives back
    # only half of any price rise, so market value overstates your budget; the
    # caller passes real selling prices when it has them.
    sell_at = dict(selling_price or {})
    owned = [p for p in current_squad if p in players.index]
    squad_value = sum(sell_at.get(p, price.get(p, 0.0)) for p in owned)

    if owned:
        budget = round(squad_value + bank, 1)
    else:
        # No readable squad (first gameweek, or a private entry). Market value of
        # nothing is zero, which would make every 15-player squad infeasible, so
        # fall back to the standard starting budget.
        budget = float(cfg.get("planning", "fallback_budget", default=100.0))
        log.warning("no current squad known — planning from scratch on a %.1fm budget", budget)

    decay = float(cfg.get("planning", "decay", default=0.86))
    bench_w = [float(x) for x in cfg.get(
        "model", "auto_sub_slot_probability",
        default=cfg.get("strategy", "bench_weight", default=[0.0, 0.16, 0.10, 0.05]))]
    if len(bench_w) != 4:
        raise ValueError("model.auto_sub_slot_probability must contain GK, 1, 2, 3")
    cap_mult = float(cfg.get("strategy", "captain_multiplier", default=2.0))
    max_hit_gw = int(cfg.get("planning", "max_hit_per_gw", default=8))
    max_hits_total = int(cfg.get("planning", "max_total_hits", default=12))
    # A free transfer is not free: spending it now means not having it later, and
    # every move carries price-change and injury risk the model cannot see.
    # Without this the solver churns the bench every single week for hundredths
    # of a point, which is the opposite of what a time-poor manager wants.
    friction = float(cfg.get("planning", "transfer_friction", default=0.8))
    raw_cap = cfg.get("planning", "max_transfers_per_gw", default=1)
    max_moves_gw = None if raw_cap in (None, 0, "none") else int(raw_cap)

    prob = pulp.LpProblem("fpl_multi_gw", pulp.LpMaximize)
    V = prob.add_variable_dicts
    squad = V("squad", (cand, gws), cat="Binary")
    start = V("start", (cand, gws), cat="Binary")
    cap = V("cap", (cand, gws), cat="Binary")
    buy = V("buy", (cand, gws), cat="Binary")
    sell = V("sell", (cand, gws), cat="Binary")
    bench_slot = V("bench_slot", (cand, gws, range(4)), cat="Binary")
    free_used = V("free_used", gws, lowBound=0, upBound=MAX_SAVED_TRANSFERS, cat="Integer")
    hits = V("hits", gws, lowBound=0, upBound=max_hit_gw // HIT_COST, cat="Integer")
    hit_active = V("hit_active", gws, cat="Binary")
    ft = V("ft", gws, lowBound=0, upBound=MAX_SAVED_TRANSFERS, cat="Integer")
    ft_overflow = V("ft_overflow", gws, cat="Binary")
    cash = V("cash", gws, lowBound=0, cat="Continuous")

    # ---- objective --------------------------------------------------------
    prob += pulp.lpSum(
        (decay ** i) * (
            pulp.lpSum(ep[(p, g)] * start[p][g] for p in cand)
            + pulp.lpSum(ep[(p, g)] * (cap_mult - 1.0) * cap[p][g] for p in cand)
            + pulp.lpSum(bench_w[slot] * ep[(p, g)] * bench_slot[p][g][slot]
                         for p in cand for slot in range(4))
            - HIT_COST * hits[g]
            - friction * pulp.lpSum(buy[p][g] for p in cand)
            - 1e-6 * pulp.lpSum(stable_rank[p] * squad[p][g] for p in cand)
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
        prob += pulp.lpSum(start[p][g] for p in cand) == XI_SIZE
        for position in (1, 2, 3, 4):
            members = [start[p][g] for p in cand if pos[p] == position]
            prob += pulp.lpSum(members) >= XI_MIN[position]
            prob += pulp.lpSum(members) <= XI_MAX[position]
        prob += pulp.lpSum(cap[p][g] for p in cand) == 1
        for slot in range(4):
            prob += pulp.lpSum(bench_slot[p][g][slot] for p in cand) == 1
        for p in cand:
            prob += start[p][g] <= squad[p][g]
            prob += cap[p][g] <= start[p][g]
            prob += pulp.lpSum(bench_slot[p][g][slot] for slot in range(4)) \
                == squad[p][g] - start[p][g]
            if pos[p] == 1:
                for slot in (1, 2, 3):
                    prob += bench_slot[p][g][slot] == 0
            else:
                prob += bench_slot[p][g][0] == 0

    # Players you never want sold, whatever the model thinks of them.
    for pid in (locked or []):
        if pid in cand:
            for g in gws:
                prob += squad[pid][g] == 1

    # ---- transfers link the gameweeks together ---------------------------
    # With no squad to start from, the first gameweek is a fresh build, not a
    # set of swaps: fifteen players arrive and nobody leaves. Holding it to
    # "one player out for every player in" makes the whole problem infeasible.
    fresh_build = not owned

    for i, g in enumerate(gws):
        for p in cand:
            previous = (1 if p in current_squad else 0) if i == 0 else squad[p][gws[i - 1]]
            prob += squad[p][g] == previous + buy[p][g] - sell[p][g]
            prob += buy[p][g] + sell[p][g] <= 1
        if i == 0 and fresh_build:
            prob += ft[g] == min(free_transfers, MAX_SAVED_TRANSFERS)
            prob += free_used[g] == 0
            prob += hits[g] == 0
            prob += hit_active[g] == 0
            prob += ft_overflow[g] == 0
            prob += cash[g] == budget - pulp.lpSum(price[p] * buy[p][g] for p in cand)
            continue
        previous_cash = float(bank) if i == 0 else cash[gws[i - 1]]
        sale_revenue = pulp.lpSum(
            sell_at.get(p, price[p]) * sell[p][g] for p in cand)
        purchase_cost = pulp.lpSum(price[p] * buy[p][g] for p in cand)
        prob += cash[g] == previous_cash + sale_revenue - purchase_cost
        moves = pulp.lpSum(buy[p][g] for p in cand)
        prob += moves == pulp.lpSum(sell[p][g] for p in cand)
        # FPL always consumes available free transfers before charging a hit.
        # Equality accounts for every move; hit_active makes free_used equal
        # the entire FT bank whenever at least one paid transfer is required.
        prob += moves == free_used[g] + hits[g]
        prob += free_used[g] <= ft[g]
        prob += free_used[g] <= moves
        hit_limit = max_hit_gw // HIT_COST
        prob += hits[g] <= hit_limit * hit_active[g]
        prob += hits[g] >= hit_active[g]
        prob += free_used[g] >= ft[g] - MAX_SAVED_TRANSFERS * (1 - hit_active[g])
        # One change a week is the working rhythm, even with transfers banked.
        # Spending three at once is a different kind of decision — it rebuilds
        # the squad rather than improving it, and it is not what a manager who
        # wants a single considered move each week is asking for.
        if max_moves_gw is not None:
            cap_for_week = max_moves_gw
            if i == 0 and exact_first_moves is not None:
                cap_for_week = max(cap_for_week, int(exact_first_moves))
            prob += moves <= cap_for_week
        if i == 0 and forced_first_move:
            out_id, in_id = forced_first_move
            if out_id not in cand or in_id not in cand:
                raise ValueError("forced transfer is outside the optimiser candidate pool")
            prob += sell[out_id][g] == 1
            prob += buy[in_id][g] == 1
        elif i == 0 and hold_first_week:
            prob += moves == 0
        if i == 0 and exact_first_moves is not None:
            prob += moves == int(exact_first_moves)
        if i == 0:
            for forbidden in (forbidden_first_buys or []):
                members = [buy[p][g] for p in forbidden if p in cand]
                if members:
                    prob += pulp.lpSum(members) <= len(forbidden) - 1
        if i == 0:
            prob += ft[g] == min(free_transfers, MAX_SAVED_TRANSFERS)
            prob += ft_overflow[g] == 0
        else:
            previous_gw = gws[i - 1]
            prob += ft[g] == ft[previous_gw] - free_used[previous_gw] + 1 - ft_overflow[g]
            # Overflow is exactly the one transfer discarded when a full bank
            # of five rolls into another unused week.
            prob += MAX_SAVED_TRANSFERS * ft_overflow[g] <= ft[previous_gw]
            prob += MAX_SAVED_TRANSFERS * ft_overflow[g] \
                <= MAX_SAVED_TRANSFERS - free_used[previous_gw]
    prob += pulp.lpSum(hits[g] for g in gws) <= max_hits_total // HIT_COST

    time_limit = int(cfg.get("planning", "solver_time_limit", default=180))
    # PuLP 4 removes PULP_CBC_CMD. Use the supported COIN_CMD interface while
    # pointing it at the CBC binary bundled by the `pulp[cbc]` extra.
    cbc_path = getattr(pulp.apis.PULP_CBC_CMD, "pulp_cbc_path", None)
    prob.solve(pulp.COIN_CMD(path=cbc_path, msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    log.info("solver finished: %s (%d candidates, %d gameweeks)", status, len(cand), len(gws))
    if status != "Optimal":
        raise RuntimeError(f"transfer plan has no usable solution: {status}")

    plans: list[GameweekPlan] = []
    for g in gws:
        chosen = [p for p in cand if squad[p][g].value() and squad[p][g].value() > 0.5]
        xi = [p for p in cand if start[p][g].value() and start[p][g].value() > 0.5]
        captain = next((p for p in cand if cap[p][g].value() and cap[p][g].value() > 0.5), None)
        vice = max((p for p in xi if p != captain), key=lambda p: ep[(p, g)], default=captain)

        # FPL benches the reserve keeper in his own slot; the other three are
        # ordered, and that order decides who gets auto-subbed in first.
        bench_by_slot = [next(
            (p for p in cand if bench_slot[p][g][slot].value()
             and bench_slot[p][g][slot].value() > 0.5), None)
            for slot in range(4)]
        bench_gk = [bench_by_slot[0]] if bench_by_slot[0] is not None else []
        bench_out = [p for p in bench_by_slot[1:] if p is not None]

        buys = [p for p in cand if buy[p][g].value() and buy[p][g].value() > 0.5]
        sells = [p for p in cand if sell[p][g].value() and sell[p][g].value() > 0.5]
        # `moves` is for display; `buys`/`sells` stay as the solver reported them
        # so a fresh build (fifteen in, nobody out) is not silently emptied.
        moves = _pair_by_position(sells, buys, pos, ep, g)
        gw_hits = int(round(hits[g].value() or 0))
        gw_free_used = int(round(free_used[g].value() or 0))
        plans.append(GameweekPlan(
            gw=int(g), squad=chosen, xi=sorted(xi, key=lambda p: (pos[p], -ep[(p, g)])),
            bench=bench_gk + bench_out, bench_gk=bench_gk, bench_outfield=bench_out,
            captain=captain, vice=vice,
            buys=buys, sells=sells, moves=moves,
            hits=gw_hits,
            free_transfers_before=int(round(ft[g].value() or 0)),
            free_transfers_used=gw_free_used,
            expected_points=round(
                sum(ep[(p, g)] for p in xi)
                + (ep[(captain, g)] * (cap_mult - 1.0) if captain else 0.0)
                + sum(bench_w[slot] * ep[(pid, g)]
                      for slot, pid in enumerate(bench_by_slot) if pid is not None)
                - HIT_COST * gw_hits, 2),
        ))

    notes = []
    if status != "Optimal":
        notes.append(
            f"Solver returned {status} within {time_limit}s — the plan is the best found, "
            "not a proven optimum. Raise planning.solver_time_limit or lower candidate_pool.")
    return Plan(gameweeks=plans, status=status,
                objective=float(pulp.value(prob.objective) or 0.0),
                budget=budget, notes=notes)


def solve_scenarios(ep_grid: pd.DataFrame, players: pd.DataFrame,
                    current_squad: list[int], bank: float, free_transfers: int,
                    cfg: Config, *, selling_price: dict[int, float] | None = None,
                    locked: list[int] | None = None) -> list[TransferScenario]:
    """Solve holding and each usable free-transfer count on the same horizon."""
    maximum = min(MAX_SAVED_TRANSFERS, max(0, int(free_transfers)))
    configured = int(cfg.get("planning", "scenario_max_transfers", default=maximum))
    maximum = min(maximum, max(0, configured))
    scenarios: list[TransferScenario] = []
    alternatives = max(1, int(cfg.get("planning", "scenario_alternatives", default=2)))
    for count in range(maximum + 1):
        excluded: list[set[int]] = []
        variants = 1 if count == 0 else alternatives
        for variant in range(1, variants + 1):
            try:
                plan = solve(
                    ep_grid, players, current_squad, bank, free_transfers, cfg,
                    selling_price=selling_price, locked=locked,
                    hold_first_week=count == 0, exact_first_moves=count,
                    forbidden_first_buys=excluded,
                )
            except RuntimeError as exc:
                log.warning("%d-transfer scenario variant %d unavailable: %s",
                            count, variant, exc)
                break
            scenarios.append(TransferScenario(
                transfers=count, plan=plan, variant=variant))
            excluded.append(set(plan.gameweeks[0].buys))
    if not scenarios:
        raise RuntimeError("no transfer scenario has a usable optimal solution")
    baseline = next((s for s in scenarios if s.transfers == 0), scenarios[0])
    for scenario in scenarios:
        scenario.objective_gain = scenario.plan.objective - baseline.plan.objective
        scenario.next_gw_gain = (
            scenario.plan.gameweeks[0].expected_points
            - baseline.plan.gameweeks[0].expected_points
        )
    threshold = float(cfg.get("planning", "min_transfer_gain", default=3.0))
    eligible = [s for s in scenarios if s.objective_gain >= threshold]
    chosen = max(eligible or [baseline], key=lambda s: (s.plan.objective, -s.transfers))
    chosen.recommended = True
    return scenarios
