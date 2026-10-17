from test_round23 import universe
from test_fplbot import _advice_rows, _cfg
from fplbot import optimize


def test_priority_prefixes_preserve_budget_clubs_and_previous_moves():
    players, _, squad, _, incoming = universe()
    rows = _advice_rows({p: (1. if p in squad else 5.) for p in players.index}, games=5)
    sale = {p: float(players.price[p]) - .1 for p in squad}
    result = optimize.rank_transfer_priorities(rows, players, squad, 20., 1, _cfg(), sale)
    assert len(result) >= 2
    current = set(squad)
    bank = 20.
    bought = set()
    for rank, priority in enumerate(result, 1):
        c = priority.candidate
        assert c.out_id in set(squad) - bought
        assert c.in_id not in set(squad) | bought
        assert players.element_type[c.out_id] == players.element_type[c.in_id]
        current.remove(c.out_id)
        current.add(c.in_id)
        bought.add(c.in_id)
        bank = round(bank + sale[c.out_id] - float(players.price[c.in_id]), 1)
        assert bank >= 0
        assert priority.bank_after == bank
        assert players.loc[list(current)].team.value_counts().max() <= 3
        assert priority.cumulative_hit_cost == 4 * max(0, rank - 1)
        assert priority.gain_after_hit == c.gain - (4 if rank > 1 else 0)
    repeated = optimize.rank_transfer_priorities(rows, players, squad, 20., 1, _cfg(), sale)
    assert [(p.candidate.out_id, p.candidate.in_id) for p in result] == [(p.candidate.out_id, p.candidate.in_id) for p in repeated]


def test_priorities_require_positive_gain_and_full_fixture_window():
    players, _, squad, _, _ = universe()
    rows = _advice_rows({p: 1. for p in players.index}, games=5)
    assert not optimize.rank_transfer_priorities(rows, players, squad, 20, 3, _cfg())
    rows = _advice_rows({p: 1. if p in squad else 10. for p in players.index}, games=3)
    assert not optimize.rank_transfer_priorities(rows, players, squad, 20, 3, _cfg())
