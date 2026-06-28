import pytest

from poker.bot import (
    canonical_hands,
    format_pushfold_grid,
    pot_odds_decision,
    pushfold_ev,
    pushfold_range,
)

# A typical, fixed heads-up calling range used across the push/fold tests.
CALL_RANGE = "22+, A2s+, A7o+, K9s+, KTo+, QTs+, JTs"


# --- pot odds ------------------------------------------------------------------

def test_pot_odds_required_equity():
    d = pot_odds_decision(0.5, to_call=10, pot=20)
    assert d.required_equity == pytest.approx(10 / 30)


def test_pot_odds_calls_when_profitable():
    d = pot_odds_decision(0.40, to_call=10, pot=20)  # required 0.333
    assert d.action == "call"
    assert d.ev > 0


def test_pot_odds_folds_when_unprofitable():
    d = pot_odds_decision(0.30, to_call=10, pot=20)  # required 0.333
    assert d.action == "fold"
    assert d.ev < 0


def test_pot_odds_decision_flips_at_breakeven():
    required = 10 / 30
    just_above = pot_odds_decision(required + 1e-6, 10, 20)
    just_below = pot_odds_decision(required - 1e-6, 10, 20)
    assert just_above.action == "call"
    assert just_below.action == "fold"


def test_pot_odds_ev_formula():
    d = pot_odds_decision(0.6, to_call=20, pot=80)
    assert d.ev == pytest.approx(0.6 * 80 - 0.4 * 20)


def test_pot_odds_rejects_negative():
    with pytest.raises(ValueError):
        pot_odds_decision(0.5, to_call=-1, pot=10)


# --- push / fold ---------------------------------------------------------------

def test_aces_always_push():
    d = pushfold_ev("AsAh", CALL_RANGE, stack_bb=50, n=3000, seed=1)
    assert d.action == "push"
    assert d.equity_vs_call > 0.7  # aces crush a calling range


def test_trash_folds_when_deep():
    d = pushfold_ev("7h2c", CALL_RANGE, stack_bb=50, n=3000, seed=1)
    assert d.action == "fold"


def test_any_two_push_when_extremely_short():
    # At a tiny stack the blinds you win dominate; even 72o is +EV to shove.
    d = pushfold_ev("7h2c", CALL_RANGE, stack_bb=1.0, n=3000, seed=2)
    assert d.action == "push"


def test_call_frequency_in_unit_interval():
    d = pushfold_ev("AsAh", CALL_RANGE, stack_bb=10, n=1000, seed=3)
    assert 0.0 < d.call_freq < 1.0


def test_pushfold_ev_is_reproducible():
    a = pushfold_ev("AsKs", CALL_RANGE, stack_bb=15, n=2000, seed=5)
    b = pushfold_ev("AsKs", CALL_RANGE, stack_bb=15, n=2000, seed=5)
    assert a.ev_push == b.ev_push


# --- push/fold chart -----------------------------------------------------------

def test_canonical_hands_count_and_uniqueness():
    hands = canonical_hands()
    assert len(hands) == 169
    labels = {label for label, _ in hands}
    assert len(labels) == 169
    # representative combos are always two distinct cards
    assert all(a != b for _, (a, b) in hands)


def test_push_range_widens_as_stack_shortens():
    tight = pushfold_range(25, CALL_RANGE, n=400, seed=10)
    wide = pushfold_range(6, CALL_RANGE, n=400, seed=10)
    n_tight = sum(d.action == "push" for d in tight.values())
    n_wide = sum(d.action == "push" for d in wide.values())
    assert n_wide > n_tight
    assert tight["AA"].action == wide["AA"].action == "push"


def test_format_grid_shape():
    chart = pushfold_range(10, CALL_RANGE, n=300, seed=12)
    grid = format_pushfold_grid(chart)
    rows = grid.splitlines()
    assert len(rows) == 14  # header + 13 rank rows
    assert chart["AA"] is not None
