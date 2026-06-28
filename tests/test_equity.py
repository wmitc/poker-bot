import pytest

from poker.equity import equity, equity_curve, equity_exact


def test_equity_is_deterministic_with_seed():
    a = equity("AhKh", "QsQc", n=5000, seed=1)
    b = equity("AhKh", "QsQc", n=5000, seed=1)
    assert a.equity == b.equity
    assert a.std_error == b.std_error


def test_aa_vs_kk_known_equity():
    # AA vs KK preflop is ~0.82 for the aces.
    res = equity("AsAh", "KsKh", n=40000, seed=7)
    assert abs(res.equity - 0.82) < 0.02
    assert res.ci_low < 0.82 < res.ci_high  # true value inside the 95% CI


def test_aa_vs_kk_matches_exact_on_river():
    # With four board cards known, only the river varies (44 outcomes): exact is cheap.
    board = "2c7d9hQs"
    exact = equity_exact("AsAh", "KsKh", board=board)
    mc = equity("AsAh", "KsKh", board=board, n=20000, seed=3)
    assert exact.exact and exact.samples == 44
    assert abs(mc.equity - exact.equity) < 0.02


def test_probabilities_sum_to_one():
    res = equity("AhKh", "QsQc", n=3000, seed=2)
    assert res.win + res.tie + res.loss == pytest.approx(1.0)
    assert res.equity == pytest.approx(res.win + res.tie / 2, abs=1e-9)


def test_identical_made_hands_split():
    # Both players hold the same straight on a fully-known board -> certain tie.
    res = equity_exact("AsKs", "AdKd", board="2c3h4d5s6c")
    assert res.tie == pytest.approx(1.0)
    assert res.equity == pytest.approx(0.5)


def test_board_quads_kicker_decides():
    # Quad aces on board; the higher (king) kicker plays and wins outright.
    res = equity_exact("KsKh", "QsQh", board="AcAdAhAs2c")
    assert res.equity == pytest.approx(1.0)
    # Villain holds the case ace for quads; hero's full house is drawing dead.
    res2 = equity_exact("KsKh", "AdQh", board="AcAhAs2c3d")
    assert res2.equity == pytest.approx(0.0)


def test_equity_against_range_runs_and_is_reasonable():
    # AKs should be a modest favourite against a tight pair-heavy range.
    res = equity("AsKs", "QQ+, AKo", n=8000, seed=11)
    assert 0.3 < res.equity < 0.7


def test_multiway_equity_lower_than_heads_up():
    hu = equity("AsAh", "KsKh", n=8000, seed=5)
    multi = equity("AsAh", ["KsKh", "QsQh"], n=8000, seed=5)
    assert multi.equity < hu.equity  # more opponents -> less equity


def test_exact_vs_mc_on_flop():
    # Two known hands on a flop: turn+river enumerated, C(45, 2) = 990 outcomes.
    board = "2c7dTh"
    exact = equity_exact("AsAd", "KsKd", board=board)
    mc = equity("AsAd", "KsKd", board=board, n=15000, seed=4)
    assert exact.samples == 990
    assert abs(mc.equity - exact.equity) < 0.02


def test_equity_curve_converges():
    counts, est, half = equity_curve("AsAh", "KsKh", n=5000, seed=9)
    assert counts[0] == 1 and counts[-1] == 5000
    assert est.shape == half.shape == (5000,)
    # Final estimate near the known value; SE band shrinks over time.
    assert abs(est[-1] - 0.82) < 0.03
    assert half[-1] < half[100]


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        equity("AhKh", "QsQc", board="2c3d", n=10)  # board of 2 is illegal
    with pytest.raises(ValueError):
        equity("Ah", "QsQc", n=10)  # hero needs 2 cards
    with pytest.raises(ValueError):
        equity("AhKh", "Ah2d", n=10)  # opponent shares a card with hero


def test_std_error_scales_like_one_over_sqrt_n():
    se_small = equity("AhKh", "QsQc", n=2000, seed=21).std_error
    se_large = equity("AhKh", "QsQc", n=8000, seed=22).std_error
    ratio = se_small / se_large
    assert 1.6 < ratio < 2.4  # ~ sqrt(4) = 2
