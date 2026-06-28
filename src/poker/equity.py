"""Monte Carlo (and exact) equity estimation for Texas Hold'em.

Equity is the share of the pot a hand wins on average at showdown, counting
split pots fractionally. :func:`equity` estimates it by simulating random
runouts and opponent holdings; it reports the estimate together with a standard
error and a 95% confidence interval, the quantities that make the Monte Carlo
uncertainty explicit. :func:`equity_exact` enumerates the remaining cards for
small spaces and is used to cross-check the estimator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .card import FULL_DECK, Card, parse_cards
from .evaluator import evaluate7
from .ranges import Combo, parse_range

Z_95 = 1.959963984540054  # standard normal quantile for a 95% two-sided interval

# An opponent is specified as a fixed hand ("AhKh") or a range ("QQ+, AKs"),
# either as a string or already-parsed cards/combos.


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Outcome of an equity calculation.

    ``equity`` is win probability plus half the tie probability (the pot share).
    ``std_error`` and the confidence interval are zero for exact results.
    """

    equity: float
    win: float
    tie: float
    loss: float
    samples: int
    std_error: float
    ci_low: float
    ci_high: float
    exact: bool

    def __str__(self) -> str:
        if self.exact:
            return f"equity {self.equity:.4f} (exact, {self.samples} outcomes)"
        return (
            f"equity {self.equity:.4f} +/- {self.std_error:.4f} "
            f"(95% CI [{self.ci_low:.4f}, {self.ci_high:.4f}], n={self.samples})"
        )


def _as_cards(value: str | Sequence[Card]) -> list[Card]:
    if isinstance(value, str):
        return parse_cards(value)
    return list(value)


def _opponent_combos(spec, dead: set[Card]) -> list[Combo]:
    """Resolve an opponent spec to the list of legal combos given dead cards."""
    if isinstance(spec, str):
        try:
            cards = parse_cards(spec)
        except ValueError:
            cards = None
        if cards is not None and len(cards) == 2:
            combos = [_ordered(cards[0], cards[1])]
        else:
            combos = parse_range(spec, exclude=dead)
    else:
        items = list(spec)
        if items and isinstance(items[0], Card):  # a fixed two-card hand
            if len(items) != 2:
                raise ValueError("a fixed opponent hand must have exactly 2 cards")
            combos = [_ordered(items[0], items[1])]
        else:  # already a list of combos
            combos = [tuple(c) for c in items]
    legal = [c for c in combos if c[0] not in dead and c[1] not in dead]
    if not legal:
        raise ValueError("opponent has no legal combos given the known cards")
    return legal


def _ordered(a: Card, b: Card) -> Combo:
    return (a, b) if a.code > b.code else (b, a)


def _normalize_opponents(opponents) -> list:
    # A bare string, or a single fixed hand given as a list of Cards, is one opponent.
    if isinstance(opponents, str):
        return [opponents]
    items = list(opponents)
    if items and isinstance(items[0], Card):
        return [items]
    return list(items)


def _hero_share(hero_score: int, opp_scores: list[int]) -> float:
    """Hero's pot share for one showdown (1 sole win, 1/k for a k-way tie, 0 loss)."""
    best = hero_score
    for s in opp_scores:
        if s > best:
            return 0.0
    winners = 1 + sum(1 for s in opp_scores if s == hero_score)
    return 1.0 / winners


def _summarize(shares: np.ndarray, exact: bool) -> EquityResult:
    n = int(shares.size)
    equity = float(shares.mean())
    win = float(np.mean(shares == 1.0))
    loss = float(np.mean(shares == 0.0))
    tie = float(1.0 - win - loss)
    if exact:
        return EquityResult(equity, win, tie, loss, n, 0.0, equity, equity, True)
    std_error = float(shares.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    half = Z_95 * std_error
    return EquityResult(
        equity, win, tie, loss, n, std_error,
        max(0.0, equity - half), min(1.0, equity + half), False,
    )


def equity(
    hero: str | Sequence[Card],
    opponents,
    board: str | Sequence[Card] = (),
    n: int = 100_000,
    seed: int | np.random.Generator | None = None,
) -> EquityResult:
    """Estimate hero's equity by Monte Carlo simulation.

    ``opponents`` is a single spec (fixed hand or range) or a list of specs for
    multiway pots. ``board`` holds 0, 3, 4 or 5 known community cards.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    hero_cards = _as_cards(hero)
    board_cards = _as_cards(board)
    if len(hero_cards) != 2:
        raise ValueError("hero must have exactly 2 cards")
    if len(board_cards) not in (0, 3, 4, 5):
        raise ValueError("board must have 0, 3, 4 or 5 cards")

    dead = set(hero_cards) | set(board_cards)
    if len(dead) != len(hero_cards) + len(board_cards):
        raise ValueError("duplicate cards among hero and board")

    opp_specs = _normalize_opponents(opponents)
    opp_combo_lists = [_opponent_combos(spec, dead) for spec in opp_specs]

    need = 5 - len(board_cards)
    shares = np.empty(n, dtype=np.float64)

    for i in range(n):
        used = set(dead)
        opp_hands = []
        for combos in opp_combo_lists:
            combo = combos[rng.integers(len(combos))]
            while combo[0] in used or combo[1] in used:
                combo = combos[rng.integers(len(combos))]
            opp_hands.append(combo)
            used.add(combo[0])
            used.add(combo[1])

        available = [c for c in FULL_DECK if c not in used]
        drawn = rng.choice(len(available), size=need, replace=False) if need else ()
        runout = board_cards + [available[j] for j in drawn]

        hero_score = evaluate7(hero_cards + runout)
        opp_scores = [evaluate7([oh[0], oh[1], *runout]) for oh in opp_hands]
        shares[i] = _hero_share(hero_score, opp_scores)

    return _summarize(shares, exact=False)


def equity_exact(
    hero: str | Sequence[Card],
    opponents,
    board: str | Sequence[Card] = (),
) -> EquityResult:
    """Exact equity by full enumeration. Only practical for small remaining spaces.

    Enumerates every legal assignment of opponent holdings and every board
    completion with uniform weight. Intended for turn/river spots (and as a
    cross-check for :func:`equity`), not preflop multiway.
    """
    hero_cards = _as_cards(hero)
    board_cards = _as_cards(board)
    if len(hero_cards) != 2:
        raise ValueError("hero must have exactly 2 cards")

    dead = set(hero_cards) | set(board_cards)
    opp_specs = _normalize_opponents(opponents)
    opp_combo_lists = [_opponent_combos(spec, dead) for spec in opp_specs]
    need = 5 - len(board_cards)

    shares: list[float] = []
    for assignment in product(*opp_combo_lists):
        used = set(dead)
        cards = [c for combo in assignment for c in combo]
        if len(set(cards)) != len(cards) or any(c in used for c in cards):
            continue  # overlapping opponent holdings
        used.update(cards)
        pool = [c for c in FULL_DECK if c not in used]
        for extra in combinations(pool, need):
            runout = board_cards + list(extra)
            hero_score = evaluate7(hero_cards + runout)
            opp_scores = [evaluate7([oh[0], oh[1], *runout]) for oh in assignment]
            shares.append(_hero_share(hero_score, opp_scores))

    if not shares:
        raise ValueError("no legal outcomes to enumerate")
    return _summarize(np.asarray(shares, dtype=np.float64), exact=True)


def equity_curve(
    hero: str | Sequence[Card],
    opponents,
    board: str | Sequence[Card] = (),
    n: int = 100_000,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return running equity estimate and +/-1.96 SE band vs sample size.

    Reuses the same simulation as :func:`equity` but exposes the cumulative
    estimate after each sample, for convergence plots. Returns
    ``(n_values, estimate, half_width)`` arrays of length ``n``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    hero_cards = _as_cards(hero)
    board_cards = _as_cards(board)
    dead = set(hero_cards) | set(board_cards)
    opp_specs = _normalize_opponents(opponents)
    opp_combo_lists = [_opponent_combos(spec, dead) for spec in opp_specs]
    need = 5 - len(board_cards)

    shares = np.empty(n, dtype=np.float64)
    for i in range(n):
        used = set(dead)
        opp_hands = []
        for combos in opp_combo_lists:
            combo = combos[rng.integers(len(combos))]
            while combo[0] in used or combo[1] in used:
                combo = combos[rng.integers(len(combos))]
            opp_hands.append(combo)
            used.add(combo[0])
            used.add(combo[1])
        available = [c for c in FULL_DECK if c not in used]
        drawn = rng.choice(len(available), size=need, replace=False) if need else ()
        runout = board_cards + [available[j] for j in drawn]
        hero_score = evaluate7(hero_cards + runout)
        opp_scores = [evaluate7([oh[0], oh[1], *runout]) for oh in opp_hands]
        shares[i] = _hero_share(hero_score, opp_scores)

    counts = np.arange(1, n + 1)
    cum_mean = np.cumsum(shares) / counts
    cum_sq = np.cumsum(shares**2) / counts
    var = np.maximum(cum_sq - cum_mean**2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        half = Z_95 * np.sqrt(var / counts)
    return counts, cum_mean, half


__all__ = [
    "EquityResult",
    "equity",
    "equity_exact",
    "equity_curve",
]
