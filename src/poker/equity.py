"""Monte Carlo (and exact) equity estimation for Texas Hold'em.

Equity is the share of the pot a hand wins on average at showdown, counting a
split pot as half. This module estimates hero's equity against a single
opponent -- a fixed hand or a range -- on any board.

:func:`equity` runs a Monte Carlo simulation and reports the estimate together
with a standard error and a 95% confidence interval, making the sampling
uncertainty explicit. :func:`equity_exact` enumerates the remaining cards for
small spaces and is used to cross-check the estimator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .card import FULL_DECK, Card, parse_cards
from .evaluator import evaluate7
from .ranges import Combo, parse_range

Z_95 = 1.959963984540054  # standard normal quantile for a 95% two-sided interval


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
    return parse_cards(value) if isinstance(value, str) else list(value)


def _ordered(a: Card, b: Card) -> Combo:
    return (a, b) if a.code > b.code else (b, a)


def _opponent_combos(opponent, dead: set[Card]) -> list[Combo]:
    """Resolve an opponent (fixed hand or range) to its legal combos."""
    if isinstance(opponent, str):
        try:
            cards = parse_cards(opponent)
        except ValueError:
            cards = None
        if cards is not None and len(cards) == 2:
            combos = [_ordered(cards[0], cards[1])]
        else:
            combos = parse_range(opponent, exclude=dead)
    else:
        items = list(opponent)
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


def _validate(hero_cards: list[Card], board_cards: list[Card]) -> set[Card]:
    if len(hero_cards) != 2:
        raise ValueError("hero must have exactly 2 cards")
    if len(board_cards) not in (0, 3, 4, 5):
        raise ValueError("board must have 0, 3, 4 or 5 cards")
    dead = set(hero_cards) | set(board_cards)
    if len(dead) != len(hero_cards) + len(board_cards):
        raise ValueError("duplicate cards among hero and board")
    return dead


def _summarize(shares: np.ndarray, exact: bool) -> EquityResult:
    n = int(shares.size)
    eq = float(shares.mean())
    win = float(np.mean(shares == 1.0))
    loss = float(np.mean(shares == 0.0))
    tie = float(1.0 - win - loss)
    if exact:
        return EquityResult(eq, win, tie, loss, n, 0.0, eq, eq, True)
    std_error = float(shares.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    half = Z_95 * std_error
    return EquityResult(
        eq, win, tie, loss, n, std_error,
        max(0.0, eq - half), min(1.0, eq + half), False,
    )


def _simulate(hero, opponent, board, n, seed) -> np.ndarray:
    """Run ``n`` showdowns and return hero's pot share for each."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    hero_cards = _as_cards(hero)
    board_cards = _as_cards(board)
    dead = _validate(hero_cards, board_cards)
    opp_combos = _opponent_combos(opponent, dead)
    need = 5 - len(board_cards)

    shares = np.empty(n, dtype=np.float64)
    for i in range(n):
        villain = opp_combos[rng.integers(len(opp_combos))]
        while villain[0] in dead or villain[1] in dead:
            villain = opp_combos[rng.integers(len(opp_combos))]
        used = (*dead, villain[0], villain[1])
        available = [c for c in FULL_DECK if c not in used]
        drawn = rng.choice(len(available), size=need, replace=False) if need else ()
        runout = board_cards + [available[j] for j in drawn]
        hero_score = evaluate7(hero_cards + runout)
        villain_score = evaluate7([villain[0], villain[1], *runout])
        shares[i] = _share(hero_score, villain_score)
    return shares


def _share(hero_score: int, villain_score: int) -> float:
    if hero_score > villain_score:
        return 1.0
    if hero_score < villain_score:
        return 0.0
    return 0.5


def equity(
    hero: str | Sequence[Card],
    opponent,
    board: str | Sequence[Card] = (),
    n: int = 100_000,
    seed: int | np.random.Generator | None = None,
) -> EquityResult:
    """Estimate hero's equity against one opponent by Monte Carlo simulation.

    ``opponent`` is a fixed hand (``"QsQc"``) or a range (``"QQ+, AKs"``).
    ``board`` holds 0, 3, 4 or 5 known community cards.
    """
    return _summarize(_simulate(hero, opponent, board, n, seed), exact=False)


def equity_exact(
    hero: str | Sequence[Card],
    opponent,
    board: str | Sequence[Card] = (),
) -> EquityResult:
    """Exact equity by enumerating every opponent hand and board completion.

    Only practical for small remaining spaces (turn/river, or two fixed hands on
    an early board). Used as a cross-check for :func:`equity`.
    """
    hero_cards = _as_cards(hero)
    board_cards = _as_cards(board)
    dead = _validate(hero_cards, board_cards)
    opp_combos = _opponent_combos(opponent, dead)
    need = 5 - len(board_cards)

    shares: list[float] = []
    for villain in opp_combos:
        used = {*dead, villain[0], villain[1]}
        pool = [c for c in FULL_DECK if c not in used]
        for extra in combinations(pool, need):
            runout = board_cards + list(extra)
            hero_score = evaluate7(hero_cards + runout)
            villain_score = evaluate7([villain[0], villain[1], *runout])
            shares.append(_share(hero_score, villain_score))

    if not shares:
        raise ValueError("no legal outcomes to enumerate")
    return _summarize(np.asarray(shares, dtype=np.float64), exact=True)


def equity_curve(
    hero: str | Sequence[Card],
    opponent,
    board: str | Sequence[Card] = (),
    n: int = 100_000,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running equity estimate and +/-1.96 SE band after each sample.

    Returns ``(n_values, estimate, half_width)`` arrays of length ``n`` for
    convergence plots.
    """
    shares = _simulate(hero, opponent, board, n, seed)
    counts = np.arange(1, n + 1)
    cum_mean = np.cumsum(shares) / counts
    cum_sq = np.cumsum(shares**2) / counts
    var = np.maximum(cum_sq - cum_mean**2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        half = Z_95 * np.sqrt(var / counts)
    return counts, cum_mean, half


__all__ = ["EquityResult", "equity", "equity_exact", "equity_curve"]
