"""From-scratch poker hand evaluator.

Every hand is mapped to a single integer score so that two hands compare with a
plain ``>``: a higher score is a stronger hand, equal scores tie. The score
packs a hand *category* (high card .. straight flush) in the most significant
position followed by up to five tie-break ranks.

Two independent implementations are provided:

* :func:`evaluate5` scores exactly five cards.
* :func:`evaluate7` scores the best five-card hand contained in five-to-seven
  cards directly (no enumeration) and is the hot path for Monte Carlo.

:func:`best_of` is a deliberately simple brute-force evaluator (max over all
five-card subsets, scored by :func:`evaluate5`). It is slower but obviously
correct, and is used as a test oracle to validate :func:`evaluate7`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from .card import Card

# Hand categories, weakest to strongest.
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = {
    HIGH_CARD: "high card",
    ONE_PAIR: "one pair",
    TWO_PAIR: "two pair",
    THREE_OF_A_KIND: "three of a kind",
    STRAIGHT: "straight",
    FLUSH: "flush",
    FULL_HOUSE: "full house",
    FOUR_OF_A_KIND: "four of a kind",
    STRAIGHT_FLUSH: "straight flush",
}

_BASE = 15  # ranks are 2..14, so base 15 keeps each tie-break in its own digit
_SLOTS = 5
_CATEGORY_UNIT = _BASE**_SLOTS


def _pack(category: int, tiebreaks: Sequence[int]) -> int:
    """Combine a category and ordered tie-break ranks into one comparable int."""
    vals = list(tiebreaks)
    if len(vals) > _SLOTS:
        raise ValueError("too many tie-break ranks")
    vals += [0] * (_SLOTS - len(vals))
    score = category
    for v in vals:
        score = score * _BASE + v
    return score


def category_of(score: int) -> int:
    """Recover the hand category from a packed score."""
    return score // _CATEGORY_UNIT


def category_name(score: int) -> str:
    return CATEGORY_NAMES[category_of(score)]


def _straight_high(ranks: set[int]) -> int | None:
    """Return the high card of the best straight in ``ranks``, or ``None``.

    Handles the ace-low ("wheel") straight A-2-3-4-5 by treating an ace as a 1.
    """
    present = set(ranks)
    if 14 in present:
        present = present | {1}
    for high in sorted(present, reverse=True):
        if all((high - offset) in present for offset in range(5)):
            return high
    return None


def evaluate5(cards: Sequence[Card]) -> int:
    """Score exactly five cards."""
    if len(cards) != 5:
        raise ValueError(f"evaluate5 expects 5 cards, got {len(cards)}")
    ranks = sorted((c.rank for c in cards), reverse=True)
    counts = Counter(ranks)
    # Sort ranks by (frequency, rank) descending: the primary hand shape first.
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(count for _, count in by_count)
    is_flush = len({c.suit for c in cards}) == 1
    straight_high = _straight_high(set(ranks))

    if is_flush and straight_high is not None:
        return _pack(STRAIGHT_FLUSH, [straight_high])
    if shape == (4, 1):
        return _pack(FOUR_OF_A_KIND, [by_count[0][0], by_count[1][0]])
    if shape == (3, 2):
        return _pack(FULL_HOUSE, [by_count[0][0], by_count[1][0]])
    if is_flush:
        return _pack(FLUSH, ranks)
    if straight_high is not None:
        return _pack(STRAIGHT, [straight_high])
    if shape == (3, 1, 1):
        return _pack(THREE_OF_A_KIND, [by_count[0][0], by_count[1][0], by_count[2][0]])
    if shape == (2, 2, 1):
        return _pack(TWO_PAIR, [by_count[0][0], by_count[1][0], by_count[2][0]])
    if shape == (2, 1, 1, 1):
        return _pack(ONE_PAIR, [r for r, _ in by_count])
    return _pack(HIGH_CARD, ranks)


def evaluate7(cards: Sequence[Card]) -> int:
    """Score the best five-card hand among five-to-seven cards, directly.

    Independent of :func:`evaluate5`; this is the function used in the Monte
    Carlo inner loop.
    """
    n = len(cards)
    if not 5 <= n <= 7:
        raise ValueError(f"evaluate7 expects 5-7 cards, got {n}")

    ranks = [c.rank for c in cards]
    counts = Counter(ranks)
    distinct_desc = sorted(counts, reverse=True)
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    # At most one suit can hold five-plus cards out of seven.
    suit_counts = Counter(c.suit for c in cards)
    flush_suit = next((s for s, k in suit_counts.items() if k >= 5), None)
    flush_ranks = (
        sorted((c.rank for c in cards if c.suit == flush_suit), reverse=True)
        if flush_suit is not None
        else []
    )

    straight_high = _straight_high(set(ranks))
    straight_flush_high = _straight_high(set(flush_ranks)) if flush_suit is not None else None

    def kickers(exclude: set[int], how_many: int) -> list[int]:
        return [r for r in distinct_desc if r not in exclude][:how_many]

    top_count = by_count[0][1]
    second_count = by_count[1][1] if len(by_count) > 1 else 0

    if straight_flush_high is not None:
        return _pack(STRAIGHT_FLUSH, [straight_flush_high])
    if top_count == 4:
        quad = by_count[0][0]
        return _pack(FOUR_OF_A_KIND, [quad, *kickers({quad}, 1)])
    if top_count == 3 and second_count >= 2:
        # Best trips plus the best pair (a second set of trips counts as a pair).
        return _pack(FULL_HOUSE, [by_count[0][0], by_count[1][0]])
    if flush_suit is not None:
        return _pack(FLUSH, flush_ranks[:5])
    if straight_high is not None:
        return _pack(STRAIGHT, [straight_high])
    if top_count == 3:
        trips = by_count[0][0]
        return _pack(THREE_OF_A_KIND, [trips, *kickers({trips}, 2)])
    if top_count == 2 and second_count == 2:
        high_pair, low_pair = by_count[0][0], by_count[1][0]
        return _pack(TWO_PAIR, [high_pair, low_pair, *kickers({high_pair, low_pair}, 1)])
    if top_count == 2:
        pair = by_count[0][0]
        return _pack(ONE_PAIR, [pair, *kickers({pair}, 3)])
    return _pack(HIGH_CARD, distinct_desc[:5])


def best_of(cards: Sequence[Card]) -> int:
    """Brute-force best five-card score (test oracle).

    Enumerates every five-card subset and scores it with :func:`evaluate5`.
    Obviously correct, but slower than :func:`evaluate7`.
    """
    if len(cards) < 5:
        raise ValueError("need at least 5 cards")
    if len(cards) == 5:
        return evaluate5(cards)
    return max(evaluate5(combo) for combo in combinations(cards, 5))
