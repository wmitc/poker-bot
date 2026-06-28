"""A simple equity/EV-driven poker bot.

Two decisions, both grounded in expected value:

* :func:`pot_odds_decision` — given an equity estimate, call iff the call is
  +EV, i.e. ``equity > to_call / (pot + to_call)``.
* :func:`pushfold_ev` — the chip-EV of an all-in shove in a heads-up blind-vs-
  blind spot, against an assumed villain calling range. :func:`pushfold_range`
  maps this over every starting hand to produce a push/fold chart.

The push/fold model is the textbook one: both players are ``stack_bb`` deep,
hero either open-shoves or folds the small blind, and villain either calls with
a fixed range or folds. Equity-vs-call is estimated by Monte Carlo, so the
chart depends on the simulation seed/size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .card import RANK_CHARS, Card
from .equity import equity
from .ranges import Combo, parse_range

# Total distinct villain combos once hero's two cards are removed: C(50, 2).
_VILLAIN_COMBOS = 50 * 49 // 2


@dataclass(frozen=True, slots=True)
class PotOddsDecision:
    action: str  # "call" or "fold"
    equity: float
    to_call: float
    pot: float
    required_equity: float
    ev: float

    def __str__(self) -> str:
        return (
            f"{self.action.upper()} (equity {self.equity:.3f} vs required "
            f"{self.required_equity:.3f}, EV {self.ev:+.3f})"
        )


@dataclass(frozen=True, slots=True)
class PushFoldDecision:
    action: str  # "push" or "fold"
    stack_bb: float
    equity_vs_call: float
    call_freq: float
    ev_push: float
    ev_fold: float
    hand: str | None = None

    def __str__(self) -> str:
        label = f"{self.hand} " if self.hand else ""
        return (
            f"{label}{self.action.upper()} @ {self.stack_bb}bb "
            f"(EV_push {self.ev_push:+.3f} vs EV_fold {self.ev_fold:+.3f})"
        )


def pot_odds_decision(equity_value: float, to_call: float, pot: float) -> PotOddsDecision:
    """Call/fold from pot odds.

    ``pot`` is the amount in the middle before hero calls (what hero stands to
    win); ``to_call`` is the amount hero must put in. Calling is +EV exactly
    when ``equity > to_call / (pot + to_call)``.
    """
    if to_call < 0 or pot < 0:
        raise ValueError("pot and to_call must be non-negative")
    required = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    ev = equity_value * pot - (1.0 - equity_value) * to_call
    action = "call" if ev > 0 else "fold"
    return PotOddsDecision(action, equity_value, to_call, pot, required, ev)


def pushfold_ev(
    hero: str | list[Card] | Combo,
    villain_call_range: str,
    stack_bb: float,
    *,
    n: int = 20_000,
    seed: int | np.random.Generator | None = None,
    sb: float = 0.5,
    bb: float = 1.0,
    hand_label: str | None = None,
) -> PushFoldDecision:
    """Chip-EV of open-shoving ``stack_bb`` heads-up, vs a fixed calling range.

    EV_push = (1 - f) * bb + f * (2 * e - 1) * stack_bb, where ``f`` is villain's
    calling frequency and ``e`` is hero's equity when called. EV_fold = -sb.
    Hero shoves when EV_push exceeds EV_fold.
    """
    hero_cards = _as_hero_cards(hero)
    call_combos = parse_range(villain_call_range, exclude=hero_cards)
    call_freq = len(call_combos) / _VILLAIN_COMBOS

    if call_combos:
        e = equity(hero_cards, call_combos, n=n, seed=seed).equity
    else:
        e = 0.0  # never called; equity-when-called is irrelevant

    ev_push = (1.0 - call_freq) * bb + call_freq * (2.0 * e - 1.0) * stack_bb
    ev_fold = -sb
    action = "push" if ev_push > ev_fold else "fold"
    return PushFoldDecision(action, stack_bb, e, call_freq, ev_push, ev_fold, hand_label)


def canonical_hands() -> list[tuple[str, tuple[Card, Card]]]:
    """The 169 distinct starting hands with a representative two-card combo each.

    Labels follow the usual grid convention: pairs (``AA``), suited (``AKs``)
    above the diagonal, offsuit (``AKo``) below it.
    """
    ranks_desc = list(range(14, 1, -1))  # ace (14) down to two (2)
    out: list[tuple[str, tuple[Card, Card]]] = []
    for i, a in enumerate(ranks_desc):
        for j, b in enumerate(ranks_desc):
            a_ch, b_ch = _rank_char(a), _rank_char(b)
            if i == j:  # pair
                out.append((f"{a_ch}{a_ch}", (Card(a, 0), Card(a, 1))))
            elif i < j:  # above the diagonal: suited, a is the higher rank
                out.append((f"{a_ch}{b_ch}s", (Card(a, 0), Card(b, 0))))
            else:  # below the diagonal: offsuit, b is the higher rank
                out.append((f"{b_ch}{a_ch}o", (Card(b, 0), Card(a, 1))))
    return out


def _rank_char(rank: int) -> str:
    return RANK_CHARS[rank - 2]


def pushfold_range(
    stack_bb: float,
    villain_call_range: str,
    *,
    n: int = 5_000,
    seed: int | np.random.Generator | None = None,
    sb: float = 0.5,
    bb: float = 1.0,
) -> dict[str, PushFoldDecision]:
    """Compute the push/fold decision for every starting hand at one stack depth.

    Returns a mapping from hand label (``"AA"``, ``"72o"``, ...) to its decision.
    A shared seeded generator is threaded through so the whole chart is
    reproducible from a single ``seed``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    chart: dict[str, PushFoldDecision] = {}
    for label, combo in canonical_hands():
        chart[label] = pushfold_ev(
            combo, villain_call_range, stack_bb,
            n=n, seed=rng, sb=sb, bb=bb, hand_label=label,
        )
    return chart


def format_pushfold_grid(chart: dict[str, PushFoldDecision]) -> str:
    """Render a push/fold chart as a 13x13 text grid (``#`` push, ``.`` fold)."""
    order = list(reversed(RANK_CHARS))  # A, K, ..., 2
    lines = ["   " + " ".join(order)]
    for hi in order:
        cells = []
        for lo in order:
            if hi == lo:
                label = f"{hi}{hi}"
            elif order.index(hi) < order.index(lo):
                label = f"{hi}{lo}s"
            else:
                label = f"{lo}{hi}o"
            cells.append("#" if chart[label].action == "push" else ".")
        lines.append(f"{hi}  " + " ".join(cells))
    return "\n".join(lines)


def _as_hero_cards(hero: str | list[Card] | Combo) -> list[Card]:
    if isinstance(hero, str):
        from .card import parse_cards

        cards = parse_cards(hero)
    else:
        cards = list(hero)
    if len(cards) != 2:
        raise ValueError("hero must have exactly 2 cards")
    return cards
