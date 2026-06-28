"""Hand-range notation parser.

Expands the standard shorthand used to describe ranges of starting hands into
the concrete two-card combinations they represent. Supported tokens:

* Pairs: ``QQ``; with ``+``: ``QQ+`` (QQ, KK, AA); spans: ``22-99``.
* Suited / offsuit: ``AKs``, ``AKo``; unspecified ``AK`` means both.
* Kicker ``+``: ``ATs+`` (ATs, AJs, AQs, AKs), ``KTo+``, ``AT+`` (both).
* Connector-style spans: ``T9s-65s`` (T9s, 98s, 87s, 76s, 65s).

A *combo* is a 2-tuple of :class:`~poker.card.Card` ordered by descending card
code. A *range* is the de-duplicated, deterministically ordered list of combos.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from .card import NUM_SUITS, RANK_TO_INT, Card

Combo = tuple[Card, Card]
# A hand class is (high_rank, low_rank, suitedness) with suitedness in {"s", "o", None};
# None means "both" for non-pairs (a pair always has suitedness None).
HandClass = tuple[int, int, str | None]


def _rank(ch: str) -> int:
    try:
        return RANK_TO_INT[ch.upper()]
    except KeyError:
        raise ValueError(f"invalid rank character: {ch!r}") from None


def _combo(a: Card, b: Card) -> Combo:
    return (a, b) if a.code > b.code else (b, a)


def _parse_handclass(token: str) -> HandClass:
    s = token.strip()
    suit: str | None = None
    if s and s[-1].lower() in ("s", "o"):
        suit = s[-1].lower()
        s = s[:-1]
    if len(s) != 2:
        raise ValueError(f"invalid hand token: {token!r}")
    r1, r2 = _rank(s[0]), _rank(s[1])
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        if suit is not None:
            raise ValueError(f"a pair cannot be suited/offsuit: {token!r}")
        return (r1, r2, None)
    return (r1, r2, suit)


def _expand_plus(base: str) -> list[HandClass]:
    r1, r2, suit = _parse_handclass(base)
    if r1 == r2:  # pair and up, e.g. QQ+
        return [(r, r, None) for r in range(r1, RANK_TO_INT["A"] + 1)]
    # Fixed high card, kicker climbs to one below it, e.g. ATs+ -> AT..AK suited.
    return [(r1, low, suit) for low in range(r2, r1)]


def _expand_span(lhs: str, rhs: str) -> list[HandClass]:
    a, b = _parse_handclass(lhs), _parse_handclass(rhs)
    a_pair, b_pair = a[0] == a[1], b[0] == b[1]
    if a_pair != b_pair:
        raise ValueError(f"mismatched span endpoints: {lhs!r}-{rhs!r}")
    if a_pair:
        lo, hi = sorted((a[0], b[0]))
        return [(r, r, None) for r in range(lo, hi + 1)]
    if a[2] != b[2]:
        raise ValueError(f"span endpoints differ in suitedness: {lhs!r}-{rhs!r}")
    if (a[0] - a[1]) != (b[0] - b[1]):
        raise ValueError(f"span endpoints have different gaps: {lhs!r}-{rhs!r}")
    gap = a[0] - a[1]
    hi_top, lo_top = max(a[0], b[0]), min(a[0], b[0])
    return [(top, top - gap, a[2]) for top in range(lo_top, hi_top + 1)]


def _parse_token(token: str) -> list[HandClass]:
    tok = token.strip()
    if not tok:
        return []
    if tok.endswith("+"):
        return _expand_plus(tok[:-1])
    if "-" in tok:
        lhs, rhs = tok.split("-", 1)
        return _expand_span(lhs, rhs)
    return [_parse_handclass(tok)]


def expand_handclass(hc: HandClass) -> list[Combo]:
    """All concrete combos for a single hand class."""
    r1, r2, suit = hc
    if r1 == r2:  # pair: choose 2 of 4 suits
        cards = [Card(r1, s) for s in range(NUM_SUITS)]
        return [_combo(a, b) for a, b in combinations(cards, 2)]
    combos: list[Combo] = []
    if suit in (None, "s"):
        combos += [_combo(Card(r1, s), Card(r2, s)) for s in range(NUM_SUITS)]
    if suit in (None, "o"):
        combos += [
            _combo(Card(r1, s1), Card(r2, s2))
            for s1 in range(NUM_SUITS)
            for s2 in range(NUM_SUITS)
            if s1 != s2
        ]
    return combos


def parse_range(text: str, exclude: Iterable[Card] = ()) -> list[Combo]:
    """Expand a comma-separated range string into concrete combos.

    Combos that contain any card in ``exclude`` (e.g. hero's cards or the board)
    are dropped. The result is de-duplicated and ordered deterministically.
    """
    blocked = set(exclude)
    seen: set[frozenset[Card]] = set()
    out: list[Combo] = []
    for token in text.replace(";", ",").split(","):
        for hc in _parse_token(token):
            for combo in expand_handclass(hc):
                if combo[0] in blocked or combo[1] in blocked:
                    continue
                key = frozenset(combo)
                if key not in seen:
                    seen.add(key)
                    out.append(combo)
    out.sort(key=lambda c: (c[0].code, c[1].code), reverse=True)
    return out
