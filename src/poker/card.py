"""Card and deck representation.

Cards are immutable ``(rank, suit)`` pairs:

* ``rank`` is an int ``2..14`` where ``J=11, Q=12, K=13, A=14`` (ace high).
* ``suit`` is an int ``0..3`` mapping to ``c, d, h, s``.

A card serialises to a two-character string like ``"Ah"`` (ace of hearts) or
``"Td"`` (ten of diamonds), the notation used throughout the project.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

RANK_TO_INT = {ch: i + 2 for i, ch in enumerate(RANK_CHARS)}
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}
SUIT_TO_INT = {ch: i for i, ch in enumerate(SUIT_CHARS)}
INT_TO_SUIT = {v: k for k, v in SUIT_TO_INT.items()}

MIN_RANK = 2
MAX_RANK = 14
NUM_SUITS = 4
NUM_CARDS = 52


@dataclass(frozen=True, slots=True, order=True)
class Card:
    """An immutable playing card.

    Ordering is by ``(rank, suit)`` which makes sorting hands deterministic; it
    has no poker meaning beyond convenience.
    """

    rank: int
    suit: int

    def __post_init__(self) -> None:
        if not MIN_RANK <= self.rank <= MAX_RANK:
            raise ValueError(f"rank out of range: {self.rank}")
        if not 0 <= self.suit < NUM_SUITS:
            raise ValueError(f"suit out of range: {self.suit}")

    @classmethod
    def from_str(cls, text: str) -> Card:
        """Parse a single card such as ``"Ah"`` (case-insensitive)."""
        s = text.strip()
        if len(s) != 2:
            raise ValueError(f"invalid card string: {text!r}")
        rank_ch, suit_ch = s[0].upper(), s[1].lower()
        if rank_ch not in RANK_TO_INT:
            raise ValueError(f"invalid rank in {text!r}")
        if suit_ch not in SUIT_TO_INT:
            raise ValueError(f"invalid suit in {text!r}")
        return cls(RANK_TO_INT[rank_ch], SUIT_TO_INT[suit_ch])

    @property
    def code(self) -> int:
        """Integer in ``0..51`` (``(rank - 2) * 4 + suit``)."""
        return (self.rank - MIN_RANK) * NUM_SUITS + self.suit

    @classmethod
    def from_code(cls, code: int) -> Card:
        if not 0 <= code < NUM_CARDS:
            raise ValueError(f"card code out of range: {code}")
        return cls(MIN_RANK + code // NUM_SUITS, code % NUM_SUITS)

    def __str__(self) -> str:
        return f"{INT_TO_RANK[self.rank]}{INT_TO_SUIT[self.suit]}"

    def __repr__(self) -> str:
        return f"Card('{self}')"


# The canonical 52-card deck, ordered by card code.
FULL_DECK: tuple[Card, ...] = tuple(Card.from_code(c) for c in range(NUM_CARDS))


def parse_cards(text: str | Iterable[str]) -> list[Card]:
    """Parse multiple cards.

    Accepts a single concatenated string (``"AhKsQd"``), a whitespace- or
    comma-separated string (``"Ah Ks Qd"``), or an iterable of card strings.
    """
    if isinstance(text, str):
        cleaned = text.replace(",", " ").strip()
        tokens = cleaned.split() if " " in cleaned else _chunk_pairs(cleaned)
    else:
        tokens = list(text)
    return [Card.from_str(tok) for tok in tokens if tok]


def _chunk_pairs(s: str) -> list[str]:
    if len(s) % 2 != 0:
        raise ValueError(f"cannot parse cards from {s!r}: odd length")
    return [s[i : i + 2] for i in range(0, len(s), 2)]


def remaining_deck(exclude: Iterable[Card]) -> list[Card]:
    """Return the deck minus ``exclude`` (duplicates in ``exclude`` are ignored)."""
    blocked = set(exclude)
    return [c for c in FULL_DECK if c not in blocked]


class Deck:
    """A shuffleable deck backed by a seeded ``numpy`` generator.

    Construction is reproducible: pass an int ``seed`` or an existing
    ``numpy.random.Generator``.
    """

    def __init__(self, seed: int | np.random.Generator | None = None) -> None:
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.cards: list[Card] = list(FULL_DECK)

    def shuffle(self) -> None:
        order = self._rng.permutation(len(self.cards))
        self.cards = [self.cards[i] for i in order]

    def deal(self, n: int) -> list[Card]:
        if n > len(self.cards):
            raise ValueError(f"cannot deal {n} cards from {len(self.cards)} remaining")
        dealt, self.cards = self.cards[:n], self.cards[n:]
        return dealt


def draw(
    rng: np.random.Generator,
    n: int,
    exclude: Sequence[Card] | Iterable[Card] = (),
) -> list[Card]:
    """Draw ``n`` distinct random cards not in ``exclude`` using ``rng``."""
    pool = remaining_deck(exclude)
    if n > len(pool):
        raise ValueError(f"cannot draw {n} cards from {len(pool)} available")
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]
