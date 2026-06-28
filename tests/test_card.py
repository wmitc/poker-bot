import numpy as np
import pytest

from poker.card import (
    FULL_DECK,
    NUM_CARDS,
    Card,
    Deck,
    draw,
    parse_cards,
    remaining_deck,
)


def test_card_from_str_roundtrip():
    for code in range(NUM_CARDS):
        card = Card.from_code(code)
        assert Card.from_str(str(card)) == card
        assert card.code == code


def test_card_from_str_case_insensitive():
    assert Card.from_str("ah") == Card.from_str("AH") == Card.from_str("Ah")
    assert Card.from_str("Td").rank == 10
    assert Card.from_str("Ah").rank == 14  # ace high


@pytest.mark.parametrize("bad", ["", "A", "Ahh", "Xh", "As!", "1h"])
def test_card_from_str_rejects_invalid(bad):
    with pytest.raises(ValueError):
        Card.from_str(bad)


def test_card_rejects_out_of_range():
    with pytest.raises(ValueError):
        Card(1, 0)
    with pytest.raises(ValueError):
        Card(15, 0)
    with pytest.raises(ValueError):
        Card(10, 4)


def test_card_is_hashable_and_immutable():
    from dataclasses import FrozenInstanceError

    c = Card.from_str("Ks")
    assert c in {c}
    with pytest.raises(FrozenInstanceError):
        c.rank = 3  # type: ignore[misc]


def test_full_deck_is_52_unique():
    assert len(FULL_DECK) == NUM_CARDS
    assert len(set(FULL_DECK)) == NUM_CARDS


def test_parse_cards_formats():
    expected = [Card.from_str("Ah"), Card.from_str("Ks"), Card.from_str("Qd")]
    assert parse_cards("AhKsQd") == expected
    assert parse_cards("Ah Ks Qd") == expected
    assert parse_cards("Ah, Ks, Qd") == expected
    assert parse_cards(["Ah", "Ks", "Qd"]) == expected


def test_parse_cards_odd_length_raises():
    with pytest.raises(ValueError):
        parse_cards("AhK")


def test_remaining_deck_excludes():
    blocked = parse_cards("AhKs")
    rem = remaining_deck(blocked)
    assert len(rem) == NUM_CARDS - 2
    assert all(c not in blocked for c in rem)


def test_deck_shuffle_is_reproducible():
    d1, d2 = Deck(seed=42), Deck(seed=42)
    d1.shuffle()
    d2.shuffle()
    assert d1.cards == d2.cards
    assert Deck(seed=1).cards == list(FULL_DECK)  # unshuffled is canonical order


def test_deck_deal_consumes_cards():
    d = Deck(seed=7)
    d.shuffle()
    hand = d.deal(5)
    assert len(hand) == 5
    assert len(d.cards) == NUM_CARDS - 5
    assert all(c not in d.cards for c in hand)


def test_draw_excludes_and_is_seeded():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    blocked = parse_cards("AhKsQc")
    a = draw(rng1, 5, blocked)
    b = draw(rng2, 5, blocked)
    assert a == b  # same seed -> same draw
    assert len(set(a)) == 5
    assert all(c not in blocked for c in a)


def test_draw_too_many_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw(rng, 51, parse_cards("AhKs"))
