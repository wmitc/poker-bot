import random

import pytest

from poker.card import FULL_DECK, Card, parse_cards
from poker.evaluator import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    best_of,
    category_of,
    evaluate5,
    evaluate7,
)


def h(s: str) -> list[Card]:
    return parse_cards(s)


# --- category detection on canonical five-card hands ---------------------------

CATEGORY_EXAMPLES = [
    ("AhKhQhJhTh", STRAIGHT_FLUSH),
    ("5h4h3h2hAh", STRAIGHT_FLUSH),  # wheel straight flush
    ("9s9h9d9cKs", FOUR_OF_A_KIND),
    ("8s8h8dKsKh", FULL_HOUSE),
    ("Ah9h7h4h2h", FLUSH),
    ("Ts9d8h7s6c", STRAIGHT),
    ("5h4d3c2sAh", STRAIGHT),  # wheel
    ("QsQhQd9c2s", THREE_OF_A_KIND),
    ("JsJhTdTcKs", TWO_PAIR),
    ("7s7hKd9c2s", ONE_PAIR),
    ("Ah9d7c5s3h", HIGH_CARD),
]


@pytest.mark.parametrize("cards,category", CATEGORY_EXAMPLES)
def test_evaluate5_category(cards, category):
    assert category_of(evaluate5(h(cards))) == category


def test_category_strict_ordering():
    # One representative hand per category, in increasing strength.
    ladder = [
        "Ah9d7c5s3h",  # high card
        "7s7hKd9c2s",  # pair
        "JsJhTdTcKs",  # two pair
        "QsQhQd9c2s",  # trips
        "Ts9d8h7s6c",  # straight
        "Ah9h7h4h2h",  # flush
        "8s8h8dKsKh",  # full house
        "9s9h9d9cKs",  # quads
        "AhKhQhJhTh",  # straight flush
    ]
    scores = [evaluate5(h(c)) for c in ladder]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)


# --- tie-breaks ----------------------------------------------------------------

def test_higher_kicker_wins():
    assert evaluate5(h("AhAdKsQc7h")) > evaluate5(h("AhAdKsJc7h"))


def test_straight_high_card_breaks_tie():
    assert evaluate5(h("Ts9d8h7s6c")) > evaluate5(h("9s8d7h6s5c"))


def test_wheel_is_lowest_straight():
    assert evaluate5(h("6s5d4h3s2c")) > evaluate5(h("5h4d3c2sAh"))


def test_full_house_compares_trips_first():
    assert evaluate5(h("KsKhKd2c2s")) > evaluate5(h("QsQhQdAcAs"))


def test_flush_compares_high_cards():
    assert evaluate5(h("AhQh8h5h3h")) > evaluate5(h("KhQh8h5h3h"))


# --- evaluate7 specifics -------------------------------------------------------

def test_evaluate7_picks_best_five():
    # Board makes a flush; hole cards add nothing better.
    score = evaluate7(h("AhKh") + h("Qh7h2h3s4d"))
    assert category_of(score) == FLUSH


def test_evaluate7_two_trips_makes_full_house():
    # Two sets of trips -> full house using the higher trips and lower as pair.
    score = evaluate7(h("KsKhKd2c2s2h9d"))
    assert category_of(score) == FULL_HOUSE
    assert score == evaluate7(h("KsKhKd2c2s9d"))  # the 7th card is irrelevant here


def test_evaluate7_three_pairs_uses_top_two():
    score = evaluate7(h("AsAhKsKhQsQh2d"))
    assert category_of(score) == TWO_PAIR
    # Best two pair is aces and kings with a queen kicker.
    assert score == evaluate5(h("AsAhKsKhQs"))


def test_evaluate7_quads_kicker():
    strong = evaluate7(h("9s9h9d9cAhKh2s"))
    weak = evaluate7(h("9s9h9d9cQhKh2s"))
    assert category_of(strong) == FOUR_OF_A_KIND
    assert strong > weak  # ace kicker beats queen kicker


def test_evaluate7_accepts_five_and_six_cards():
    assert evaluate7(h("AhKhQhJhTh")) == best_of(h("AhKhQhJhTh"))
    assert evaluate7(h("AhKhQhJhTh2s")) == best_of(h("AhKhQhJhTh2s"))


# --- the cross-check: evaluate7 must match the brute-force oracle ---------------

def test_evaluate7_matches_bruteforce_random():
    rng = random.Random(2024)
    deck = list(FULL_DECK)
    for _ in range(3000):
        hand = rng.sample(deck, 7)
        assert evaluate7(hand) == best_of(hand), [str(c) for c in hand]


def test_evaluate7_matches_bruteforce_six_cards():
    rng = random.Random(99)
    deck = list(FULL_DECK)
    for _ in range(1000):
        hand = rng.sample(deck, 6)
        assert evaluate7(hand) == best_of(hand), [str(c) for c in hand]
