import pytest

from poker.card import parse_cards
from poker.ranges import expand_handclass, parse_range


def n_combos(text: str) -> int:
    return len(parse_range(text))


# --- combo counts for single tokens --------------------------------------------

def test_pair_has_six_combos():
    assert n_combos("AA") == 6


def test_suited_has_four_combos():
    assert n_combos("AKs") == 4


def test_offsuit_has_twelve_combos():
    assert n_combos("AKo") == 12


def test_unspecified_has_sixteen_combos():
    assert n_combos("AK") == 16
    assert n_combos("AK") == n_combos("AKs") + n_combos("AKo")


def test_suited_combos_share_suit():
    for a, b in parse_range("AKs"):
        assert a.suit == b.suit


def test_offsuit_combos_differ_in_suit():
    for a, b in parse_range("AKo"):
        assert a.suit != b.suit


# --- plus expansion ------------------------------------------------------------

def test_pair_plus():
    # QQ, KK, AA
    assert n_combos("QQ+") == 18
    assert n_combos("22+") == 13 * 6  # all pairs


def test_suited_kicker_plus():
    # ATs, AJs, AQs, AKs -> 4 hand classes x 4 combos
    assert n_combos("ATs+") == 16


def test_unspecified_kicker_plus():
    assert n_combos("AT+") == n_combos("ATs+") + n_combos("ATo+")


# --- span expansion ------------------------------------------------------------

def test_pair_span():
    assert n_combos("22-99") == 8 * 6  # 22..99


def test_connector_span():
    # T9s, 98s, 87s, 76s, 65s
    assert n_combos("T9s-65s") == 5 * 4


def test_span_endpoints_order_insensitive():
    assert parse_range("22-99") == parse_range("99-22")


@pytest.mark.parametrize("bad", ["T9s-66s", "T9s-65o", "AA-AKs"])
def test_invalid_spans_raise(bad):
    with pytest.raises(ValueError):
        parse_range(bad)


# --- composition, dedup, exclusion ---------------------------------------------

def test_multiple_tokens_combine_and_dedup():
    combos = parse_range("AKs, AKs, AKo")  # duplicate AKs collapses
    assert len(combos) == 16


def test_full_enumeration_is_1326():
    # Build the whole range explicitly: all pairs and all unspecified non-pairs.
    ranks = "23456789TJQKA"
    tokens = [r + r for r in ranks]
    for i, hi in enumerate(ranks):
        for lo in ranks[:i]:
            tokens.append(hi + lo)
    assert n_combos(",".join(tokens)) == 1326


def test_exclude_blocks_cards():
    blocked = parse_cards("AsKs")
    combos = parse_range("AKs", exclude=blocked)
    # AsKs is removed; the other three suited AK combos remain.
    assert len(combos) == 3
    assert all(blocked[0] not in c and blocked[1] not in c for c in combos)


def test_expand_handclass_pair_count():
    assert len(expand_handclass((14, 14, None))) == 6
