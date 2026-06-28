# poker-equity

A Texas Hold'em **equity calculator** (hand vs. hand or hand vs. range via Monte Carlo) and a
**simple bot** that makes push/fold and pot-odds decisions. Built as a quant-portfolio project to
demonstrate **Monte Carlo estimation** (with convergence and standard-error analysis) and
**expected-value (EV) reasoning**.

## Why this project

Two ideas that come up constantly in quant interviews show up here in a concrete, testable form:

- **Monte Carlo methods** — estimate the probability of winning a hand by simulating thousands of
  random runouts, and quantify the estimate's uncertainty with a standard error / confidence
  interval. The [convergence notebook](notebooks/convergence.ipynb) shows the estimate converging
  and the standard error falling like `1/sqrt(N)` (fitted log-log slope ≈ −0.50).
- **EV reasoning** — turn an equity number into a decision. The bot compares equity to pot odds
  (call iff `equity > to_call / (pot + to_call)`) and evaluates the chip-EV of an all-in shove.

## Features

- From-scratch 7-card hand evaluator, validated against an exhaustive brute-force oracle.
- Monte Carlo equity: hand vs. hand or hand vs. range, on any board, with a 95% confidence
  interval.
- Exact enumeration for small remaining spaces, used to cross-check the Monte Carlo estimates.
- Hand-range notation parser (`QQ+`, `AKs`, `ATo+`, `22-99`, `T9s-65s`, ...).
- A push/fold + pot-odds bot, plus a 13×13 push/fold chart.

## Install

```bash
git clone https://github.com/wmitc/poker-bot
cd poker-bot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"
```

## Usage

### Command line

```bash
# Equity of a hand vs a hand
poker equity --hero AhKh --vs QsQc --sims 100000 --seed 1
#   AhKh vs QsQc
#     equity 0.4637 +/- 0.0035 (95% CI [0.4568, 0.4706], n=100000)
#     win 0.462  tie 0.004  loss 0.534

# Equity of a hand vs a range, on a flop
poker equity --hero AsKs --vs "QQ+, AKo" --board Ah7d2c --sims 100000

# Heads-up push/fold decision
poker pushfold --hero AsKs --stack 12
#   AsKs at 12.0bb vs calling range [...]
#     decision: PUSH
#     EV(push) +1.41bb   EV(fold) -0.50bb
```

### Library

```python
from poker.equity import equity, equity_exact
from poker.bot import pot_odds_decision, pushfold_ev

equity("AsAh", "KsKh", n=100_000, seed=1)        # AA vs KK ≈ 0.82
equity("AsKs", "QQ+, AKo", board="Ah7d2c")        # vs a range, on a flop
equity_exact("AsAh", "KsKh", board="2c7d9hQs")    # exact (river enumeration)

pot_odds_decision(equity_value=0.40, to_call=10, pot=20)   # call/fold from pot odds
pushfold_ev("AsKs", "22+, A2s+, KTo+", stack_bb=12)        # shove/fold EV
```

## How it works

- **Hand evaluator** (`poker.evaluator`) maps any 5–7 card hand to a single integer so hands
  compare with `>`. A direct 7-card evaluator is the Monte Carlo hot path; an independent
  brute-force evaluator serves as a test oracle.
- **Equity** (`poker.equity`) deals random runouts (and random opponent holdings from a range),
  scores the showdown, and averages hero's pot share. It reports the standard error and a 95%
  confidence interval; `equity_exact` enumerates outcomes when few cards remain.
- **Bot** (`poker.bot`) turns equity into decisions: pot-odds calls and the chip-EV of an open
  shove, `EV_push = (1 − f)·bb + f·(2e − 1)·stack`, where `f` is villain's call frequency and `e`
  hero's equity when called.

A note on performance: the evaluator is pure Python for clarity, which is the Monte Carlo
bottleneck (~30 µs per simulated showdown) — comfortably fast enough for `1e4`–`1e6` simulations.

## Development

```bash
ruff check .
pytest
```

## License

MIT
