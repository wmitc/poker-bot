# poker-equity

A Texas Hold'em **equity calculator** (hand vs. hand or hand vs. range via Monte Carlo) and a
**simple bot** that makes push/fold and pot-odds decisions. Built as a quant-portfolio project to
demonstrate **Monte Carlo estimation** (with convergence and standard-error analysis) and
**expected-value (EV) reasoning**.

> Status: under active development. See [the roadmap](#roadmap) below.

## Why this project

Two ideas that come up constantly in quant interviews show up here in a concrete, testable form:

- **Monte Carlo methods** — estimate the probability of winning a hand by simulating thousands of
  random runouts, and quantify the estimate's uncertainty with a standard error / confidence
  interval. The accompanying notebook shows the estimate converging as `O(1/sqrt(N))`.
- **EV reasoning** — turn an equity number into a decision. The bot compares equity to pot odds
  (call iff `equity > to_call / (pot + to_call)`) and evaluates the chip-EV of an all-in shove.

## Features (planned)

- From-scratch 7-card hand evaluator (validated against an exhaustive brute-force oracle).
- Monte Carlo equity: hand vs. hand, hand vs. range, with partial boards (flop/turn).
- Exact enumeration for small remaining spaces, used to cross-check the Monte Carlo estimates.
- Hand-range notation parser (`QQ+`, `AKs`, `ATo+`, `22-99`, ...).
- A push/fold + pot-odds bot.
- A convergence notebook visualising estimate vs. sample size and standard-error scaling.

## Install

```bash
git clone https://github.com/wmitc/poker-bot
cd poker-bot
pip install -e ".[dev]"
```

## Usage

CLI and library usage are documented here as the features land. Example (planned):

```bash
poker equity --hero AhKh --villain QsQc --sims 100000 --seed 1
```

## Development

```bash
ruff check .
pytest
```

A note on performance: the hand evaluator is pure Python for clarity, which is the Monte Carlo
bottleneck. It is comfortably fast enough for `1e4`–`1e6` simulations.

## Roadmap

1. Project scaffold
2. Cards & deck
3. Hand evaluator
4. Range parser
5. Monte Carlo equity
6. Push/fold + pot-odds bot
7. CLI
8. Convergence notebook

## License

MIT
