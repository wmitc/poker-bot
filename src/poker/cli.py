"""Command-line interface.

Two subcommands:

* ``poker equity`` -- estimate hero's equity vs a hand or range.
* ``poker pushfold`` -- the heads-up shove/fold decision for a hand and stack.
"""

from __future__ import annotations

import argparse
import sys

from .bot import pushfold_ev
from .equity import equity


def _add_equity_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("equity", help="Monte Carlo equity of a hand vs a hand or range")
    p.add_argument("--hero", required=True, help="hero's two cards, e.g. AhKh")
    p.add_argument("--vs", required=True, help='opponent hand or range, e.g. "QsQc" or "QQ+, AKs"')
    p.add_argument("--board", default="", help="known board cards, e.g. Ah7d2c")
    p.add_argument("--sims", type=int, default=100_000, help="number of simulations")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")


def _add_pushfold_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("pushfold", help="heads-up push/fold decision for a hand")
    p.add_argument("--hero", required=True, help="hero's two cards, e.g. AsKs")
    p.add_argument("--stack", type=float, required=True, help="effective stack in big blinds")
    p.add_argument(
        "--call-range",
        default="22+, A2s+, A7o+, K9s+, KTo+, QTs+, JTs",
        help="villain's assumed calling range",
    )
    p.add_argument("--sims", type=int, default=20_000, help="simulations for equity-vs-call")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")


def _run_equity(args: argparse.Namespace) -> None:
    res = equity(args.hero, args.vs, board=args.board, n=args.sims, seed=args.seed)
    board = f" on {args.board}" if args.board else ""
    print(f"{args.hero} vs {args.vs}{board}")
    print(f"  {res}")
    print(f"  win {res.win:.3f}  tie {res.tie:.3f}  loss {res.loss:.3f}")


def _run_pushfold(args: argparse.Namespace) -> None:
    d = pushfold_ev(
        args.hero, args.call_range, args.stack, n=args.sims, seed=args.seed
    )
    print(f"{args.hero} at {args.stack}bb vs calling range [{args.call_range}]")
    print(f"  decision: {d.action.upper()}")
    print(f"  EV(push) {d.ev_push:+.3f}bb   EV(fold) {d.ev_fold:+.3f}bb")
    print(f"  villain calls {d.call_freq:.1%} of hands; equity when called {d.equity_vs_call:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_equity_parser(sub)
    _add_pushfold_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "equity":
            _run_equity(args)
        elif args.command == "pushfold":
            _run_pushfold(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
