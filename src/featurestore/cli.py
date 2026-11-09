"""`featurestore`: run the evaluation, or look up one customer's features as of a moment, offline and online."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import check, run, to_json, to_markdown
    from .report import render

    r = run(n_users=args.users, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(to_json(r), newline="\n", encoding="utf-8")
    (out / "evaluation.md").write_text(to_markdown(r), newline="\n", encoding="utf-8")
    (out / "report.html").write_text(render(r), newline="\n", encoding="utf-8")
    print(f"{r.labels:,} labelled moments ({r.seconds:.0f}s) -> {out}")
    if args.gate:
        problems = check(r)
        for p in problems:
            print(f"GATE FAILED: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    import numpy as np

    from . import data, stores
    from .features import FEATURES

    ev, _ = data.generate(args.users, args.seed)
    moment = data.Labels(np.array([args.customer]), np.array([args.at]), np.array([False]))
    print(f"customer {args.customer} as of day {args.at}")
    rows = {mode: stores.offline(ev, moment, FEATURES, mode)[0] for mode in ("point_in_time", "label_date", "snapshot")}
    online = stores.replay(ev, moment, stores.OnlineStore(FEATURES))[0]
    print(f"{'feature':18} {'online':>10} {'point in time':>14} {'label date':>11} {'snapshot':>10}")
    for k, f in enumerate(FEATURES):
        print(
            f"{f.name:18} {online[k]:>10.2f} {rows['point_in_time'][k]:>14.2f} {rows['label_date'][k]:>11.2f} "
            f"{rows['snapshot'][k]:>10.2f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="featurestore", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="consistency, leakage and skew detection")
    e.add_argument("--users", type=int, default=4000)
    e.add_argument("--seed", type=int, default=3)
    e.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    e.add_argument("--gate", action="store_true")
    e.set_defaults(fn=cmd_evaluate)
    lk = sub.add_parser("lookup", help="one customer's features, online and offline")
    lk.add_argument("customer", type=int)
    lk.add_argument("--at", type=float, default=80.0)
    lk.add_argument("--users", type=int, default=4000)
    lk.add_argument("--seed", type=int, default=3)
    lk.set_defaults(fn=cmd_lookup)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
