"""Command-line entry point for the PropAlgo pipeline.

Subcommands:

    fetch       Download/refresh H1 data into the parquet cache (dukascopy).
    backtest    Anchored walk-forward over the basket; prints acceptance metrics
                and a cost-sensitivity table; writes an equity curve CSV.
    signals     Train on trailing 18mo and emit signals.csv for the EA.

Everything runs offline on synthetic data with --synthetic (the default when no
cache exists), so CI and first-time users get a working pipeline with no network.
Real orders require nothing here -- live trading happens in the EA, which only
*reads* signals.csv. The --live flag merely affects which data source is used.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd

from .backtest import run_walk_forward
from .config import StrategyConfig, load_overrides
from .data import generate_synthetic_basket, get_data, save_cache, SyntheticSpec
from .signals import generate_signals, write_signals


def _make_cfg(args) -> StrategyConfig:
    cfg = StrategyConfig()
    if getattr(args, "config", None):
        load_overrides(cfg, args.config)
        print(f"[config] applied overrides from {args.config}")
    return cfg


def _load_basket(args, cfg: StrategyConfig) -> Dict[str, pd.DataFrame]:
    if args.synthetic:
        print("[data] using synthetic planted-signal dataset (offline)")
        spec = SyntheticSpec(months=args.months)
        return generate_synthetic_basket(cfg.instruments, spec)

    data = {}
    for inst in cfg.instruments:
        try:
            df = get_data(inst, data_dir=args.data_dir, start=args.start,
                          fetch_missing=not args.no_fetch)
            data[inst] = df
            print(f"[data] {inst}: {len(df)} bars")
        except Exception as exc:
            print(f"[data] {inst}: unavailable ({exc})")
    if not data:
        print("[data] no real data available; falling back to synthetic")
        spec = SyntheticSpec(months=args.months)
        return generate_synthetic_basket(cfg.instruments, spec)
    return data


def _print_metrics(title, m):
    print(f"\n=== {title} ===")
    keys = ["trades", "trades_per_week", "win_rate", "realized_rrr", "mean_r",
            "t_stat", "total_return", "max_drawdown", "worst_day", "final_equity"]
    for k in keys:
        if k in m:
            v = m[k]
            print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k:18s}: {v}")


def _per_instrument_report(result, cfg):
    print("\n=== per-instrument (pre-cap) candidate stats ===")
    import numpy as np
    for inst, cands in result["per_instrument"].items():
        if not cands:
            print(f"  {inst:8s}: no signals")
            continue
        r = np.array([c.r_multiple for c in cands])
        w = np.array([c.win for c in cands])
        print(f"  {inst:8s}: n={len(cands):4d}  win={w.mean():.3f}  meanR={r.mean():+.3f}")


def cmd_backtest(args):
    cfg = _make_cfg(args)
    data = _load_basket(args, cfg)
    result = run_walk_forward(data, cfg, verbose=True)

    base_m = result["baseline"].metrics
    _per_instrument_report(result, cfg)
    _print_metrics("COMBINED walk-forward (config cost)", base_m)

    print("\n=== cost sensitivity (extra slippage) ===")
    print(f"  {'extra_bp':>8s} {'win_rate':>9s} {'mean_r':>8s} {'t_stat':>8s} "
          f"{'tot_ret':>8s} {'max_dd':>8s}")
    for bp, m in result["sensitivity"].items():
        print(f"  {bp:8.1f} {m['win_rate']:9.3f} {m['mean_r']:8.3f} "
              f"{m['t_stat']:8.2f} {m['total_return']:8.3f} {m['max_drawdown']:8.3f}")

    # Honesty flag: edge dies near +2bp.
    s2 = result["sensitivity"].get(2.0)
    if s2 and (s2["t_stat"] < 2.0 or s2["mean_r"] <= 0):
        print("\n  [!] WARNING: the edge does NOT survive +2bp of extra cost.")
        print("      You MUST re-measure your broker's real round-trip cost; if it")
        print("      exceeds the config values, this strategy is not tradeable.")

    # VectorBT independent cross-check of portfolio stats
    from .vbt_report import vbt_stats
    vstats = vbt_stats(result["baseline"].equity_curve, cfg.account_start)
    if vstats:
        print("\n=== VectorBT cross-check (independent) ===")
        for k, v in vstats.items():
            print(f"  {k:18s}: {v:.4f}")

    # last-6-months OOS focus
    six = _six_month_report(result, cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    eq_path = os.path.join(args.out_dir, "equity_curve.csv")
    result["baseline"].equity_curve.to_csv(eq_path, index=False)
    tr_path = os.path.join(args.out_dir, "trades.csv")
    result["baseline"].trades.to_csv(tr_path, index=False)
    rpt_path = os.path.join(args.out_dir, "backtest_report.md")
    _write_report(rpt_path, result, base_m, six, vstats, cfg)
    print(f"\n[out] equity curve -> {eq_path}")
    print(f"[out] trades       -> {tr_path}")
    print(f"[out] report       -> {rpt_path}")


def _six_month_report(result, cfg):
    from .backtest import apply_weekly_cap, simulate_portfolio
    cands = result["candidates"]
    if not cands:
        return
    last = max(c.entry_time for c in cands)
    cutoff = last - pd.DateOffset(months=6)
    recent = [c for c in cands if c.entry_time >= cutoff]
    capped = apply_weekly_cap(recent, cfg.weekly_trade_cap)
    res = simulate_portfolio(capped, cfg)
    _print_metrics("LAST 6 MONTHS out-of-sample (config cost)", res.metrics)
    return res.metrics


def _fmt(m, k):
    v = m.get(k, float("nan"))
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _write_report(path, result, base_m, six_m, vstats, cfg):
    lines = []
    lines.append("# PropAlgo backtest report\n")
    lines.append("Regime-filtered, multi-instrument mean-reversion (H1). "
                 "Anchored walk-forward, all profits NET of costs.\n")
    lines.append("## Combined walk-forward (config cost)\n")
    lines.append("| metric | value |\n|---|---|")
    for k in ["trades", "trades_per_week", "win_rate", "realized_rrr", "mean_r",
              "t_stat", "total_return", "max_drawdown", "worst_day", "final_equity"]:
        lines.append(f"| {k} | {_fmt(base_m, k)} |")
    if six_m:
        lines.append("\n## Last 6 months out-of-sample (config cost)\n")
        lines.append("| metric | value |\n|---|---|")
        for k in ["trades", "trades_per_week", "win_rate", "realized_rrr", "mean_r",
                  "t_stat", "total_return", "max_drawdown", "worst_day", "final_equity"]:
            lines.append(f"| {k} | {_fmt(six_m, k)} |")
    lines.append("\n## Cost sensitivity (extra slippage)\n")
    lines.append("| extra_bp | win_rate | mean_r | t_stat | total_return | max_drawdown |\n|---|---|---|---|---|---|")
    for bp, m in result["sensitivity"].items():
        lines.append(f"| {bp:.1f} | {_fmt(m,'win_rate')} | {_fmt(m,'mean_r')} | "
                     f"{_fmt(m,'t_stat')} | {_fmt(m,'total_return')} | {_fmt(m,'max_drawdown')} |")
    if vstats:
        lines.append("\n## VectorBT independent cross-check\n")
        lines.append("| metric | value |\n|---|---|")
        for k, v in vstats.items():
            lines.append(f"| {k} | {v:.4f} |")
    lines.append("\n## Per-instrument (pre-cap candidate stats)\n")
    lines.append("| instrument | n | win_rate | mean_r |\n|---|---|---|---|")
    for inst, cands in result["per_instrument"].items():
        if not cands:
            lines.append(f"| {inst} | 0 | - | - |")
            continue
        r = np.array([c.r_multiple for c in cands])
        w = np.array([c.win for c in cands])
        lines.append(f"| {inst} | {len(cands)} | {w.mean():.3f} | {r.mean():+.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_signals(args):
    cfg = _make_cfg(args)
    data = _load_basket(args, cfg)
    df = generate_signals(data, cfg, scan_bars=args.scan_bars)
    path = args.out or os.path.join(args.out_dir, "signals.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_signals(df, path)
    print(f"[signals] wrote {len(df)} signal(s) -> {path}")
    if len(df):
        print(df.to_string(index=False))
    if args.live:
        print("[signals] --live set: EA may act on these on a real/demo account.")
    else:
        print("[signals] paper mode (default). Pass --live to authorise real orders in the EA.")


def cmd_fetch(args):
    cfg = _make_cfg(args)
    for inst in cfg.instruments:
        try:
            df = get_data(inst, data_dir=args.data_dir, start=args.start,
                          fetch_missing=True)
            save_cache(df, args.data_dir, inst)
            print(f"[fetch] {inst}: {len(df)} bars cached")
        except Exception as exc:
            print(f"[fetch] {inst}: FAILED ({exc})")


def build_parser():
    p = argparse.ArgumentParser(prog="propalgo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data")
    common.add_argument("--out-dir", default="signals_out")
    common.add_argument("--synthetic", action="store_true",
                        help="use the offline synthetic planted-signal dataset")
    common.add_argument("--months", type=int, default=36,
                        help="months of synthetic history to generate")
    common.add_argument("--start", default="2018-01-01")
    common.add_argument("--no-fetch", action="store_true",
                        help="use only cached data, do not hit the network")
    common.add_argument("--config", default=None,
                        help="YAML/JSON file overriding per-instrument costs and "
                             "the broker symbol map (strategy params stay locked)")

    bt = sub.add_parser("backtest", parents=[common], help="walk-forward backtest")
    bt.set_defaults(func=cmd_backtest)

    sg = sub.add_parser("signals", parents=[common], help="generate signals.csv")
    sg.add_argument("--scan-bars", type=int, default=5)
    sg.add_argument("--out", default=None)
    sg.add_argument("--live", action="store_true",
                    help="authorise the EA to place real orders from these signals")
    sg.set_defaults(func=cmd_signals)

    ft = sub.add_parser("fetch", parents=[common], help="refresh the data cache")
    ft.set_defaults(func=cmd_fetch)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
