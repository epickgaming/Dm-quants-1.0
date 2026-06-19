"""End-to-end pipeline smoke test on a small synthetic dataset.

Kept deliberately small (one instrument, short history) so it runs in CI in a
few seconds while still exercising discovery -> selection -> walk-forward ->
portfolio simulation and the signals generator.
"""

import numpy as np

from propalgo.backtest import (
    _build_instrument,
    apply_weekly_cap,
    simulate_portfolio,
    walk_forward_instrument,
)
from propalgo.config import StrategyConfig
from propalgo.data import SyntheticSpec, generate_synthetic
from propalgo.signals import generate_signals


def test_walk_forward_produces_positive_edge():
    spec = SyntheticSpec(months=26, seed=7)
    df = generate_synthetic("XAUUSD", spec)
    cfg = StrategyConfig()
    idata = _build_instrument("XAUUSD", df, cfg)
    cands = walk_forward_instrument(idata, cfg)

    assert len(cands) > 20, "expected the planted edge to generate signals"
    r = np.array([c.r_multiple for c in cands])
    assert r.mean() > 0.05, f"expected positive mean R, got {r.mean():.3f}"

    # ER gate respected for every emitted candidate
    assert all(c.er <= cfg.er_gate for c in cands)

    # portfolio sim runs and stays within prop-firm drawdown limits
    capped = apply_weekly_cap(cands, cfg.weekly_trade_cap)
    res = simulate_portfolio(capped, cfg)
    assert res.metrics["trades"] > 0
    assert res.metrics["max_drawdown"] < cfg.max_total_drawdown
    assert np.isfinite(res.metrics["t_stat"])


def test_signals_generation_schema():
    spec = SyntheticSpec(months=22, seed=7)
    df = generate_synthetic("XAUUSD", spec)
    cfg = StrategyConfig()
    sigs = generate_signals({"XAUUSD": df}, cfg, scan_bars=80)
    # may or may not have rows depending on recent bars; schema must hold
    expected = ["symbol", "timeframe", "direction", "entry", "stop",
                "target", "atr", "risk_pct", "er", "asof"]
    assert list(sigs.columns) == expected
    for _, row in sigs.iterrows():
        assert row["direction"] in ("BUY", "SELL")
        if row["direction"] == "BUY":
            assert row["stop"] < row["entry"] < row["target"]
        else:
            assert row["target"] < row["entry"] < row["stop"]
        assert row["er"] <= cfg.er_gate
