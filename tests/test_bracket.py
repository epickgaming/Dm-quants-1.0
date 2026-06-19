"""Unit tests for bracket simulation and direction inference."""

import numpy as np
import pytest

from propalgo.bracket import infer_direction, simulate_bracket


def _flat(n, price=100.0):
    return np.full(n, price)


def test_long_hits_target():
    n = 60
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    atr_value = 1.0  # SL=2, TP=3
    # bar 5 spikes up to target (103+)
    high[5] = 103.5
    res = simulate_bracket(high, low, close, entry_idx=0, direction=1,
                           atr_value=atr_value, cost_frac=0.0)
    assert res.exit_reason == "target"
    assert res.win is True
    assert res.r_multiple == pytest.approx(1.5, abs=1e-9)  # 3/2 net of zero cost


def test_long_hits_stop():
    n = 60
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    low[3] = 97.5  # below stop at 98
    res = simulate_bracket(high, low, close, 0, 1, atr_value=1.0, cost_frac=0.0)
    assert res.exit_reason == "stop"
    assert res.win is False
    assert res.r_multiple == pytest.approx(-1.0, abs=1e-9)


def test_stop_first_when_both_touched():
    n = 60
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    high[2] = 104.0   # would be target
    low[2] = 97.0     # would be stop -- same bar
    res = simulate_bracket(high, low, close, 0, 1, atr_value=1.0, cost_frac=0.0)
    assert res.exit_reason == "stop"  # conservative: stop assumed first


def test_time_stop_exit_at_close():
    n = 60
    close = _flat(n, 100.0)
    high = _flat(n, 100.4)  # never reaches TP (103) or SL (98)
    low = _flat(n, 99.6)
    close[48] = 100.5  # the 48th bar close
    res = simulate_bracket(high, low, close, 0, 1, atr_value=1.0, cost_frac=0.0,
                           time_stop=48)
    assert res.exit_reason == "time"
    assert res.bars_held == 48


def test_cost_reduces_r():
    n = 60
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    high[5] = 103.5
    # cost_frac 0.001 of price 100 = 0.1 price; sl_distance=2 -> 0.05 R reduction
    res = simulate_bracket(high, low, close, 0, 1, atr_value=1.0, cost_frac=0.001)
    assert res.r_multiple == pytest.approx(1.5 - 0.05, abs=1e-9)


def test_infer_direction_up():
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108], dtype=float)
    # entry at 0, +8 bars rises -> +1
    assert infer_direction(close, [0], horizon=8) == 1


def test_infer_direction_down():
    close = np.array([108, 107, 106, 105, 104, 103, 102, 101, 100], dtype=float)
    assert infer_direction(close, [0], horizon=8) == -1


def test_short_hits_target():
    n = 60
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    low[4] = 96.5  # short target at 97
    res = simulate_bracket(high, low, close, 0, direction=-1, atr_value=1.0,
                           cost_frac=0.0)
    assert res.exit_reason == "target"
    assert res.win is True
