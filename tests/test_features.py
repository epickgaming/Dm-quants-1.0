"""Unit tests for ATR, ER, and z-score."""

import numpy as np
import pytest

from propalgo.features import atr, efficiency_ratio, true_range, zscore


def test_true_range_first_bar_is_high_low():
    high = np.array([10.0, 11.0, 12.0])
    low = np.array([9.0, 10.0, 11.0])
    close = np.array([9.5, 10.5, 11.5])
    tr = true_range(high, low, close)
    assert tr[0] == pytest.approx(1.0)  # high-low on first bar
    # bar 1: max(11-10, |11-9.5|, |10-9.5|) = max(1, 1.5, 0.5) = 1.5
    assert tr[1] == pytest.approx(1.5)


def test_atr_min_periods_one_expanding_then_trailing():
    high = np.arange(1, 21, dtype=float) + 0.5
    low = np.arange(1, 21, dtype=float) - 0.5
    close = np.arange(1, 21, dtype=float)
    a = atr(high, low, close, period=14)
    assert len(a) == len(close)
    assert np.all(np.isfinite(a))      # never NaN with min_periods=1
    assert a[0] == pytest.approx(1.0)  # first bar TR = high-low = 1


def test_atr_constant_range_equals_range():
    n = 50
    close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    a = atr(high, low, close, period=14)
    # constant 2-wide bars, no gaps -> ATR converges to 2.0
    assert a[-1] == pytest.approx(2.0, abs=1e-9)


def test_zscore_basic():
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(w)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0, abs=1e-12)


def test_zscore_flat_returns_zeros():
    w = np.full(10, 7.0)
    z = zscore(w)
    assert np.allclose(z, 0.0)


def test_efficiency_ratio_trending_is_one():
    # perfectly trending line: |c[t]-c[t-50]| == sum of |diffs| -> ER == 1
    close = np.arange(0, 200, dtype=float)
    er = efficiency_ratio(close, period=50)
    assert er[100] == pytest.approx(1.0, abs=1e-9)


def test_efficiency_ratio_ranging_is_low():
    # oscillating series: net move tiny vs path length -> ER near 0
    x = np.arange(0, 300)
    close = 100 + np.sin(x / 2.0)
    er = efficiency_ratio(close, period=50)
    vals = er[~np.isnan(er)]
    assert np.nanmedian(vals) < 0.3


def test_efficiency_ratio_insufficient_history_is_nan():
    close = np.arange(0, 100, dtype=float)
    er = efficiency_ratio(close, period=50)
    assert np.all(np.isnan(er[:50]))
