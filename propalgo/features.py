"""Price features: ATR(14), z-score, Kaufman Efficiency Ratio(50).

These are implemented exactly as specified. Functions accept numpy arrays (or
pandas Series, which are coerced) and return numpy arrays of the same length so
they line up bar-for-bar with the input series. Indices that cannot be computed
yet (insufficient history) follow the rules stated in the spec.
"""

from __future__ import annotations

import numpy as np

from .config import ATR_PERIOD, ER_PERIOD


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def true_range(high, low, close) -> np.ndarray:
    """True Range per bar.

    TR = max(high-low, |high-prev_close|, |low-prev_close|).
    For the first bar (no prev_close) TR = high-low.
    """
    high = _as_float_array(high)
    low = _as_float_array(low)
    close = _as_float_array(close)

    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)

    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = hl[0]  # no previous close on the first bar
    return tr


def atr(high, low, close, period: int = ATR_PERIOD) -> np.ndarray:
    """ATR(14) in price units: rolling mean of True Range, min_periods=1.

    Using min_periods=1 means early bars report the running mean of however many
    TR values exist so far (never NaN after the first bar).
    """
    tr = true_range(high, low, close)
    n = len(tr)
    out = np.empty(n, dtype=float)
    # Expanding mean until we have `period` values, then trailing window mean.
    csum = np.cumsum(tr)
    for i in range(n):
        if i < period:
            out[i] = csum[i] / (i + 1)
        else:
            out[i] = (csum[i] - csum[i - period]) / period
    return out


def zscore(window) -> np.ndarray:
    """Z-score of a 1-D window: (w - mean) / std.

    If std < 1e-8 the window is (numerically) flat and we return zeros, which
    keeps the z-norm distance well defined for matrix-profile matching.
    """
    w = _as_float_array(window)
    mu = w.mean()
    sd = w.std()
    if sd < 1e-8:
        return np.zeros_like(w)
    return (w - mu) / sd


def efficiency_ratio(close, period: int = ER_PERIOD) -> np.ndarray:
    """Kaufman Efficiency Ratio over `period` bars.

    ER[t] = |close[t] - close[t-period]| / sum_{i=t-period+1..t} |close[i]-close[i-1]|

    ER ~ 0 => ranging (tradeable); ER ~ 1 => trending (avoid).
    Bars with insufficient history, or a zero denominator (perfectly flat),
    return NaN so the regime gate naturally rejects them.
    """
    close = _as_float_array(close)
    n = len(close)
    out = np.full(n, np.nan, dtype=float)

    abs_diff = np.abs(np.diff(close))  # length n-1; abs_diff[i] = |c[i+1]-c[i]|
    # Prefix sums of absolute one-bar moves for O(1) window sums.
    pref = np.concatenate([[0.0], np.cumsum(abs_diff)])  # pref[k] = sum of first k diffs

    for t in range(period, n):
        direction = abs(close[t] - close[t - period])
        # Sum of |c[i]-c[i-1]| for i in (t-period+1 .. t)
        volatility = pref[t] - pref[t - period]
        if volatility <= 1e-12:
            out[t] = np.nan
        else:
            out[t] = direction / volatility
    return out
