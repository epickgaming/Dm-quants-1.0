"""Bracket-trade simulation and direction inference.

A trade enters at the window-end close, brackets at SL = 2*ATR and TP = 3*ATR
(1:1.5), and is walked forward bar by bar:

  * if a bar touches the stop -> loss (if a bar touches BOTH stop and target in
    the same bar, the stop is assumed first -- the conservative assumption);
  * if a bar touches the target -> win;
  * otherwise, at the time-stop (48th bar) exit at that bar's close.

The per-instrument round-trip cost (a fraction of price) is subtracted from the
PnL of every trade. The result is reported as an R-multiple = pnl / sl_distance,
so +1.5 is a clean target hit and -1.0 a clean stop, both NET of cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DIRECTION_HORIZON, SL_ATR_MULT, TIME_STOP_BARS, TP_ATR_MULT


@dataclass
class TradeResult:
    direction: int          # +1 long, -1 short
    entry_price: float
    sl_distance: float      # price distance to stop (2*ATR)
    tp_distance: float      # price distance to target (3*ATR)
    exit_reason: str        # "stop" | "target" | "time"
    bars_held: int
    r_multiple: float       # pnl / sl_distance, net of cost
    win: bool


def infer_direction(
    close: np.ndarray, occurrences, horizon: int = DIRECTION_HORIZON
) -> int:
    """Fixed direction from TRAINING occurrences only (no look-ahead).

    For each training occurrence entry index e, compute the forward return over
    `horizon` bars r = (close[e+horizon]-close[e]) / close[e]. Direction is +1
    if the mean of those returns is >= 0, else -1. Occurrences without enough
    forward bars are skipped. With no usable occurrences we default to +1.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    rets = []
    for e in occurrences:
        if e + horizon < n and close[e] != 0:
            rets.append((close[e + horizon] - close[e]) / close[e])
    if not rets:
        return 1
    return 1 if np.mean(rets) >= 0 else -1


def simulate_bracket(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: int,
    direction: int,
    atr_value: float,
    cost_frac: float,
    sl_mult: float = SL_ATR_MULT,
    tp_mult: float = TP_ATR_MULT,
    time_stop: int = TIME_STOP_BARS,
) -> TradeResult | None:
    """Simulate one bracket trade entered at the close of bar `entry_idx`.

    Returns None if there is not enough data or the ATR is non-positive.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)

    if atr_value <= 0 or entry_idx >= n - 1:
        return None

    entry_price = close[entry_idx]
    sl_distance = sl_mult * atr_value
    tp_distance = tp_mult * atr_value

    if direction > 0:
        stop_price = entry_price - sl_distance
        target_price = entry_price + tp_distance
    else:
        stop_price = entry_price + sl_distance
        target_price = entry_price - tp_distance

    # Round-trip cost expressed in price units (fraction of entry price).
    cost_price = cost_frac * entry_price

    last_bar = min(entry_idx + time_stop, n - 1)
    for i in range(entry_idx + 1, last_bar + 1):
        hi, lo = high[i], low[i]
        if direction > 0:
            hit_stop = lo <= stop_price
            hit_target = hi >= target_price
        else:
            hit_stop = hi >= stop_price
            hit_target = lo <= target_price

        if hit_stop:  # conservative: stop checked first when both touch
            pnl = -sl_distance - cost_price
            return TradeResult(
                direction, entry_price, sl_distance, tp_distance, "stop",
                i - entry_idx, pnl / sl_distance, False,
            )
        if hit_target:
            pnl = tp_distance - cost_price
            return TradeResult(
                direction, entry_price, sl_distance, tp_distance, "target",
                i - entry_idx, pnl / sl_distance, True,
            )

    # Time stop: exit at the last bar's close.
    exit_price = close[last_bar]
    raw = (exit_price - entry_price) * direction
    pnl = raw - cost_price
    return TradeResult(
        direction, entry_price, sl_distance, tp_distance, "time",
        last_bar - entry_idx, pnl / sl_distance, pnl > 0,
    )
