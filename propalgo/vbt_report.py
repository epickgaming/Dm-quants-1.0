"""VectorBT cross-check for the big-data sweep.

Our bracket simulation is path-dependent (intrabar stop-before-target ordering,
time stop), so the trade outcomes are produced by the custom engine in
`backtest.py`. VectorBT is then used as an INDEPENDENT, vectorized verifier of
the portfolio-level statistics (Sharpe, max drawdown, total return) computed
from the realised equity curve -- exactly the "fast on huge data" role the spec
recommends it for. If vectorbt is not installed this module degrades gracefully.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt  # noqa: F401
    _HAS_VBT = True
except Exception:  # pragma: no cover
    vbt = None
    _HAS_VBT = False


def equity_to_returns(equity_curve: pd.DataFrame, account_start: float) -> pd.Series:
    """Convert the event-time equity curve into a returns series."""
    if equity_curve is None or len(equity_curve) == 0:
        return pd.Series(dtype=float)
    s = equity_curve.copy()
    s["time"] = pd.to_datetime(s["time"])
    s = s.set_index("time")["equity"]
    s = pd.concat([pd.Series([account_start], index=[s.index[0] - pd.Timedelta(hours=1)]), s])
    return s.pct_change().dropna()


def vbt_stats(equity_curve: pd.DataFrame, account_start: float,
              freq: str = "h") -> Optional[Dict[str, float]]:
    """Independent VectorBT portfolio statistics from the equity curve.

    Returns None if vectorbt is unavailable or there is nothing to analyse.
    """
    if not _HAS_VBT:
        return None
    rets = equity_to_returns(equity_curve, account_start)
    if len(rets) < 3:
        return None
    acc = rets.vbt.returns(freq=freq)
    try:
        return {
            "vbt_total_return": float(acc.total()),
            "vbt_sharpe": float(acc.sharpe_ratio()),
            "vbt_max_drawdown": float(acc.max_drawdown()),
        }
    except Exception:  # pragma: no cover
        return None
