"""Live signal generation: the Python -> MT5 interface.

In live mode we train on the trailing 18 months of each instrument, then scan
the most recent bars for matches of the selected templates. Each fresh, regime-
valid match becomes a row in signals.csv with ABSOLUTE prices the EA can act on:

    symbol,timeframe,direction,entry,stop,target,atr,risk_pct,er,asof

`asof` is the entry bar's timestamp (ISO). The EA de-duplicates by symbol+asof.
The broker `symbol` is taken from the configurable symbol map.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import _build_instrument, _train_and_select
from .config import StrategyConfig
from .discovery import MatrixProfileDiscovery, match_template

SIGNAL_COLUMNS = [
    "symbol", "timeframe", "direction", "entry", "stop",
    "target", "atr", "risk_pct", "er", "asof",
]


def generate_signals(
    data: Dict[str, pd.DataFrame],
    cfg: Optional[StrategyConfig] = None,
    scan_bars: int = 5,
    discovery=None,
) -> pd.DataFrame:
    """Train on trailing 18mo per instrument and scan the last `scan_bars` bars
    for fresh, regime-valid signals. Returns a DataFrame in SIGNAL_COLUMNS order.
    """
    cfg = cfg or StrategyConfig()
    discovery = discovery or MatrixProfileDiscovery(
        window=cfg.window, max_motifs=cfg.max_motifs,
        min_neighbors=cfg.min_neighbors, max_distance=cfg.max_distance,
        cutoff=cfg.motif_cutoff, max_matches=cfg.max_matches,
        match_tolerance=cfg.match_tolerance,
    )

    rows: List[dict] = []
    for inst, df in data.items():
        if len(df) < cfg.window * 10:
            continue
        idata = _build_instrument(inst, df, cfg)
        times = pd.to_datetime(idata.times)
        n = len(idata.close)

        # trailing 18 months training window ending at the most recent bar
        last_time = times[-1]
        train_lo_date = last_time - pd.DateOffset(months=cfg.train_months)
        train_lo = int(np.searchsorted(times.values, np.datetime64(train_lo_date)))
        train_hi = n  # train through the latest bar
        if train_hi - train_lo < cfg.window * 5:
            continue

        survivors = _train_and_select(idata, cfg, train_lo, train_hi, discovery)
        scan_start = max(cfg.window, n - scan_bars)

        for tpl, direction in survivors:
            ext_lo = max(0, scan_start - (cfg.window - 1))
            seg = idata.close[ext_lo:n]
            occ = match_template(tpl.pattern, seg, max_distance=tpl.match_tolerance)
            for o in occ:
                e = ext_lo + int(o) + tpl.window - 1
                if e < scan_start or e >= n:
                    continue
                er_val = idata.er[e]
                if not np.isfinite(er_val) or er_val > cfg.er_gate:
                    continue
                atr_val = idata.atr[e]
                if atr_val <= 0:
                    continue
                entry = float(idata.close[e])
                sl_dist = cfg.sl_atr_mult * atr_val
                tp_dist = cfg.tp_atr_mult * atr_val
                if direction > 0:
                    stop, target = entry - sl_dist, entry + tp_dist
                    side = "BUY"
                else:
                    stop, target = entry + sl_dist, entry - tp_dist
                    side = "SELL"
                rows.append({
                    "symbol": cfg.symbol_map.get(inst, inst),
                    "timeframe": cfg.timeframe,
                    "direction": side,
                    "entry": round(entry, 5),
                    "stop": round(stop, 5),
                    "target": round(target, 5),
                    "atr": round(float(atr_val), 5),
                    "risk_pct": cfg.risk_pct,
                    "er": round(float(er_val), 4),
                    "asof": pd.Timestamp(times.values[e]).isoformat(),
                })

    df_out = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    if len(df_out):
        df_out = df_out.drop_duplicates(subset=["symbol", "asof"]).reset_index(drop=True)
    return df_out


def write_signals(df: pd.DataFrame, path: str) -> str:
    """Write signals to CSV (creating an empty, header-only file if no signals)."""
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=SIGNAL_COLUMNS)
    df.to_csv(path, index=False)
    return path
