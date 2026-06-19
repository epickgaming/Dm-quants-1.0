"""PropAlgo: regime-filtered, multi-instrument mean-reversion trading algorithm.

A faithful implementation of the locked, pre-validated strategy specification:
matrix-profile pattern discovery on H1 closes, traded only in ranging regimes
(Kaufman Efficiency Ratio gate), with fixed 1:1.5 ATR brackets and prop-firm
risk controls.

The public surface is small on purpose; the modules are:

    config      - all locked parameters, per-instrument costs, symbol map
    features    - ATR(14), z-score, Kaufman Efficiency Ratio(50)
    data        - canonical OHLCV schema, parquet cache, synthetic generator
    discovery   - pattern-discovery interface + matrix-profile implementation
    bracket     - bracket-trade simulation and direction inference
    selection   - anti-overfit gates (occurrence / mean-R / split-half / FDR)
    backtest    - anchored walk-forward engine + portfolio risk simulation
    signals     - signals.csv generator (Python -> MT5 interface)
"""

from . import config  # noqa: F401

__version__ = "1.0.0"
