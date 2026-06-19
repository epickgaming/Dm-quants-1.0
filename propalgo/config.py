"""Locked strategy configuration.

Every number here is part of the fixed, pre-validated specification. Do NOT
tune these to fit a backtest -- the anti-overfit story depends on them being
frozen. The only values a *user* is expected to change are the per-instrument
round-trip costs and the broker symbol map, and only after re-measuring them
against their own broker (the edge is cost-critical).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Instruments & timeframe (LOCKED)
# --------------------------------------------------------------------------- #
# Canonical internal instrument keys. The broker may name them differently;
# see SYMBOL_MAP for the CFD ticker translation used by the EA.
INSTRUMENTS: List[str] = ["XAUUSD", "FTSE100", "SP500", "COPPER", "DAX"]

TIMEFRAME: str = "H1"  # 1-hour only. 5m/15m proven untradeable -- do not change.
BARS_PER_DAY: int = 24
BARS_PER_WEEK: int = 24 * 5  # H1 bars in a trading week (approx, used for rates)

# --------------------------------------------------------------------------- #
# Feature parameters
# --------------------------------------------------------------------------- #
ATR_PERIOD: int = 14
ER_PERIOD: int = 50
ER_GATE: float = 0.45  # signal valid only when entry-bar ER <= this (ranging)

# --------------------------------------------------------------------------- #
# Pattern discovery
# --------------------------------------------------------------------------- #
WINDOW: int = 20  # motif length m
MAX_MOTIFS: int = 20
MIN_NEIGHBORS: int = 1
MAX_DISTANCE: float = 3.0  # z-norm Euclidean distance for motif grouping
MOTIF_CUTOFF: float = 3.0
MAX_MATCHES: int = 5000
MATCH_TOLERANCE: float = 3.0  # z-norm distance for live matching

# --------------------------------------------------------------------------- #
# Trade simulation
# --------------------------------------------------------------------------- #
DIRECTION_HORIZON: int = 8  # bars used to infer fixed direction (training only)
SL_ATR_MULT: float = 2.0
TP_ATR_MULT: float = 3.0  # => 1:1.5 reward:risk
TIME_STOP_BARS: int = 48  # max bars in trade before exit-at-close

# --------------------------------------------------------------------------- #
# Pattern selection (anti-overfit gates)
# --------------------------------------------------------------------------- #
MIN_OCCURRENCES: int = 25
SPLIT_HALF_MIN: int = 8  # each half needs >= this many trades
FDR_Q: float = 0.10  # Benjamini-Hochberg false-discovery rate

# --------------------------------------------------------------------------- #
# Walk-forward / retrain
# --------------------------------------------------------------------------- #
TRAIN_MONTHS: int = 18
RETRAIN_EVERY_DAYS: int = 30  # retrain monthly

# --------------------------------------------------------------------------- #
# Portfolio / prop-firm risk (HARD limits)
# --------------------------------------------------------------------------- #
ACCOUNT_START: float = 5_000.0
RISK_PCT: float = 0.005  # 0.5% of equity per trade
WEEKLY_TRADE_CAP: int = 6  # across the whole basket; lowest-ER wins ties
DAILY_LOSS_CUTOFF: float = 0.02  # stop opening new trades after -2% on the day
MAX_TOTAL_DRAWDOWN: float = 0.10  # hard halt at -10% (i.e. $4,500 floor)
MAX_CONCURRENT_RISK: float = 0.03  # cap total open risk to ~3% of equity

# Decay monitor: disable an instrument whose trailing win-rate drops below this.
DECAY_WINRATE_FLOOR: float = 0.45
DECAY_WINDOW_TRADES: int = 20  # trailing trades used for the win-rate estimate

# --------------------------------------------------------------------------- #
# Costs (round-trip, as a fraction of price). ESTIMATES -- re-measure!
# --------------------------------------------------------------------------- #
COSTS: Dict[str, float] = {
    "XAUUSD": 0.0003,
    "FTSE100": 0.0003,
    "SP500": 0.0002,
    "COPPER": 0.0007,
    "DAX": 0.0003,
}

# Extra-slippage scenarios layered on top of COSTS for the robustness report.
# 1 basis point = 0.0001 of price.
SLIPPAGE_SCENARIOS_BP: List[float] = [0.0, 1.0, 2.0]

# --------------------------------------------------------------------------- #
# Broker symbol map (internal key -> broker CFD ticker). EA input mirror.
# These are common defaults; the user MUST confirm them against their broker.
# --------------------------------------------------------------------------- #
SYMBOL_MAP: Dict[str, str] = {
    "XAUUSD": "XAUUSD",
    "FTSE100": "UK100",
    "SP500": "US500",
    "COPPER": "COPPER",
    "DAX": "GER40",
}

# Contract value per 1.0 price unit per 1.0 lot (a.k.a. tick value scaling).
# Used by the Python sizing illustration; the EA reads these live from the
# broker via SymbolInfo, so these are only defaults for offline reporting.
CONTRACT_VALUE_PER_PRICE_UNIT: Dict[str, float] = {
    "XAUUSD": 100.0,   # 100 oz per lot
    "FTSE100": 1.0,
    "SP500": 1.0,
    "COPPER": 25_000.0,
    "DAX": 1.0,
}


@dataclass
class StrategyConfig:
    """Bundle of all parameters, so callers can pass one object around.

    Defaults mirror the module-level locked constants. Tests and the synthetic
    pipeline may override *non-strategy* knobs (e.g. costs) but should leave the
    strategy parameters alone.
    """

    instruments: List[str] = field(default_factory=lambda: list(INSTRUMENTS))
    timeframe: str = TIMEFRAME

    atr_period: int = ATR_PERIOD
    er_period: int = ER_PERIOD
    er_gate: float = ER_GATE

    window: int = WINDOW
    max_motifs: int = MAX_MOTIFS
    min_neighbors: int = MIN_NEIGHBORS
    max_distance: float = MAX_DISTANCE
    motif_cutoff: float = MOTIF_CUTOFF
    max_matches: int = MAX_MATCHES
    match_tolerance: float = MATCH_TOLERANCE

    direction_horizon: int = DIRECTION_HORIZON
    sl_atr_mult: float = SL_ATR_MULT
    tp_atr_mult: float = TP_ATR_MULT
    time_stop_bars: int = TIME_STOP_BARS

    min_occurrences: int = MIN_OCCURRENCES
    split_half_min: int = SPLIT_HALF_MIN
    fdr_q: float = FDR_Q

    train_months: int = TRAIN_MONTHS
    retrain_every_days: int = RETRAIN_EVERY_DAYS

    account_start: float = ACCOUNT_START
    risk_pct: float = RISK_PCT
    weekly_trade_cap: int = WEEKLY_TRADE_CAP
    daily_loss_cutoff: float = DAILY_LOSS_CUTOFF
    max_total_drawdown: float = MAX_TOTAL_DRAWDOWN
    max_concurrent_risk: float = MAX_CONCURRENT_RISK

    decay_winrate_floor: float = DECAY_WINRATE_FLOOR
    decay_window_trades: int = DECAY_WINDOW_TRADES

    costs: Dict[str, float] = field(default_factory=lambda: dict(COSTS))
    slippage_scenarios_bp: List[float] = field(
        default_factory=lambda: list(SLIPPAGE_SCENARIOS_BP)
    )
    symbol_map: Dict[str, str] = field(default_factory=lambda: dict(SYMBOL_MAP))
    contract_value: Dict[str, float] = field(
        default_factory=lambda: dict(CONTRACT_VALUE_PER_PRICE_UNIT)
    )


def load_overrides(cfg: "StrategyConfig", path: str) -> "StrategyConfig":
    """Apply user overrides from a YAML/JSON file onto a StrategyConfig.

    Only the USER-tunable knobs are honoured -- the broker symbol map and the
    per-instrument round-trip costs -- so the locked strategy parameters can
    never be changed from a config file. Example file::

        costs:
          XAUUSD: 0.00035
          COPPER: 0.0008
        symbol_map:
          FTSE100: FTSE100
          DAX: DE40

    Unknown keys are ignored with a warning.
    """
    import json
    import os

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path) as f:
        text = f.read()
    data = None
    try:
        import yaml  # optional dependency
        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)  # fall back to JSON

    data = data or {}
    allowed = {"costs", "symbol_map"}
    for key in data:
        if key not in allowed:
            print(f"[config] ignoring non-tunable key: {key!r}")

    if isinstance(data.get("costs"), dict):
        for inst, val in data["costs"].items():
            if inst in cfg.instruments:
                cfg.costs[inst] = float(val)
            else:
                print(f"[config] ignoring cost for unknown instrument: {inst!r}")
    if isinstance(data.get("symbol_map"), dict):
        for inst, sym in data["symbol_map"].items():
            if inst in cfg.instruments:
                cfg.symbol_map[inst] = str(sym)
            else:
                print(f"[config] ignoring symbol map for unknown instrument: {inst!r}")
    return cfg
