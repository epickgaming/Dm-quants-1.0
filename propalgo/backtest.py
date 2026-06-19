"""Anchored walk-forward backtest + prop-firm portfolio simulation.

Pipeline per instrument:
  * compute ATR(14) and ER(50) on the full series (features only ever look back);
  * anchored walk-forward: at each monthly retrain date, train on the trailing
    18 months -> discover motifs -> simulate their in-training bracket trades ->
    select with the anti-overfit gates -> fix each survivor's direction;
  * scan the forward (test) month for matches of the selected templates and emit
    candidate signals (entry time, ER, ATR, direction, bracket outcome).

Then a single chronological portfolio simulation across the whole basket applies
the regime gate, the weekly cap (lowest-ER wins), one-position-per-instrument,
the concurrent-risk cap, the -2%/day cutoff, the -10% hard halt, 0.5% sizing,
and the per-instrument decay monitor -- producing the equity curve and metrics.

Selection NEVER sees test-period data, so the walk-forward is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .bracket import infer_direction, simulate_bracket
from .config import StrategyConfig
from .discovery import MatrixProfileDiscovery, match_template
from .features import atr as atr_feature
from .features import efficiency_ratio
from .selection import TemplateStats, evaluate_templates


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class CandidateSignal:
    instrument: str
    entry_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    sl_distance: float
    tp_distance: float
    atr: float
    er: float
    r_multiple: float
    win: bool
    exit_reason: str
    template_id: str


@dataclass
class InstrumentData:
    instrument: str
    df: pd.DataFrame
    close: np.ndarray = field(init=False)
    high: np.ndarray = field(init=False)
    low: np.ndarray = field(init=False)
    times: np.ndarray = field(init=False)
    atr: np.ndarray = field(init=False)
    er: np.ndarray = field(init=False)

    def __post_init__(self, atr_period: int = 14, er_period: int = 50):
        d = self.df
        self.close = d["close"].values.astype(float)
        self.high = d["high"].values.astype(float)
        self.low = d["low"].values.astype(float)
        self.times = d["timestamp"].values
        self.atr = atr_feature(self.high, self.low, self.close, atr_period)
        self.er = efficiency_ratio(self.close, er_period)


def _build_instrument(instrument, df, cfg) -> InstrumentData:
    idata = InstrumentData.__new__(InstrumentData)
    idata.instrument = instrument
    idata.df = df
    d = df
    idata.close = d["close"].values.astype(float)
    idata.high = d["high"].values.astype(float)
    idata.low = d["low"].values.astype(float)
    idata.times = pd.to_datetime(d["timestamp"]).values
    idata.atr = atr_feature(idata.high, idata.low, idata.close, cfg.atr_period)
    idata.er = efficiency_ratio(idata.close, cfg.er_period)
    return idata


# --------------------------------------------------------------------------- #
# Per-instrument walk-forward candidate generation
# --------------------------------------------------------------------------- #
def _train_and_select(idata: InstrumentData, cfg: StrategyConfig,
                      train_lo: int, train_hi: int, discovery):
    """Discover + select templates on close[train_lo:train_hi].

    Returns a list of (Template, direction) for survivors.
    """
    train_close = idata.close[train_lo:train_hi]
    train_high = idata.high[train_lo:train_hi]
    train_low = idata.low[train_lo:train_hi]
    cost = cfg.costs.get(idata.instrument, 0.0)

    templates = discovery.discover(idata.instrument, train_close)
    if not templates:
        return []

    stats_list: List[TemplateStats] = []
    template_dir: Dict[str, int] = {}
    template_obj: Dict[str, object] = {}

    for tpl in templates:
        # Re-match the template over the training series for a uniform, robust
        # occurrence set (consistent with how we match live in the test window).
        occ = match_template(tpl.pattern, train_close, max_distance=tpl.match_tolerance)
        # entry = window-end of each occurrence
        entries = [int(o + tpl.window - 1) for o in occ
                   if o + tpl.window - 1 < len(train_close)]
        if len(entries) < 2:
            continue

        direction = infer_direction(train_close, entries, cfg.direction_horizon)
        rs, ets = [], []
        for e in entries:
            res = simulate_bracket(
                train_high, train_low, train_close, e, direction,
                idata.atr[train_lo + e], cost,
                cfg.sl_atr_mult, cfg.tp_atr_mult, cfg.time_stop_bars,
            )
            if res is not None:
                rs.append(res.r_multiple)
                ets.append(e)
        if len(rs) < 2:
            continue

        ts = TemplateStats(
            template_id=tpl.template_id,
            instrument=idata.instrument,
            direction=direction,
            r_multiples=np.array(rs),
            entry_times=np.array(ets),
        )
        stats_list.append(ts)
        template_dir[tpl.template_id] = direction
        template_obj[tpl.template_id] = tpl

    evaluate_templates(stats_list, cfg.min_occurrences, cfg.split_half_min, cfg.fdr_q)

    survivors = []
    for ts in stats_list:
        if ts.selected:
            tpl = template_obj[ts.template_id]
            survivors.append((tpl, ts.direction))
    return survivors


def walk_forward_instrument(idata: InstrumentData, cfg: StrategyConfig,
                            discovery=None) -> List[CandidateSignal]:
    """Anchored walk-forward over one instrument -> candidate test signals."""
    discovery = discovery or MatrixProfileDiscovery(
        window=cfg.window, max_motifs=cfg.max_motifs,
        min_neighbors=cfg.min_neighbors, max_distance=cfg.max_distance,
        cutoff=cfg.motif_cutoff, max_matches=cfg.max_matches,
        match_tolerance=cfg.match_tolerance,
    )

    times = pd.to_datetime(idata.times)
    n = len(idata.close)
    cost = cfg.costs.get(idata.instrument, 0.0)

    train_delta = pd.DateOffset(months=cfg.train_months)
    step_delta = pd.Timedelta(days=cfg.retrain_every_days)

    first_time = times[0]
    last_time = times[-1]
    # First retrain date is once we have a full training window.
    retrain_date = first_time + train_delta
    if retrain_date >= last_time:
        return []  # not enough history for even one walk-forward step

    candidates: List[CandidateSignal] = []
    seen_entries = set()

    while retrain_date < last_time:
        test_end_date = retrain_date + step_delta
        train_lo_date = retrain_date - train_delta

        train_lo = int(np.searchsorted(times.values, np.datetime64(train_lo_date)))
        train_hi = int(np.searchsorted(times.values, np.datetime64(retrain_date)))
        test_lo = train_hi
        test_hi = int(np.searchsorted(times.values, np.datetime64(test_end_date)))
        test_hi = min(test_hi, n)

        if train_hi - train_lo < cfg.window * 5 or test_hi <= test_lo:
            retrain_date = retrain_date + step_delta
            continue

        survivors = _train_and_select(idata, cfg, train_lo, train_hi, discovery)

        for tpl, direction in survivors:
            # Match on an extended slice so windows ending in the test month are
            # caught, then keep only entries whose ENTRY bar is in the test month.
            ext_lo = max(0, test_lo - (cfg.window - 1))
            seg = idata.close[ext_lo:test_hi]
            occ = match_template(tpl.pattern, seg, max_distance=tpl.match_tolerance)
            for o in occ:
                e = ext_lo + int(o) + tpl.window - 1  # global entry index
                if e < test_lo or e >= test_hi:
                    continue
                key = (idata.instrument, e)  # dedup: first template to fire wins
                if key in seen_entries:
                    continue
                er_val = idata.er[e]
                if not np.isfinite(er_val) or er_val > cfg.er_gate:
                    continue
                res = simulate_bracket(
                    idata.high, idata.low, idata.close, e, direction,
                    idata.atr[e], cost, cfg.sl_atr_mult, cfg.tp_atr_mult,
                    cfg.time_stop_bars,
                )
                if res is None:
                    continue
                exit_idx = min(e + res.bars_held, n - 1)
                seen_entries.add(key)
                candidates.append(CandidateSignal(
                    instrument=idata.instrument,
                    entry_idx=e,
                    entry_time=pd.Timestamp(times.values[e]),
                    exit_time=pd.Timestamp(times.values[exit_idx]),
                    direction=direction,
                    entry_price=res.entry_price,
                    sl_distance=res.sl_distance,
                    tp_distance=res.tp_distance,
                    atr=idata.atr[e],
                    er=float(er_val),
                    r_multiple=res.r_multiple,
                    win=res.win,
                    exit_reason=res.exit_reason,
                    template_id=tpl.template_id,
                ))

        retrain_date = retrain_date + step_delta

    candidates.sort(key=lambda c: c.entry_time)
    return collapse_overlaps(candidates, min_gap=cfg.window)


def collapse_overlaps(candidates: List[CandidateSignal], min_gap: int) -> List[CandidateSignal]:
    """Collapse overlapping signals on the same instrument into one.

    Several survivor templates can fire on the same bar cluster (the same
    underlying shape) at slightly different offsets. We would never open several
    trades on one cluster, so we keep a single representative per cluster: the
    lowest-ER one (highest conviction, matching the weekly-cap tie-break), then
    enforce a `min_gap`-bar exclusion zone around it.
    """
    if not candidates:
        return candidates
    by_inst: Dict[str, List[CandidateSignal]] = {}
    for c in candidates:
        by_inst.setdefault(c.instrument, []).append(c)

    kept: List[CandidateSignal] = []
    for inst, group in by_inst.items():
        group_sorted = sorted(group, key=lambda c: c.er)  # lowest ER first
        chosen: List[CandidateSignal] = []
        chosen_idx: List[int] = []
        for c in group_sorted:
            if all(abs(c.entry_idx - ci) >= min_gap for ci in chosen_idx):
                chosen.append(c)
                chosen_idx.append(c.entry_idx)
        kept.extend(chosen)
    kept.sort(key=lambda c: c.entry_time)
    return kept


# --------------------------------------------------------------------------- #
# Weekly cap + portfolio simulation
# --------------------------------------------------------------------------- #
def apply_weekly_cap(candidates: List[CandidateSignal], cap: int) -> List[CandidateSignal]:
    """Across the basket, keep at most `cap` signals per ISO week, choosing the
    lowest-ER (most ranging = highest conviction) when more qualify."""
    by_week: Dict[tuple, List[CandidateSignal]] = {}
    for c in candidates:
        iso = c.entry_time.isocalendar()
        key = (iso[0], iso[1])
        by_week.setdefault(key, []).append(c)
    kept: List[CandidateSignal] = []
    for key, group in by_week.items():
        group_sorted = sorted(group, key=lambda c: c.er)
        kept.extend(group_sorted[:cap])
    kept.sort(key=lambda c: c.entry_time)
    return kept


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]


def simulate_portfolio(candidates: List[CandidateSignal], cfg: StrategyConfig,
                       extra_slippage_bp: float = 0.0) -> PortfolioResult:
    """Chronological portfolio sim enforcing all prop-firm risk rules.

    `extra_slippage_bp` adds basis points of round-trip cost on top of the
    per-instrument config cost (for the robustness report). It is converted to R
    using each trade's entry price and stop distance.
    """
    candidates = sorted(candidates, key=lambda c: c.entry_time)

    equity = cfg.account_start
    peak = equity
    halted = False

    open_positions: List[dict] = []        # {instrument, exit_time, risk$, r}
    day_pnl: Dict[pd.Timestamp, float] = {}
    day_start_equity: Dict[pd.Timestamp, float] = {}
    disabled: Dict[str, bool] = {i: False for i in cfg.instruments}
    recent_wins: Dict[str, List[int]] = {i: [] for i in cfg.instruments}

    trade_rows = []
    equity_points = []

    def settle_until(t):
        nonlocal equity, peak, halted
        due = [p for p in open_positions if p["exit_time"] <= t]
        due.sort(key=lambda p: p["exit_time"])
        for p in due:
            pnl = p["r"] * p["risk_dollars"]
            equity += pnl
            d = pd.Timestamp(p["exit_time"]).normalize()
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            peak = max(peak, equity)
            equity_points.append((p["exit_time"], equity))
            # decay monitor
            inst = p["instrument"]
            recent_wins[inst].append(1 if p["r"] > 0 else 0)
            rw = recent_wins[inst][-cfg.decay_window_trades:]
            if len(rw) >= cfg.decay_window_trades:
                if np.mean(rw) < cfg.decay_winrate_floor:
                    disabled[inst] = True
                elif np.mean(rw) >= cfg.decay_winrate_floor + 0.05:
                    disabled[inst] = False
            open_positions.remove(p)
            if (peak - equity) / peak >= cfg.max_total_drawdown:
                halted = True

    for c in candidates:
        if halted:
            break
        settle_until(c.entry_time)
        if halted:
            break

        day = pd.Timestamp(c.entry_time).normalize()
        if day not in day_start_equity:
            day_start_equity[day] = equity

        # -- risk gates -------------------------------------------------- #
        if disabled.get(c.instrument, False):
            continue
        # one position per instrument
        if any(p["instrument"] == c.instrument for p in open_positions):
            continue
        # daily -2% cutoff (based on realized pnl so far today)
        if day_pnl.get(day, 0.0) <= -cfg.daily_loss_cutoff * day_start_equity[day]:
            continue

        risk_dollars = cfg.risk_pct * equity
        # concurrent open-risk cap (~3% of equity)
        open_risk = sum(p["risk_dollars"] for p in open_positions)
        if open_risk + risk_dollars > cfg.max_concurrent_risk * equity + 1e-9:
            continue

        # -- apply extra slippage to R ----------------------------------- #
        r = c.r_multiple
        if extra_slippage_bp > 0 and c.sl_distance > 0:
            extra_frac = extra_slippage_bp * 1e-4
            r = r - (extra_frac * c.entry_price) / c.sl_distance

        open_positions.append({
            "instrument": c.instrument,
            "exit_time": c.exit_time,
            "risk_dollars": risk_dollars,
            "r": r,
        })
        trade_rows.append({
            "instrument": c.instrument,
            "entry_time": c.entry_time,
            "exit_time": c.exit_time,
            "direction": "BUY" if c.direction > 0 else "SELL",
            "entry_price": c.entry_price,
            "sl_distance": c.sl_distance,
            "atr": c.atr,
            "er": c.er,
            "r_multiple": r,
            "win": r > 0,
            "exit_reason": c.exit_reason,
            "risk_dollars": risk_dollars,
            "template_id": c.template_id,
        })

    # settle any still-open positions at the end
    if open_positions:
        settle_until(pd.Timestamp.max)

    trades = pd.DataFrame(trade_rows)
    equity_points.sort(key=lambda x: x[0])
    equity_curve = pd.DataFrame(equity_points, columns=["time", "equity"])

    metrics = _compute_metrics(trades, equity_curve, cfg, candidates)
    return PortfolioResult(trades=trades, equity_curve=equity_curve, metrics=metrics)


def _compute_metrics(trades, equity_curve, cfg, candidates) -> Dict[str, float]:
    m: Dict[str, float] = {}
    n = len(trades)
    m["trades"] = n
    if n == 0:
        m.update({k: float("nan") for k in
                  ["win_rate", "mean_r", "t_stat", "realized_rrr",
                   "trades_per_week", "total_return", "max_drawdown", "worst_day"]})
        return m

    r = trades["r_multiple"].values.astype(float)
    wins = r > 0
    m["win_rate"] = float(wins.mean())
    m["mean_r"] = float(r.mean())
    sd = r.std(ddof=1) if n > 1 else 0.0
    m["t_stat"] = float(r.mean() / (sd / np.sqrt(n))) if sd > 1e-12 else float("nan")

    win_r = r[wins]
    loss_r = -r[~wins]
    if len(win_r) and len(loss_r) and loss_r.mean() > 1e-12:
        m["realized_rrr"] = float(win_r.mean() / loss_r.mean())
    else:
        m["realized_rrr"] = float("nan")

    span_days = (trades["entry_time"].max() - trades["entry_time"].min()).days
    weeks = max(span_days / 7.0, 1.0)
    m["trades_per_week"] = float(n / weeks)

    # equity-based metrics
    if len(equity_curve) > 0:
        eq = equity_curve["equity"].values.astype(float)
        eq = np.concatenate([[cfg.account_start], eq])
        m["total_return"] = float(eq[-1] / cfg.account_start - 1.0)
        running_peak = np.maximum.accumulate(eq)
        dd = (eq - running_peak) / running_peak
        m["max_drawdown"] = float(-dd.min())
        m["final_equity"] = float(eq[-1])
        # worst day
        ec = equity_curve.copy()
        ec["day"] = pd.to_datetime(ec["time"]).dt.normalize()
        daily = ec.groupby("day")["equity"].last()
        daily = pd.concat([pd.Series([cfg.account_start]), daily])
        day_ret = daily.pct_change().dropna()
        m["worst_day"] = float(day_ret.min()) if len(day_ret) else float("nan")
    else:
        m.update({"total_return": float("nan"), "max_drawdown": float("nan"),
                  "worst_day": float("nan"), "final_equity": cfg.account_start})
    return m


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def run_walk_forward(data: Dict[str, pd.DataFrame], cfg: Optional[StrategyConfig] = None,
                     discovery=None, verbose: bool = True) -> Dict[str, object]:
    """Run the full anchored walk-forward over a basket and return results.

    Returns a dict with raw candidates, the post-cap candidates, the baseline
    portfolio result, and a cost-sensitivity table across slippage scenarios.
    """
    cfg = cfg or StrategyConfig()
    all_candidates: List[CandidateSignal] = []
    per_instrument: Dict[str, List[CandidateSignal]] = {}

    for inst, df in data.items():
        if verbose:
            print(f"[wf] {inst}: {len(df)} bars")
        idata = _build_instrument(inst, df, cfg)
        cands = walk_forward_instrument(idata, cfg, discovery)
        per_instrument[inst] = cands
        all_candidates.extend(cands)
        if verbose:
            print(f"[wf] {inst}: {len(cands)} candidate signals")

    capped = apply_weekly_cap(all_candidates, cfg.weekly_trade_cap)
    baseline = simulate_portfolio(capped, cfg, extra_slippage_bp=0.0)

    # cost sensitivity
    sensitivity = {}
    for bp in cfg.slippage_scenarios_bp:
        res = simulate_portfolio(capped, cfg, extra_slippage_bp=bp)
        sensitivity[bp] = res.metrics

    return {
        "config": cfg,
        "candidates": all_candidates,
        "per_instrument": per_instrument,
        "capped": capped,
        "baseline": baseline,
        "sensitivity": sensitivity,
    }
