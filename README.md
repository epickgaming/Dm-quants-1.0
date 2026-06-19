# PropAlgo — Regime-Filtered Mean-Reversion (H1)

A production implementation of a **fixed, pre-validated** multi-instrument
mean-reversion strategy. It learns recurring price shapes from recent history
(unsupervised matrix profile), trades them **only in ranging regimes** (Kaufman
Efficiency Ratio gate), with fixed **1:1.5 ATR brackets**, under hard prop-firm
risk limits.

The Python package is the "brain" (data, discovery, selection, walk-forward
backtest, live signal file). The MQL5 Expert Advisor (`ea/PropAlgo.mq5`) is the
risk-aware execution layer that *reads* the signal file and trades it on MT5.

> **The strategy parameters are locked.** Do not tune them. The only values a
> user should change are the **per-instrument round-trip costs** and the
> **broker symbol map**, and only after re-measuring them against their own
> broker — the edge is cost-critical (see *Re-measuring broker costs*).

---

## Strategy at a glance

| Element | Setting |
|---|---|
| Instruments | XAUUSD, FTSE100, S&P500, Copper, DAX |
| Timeframe | **H1 only** (5m/15m proven untradeable — spread eats the edge) |
| Pattern discovery | Matrix Profile (`stumpy`), window `m=20`, ≤20 motifs |
| Regime filter | Kaufman Efficiency Ratio(50), trade only when **ER ≤ 0.45** |
| Direction | fixed from training (mean forward 8-bar return) |
| Bracket | SL = 2·ATR(14), TP = 3·ATR(14) → 1:1.5; 48-bar time stop |
| Anti-overfit gates | occ ≥ 25, mean R > 0, split-half robust, BH FDR q=0.10 |
| Walk-forward | train 18 months, retrain monthly, anchored |
| Sizing / risk | 0.5% equity/trade; −2%/day cutoff; −10% hard halt; ~3% concurrent |
| Weekly cap | 6 trades/week across basket (lowest-ER wins ties) |
| Account | $5,000 |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core deps: `numpy pandas scipy pyarrow stumpy`. Optional: `vectorbt` (independent
big-data cross-check), `dukascopy-python` (free real H1 data), `pytest` (tests).

The whole pipeline runs **offline** on a deterministic synthetic dataset, so you
can verify everything with no network and no broker.

---

## Quick start (offline, reproducible)

```bash
# Anchored walk-forward over a synthetic basket (planted-signal dataset).
python -m propalgo.cli backtest --synthetic --months 36

# Generate a signals.csv from the most recent bars.
python -m propalgo.cli signals --synthetic --scan-bars 50
```

Outputs land in `signals_out/`: `equity_curve.csv`, `trades.csv`,
`backtest_report.md`, and `signals.csv`.

### Acceptance metrics reproduced (synthetic, `--months 36`, last-6-months OOS)

| metric | target | reproduced |
|---|---|---|
| trades / week | ~6 | **5.85** |
| win rate | ~55–56% | **55.9%** |
| realized RRR | ~1.2–1.5 | **1.30** |
| mean R | +0.2 … +0.3 | **+0.30** |
| t-stat (one-sided) | > 2 | **2.98** |
| total return | positive | **+24.9%** |
| max drawdown | < 8% | **3.0%** |
| worst day | > −3% | **−1.6%** |

All profits are **net of costs**. The backtest also prints a **cost-sensitivity**
table at the configured cost and at +1bp / +2bp of extra slippage, and loudly
warns if the edge does not survive +2bp.

> The synthetic generator plants recurring motifs with a controlled forward edge
> purely so the *pipeline* is verifiable offline and in CI. Real-edge numbers
> require real data (below); the synthetic numbers are a pipeline check, not a
> claim about markets.

---

## Big-data backtest on real H1 history

```bash
# 1. Fetch & cache H1 OHLCV per instrument (dukascopy, free, ~2018+).
python -m propalgo.cli fetch

# 2. Anchored walk-forward over ALL cached history.
python -m propalgo.cli backtest --no-fetch
```

* Data is cached to `data/<INSTRUMENT>_H1.parquet`; re-runs fetch only the
  missing tail and normalise to the canonical schema
  (`timestamp` UTC tz-naive, `open/high/low/close/volume`, sorted, de-duped).
* `vectorbt` is used as an **independent** verifier of portfolio stats (Sharpe,
  max drawdown, total return) computed from the realised equity curve — the
  "fast on huge data" cross-check. The path-dependent bracket simulation itself
  is done by the custom engine in `propalgo/backtest.py`.
* A per-instrument **decay monitor** auto-disables any instrument whose trailing
  win rate falls below ~45% until it recovers.

---

## Live signals → MT5

```bash
# Train on trailing 18 months, scan recent bars, write signals.csv.
python -m propalgo.cli signals --no-fetch --out signals_out/signals.csv
# add --live to authorise the EA to place REAL orders from these signals
```

`signals.csv` schema (absolute prices):

```
symbol,timeframe,direction,entry,stop,target,atr,risk_pct,er,asof
```

`direction` is BUY/SELL, `risk_pct=0.005`, `asof` is the entry bar's ISO
timestamp. The EA de-duplicates by `symbol+asof`.

---

## Deploy the Expert Advisor

1. Copy `ea/PropAlgo.mq5` into `MQL5/Experts/` and compile in MetaEditor.
2. Place `signals.csv` where the EA reads it:
   * default: `MQL5/Files/signals.csv` (terminal data folder), or
   * set `InpUseCommonFolder=true` to read the terminal **common** folder.
3. Set EA inputs:
   * **broker symbol map** (`InpMapXAUUSD`, `InpMapFTSE100=UK100`,
     `InpMapSP500=US500`, `InpMapCOPPER`, `InpMapDAX=GER40`) — confirm against
     your broker's exact CFD tickers;
   * `InpRiskPct=0.005`, `InpDailyLossCutoff=0.02`, `InpMaxDrawdown=0.10`,
     `InpMaxConcurrentRisk=0.03`, `InpWeeklyTradeCap=6`;
   * **`InpLive` defaults to `false` (paper)** — it logs intended orders without
     sending. Set `InpLive=true` only when you are ready to trade.
4. **Validate in the MT5 Strategy Tester first**, using *Every tick based on
   real ticks*, before any live/demo use.

The EA enforces, on every signal: one position per symbol, the 6/week cap, the
−2%/day cutoff, the −10% hard halt, and the concurrent-risk cap. It sizes from
the broker's live `SYMBOL_TRADE_TICK_VALUE`/`TICK_SIZE` and attaches SL/TP. Every
action is logged via `Print`.

---

## Re-measuring broker costs (do this — the edge is cost-critical)

Costs are configured in `propalgo/config.py` as a **round-trip fraction of
price**:

```python
COSTS = {"XAUUSD":0.0003, "FTSE100":0.0003, "SP500":0.0002,
         "COPPER":0.0007, "DAX":0.0003}
```

To re-measure for your broker, for each instrument:

1. Read the **typical spread** in price units at the H1 you trade (e.g. from the
   Market Watch / a tick log), and add any **commission** converted to price.
2. round-trip cost ≈ `(spread + 2·commission_in_price) / price`.
3. Put that fraction in `COSTS`. Re-run `backtest` and read the **cost
   sensitivity** table: if your real cost pushes the edge below t≈2, **the
   strategy is not tradeable on that instrument at that broker** — the backtest
   prints a loud warning when this happens.

On H1 the edge survives ~+1bp of extra slippage but dies near +2bp, so accurate
costs matter more than anything else here.

### Override costs & symbols without editing source

Both the per-instrument costs and the broker symbol map are configurable via an
external YAML/JSON file (locked strategy params are never read from it):

```bash
cp config.example.yaml my_broker.yaml   # edit costs + symbol_map for your broker
python -m propalgo.cli backtest --no-fetch --config my_broker.yaml
python -m propalgo.cli signals  --no-fetch --config my_broker.yaml
```

The EA mirrors the same knobs as inputs (`InpMap*`, `InpRiskPct`, limits), so the
broker mapping lives in both layers.

---

## Tests

```bash
pytest -q          # unit tests (ATR, ER, z-score, bracket, selection, FDR)
                   # + schema/normalisation + an end-to-end pipeline smoke test
```

---

## Project layout

```
propalgo/
  config.py      locked parameters, per-instrument costs, broker symbol map
  features.py    ATR(14), z-score, Kaufman Efficiency Ratio(50)
  data.py        canonical schema, parquet cache, dukascopy fetch, synthetic gen
  discovery.py   matrix-profile motif discovery (swappable) + numpy match fallback
  bracket.py     bracket-trade simulation + direction inference
  selection.py   anti-overfit gates (occ / mean-R / split-half) + BH FDR
  backtest.py    anchored walk-forward + prop-firm portfolio simulation
  signals.py     signals.csv generator (Python -> MT5 interface)
  vbt_report.py  VectorBT independent portfolio-stats cross-check
  cli.py         `fetch` / `backtest` / `signals` entry points
ea/PropAlgo.mq5  MQL5 Expert Advisor (risk-aware executor)
scripts/         run_backtest.py wrapper
tests/           unit + integration tests
```

---

## Honesty notes

* Parameters are **fixed**; selection uses significance + split-half + FDR
  gates; **test-period data is never used for selection** (anchored walk-forward).
* All reported PnL is **net of costs**. Nothing here is gross.
* Default is **paper** (`InpLive=false`); real orders require an explicit opt-in.
* Past performance — synthetic or historical — does not guarantee future results.
  Re-measure your costs and validate on the real-tick Strategy Tester before
  risking capital.
