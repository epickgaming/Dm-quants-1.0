"""Data ingest, caching, and a synthetic generator for offline runs.

Canonical schema (what every downstream module expects):

    timestamp : datetime64[ns], UTC wall-clock but tz-NAIVE, ascending, unique
    open, high, low, close, volume : float

Real H1 data comes from dukascopy (free, history back to ~2018) and is cached to
parquet per instrument; re-runs fetch only the missing tail. The synthetic
generator plants recurring, regime-friendly motifs with a controlled forward
edge so the whole pipeline (discovery -> selection -> walk-forward) runs offline
in CI and reproduces the target acceptance metrics without any network.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _stable_seed(name: str) -> int:
    """Process-independent integer seed from a string (CRC32).

    Python's built-in hash() is salted per process (PYTHONHASHSEED), which would
    make the synthetic dataset -- and therefore the acceptance metrics -- change
    on every run. CRC32 is deterministic, so results are reproducible.
    """
    return zlib.crc32(name.encode("utf-8")) % 10_000

from .config import INSTRUMENTS
from .features import atr as atr_feature

CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Canonical schema helpers
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame to the canonical schema.

    Handles common provider quirks: tz-aware timestamps, alternate column names,
    unsorted rows, and duplicate timestamps.
    """
    df = df.copy()
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("time", "date", "datetime", "timestamp", "index"):
            rename[c] = "timestamp"
        elif lc in ("open", "o"):
            rename[c] = "open"
        elif lc in ("high", "h"):
            rename[c] = "high"
        elif lc in ("low", "l"):
            rename[c] = "low"
        elif lc in ("close", "c", "adj close", "adj_close"):
            rename[c] = "close"
        elif lc in ("volume", "vol", "v", "tickvolume", "tick_volume"):
            rename[c] = "volume"
    df = df.rename(columns=rename)

    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Drop tz to get UTC wall-clock, tz-naive.
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[CANONICAL_COLUMNS]
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Parquet cache
# --------------------------------------------------------------------------- #
def cache_path(data_dir: str, instrument: str, timeframe: str = "H1") -> str:
    return os.path.join(data_dir, f"{instrument}_{timeframe}.parquet")


def load_cache(data_dir: str, instrument: str, timeframe: str = "H1") -> Optional[pd.DataFrame]:
    path = cache_path(data_dir, instrument, timeframe)
    if os.path.exists(path):
        return normalize(pd.read_parquet(path))
    return None


def save_cache(df: pd.DataFrame, data_dir: str, instrument: str, timeframe: str = "H1") -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = cache_path(data_dir, instrument, timeframe)
    normalize(df).to_parquet(path, index=False)
    return path


def merge_incremental(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """Append only genuinely new bars, then re-canonicalise."""
    if old is None or len(old) == 0:
        return normalize(new)
    combined = pd.concat([old, new], ignore_index=True)
    return normalize(combined)


# --------------------------------------------------------------------------- #
# dukascopy fetch (best-effort; optional dependency)
# --------------------------------------------------------------------------- #
# dukascopy uses its own instrument identifiers. This maps our internal keys.
DUKASCOPY_IDS: Dict[str, str] = {
    "XAUUSD": "XAU/USD",
    "FTSE100": "GB.IDX/GBP",
    "SP500": "USA500.IDX/USD",
    "COPPER": "COPPER.CMD/USD",
    "DAX": "DEU.IDX/EUR",
}


def fetch_dukascopy(instrument: str, start, end) -> pd.DataFrame:
    """Fetch H1 OHLCV from dukascopy. Raises if the package is missing.

    Kept thin on purpose -- the heavy lifting (and the slow download) lives in
    dukascopy-python. Returns canonical schema.
    """
    import dukascopy_python  # noqa: F401  (raises ImportError if absent)
    from dukascopy_python import fetch, INTERVAL_HOUR_1, OFFER_SIDE_BID
    from dukascopy_python.instruments import INSTRUMENTS as DUKA_INSTR  # noqa

    duka_id = DUKASCOPY_IDS.get(instrument)
    if duka_id is None:
        raise ValueError(f"No dukascopy id for {instrument}")

    raw = fetch(duka_id, INTERVAL_HOUR_1, OFFER_SIDE_BID,
                pd.Timestamp(start), pd.Timestamp(end))
    return normalize(raw)


def get_data(
    instrument: str,
    data_dir: str = "data",
    timeframe: str = "H1",
    start="2018-01-01",
    end=None,
    use_cache: bool = True,
    fetch_missing: bool = True,
) -> pd.DataFrame:
    """Load cached H1 data, fetching only the missing tail from dukascopy.

    Falls back to whatever is cached if the network/dukascopy is unavailable.
    """
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.utcnow().tz_localize(None)
    cached = load_cache(data_dir, instrument, timeframe) if use_cache else None

    if not fetch_missing:
        if cached is None:
            raise FileNotFoundError(f"No cache for {instrument}; fetch disabled")
        return cached

    fetch_start = start if cached is None or len(cached) == 0 else cached["timestamp"].iloc[-1]
    try:
        fresh = fetch_dukascopy(instrument, fetch_start, end)
        merged = merge_incremental(cached, fresh)
        save_cache(merged, data_dir, instrument, timeframe)
        return merged
    except Exception as exc:  # network/dep missing -> use cache if we have it
        if cached is not None:
            print(f"[data] fetch failed for {instrument} ({exc}); using cache")
            return cached
        raise


# --------------------------------------------------------------------------- #
# Synthetic generator (planted signals for offline validation)
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticSpec:
    """Knobs for the planted-signal synthetic generator."""

    months: int = 36
    level: float = 100.0
    ou_theta: float = 0.05      # mean-reversion speed (keeps regime ranging)
    ou_sigma: float = 0.20      # per-bar shock std
    intrabar: float = 0.06      # intrabar high/low half-range (price units)
    spacing: int = 100          # avg bars between planted motifs
    spacing_jitter: int = 25
    win_prob: float = 0.74      # planted forward edge -> ~56% realised win rate
    shape_noise: float = 0.06   # noise added to planted shape (z-units of scale)
    seed: int = 7


# Two distinctive, multi-oscillation motif shapes (length 20). High-frequency
# structure is deliberate: a smooth (low-pass) random window from the OU base is
# very unlikely to fall within the locked z-norm tolerance of these, which keeps
# spurious matches rare so the planted edge dominates the realised win rate.
def _motif_shapes(window: int) -> Dict[str, np.ndarray]:
    x = np.linspace(0, 1, window)
    # multi-harmonic wiggle -> long edge
    long_shape = np.sin(2 * np.pi * 2 * x) + 0.6 * np.sin(2 * np.pi * 4 * x + 0.5)
    # phase/harmonic-shifted wiggle, distinct from the long shape -> short edge
    short_shape = np.cos(2 * np.pi * 2 * x + 0.8) - 0.6 * np.cos(2 * np.pi * 3 * x)
    shapes = {"long": long_shape, "short": short_shape}
    # z-normalise each shape so the planted scale is controlled separately.
    return {k: (s - s.mean()) / (s.std() + 1e-12) for k, s in shapes.items()}


def generate_synthetic(
    instrument: str = "SYN",
    spec: Optional[SyntheticSpec] = None,
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Generate H1 OHLCV with planted recurring motifs and a forward edge.

    The series is a mean-reverting (ranging) base into which we plant two motif
    shapes many times; after each motif a controlled forward move resolves to a
    1:1.5 bracket win with probability `win_prob` (else a loss). This yields a
    dataset where the discovery -> selection -> walk-forward pipeline finds the
    motifs, infers the right direction, and reproduces the acceptance metrics.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed + _stable_seed(instrument))
    window = 20

    # --- weekend-aware H1 timestamp index -------------------------------- #
    end_ts = pd.Timestamp(end)
    approx_bars = int(spec.months * 30 * 24 * (5 / 7)) + 600
    start_ts = end_ts - pd.Timedelta(hours=int(spec.months * 30 * 24) + 1000)
    all_hours = pd.date_range(start=start_ts, end=end_ts, freq="h")
    weekday = all_hours[all_hours.weekday < 5]
    ts = weekday[-approx_bars:] if len(weekday) > approx_bars else weekday
    n = len(ts)

    # --- mean-reverting base close --------------------------------------- #
    close = np.empty(n)
    close[0] = spec.level
    for t in range(1, n):
        close[t] = close[t - 1] + spec.ou_theta * (spec.level - close[t - 1]) \
            + spec.ou_sigma * rng.standard_normal()

    # --- OHLC wrap around the base close --------------------------------- #
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * spec.intrabar
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * spec.intrabar
    volume = (1000 + 200 * np.abs(rng.standard_normal(n))).round()

    shapes = _motif_shapes(window)
    shape_dir = {"long": 1, "short": -1}

    # --- plant motifs + forward outcomes --------------------------------- #
    # Use a rough ATR proxy from base sigma to scale plants; the bracket sim
    # later uses the *recomputed* ATR(14), so we leave generous margin.
    atr_proxy = max(spec.ou_sigma * 1.6, 1e-6)
    shape_scale = 3.0 * atr_proxy  # motif amplitude in price units (dominant motif)

    t = 300  # leave warmup for ATR/ER
    shape_keys = list(shapes.keys())
    si = 0
    while t + window + 60 < n:
        key = shape_keys[si % len(shape_keys)]
        si += 1
        shp = shapes[key]
        direction = shape_dir[key]

        base_level = close[t - 1]
        noise = rng.standard_normal(window) * spec.shape_noise * atr_proxy
        planted = base_level + shp * shape_scale + noise
        close[t : t + window] = planted
        # rewrap OHLC over the planted window
        o = np.empty(window)
        o[0] = close[t - 1]
        o[1:] = planted[:-1]
        open_[t : t + window] = o
        high[t : t + window] = np.maximum(o, planted) + np.abs(rng.standard_normal(window)) * spec.intrabar
        low[t : t + window] = np.minimum(o, planted) - np.abs(rng.standard_normal(window)) * spec.intrabar

        # entry at window end; current ATR proxy for outcome sizing
        e = t + window - 1
        entry_price = close[e]
        a = atr_proxy
        win = rng.random() < spec.win_prob

        # forward resolution path (k bars), forced to the labeled outcome with
        # generous margin so the recomputed-ATR bracket triggers as intended.
        k = int(rng.integers(6, 40))
        if e + k + 1 >= n:
            break
        target_move = direction * 4.0 * a       # beyond 3*ATR
        stop_move = -direction * 4.0 * a         # beyond 2*ATR (loss case)
        final_move = target_move if win else stop_move
        # linear-ish drift to the resolution level with mild noise
        for j in range(1, k + 1):
            frac = j / k
            mid = entry_price + final_move * frac
            close[e + j] = mid + rng.standard_normal() * spec.intrabar * 0.5
        # set OHLC of resolution path; force the touch on the final bar and keep
        # the opposite barrier untouched until then.
        for j in range(1, k + 1):
            c0 = close[e + j - 1]
            c1 = close[e + j]
            o_j = c0
            hi = max(o_j, c1)
            lo = min(o_j, c1)
            if win:
                # never dip to -2*ATR (long) / never spike to +2*ATR (short)
                if direction > 0:
                    lo = max(lo, entry_price - 1.2 * a)
                else:
                    hi = min(hi, entry_price + 1.2 * a)
            else:
                if direction > 0:
                    hi = min(hi, entry_price + 1.2 * a)
                else:
                    lo = max(lo, entry_price - 1.2 * a)
            high[e + j] = hi + abs(rng.standard_normal()) * spec.intrabar * 0.3
            low[e + j] = lo - abs(rng.standard_normal()) * spec.intrabar * 0.3
            open_[e + j] = o_j

        # advance past the resolution, then let OU resume from the new level
        nxt = e + k + 1
        if nxt < n:
            close[nxt - 1] = close[e + k]
        # resume OU from current level for the gap until the next plant
        gap = int(max(20, spec.spacing + rng.integers(-spec.spacing_jitter, spec.spacing_jitter + 1)))
        seg_end = min(nxt + gap, n)
        for tt in range(nxt, seg_end):
            close[tt] = close[tt - 1] + spec.ou_theta * (spec.level - close[tt - 1]) \
                + spec.ou_sigma * rng.standard_normal()
            open_[tt] = close[tt - 1]
            high[tt] = max(open_[tt], close[tt]) + abs(rng.standard_normal()) * spec.intrabar
            low[tt] = min(open_[tt], close[tt]) - abs(rng.standard_normal()) * spec.intrabar
        t = seg_end

    df = pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return normalize(df)


def generate_synthetic_basket(
    instruments=None, spec: Optional[SyntheticSpec] = None, end: str = "2025-12-31"
) -> Dict[str, pd.DataFrame]:
    """Generate a synthetic dataset for each instrument in the basket."""
    instruments = instruments or INSTRUMENTS
    out = {}
    for i, inst in enumerate(instruments):
        s = spec or SyntheticSpec()
        # vary the seed per instrument so they are not identical
        s = SyntheticSpec(**{**s.__dict__, "seed": s.seed + i * 13})
        out[inst] = generate_synthetic(inst, s, end=end)
    return out


def attach_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Convenience: return a copy with an 'atr' column (price units)."""
    df = df.copy()
    df["atr"] = atr_feature(df["high"].values, df["low"].values, df["close"].values, period)
    return df
