"""Tests for canonical schema normalization and the synthetic generator."""

import numpy as np
import pandas as pd

from propalgo.data import (
    CANONICAL_COLUMNS,
    SyntheticSpec,
    generate_synthetic,
    merge_incremental,
    normalize,
)
from propalgo.features import efficiency_ratio


def test_normalize_handles_tz_and_aliases_and_dupes():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 00:00",
                                 "2020-01-01 01:00"], utc=True),
        "O": [1.0, 2.0, 1.1], "H": [2, 3, 2], "L": [0.5, 1, 0.6],
        "C": [1.5, 2.5, 1.6], "Vol": [10, 20, 30],
    })
    out = normalize(df)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].is_unique               # duplicate dropped
    assert out["timestamp"].dt.tz is None           # tz-naive
    assert len(out) == 2


def test_merge_incremental_appends_only_new():
    base = normalize(pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
        "open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [1, 1],
    }))
    new = normalize(pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00"]),
        "open": [2, 2], "high": [2, 2], "low": [2, 2], "close": [2, 2], "volume": [2, 2],
    }))
    merged = merge_incremental(base, new)
    assert len(merged) == 3                          # one overlap collapsed
    assert merged["timestamp"].is_unique


def test_synthetic_schema_and_regime():
    spec = SyntheticSpec(months=8, seed=3)
    df = generate_synthetic("XAUUSD", spec)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert df["timestamp"].is_monotonic_increasing
    # synthetic is mostly ranging -> ER predominantly below the gate
    er = efficiency_ratio(df["close"].values, 50)
    er = er[np.isfinite(er)]
    assert np.mean(er <= 0.45) > 0.8
