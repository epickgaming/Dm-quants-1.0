"""Anti-overfit pattern selection.

Given templates whose in-training trades have been simulated, keep a template
only if ALL of these gates pass:

  1. occurrences >= 25 (enough training trades to mean anything);
  2. mean R > 0 (positive expectancy in training);
  3. split-half robustness: split trades by their median entry time; each half
     must have >= 8 trades AND mean R > 0 in BOTH halves;
  4. Benjamini-Hochberg FDR at q = 0.10 across the one-sided t-test p-values of
     all templates that survived gates 1-3.

Selection uses TRAINING data only. The test/forward period is never consulted
for selection -- that is what keeps the walk-forward honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy import stats

from .config import FDR_Q, MIN_OCCURRENCES, SPLIT_HALF_MIN


@dataclass
class TemplateStats:
    """Per-template training statistics used by the selection gates."""

    template_id: str
    instrument: str
    direction: int
    r_multiples: np.ndarray       # R per training trade
    entry_times: np.ndarray       # entry index/time per training trade (for split)
    n: int = 0
    mean_r: float = float("nan")
    p_value: float = float("nan")
    passed_pre_fdr: bool = False
    selected: bool = False
    reject_reason: str = ""


def one_sided_t_pvalue(r: Sequence[float]) -> float:
    """One-sided t-test p-value for H1: mean(R) > 0.

    Returns 1.0 (no evidence) for degenerate samples.
    """
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return 1.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        # All identical: significant iff positive, else not.
        return 0.0 if r.mean() > 0 else 1.0
    t = r.mean() / (sd / np.sqrt(len(r)))
    # Survival function of t-distribution (one-sided, upper tail).
    return float(stats.t.sf(t, df=len(r) - 1))


def benjamini_hochberg(p_values: Sequence[float], q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns a boolean mask of rejected nulls
    (i.e. selected templates) at false-discovery rate q.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresholds
    selected = np.zeros(n, dtype=bool)
    if below.any():
        k = np.max(np.where(below)[0])  # largest index satisfying BH condition
        selected_order = order[: k + 1]
        selected[selected_order] = True
    return selected


def _split_half_ok(r: np.ndarray, t: np.ndarray, min_each: int) -> bool:
    """Split trades by median entry time; require each half >= min_each trades
    and mean R > 0 in BOTH halves."""
    if len(r) < 2 * min_each:
        return False
    median_t = np.median(t)
    first = r[t <= median_t]
    second = r[t > median_t]
    # If the median lands on many ties, rebalance by ordering on time then index.
    if len(first) < min_each or len(second) < min_each:
        order = np.argsort(t, kind="stable")
        half = len(r) // 2
        first = r[order[:half]]
        second = r[order[half:]]
    if len(first) < min_each or len(second) < min_each:
        return False
    return first.mean() > 0 and second.mean() > 0


def evaluate_templates(
    stats_list: List[TemplateStats],
    min_occurrences: int = MIN_OCCURRENCES,
    split_half_min: int = SPLIT_HALF_MIN,
    fdr_q: float = FDR_Q,
) -> List[TemplateStats]:
    """Apply all gates in order and mark each template's outcome.

    Mutates and returns the input list (with stats filled in) for convenience.
    """
    # Gates 1-3 (per-template).
    pre_fdr: List[TemplateStats] = []
    for ts in stats_list:
        r = np.asarray(ts.r_multiples, dtype=float)
        ts.n = len(r)
        ts.mean_r = float(r.mean()) if ts.n else float("nan")

        if ts.n < min_occurrences:
            ts.reject_reason = f"occurrences {ts.n} < {min_occurrences}"
            continue
        if not (ts.mean_r > 0):
            ts.reject_reason = f"mean R {ts.mean_r:.3f} <= 0"
            continue
        if not _split_half_ok(r, np.asarray(ts.entry_times, dtype=float), split_half_min):
            ts.reject_reason = "failed split-half robustness"
            continue

        ts.p_value = one_sided_t_pvalue(r)
        ts.passed_pre_fdr = True
        pre_fdr.append(ts)

    # Gate 4 (FDR across survivors).
    if pre_fdr:
        mask = benjamini_hochberg([ts.p_value for ts in pre_fdr], q=fdr_q)
        for ts, keep in zip(pre_fdr, mask):
            ts.selected = bool(keep)
            if not keep:
                ts.reject_reason = f"FDR q={fdr_q} (p={ts.p_value:.3f})"
    return stats_list
