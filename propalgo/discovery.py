"""Pattern discovery: matrix-profile motifs with a swappable interface.

The default engine is the Matrix Profile (via `stumpy`). Discovery is kept
behind a small interface so an alternative (e.g. SAX, autoencoder, random
shapelets) could be dropped in without touching the backtest/selection code.

A motif is a recurring z-normalised 20-bar shape. For each motif group with at
least two occurrences we keep a `Template`: the z-scored window at the anchor
index plus the list of occurrence start indices found in the training window.

Matching of a template against a series uses `stumpy.match` (z-norm Euclidean,
distance <= tolerance, with an exclusion zone). A pure-numpy fallback is
provided so the pipeline still runs if stumpy is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np

from .config import (
    MATCH_TOLERANCE,
    MAX_DISTANCE,
    MAX_MATCHES,
    MAX_MOTIFS,
    MIN_NEIGHBORS,
    MOTIF_CUTOFF,
    WINDOW,
)
from .features import zscore

try:  # pragma: no cover - exercised implicitly when stumpy is installed
    import stumpy

    _HAS_STUMPY = True
except Exception:  # pragma: no cover
    stumpy = None
    _HAS_STUMPY = False


@dataclass
class Template:
    """A discovered recurring shape."""

    instrument: str
    pattern: np.ndarray            # z-scored anchor window, length = window
    window: int
    occurrences: List[int] = field(default_factory=list)  # start indices (train)
    direction: int = 0             # +1/-1, set during selection from training
    match_tolerance: float = MATCH_TOLERANCE
    # Optional bookkeeping filled in by selection.
    train_mean_r: float = float("nan")
    train_n: int = 0
    p_value: float = float("nan")
    template_id: str = ""


class PatternDiscovery(Protocol):
    """Interface for swappable discovery engines."""

    def discover(self, instrument: str, train_close: np.ndarray) -> List[Template]:
        ...


# --------------------------------------------------------------------------- #
# Numpy fallback matcher (also used to make matching deterministic in tests)
# --------------------------------------------------------------------------- #
def _znorm(a: np.ndarray) -> np.ndarray:
    return zscore(a)


def numpy_match(
    template: np.ndarray,
    series: np.ndarray,
    max_distance: float = MATCH_TOLERANCE,
    exclusion: Optional[int] = None,
) -> np.ndarray:
    """Return start indices where the z-normalised subsequence distance to the
    template is <= max_distance. Applies a simple exclusion zone so overlapping
    windows are not double-counted (keeps the closest in each cluster).

    Distance is z-norm Euclidean: both template and each window are z-scored,
    then the Euclidean distance is taken. This matches stumpy's convention.
    """
    template = np.asarray(template, dtype=float)
    series = np.asarray(series, dtype=float)
    m = len(template)
    n = len(series)
    if n < m:
        return np.array([], dtype=int)

    if exclusion is None:
        exclusion = max(1, m // 2)

    tz = _znorm(template)
    dists = np.full(n - m + 1, np.inf)
    for i in range(n - m + 1):
        wz = _znorm(series[i : i + m])
        dists[i] = np.sqrt(np.sum((wz - tz) ** 2))

    # Greedy selection of below-threshold matches with exclusion zone.
    order = np.argsort(dists)
    taken: List[int] = []
    blocked = np.zeros(len(dists), dtype=bool)
    for idx in order:
        if dists[idx] > max_distance:
            break
        if blocked[idx]:
            continue
        taken.append(int(idx))
        lo = max(0, idx - exclusion)
        hi = min(len(dists), idx + exclusion + 1)
        blocked[lo:hi] = True
    return np.array(sorted(taken), dtype=int)


def match_template(
    template: np.ndarray,
    series: np.ndarray,
    max_distance: float = MATCH_TOLERANCE,
) -> np.ndarray:
    """Match a template against a series, preferring stumpy, numpy as fallback.

    Returns sorted start indices of accepted matches.
    """
    template = np.asarray(template, dtype=float)
    series = np.asarray(series, dtype=float)
    if len(series) < len(template):
        return np.array([], dtype=int)

    if _HAS_STUMPY:
        try:
            res = stumpy.match(template, series, max_distance=max_distance)
            # stumpy.match returns array of [distance, index] rows.
            if res is None or len(res) == 0:
                return np.array([], dtype=int)
            idx = np.array(sorted(int(r[1]) for r in res), dtype=int)
            return idx
        except Exception:
            pass
    return numpy_match(template, series, max_distance=max_distance)


# --------------------------------------------------------------------------- #
# Matrix-profile discovery (default)
# --------------------------------------------------------------------------- #
class MatrixProfileDiscovery:
    """Discover motifs with stumpy's matrix profile.

    Falls back to a self-matching numpy scan if stumpy is unavailable, so the
    pipeline still produces templates (just more slowly) offline.
    """

    def __init__(
        self,
        window: int = WINDOW,
        max_motifs: int = MAX_MOTIFS,
        min_neighbors: int = MIN_NEIGHBORS,
        max_distance: float = MAX_DISTANCE,
        cutoff: float = MOTIF_CUTOFF,
        max_matches: int = MAX_MATCHES,
        match_tolerance: float = MATCH_TOLERANCE,
    ):
        self.window = window
        self.max_motifs = max_motifs
        self.min_neighbors = min_neighbors
        self.max_distance = max_distance
        self.cutoff = cutoff
        self.max_matches = max_matches
        self.match_tolerance = match_tolerance

    def discover(self, instrument: str, train_close: np.ndarray) -> List[Template]:
        train_close = np.asarray(train_close, dtype=float)
        m = self.window
        if len(train_close) < m * 3:
            return []

        if _HAS_STUMPY:
            return self._discover_stumpy(instrument, train_close)
        return self._discover_numpy(instrument, train_close)

    # -- stumpy path ------------------------------------------------------- #
    def _discover_stumpy(self, instrument: str, train_close: np.ndarray):
        m = self.window
        try:
            mp = stumpy.stump(train_close, m=m)
            profile = mp[:, 0].astype(float)
            motif_distances, motif_indices = stumpy.motifs(
                train_close,
                profile,
                min_neighbors=self.min_neighbors,
                max_distance=self.max_distance,
                cutoff=self.cutoff,
                max_matches=self.max_matches,
                max_motifs=self.max_motifs,
            )
        except Exception:
            return self._discover_numpy(instrument, train_close)

        templates: List[Template] = []
        for gi, group in enumerate(motif_indices):
            occ = [int(x) for x in group if x >= 0]
            occ = sorted(set(occ))
            if len(occ) < 2:
                continue
            anchor = occ[0]
            pattern = zscore(train_close[anchor : anchor + m])
            templates.append(
                Template(
                    instrument=instrument,
                    pattern=pattern,
                    window=m,
                    occurrences=occ,
                    match_tolerance=self.match_tolerance,
                    template_id=f"{instrument}_mp{gi}",
                )
            )
        return templates

    # -- numpy fallback path ---------------------------------------------- #
    def _discover_numpy(self, instrument: str, train_close: np.ndarray):
        """Cheap approximate discovery: scan candidate anchors, group by
        z-norm self-matches. Used only when stumpy is missing."""
        m = self.window
        n = len(train_close)
        templates: List[Template] = []
        used = np.zeros(n - m + 1, dtype=bool)
        # Stride anchors to keep the O(n^2) fallback affordable.
        stride = max(1, m // 2)
        count = 0
        for anchor in range(0, n - m + 1, stride):
            if used[anchor] or count >= self.max_motifs:
                continue
            pattern = train_close[anchor : anchor + m]
            occ = numpy_match(pattern, train_close, max_distance=self.max_distance)
            if len(occ) < 2:
                continue
            for o in occ:
                lo = max(0, o - m // 2)
                hi = min(len(used), o + m // 2 + 1)
                used[lo:hi] = True
            templates.append(
                Template(
                    instrument=instrument,
                    pattern=zscore(pattern),
                    window=m,
                    occurrences=[int(x) for x in occ],
                    match_tolerance=self.match_tolerance,
                    template_id=f"{instrument}_np{count}",
                )
            )
            count += 1
        return templates


def get_discovery(name: str = "matrix_profile", **kwargs) -> PatternDiscovery:
    """Factory for discovery engines (keeps discovery swappable)."""
    name = (name or "matrix_profile").lower()
    if name in ("matrix_profile", "mp", "stumpy", "default"):
        return MatrixProfileDiscovery(**kwargs)
    raise ValueError(f"Unknown discovery engine: {name!r}")
