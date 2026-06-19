"""Unit tests for the anti-overfit selection gates and FDR."""

import numpy as np

from propalgo.selection import (
    TemplateStats,
    benjamini_hochberg,
    evaluate_templates,
    one_sided_t_pvalue,
)


def test_bh_all_significant():
    p = [0.001, 0.002, 0.003, 0.004]
    sel = benjamini_hochberg(p, q=0.10)
    assert sel.all()


def test_bh_none_significant():
    p = [0.5, 0.6, 0.7, 0.8]
    sel = benjamini_hochberg(p, q=0.10)
    assert not sel.any()


def test_bh_step_up_behaviour():
    # one tiny p plus large ones; BH should select at least the smallest.
    p = [0.001, 0.9, 0.9, 0.9, 0.9]
    sel = benjamini_hochberg(p, q=0.10)
    assert sel[0]
    assert not sel[1:].any()


def test_one_sided_p_positive_mean_small():
    rng = np.random.default_rng(0)
    r = rng.normal(0.5, 1.0, size=200)  # clearly positive mean
    p = one_sided_t_pvalue(r)
    assert p < 0.01


def test_one_sided_p_zero_mean_large():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 1.0, size=200)
    p = one_sided_t_pvalue(r)
    assert p > 0.05


def _mk(r_vals, name="t"):
    r = np.array(r_vals, dtype=float)
    return TemplateStats(
        template_id=name, instrument="X", direction=1,
        r_multiples=r, entry_times=np.arange(len(r), dtype=float),
    )


def test_gate_rejects_too_few_occurrences():
    ts = _mk([1.5] * 10)  # only 10 < 25
    evaluate_templates([ts])
    assert not ts.selected
    assert "occurrences" in ts.reject_reason


def test_gate_rejects_negative_mean():
    rng = np.random.default_rng(2)
    r = rng.normal(-0.2, 1.0, size=40)
    ts = _mk(r)
    evaluate_templates([ts])
    assert not ts.selected


def test_good_template_selected():
    rng = np.random.default_rng(3)
    # strong, consistent positive edge across both halves
    r = np.concatenate([rng.normal(0.6, 0.8, size=40)])
    ts = _mk(r)
    evaluate_templates([ts])
    assert ts.passed_pre_fdr
    assert ts.selected


def test_split_half_blocks_one_sided_edge():
    # first half strongly positive, second half negative -> split-half fails
    first = np.full(20, 1.2)
    second = np.full(20, -0.5)
    ts = _mk(np.concatenate([first, second]))
    evaluate_templates([ts])
    assert not ts.selected
    assert "split-half" in ts.reject_reason
