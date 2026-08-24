"""Tests for the third-pass modules."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import realism as RE  # noqa: E402


# ---------------------------------------------------------------------------
# product mix
# ---------------------------------------------------------------------------

def test_a_wide_mix_moves_the_bottleneck():
    """A line balanced for the average is balanced for a product it never makes."""
    cycles = [40.0, 42.0, 44.0, 41.0]
    mix = RE.product_mix(4, len(cycles), spread=0.45, seed=1)
    out = RE.bottleneck_by_product(cycles, mix)
    assert out["bottleneck_moves"]
    assert len(out["distinct_bottlenecks"]) > 1


def test_a_zero_spread_mix_leaves_the_bottleneck_alone():
    cycles = [40.0, 42.0, 58.0, 41.0]
    mix = RE.product_mix(3, len(cycles), spread=0.0, seed=1)
    out = RE.bottleneck_by_product(cycles, mix)
    assert not out["bottleneck_moves"]
    assert out["distinct_bottlenecks"] == [2]


# ---------------------------------------------------------------------------
# changeovers
# ---------------------------------------------------------------------------

def test_the_changeover_matrix_is_asymmetric():
    """A symmetric matrix makes sequencing trivial and removes the problem."""
    m = RE.changeover_matrix(["A", "B", "C"], seed=2)
    assert m[("A", "B")] != m[("B", "A")]
    assert m[("A", "A")] == 0.0


def test_bigger_batches_mean_fewer_changeovers_and_more_wip():
    m = RE.changeover_matrix(["A", "B"], seed=3)
    rows = RE.batch_size_sweep(["A", "B"], m, {"A": 200, "B": 200}, 50.0, 86400.0)
    assert rows[0]["n_changeovers"] > rows[-1]["n_changeovers"]
    assert rows[0]["wip_parts"] < rows[-1]["wip_parts"]
    assert rows[0]["throughput_per_hour"] < rows[-1]["throughput_per_hour"]


def test_a_setup_share_above_one_means_the_scenario_is_wrong():
    """Regression: demand of 600 against a horizon good for 320 reported 347%."""
    m = RE.changeover_matrix(["A", "B", "C"], seed=4)
    rows = RE.batch_size_sweep(["A", "B", "C"], m, {"A": 5000, "B": 5000,
                                                    "C": 5000}, 58.0, 28800.0)
    assert rows[0]["setup_share_of_horizon"] > 1.0, (
        "an impossible scenario should produce an obviously impossible number, "
        "not a plausible one")


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def test_enough_operators_means_no_starvation():
    out = RE.operator_starvation(6, 6, attention_frac=0.5, seed=1)
    assert out["fraction_of_demands_unmet"] == pytest.approx(0.0, abs=1e-9)


def test_too_few_operators_costs_throughput():
    out = RE.operator_starvation(1, 6, attention_frac=0.6, seed=1)
    assert out["fraction_of_demands_unmet"] > 0.3
    assert out["throughput_penalty_estimate"] > 0


def test_operator_penalty_grows_as_operators_are_removed():
    prev = -1.0
    for n in (6, 4, 2, 1):
        p = RE.operator_starvation(n, 6, 0.5, seed=2)["fraction_of_demands_unmet"]
        assert p >= prev
        prev = p


# ---------------------------------------------------------------------------
# quality loops
# ---------------------------------------------------------------------------

def test_rework_through_the_bottleneck_costs_more_than_rework_after_it():
    """Same defect rate, completely different consequence."""
    before = RE.quality_loop(0.90, rework_frac=1.0, rework_station_index=0,
                             bottleneck_index=3, base_throughput=50.0)
    after = RE.quality_loop(0.90, rework_frac=1.0, rework_station_index=5,
                            bottleneck_index=3, base_throughput=50.0)
    assert before["rework_passes_bottleneck"]
    assert not after["rework_passes_bottleneck"]
    assert before["loss_per_hour"] > after["loss_per_hour"]


def test_perfect_yield_costs_nothing():
    out = RE.quality_loop(1.0, 0.5, 0, 3, 50.0)
    assert out["loss_per_hour"] == pytest.approx(0.0, abs=1e-9)


def test_scrap_reduces_good_output_even_after_the_bottleneck():
    out = RE.quality_loop(0.90, rework_frac=0.0, rework_station_index=5,
                          bottleneck_index=3, base_throughput=50.0)
    assert out["throughput_after"] == pytest.approx(45.0, rel=1e-6)


# ---------------------------------------------------------------------------
# the combined penalty
# ---------------------------------------------------------------------------

def test_the_combined_penalty_only_ever_reduces_throughput():
    out = RE.combined_penalty(100.0, mix_penalty=0.1, changeover_penalty=0.1,
                              operator_penalty=0.1, quality_penalty=0.1)
    assert out["adjusted"] < out["base"]
    assert out["total_overstatement"] > 1.0
    # Multiplicative, not additive.
    assert out["adjusted"] == pytest.approx(100.0 * 0.9 ** 4, rel=1e-9)


def test_zero_penalties_leave_throughput_alone():
    out = RE.combined_penalty(100.0, mix_penalty=0, changeover_penalty=0,
                              operator_penalty=0, quality_penalty=0)
    assert out["adjusted"] == pytest.approx(100.0)
    assert out["total_overstatement"] == pytest.approx(1.0)


def test_the_ladder_records_every_stage():
    out = RE.combined_penalty(100.0, mix_penalty=0.05, changeover_penalty=0.05,
                              operator_penalty=0.05, quality_penalty=0.05)
    assert len(out["ladder"]) == 5
    tps = [r["throughput"] for r in out["ladder"]]
    assert tps == sorted(tps, reverse=True), "each stage can only reduce"


def test_the_caveat_names_the_interaction():
    """Multiplicative stacking is approximate and the report must say so."""
    out = RE.combined_penalty(100.0, mix_penalty=0.1, changeover_penalty=0.1,
                              operator_penalty=0.1, quality_penalty=0.1)
    assert "interact" in out["caveat"]
