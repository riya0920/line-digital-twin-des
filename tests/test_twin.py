"""SE-3 tests: the checks that make the simulation's answers worth reading.

These are deliberately the *validation* tests rather than unit tests of the
engine's plumbing. A DES that passes M/M/1 and conserves entities is trustworthy
in a way that one with 100% line coverage and no validation is not.
"""
from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import experiment as ex  # noqa: E402
import line as line_mod  # noqa: E402
import validation as V  # noqa: E402


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.85])
def test_mm1_matches_closed_form(rho):
    c = V.mm1_case(rho, horizon_s=150 * 3600, n_reps=10)
    assert c["L_in_ci"], (rho, c["L_theory"], c["L_sim"])
    # W gets a 10% tolerance rather than a CI check, and the reason is documented
    # rather than tuned-until-green. Two effects push it:
    #   * finite-horizon boundary bias, which is POSITIVE (parts still in the
    #     system at the end contributed to L but never to W) and grows with rho
    #   * at high rho an M/M/1 queue is strongly autocorrelated, so a 150-hour run
    #     is a much smaller effective sample than it looks
    # Measured overshoot at rho=0.85 on this horizon is about 6%. Asserting a
    # tighter bound would be asserting something the estimator cannot deliver at
    # this run length; asserting a CI would fail for a reason that is not a bug.
    # MODEL_VALIDITY section 5 carries the same caveat.
    assert c["W_sim"].mean == pytest.approx(c["W_theory"], rel=0.10)
    assert c["W_sim"].mean >= c["W_theory"] * 0.98, "bias should be positive, not negative"
    assert c["littles_law_worst_pct"] < 3.0


def test_littles_law_holds_on_the_default_line():
    for seed in range(4):
        r = line_mod.simulate(line_mod.default_line(), 24 * 3600, 500 + seed, 1800)
        chk = V.assert_littles_law(r, tol_pct=5.0)
        assert chk["pass"], chk


def test_littles_law_residual_shrinks_with_horizon():
    """The residual is a finite-window boundary effect, not an entity leak.
    A leak would get worse with a longer run, not better."""
    def worst(h):
        return max(abs(V.littles_law(
            line_mod.simulate(line_mod.default_line(), h * 3600, 900 + r, 1800)
        )["relative_residual"]) for r in range(4))
    assert worst(24) < worst(4)


def test_throughput_never_exceeds_the_constraint_ceiling():
    spec = line_mod.default_line()
    s = ex.replicate(spec, 24 * 3600, 1800, n_reps=8)
    chk = V.check_ceiling(spec, s)
    assert not chk["exceeds_ceiling"], chk
    assert chk["pct_of_ceiling"] < 100.0


def test_moving_the_constraint_moves_the_ceiling():
    spec = line_mod.default_line()
    before = V.bottleneck_ceiling(spec)
    faster = copy.deepcopy(spec)
    for st in faster.stations:
        if st.name == before["constraint"]:
            st.mean_cycle_s *= 0.5          # no longer the slowest
    after = V.bottleneck_ceiling(faster)
    assert after["constraint"] != before["constraint"]
    assert after["ceiling_per_hour"] > before["ceiling_per_hour"]


def test_constraint_speedup_beats_non_constraint_speedup():
    """Theory of constraints, as a test rather than a slogan."""
    base = line_mod.default_line()
    con = V.bottleneck_ceiling(base)["constraint"]
    non = next(s.name for s in base.stations if s.name != con)

    def sped(name):
        s = copy.deepcopy(base)
        for st in s.stations:
            if st.name == name:
                st.mean_cycle_s *= 0.90
        return ex.replicate(s, 24 * 3600, 1800, n_reps=12)["throughput_per_hour"]

    b = ex.replicate(base, 24 * 3600, 1800, n_reps=12)["throughput_per_hour"]
    at_con, at_non = sped(con), sped(non)
    assert at_con.mean - b.mean > at_non.mean - b.mean
    # And the constraint gain must be big enough to clear the intervals.
    assert at_con.mean - at_con.half_width > b.mean + b.half_width


def test_finite_buffers_actually_block():
    """With finite buffers and variability, upstream stations must spend real
    time blocked. If they do not, the buffers are effectively infinite and the
    model is answering a different question."""
    r = line_mod.simulate(line_mod.default_line(), 12 * 3600, 77, 1800)
    assert max(r.blocked_frac.values()) > 0.01


def test_common_random_numbers_reduce_variance_of_the_difference():
    base = line_mod.default_line()
    faster = copy.deepcopy(base)
    con = V.bottleneck_ceiling(base)["constraint"]
    for st in faster.stations:
        if st.name == con:
            st.mean_cycle_s *= 0.90
    c = ex.crn_variance_reduction(base, faster, 12 * 3600, 1800, n_reps=12)
    assert c["variance_reduction_factor"] > 3.0, c
    assert c["correlation_between_scenarios"] > 0.5, c


def test_streams_are_independent_across_stations():
    """Changing station 3 must not change what station 1 draws. This is the
    property CRN depends on, and it is worth a direct test rather than only an
    indirect one through the variance-reduction number."""
    a = line_mod.streams(1234, 6)
    b = line_mod.streams(1234, 6)
    first_a = a["cycle"][0].standard_normal(5)
    a["cycle"][2].standard_normal(500)          # consume heavily from station 3
    first_b = b["cycle"][0].standard_normal(5)
    assert np.allclose(first_a, first_b)


def test_mser_warmup_is_not_the_whole_run():
    w = ex.welch_warmup(line_mod.default_line(), 24 * 3600, n_reps=6)
    assert 0 < w["warmup_s"] < 12 * 3600
    assert w["settled"]


def test_reps_needed_scales_with_variance_squared():
    # n = (z*sd/h)^2, so doubling sd quadruples n. Compared as a ratio rather than
    # an exact equality because reps_needed ceilings to a whole replication, and
    # ceil(61.46) = 62 is not 4 * ceil(15.37) = 64.
    assert ex.reps_needed(2.0, 0.5) / ex.reps_needed(1.0, 0.5) == pytest.approx(4.0, rel=0.05)
    assert ex.reps_needed(1.0, 0.25) / ex.reps_needed(1.0, 0.5) == pytest.approx(4.0, rel=0.05)
