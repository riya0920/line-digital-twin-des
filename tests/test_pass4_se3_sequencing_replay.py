"""Pass 4: sequencing, busy-time failure clocking, and the animation replay.

The tests that matter here are the ones that would have caught the two bugs this
pass actually had: the backward schedule shifting by the wrong set of setups, and
or-opt being tested on an instance where it had nothing to do.
"""
from __future__ import annotations

import copy
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import animate, line, realism, sequencing as sq  # noqa: E402


# ---------------------------------------------------------------------------
# sequencing
# ---------------------------------------------------------------------------

def _instance(seed=3, n_products=4, n_jobs=9):
    jobs = sq.demo_jobs(n_products=n_products, n_jobs=n_jobs, seed=seed)
    prods = sorted({j.product for j in jobs})
    return jobs, realism.changeover_matrix(prods, seed=seed)


def test_every_rule_returns_a_permutation():
    jobs, m = _instance()
    for name, fn in sq.RULES.items():
        seq = fn(list(jobs), m, start_product="A")
        assert sorted(j.jid for j in seq) == sorted(j.jid for j in jobs), name


def test_or_opt_never_makes_the_sequence_worse():
    """And on at least one instance it must make it BETTER.

    The first version of this test used a 4-product instance where
    nearest-neighbour was already at a local optimum, so or-opt returned its
    input unchanged and the test passed while proving nothing about or-opt.
    """
    improved_somewhere = False
    for seed in range(1, 9):
        jobs, m = _instance(seed=seed, n_products=8, n_jobs=12)
        nn = sq.nearest_neighbour(jobs, m, "A")
        before = sq._setup_of(nn, m, "A")
        after = sq._setup_of(sq.improve(nn, m, start_product="A"), m, "A")
        assert after <= before + 1e-9, seed
        improved_somewhere |= after < before - 1e-6
    assert improved_somewhere, "or-opt improved nothing on any instance"


def test_or_opt_stays_at_or_above_the_exact_optimum():
    for seed in (1, 2, 5, 7):
        jobs, m = _instance(seed=seed, n_products=8, n_jobs=10)
        opt = sq.optimal_setup(jobs, m, "A")
        heur = sq._setup_of(sq.rule_min_setup(jobs, m, start_product="A"), m, "A")
        assert heur >= opt["setup_s"] - 1e-6, "heuristic beat the exact optimum"


def test_held_karp_matches_brute_force_on_a_tiny_instance():
    import itertools
    jobs, m = _instance(seed=4, n_products=4, n_jobs=7)
    opt = sq.optimal_setup(jobs, m, "A")
    best = min(sq._setup_of(list(p), m, "A") for p in itertools.permutations(jobs))
    assert opt["setup_s"] == pytest.approx(best, rel=1e-9)


def test_held_karp_refuses_rather_than_hangs():
    jobs, m = _instance(seed=1, n_products=6, n_jobs=20)
    out = sq.optimal_setup(jobs, m, "A", max_n=12)
    assert out["feasible"] is False and "2^n" in out["why"]


def test_asymmetry_is_real_or_the_problem_is_not_interesting():
    _, m = _instance()
    pairs = [(a, b) for (a, b) in m if a != b]
    diffs = [abs(m[(a, b)] - m[(b, a)]) for a, b in pairs]
    assert max(diffs) > 60.0, "matrix is near-symmetric; or-opt vs 2-opt is moot"


def test_setup_cost_of_a_sequence_matches_evaluate():
    jobs, m = _instance()
    seq = sq.rule_edd(jobs, m)
    assert sq._setup_of(seq, m, "A") == pytest.approx(
        sq.evaluate(seq, m, start_product="A")["setup_s"])


# --- backward scheduling ---------------------------------------------------

def test_backward_schedule_is_the_exact_inverse_of_the_forward_walk():
    """The bug this catches: shifting job i by the setups BEFORE it instead of
    the setups AFTER it. Both versions land the last job exactly on the due
    date, so only the release time and the interior starts give it away."""
    jobs, m = _instance()
    seq = sq.rule_min_setup(jobs, m, start_product="A")
    due = sum(j.run_s for j in seq) * 1.4
    b = sq.backward_schedule(seq, m, due_s=due, start_product="A")
    free = [copy.replace(j, ready_s=-1e12) for j in seq]
    f = sq.evaluate(free, m, start_product="A", t0=b["release_s"])
    assert f["schedule"][-1]["finish_s"] == pytest.approx(due, abs=1e-6)
    for r, g in zip(b["schedule"], f["schedule"]):
        assert r["latest_start_s"] == pytest.approx(g["start_s"], abs=1e-6)


def test_backward_schedule_reports_infeasible_when_the_promise_is_impossible():
    jobs, m = _instance()
    seq = sq.rule_min_setup(jobs, m, start_product="A")
    due = sum(j.run_s for j in seq)          # no room at all for the setups
    b = sq.backward_schedule(seq, m, due_s=due, start_product="A")
    assert b["feasible"] is False
    assert b["release_s"] == pytest.approx(-b["total_setup_s"], abs=1e-6)


def test_backward_schedule_flags_ready_violations():
    jobs, m = _instance()
    seq = sq.rule_min_setup(jobs, m, start_product="A")
    for j in seq:
        j.ready_s = 1e6
    b = sq.backward_schedule(seq, m, due_s=sum(j.run_s for j in seq) * 1.4,
                             start_product="A")
    assert len(b["ready_violations"]) == len(seq)


# --- the trade -------------------------------------------------------------

def test_min_setup_wins_on_setup_and_atc_wins_on_tardiness():
    """Neither rule wins both, which is the entire finding."""
    jobs, m = _instance()
    c = sq.compare(jobs, m, start_product="A")
    assert c["min_setup"]["setup_s"] == min(v["setup_s"] for v in c.values())
    assert c["atc"]["total_tardiness_s"] < c["edd"]["total_tardiness_s"]
    assert c["atc"]["setup_s"] > c["min_setup"]["setup_s"]


def test_edd_loses_max_lateness_to_a_setup_aware_rule():
    """Jackson's rule says EDD minimises maximum lateness. That guarantee is for
    a single machine with NO sequence-dependent setups, and this asserts the
    guarantee is genuinely broken here rather than merely assumed to be."""
    jobs, m = _instance()
    c = sq.compare(jobs, m, start_product="A")
    assert c["atc"]["max_lateness_s"] < c["edd"]["max_lateness_s"]


# ---------------------------------------------------------------------------
# busy-time failure clocking
# ---------------------------------------------------------------------------

def test_busy_clock_and_wall_clock_agree_when_nothing_is_ever_idle():
    """The mechanism check. If a station never idles there is no idle time for a
    repair to overlap, so the two clocks must give the same availability. If this
    fails, the difference measured elsewhere is not the effect it is claimed to
    be."""
    def spec(mode):
        return line.LineSpec(name="sat", stations=[
            line.StationSpec("A", 90.0, cv=0.10, mtbf_s=3600, mttr_s=300,
                             buffer_after=20, failure_clock=mode),
            line.StationSpec("B", 20.0, cv=0.10, buffer_after=20)])
    out = {}
    for mode in ("wall", "busy"):
        d = [line.simulate(spec(mode), 3600 * 60, seed=s, warmup_s=3600).down_frac["A"]
             for s in range(20, 28)]
        out[mode] = float(np.mean(d))
    assert out["wall"] == pytest.approx(out["busy"], rel=0.10), out


def test_busy_clock_lowers_downtime_for_a_lightly_loaded_station():
    def spec(mode):
        return line.LineSpec(name="idle", arrival_mean_s=200.0, stations=[
            line.StationSpec("A", 40.0, cv=0.10, mtbf_s=3600, mttr_s=300,
                             buffer_after=10, failure_clock=mode)])
    res = {m: float(np.mean([
        line.simulate(spec(m), 3600 * 60, seed=s, warmup_s=3600).down_frac["A"]
        for s in range(30, 38)])) for m in ("wall", "busy")}
    assert res["busy"] < res["wall"] * 0.6, res


def test_busy_clock_can_fail_a_station_mid_cycle():
    """A failure that only ever lands between parts is a scheduled stop, not a
    breakdown. Splitting the cycle is what makes it a breakdown, and the tell is
    that busy time accumulates in pieces smaller than one cycle."""
    spec = line.LineSpec(name="m", stations=[
        line.StationSpec("A", 300.0, cv=0.0, mtbf_s=400, mttr_s=60,
                         buffer_after=10, failure_clock="busy"),
        line.StationSpec("B", 10.0, cv=0.0, buffer_after=10)])
    r = line.simulate(spec, 3600 * 10, seed=5, log_events=True)
    runs = [e for e in r.events if e[1] == "A" and e[2] == "running"]
    downs = [e for e in r.events if e[1] == "A" and e[2] == "down"]
    assert len(downs) > 5
    # more "running" starts than parts completed => cycles were split
    assert len(runs) > r.completed


def test_logging_events_does_not_perturb_the_run():
    a = line.simulate(line.default_line(), 3600 * 6, seed=7, warmup_s=1800)
    b = line.simulate(line.default_line(), 3600 * 6, seed=7, warmup_s=1800,
                      log_events=True)
    assert a.throughput_per_hour == b.throughput_per_hour
    assert a.utilisation == b.utilisation
    assert b.events and not a.events


def test_wall_clock_is_still_the_default():
    """Every number published before this pass came from wall-clock failures.
    Changing the default would silently invalidate them."""
    assert all(st.failure_clock == "wall" for st in line.default_line().stations)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def logged_run():
    spec = line.default_line()
    return spec, line.simulate(spec, 3600 * 8, seed=7, warmup_s=3600,
                               log_events=True)


def test_replay_frames_match_the_measured_time_fractions(logged_run):
    spec, r = logged_run
    names = [s.name for s in spec.stations]
    frames = animate.replay_frames(spec, r, 2000, t0=r.warmup_s, t1=r.sim_time_s)
    fr = animate.state_fractions(frames, names)
    for n in names:
        # the run's fractions use the full horizon as the denominator; the
        # frames cover post-warmup only, so a few points of slack is expected
        assert fr[n].get("running", 0.0) == pytest.approx(
            r.utilisation[n], abs=0.06), n


def test_replay_refuses_rather_than_falling_back(logged_run):
    spec, _ = logged_run
    r = line.simulate(spec, 3600 * 2, seed=1)
    with pytest.raises(ValueError, match="log_events"):
        animate.replay_frames(spec, r)


def test_replay_gets_the_bottleneck_signature_right_and_the_reconstruction_does_not(
        logged_run):
    """WIP accumulates in front of the constraint and drains behind it. The
    reconstruction's buffer heuristic inverts it -- it shows the line filling up
    DOWNSTREAM of the bottleneck, which is the one thing an animation of a line
    exists to show."""
    spec, r = logged_run
    c = animate.compare_modes(spec, r, n_frames=2000)
    assert c["bottleneck"] == "S3-weld"
    assert c["replay_signature"]["correct_sign"] is True
    assert c["reconstruction_signature"]["correct_sign"] is False


def test_replay_buffers_never_exceed_capacity(logged_run):
    spec, r = logged_run
    caps = [s.buffer_after for s in spec.stations[:-1]]
    frames = animate.replay_frames(spec, r, 1000, t0=r.warmup_s, t1=r.sim_time_s)
    b = np.array([f["b"] for f in frames], dtype=float)
    assert b.shape[1] == len(caps)
    assert (b <= np.array(caps) + 1e-9).all()
    assert (b >= -1e-9).all()


def test_render_labels_its_own_provenance(logged_run, tmp_path):
    spec, r = logged_run
    rep = animate.render(tmp_path / "a.html", spec, r, {}, 120, mode="replay")
    rec = animate.render(tmp_path / "b.html", spec, r, {}, 120, mode="reconstruct")
    assert rep["mode"] == "replay" and "event log" in rep["limit"]
    assert rec["mode"] == "reconstruct" and "not replayed" in rec["limit"]
    assert "event log" in (tmp_path / "a.html").read_text(encoding="utf-8")


def test_render_auto_does_not_claim_a_replay_it_does_not_have(logged_run, tmp_path):
    spec, _ = logged_run
    r = line.simulate(spec, 3600 * 2, seed=1)
    out = animate.render(tmp_path / "c.html", spec, r, {}, 60)
    assert out["mode"] == "reconstruct"
