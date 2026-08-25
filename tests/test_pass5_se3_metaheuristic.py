"""Pass 5: metaheuristics over the or-opt neighbourhood.

The tests that matter guard the two things easiest to get wrong: that the search
still never reverses a segment (the asymmetry argument), and that the temperature
is derived from the data rather than fitted to one instance.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import realism as R              # noqa: E402
import sequencing as sq          # noqa: E402

_spec = importlib.util.spec_from_file_location("p5", ROOT / "run_pass5.py")
P5 = importlib.util.module_from_spec(_spec)
sys.modules["p5"] = P5
_spec.loader.exec_module(P5)

RESULT = ROOT / "out" / "pass5.json"


def _inst(n_jobs=10, n_products=8, seed=2):
    jobs = sq.demo_jobs(n_products=n_products, n_jobs=n_jobs, seed=seed)
    prods = sorted({j.product for j in jobs})
    return jobs, R.changeover_matrix(prods, seed=seed)


# --- the neighbourhood -------------------------------------------------------

def test_a_random_move_is_still_a_permutation():
    jobs, _ = _inst()
    rng = np.random.default_rng(0)
    for _ in range(200):
        cand = sq._neighbour(jobs, rng)
        assert sorted(j.jid for j in cand) == sorted(j.jid for j in jobs)


def test_the_search_never_reverses_a_segment():
    """The asymmetry argument does not stop applying because the search got
    cleverer: 2-opt's O(1) evaluation is only valid on a symmetric matrix."""
    src = (ROOT / "src" / "sequencing.py").read_text(encoding="utf-8")
    body = src[src.index("def _neighbour"):]
    assert "[::-1]" not in body
    assert "reverse" not in body.split("def gap_to_optimal")[0].lower().replace(
        "without reversing", "").replace("no reversal", "")


def test_a_two_job_sequence_is_returned_untouched():
    jobs, m = _inst(n_jobs=2, n_products=2)
    out = sq.simulated_annealing(jobs, m, start_product="A", iters=100)
    assert [j.jid for j in out] == [j.jid for j in jobs]


# --- the temperature ---------------------------------------------------------

def test_the_temperature_is_derived_from_the_data():
    """A hand-picked t0 is a hidden fit to one instance. Scaling every
    changeover by 60 must not change the search's behaviour."""
    jobs, m = _inst()
    m60 = {k: v * 60.0 for k, v in m.items()}
    a = sq.simulated_annealing(jobs, m, start_product="A", seed=3, iters=4000)
    b = sq.simulated_annealing(jobs, m60, start_product="A", seed=3, iters=4000)
    assert [j.jid for j in a] == [j.jid for j in b]


def test_the_schedule_actually_accepts_worse_moves():
    """Otherwise this is `improve` with extra steps and a longer runtime."""
    jobs, m = _inst()
    sq.simulated_annealing(jobs, m, start_product="A", seed=1, iters=5000)
    st = sq.simulated_annealing.last_stats
    assert st["worse_accepted"] > 0
    assert 0.0 < st["worse_accept_frac"] < 1.0
    assert st["t_end"] < st["t0"]


# --- never worse than what it replaces ---------------------------------------

def test_multi_start_is_never_worse_than_or_opt():
    """Its first start IS nearest-neighbour plus or-opt, so it cannot be."""
    for seed in (1, 2, 5, 7):
        jobs, m = _inst(seed=seed)
        base = sq._setup_of(sq.rule_min_setup(jobs, m, start_product="A"), m, "A")
        ms = sq._setup_of(sq.multi_start(jobs, m, start_product="A", seed=0,
                                         restarts=20), m, "A")
        assert ms <= base + 1e-9, seed


def test_neither_method_ever_beats_the_exact_optimum():
    """If a heuristic comes in under Held-Karp, one of them is wrong."""
    for seed in (1, 2, 5, 7):
        jobs, m = _inst(seed=seed)
        opt = sq.optimal_setup(jobs, m, "A")["setup_s"]
        for fn in (lambda: sq.multi_start(jobs, m, start_product="A", seed=0,
                                          restarts=20),
                   lambda: sq.simulated_annealing(jobs, m, start_product="A",
                                                  seed=0, iters=5000)):
            assert sq._setup_of(fn(), m, "A") >= opt - 1e-6, seed


def test_multi_start_closes_the_gap_on_every_instance():
    """The item: or-opt's 10-25% gap. Multi-start closes it on all four, at the
    production restart count."""
    for seed in (1, 2, 5, 7):
        jobs, m = _inst(seed=seed)
        g = sq.gap_to_optimal(jobs, m, "A", seed=0, restarts=P5.RESTARTS,
                              iters=P5.SA_ITERS)
        assert g["exact_feasible"]
        assert g["methods"]["multi-start or-opt"]["optimal"], seed
        assert g["methods"]["or-opt"]["gap_pct"] >= -1e-9


def test_annealing_is_not_monotone_in_its_budget():
    """More iterations is not a finer search: the cooling rate is derived from
    the iteration count, so doubling the budget runs a DIFFERENT search rather
    than a longer one. Seed 5 is optimal at 2k, 11% off at 8k, optimal at 20k.

    This is why annealing is reported as not earning its complexity -- a method
    whose answer does not improve monotonically with effort cannot be tuned by
    giving it more.
    """
    jobs, m = _inst(seed=5)
    opt = sq.optimal_setup(jobs, m, "A")["setup_s"]
    gaps = []
    for it in (2000, 8000, 20000):
        c = sq._setup_of(sq.simulated_annealing(jobs, m, start_product="A",
                                                seed=0, iters=it), m, "A")
        gaps.append(100.0 * (c - opt) / opt)
    assert max(gaps) > 1.0, gaps
    assert gaps != sorted(gaps, reverse=True), (
        "if this became monotone the write-up needs changing")


def test_the_search_is_reproducible():
    jobs, m = _inst()
    a = sq.simulated_annealing(jobs, m, start_product="A", seed=5, iters=3000)
    b = sq.simulated_annealing(jobs, m, start_product="A", seed=5, iters=3000)
    assert [j.jid for j in a] == [j.jid for j in b]


# --- the reported result -----------------------------------------------------

@pytest.mark.skipif(not RESULT.exists(), reason="run run_pass5.py first")
def test_multi_start_wins_and_annealing_does_not():
    """The finding: the more elaborate method loses to the simpler one."""
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    s = d["scaled"]["summary"]
    ms, sa = s["multi-start or-opt"], s["simulated annealing"]
    assert ms["n_best"] == ms["of"], "multi-start should win or tie everywhere"
    assert sa["n_best"] < ms["n_best"]
    assert sa["mean_vs_best_pct"] > ms["mean_vs_best_pct"]


@pytest.mark.skipif(not RESULT.exists(), reason="run run_pass5.py first")
def test_nothing_predicts_when_a_local_search_is_enough():
    """Both explanations tried -- job count and product diversity -- failed, and
    the document has to say so rather than draw a clean rule."""
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    sc = d["scaled"]
    assert sc["oropt_enough_at"] and sc["oropt_short_at"], "need both cases"
    doc = (ROOT / "docs" / "SEQUENCING_GAP.md").read_text(encoding="utf-8")
    assert "nothing predicts" in doc.lower()
    assert "both failed" in doc


@pytest.mark.skipif(not RESULT.exists(), reason="run run_pass5.py first")
def test_the_document_says_best_of_four_is_a_floor():
    doc = (ROOT / "docs" / "SEQUENCING_GAP.md").read_text(encoding="utf-8")
    assert "floor and not the optimum" in doc
