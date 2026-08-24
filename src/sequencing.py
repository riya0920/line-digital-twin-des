"""Sequencing: deciding what order to build in, and what that order costs.

`realism.py` has an asymmetric changeover matrix and a `sequence_cost` that
prices a sequence. It has no way to PRODUCE one -- `batch_size_sweep` builds a
round-robin (A, B, C, A, B, C ...), which on an asymmetric matrix is close to the
worst thing you can do, so every batch-size number it published carried an
avoidable setup penalty that had nothing to do with batch size.

What makes this problem worth solving rather than sorting:

  * The matrix is ASYMMETRIC. A->B is not B->A, so this is an asymmetric TSP,
    and the usual 2-opt move is wrong here: 2-opt works by REVERSING a segment,
    which on a symmetric matrix leaves the interior cost unchanged and on an
    asymmetric one re-prices every arc inside the reversed segment. The correct
    cheap neighbourhood is or-opt -- relocate a short segment without reversing
    it -- so that is what `improve()` uses.

  * Setup and due dates pull in opposite directions. The minimum-setup sequence
    is free to build in whatever order the tooling likes, which is exactly the
    order that ignores who is waiting. Nothing here resolves that; it prices
    both so the trade is visible.

Sequences here are sequences of BATCHES, not of parts. Sequencing individual
parts with a ten-minute changeover between each is not a scheduling problem, it
is a decision not to run the line.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Job:
    """One batch. `due_s` is a promise, `ready_s` is a constraint."""
    jid: str
    product: str
    qty: int
    due_s: float
    ready_s: float = 0.0
    run_s_per_part: float = 60.0

    @property
    def run_s(self) -> float:
        return self.qty * self.run_s_per_part


# ---------------------------------------------------------------------------
# pricing a sequence
# ---------------------------------------------------------------------------

def evaluate(sequence, matrix: dict, *, start_product=None, t0: float = 0.0) -> dict:
    """Walk the sequence forward and price it on BOTH objectives.

    Tardiness is summed, not averaged: ten jobs one hour late and one job ten
    hours late are different situations, and an average hides which one you are
    in. Max lateness is reported alongside for that reason.

    A job cannot start before it is ready, so the walk carries idle time. Setup
    is charged BEFORE the ready check, because the changeover can be done while
    waiting for material -- assuming otherwise would price a setup twice on any
    sequence with a gap in it.
    """
    t = float(t0)
    prev = start_product
    setup_total = 0.0
    idle_total = 0.0
    rows = []
    for job in sequence:
        setup = 0.0 if prev is None else float(matrix.get((prev, job.product), 0.0))
        setup_total += setup
        t += setup
        if t < job.ready_s:
            idle_total += job.ready_s - t
            t = job.ready_s
        start = t
        t += job.run_s
        rows.append({"jid": job.jid, "product": job.product, "setup_s": setup,
                     "start_s": start, "finish_s": t, "due_s": job.due_s,
                     "lateness_s": t - job.due_s})
        prev = job.product

    late = [r["lateness_s"] for r in rows]
    tard = [max(0.0, x) for x in late]
    return {"makespan_s": t - t0, "setup_s": setup_total, "idle_s": idle_total,
            "run_s": sum(j.run_s for j in sequence),
            "total_tardiness_s": float(sum(tard)),
            "max_lateness_s": float(max(late)) if late else 0.0,
            "n_late": int(sum(1 for x in late if x > 1e-9)),
            "n_jobs": len(sequence),
            "on_time_frac": 1.0 - (sum(1 for x in late if x > 1e-9)
                                   / max(len(late), 1)),
            "schedule": rows}


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def rule_fifo(jobs, matrix: dict, **kw):
    return list(jobs)


def rule_spt(jobs, matrix: dict, **kw):
    """Shortest processing time. Minimises MEAN flow time, provably, and it is
    also the rule most likely to leave one large job permanently at the back."""
    return sorted(jobs, key=lambda j: j.run_s)


def rule_edd(jobs, matrix: dict, **kw):
    """Earliest due date. Minimises MAXIMUM lateness on a single machine with no
    setups (Jackson's rule). The setups are what break the guarantee here."""
    return sorted(jobs, key=lambda j: j.due_s)


def rule_min_setup(jobs, matrix: dict, start_product=None, **kw):
    """Nearest-neighbour on the changeover matrix, then or-opt.

    Nearest-neighbour alone is typically well above optimal on an asymmetric
    instance; the improvement pass is what makes it worth using.
    """
    return improve(nearest_neighbour(jobs, matrix, start_product), matrix,
                   start_product=start_product)


def rule_atc(jobs, matrix: dict, start_product=None, k: float = 2.0, **kw):
    """Apparent Tardiness Cost with setups -- the compromise rule.

    Builds greedily, scoring each candidate by how urgent it is DISCOUNTED by
    the setup it would cost:

        score = (1 / p) * exp(-max(0, d - t - p) / (k * p_bar)) * exp(-s / s_bar)

    The first exponential is slack: a job whose due date is far away scores low,
    and the discount vanishes as its slack goes to zero, so urgency wins in the
    end. The second is the setup penalty. `k` sets how far ahead the rule looks
    -- large k and it behaves like min-setup, small k and it behaves like EDD.
    """
    remaining = list(jobs)
    p_bar = float(np.mean([j.run_s for j in jobs])) if jobs else 1.0
    offs = [v for kk, v in matrix.items() if kk[0] != kk[1]]
    s_bar = float(np.mean(offs)) if offs else 1.0
    t = 0.0
    prev = start_product
    out = []
    while remaining:
        best, best_score = None, -np.inf
        for j in remaining:
            s = 0.0 if prev is None else float(matrix.get((prev, j.product), 0.0))
            slack = max(0.0, j.due_s - t - j.run_s - s)
            score = ((1.0 / max(j.run_s, 1e-9))
                     * np.exp(-slack / max(k * p_bar, 1e-9))
                     * np.exp(-s / max(s_bar, 1e-9)))
            if score > best_score:
                best, best_score = j, score
        s = 0.0 if prev is None else float(matrix.get((prev, best.product), 0.0))
        t = max(t + s, best.ready_s) + best.run_s
        prev = best.product
        remaining.remove(best)
        out.append(best)
    return out


RULES = {"fifo": rule_fifo, "spt": rule_spt, "edd": rule_edd,
         "min_setup": rule_min_setup, "atc": rule_atc}


# ---------------------------------------------------------------------------
# construction and improvement
# ---------------------------------------------------------------------------

def nearest_neighbour(jobs, matrix: dict, start_product=None):
    remaining = list(jobs)
    prev = start_product
    out = []
    while remaining:
        if prev is None:
            nxt = remaining[0]
        else:
            nxt = min(remaining, key=lambda j: matrix.get((prev, j.product), 0.0))
        remaining.remove(nxt)
        out.append(nxt)
        prev = nxt.product
    return out


def improve(sequence, matrix: dict, *, start_product=None, max_seg: int = 3,
            max_passes: int = 200):
    """Or-opt: relocate a segment of 1..max_seg jobs, WITHOUT reversing it.

    Reversal is the reason 2-opt is not used. On a symmetric matrix reversing a
    segment leaves every interior arc costing the same, so 2-opt evaluates a move
    in O(1). On an asymmetric matrix every interior arc flips direction and gets
    a different price, so that O(1) evaluation is simply wrong -- it will accept
    moves that make the sequence worse. Or-opt never reverses, so its interior
    cost is invariant and the cheap evaluation is exact.
    """
    seq = list(sequence)
    best = _setup_of(seq, matrix, start_product)
    for _ in range(max_passes):
        improved = False
        for seg in range(1, max_seg + 1):
            for i in range(len(seq) - seg + 1):
                block = seq[i:i + seg]
                rest = seq[:i] + seq[i + seg:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    cand = rest[:j] + block + rest[j:]
                    c = _setup_of(cand, matrix, start_product)
                    if c < best - 1e-9:
                        seq, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return seq


def _setup_of(seq, matrix: dict, start_product) -> float:
    total = 0.0
    prev = start_product
    for j in seq:
        if prev is not None:
            total += float(matrix.get((prev, j.product), 0.0))
        prev = j.product
    return total


def optimal_setup(jobs, matrix: dict, start_product=None, max_n: int = 12) -> dict:
    """Held-Karp exact minimum-setup sequence, for small instances only.

    O(n^2 * 2^n) -- 12 jobs is about 600k states and runs in a couple of
    seconds, 20 jobs is 400 million and does not. It exists to answer the
    question a heuristic cannot answer about itself: how far from optimal is it?
    Publishing a heuristic's cost without that number is publishing a number
    with no scale.
    """
    n = len(jobs)
    if n == 0:
        return {"feasible": True, "setup_s": 0.0, "sequence": []}
    if n > max_n:
        return {"feasible": False,
                "why": f"{n} jobs exceeds the exact solver's limit of {max_n}; "
                       f"Held-Karp is O(n^2 * 2^n)"}
    prods = [j.product for j in jobs]

    def cost(a: int, b: int) -> float:
        return float(matrix.get((prods[a], prods[b]), 0.0))

    start_cost = [0.0 if start_product is None
                  else float(matrix.get((start_product, p), 0.0)) for p in prods]
    dp = {(1 << i, i): (start_cost[i], -1) for i in range(n)}
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for i in subset:
                mask |= 1 << i
            for last in subset:
                prev_mask = mask ^ (1 << last)
                best = (np.inf, -1)
                for k in subset:
                    if k == last or (prev_mask, k) not in dp:
                        continue
                    c = dp[(prev_mask, k)][0] + cost(k, last)
                    if c < best[0]:
                        best = (c, k)
                if best[1] >= 0:
                    dp[(mask, last)] = best
    full = (1 << n) - 1
    end = min(range(n), key=lambda i: dp[(full, i)][0])
    order, mask, last = [], full, end
    while last >= 0:
        order.append(last)
        prev = dp[(mask, last)][1]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return {"feasible": True, "setup_s": float(dp[(full, end)][0]),
            "sequence": [jobs[i].jid for i in order],
            "jobs": [jobs[i] for i in order]}


# ---------------------------------------------------------------------------
# backward scheduling
# ---------------------------------------------------------------------------

def backward_schedule(sequence, matrix: dict, due_s: float,
                      start_product=None) -> dict:
    """Latest start that still hits `due_s`, walking the sequence backwards.

    Forward scheduling answers "when will it be done"; backward answers "when
    must I start", and only the second one tells you that you are already late.
    A backward schedule with a release time in the PAST is the useful output --
    it says the promise cannot be met with this sequence, before anybody cuts
    metal, and the size of the negative number is how much has to give.
    """
    # Each setup sits immediately BEFORE its job. Walking BACKWARDS, a setup
    # pushes everything ahead of it earlier -- so the shift applied to job i is
    # the total of the setups that come AFTER it, jobs i+1..n-1. Getting this
    # the wrong way round (shifting job i by the setups 0..i) is the natural
    # mistake, and it is silent: the numbers all move in the right direction and
    # the last job still lands exactly on the due date. The tell is the release
    # time, which comes out too late by the whole suffix -- it reported a
    # feasible start of 0.0 on an instance that needed to begin 46 minutes
    # before time zero.
    setups = []
    prev = start_product
    for job in sequence:
        setups.append(0.0 if prev is None
                      else float(matrix.get((prev, job.product), 0.0)))
        prev = job.product

    n = len(sequence)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + setups[i]

    run_suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        run_suffix[i] = run_suffix[i + 1] + sequence[i].run_s

    rows = []
    for i, job in enumerate(sequence):
        start = due_s - run_suffix[i] - suffix[i + 1]
        rows.append({"jid": job.jid, "product": job.product,
                     "latest_start_s": start,
                     "latest_finish_s": start + job.run_s,
                     "setup_s": setups[i]})
    cum = suffix[0]
    release = (rows[0]["latest_start_s"] - setups[0]) if rows else due_s
    ready_violations = [
        {"jid": r["jid"], "ready_s": j.ready_s,
         "latest_start_s": r["latest_start_s"]}
        for r, j in zip(rows, sequence) if r["latest_start_s"] < j.ready_s - 1e-9]
    return {"release_s": release, "feasible": release >= -1e-9,
            "slack_s": release, "total_setup_s": cum,
            "ready_violations": ready_violations, "schedule": rows}


# ---------------------------------------------------------------------------
# comparing rules
# ---------------------------------------------------------------------------

def compare(jobs, matrix: dict, start_product=None, **rule_kw) -> dict:
    """Every rule on the same jobs, priced on both objectives."""
    out = {}
    for name, fn in RULES.items():
        seq = fn(list(jobs), matrix, start_product=start_product, **rule_kw)
        assert sorted(j.jid for j in seq) == sorted(j.jid for j in jobs), \
            f"{name} did not return a permutation of the jobs"
        out[name] = evaluate(seq, matrix, start_product=start_product)
        out[name]["order"] = [j.jid for j in seq]
    return out


def demo_jobs(n_products: int = 4, n_jobs: int = 9, seed: int = 3,
              qty: int = 40, run_s_per_part: float = 58.0,
              tightness: float = 1.35):
    """Jobs whose due dates are TIGHT enough that the trade is real.

    `tightness` multiplies total run time to set the due-date horizon. Above
    about 2.0 every rule meets every date and the comparison says nothing; below
    about 1.1 no rule meets any and it says nothing either. The interesting band
    is where some rules make it and others do not, and that is a property of the
    instance, not of the rules -- so it is a parameter and it is stated.
    """
    rng = np.random.default_rng(seed)
    prods = [chr(ord("A") + i) for i in range(n_products)]
    total_run = n_jobs * qty * run_s_per_part
    jobs = []
    for i in range(n_jobs):
        jobs.append(Job(
            jid=f"J{i + 1:02d}", product=prods[int(rng.integers(n_products))],
            qty=int(qty * rng.uniform(0.6, 1.4)),
            due_s=float(rng.uniform(0.35, 1.0) * tightness * total_run),
            ready_s=0.0, run_s_per_part=run_s_per_part))
    return jobs
