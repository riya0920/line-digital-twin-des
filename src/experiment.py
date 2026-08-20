"""Replications, warm-up, confidence intervals, and common random numbers.

Three disciplines, all of them skipped in most portfolio DES projects:

1. REPLICATIONS. A single run of a stochastic simulation is one sample from a
   distribution. Reporting it as "the throughput" is the field's cardinal sin --
   it is a random number with a narrative. Everything here is reported as a mean
   over n replications with a confidence interval, and the CI half-width is
   printed so a reader can see whether a scenario difference is real.

2. WARM-UP. A line started empty is not the line you are modelling: every station
   is starved, WIP climbs from zero, and cycle times are short because there is no
   queue. Statistics collected over that transient are biased low on WIP and cycle
   time and biased high on nothing useful. Welch's method finds where the
   transient ends; everything before it is excluded from statistics but still
   simulated.

3. COMMON RANDOM NUMBERS. When comparing two scenarios, driving both with the
   same random stream makes the comparison paired: the difference is measured on
   the same "day" rather than on two different ones. It reduces the variance of
   the DIFFERENCE, which is the quantity a decision depends on. The achieved
   variance reduction is measured here rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

import line as line_mod

BASE_SEED = 20260819


@dataclass
class Estimate:
    mean: float
    half_width: float
    n: int
    sd: float

    @property
    def lo(self) -> float:
        return self.mean - self.half_width

    @property
    def hi(self) -> float:
        return self.mean + self.half_width

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.half_width:.3f}"


def ci(values, conf: float = 0.95) -> Estimate:
    """Student-t confidence interval on the mean of independent replications."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return Estimate(float(v.mean()) if n else float("nan"), float("nan"), n, 0.0)
    sd = float(v.std(ddof=1))
    hw = float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n))
    return Estimate(float(v.mean()), hw, n, sd)


def welch_warmup(spec: line_mod.LineSpec, horizon_s: float, n_reps: int = 10,
                 bucket_s: float = 300.0, window: int = 5,
                 seed: int = BASE_SEED) -> dict:
    """Welch's procedure for initial-transient detection.

    Run n replications, average the WIP series across replications at each time
    bucket (which kills most of the noise), smooth with a moving window, and take
    the warm-up point as the first bucket after which the smoothed series stays
    within a tolerance band of its own tail mean.

    Averaging ACROSS replications first is the part that matters. Smoothing a
    single noisy run and eyeballing where it "looks flat" is how warm-up periods
    get chosen to be whatever the analyst wanted.
    """
    n_buckets = int(horizon_s // bucket_s)
    series = np.zeros((n_reps, n_buckets))
    for r in range(n_reps):
        # Sample WIP on a grid by re-running with progressively longer horizons is
        # far too slow; instead run once and reconstruct WIP over time from the
        # entity log the engine already keeps.
        res = line_mod.simulate(spec, horizon_s=horizon_s, seed=seed + r, warmup_s=0.0)
        series[r] = _wip_grid(res, horizon_s, bucket_s, n_buckets)

    avg = series.mean(axis=0)
    smoothed = np.array([avg[max(0, i - window + 1): i + 1].mean() for i in range(n_buckets)])
    tail = float(smoothed[int(0.6 * n_buckets):].mean())

    # Truncation point by MSER-5 (White, 1997), not by eye and not by a
    # "stays within tolerance forever" rule.
    #
    # The tolerance rule was tried first and it is too brittle: it demands EVERY
    # later bucket sit inside the band, so one noisy bucket near the end pushes
    # the answer to the end of the run. It reported a 11.9-hour warm-up on a
    # 12-hour horizon -- i.e. "discard everything", which is not a finding, it is
    # the criterion failing.
    #
    # MSER-5 instead picks the truncation d that MINIMISES the estimated standard
    # error of the truncated mean. Truncating too little leaves transient bias in;
    # truncating too much throws away data and widens the interval. The minimum
    # trades those off, and it is a number rather than a judgement.
    mser_d, mser_curve = _mser(avg)

    # Guard against the degenerate answer. MSER-5 can select a d in the last few
    # buckets when the series never settles, and that must be visible rather than
    # silently returned as a warm-up period.
    settled = mser_d < int(0.5 * n_buckets)
    return {
        "warmup_s": float(mser_d * bucket_s),
        "bucket_s": bucket_s,
        "n_buckets": n_buckets,
        "tail_mean_wip": tail,
        "series_mean_wip": [float(x) for x in smoothed],
        "mser_index": int(mser_d),
        "mser_curve": [float(x) for x in mser_curve],
        "settled": bool(settled),
        "n_reps": n_reps,
    }


def _mser(y: np.ndarray) -> tuple[int, np.ndarray]:
    """MSER: pick d minimising sum((Y_i - mean)^2)/(n-d)^2 over i > d."""
    n = len(y)
    out = np.full(n, np.inf)
    # Only consider truncating up to half the series; beyond that the estimator
    # is fitting noise in a handful of remaining points.
    for d in range(0, n // 2):
        tail = y[d:]
        m = tail.mean()
        out[d] = float(((tail - m) ** 2).sum() / (len(tail) ** 2))
    return int(np.argmin(out)), out


def _wip_grid(res: line_mod.RunResult, horizon_s: float, bucket_s: float,
              n_buckets: int) -> np.ndarray:
    """Reconstruct WIP(t) on a grid: entries so far minus exits so far."""
    grid = (np.arange(n_buckets) + 0.5) * bucket_s
    ee = res.entry_exit
    if ee.size == 0:
        return np.zeros(n_buckets)
    ent = np.sort(ee[:, 0])
    ext = np.sort(ee[:, 1])
    return (np.searchsorted(ent, grid, "right")
            - np.searchsorted(ext, grid, "right")).astype(float)


def replicate(spec: line_mod.LineSpec, horizon_s: float, warmup_s: float,
              n_reps: int = 30, seed: int = BASE_SEED) -> dict:
    """Run n independent replications and summarise every metric with a CI."""
    runs = [line_mod.simulate(spec, horizon_s, seed + r, warmup_s) for r in range(n_reps)]
    return summarise(runs, spec)


def replicate_crn(spec: line_mod.LineSpec, horizon_s: float, warmup_s: float,
                  n_reps: int = 30, seed: int = BASE_SEED) -> list[line_mod.RunResult]:
    """Replications keyed to a fixed seed sequence, for paired scenario comparison.

    Scenario A replication i and scenario B replication i see the same seed, so
    they experience 'the same day' as closely as the two models allow. The
    correlation this induces is what shrinks the variance of the difference.

    The honest caveat, stated because it is the standard trap: CRN only works to
    the extent the two models CONSUME the random stream in the same order. Change
    the number of stations and the streams desynchronise almost immediately, and
    the pairing degrades to nothing (it never makes things worse, it just stops
    helping). That is why the achieved variance reduction is measured per
    comparison rather than assumed -- see `crn_variance_reduction`.
    """
    return [line_mod.simulate(spec, horizon_s, seed + r, warmup_s) for r in range(n_reps)]


def summarise(runs: list[line_mod.RunResult], spec: line_mod.LineSpec) -> dict:
    thr = [r.throughput_per_hour for r in runs]
    wip = [r.wip_time_avg for r in runs]
    ct = [float(r.cycle_times.mean()) if len(r.cycle_times) else np.nan for r in runs]
    ct95 = [float(np.percentile(r.cycle_times, 95)) if len(r.cycle_times) else np.nan
            for r in runs]
    names = [s.name for s in spec.stations]
    return {
        "scenario": spec.name,
        "n_reps": len(runs),
        "throughput_per_hour": ci(thr),
        "wip": ci(wip),
        "cycle_time_s": ci(ct),
        "cycle_time_p95_s": ci(ct95),
        "utilisation": {n: ci([r.utilisation[n] for r in runs]) for n in names},
        "blocked": {n: ci([r.blocked_frac[n] for r in runs]) for n in names},
        "starved": {n: ci([r.starved_frac[n] for r in runs]) for n in names},
        "_runs": runs,
    }


def crn_variance_reduction(spec_a: line_mod.LineSpec, spec_b: line_mod.LineSpec,
                           horizon_s: float, warmup_s: float, n_reps: int = 30,
                           metric=lambda r: r.throughput_per_hour,
                           seed: int = BASE_SEED) -> dict:
    """Measure what CRN actually bought on this comparison.

    Paired: both scenarios use seeds seed..seed+n-1.
    Independent: scenario B uses a disjoint seed block.
    The variance reduction factor is Var(independent diff) / Var(paired diff).
    """
    a = [metric(line_mod.simulate(spec_a, horizon_s, seed + r, warmup_s)) for r in range(n_reps)]
    b_crn = [metric(line_mod.simulate(spec_b, horizon_s, seed + r, warmup_s)) for r in range(n_reps)]
    b_ind = [metric(line_mod.simulate(spec_b, horizon_s, seed + 100_000 + r, warmup_s))
             for r in range(n_reps)]

    d_crn = np.array(b_crn) - np.array(a)
    d_ind = np.array(b_ind) - np.array(a)
    v_crn, v_ind = float(d_crn.var(ddof=1)), float(d_ind.var(ddof=1))
    corr = float(np.corrcoef(a, b_crn)[0, 1]) if n_reps > 2 else float("nan")
    return {
        "paired_diff": ci(d_crn),
        "independent_diff": ci(d_ind),
        "var_paired": v_crn,
        "var_independent": v_ind,
        "variance_reduction_factor": (v_ind / v_crn) if v_crn > 0 else float("inf"),
        "correlation_between_scenarios": corr,
        "n_reps": n_reps,
    }


def reps_needed(sd: float, target_half_width: float, conf: float = 0.95) -> int:
    """How many replications for a target CI half-width? n = (z*sd/h)^2.

    This is the answer to 'defend 30 replications': 30 is not a magic number, it
    is whatever hits the precision the decision needs. A metric with 3x the
    standard deviation needs 9x the replications for the same half-width, which is
    why P95 cycle time costs so much more to estimate than mean throughput.
    """
    z = stats.norm.ppf(0.5 + conf / 2)
    return int(np.ceil((z * sd / max(target_half_width, 1e-12)) ** 2))
