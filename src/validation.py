"""The credibility core: check the simulation against results that are known.

A simulation is an argument, not an oracle. The only thing that makes its answers
worth anything is that it reproduces the cases where the answer is already known,
and that it never violates the identities that must hold regardless of model.

Three layers, cheapest first:

1. LITTLE'S LAW, on every run, always. L = lambda*W is a conservation identity,
   not a modelling assumption: it holds for any arrival process, any service
   distribution, any queue discipline, in any system in steady state. If it fails,
   the model has a bug -- entities are being lost, duplicated, counted at the
   wrong boundary, or the warm-up is leaking into the statistics. It is checked
   as an INVARIANT MONITOR rather than as a one-off exhibit.

2. M/M/1, against the closed-form results, across utilisation levels. This is the
   degenerate case the whole engine has to get right before any of its
   non-analytic answers deserve attention.

3. BOTTLENECK ARITHMETIC. A line's throughput cannot exceed the effective
   (availability-adjusted) capacity of its constraint, and moving the constraint
   must move the ceiling. Designed-bottleneck scenarios check both directions.

The three debugging suspects when Little's Law fails, in order of likelihood:
warm-up leakage (statistics collected over a window the WIP integral does not
match), WIP counting at the boundaries (is a part in service "in the system"?),
and lost or duplicated entities (a `get` that never returned, a part put twice).
"""
from __future__ import annotations

import numpy as np

import line as line_mod
from experiment import ci


def littles_law(res: line_mod.RunResult) -> dict:
    """L == lambda * W over the post-warm-up window.

    lambda is measured as completions per unit observed time, and W as the mean
    time in system of parts that COMPLETED in that window. Both must refer to the
    same window or the identity fails for a reason that has nothing to do with the
    model -- which is itself the most common false alarm here.
    """
    obs = max(1e-9, res.sim_time_s - res.warmup_s)
    lam = res.completed / obs
    w = float(res.cycle_times.mean()) if len(res.cycle_times) else 0.0
    L = res.wip_time_avg
    pred = lam * w
    resid = (L - pred) / L if L > 0 else 0.0
    return {
        "L_observed": L, "lambda_per_s": lam, "W_mean_s": w,
        "L_predicted": pred, "relative_residual": resid,
        "abs_pct": abs(resid) * 100.0,
    }


def assert_littles_law(res: line_mod.RunResult, tol_pct: float = 5.0) -> dict:
    r = littles_law(res)
    r["pass"] = bool(r["abs_pct"] <= tol_pct)
    return r


# --------------------------------------------------------------------------
# M/M/1
# --------------------------------------------------------------------------

def mm1_theory(lam: float, mu: float) -> dict:
    """Closed form for M/M/1. rho = lam/mu must be < 1 or the queue is unstable."""
    rho = lam / mu
    if rho >= 1:
        return {"rho": rho, "L": float("inf"), "W": float("inf"), "Lq": float("inf")}
    return {
        "rho": rho,
        "L": rho / (1 - rho),              # number in system
        "Lq": rho**2 / (1 - rho),          # number waiting
        "W": 1.0 / (mu - lam),             # time in system
        "Wq": rho / (mu - lam),            # time waiting
    }


def mm1_case(rho: float, mu_per_s: float = 1 / 60.0, horizon_s: float = 400 * 3600,
             n_reps: int = 20, seed: int = 424242) -> dict:
    """Simulate a single-station M/M/1 and compare to theory with CIs.

    Long horizons on purpose: at high utilisation an M/M/1 queue has enormous
    autocorrelation and a short run has not seen the tail of the queue-length
    distribution at all. Reporting a tight CI from a short run at rho=0.9 is the
    classic way to 'validate' a queue model against a number it never sampled.
    """
    lam = rho * mu_per_s
    spec = line_mod.LineSpec(
        name=f"M/M/1 rho={rho}",
        stations=[line_mod.StationSpec("server", 1.0 / mu_per_s, cv=1.0,
                                       dist="exponential", buffer_after=10_000_000)],
        release="push",
        arrival_mean_s=1.0 / lam,
    )
    warm = min(0.25 * horizon_s, 50_000.0 / max(1e-9, (1 - rho)))
    runs = [line_mod.simulate(spec, horizon_s, seed + r, warm) for r in range(n_reps)]
    th = mm1_theory(lam, mu_per_s)
    L = ci([r.wip_time_avg for r in runs])
    W = ci([float(r.cycle_times.mean()) for r in runs if len(r.cycle_times)])
    U = ci([r.utilisation["server"] for r in runs])
    ll = [littles_law(r)["abs_pct"] for r in runs]
    return {
        "rho_target": rho,
        "L_theory": th["L"], "L_sim": L, "L_in_ci": bool(L.lo <= th["L"] <= L.hi),
        "W_theory": th["W"], "W_sim": W, "W_in_ci": bool(W.lo <= th["W"] <= W.hi),
        "rho_theory": th["rho"], "rho_sim": U,
        "rho_in_ci": bool(U.lo <= th["rho"] <= U.hi),
        "littles_law_worst_pct": float(max(ll)) if ll else float("nan"),
        "warmup_s": warm, "horizon_s": horizon_s, "n_reps": n_reps,
    }


# --------------------------------------------------------------------------
# bottleneck arithmetic
# --------------------------------------------------------------------------

def bottleneck_ceiling(spec: line_mod.LineSpec) -> dict:
    """The constraint and the throughput ceiling it implies."""
    caps = {s.name: line_mod.effective_capacity_per_hour(s) for s in spec.stations}
    name = min(caps, key=caps.get)
    return {"constraint": name, "ceiling_per_hour": caps[name], "capacities": caps}


def check_ceiling(spec: line_mod.LineSpec, summary: dict) -> dict:
    """Simulated throughput must not exceed the constraint's effective capacity.

    It will normally sit BELOW it, and the gap is the interesting part: blocking
    and starvation caused by variability propagating through finite buffers mean a
    line never achieves its constraint's isolated capacity. A model that hits the
    ceiling exactly has either no variability or no finite buffers.
    """
    b = bottleneck_ceiling(spec)
    thr = summary["throughput_per_hour"]
    return {
        **b,
        "throughput": thr.mean,
        "throughput_ci_hi": thr.hi,
        "pct_of_ceiling": 100.0 * thr.mean / b["ceiling_per_hour"],
        "exceeds_ceiling": bool(thr.lo > b["ceiling_per_hour"]),
    }
