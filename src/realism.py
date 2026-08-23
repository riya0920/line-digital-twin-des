"""Product mix, changeovers, operators and quality loops -- the four omissions
MODEL_VALIDITY §2 says all bias throughput UPWARD.

WHY THIS IS THE MOST IMPORTANT REMAINING ITEM. The project's own validity
document lists four effects it does not model and states that every one of them
reduces throughput. That is an admission that **every number the twin produces is
an upper bound**, and an upper bound of unknown size is not much use for an
investment decision: "buy the buffer, it gains 12 parts/hour" means nothing if
the unmodelled effects cost 30.

So the point of this module is not realism for its own sake. It is to put a
number on the gap, so the recommendation memo can say how far the simulated
answer can be trusted.

THE FOUR, and the mechanism by which each one costs throughput:

  PRODUCT MIX      different SKUs have different cycle times per station, so the
                   bottleneck can MOVE between products. A line balanced for the
                   average is balanced for a product it never makes.

  CHANGEOVERS      sequence-dependent setup. Going A->B may cost 10 minutes and
                   B->A 40, because one direction needs a purge and the other
                   does not. This is the one that interacts with batch size, and
                   it is why the batch-size decision is a throughput decision and
                   not a warehouse one.

  OPERATORS        a shared resource across stations. Two operators covering six
                   stations means a station can be up, unblocked, unstarved, and
                   still not running -- a state the current model has no way to
                   represent, so it counts that time as running.

  QUALITY LOOPS    a rejected part either leaves (yield loss) or comes BACK for
                   rework. Rework is the worse of the two, because the part
                   consumes bottleneck capacity twice, and it arrives out of
                   sequence.

WHAT IS STILL NOT MODELLED after this: operator skill differences, tool wear
raising cycle time over a run, material shortages, and the scheduler itself.
Those stay named rather than silently absent.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# product mix and changeovers
# ---------------------------------------------------------------------------

def product_mix(n_products: int = 3, n_stations: int = 6, spread: float = 0.25,
                seed: int = 0) -> dict:
    """Per-product cycle-time multipliers per station.

    `spread` is the coefficient of variation ACROSS products at one station. At
    0.25 the bottleneck moves between products, which is the effect worth
    modelling -- a smaller spread would just add noise to the average.
    """
    rng = np.random.default_rng(seed)
    names = [chr(ord("A") + i) for i in range(n_products)]
    mult = {p: np.clip(rng.normal(1.0, spread, n_stations), 0.5, 2.0)
            for p in names}
    return {"products": names, "multipliers": {k: v.tolist()
                                               for k, v in mult.items()},
            "_mult": mult}


def bottleneck_by_product(base_cycles: list[float], mix: dict) -> dict:
    """Which station is the constraint for each product.

    If the answer differs across products, a line balanced for the average is
    balanced for a product it never makes -- and the buffer sized for the average
    bottleneck sits in the wrong place for most of the schedule.
    """
    out = {}
    for p, m in mix["_mult"].items():
        eff = np.asarray(base_cycles) * m
        out[p] = {"bottleneck_index": int(np.argmax(eff)),
                  "bottleneck_cycle_s": float(eff.max()),
                  "capacity_per_hour": 3600.0 / float(eff.max())}
    idxs = {v["bottleneck_index"] for v in out.values()}
    return {"by_product": out, "bottleneck_moves": len(idxs) > 1,
            "distinct_bottlenecks": sorted(idxs)}


def changeover_matrix(products: list[str], seed: int = 0,
                      base_s: float = 600.0) -> dict:
    """Sequence-dependent setup times. Deliberately ASYMMETRIC.

    A->B costing 10 minutes and B->A costing 40 is the normal case, not an edge
    case: one direction needs a purge, a colour flush or a tool change and the
    other does not. A symmetric matrix makes the sequencing problem trivial and
    removes the only interesting thing about it.
    """
    rng = np.random.default_rng(seed)
    m = {}
    for a in products:
        for b in products:
            m[(a, b)] = 0.0 if a == b else float(
                base_s * rng.uniform(0.4, 3.0))
    return m


def sequence_cost(sequence: list[str], matrix: dict) -> float:
    return sum(matrix.get((a, b), 0.0)
               for a, b in zip(sequence, sequence[1:]))


def batch_size_sweep(products: list[str], matrix: dict, demand: dict,
                     bottleneck_cycle_s: float, horizon_s: float) -> list[dict]:
    """The batch-size trade, which is a throughput decision and not a storage one.

    Small batches mean more changeovers, and a changeover on the BOTTLENECK is
    lost throughput that can never be recovered -- the hour is gone. Large batches
    mean fewer changeovers and more work-in-process, longer lead times, and a
    slower response to a quality problem, because more parts are made before
    anybody sees the first one.

    Only the first half of that is modelled here, so the optimum this produces
    is biased toward LARGE batches. Said plainly rather than left for the reader
    to notice.
    """
    total = sum(demand.values())
    rows = []
    for batch in (5, 10, 20, 40, 80, 160):
        n_batches = max(int(np.ceil(total / batch)), 1)
        seq = []
        for i in range(n_batches):
            seq.append(products[i % len(products)])
        setup = sequence_cost(seq, matrix)
        run = total * bottleneck_cycle_s
        avail = max(horizon_s - setup, 0.0)
        rows.append({
            "batch": batch, "n_changeovers": max(n_batches - 1, 0),
            "setup_seconds": setup,
            "setup_share_of_horizon": setup / max(horizon_s, 1e-9),
            "parts_possible": avail / max(bottleneck_cycle_s, 1e-9),
            "throughput_per_hour": (avail / max(bottleneck_cycle_s, 1e-9))
            / max(horizon_s / 3600.0, 1e-9),
            "wip_parts": batch,
        })
    return rows


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def operator_starvation(n_operators: int, n_stations: int,
                        attention_frac: float, seed: int = 0,
                        n_samples: int = 20000) -> dict:
    """How often a station is up, unblocked, unstarved -- and still not running.

    `attention_frac` is the share of its cycle a station needs an operator
    present (load, unload, gauge). With fewer operators than stations, the
    demands collide, and the current model has no state for "waiting for a
    person" -- so it counts that time as running.

    A binomial occupancy model rather than a queueing one. It is crude, and the
    direction is what matters: the current twin's throughput is too high by
    roughly this fraction, and knowing whether that is 2% or 20% decides whether
    the buffer recommendation survives.
    """
    rng = np.random.default_rng(seed)
    demand = rng.random((n_samples, n_stations)) < attention_frac
    needed = demand.sum(axis=1)
    unmet = np.maximum(needed - n_operators, 0)
    # A station wanting attention when demand exceeds supply waits with
    # probability unmet/needed.
    p_wait = np.where(needed > 0, unmet / np.maximum(needed, 1), 0.0)
    return {"n_operators": n_operators, "n_stations": n_stations,
            "attention_frac": attention_frac,
            "mean_stations_wanting_attention": float(needed.mean()),
            "fraction_of_demands_unmet": float(p_wait.mean()),
            "throughput_penalty_estimate": float(attention_frac * p_wait.mean()),
            "model": "binomial occupancy; crude, and the direction is the point"}


# ---------------------------------------------------------------------------
# quality loops
# ---------------------------------------------------------------------------

def quality_loop(first_pass_yield: float, rework_frac: float,
                 rework_station_index: int, bottleneck_index: int,
                 base_throughput: float) -> dict:
    """Scrap versus rework, and why rework is the expensive one.

    A part that fails either leaves (yield loss) or comes back. Rework is worse
    when the loop passes back THROUGH the bottleneck, because the part consumes
    the constraint's capacity twice -- and the constraint's capacity is the
    line's capacity, so every reworked part is a part not made.

    A part reworked at a station downstream of the bottleneck costs almost
    nothing in throughput terms. Same defect rate, completely different
    consequence, and it is the sort of thing a line-level average hides.
    """
    fail = 1.0 - first_pass_yield
    scrap = fail * (1 - rework_frac)
    rework = fail * rework_frac
    through_bottleneck = rework_station_index <= bottleneck_index
    # Scrap reduces good output directly. Rework consumes capacity again only if
    # the loop re-enters at or before the constraint.
    good_frac = 1.0 - scrap
    capacity_consumed = 1.0 + (rework if through_bottleneck else 0.0)
    eff = base_throughput * good_frac / capacity_consumed
    return {
        "first_pass_yield": first_pass_yield, "scrap_frac": scrap,
        "rework_frac_of_all": rework,
        "rework_passes_bottleneck": through_bottleneck,
        "throughput_before": base_throughput, "throughput_after": eff,
        "loss_per_hour": base_throughput - eff,
        "loss_pct": 100 * (base_throughput - eff) / max(base_throughput, 1e-9),
    }


# ---------------------------------------------------------------------------
# the combined correction
# ---------------------------------------------------------------------------

def combined_penalty(base_throughput: float, *, mix_penalty: float,
                     changeover_penalty: float, operator_penalty: float,
                     quality_penalty: float) -> dict:
    """Stack the four, multiplicatively, and say why that is approximate.

    Multiplicative rather than additive because each is a fractional loss of the
    remaining capacity, not an independent subtraction. It is still an
    approximation: the effects INTERACT, and mostly in the bad direction -- a
    changeover during an operator shortage costs more than either alone, because
    the setup itself needs the operator that is not there.

    So this is a better upper bound than the unadjusted number, and still an
    upper bound.
    """
    factors = {"product mix": 1 - mix_penalty,
               "changeovers": 1 - changeover_penalty,
               "operators": 1 - operator_penalty,
               "quality loop": 1 - quality_penalty}
    out = base_throughput
    ladder = [{"stage": "twin as built", "throughput": base_throughput}]
    for name, f in factors.items():
        out *= max(f, 0.0)
        ladder.append({"stage": f"after {name}", "throughput": out,
                       "factor": f})
    return {"ladder": ladder, "base": base_throughput, "adjusted": out,
            "total_overstatement": base_throughput / max(out, 1e-9),
            "caveat": ("multiplicative and therefore approximate; the effects "
                       "interact, mostly in the bad direction -- a changeover "
                       "during an operator shortage costs more than either "
                       "alone, because the setup needs the operator who is not "
                       "there")}
