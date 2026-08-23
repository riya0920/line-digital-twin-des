"""SE-3, the rest: the four unmodelled effects quantified, a distribution-family
sensitivity, validation as a CI gate, busy-time failures, and an animation.

    python complete.py
    python complete.py --quick
    python complete.py --report-only

Mapping to the README's not-built list:

  1  no animation                                  -> stage 5
  2  no dashboards                                 -> stage 5
  3  the validation suite is not a gate            -> stage 4
  4  no product mix, changeovers, operators, quality-> stage 1
  5  no sensitivity over the cycle-time family     -> stage 2
  7  failures clocked on wall time, not busy time  -> stage 3
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import animate as ANIM  # noqa: E402
import line  # noqa: E402
import realism as RE  # noqa: E402
import validation as VAL  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
QUICK = "--quick" in sys.argv
HORIZON = 8 * 3600.0 if QUICK else 24 * 3600.0
WARMUP = 1800.0


# ---------------------------------------------------------------------------
# 1. the four unmodelled effects
# ---------------------------------------------------------------------------

def stage_realism() -> dict:
    spec = line.default_line()
    base = line.simulate(spec, HORIZON, seed=1, warmup_s=WARMUP)
    base_tp = base.throughput_per_hour
    cycles = [s.mean_cycle_s for s in spec.stations]
    bott = int(np.argmax([s.mean_cycle_s / (
        (s.mtbf_s / (s.mtbf_s + s.mttr_s)) if s.mtbf_s else 1.0)
        for s in spec.stations]))

    mix = RE.product_mix(3, len(spec.stations), seed=4)
    bp = RE.bottleneck_by_product(cycles, mix)
    caps = [v["capacity_per_hour"] for v in bp["by_product"].values()]
    mix_penalty = 1 - (float(np.mean(caps)) / max(max(caps), 1e-9))

    matrix = RE.changeover_matrix(mix["products"], seed=5)
    # Demand has to be something the line can actually make in the horizon.
    # The first version asked for 600 parts against a horizon good for ~320, so
    # setup came out at 86% of the horizon at batch 20 and 347% at batch 5 --
    # and a setup share above 1.0 is the tell that the SCENARIO is wrong rather
    # than the model.
    achievable = int(base_tp * HORIZON / 3600.0)
    per_product = max(achievable // len(mix["products"]), 1)
    demand = {p: per_product for p in mix["products"]}
    sweep = RE.batch_size_sweep(mix["products"], matrix,
                                demand, cycles[bott], HORIZON)
    # Pick the batch size a planner would actually choose: the largest setup
    # share that still leaves the line able to meet demand, rather than an
    # arbitrary index into the sweep.
    feasible = [r for r in sweep if r["setup_share_of_horizon"] < 0.5]
    chosen = min(feasible, key=lambda r: r["batch"]) if feasible else sweep[-1]
    changeover_penalty = chosen["setup_share_of_horizon"]

    ops = RE.operator_starvation(2, len(spec.stations), attention_frac=0.35,
                                 seed=6)
    quality = RE.quality_loop(first_pass_yield=0.94, rework_frac=0.6,
                              rework_station_index=1, bottleneck_index=bott,
                              base_throughput=base_tp)
    combined = RE.combined_penalty(
        base_tp, mix_penalty=mix_penalty,
        changeover_penalty=changeover_penalty,
        operator_penalty=ops["throughput_penalty_estimate"],
        quality_penalty=quality["loss_pct"] / 100.0)

    return {"base_throughput": base_tp, "bottleneck_index": bott,
            "mix": {k: v for k, v in bp.items() if k != "by_product"},
            "mix_by_product": bp["by_product"], "mix_penalty": mix_penalty,
            "changeover_sweep": sweep, "changeover_penalty": changeover_penalty,
            "chosen_batch": chosen["batch"], "demand_per_product": per_product,
            "achievable_in_horizon": achievable,
            "operators": ops, "quality": quality, "combined": combined,
            "asymmetry_example": {
                f"{a}->{b}": round(v, 0) for (a, b), v in list(matrix.items())[:6]}}


# ---------------------------------------------------------------------------
# 2. distribution-family sensitivity
# ---------------------------------------------------------------------------

def stage_distribution() -> dict:
    """The assumption the buffer result is most sensitive to.

    Lognormal, exponential and constant cycle times at the SAME mean and (where
    it exists) the same cv. If the buffer recommendation flips between families,
    it is a recommendation about my distributional choice rather than about the
    line -- and nobody measured the real distribution.
    """
    rows = []
    for dist in ("lognormal", "exponential", "constant"):
        for buf in (2, 5, 20):
            spec = line.default_line(f"{dist}-b{buf}")
            for s in spec.stations:
                s.dist = dist
                s.buffer_after = buf
                if dist == "constant":
                    s.cv = 0.0
            reps = 2 if QUICK else 4
            tps = [line.simulate(spec, HORIZON, seed=100 + r,
                                 warmup_s=WARMUP).throughput_per_hour
                   for r in range(reps)]
            rows.append({"dist": dist, "buffer": buf,
                         "throughput": float(np.mean(tps)),
                         "sd": float(np.std(tps)), "reps": reps})
    by_dist: dict[str, dict] = {}
    for r in rows:
        by_dist.setdefault(r["dist"], {})[r["buffer"]] = r["throughput"]
    gains = {d: v[20] - v[2] for d, v in by_dist.items()}
    return {"rows": rows, "by_dist": by_dist, "buffer_gain_b2_to_b20": gains,
            "gain_spread": max(gains.values()) - min(gains.values()),
            "recommendation_stable": all(g > 0 for g in gains.values())}


# ---------------------------------------------------------------------------
# 3. busy-time failures
# ---------------------------------------------------------------------------

def stage_busy_time() -> dict:
    """MTBF clocked on WALL time overstates availability for an idle station.

    A machine that is starved half the day does not accumulate wear while it
    sits there. Clocking failures on wall time gives it the same failure rate as
    one running flat out, which understates the availability of lightly-loaded
    stations and, more importantly, MISATTRIBUTES failures to stations that were
    not working.

    The correction is to scale each station's MTBF by its utilisation, which is
    the first-order version of busy-time clocking: a station running 50% of the
    time should see failures at half the wall-clock rate.
    """
    spec = line.default_line()
    base = line.simulate(spec, HORIZON, seed=11, warmup_s=WARMUP)

    corrected = line.default_line("busy-time")
    for st, name in zip(corrected.stations, [s.name for s in spec.stations]):
        u = base.utilisation.get(name, 1.0)
        if st.mtbf_s:
            st.mtbf_s = st.mtbf_s / max(u, 0.05)
    fixed = line.simulate(corrected, HORIZON, seed=11, warmup_s=WARMUP)

    return {"wall_clock_throughput": base.throughput_per_hour,
            "busy_time_throughput": fixed.throughput_per_hour,
            "delta": fixed.throughput_per_hour - base.throughput_per_hour,
            "pct": 100 * (fixed.throughput_per_hour - base.throughput_per_hour)
            / max(base.throughput_per_hour, 1e-9),
            "utilisation": base.utilisation,
            "down_frac_before": base.down_frac,
            "down_frac_after": fixed.down_frac,
            "method": ("first-order: MTBF scaled by utilisation. A proper "
                       "implementation clocks the failure process only while the "
                       "station is busy, which needs a change in the engine "
                       "rather than in the spec.")}


# ---------------------------------------------------------------------------
# 4. validation as a gate
# ---------------------------------------------------------------------------

def stage_gate() -> dict:
    """Make the validation suite FAIL a build rather than print a number.

    The README's item 3: Little's Law violations were counted and reported, not
    raised. A check that cannot fail is documentation.
    """
    spec = line.default_line()
    r = line.simulate(spec, HORIZON, seed=21, warmup_s=WARMUP)
    checks = []

    lil = VAL.littles_law(r)
    # `relative_residual`, not `relative_error`. Reading a missing key gave NaN
    # and failed the gate -- and a gate that fails on a typo is worse than no
    # gate, because it teaches people to ignore it.
    resid = lil.get("relative_residual")
    if resid is None:
        raise KeyError(f"littles_law returned {sorted(lil)}, no residual")
    checks.append({"check": "Little's Law (L = lambda W)",
                   "value": resid, "limit": 0.05,
                   "passed": abs(resid) < 0.05,
                   "why": ("WIP, throughput and cycle time must be mutually "
                           "consistent; if they are not, one of them is measured "
                           "over the wrong window")})

    cap = line.effective_capacity_per_hour(
        min(spec.stations,
            key=lambda s: line.effective_capacity_per_hour(s)))
    checks.append({"check": "throughput <= bottleneck effective capacity",
                   "value": r.throughput_per_hour, "limit": cap,
                   "passed": r.throughput_per_hour <= cap * 1.02,
                   "why": ("a line cannot beat its constraint; exceeding it means "
                           "the constraint is mis-identified or the warm-up is "
                           "being counted")})

    checks.append({"check": "conservation: created >= completed",
                   "value": r.entities_completed, "limit": r.entities_created,
                   "passed": r.entities_completed <= r.entities_created,
                   "why": "parts cannot be created by the simulation"})

    fracs_ok = all(
        0.0 <= r.utilisation.get(s.name, 0) + r.blocked_frac.get(s.name, 0)
        + r.starved_frac.get(s.name, 0) + r.down_frac.get(s.name, 0) <= 1.05
        for s in spec.stations)
    checks.append({"check": "station time fractions sum to <= 1",
                   "value": 1.0 if fracs_ok else 0.0, "limit": 1.0,
                   "passed": fracs_ok,
                   "why": "a station cannot be in two states at once"})

    failed = [c for c in checks if not c["passed"]]
    return {"checks": checks, "n_failed": len(failed),
            "exit_code": 1 if failed else 0,
            "gate": "python complete.py --gate  (exits non-zero on failure)"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        res = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
        (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
        print("re-rendered docs/COMPLETION.md")
        return

    t0 = time.perf_counter()
    res: dict = {"quick": QUICK}

    print("1/5 the four unmodelled effects ...", flush=True)
    res["realism"] = stage_realism()
    c = res["realism"]["combined"]
    print(f"    {c['base']:.1f} -> {c['adjusted']:.1f} parts/h "
          f"({c['total_overstatement']:.2f}x overstated)", flush=True)

    print("2/5 distribution-family sensitivity ...", flush=True)
    res["distribution"] = stage_distribution()
    print(f"    buffer gain by family: "
          f"{ {k: round(v, 1) for k, v in res['distribution']['buffer_gain_b2_to_b20'].items()} }",
          flush=True)

    print("3/5 busy-time failures ...", flush=True)
    res["busy_time"] = stage_busy_time()
    print(f"    {res['busy_time']['pct']:+.1f}% throughput", flush=True)

    print("4/5 validation as a gate ...", flush=True)
    res["gate"] = stage_gate()
    print(f"    {res['gate']['n_failed']} checks failed", flush=True)

    print("5/5 animation and dashboard ...", flush=True)
    spec = line.default_line()
    r = line.simulate(spec, 3600.0, seed=31, warmup_s=600.0)
    res["animation"] = ANIM.render(OUT / "line.html", spec, r, res)

    res["wall_seconds"] = time.perf_counter() - t0
    (OUT / "completion.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/COMPLETION.md and out/line.html "
          f"({res['wall_seconds']:.0f}s)")

    if "--gate" in sys.argv and res["gate"]["exit_code"]:
        sys.exit(res["gate"]["exit_code"])


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    rl, di, bt, gt = res["realism"], res["distribution"], res["busy_time"], res["gate"]
    A("# SE-3 completion — generated by `complete.py`, not hand-edited\n")

    A("## 1. The four unmodelled effects, priced\n")
    A("MODEL_VALIDITY §2 lists four things the twin does not model and states "
      "that **every one of them biases throughput upward**. That is an admission "
      "that every number the twin produces is an upper bound — and an upper bound "
      "of unknown size is not usable for an investment decision. Here is the "
      "size.\n")
    c = rl["combined"]
    A("| stage | throughput (parts/h) | factor |")
    A("|---|---|---|")
    for row in c["ladder"]:
        f = f"{row['factor']:.3f}" if "factor" in row else "—"
        A(f"| {row['stage']} | {row['throughput']:.1f} | {f} |")
    A(f"\n**The twin overstates throughput by {c['total_overstatement']:.2f}× "
      f"({c['base']:.1f} → {c['adjusted']:.1f} parts/h.)**\n")
    A(f"*{c['caveat']}.*\n")

    A("### Product mix moves the bottleneck\n")
    A("| product | bottleneck station | capacity/h |")
    A("|---|---|---|")
    for p, v in rl["mix_by_product"].items():
        A(f"| {p} | S{v['bottleneck_index'] + 1} | {v['capacity_per_hour']:.1f} |")
    if rl["mix"]["bottleneck_moves"]:
        A(f"\n**The bottleneck is not the same station for every product** — "
          f"stations {[i + 1 for i in rl['mix']['distinct_bottlenecks']]} each take "
          "the constraint depending on what is running. A line balanced for the "
          "average is balanced for a product it never makes, and the buffer sized "
          "for the average bottleneck sits in the wrong place for most of the "
          "schedule.\n")
    else:
        A("\nThe bottleneck happens to stay at one station across this mix, so "
          "the moving-constraint effect does not bite here. It would at a wider "
          "spread, and the multiplier spread is a parameter rather than a "
          "measurement.\n")

    A("### Changeovers make batch size a throughput decision\n")
    A("| batch | changeovers | setup as share of horizon | throughput/h | WIP |")
    A("|---|---|---|---|---|")
    for r in rl["changeover_sweep"]:
        A(f"| {r['batch']} | {r['n_changeovers']} "
          f"| {r['setup_share_of_horizon'] * 100:.1f}% "
          f"| {r['throughput_per_hour']:.1f} | {r['wip_parts']} |")
    A(f"\nDemand is scaled to what the line can actually make in the horizon "
      f"({rl['achievable_in_horizon']} parts, {rl['demand_per_product']} per "
      f"product), and the batch chosen is **{rl['chosen_batch']}** — the "
      "smallest that keeps setup under half the horizon. The first version of "
      "this asked for 600 parts against a horizon good for ~320 and reported a "
      "setup share of 347%, which is the tell that the scenario was wrong rather "
      "than the model.\n")
    A(f"The matrix is **asymmetric on purpose** — e.g. "
      f"{', '.join(f'{k} {v:.0f}s' for k, v in list(rl['asymmetry_example'].items())[:3])} "
      "— because one direction needs a purge and the other does not. A symmetric "
      "matrix makes the sequencing problem trivial and removes the only "
      "interesting thing about it.\n")
    A("**This sweep is biased toward large batches** and says so: it models the "
      "changeover cost and not the WIP cost, the lead-time cost, or the slower "
      "response to a quality problem. A changeover on the *bottleneck* is lost "
      "throughput that can never be recovered, which is what makes this a "
      "throughput decision rather than a warehouse one.\n")

    o = rl["operators"]
    A("### Operators are a shared resource the model has no state for\n")
    A(f"{o['n_operators']} operators across {o['n_stations']} stations, each "
      f"needing attention {o['attention_frac'] * 100:.0f}% of its cycle: "
      f"**{o['fraction_of_demands_unmet'] * 100:.1f}% of attention demands go "
      f"unmet**, costing about "
      f"{o['throughput_penalty_estimate'] * 100:.1f}% of throughput.\n")
    A("The current model has no state for *waiting for a person* — a station can "
      "be up, unblocked and unstarved and still not running — so it counts that "
      f"time as running. ({o['model']}.)\n")

    q = rl["quality"]
    A("### Rework is the expensive failure, and only sometimes\n")
    A(f"First-pass yield {q['first_pass_yield']:.2f}, of which "
      f"{q['rework_frac_of_all'] * 100:.1f}% is reworked and "
      f"{q['scrap_frac'] * 100:.1f}% scrapped. Loss: "
      f"**{q['loss_per_hour']:.1f} parts/h ({q['loss_pct']:.1f}%)**.\n")
    A(f"The rework loop re-enters "
      f"{'at or before' if q['rework_passes_bottleneck'] else 'after'} the "
      "constraint, so each reworked part "
      f"{'consumes bottleneck capacity twice' if q['rework_passes_bottleneck'] else 'costs almost nothing in throughput terms'}. "
      "Same defect rate, completely different consequence depending on where the "
      "loop closes — which is exactly what a line-level average hides.\n")

    A("## 2. Does the buffer recommendation survive the distribution family?\n")
    A("| cycle-time family | buffer 2 | buffer 5 | buffer 20 | gain 2→20 |")
    A("|---|---|---|---|---|")
    for d, v in di["by_dist"].items():
        A(f"| {d} | {v[2]:.1f} | {v[5]:.1f} | {v[20]:.1f} "
          f"| **{di['buffer_gain_b2_to_b20'][d]:+.1f}** |")
    if di["recommendation_stable"]:
        A(f"\n**The recommendation holds in all three families** — buffering "
          f"helps whatever the cycle-time distribution — but the SIZE of the gain "
          f"varies by {di['gain_spread']:.1f} parts/h across them. So the "
          "direction is robust and the magnitude is a statement about my "
          "distributional choice, and **nobody measured the real one**. A "
          "business case built on the magnitude needs that measurement first.\n")
    else:
        A("\n**The recommendation does NOT hold across families**, which makes it "
          "a recommendation about my distributional assumption rather than about "
          "the line. Measuring the real cycle-time distribution is a prerequisite, "
          "not an improvement.\n")

    A("## 3. Failures on busy time, not wall time\n")
    A(f"Wall-clock MTBF: **{bt['wall_clock_throughput']:.1f} parts/h**. "
      f"Utilisation-corrected: **{bt['busy_time_throughput']:.1f}** "
      f"({bt['pct']:+.1f}%).\n")
    A("A machine starved half the day does not accumulate wear while it sits "
      "there. Clocking failures on wall time gives an idle station the same "
      "failure rate as one running flat out, which understates the availability "
      "of lightly-loaded stations and misattributes failures to stations that "
      "were not working.\n")
    A(f"*{bt['method']}*\n")

    A("## 4. Validation as a gate\n")
    A(f"The README's item 3: Little's Law violations were counted and reported, "
      f"not raised. **A check that cannot fail is documentation.** "
      f"`{gt['gate']}`\n")
    A("| check | value | limit | passed | why it matters |")
    A("|---|---|---|---|---|")
    for c_ in gt["checks"]:
        A(f"| {c_['check']} | {c_['value']:.4g} | {c_['limit']:.4g} "
          f"| {'**yes**' if c_['passed'] else '**NO**'} | {c_['why']} |")
    A(f"\nExit code {gt['exit_code']} — {gt['n_failed']} failed.\n")

    a = res["animation"]
    A("## 5. Animation and dashboard\n")
    A(f"`out/line.html`, {a['bytes'] / 1024:.0f} KB, self-contained, "
      f"{a['n_frames']} frames of playback. The spec's red-flag list names "
      "\"analysis without animation\" and it is right for a reason that is not "
      "cosmetic: **trust in a simulation is built visually.** A plant manager "
      "who watches parts pile up in front of S3 and sees S4 starve believes the "
      "bottleneck result in a way no table achieves — and, more usefully, will "
      "spot a modelling error a table hides.\n")

    A("---")
    A(f"*Generated in {res.get('wall_seconds', 0):.0f}s"
      f"{' (quick mode)' if res.get('quick') else ''}.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
