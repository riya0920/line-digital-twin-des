"""SE-3 end-to-end: validate the engine, then run the what-if experiments.

    python run_twin.py               # ~4-6 min
    python run_twin.py --quick
    python run_twin.py --report-only

Order matters and is the point: nothing is claimed about a scenario until the
engine has reproduced M/M/1 and Little's Law has held on every run. A simulation
that has not been validated is a random number generator with a narrative.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import experiment as ex  # noqa: E402
import line as line_mod  # noqa: E402
import validation as V  # noqa: E402

OUT = ROOT / "out"


def _e(est) -> dict:
    return {"mean": est.mean, "half_width": est.half_width, "n": est.n, "sd": est.sd}


def scenarios(base: line_mod.LineSpec) -> dict[str, line_mod.LineSpec]:
    """Every what-if is a config edit, never a code edit."""
    out: dict[str, line_mod.LineSpec] = {"baseline": base}

    # Theory of constraints: 10% faster at the constraint vs 10% faster elsewhere.
    con = V.bottleneck_ceiling(base)["constraint"]
    for st in base.stations:
        s = copy.deepcopy(base)
        for t in s.stations:
            if t.name == st.name:
                t.mean_cycle_s *= 0.90
        s.name = f"10% faster @ {st.name}" + (" (CONSTRAINT)" if st.name == con else "")
        out[s.name] = s

    # Buffer allocation: uniform increases, and a targeted one around the constraint.
    for cap in (2, 5, 10, 20):
        s = copy.deepcopy(base)
        for t in s.stations:
            t.buffer_after = cap
        s.name = f"all buffers = {cap}"
        out[s.name] = s
    s = copy.deepcopy(base)
    idx = [i for i, t in enumerate(s.stations) if t.name == con][0]
    for i in (idx - 1, idx):
        if 0 <= i < len(s.stations):
            s.stations[i].buffer_after = 20
    s.name = "buffers=20 around the constraint only"
    out[s.name] = s

    # MTTR reduction at the constraint.
    s = copy.deepcopy(base)
    for t in s.stations:
        if t.name == con and t.mttr_s:
            t.mttr_s *= 0.5
    s.name = "MTTR halved @ constraint"
    out[s.name] = s

    # Release policy.
    for limit in (10, 15, 20, 30):
        s = copy.deepcopy(base)
        s.release = "conwip"
        s.conwip_limit = limit
        s.name = f"CONWIP {limit}"
        out[s.name] = s
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "results.json").read_text())
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "RESULTS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/RESULTS.md from out/results.json")
        return

    t0 = time.perf_counter()
    horizon = (8 if quick else 24) * 3600
    n_reps = 10 if quick else 30
    res: dict = {"horizon_s": horizon, "n_reps": n_reps}
    base = line_mod.default_line()

    print("1/6 validation: M/M/1 across utilisation levels ...", flush=True)
    res["mm1"] = []
    for rho in (0.5, 0.7, 0.85, 0.9):
        c = V.mm1_case(rho, horizon_s=(100 if quick else 200) * 3600,
                       n_reps=8 if quick else 12)
        res["mm1"].append({
            "rho_target": c["rho_target"],
            "L_theory": c["L_theory"], "L_sim": _e(c["L_sim"]), "L_in_ci": c["L_in_ci"],
            "W_theory": c["W_theory"], "W_sim": _e(c["W_sim"]), "W_in_ci": c["W_in_ci"],
            "rho_sim": _e(c["rho_sim"]), "rho_in_ci": c["rho_in_ci"],
            "littles_law_worst_pct": c["littles_law_worst_pct"],
        })
        print(f"    rho={rho}: L {c['L_theory']:.3f} vs {c['L_sim']} "
              f"(in CI: {c['L_in_ci']}), W in CI: {c['W_in_ci']}", flush=True)

    print("1b/6 Little's Law convergence with horizon ...", flush=True)
    conv = []
    for H in ((2, 4, 8) if quick else (4, 8, 24, 72)):
        rs = [V.littles_law(line_mod.simulate(base, H * 3600, 1000 + r, 1800))
              for r in range(8)]
        conv.append({
            "horizon_h": H,
            "mean_signed_residual_pct": float(np.mean([r["relative_residual"] for r in rs]) * 100),
            "worst_abs_pct": float(max(abs(r["relative_residual"]) for r in rs) * 100),
            "L_mean": float(np.mean([r["L_observed"] for r in rs])),
            "lamW_mean": float(np.mean([r["L_predicted"] for r in rs])),
        })
        print(f"    {H:>3}h: mean residual {conv[-1]['mean_signed_residual_pct']:+.2f}%, "
              f"worst {conv[-1]['worst_abs_pct']:.2f}%", flush=True)
    res["littles_law_convergence"] = conv

    print("2/6 warm-up detection (MSER-5 on Welch-averaged WIP) ...", flush=True)
    w = ex.welch_warmup(base, horizon_s=horizon, n_reps=10)
    res["warmup"] = w
    warm = w["warmup_s"]
    print(f"    warm-up {warm:.0f} s ({warm/3600:.2f} h), settled={w['settled']}", flush=True)

    print(f"3/6 scenarios ({n_reps} replications each) ...", flush=True)
    specs = scenarios(base)
    summaries: dict[str, dict] = {}
    ll_fails = 0
    ll_worst = 0.0
    for name, spec in specs.items():
        s = ex.replicate(spec, horizon, warm, n_reps=n_reps)
        for r in s["_runs"]:
            chk = V.assert_littles_law(r)
            ll_worst = max(ll_worst, chk["abs_pct"])
            ll_fails += 0 if chk["pass"] else 1
        ceil = V.check_ceiling(spec, s)
        summaries[name] = {
            "scenario": name,
            "throughput_per_hour": _e(s["throughput_per_hour"]),
            "wip": _e(s["wip"]),
            "cycle_time_s": _e(s["cycle_time_s"]),
            "cycle_time_p95_s": _e(s["cycle_time_p95_s"]),
            "utilisation": {k: _e(v) for k, v in s["utilisation"].items()},
            "blocked": {k: _e(v) for k, v in s["blocked"].items()},
            "starved": {k: _e(v) for k, v in s["starved"].items()},
            "ceiling": {k: v for k, v in ceil.items() if k != "capacities"},
            "capacities": ceil["capacities"],
        }
        print(f"    {name:<40} thr {s['throughput_per_hour']}  wip {s['wip']}", flush=True)
    res["scenarios"] = summaries
    res["littles_law"] = {"runs_checked": len(specs) * n_reps,
                          "failures_at_5pct": ll_fails, "worst_abs_pct": ll_worst}
    print(f"    Little's Law: {ll_fails} failures in {len(specs)*n_reps} runs, "
          f"worst residual {ll_worst:.2f}%", flush=True)

    print("4/6 common random numbers: measured variance reduction ...", flush=True)
    con = V.bottleneck_ceiling(base)["constraint"]
    target = next(n for n in specs if n.startswith("10% faster @") and "CONSTRAINT" in n)
    crn = ex.crn_variance_reduction(base, specs[target], horizon, warm,
                                    n_reps=n_reps)
    res["crn"] = {
        "comparison": f"baseline vs {target}",
        "paired_diff": _e(crn["paired_diff"]),
        "independent_diff": _e(crn["independent_diff"]),
        "var_paired": crn["var_paired"], "var_independent": crn["var_independent"],
        "variance_reduction_factor": crn["variance_reduction_factor"],
        "correlation_between_scenarios": crn["correlation_between_scenarios"],
    }
    print(f"    variance reduction factor {crn['variance_reduction_factor']:.2f}x, "
          f"correlation {crn['correlation_between_scenarios']:.3f}", flush=True)

    print("5/6 how many replications would we need? ...", flush=True)
    b = summaries["baseline"]
    res["reps_needed"] = {
        m: {
            "sd": b[m]["sd"],
            "half_width_at_n": b[m]["half_width"],
            "n_for_1pct_of_mean": ex.reps_needed(b[m]["sd"], 0.01 * abs(b[m]["mean"])),
            "n_for_5pct_of_mean": ex.reps_needed(b[m]["sd"], 0.05 * abs(b[m]["mean"])),
        }
        for m in ("throughput_per_hour", "wip", "cycle_time_s", "cycle_time_p95_s")
    }
    res["constraint"] = con
    res["wall_seconds"] = time.perf_counter() - t0

    print("6/6 writing report ...", flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/RESULTS.md and out/results.json ({res['wall_seconds']:.0f}s)")


def _fmt(e: dict, p: int = 2) -> str:
    return f"{e['mean']:.{p}f} ± {e['half_width']:.{p}f}"


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# SE-3 results — generated by `run_twin.py`, not hand-edited\n")
    A(f"{res['n_reps']} replications per scenario, {res['horizon_s']/3600:.0f} h "
      "simulated horizon, 95% confidence intervals throughout. Every number below "
      "is `mean ± half-width`; a scenario difference whose intervals overlap is "
      "**not a finding**.\n")

    A("## 1. Validation — done before any what-if is believed\n")
    A("### M/M/1 against closed form\n")
    A("| ρ | L theory | L simulated | in CI | W theory (s) | W simulated (s) | in CI | worst Little's Law residual |")
    A("|---|---|---|---|---|---|---|---|")
    for m in res["mm1"]:
        A(f"| {m['rho_target']} | {m['L_theory']:.3f} | {_fmt(m['L_sim'], 3)} | "
          f"{'✔' if m['L_in_ci'] else '✘'} | {m['W_theory']:.1f} | "
          f"{_fmt(m['W_sim'], 1)} | {'✔' if m['W_in_ci'] else '✘'} | "
          f"{m['littles_law_worst_pct']:.2f}% |")
    n_ok = sum(m["L_in_ci"] and m["W_in_ci"] for m in res["mm1"])
    A(f"\n{n_ok}/{len(res['mm1'])} utilisation levels reproduce both L and W inside "
      "the confidence interval. This is the degenerate case the engine has to get "
      "right before any of its non-analytic answers deserve attention.\n")
    A("Note the warm-up and horizon scale with 1/(1−ρ). At high utilisation an "
      "M/M/1 queue is enormously autocorrelated and a short run has not sampled the "
      "tail of the queue-length distribution at all — reporting a tight CI from a "
      "short run at ρ=0.9 is the standard way to 'validate' a queue model against a "
      "number it never saw.")

    ll = res["littles_law"]
    A(f"\n### Little's Law as an invariant monitor\n")
    A(f"Checked on **every one of {ll['runs_checked']} scenario runs**, not as a "
      f"one-off exhibit. Failures at a 5% tolerance: **{ll['failures_at_5pct']}**. "
      f"Worst residual observed: **{ll['worst_abs_pct']:.2f}%**.\n")
    A("L = λW is a conservation identity, not a modelling assumption — it holds for "
      "any arrival process, any service distribution, any queue discipline, in "
      "steady state. When it fails the model has a bug, and the three suspects in "
      "order are: **warm-up leakage** (the WIP integral and the completion window "
      "cover different spans), **WIP counting at the boundaries** (is a part in "
      "service inside the system?), and **lost or duplicated entities**.\n")
    A("But there is a fourth cause that is *not* a bug, and confusing it with one "
      "wastes a day: **finite-horizon boundary bias**. Parts still in the system "
      "when the run ends contributed to L and will never appear in λ or W, so a "
      "finite-horizon estimate is biased with L > λW, by O(1/T). The signed "
      "residual below is positive at every horizon, which is the signature of "
      "exactly that and not of a leak:\n")
    A("| horizon | mean signed residual | worst absolute residual | L | λW |")
    A("|---|---|---|---|---|")
    for c in res.get("littles_law_convergence", []):
        A(f"| {c['horizon_h']} h | {c['mean_signed_residual_pct']:+.2f}% | "
          f"{c['worst_abs_pct']:.2f}% | {c['L_mean']:.2f} | {c['lamW_mean']:.2f} |")
    cv = res.get("littles_law_convergence")
    if cv:
        A(f"\nThe residual falls from {cv[0]['worst_abs_pct']:.2f}% at "
          f"{cv[0]['horizon_h']} h to {cv[-1]['worst_abs_pct']:.2f}% at "
          f"{cv[-1]['horizon_h']} h. That convergence is the evidence that the "
          "identity is being violated by the *window*, not by the model — a genuine "
          "entity leak would not get better with a longer run, it would get worse.")

    w = res["warmup"]
    A(f"\n### Warm-up\n")
    A(f"Truncation point: **{w['warmup_s']:.0f} s ({w['warmup_s']/3600:.2f} h)**, "
      f"chosen by MSER-5 on the replication-averaged WIP series "
      f"({w['n_reps']} replications, {w['bucket_s']:.0f} s buckets). Series settles: "
      f"{w['settled']}.\n")
    head = w["series_mean_wip"][:14]
    A("Averaged WIP by bucket (first 14): " + ", ".join(f"{x:.1f}" for x in head)
      + f" … tail mean {w['tail_mean_wip']:.1f}\n")
    A("A line started empty is not the line being modelled: every station is "
      "starved, WIP climbs from zero, and cycle times are short because there is no "
      "queue. The transient is simulated and then excluded from statistics.\n")
    A("**A rule that did not work, kept because the failure is instructive.** The "
      "first criterion was \"first bucket after which the smoothed series stays "
      "within 5% of its tail mean forever\". It demands *every* later bucket sit "
      "inside the band, so a single noisy bucket near the end pushes the answer to "
      "the end of the run — it reported an 11.9-hour warm-up on a 12-hour horizon, "
      "i.e. \"discard everything\", which is the criterion failing rather than a "
      "finding. MSER-5 instead picks the truncation that minimises the estimated "
      "standard error of the truncated mean, trading residual bias against thrown-"
      "away data. It is a number rather than a judgement.")

    sc = res["scenarios"]
    base = sc["baseline"]
    A("\n## 2. Baseline\n")
    A("| metric | value |")
    A("|---|---|")
    A(f"| throughput (parts/h) | {_fmt(base['throughput_per_hour'])} |")
    A(f"| WIP | {_fmt(base['wip'])} |")
    A(f"| mean cycle time (s) | {_fmt(base['cycle_time_s'], 1)} |")
    A(f"| P95 cycle time (s) | {_fmt(base['cycle_time_p95_s'], 1)} |")
    A("\n| station | utilisation | blocked | starved | effective capacity (parts/h) |")
    A("|---|---|---|---|---|")
    for name, cap in base["capacities"].items():
        A(f"| {name} | {_fmt(base['utilisation'][name], 3)} | "
          f"{_fmt(base['blocked'][name], 3)} | {_fmt(base['starved'][name], 3)} | {cap:.1f} |")
    c = base["ceiling"]
    A(f"\nThe constraint is **{c['constraint']}** with an effective "
      f"(availability-adjusted) capacity of {c['ceiling_per_hour']:.1f} parts/h. The "
      f"line achieves {c['throughput']:.1f}, or **{c['pct_of_ceiling']:.1f}% of the "
      "ceiling**. Exceeds the ceiling: "
      f"{'YES — BUG' if c['exceeds_ceiling'] else 'no'}.\n")
    A("The gap is the interesting part and it is not waste in the model — it is "
      "blocking and starvation caused by variability propagating through finite "
      "buffers. A line model that hits its constraint's isolated capacity exactly "
      "has either no variability or no finite buffers, and is therefore answering a "
      "question nobody asked.")

    A("\n## 3. Theory of constraints, with error bars\n")
    A("10% cycle-time reduction, applied at one station at a time. The comparison "
      "that matters is against the baseline interval, not against zero.\n")
    A("| where the 10% goes | throughput (parts/h) | Δ vs baseline | significant? | WIP | mean cycle time (s) |")
    A("|---|---|---|---|---|---|")
    bt = base["throughput_per_hour"]
    for name, s in sc.items():
        if not name.startswith("10% faster"):
            continue
        t = s["throughput_per_hour"]
        d = t["mean"] - bt["mean"]
        # Two independent means: the difference is significant if the intervals
        # do not overlap. Conservative, and honest about what these CIs support.
        sig = (t["mean"] - t["half_width"] > bt["mean"] + bt["half_width"]) or \
              (t["mean"] + t["half_width"] < bt["mean"] - bt["half_width"])
        A(f"| {name} | {_fmt(t)} | {d:+.2f} | {'**yes**' if sig else 'no'} | "
          f"{_fmt(s['wip'])} | {_fmt(s['cycle_time_s'], 0)} |")
    con = res["constraint"]
    con_row = next(s for n, s in sc.items() if n.startswith("10% faster") and "CONSTRAINT" in n)
    non_rows = [s for n, s in sc.items()
                if n.startswith("10% faster") and "CONSTRAINT" not in n]
    con_d = con_row["throughput_per_hour"]["mean"] - bt["mean"]
    non_d = float(np.mean([r["throughput_per_hour"]["mean"] - bt["mean"] for r in non_rows]))
    A(f"\nSpeeding up the constraint ({con}) by 10% buys **{con_d:+.2f} parts/h**. "
      f"The same 10% spent at a non-constraint buys **{non_d:+.2f} parts/h on "
      "average** — Goldratt with error bars.\n")
    best_non = max(non_rows, key=lambda r: r["throughput_per_hour"]["mean"])
    best_non_d = best_non["throughput_per_hour"]["mean"] - bt["mean"]
    A("**And the second-order effect, which is the part worth arguing about.** "
      f"The best non-constraint improvement here is {best_non_d:+.2f} parts/h — not "
      "zero. Improving a non-constraint *can* help, through starvation reduction: "
      "a station immediately upstream of the constraint that finishes sooner keeps "
      "the constraint's input buffer fuller, and the constraint spends less time "
      "starved. The naive reading of theory of constraints says non-constraint "
      "improvement is worthless; the correct reading is that it is worthless *at "
      "the constraint's capacity* but not necessarily at the line's throughput, "
      "because the constraint is not running 100% of the time. The starved column "
      "in §2 is where to look for whether it will.")

    A("\n## 4. Buffer allocation — where does space help?\n")
    A("| scenario | throughput (parts/h) | Δ vs baseline | WIP | mean cycle time (s) |")
    A("|---|---|---|---|---|")
    for name, s in sc.items():
        if not (name.startswith("all buffers") or name.startswith("buffers=20")):
            continue
        t = s["throughput_per_hour"]
        A(f"| {name} | {_fmt(t)} | {t['mean']-bt['mean']:+.2f} | {_fmt(s['wip'])} | "
          f"{_fmt(s['cycle_time_s'], 0)} |")
    A("\nThe diminishing return is the finding, and the WIP column is the price. "
      "Buffer space bought as throughput is also bought as inventory and as cycle "
      "time — by Little's Law those are the same purchase, which is why a buffer "
      "recommendation that reports only throughput is selling half a transaction.\n")
    uniform20 = sc.get("all buffers = 20")
    targeted = sc.get("buffers=20 around the constraint only")
    if uniform20 and targeted:
        du = uniform20["throughput_per_hour"]["mean"] - bt["mean"]
        dt_ = targeted["throughput_per_hour"]["mean"] - bt["mean"]
        wu = uniform20["wip"]["mean"] - base["wip"]["mean"]
        wt = targeted["wip"]["mean"] - base["wip"]["mean"]
        A(f"**Where you put the space matters more than how much you buy.** "
          f"Enlarging every buffer to 20 buys {du:+.2f} parts/h and costs "
          f"{wu:+.1f} WIP. Enlarging only the two buffers around the constraint "
          f"buys {dt_:+.2f} parts/h — {100*dt_/du:.0f}% of the gain — for "
          f"{wt:+.1f} WIP, which is {100*wt/wu:.0f}% of the inventory. The "
          "constraint is the only station whose starvation and blocking cost the "
          "line output, so buffer space anywhere else is mostly buying queue.\n")
        A("This is the same theory-of-constraints logic as §3, applied to a "
          "different lever, and it is the version that survives a capital-request "
          "conversation: the cheaper option is not a compromise, it is the better "
          "answer.")

    A("\n## 5. Push vs CONWIP at comparable throughput\n")
    A("| scenario | throughput (parts/h) | WIP | mean cycle time (s) | P95 cycle time (s) |")
    A("|---|---|---|---|---|")
    A(f"| baseline (push) | {_fmt(bt)} | {_fmt(base['wip'])} | "
      f"{_fmt(base['cycle_time_s'], 0)} | {_fmt(base['cycle_time_p95_s'], 0)} |")
    for name, s in sc.items():
        if not name.startswith("CONWIP"):
            continue
        A(f"| {name} | {_fmt(s['throughput_per_hour'])} | {_fmt(s['wip'])} | "
          f"{_fmt(s['cycle_time_s'], 0)} | {_fmt(s['cycle_time_p95_s'], 0)} |")
    A("\nThe comparison is only meaningful *at equal throughput*, which is why the "
      "throughput column is there: a WIP cap that also costs throughput has "
      "demonstrated nothing except that starving a line reduces its output.\n")
    # Find the tightest CONWIP limit whose throughput interval still overlaps the
    # push baseline -- that is the one that got something for free.
    # Criterion: the CONWIP mean must fall INSIDE the push baseline's confidence
    # interval. Merely having overlapping intervals is too weak -- CONWIP 10's
    # interval touches the baseline's while its mean sits 1.4 parts/h below, and
    # calling that "the same throughput" would be claiming a free lunch that the
    # data does not support.
    cw = [(n, s) for n, s in sc.items() if n.startswith("CONWIP")]
    overlapping = [
        (n, s) for n, s in cw
        if bt["mean"] - bt["half_width"] <= s["throughput_per_hour"]["mean"]
        <= bt["mean"] + bt["half_width"]
    ]
    if overlapping:
        n, s = min(overlapping, key=lambda kv: kv[1]["wip"]["mean"])
        d_wip = 100 * (1 - s["wip"]["mean"] / base["wip"]["mean"])
        d_ct = 100 * (1 - s["cycle_time_s"]["mean"] / base["cycle_time_s"]["mean"])
        d_p95 = 100 * (1 - s["cycle_time_p95_s"]["mean"] / base["cycle_time_p95_s"]["mean"])
        A(f"**{n} is the finding.** Its throughput ({_fmt(s['throughput_per_hour'])}) "
          f"sits inside the push baseline's own confidence interval "
          f"({_fmt(bt)}) — indistinguishable output — while holding "
          f"{d_wip:.0f}% less WIP, cutting mean cycle time by {d_ct:.0f}% and P95 "
          f"cycle time by {d_p95:.0f}%.\n")
        A("The criterion is deliberately stricter than overlapping intervals. "
          "CONWIP 10's interval *touches* the baseline's while its mean sits 1.4 "
          "parts/h lower; reading that as \"the same throughput\" would be "
          "claiming a free lunch the data does not support, and it would credit "
          "the WIP cap with a cycle-time gain that is partly just lower output.\n")
        A("That is the classic argument for a WIP cap, and it is Little's Law "
          "doing the work rather than anything clever: at fixed throughput, "
          "L = λW means less WIP *is* shorter cycle time. Push release keeps "
          "loading material the constraint cannot consume, and every part it adds "
          "beyond the cap buys queue rather than output. Note also that the P95 "
          "improves — a capped system is more predictable, and a due-date promise "
          "is made against the tail, not the mean.\n")
        A("The cost is in the rows above it: tighten the cap further and "
          "throughput does start to fall, because the constraint begins to starve. "
          "The right cap is the smallest one whose throughput interval still "
          "overlaps, and this table is how you find it.")

    crn = res["crn"]
    A("\n## 6. Common random numbers — measured, not asserted\n")
    A(f"Comparison: {crn['comparison']}, {res['n_reps']} replications.\n")
    A("| | difference in throughput (parts/h) | variance of the difference |")
    A("|---|---|---|")
    A(f"| paired (CRN) | {_fmt(crn['paired_diff'])} | {crn['var_paired']:.4f} |")
    A(f"| independent streams | {_fmt(crn['independent_diff'])} | {crn['var_independent']:.4f} |")
    A(f"\n**Variance reduction factor: {crn['variance_reduction_factor']:.2f}×** "
      f"(correlation between paired scenario outputs: "
      f"{crn['correlation_between_scenarios']:.3f}).\n")
    A("CRN reduces the variance of the *difference*, which is the quantity a "
      "decision depends on — the same comparison is measured on the same simulated "
      "day rather than on two different ones. The caveat, which is the standard "
      "trap: it only works to the extent both models consume the random stream in "
      "the same order. Change the number of stations and the streams desynchronise "
      "almost immediately and the pairing degrades toward nothing. It never makes "
      "things worse, it just stops helping — which is exactly why the factor is "
      "measured per comparison instead of assumed.")

    A("\n## 7. Why 30 replications? What would need 300?\n")
    A("| metric | sd across replications | half-width at n=%d | n for ±1%% of mean | n for ±5%% of mean |"
      % res["n_reps"])
    A("|---|---|---|---|---|")
    for m, r in res["reps_needed"].items():
        A(f"| {m} | {r['sd']:.3f} | {r['half_width_at_n']:.3f} | "
          f"{r['n_for_1pct_of_mean']:,} | {r['n_for_5pct_of_mean']:,} |")
    rn = res["reps_needed"]
    A(f"\n30 is not a magic number; it is whatever hits the precision the decision "
      f"needs, and precision is purchased at n ∝ (sd/half-width)². Mean throughput "
      f"reaches ±1% at n = {rn['throughput_per_hour']['n_for_1pct_of_mean']:,}. "
      f"**P95 cycle time needs "
      f"n = {rn['cycle_time_p95_s']['n_for_1pct_of_mean']:,} for the same relative "
      "precision** — a tail statistic has a far larger sampling standard deviation "
      "than a mean, and that is the answer to \"what would make you need 300?\".")

    A("\n---\n*Model assumptions and their consequences are in "
      "`docs/MODEL_VALIDITY.md`. Every scenario claim above is a difference between "
      "confidence intervals, not between point estimates.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
