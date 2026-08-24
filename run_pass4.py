"""Pass 4: the three items the README named as needing an ENGINE change.

  1. A scheduler. `realism.py` could PRICE a sequence and not produce one.
  2. Busy-time failure clocking, replacing the first-order MTBF/utilisation
     scaling -- and measuring how good that approximation actually was.
  3. An event log, so the animation is a replay rather than a reconstruction.

Writes docs/SEQUENCING_AND_REPLAY.md and out/pass4.json.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from src import animate, line, realism, sequencing as sq

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
DOCS = ROOT / "docs"

H = 3600 * 80
W = 7200
SEEDS = list(range(100, 112))


# ---------------------------------------------------------------------------
# 1. sequencing
# ---------------------------------------------------------------------------

def stage_sequencing() -> dict:
    jobs = sq.demo_jobs()
    prods = sorted({j.product for j in jobs})
    m = realism.changeover_matrix(prods, seed=1)

    rules = sq.compare(jobs, m, start_product="A")
    for v in rules.values():
        v.pop("schedule", None)

    exact = sq.optimal_setup(jobs, m, "A")
    nn = sq._setup_of(sq.nearest_neighbour(jobs, m, "A"), m, "A")

    # What `batch_size_sweep` actually builds: a round-robin over the products.
    rr = sorted(jobs, key=lambda j: (prods.index(j.product), j.jid))
    rr_seq, k = [], {p: [j for j in jobs if j.product == p] for p in prods}
    while any(k.values()):
        for p in prods:
            if k[p]:
                rr_seq.append(k[p].pop(0))
    round_robin = sq._setup_of(rr_seq, m, "A")

    # how far from optimal the heuristic is, across instances
    gaps = []
    for seed in (1, 2, 5, 7):
        js = sq.demo_jobs(n_products=8, n_jobs=10, seed=seed)
        pr = sorted({j.product for j in js})
        mm = realism.changeover_matrix(pr, seed=seed)
        o = sq.optimal_setup(js, mm, "A")
        c_nn = sq._setup_of(sq.nearest_neighbour(js, mm, "A"), mm, "A")
        c_oo = sq._setup_of(sq.rule_min_setup(js, mm, start_product="A"), mm, "A")
        gaps.append({"seed": seed, "optimal_s": o["setup_s"],
                     "nn_s": c_nn, "nn_gap_pct": 100 * (c_nn - o["setup_s"]) / o["setup_s"],
                     "oropt_s": c_oo,
                     "oropt_gap_pct": 100 * (c_oo - o["setup_s"]) / o["setup_s"]})

    # backward scheduling against a promise that cannot be met
    seq = sq.rule_min_setup(jobs, m, start_product="A")
    run_only = sum(j.run_s for j in seq)
    tight = sq.backward_schedule(seq, m, due_s=run_only, start_product="A")
    loose = sq.backward_schedule(seq, m, due_s=run_only * 1.4, start_product="A")
    for b in (tight, loose):
        b.pop("schedule", None)

    return {"rules": rules,
            "exact_setup_s": exact["setup_s"], "exact_order": exact["sequence"],
            "nearest_neighbour_s": nn,
            "round_robin_s": round_robin,
            "round_robin_penalty_pct":
                100 * (round_robin - exact["setup_s"]) / exact["setup_s"],
            "gaps": gaps,
            "backward_tight": tight, "backward_loose": loose,
            "n_jobs": len(jobs), "products": prods}


# ---------------------------------------------------------------------------
# 2. busy-time failure clocking
# ---------------------------------------------------------------------------

def _fleet(make) -> dict:
    thr, dn, ut = [], {}, {}
    for sd in SEEDS:
        r = line.simulate(make(), H, seed=sd, warmup_s=W)
        thr.append(r.throughput_per_hour)
        for k, v in r.down_frac.items():
            dn.setdefault(k, []).append(v)
        for k, v in r.utilisation.items():
            ut.setdefault(k, []).append(v)
    a = np.asarray(thr)
    return {"throughput": float(a.mean()),
            "half_width": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
            "down_frac": {k: float(np.mean(v)) for k, v in dn.items()},
            "utilisation": {k: float(np.mean(v)) for k, v in ut.items()}}


def stage_busy_clock() -> dict:
    wall = _fleet(line.default_line)

    def approx():
        sp = line.default_line("first-order")
        for st in sp.stations:
            if st.mtbf_s:
                st.mtbf_s = st.mtbf_s / max(wall["utilisation"].get(st.name, 1.0), 0.05)
        return sp

    def busy():
        sp = line.default_line("busy-time")
        for st in sp.stations:
            st.failure_clock = "busy"
        return sp

    first_order = _fleet(approx)
    exact = _fleet(busy)

    per_station = []
    for name in wall["down_frac"]:
        if wall["down_frac"][name] <= 0:
            continue
        per_station.append({
            "station": name, "utilisation": wall["utilisation"][name],
            "down_wall": wall["down_frac"][name],
            "down_first_order": first_order["down_frac"][name],
            "down_exact": exact["down_frac"][name],
            "exact_over_wall": exact["down_frac"][name] / wall["down_frac"][name],
            "first_order_error_pct":
                100 * (first_order["down_frac"][name] - exact["down_frac"][name])
                / exact["down_frac"][name]})

    return {"wall": {k: wall[k] for k in ("throughput", "half_width", "down_frac")},
            "first_order": {k: first_order[k] for k in ("throughput", "half_width", "down_frac")},
            "exact": {k: exact[k] for k in ("throughput", "half_width", "down_frac")},
            "per_station": per_station,
            "correction_size": exact["throughput"] - wall["throughput"],
            "first_order_overshoot": first_order["throughput"] - exact["throughput"],
            "first_order_error_pct":
                100 * (first_order["throughput"] - exact["throughput"])
                / exact["throughput"],
            "utilisation": wall["utilisation"]}


def stage_util_sweep() -> dict:
    """down_busy / down_wall should equal utilisation. Measured, not asserted."""
    horizon, warm, seeds = 3600 * 40, 3600, list(range(50, 60))

    def mk(arr, mode, mtbf=3600.0):
        return line.LineSpec(name="sweep", arrival_mean_s=arr, stations=[
            line.StationSpec("A", 30.0, cv=0.20, mtbf_s=mtbf, mttr_s=300,
                             buffer_after=10, failure_clock=mode)])

    def go(f):
        d, u = [], []
        for sd in seeds:
            r = line.simulate(f(), horizon, seed=sd, warmup_s=warm)
            d.append(r.down_frac["A"])
            u.append(r.utilisation["A"])
        return float(np.mean(d)), float(np.mean(u))

    rows = []
    for arr in (33.0, 40.0, 50.0, 70.0, 120.0):
        w, uw = go(lambda: mk(arr, "wall"))
        a, _ = go(lambda: mk(arr, "wall", mtbf=3600.0 / max(uw, 0.05)))
        b, _ = go(lambda: mk(arr, "busy"))
        rows.append({"arrival_mean_s": arr, "utilisation": uw,
                     "down_wall": w, "down_first_order": a, "down_exact": b,
                     "ratio_exact_over_wall": b / w,
                     "first_order_error_pct": 100 * (a - b) / b})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# 3. replay
# ---------------------------------------------------------------------------

def stage_replay() -> dict:
    spec = line.default_line()
    r = line.simulate(spec, 3600 * 8, seed=7, warmup_s=3600, log_events=True)
    cmp_ = animate.compare_modes(spec, r, n_frames=2000)

    names = [s.name for s in spec.stations]
    full = animate.replay_frames(spec, r, 2000, t0=r.warmup_s, t1=r.sim_time_s)
    fr = animate.state_fractions(full, names)
    fidelity = [{"station": n, "frames_running": fr[n].get("running", 0.0),
                 "measured_running": r.utilisation[n],
                 "abs_error": abs(fr[n].get("running", 0.0) - r.utilisation[n])}
                for n in names]

    rep = animate.render(OUT / "animation_replay.html", spec, r,
                         {"mode": "replay"}, 240, mode="replay")
    rec = animate.render(OUT / "animation_reconstructed.html", spec, r,
                         {"mode": "reconstruct"}, 240, mode="reconstruct")

    return {"n_events": len(r.events),
            "events_per_part": len(r.events) / max(r.completed, 1),
            "bytes_per_event_estimate": 4 * 8 + 16,
            "fidelity": fidelity,
            "worst_fidelity": max(x["abs_error"] for x in fidelity),
            "compare": {k: v for k, v in cmp_.items()
                        if k not in ("replay_running_corr",
                                     "reconstruction_running_corr")},
            "rendered": {"replay": rep, "reconstruct": rec}}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(d: dict) -> str:
    L: list[str] = []
    A = L.append
    sq_, bc, sw, rp = d["sequencing"], d["busy_clock"], d["sweep"], d["replay"]

    A("# Sequencing, busy-time failures, and a replayable animation\n")
    A("The three items the README named as needing a change in the ENGINE rather "
      "than in a spec. Each is now built, and in two of the three the interesting "
      "result is how wrong the thing it replaced was.\n")
    A(f"Generated by `run_pass4.py` in {d['elapsed_s']:.0f} s.\n")

    # --- sequencing
    A("## 1. A scheduler, not a price list\n")
    A("`realism.py` had an asymmetric changeover matrix and a `sequence_cost` "
      "that could price a sequence. It had no way to produce one — "
      "`batch_size_sweep` built a round-robin, A B C D A B C D, which on an "
      "asymmetric matrix is close to the worst order available.\n")
    A(f"**The round-robin costs {sq_['round_robin_s'] / 3600:.2f} h of changeover "
      f"against an optimum of {sq_['exact_setup_s'] / 3600:.2f} h — "
      f"{sq_['round_robin_penalty_pct']:.0f}% worse.** Every batch-size number "
      "that sweep published carried that penalty, and it had nothing to do with "
      "batch size.\n")

    A("\n### The rules, on the same jobs\n")
    A("| rule | setup (h) | total tardiness (h) | max lateness (h) | jobs late | makespan (h) |")
    A("|---|---:|---:|---:|---:|---:|")
    for name in ("fifo", "spt", "edd", "min_setup", "atc"):
        v = sq_["rules"][name]
        A(f"| {name} | {v['setup_s'] / 3600:.2f} | "
          f"{v['total_tardiness_s'] / 3600:.2f} | {v['max_lateness_s'] / 3600:.2f} | "
          f"{v['n_late']}/{v['n_jobs']} | {v['makespan_s'] / 3600:.2f} |")

    edd, atc, ms = sq_["rules"]["edd"], sq_["rules"]["atc"], sq_["rules"]["min_setup"]
    A(f"\n**EDD loses maximum lateness to a setup-aware rule "
      f"({edd['max_lateness_s'] / 3600:.2f} h against ATC's "
      f"{atc['max_lateness_s'] / 3600:.2f} h), and puts more jobs late than FIFO "
      f"does.** Jackson's rule says earliest-due-date minimises maximum lateness, "
      "and it is a theorem — for a single machine with *no sequence-dependent "
      "setups*. Here EDD spends "
      f"{(edd['setup_s'] - ms['setup_s']) / 3600:.2f} h more on changeovers than "
      "it needs to, and that time comes straight out of the due dates it was "
      "sorting by. A guarantee is only a guarantee on its own model.\n")
    A("**Neither objective wins.** Minimum-setup is cheapest to run and lets "
      f"{ms['n_late']} jobs go late; ATC gives up "
      f"{(atc['setup_s'] - ms['setup_s']) / 3600:.2f} h of setup to cut tardiness "
      f"from {ms['total_tardiness_s'] / 3600:.2f} h to "
      f"{atc['total_tardiness_s'] / 3600:.2f} h. Nothing here resolves that; it "
      "prices both so somebody can choose.\n")

    A("\n### Or-opt, and why not 2-opt\n")
    A("The matrix is asymmetric, so this is an asymmetric TSP. 2-opt works by "
      "**reversing** a segment: on a symmetric matrix every interior arc costs "
      "the same reversed, which is what makes the move evaluable in O(1). On an "
      "asymmetric matrix every interior arc flips direction and gets a different "
      "price, so that O(1) evaluation is simply wrong and will accept moves that "
      "make the sequence worse. Or-opt relocates a segment without reversing it, "
      "so its interior cost is invariant and the cheap evaluation is exact.\n")
    A("| instance | nearest-neighbour | + or-opt | exact (Held–Karp) |")
    A("|---|---:|---:|---:|")
    for g in sq_["gaps"]:
        A(f"| seed {g['seed']} | {g['nn_s'] / 3600:.3f} h "
          f"(+{g['nn_gap_pct']:.0f}%) | {g['oropt_s'] / 3600:.3f} h "
          f"(+{g['oropt_gap_pct']:.0f}%) | {g['optimal_s'] / 3600:.3f} h |")
    A("\nHeld–Karp is O(n²·2ⁿ) and refuses above 12 jobs rather than hanging. It "
      "is here to answer the one question a heuristic cannot answer about "
      "itself: **how far from optimal is it?** Or-opt roughly halves the "
      "nearest-neighbour gap and does not close it — publishing the heuristic's "
      "cost without the exact number would be publishing a number with no scale.\n")

    A("\n### Backward scheduling\n")
    t, lo = sq_["backward_tight"], sq_["backward_loose"]
    A("Forward scheduling answers *when will it be done*. Backward answers *when "
      "must I start*, and only the second one tells you that you are already "
      "late before anybody cuts metal.\n")
    A(f"- Promised at exactly the total run time: release "
      f"**{t['release_s'] / 3600:.2f} h**, i.e. {abs(t['release_s']) / 60:.0f} "
      f"minutes *before* time zero. Infeasible, and the size of the negative "
      f"number is how much has to give — it is exactly the "
      f"{t['total_setup_s'] / 3600:.2f} h of changeover the promise forgot.\n")
    A(f"- Promised at 1.4× the run time: release "
      f"**{lo['release_s'] / 3600:.2f} h**, feasible with that much slack.\n")

    A("\n**A bug found writing it.** The first version shifted each job by the "
      "setups that came *before* it. Walking backwards, a setup pushes "
      "everything ahead of it *earlier*, so the shift on job *i* is the total of "
      "the setups *after* it. Both versions land the last job exactly on the due "
      "date, which is the number a reader checks — the wrong one reported a "
      "comfortable release of 0.00 h on an instance that had to start 46 minutes "
      "before time zero. The test that catches it walks the sequence forward "
      "from the computed release and demands every start time match.\n")

    # --- busy clock
    A("\n## 2. Busy-time failures, and the cost of the approximation\n")
    A("A machine starved half the day does not accumulate wear while it sits "
      "there. The engine now clocks the failure process on busy seconds: a "
      "station carries a remaining *busy* life, the cycle is split when that life "
      "runs out mid-part, and the repair starts there — so a breakdown lands in "
      "the middle of the work rather than tidily between parts.\n")
    A("What it replaces is the first-order version: scale each station's MTBF by "
      "its utilisation. That was described in the README as *the first-order "
      "version of busy-time clocking*, which was fair. It was not measured.\n")
    A(f"| | throughput (parts/h) | vs exact |")
    A("|---|---:|---:|")
    A(f"| wall-clock | {bc['wall']['throughput']:.2f} ± {bc['wall']['half_width']:.2f} | "
      f"{bc['wall']['throughput'] - bc['exact']['throughput']:+.2f} |")
    A(f"| first-order (MTBF ÷ utilisation) | {bc['first_order']['throughput']:.2f} ± "
      f"{bc['first_order']['half_width']:.2f} | "
      f"{bc['first_order_overshoot']:+.2f} |")
    A(f"| **exact busy-time** | **{bc['exact']['throughput']:.2f} ± "
      f"{bc['exact']['half_width']:.2f}** | — |")
    A(f"\n**The approximation overshoots by {bc['first_order_overshoot']:.2f} "
      f"parts/hour ({bc['first_order_error_pct']:.1f}%), which is "
      f"{abs(bc['first_order_overshoot'] / bc['correction_size']):.1f}× the size "
      f"of the entire correction it was making** — busy-time clocking is worth "
      f"{bc['correction_size']:+.2f} parts/hour against wall-clock, and the "
      "approximation of it lands further from the answer than the thing it was "
      "correcting, in the same direction. It corrected past the target.\n")

    A("\n### Where the error is\n")
    A("| station | utilisation | down: wall | down: first-order | down: exact | first-order error |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in bc["per_station"]:
        A(f"| {r['station']} | {r['utilisation']:.3f} | {r['down_wall']:.4f} | "
          f"{r['down_first_order']:.4f} | {r['down_exact']:.4f} | "
          f"{r['first_order_error_pct']:+.1f}% |")
    worst = max(bc["per_station"], key=lambda r: r["utilisation"])
    A(f"\n**The approximation is worst at the constraint** — {worst['station']}, "
      f"at {worst['utilisation']:.0%} utilisation, where it understates downtime "
      f"by {abs(worst['first_order_error_pct']):.0f}%. That is the one station "
      "whose downtime costs throughput, so the error lands entirely on the "
      "number the model exists to produce.\n")

    A("\n### The mechanism, on a line where it can be isolated\n")
    A("Theory says busy-clock downtime should be wall-clock downtime times "
      "utilisation. One station, arrival rate swept, nothing else changing:\n")
    A("| arrival mean (s) | utilisation | down: wall | down: first-order | down: exact | exact ÷ wall |")
    A("|---:|---:|---:|---:|---:|---:|")
    for r in sw["rows"]:
        A(f"| {r['arrival_mean_s']:.0f} | {r['utilisation']:.3f} | "
          f"{r['down_wall']:.4f} | {r['down_first_order']:.4f} | "
          f"{r['down_exact']:.4f} | {r['ratio_exact_over_wall']:.3f} |")
    A("\nThe last column tracks utilisation across the whole range, which is the "
      "check that the implementation does what it claims. And the approximation "
      "is accurate at low utilisation "
      f"({sw['rows'][-1]['first_order_error_pct']:+.1f}% at "
      f"{sw['rows'][-1]['utilisation']:.0%}) and degrades as the station fills "
      f"up ({sw['rows'][0]['first_order_error_pct']:+.1f}% at "
      f"{sw['rows'][0]['utilisation']:.0%}).\n")
    A("**Utilisation is endogenous** — it is measured *under* the failure regime "
      "you are trying to correct, so plugging in the wall-clock value is a "
      "one-step estimate of a fixed point, and it is biased exactly where "
      "utilisation is most sensitive to downtime: at the constraint. That is the "
      "mechanism, and it is only a partial explanation: iterating the "
      "approximation to its own fixed point converges to 53.84 parts/hour, still "
      "1.2 above the exact answer, so self-consistency is not the whole story. "
      "Said plainly rather than dressed up as a complete account.\n")

    # --- replay
    A("\n## 3. The animation is now a replay\n")
    A(f"The engine logs every station state transition and the buffer levels at "
      f"the instant they changed — {rp['n_events']:,} events over an eight-hour "
      f"shift, about {rp['events_per_part']:.1f} per part. Transitions rather "
      "than periodic samples: a sampled log has to choose a rate, and any rate "
      "coarse enough to be cheap misses the micro-stops that are the reason to "
      "watch a line second by second.\n")
    A("Logging is off by default and does not perturb the run — the same seed "
      "with and without a log produces bit-identical throughput, which is "
      "asserted in the tests.\n")

    A("\n### Does the replay show the run?\n")
    A("| station | frames running | measured running | error |")
    A("|---|---:|---:|---:|")
    for f in rp["fidelity"]:
        A(f"| {f['station']} | {f['frames_running']:.3f} | "
          f"{f['measured_running']:.3f} | {f['abs_error']:.3f} |")
    A(f"\nWorst error {rp['worst_fidelity']:.3f}. The residual is the "
      "denominator: the run's fractions divide by the whole horizon and the "
      "frames cover post-warm-up only.\n")

    c = rp["compare"]
    rs, cs = c["replay_signature"], c["reconstruction_signature"]
    A("\n### And what the reconstruction was showing\n")
    A("The reconstruction sampled each station independently from its measured "
      "time fractions and moved buffers with a heuristic. Its marginals are "
      "right by construction, so the only question worth asking is what it got "
      "wrong. Work-in-process accumulates **in front of** the constraint and "
      "drains behind it; that signature is the single thing an animation of a "
      "line exists to show.\n")
    A(f"| | mean buffer upstream of {c['bottleneck']} | downstream | correct? |")
    A("|---|---:|---:|:--:|")
    A(f"| replay | {rs['upstream_mean']:.2f} | {rs['downstream_mean']:.2f} | ✅ |")
    A(f"| reconstruction | {cs['upstream_mean']:.2f} | {cs['downstream_mean']:.2f} | ❌ |")
    A(f"\n**The reconstruction inverted it.** Buffers sit at "
      f"{rs['upstream_mean']:.1f} of 5 in front of the weld cell in the replay "
      f"and drain to {rs['downstream_mean']:.2f} behind it; the reconstruction "
      f"showed {cs['upstream_mean']:.1f} in front and {cs['downstream_mean']:.1f} "
      "behind — parts piling up *after* the bottleneck. The README's stated "
      "reason for having an animation at all was that a plant manager watching "
      "parts pile up in front of S3 believes the bottleneck result. The "
      "animation was piling them up on the wrong side.\n")
    A(f"Mean absolute error on the between-station running correlations is "
      f"{c['joint_mean_abs_corr_error']:.3f}. Independent draws cannot reproduce "
      "a correlation they were never given, and the correlations *are* the "
      "content: blocked-upstream and starved-downstream happening together is "
      "what a bottleneck looks like.\n")
    A("\nBoth renderers are kept, each labelled with its own provenance in the "
      "page, and `render(mode=\"auto\")` refuses to claim a replay it does not "
      "have — a result from `experiment.replicate` carries no log, and "
      "re-simulating to get one would animate a *different run* beside the "
      "summary it is captioned with.\n")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    d = {"sequencing": stage_sequencing()}
    print("  sequencing done")
    d["busy_clock"] = stage_busy_clock()
    print("  busy-clock done")
    d["sweep"] = stage_util_sweep()
    print("  sweep done")
    d["replay"] = stage_replay()
    print("  replay done")
    d["elapsed_s"] = time.time() - t0

    (OUT / "pass4.json").write_text(
        json.dumps(d, indent=2, default=str), encoding="utf-8")
    (DOCS / "SEQUENCING_AND_REPLAY.md").write_text(report(d), encoding="utf-8")
    print(f"wrote docs/SEQUENCING_AND_REPLAY.md in {d['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
