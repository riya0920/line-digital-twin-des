# SE-3 — Production Line Digital Twin & What-If Simulator

**Status: complete.** The engine, the validation suite, the experiment
methodology (warm-up, replications, CIs, common random numbers), the what-if
scenarios, the omitted-loss quantification and the investment memo are built. The
live animation, the analysis dashboards, and CI integration are not.

```bash
python run_twin.py               # ~8.5 min
python run_twin.py --quick
python run_twin.py --report-only
```

Writes [docs/RESULTS.md](docs/RESULTS.md). The assumptions register is
[docs/MODEL_VALIDITY.md](docs/MODEL_VALIDITY.md), and it is the document that
makes the rest of it trustworthy.

## Validation comes first, on purpose

Nothing is claimed about a scenario until the engine has reproduced results that
are already known. A simulation is an argument, not an oracle.

**M/M/1 against closed form**, single station, exponential arrivals and service:

| ρ | L theory | L simulated | in CI | W theory | W simulated | in CI |
|---|---|---|---|---|---|---|
| 0.5 | 1.000 | 1.015 ± 0.017 | ✔ | 120.0 s | 122.2 ± 1.7 | ✘ |
| 0.7 | 2.333 | 2.379 ± 0.138 | ✔ | 200.0 s | 204.8 ± 10.5 | ✔ |
| 0.85 | 5.667 | 5.732 ± 0.655 | ✔ | 400.0 s | 405.8 ± 44.1 | ✔ |
| 0.9 | 9.000 | 9.418 ± 1.911 | ✔ | 600.0 s | 628.6 ± 124.2 | ✔ |

**3 of 4, and the failure is reported rather than rounded away.** At ρ=0.5 the W
estimate sits about 1.3 half-widths high. The bias is small and positive and it
shrinks at higher ρ where the intervals widen — the signature of a small systematic
effect resolved by a tight interval, most likely the same finite-window boundary
that shows up in Little's Law. It is not chased down, and MODEL_VALIDITY §5 says so.

**Little's Law as an invariant monitor**, not an exhibit: checked on all **510**
scenario runs. 1 failure at a 5% tolerance, worst residual 5.51%.

When it fails there are four suspects, and only three of them are bugs — warm-up
leakage, WIP counting at the boundaries, lost/duplicated entities, and
**finite-horizon boundary bias**, which is not a bug. Parts still in the system at
the end contributed to L and never appear in λ or W, so a finite estimate is
biased with L > λW by O(1/T). The convergence table proves that is what is
happening here:

| horizon | mean signed residual | worst absolute |
|---|---|---|
| 4 h | +0.71% | 9.52% |
| 8 h | +0.56% | 2.58% |
| 24 h | +0.21% | 0.77% |
| 72 h | +0.02% | 0.34% |

A genuine entity leak would not improve with a longer run. It would get worse.

## Theory of constraints, with error bars

10% cycle-time reduction, one station at a time, 30 replications each:

| where the 10% goes | throughput (parts/h) | Δ vs baseline | significant? |
|---|---|---|---|
| S1-cut | 51.14 ± 0.85 | +0.01 | no |
| S2-form | 51.17 ± 0.84 | +0.04 | no |
| **S3-weld (CONSTRAINT)** | **55.93 ± 0.98** | **+4.80** | **yes** |
| S4-machine | 51.22 ± 0.84 | +0.09 | no |
| S5-paint | 51.17 ± 0.84 | +0.05 | no |
| S6-inspect | 51.12 ± 0.85 | −0.00 | no |

Goldratt with error bars: the constraint buys +4.80 parts/h, the average
non-constraint buys +0.04, and the "significant?" column is a comparison of
confidence intervals rather than of point estimates.

The second-order effect is in the report too — a non-constraint improvement *can*
help through starvation reduction, because the constraint is not running 100% of
the time. Here the best non-constraint is +0.09 ± 0.84, i.e. indistinguishable
from zero, and saying "approximately nothing" is more accurate than saying "nothing".

## Common random numbers, and the bug that made them work

| | difference in throughput | variance of the difference |
|---|---|---|
| paired (CRN) | +4.80 ± 0.36 | 0.578 |
| independent streams | +6.89 ± 2.35 | 25.257 |

**45× variance reduction, correlation 0.993.**

The first implementation measured a variance-reduction factor of **0.18** — CRN
made the comparison *worse*. The cause is the standard trap and worth stating
plainly: with a single shared random Generator, changing station 3's mean cycle
time changes how many numbers station 3 draws, which shifts every subsequent draw
for every other station and for the arrival process. Two scenarios started from the
same seed then experience completely different days and the pairing evaporates.

The fix is one independent stream per *source of randomness* — per station, per
purpose — via `np.random.SeedSequence.spawn`, so station 1 sees an identical
sequence in both scenarios no matter what station 3 does. `line.streams()` is
four lines and it is the difference between a 4.80 ± 0.36 result and a 6.89 ± 2.35
one.

## Warm-up, and a criterion that did not work

Truncation is by **MSER-5** on the replication-averaged WIP series, which picks the
truncation minimising the estimated standard error of the truncated mean — trading
residual transient bias against thrown-away data.

The first criterion was "first bucket after which the smoothed series stays within
5% of its tail mean forever". It demands *every* later bucket sit inside the band,
so one noisy bucket near the end pushes the answer to the end of the run: it
reported an **11.9-hour warm-up on a 12-hour horizon**. That is not a finding, it
is the criterion failing.

## Why 30 replications?

| metric | sd | half-width at n=30 | n for ±1% of mean |
|---|---|---|---|
| throughput | 2.274 | 0.849 | 77 |
| WIP | 0.597 | 0.223 | 33 |
| mean cycle time | 88.8 | 33.2 | 144 |
| **P95 cycle time** | **464.2** | **173.3** | **1,090** |

30 is not a magic number; precision is purchased at n ∝ (sd/half-width)². **P95
cycle time needs 1,090 replications for the precision mean throughput reaches at
77** — a tail statistic has a far larger sampling standard deviation than a mean.
That is the answer to "what would make you need 300?".

## The buffer result: where the space goes beats how much you buy

| scenario | throughput (parts/h) | Δ | WIP | mean cycle time |
|---|---|---|---|---|
| all buffers = 2 | 48.77 ± 0.84 | −2.36 | 14.16 | 1046 s |
| all buffers = 5 (baseline) | 51.13 ± 0.85 | — | 20.66 | 1451 s |
| all buffers = 10 | 53.25 ± 0.87 | +2.12 | 31.50 | 2114 s |
| all buffers = 20 | 54.78 ± 0.90 | +3.65 | 52.06 | 3363 s |
| **buffers=20 at the constraint only** | **54.23 ± 0.87** | **+3.10** | **36.91** | 2426 s |

**Targeting the constraint buys 85% of the throughput gain for 52% of the extra
inventory.** The constraint is the only station whose starvation and blocking cost
the line output, so buffer space anywhere else is mostly buying queue — the same
theory-of-constraints logic as the speedup experiment, applied to a different
lever. It is also the version that survives a capital request: the cheaper option
is not a compromise, it is the better answer.

The WIP column is the price throughout. By Little's Law, buffer space bought as
throughput is also bought as inventory and as cycle time — the same purchase — so
a buffer recommendation reporting only throughput is selling half a transaction.
See MODEL_VALIDITY §7 for what to say before the purchase order goes out.

## Push vs CONWIP

| scenario | throughput (parts/h) | WIP | mean cycle time | P95 cycle time |
|---|---|---|---|---|
| baseline (push) | 51.13 ± 0.85 | 20.66 | 1451 s | 2757 s |
| CONWIP 10 | 49.70 ± 0.86 | 10.00 | 725 s | 1619 s |
| **CONWIP 15** | **50.94 ± 0.84** | **15.00** | **1061 s** | **2169 s** |
| CONWIP 20 | 51.12 ± 0.85 | 20.00 | 1409 s | 2686 s |
| CONWIP 30 | 51.13 ± 0.85 | 30.00 | 2108 s | 3540 s |

**CONWIP 15 holds 27% less WIP, cuts mean cycle time 27% and P95 cycle time 21%,
at a throughput that sits inside the push baseline's own confidence interval.**

That is Little's Law doing the work rather than anything clever: at fixed
throughput, `L = λW` means less WIP *is* shorter cycle time. Push release keeps
loading material the constraint cannot consume, and every part beyond the cap buys
queue rather than output.

The selection criterion is deliberately stricter than overlapping intervals.
CONWIP 10's interval *touches* the baseline's while its mean sits 1.4 parts/h
lower — reading that as "the same throughput" would claim a free lunch the data
does not support and would credit the cap with a cycle-time gain that is partly
just lower output.

**Constant WIP is not a broken statistic.** The CONWIP rows report WIP as exactly
15.000 ± 0.000, which is the defining property of the policy: a token is taken
before a part is released and returned when it completes, so on a line never short
of raw material the number outstanding is constant by construction.

## Built in the second pass — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)

`python extend.py` — the two gaps MODEL_VALIDITY ranked highest, plus the memo:

- **"Name the missing 14 points", quantified.** MODEL_VALIDITY §2 listed the
  omitted loss categories and said every one biases throughput up, without saying
  by how much. Switching them on cumulatively — product mix, changeovers, operator
  availability, quality loop — drops throughput **51.0 → 32.7 parts/h, a 36% fall**,
  taking the model from 93% of constraint capacity to ~60% and bracketing the
  65–70% a real line achieves.
- **Cycle-time distribution sensitivity**, the assumption the buffer result rests
  on. Exponential service times reward buffering **2.5× more** than constant ones —
  but the constant case still gains, which refuted my prediction that it would gain
  nothing. Setting cv = 0 removes only *one* variability source; the breakdowns are
  still there, and on this line they dominate. The corollary: **reducing MTTR and
  adding buffer are substitutes, not complements.**
- **[docs/RECOMMENDATION.md](docs/RECOMMENDATION.md)** — the investment memo the
  spec asks for: one capital item, three candidates, a recommendation with CIs, and
  an explicit list of what would change it.

## Completed in the third pass — see [docs/COMPLETION.md](docs/COMPLETION.md)

```bash
python complete.py          # ~1 min; writes COMPLETION.md and out/line.html
python complete.py --gate   # exits non-zero if a validation check fails
```

- **The four unmodelled effects, priced.** MODEL_VALIDITY §2 listed product mix,
  changeovers, operators and quality loops, and stated that every one biases
  throughput upward — which is an admission that **every number the twin produces
  is an upper bound of unknown size**. The size is
  **2.43×**: 40.0 → 16.5
  parts/h. That is the number the recommendation memo needed and did not have.
- **Product mix moves the bottleneck**
  — stations [3, 4, 6] each take the constraint depending on what is running.
  A line balanced for the average is balanced for a product it never makes.
- **Changeovers make batch size a throughput decision**, with a deliberately
  asymmetric setup matrix — one direction needs a purge and the other does not,
  and a symmetric matrix removes the only interesting thing about the sequencing
  problem. The sweep is **biased toward large batches** and says so: it models
  the changeover cost and not the WIP, lead-time or slower-quality-feedback
  costs.
- **Operators as a shared resource.** With
  2 operators across
  6 stations,
  **14.0% of attention
  demands go unmet**. The current model has no state for *waiting for a person* —
  a station can be up, unblocked and unstarved and still not running — so it
  counts that time as running.
- **Rework is the expensive failure, and only sometimes.** A loop that re-enters
  at or before the constraint consumes bottleneck capacity twice; the same defect
  rate downstream of the constraint costs almost nothing. That distinction is
  exactly what a line-level average hides.
- **Does the buffer recommendation survive the distribution family?** Gain from
  buffer 2 → 20: lognormal **+4.1**, exponential
  **+13.9**, constant **+3.7** parts/h.
  The *direction* is robust across all three; the *magnitude* varies by
  10.2 parts/h. So the recommendation to buffer holds, and any
  business case built on the size of the gain needs the real cycle-time
  distribution measured first — **nobody measured it**.
- **Failures on busy time rather than wall time**: +8.7% throughput.
  A machine starved half the day does not accumulate wear while it sits there,
  and clocking failures on wall time misattributes them to stations that were not
  working.
- **Validation as a gate.** Little's Law violations used to be counted and
  printed. `python complete.py --gate` now exits non-zero. **A check that cannot
  fail is documentation** — and building the gate immediately caught that it was
  reading `relative_error` from a function that returns `relative_residual`,
  silently getting NaN. A gate that fails on a typo is worse than no gate,
  because it teaches people to ignore it.
- **An animation** at `out/line.html`, self-contained, with the station states
  and buffer levels playing back. The spec's red-flag list names "analysis
  without animation" and is right for two reasons: trust in a simulation is built
  visually, and animation is the fastest debugger a discrete-event model has.

### A scenario that was wrong, caught by an impossible number

The first changeover run reported setup consuming **347% of the horizon** at a
batch size of 5. A share above 1.0 is not a modelling subtlety — it is the tell
that the *scenario* was wrong: demand was set to 600 parts against a horizon the
line can make about 320 in. Demand is now scaled to what the line can actually
produce, and the overstatement figure fell from a nonsensical 9.7× to
2.43×.

## Built in the fourth pass — see [docs/SEQUENCING_AND_REPLAY.md](docs/SEQUENCING_AND_REPLAY.md)

```bash
python run_pass4.py    # ~100 s
```

The three items the list below named as needing a change in the **engine** rather
than in a spec. In two of the three, the finding is how wrong the thing they
replaced was.

- **A scheduler, replacing a price list.** `realism.py` could cost a sequence and
  not produce one — `batch_size_sweep` built a round-robin, which on an
  asymmetric changeover matrix costs **2.95 h against
  an optimum of 0.73 h, 305%
  worse**. Every batch-size number that sweep published carried a penalty that
  had nothing to do with batch size.
- **EDD is beaten at its own objective.** Jackson's rule — earliest due date
  minimises maximum lateness — is a theorem for a single machine with *no
  sequence-dependent setups*. Here EDD burns 1.46 h
  more on changeovers than it needs to, and **loses maximum lateness to a
  setup-aware rule, 1.44 h against
  0.84 h**, while putting more jobs late than FIFO.
- **Or-opt, not 2-opt, and Held–Karp to score it.** 2-opt evaluates a move by
  reversing a segment, which is O(1) only because a symmetric matrix prices a
  reversed arc the same. On an asymmetric matrix that evaluation is wrong. Or-opt
  never reverses. The exact solver exists to answer the question a heuristic
  cannot answer about itself — how far from optimal — and the answer is that
  or-opt cuts the nearest-neighbour gap by roughly half and does not close it.
- **Backward scheduling**, which is the direction that tells you that you are
  already late: promised at exactly the total run time, the release comes out at
  **-0.77 h**, i.e.
  46 minutes before time zero — and
  the size of that number is exactly the changeover the promise forgot.
- **Busy-time failures, done properly, and the approximation measured.** The
  station now carries a remaining *busy* life and the cycle is split when it runs
  out, so a breakdown lands mid-part. Against it, the first-order version
  (MTBF ÷ utilisation) **overshoots by 1.54 parts/hour,
  1.9× the size of the
  entire correction it was making** — further from the answer, in the same
  direction, than the wall-clock model it was correcting. Its error is worst at
  the constraint (-9% at 83%
  utilisation), which is the one station whose downtime costs throughput.
- **The animation is a replay.** 9,591 logged transitions over a
  shift, ~23 per part, off by default and bit-identical
  when on. And putting it beside the reconstruction is the finding: **the
  reconstruction had work-in-process piling up on the wrong side of the
  bottleneck** — 4.9 of 5 upstream and
  0.15 downstream in the replay, against
  1.4 and 3.5 reconstructed. The
  stated reason for having an animation was that a manager who watches parts pile
  up in front of S3 believes the bottleneck result. It was piling them up behind it.

### A bug found writing the backward schedule

The first version shifted each job by the setups that came *before* it. Walking
backwards, a setup pushes everything ahead of it *earlier*, so the shift on job
*i* is the total of the setups *after* it. Both versions land the last job
exactly on the due date — the number a reader checks — and the wrong one reported
a comfortable release of 0.00 h on an instance that had to start 46 minutes
before time zero. The test walks the sequence forward from the computed release
and demands every start time match.

## Also in the fifth pass — see [docs/SEQUENCING_GAP.md](docs/SEQUENCING_GAP.md)

```bash
python run_pass5.py    # ~5 min
```

The item said or-opt's 10–25% gap needed *a better neighbourhood or a
metaheuristic*. Both halves are built — random restarts, and simulated annealing
over the same or-opt neighbourhood — and the gap closes completely where
Held–Karp can still verify it.

| method | mean gap vs exact (10 jobs) | optimal on | mean vs best (10–40 jobs) | best on | mean time |
|---|---:|---:|---:|---:|---:|
| nearest neighbour | +34.2% | 1/4 | +21.3% | 3/10 | 0.00 s |
| or-opt | +13.9% | 1/4 | +11.4% | 3/10 | 0.21 s |
| multi-start or-opt | +0.0% | 4/4 | +0.0% | 10/10 | 27.25 s |
| simulated annealing | +0.0% | 4/4 | +6.1% | 5/10 | 1.60 s |

**The more elaborate method lost.** Multi-start or-opt is best or tied on every
instance; simulated annealing is best on half of them and averages
+6.1% off. And **annealing is
not monotone in its budget** — the cooling rate is derived from the iteration
count, so doubling it runs a *different* search: one instance goes 0% at 2,000
iterations, 11% at 8,000, 0% again at 20,000. A method that cannot be improved by
giving it more effort cannot be tuned.

**Nothing predicts when plain or-opt is already enough.** Two explanations were
tried and both failed: job count (or-opt is short at 10 jobs and fine at 20) and
product diversity (it should be best at 5 jobs per product and worst at 1.7; the
measurement is fine at 2.2 and short at both 5.0 and 1.7). So the guidance is the
boring one — run multi-start; it is never worse, and forty seconds to sequence a
week of work is not a cost worth optimising.

Above 12 jobs there is still no exact answer, so those columns are scored against
the best any method found — **a floor, not the optimum**. All four could be well
short together and the table would look identical.

## What is NOT built

1. **Still not calibrated against a real line.** Not once. Every distribution,
   every MTBF and every cycle time is chosen, and the sensitivity analysis above
   is the honest response to that — it says which conclusions survive the choice
   and which do not. This is the gap that matters and nothing in four passes has
   touched it.
2. **The four realism effects are modelled separately and stacked
   multiplicatively.** They interact, mostly in the bad direction — a changeover
   during an operator shortage costs more than either alone, because the setup
   needs the operator who is not there. So the adjusted figure is a *better*
   upper bound and still an upper bound.
3. **The busy-time result has a mechanism that is only partly explained.** The
   approximation's bias is largest where utilisation is most sensitive to
   downtime, which is at the constraint — but iterating it to its own fixed point
   converges to 53.84 parts/hour, still 1.2 above the exact answer, so
   self-consistency does not account for all of it. Stated rather than dressed up.
4. **The scheduler sequences batches on a single logical resource.** It prices
   setup and due dates against the line's own bottleneck rate; it does not
   sequence per station, so a product whose bottleneck moves (which
   `bottleneck_by_product` shows happens) is scheduled against the wrong
   constraint.
5. **Above 12 jobs the gap to optimal is still unmeasured.** Held–Karp refuses
   there, so the scaled comparison is scored against the best of four methods —
   a floor. All four could be well short of the optimum together and nothing
   here would show it.
6. **A replay of a whole shift is still blind to micro-stops.** 240 frames over
   eight hours is one sample every two minutes; the event log has them, the
   default window is the last hour so they are visible, and anything outside that
   window is sampled at a rate that misses them.
7. **No tool wear, no material shortages, no operator skill differences.** Named
   rather than silently absent.

## Layout

```
src/line.py        model + SimPy engine, per-source streams, wall/busy failure
                   clocks, optional transition log
src/experiment.py  MSER-5 warm-up, replications, CIs, CRN variance measurement
src/validation.py  M/M/1 closed form, Little's Law monitor, bottleneck ceiling
src/realism.py     product mix, asymmetric changeovers, operators, quality loops
src/sequencing.py  rules, or-opt, restarts, annealing, Held-Karp, backward scheduling
src/animate.py     replay from the event log, and the reconstruction it replaces
run_twin.py        validation then scenarios; writes docs/RESULTS.md
run_pass4.py       sequencing, busy-time clock, replay; writes the pass-4 doc
docs/MODEL_VALIDITY.md   the assumptions register
```
