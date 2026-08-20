# SE-3 — Production Line Digital Twin & What-If Simulator

**Status: ~50% slice.** The engine, the validation suite, the experiment
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

## What is NOT built (the other 50%)

1. **No animation.** The spec asks for a live view of parts, buffers, and
   blocking/starving states colour-coded, and is explicit that trust in
   simulations is built visually. There is none — this project serves the
   analytical audience only, which is exactly the failure mode the spec's red
   flags list ("analysis without animation").
2. **No dashboards.** No utilisation charts, no WIP traces, no cycle-time
   distributions plotted. Everything is a markdown table.
3. **The validation suite is not in CI.** It runs as stage 1 of `run_twin.py` and
   nothing fails a build. Little's Law violations are counted and reported, not
   raised.
4. **No product mix, changeovers, operators, or quality loops.** Every one of
   these biases simulated throughput *upward*; MODEL_VALIDITY §2 has the table.
5. **No sensitivity analysis over the cycle-time distribution family**, which is
   the assumption the buffer result is most sensitive to.
6. **No recommendation memo.** The spec asks for one investment question answered
   with simulated evidence, CIs, and stated limitations. The ingredients are in
   RESULTS.md §3–4 and MODEL_VALIDITY §7; the memo itself is not written.
7. **Failures are clocked on wall time, not busy time**, which slightly
   under-states availability for low-utilisation stations. Known, uncorrected.
8. **Never calibrated against a real line.** Not once.

## Layout

```
src/line.py        model definition + SimPy engine, per-source random streams
src/experiment.py  MSER-5 warm-up, replications, CIs, CRN variance measurement
src/validation.py  M/M/1 closed form, Little's Law monitor, bottleneck ceiling
run_twin.py        validation then scenarios; writes docs/RESULTS.md
docs/MODEL_VALIDITY.md   the assumptions register
```
