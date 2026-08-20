# Model validity — the assumptions register

All models are wrong. This document is the operational version of that: what is
modelled, what is not, which conclusions are sensitive to which assumption, and
what would have to be measured before this model could be pointed at a real line.

A simulation without this document is a number generator that people trust.

---

## 1. What is modelled

| element | how | fidelity |
|---|---|---|
| 6 stations in series | SimPy processes | structural — a real line has parallel machines, rework loops, and merges |
| cycle times | lognormal, mean and CV per station | shape is an assumption; see §3.1 |
| finite buffers | SimPy `Store` with capacity | blocking and starvation emerge correctly |
| failures | exponential MTBF, exponential MTTR | see §3.2 |
| release policy | saturated push, or CONWIP | no real order release, no due dates |
| product mix | **not modelled** | single product |
| changeovers | **not modelled** | see §2 |
| operators | **not modelled** | stations are always staffed |
| quality | **not modelled** | nothing is scrapped or reworked |

## 2. What is NOT modelled, and which way each one biases the answer

Every omission here makes the simulated line **better** than a real one. That is
the direction to be honest about: an optimistic model that is not labelled
optimistic gets used to justify a capital decision.

| missing | effect on simulated throughput | why it matters |
|---|---|---|
| changeovers / setup | **too high** | a mix of products with sequence-dependent setups can cost 5–15% of capacity |
| operator availability, breaks, shift handover | **too high** | stations here never wait for a person |
| quality loops (scrap, rework) | **too high** | a rework loop re-consumes constraint capacity, which is the worst place to spend it |
| material starvation from upstream supply | **too high** | station 1 is never short of raw material |
| micro-stops below the failure model's resolution | **too high** | the MTBF/MTTR model captures breakdowns, not the 30-second jams that DATA-1 shows are comparable in total |
| scheduling chaos, hot orders, expedites | **too high** | the model runs one product at a steady rate forever |
| tool wear / degradation | **too high** | cycle times are stationary |

**The 82%-of-theory question.** This model achieves ~93% of its constraint's
effective capacity. A real line commonly achieves 65–70% of theoretical. The
missing points are the table above, in roughly that order — and if a real line and
this model disagree by 20 points, the answer is almost never "the simulation is
wrong about queueing" and almost always "the simulation is missing a loss
category".

## 3. Distributional assumptions, and their sensitivity

### 3.1 Cycle time: lognormal

Chosen because a cycle time cannot be negative and a normal with CV 0.25 puts mass
below zero often enough to matter. Constants would be worse: constant cycle times
remove exactly the variability that buffers exist to absorb, and a line model with
constant times reports that buffers are worthless.

**Sensitivity: high, and untested here.** Queueing behaviour depends on the second
moment of the service distribution, and beyond that on its tail. The buffer-
allocation results in RESULTS.md §4 would move under a heavier-tailed cycle time,
and *no sensitivity analysis over the distribution family was run*. That is a gap,
not a defensible simplification.

### 3.2 Failures: exponential MTBF and MTTR

Exponential MTBF means a constant hazard rate — a machine that has run for 8 hours
is exactly as likely to fail in the next minute as one that just started. That is
wrong for wear-out failures (which is what ML-1 and ML-3 in this portfolio are
about) and roughly right for random electrical and control faults.

Exponential MTTR is worse. Real repair times are strongly bimodal: a reset takes
90 seconds, a bearing change takes six hours. An exponential distribution with the
same mean under-represents both ends.

**Sensitivity: high for the MTTR-reduction scenario.** "MTTR halved at the
constraint" is a headline result and it assumes halving is uniform across the
distribution. If the real distribution is bimodal, halving the mean by attacking
the long tail is a different project — and a different budget — from halving it by
attacking the short resets.

### 3.3 Failures occur in wall-clock time, not in operating time

`_failures()` runs on a free-running timer, so a station can "fail" while it is
starved or blocked. For a machine whose wear is driven by cycles, failure should be
clocked on busy time. **This biases availability down slightly**, and more so for
low-utilisation stations. It is not corrected.

## 4. Statistical method — what IS defensible

These are the parts that would survive review:

- **M/M/1 validation across four utilisation levels** with L and W inside the
  confidence interval at 3 of 4 (see §5 for the fourth).
- **Little's Law checked on every run**, 510 runs, 1 failure at a 5% tolerance,
  with the residual's convergence with horizon length demonstrated — which is what
  shows the residual is a finite-window boundary effect and not an entity leak.
- **Warm-up truncation by MSER-5** on the replication-averaged WIP series, rather
  than by eye.
- **30 replications with confidence intervals on every reported metric**, and the
  replication count required for a target precision computed rather than asserted.
- **Common random numbers with a measured 45× variance reduction** on the
  difference, made possible by dedicating an independent random stream to each
  station and each purpose.

## 5. Known discrepancies, unresolved

**M/M/1 at ρ = 0.5: W measures 122.2 ± 1.7 s against a theoretical 120.0.** L is
inside its interval; W is not, by about 1.3 half-widths. The bias is positive and
small, and it shrinks at higher ρ where the intervals are wider — which is the
signature of a small systematic effect being resolved by a tight interval rather
than of a large error.

The most likely cause is the same finite-horizon boundary that shows up in Little's
Law: W is averaged over parts that *completed* in the window, which over-weights
parts that entered before it. **It is not chased down.** Reporting it as 3/4 rather
than rounding it to "validated" is the honest handling.

## 6. Calibration guidance — pointing this at a real line

Per parameter, what would have to be measured and how:

| parameter | source | difficulty |
|---|---|---|
| mean cycle time per station per product | MES cycle records or a time study | easy if an MES exists — see SE-2 |
| cycle-time **distribution** | the same records, but you need the raw values, not the average | **most plants only keep the average** — this is usually the blocker |
| MTBF | maintenance work orders, filtered to genuine functional failures | hard; see ML-1's DEPLOYMENT_REALITY §5 on work-order label noise |
| MTTR distribution | work-order open/close timestamps | timestamps are when somebody typed, not when the machine stopped |
| buffer capacities | walk the line and count | easy, and frequently different from the drawing |
| blocking/starvation fractions | machine state data | this is exactly what DATA-1's OEE platform produces |

**The validation protocol before anybody spends money:** run the model against a
historical period, compare simulated throughput, WIP, and — most importantly — the
*blocking and starvation fractions per station* against measured ones. Throughput
alone is a weak test: a model can hit the right throughput with entirely the wrong
internal dynamics, and then get the buffer recommendation backwards.

## 7. What to say when the manager says "buy the conveyor"

RESULTS.md §4 shows buffer increases buying throughput with diminishing returns.
Before a purchase order:

1. **The confidence interval.** The difference between buffer 10 and buffer 20 is
   1.5 parts/h ± 1.2. That is barely distinguishable from noise at 30 replications
   and the CI is in the table.
2. **The WIP column is the price.** By Little's Law, buffer space bought as
   throughput is also bought as inventory and as cycle time — those are the same
   purchase. Buffer 20 nearly doubles WIP relative to buffer 5. A throughput-only
   recommendation is selling half a transaction.
3. **§2 and §3.** Every omitted loss category biases throughput up, and the
   cycle-time distribution — the assumption the buffer result is most sensitive to
   — is unvalidated.
4. **Calibration status: none.** This model has never been compared against a real
   line.

The model informs, the register bounds, the pilot verifies.
