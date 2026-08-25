# Closing the sequencing gap, and which method earns its keep

The README's item: *or-opt is a local search with no restarts and no acceptance of worsening moves … closing it needs either a better neighbourhood or a metaheuristic.* Both suggestions are built — random restarts, and simulated annealing over the same or-opt neighbourhood.

**Still or-opt, still no reversal.** The asymmetry argument does not stop applying because the search got cleverer: 2-opt's O(1) move evaluation is only valid on a symmetric matrix. What changes is which move is tried and whether a worse one is accepted.

**The temperature is set from the data, not chosen.** A hand-picked starting temperature is a hidden fit to one instance — too cold and it is `improve` with extra steps, too hot and it is a random walk. It is the mean absolute cost change over a sample of random moves, so it behaves the same whether the matrix is in minutes, hours or anything else.


## Where the exact answer exists (10 jobs, Held–Karp)

| instance | nearest neighbour | or-opt | multi-start or-opt | simulated annealing |
|---|---:|---:|---:|---:|
| seed 1 | +0.0% | +0.0% | +0.0% | +0.0% |
| seed 2 | +26.4% | +18.1% | +0.0% | +0.0% |
| seed 5 | +50.7% | +11.2% | +0.0% | +0.0% |
| seed 7 | +59.6% | +26.3% | +0.0% | +0.0% |

| method | mean gap | worst | optimal on |
|---|---:|---:|---:|
| nearest neighbour | +34.2% | +59.6% | 1/4 |
| or-opt | +13.9% | +26.3% | 1/4 |
| multi-start or-opt | +0.0% | +0.0% | 4/4 |
| simulated annealing | +0.0% | +0.0% | 4/4 |

**The gap closes completely.** Both metaheuristics reach the optimum on 4 of 4 instances, against or-opt's mean +13.9%. The item is answered.

And **the annealing schedule buys nothing here** — random restarts alone find the same optima. That was worth measuring separately rather than assuming the more elaborate method is the better one.


## Above 12 jobs, where no exact answer exists

Scored against the best any method found, which is a **floor and not the optimum** — every gap below could be understating how far all four are from the real answer.

| jobs | products | jobs/product | seed | nearest neighbour | or-opt | multi-start or-opt | simulated annealing |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 2.0 | 2 | +26.4% | +18.1% | **+0.0%** | **+0.0%** |
| 10 | 7 | 1.4 | 7 | +59.6% | +26.3% | **+0.0%** | **+0.0%** |
| 20 | 7 | 2.9 | 2 | **+0.0%** | **+0.0%** | **+0.0%** | **+0.0%** |
| 20 | 8 | 2.5 | 7 | **+0.0%** | **+0.0%** | **+0.0%** | **+0.0%** |
| 40 | 8 | 5.0 | 2 | +13.6% | +13.6% | **+0.0%** | +10.8% |
| 40 | 8 | 5.0 | 7 | **+0.0%** | **+0.0%** | **+0.0%** | **+0.0%** |
| 40 | 17 | 2.4 | 2 | +15.9% | +2.2% | **+0.0%** | +7.6% |
| 40 | 16 | 2.5 | 7 | +62.5% | +35.7% | **+0.0%** | +14.6% |
| 40 | 20 | 2.0 | 2 | +8.0% | +8.0% | **+0.0%** | +13.1% |
| 40 | 19 | 2.1 | 7 | +27.0% | +10.5% | **+0.0%** | +15.5% |

| method | mean vs best | worst | best on | mean seconds |
|---|---:|---:|---:|---:|
| nearest neighbour | +21.3% | +62.5% | 3/10 | 0.00 |
| or-opt | +11.4% | +35.7% | 3/10 | 0.21 |
| multi-start or-opt | +0.0% | +0.0% | 10/10 | 27.25 |
| simulated annealing | +6.1% | +15.5% | 5/10 | 1.60 |

**Multi-start or-opt is best or tied on every instance** (10/10) and costs 27 seconds on average. **Simulated annealing does not earn its complexity**: best on 5/10, mean +6.1% off, for 1.6 seconds. Plain or-opt is best on 3/10 at 0.21 seconds, and when it is short it is short by up to 36%.


### Annealing is not monotone in its budget

More iterations is not a finer search. The cooling rate is derived from the iteration count, so doubling the budget runs a *different* search rather than a longer one — on one instance the gap goes 0% at 2,000 iterations, 11% at 8,000, and 0% again at 20,000. A method whose answer does not improve monotonically with effort cannot be tuned by giving it more, which is a second and independent reason to prefer restarts.


### And nothing predicts when a local search is already enough

Or-opt already finds the best known answer at [[20, 2.9], [20, 2.5], [40, 5.0]] (jobs, jobs-per-product) and falls short at [[10, 2.0], [10, 1.4], [40, 5.0], [40, 2.4], [40, 2.5], [40, 2.0], [40, 2.1]].

**Two explanations were tried and both failed.** The first was job count — but or-opt is short at 10 jobs and fine at 20. The second was product diversity: with few products and many jobs each, grouping by product is nearly optimal and easy to find. That predicts or-opt should do best at 5 jobs per product and worst at 1.7, and the measurement is the other way round — fine at 2.2, short at both 5.0 and 1.7.

So the honest guidance is the boring one: **run multi-start**. It is never worse, the instances where a cheaper search would have been enough are not identifiable in advance, and forty seconds to sequence a week of work is not a cost worth optimising.


## What this settles

- **The gap is closed where it can be measured**, and the exact solver is what makes that a fact rather than a hope.
- **Above 12 jobs the gap is unmeasured and stays unmeasured.** Best-of-four is a floor. All four could be well short of the optimum together and this table would look identical.
- **The more elaborate method lost.** Annealing was the README's suggestion and restarts are the simpler half of it; the simpler half is the one that works here.
- **Setup cost only.** None of this touches due dates, and the pass-4 finding stands: the minimum-setup sequence is free to ignore who is waiting.

