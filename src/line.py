"""The modelled line: SimPy stations with FINITE buffers.

Finite buffers are the whole point. Blocking (a station finishes but the
downstream buffer is full, so it cannot release the part and cannot start the
next) and starvation (the upstream buffer is empty) are the phenomena a line
simulation exists to study. Infinite buffers assume the problem away and turn the
model into a throughput calculator that always returns the bottleneck rate.

Separation of concerns is deliberate and assessable:

    line.py         model definition + engine
    experiment.py   replications, warm-up, CRN, confidence intervals
    validation.py   checks against results that are known analytically
    run_twin.py     the experiments and the report

The engine never computes a statistic; the experiment runner never defines a
station. Mixing the two is how a simulation ends up with its warm-up period baked
into a mean that nobody can find.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy


@dataclass
class StationSpec:
    name: str
    mean_cycle_s: float
    cv: float = 0.25              # coefficient of variation on cycle time
    mtbf_s: float | None = None   # None = never fails
    mttr_s: float | None = None
    buffer_after: int = 5         # capacity of the buffer DOWNSTREAM of this station
    dist: str = "lognormal"       # "lognormal" | "exponential" | "constant"


@dataclass
class LineSpec:
    stations: list[StationSpec]
    release: str = "push"         # "push" | "conwip"
    conwip_limit: int = 20
    arrival_mean_s: float | None = None  # push only; None = saturated (always material)
    arrival_cv: float = 1.0
    name: str = "line"


@dataclass
class RunResult:
    completed: int
    sim_time_s: float
    throughput_per_hour: float
    wip_time_avg: float
    cycle_times: np.ndarray
    utilisation: dict[str, float]
    blocked_frac: dict[str, float]
    starved_frac: dict[str, float]
    down_frac: dict[str, float]
    warmup_s: float
    entities_created: int
    entities_completed: int
    wip_area: float = 0.0
    observed_time: float = 0.0
    # (enter_time, exit_time) for every completed part, INCLUDING those completed
    # during warm-up. Welch's method needs the transient, so the engine keeps it
    # and the experiment layer decides what to exclude -- not the other way round.
    entry_exit: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


class _Station:
    def __init__(self, env, spec: StationSpec, rng, in_store, out_store, stats,
                 fail_rng=None):
        self.env = env
        self.spec = spec
        self.rng = rng
        self.fail_rng = fail_rng or rng
        self.in_store = in_store
        self.out_store = out_store
        self.stats = stats
        self.busy_s = 0.0
        self.blocked_s = 0.0
        self.starved_s = 0.0
        self.down_s = 0.0
        self.up = True
        if spec.mtbf_s:
            env.process(self._failures())
        env.process(self._run())

    def _cycle(self) -> float:
        """Lognormal cycle time with the requested mean and CV.

        Lognormal rather than normal because a cycle time cannot be negative and a
        normal with CV 0.25 puts mass below zero often enough to matter. Constants
        are worse still: constant cycle times remove exactly the variability that
        buffers exist to absorb, and a line model with constant times will tell you
        buffers are worthless.
        """
        m, cv = self.spec.mean_cycle_s, self.spec.cv
        if self.spec.dist == "constant" or cv <= 0:
            return m
        if self.spec.dist == "exponential":
            # Needed for the M/M/1 validation case: the analytic result is only
            # the analytic result if the service distribution is the one the
            # formula assumes. A lognormal with cv=1 is NOT exponential.
            return float(self.rng.exponential(m))
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(m) - sigma**2 / 2
        return float(self.rng.lognormal(mu, sigma))

    def _failures(self):
        while True:
            yield self.env.timeout(self.fail_rng.exponential(self.spec.mtbf_s))
            self.up = False
            t0 = self.env.now
            yield self.env.timeout(self.fail_rng.exponential(self.spec.mttr_s))
            self.down_s += self.env.now - t0
            self.up = True

    def _run(self):
        while True:
            t0 = self.env.now
            part = yield self.in_store.get()
            self.starved_s += self.env.now - t0

            while not self.up:
                yield self.env.timeout(1.0)

            t0 = self.env.now
            yield self.env.timeout(self._cycle())
            self.busy_s += self.env.now - t0

            t0 = self.env.now
            yield self.out_store.put(part)   # blocks if the buffer is full
            self.blocked_s += self.env.now - t0


def streams(seed: int, n_stations: int) -> dict:
    """One INDEPENDENT random stream per source of randomness.

    This is the piece that makes common random numbers actually work, and doing it
    the lazy way (one shared Generator for the whole model) is why the first
    version of this project measured a CRN variance-reduction factor of 0.18 --
    i.e. CRN made the comparison *worse*.

    With a single shared stream, changing station 3's mean cycle time changes how
    many numbers station 3 consumes, which shifts every subsequent draw for every
    other station and for the arrival process. The two scenarios then experience
    completely different "days" despite starting from the same seed, and the
    pairing that CRN depends on evaporates.

    With one stream per (station, purpose), station 1 sees the identical sequence
    of cycle times in both scenarios no matter what station 3 does. The scenarios
    become genuinely paired and the variance of the difference collapses.

    `np.random.SeedSequence.spawn` is the right tool: it produces statistically
    independent child streams from one parent seed, rather than the classic
    seed+1, seed+2 hack which gives no such guarantee.
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(2 * n_stations + 2)
    return {
        "arrivals": np.random.default_rng(children[0]),
        "cycle": [np.random.default_rng(children[1 + i]) for i in range(n_stations)],
        "failure": [np.random.default_rng(children[1 + n_stations + i])
                    for i in range(n_stations)],
    }


def simulate(spec: LineSpec, horizon_s: float, seed: int, warmup_s: float = 0.0,
             rng: np.random.Generator | None = None) -> RunResult:
    rstreams = streams(seed, len(spec.stations))
    rng = rstreams["arrivals"]
    env = simpy.Environment()

    n = len(spec.stations)
    stores = [simpy.Store(env, capacity=1_000_000)]  # infeed
    for st in spec.stations:
        stores.append(simpy.Store(env, capacity=max(1, st.buffer_after)))
    # The last store is the finished-goods sink and must not block the line.
    stores[-1] = simpy.Store(env, capacity=1_000_000)

    created = {"n": 0}
    completed: list[tuple[float, float]] = []   # (enter_time, exit_time)
    wip_area = {"v": 0.0, "last_t": 0.0, "obs": 0.0}
    conwip = simpy.Container(env, capacity=spec.conwip_limit,
                             init=spec.conwip_limit) if spec.release == "conwip" else None

    def _wip_now() -> int:
        return created["n"] - len(completed)

    def _accrue(t_now: float):
        if t_now >= warmup_s:
            start = max(wip_area["last_t"], warmup_s)
            if t_now > start:
                wip_area["v"] += _wip_now() * (t_now - start)
                wip_area["obs"] += t_now - start
        wip_area["last_t"] = t_now

    def releaser():
        while True:
            if conwip is not None:
                yield conwip.get(1)
            elif spec.arrival_mean_s:
                yield env.timeout(rng.exponential(spec.arrival_mean_s))
            else:
                # Saturated release: keep the infeed buffer stocked so station 1 is
                # never starved. This is the "material is always available" case.
                if len(stores[0].items) > 4:
                    yield env.timeout(1.0)
                    continue
            _accrue(env.now)
            created["n"] += 1
            yield stores[0].put({"t_in": env.now})

    def sink():
        while True:
            part = yield stores[-1].get()
            _accrue(env.now)
            completed.append((part["t_in"], env.now))
            if conwip is not None:
                yield conwip.put(1)

    stations = [
        _Station(env, sp, rstreams["cycle"][i], stores[i], stores[i + 1], None,
                 fail_rng=rstreams["failure"][i])
        for i, sp in enumerate(spec.stations)
    ]
    env.process(releaser())
    env.process(sink())
    env.run(until=horizon_s)
    _accrue(horizon_s)

    post = [(a, b) for a, b in completed if b >= warmup_s]
    obs_time = max(1e-9, horizon_s - warmup_s)
    cts = np.array([b - a for a, b in post], dtype=float)
    return RunResult(
        completed=len(post),
        sim_time_s=horizon_s,
        throughput_per_hour=len(post) / obs_time * 3600.0,
        wip_time_avg=wip_area["v"] / max(1e-9, wip_area["obs"]),
        cycle_times=cts,
        utilisation={s.spec.name: s.busy_s / horizon_s for s in stations},
        blocked_frac={s.spec.name: s.blocked_s / horizon_s for s in stations},
        starved_frac={s.spec.name: s.starved_s / horizon_s for s in stations},
        down_frac={s.spec.name: s.down_s / horizon_s for s in stations},
        warmup_s=warmup_s,
        entities_created=created["n"],
        entities_completed=len(completed),
        wip_area=wip_area["v"],
        observed_time=wip_area["obs"],
        entry_exit=np.array(completed, dtype=float).reshape(-1, 2),
    )


def default_line(name: str = "baseline") -> LineSpec:
    """Six stations, station 3 is the designed constraint.

    Cycle times are chosen so station 3 is unambiguously the bottleneck on
    capacity, and its MTBF/MTTR make its *effective* capacity lower still -- which
    is the number the theory-of-constraints experiment has to match.
    """
    return LineSpec(name=name, stations=[
        StationSpec("S1-cut", 42.0, cv=0.20, mtbf_s=7200, mttr_s=300, buffer_after=5),
        StationSpec("S2-form", 45.0, cv=0.25, mtbf_s=9000, mttr_s=420, buffer_after=5),
        StationSpec("S3-weld", 58.0, cv=0.30, mtbf_s=5400, mttr_s=600, buffer_after=5),
        StationSpec("S4-machine", 47.0, cv=0.22, mtbf_s=8000, mttr_s=360, buffer_after=5),
        StationSpec("S5-paint", 44.0, cv=0.35, mtbf_s=10800, mttr_s=900, buffer_after=5),
        StationSpec("S6-inspect", 40.0, cv=0.15, mtbf_s=None, mttr_s=None, buffer_after=5),
    ])


def effective_capacity_per_hour(st: StationSpec) -> float:
    """Availability-adjusted capacity. The ceiling the line cannot beat."""
    avail = 1.0
    if st.mtbf_s and st.mttr_s:
        avail = st.mtbf_s / (st.mtbf_s + st.mttr_s)
    return 3600.0 / st.mean_cycle_s * avail
