"""Rolling-horizon random-key re-planner (RH-RK), decision J7(b).

At every decision epoch this baseline re-optimizes a complete allocation of all free targets
to all operational UAVs with a plain crow search algorithm (paper Eq. 2) over the paper's
random-key encoding (§4.1), seeded with the paper's spatiotemporal-greedy and round-robin
heuristics (§4.3). The decision UAV then executes the first target of its planned sequence.

This is OUR baseline inspired by the paper. It is NOT MDCSATA: it optimizes a weighted sum
instead of a Pareto front and has none of the five MDCSATA strategies except the two
heuristic seeds. Committed tasks of other UAVs are locked, as in the paper's encoding.
The population is additionally seeded with the plan of the greedy dispatcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uavta.baselines.base import wait_or_single
from uavta.baselines.heuristics import GreedyValueRatePolicy
from uavta.dynamics.kinematics import distance
from uavta.dynamics.survival import segment_survival
from uavta.entities.uav import UAVStatus
from uavta.environment.env import MultiUAVDynamicEnv


@dataclass(frozen=True)
class RHRKConfig:
    """Search budget and CSA parameters (AP and fl follow the paper's Table 6)."""

    population: int = 40
    iterations: int = 40
    awareness_probability: float = 0.05
    flight_length: float = 1.5
    energy_penalty: float = 10.0


class _Problem:
    """Pre-computed data of one planning instant."""

    def __init__(self, env: MultiUAVDynamicEnv) -> None:
        sim = env.sim
        assert sim is not None
        self.t = sim.t
        self.horizon = env.config.horizon
        self.reserve = env.config.energy_reserve
        w = env.config.reward
        self.w_value, self.w_time = w.w_value, w.w_time
        self.value_ref = env.value_ref
        lam_obse, lam_risk = sim.scenario.lambda_obse, sim.scenario.lambda_risk
        self.lam_obse, self.lam_risk = lam_obse, lam_risk

        self.uav_ids = [k for k, u in enumerate(sim.uavs) if u.is_operational]
        # A UAV re-deciding in transit (retask_on_event) plans from where it is now, and its
        # own committed target is free to be (re)assigned like any other.
        dec = sim.decision_uav
        redecide = dec is not None and sim.uavs[dec].has_task
        self.slots = [
            tg.slot
            for tg in sim.targets
            if not tg.completed and (tg.locked_by is None or (redecide and tg.locked_by == dec))
        ]
        S, n = len(self.uav_ids), len(self.slots)
        self.S, self.n = S, n
        tgts = [sim.targets[s] for s in self.slots]
        self.value = [tg.value for tg in tgts]
        self.tobs = [tg.spec.observation_time for tg in tgts]
        self.tref = [tg.reference_time for tg in tgts]
        self.lam = [sim.decay_rate(s) for s in self.slots]  # per-target decay (extended env)
        # Critical targets (extended env): value × late_factor if the observation starts late.
        self.deadline = [tg.deadline_time for tg in tgts]
        self.late = [tg.spec.late_factor for tg in tgts]
        pos = [tg.position for tg in tgts]
        self.ready, self.energy, self.rate, self.speed = [], [], [], []
        self.surv0, self.eta, self.xi = [], [], []
        # eta[k][j]: sensing capability of UAV k for target j; capable[k][j] False if the
        # target requires a sensor UAV k lacks (extended environment). v1: eta[k][j] = ηₖ.
        self.capable: list[list[bool]] = []
        self.d0: list[list[float]] = []
        for k in self.uav_ids:
            u = sim.uavs[k]
            free_now = redecide and k == dec
            busy = 0.0 if free_now else sim.time_until_free(k)
            start = u.position if free_now else sim.free_position(k)
            s0 = u.survival
            if not free_now and u.status == UAVStatus.TRANSIT and u.task_slot is not None:
                rem = distance(u.position, sim.targets[u.task_slot].position)
                s0 *= segment_survival(rem, u.spec.vulnerability, lam_risk)
            self.ready.append(self.t + busy)
            self.energy.append(u.energy - u.spec.consumption_rate * busy)
            self.rate.append(u.spec.consumption_rate)
            self.speed.append(u.spec.speed)
            self.surv0.append(s0)
            self.eta.append([sim.eta_for(k, s) for s in self.slots])
            self.capable.append([sim.can_sense(k, s) for s in self.slots])
            self.xi.append(u.spec.vulnerability)
            self.d0.append([distance(start, p) for p in pos])
        self.dt = [[distance(a, b) for b in pos] for a in pos]

    def _timely(self, j: int, arrival: float) -> float:
        dl = self.deadline[j]
        return 1.0 if dl is None or arrival <= dl + 1e-9 else self.late[j]

    def decode(self, x: np.ndarray) -> list[list[int]]:
        plan: list[list[int]] = [[] for _ in range(self.S)]
        uav = np.minimum(x.astype(int), self.S - 1)
        order = np.argsort(x - uav, kind="stable")
        for j in order:
            plan[int(uav[j])].append(int(j))
        return plan

    def cost(self, plan: list[list[int]], energy_penalty: float) -> float:
        collected = 0.0
        makespan = self.t
        violation = 0.0
        for k, seq in enumerate(plan):
            clock = self.ready[k]
            surv = self.surv0[k]
            prev = -1
            for j in seq:
                d = self.d0[k][j] if prev < 0 else self.dt[prev][j]
                clock += d / self.speed[k]
                surv *= math.exp(-self.lam_risk * self.xi[k] * d)
                decay = math.exp(-self.lam[j] * max(clock - self.tref[j], 0.0))
                collected += self.value[j] * self._timely(j, clock) * self.eta[k][j] * surv * decay
                clock += self.tobs[j]
                prev = j
                if not self.capable[k][j]:
                    violation += 1.0  # the simulator would never allow this assignment
            makespan = max(makespan, clock)
            g = self.energy[k] - self.rate[k] * (clock - self.ready[k]) - self.reserve
            if seq and g <= 0:
                violation += -g + 1e-3
        return (
            -self.w_value * collected / self.value_ref
            + self.w_time * (makespan - self.t) / self.horizon
            + energy_penalty * violation
        )

    def encode(self, plan: list[list[int]]) -> np.ndarray:
        x = np.zeros(self.n)
        for k, seq in enumerate(plan):
            for pos, j in enumerate(seq):
                x[j] = k + (pos + 0.5) / (len(seq) + 1)
        return x

    def greedy_seed(self) -> list[list[int]]:
        """Nearest UAV per target, within-UAV order by descending V / max(d, 1) (paper §4.3)."""
        plan: list[list[int]] = [[] for _ in range(self.S)]
        for j in range(self.n):
            capable = [k for k in range(self.S) if self.capable[k][j]] or list(range(self.S))
            k = min(capable, key=lambda k: self.d0[k][j])
            plan[k].append(j)
        for k in range(self.S):
            plan[k].sort(key=lambda j: -self.value[j] / max(self.d0[k][j], 1.0))
        return plan

    def dispatch_greedy_seed(self) -> list[list[int]]:
        """Plan produced by simulating the greedy value-rate dispatcher in the plan model.

        Seeding with it guarantees the search starts no worse than greedy dispatch on the
        planning objective, so RH-RK is a strictly stronger comparator than greedy.
        """
        plan: list[list[int]] = [[] for _ in range(self.S)]
        clock = list(self.ready)
        surv = list(self.surv0)
        energy = list(self.energy)
        last = [-1] * self.S
        free = set(range(self.n))
        while free:
            k = min(range(self.S), key=lambda k: (clock[k], k))
            best, best_score = -1, -math.inf
            for j in free:
                if not self.capable[k][j]:
                    continue
                d = self.d0[k][j] if last[k] < 0 else self.dt[last[k]][j]
                dur = d / self.speed[k] + self.tobs[j]
                if energy[k] - self.rate[k] * dur <= self.reserve:
                    continue
                arr = clock[k] + d / self.speed[k]
                val = (
                    self.value[j]
                    * self.eta[k][j]
                    * surv[k]
                    * math.exp(-self.lam_risk * self.xi[k] * d)
                    * math.exp(-self.lam[j] * max(arr - self.tref[j], 0.0))
                    * self._timely(j, arr)
                )
                if val / dur > best_score:
                    best, best_score = j, val / dur
            if best < 0:
                clock[k] = math.inf  # this UAV can take nothing more
                if all(math.isinf(c) for c in clock):
                    break
                continue
            d = self.d0[k][best] if last[k] < 0 else self.dt[last[k]][best]
            dur = d / self.speed[k] + self.tobs[best]
            surv[k] *= math.exp(-self.lam_risk * self.xi[k] * d)
            energy[k] -= self.rate[k] * dur
            clock[k] += dur
            last[k] = best
            plan[k].append(best)
            free.remove(best)
        for j in sorted(free):  # infeasible leftovers keep the encoding complete (Eq. 9)
            plan[j % self.S].append(j)
        return plan

    def round_robin_seed(self) -> list[list[int]]:
        """Load-balanced round-robin over targets sorted by value (paper §4.3)."""
        plan: list[list[int]] = [[] for _ in range(self.S)]
        for r, j in enumerate(sorted(range(self.n), key=lambda j: -self.value[j])):
            plan[r % self.S].append(j)
        return plan


class RollingHorizonRandomKeyPolicy:
    """See module docstring."""

    name = "rh_rk"

    def __init__(self, config: RHRKConfig | None = None, seed: int | None = None) -> None:
        self.config = config or RHRKConfig()
        self._rng = np.random.default_rng(seed)
        self._fallback = GreedyValueRatePolicy()
        self.fallbacks = 0

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.fallbacks = 0

    def plan(self, env: MultiUAVDynamicEnv) -> tuple[_Problem, list[list[int]]]:
        """Optimize and return the problem and best plan (indices into ``problem.slots``)."""
        cfg, rng = self.config, self._rng
        prob = _Problem(env)
        S, n = prob.S, prob.n
        hi = S - 1e-9
        seeds = [
            prob.encode(prob.dispatch_greedy_seed()),
            prob.encode(prob.greedy_seed()),
            prob.encode(prob.round_robin_seed()),
        ]
        x = rng.uniform(0.0, hi, size=(cfg.population, n))
        for r, s in enumerate(seeds[: cfg.population]):
            x[r] = s
        fit = np.array([prob.cost(prob.decode(xi), cfg.energy_penalty) for xi in x])
        mem, mem_fit = x.copy(), fit.copy()
        for _ in range(cfg.iterations):
            for c in range(cfg.population):
                j = int(rng.integers(cfg.population))
                if rng.random() >= cfg.awareness_probability:
                    cand = x[c] + rng.random() * cfg.flight_length * (mem[j] - x[c])
                else:
                    cand = rng.uniform(0.0, hi, size=n)
                cand = np.clip(cand, 0.0, hi)
                f = prob.cost(prob.decode(cand), cfg.energy_penalty)
                x[c] = cand
                if f < mem_fit[c]:
                    mem[c], mem_fit[c] = cand, f
        best = mem[int(np.argmin(mem_fit))]
        return prob, prob.decode(best)

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        wait = wait_or_single(mask)
        if wait is not None:
            return wait
        sim = env.sim
        assert sim is not None and sim.decision_uav is not None
        prob, plan = self.plan(env)
        k = prob.uav_ids.index(sim.decision_uav)
        if plan[k]:
            slot = prob.slots[plan[k][0]]
            if mask[slot]:
                return int(slot)
        self.fallbacks += 1
        return self._fallback.act(env, obs, mask)
