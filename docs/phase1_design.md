# Phase 1 Design Document — Deep RL for Dynamic Multi-UAV Task Allocation

**Status:** Draft for review. No implementation files have been written.
**Date:** 2026-10-18
**Primary source:** G. Li, Y. Liu, Q. Zheng, K. Liu, "A Multi-Swarm Dynamic Crow Search Algorithm for Multi-UAV Dynamic Task Allocation", *Drones* 2026, 10(6), 467. DOI 10.3390/drones10060467. Full text (26 pp.) read from the MDPI PDF.

**Labels used throughout**

- **[PAPER]**: stated explicitly in the paper (equation, table or section number given).
- **[GAP]**: the paper does not give enough detail to reproduce this exactly.
- **[INCONSISTENCY]**: the paper's own values or statements appear to contradict each other.
- **[PROPOSAL]**: my design recommendation, which needs your approval.
- **[VERIFIED]**: checked on your machine (Apple M3, macOS, 2026-10-18).

---

## A. Technical analysis of the paper

### A.1 Real-world problem [PAPER §1, §3.1]

The paper addresses centralized, cooperative allocation of observation tasks to a heterogeneous UAV swarm. Example applications are environmental monitoring, disaster assessment and reconnaissance. Each target has an *observational value* that decays over time and is degraded by sensor limitations and the environment. During the mission, dynamic events change the scenario, and a central control center re-allocates "immediately upon detecting" an event.

The authors name three shortcomings of prior EA-based work:

1. Allocation decisions are decoupled from the actual execution state (no "state inheritance").
2. Only a narrow set of events is modeled, usually just crashes and new targets.
3. Response speed and solution quality can still be improved.

### A.2 Assumptions [PAPER §3.1]

- UAVs and targets are point masses. Flight paths are straight-line 3D Euclidean segments. While observing, a UAV hovers at the target.
- A UAV carries a subset of {VL (visible light), IR (infrared), RD (radar)} sensors. The environment's effect on sensing is collapsed into one effectiveness factor per sensor, ω_m.
- A centralized control center has full knowledge of the current scenario and receives external intelligence.
- **No** dynamics are modeled for communication, collisions, obstacles, turning radius, wind or return-to-base. The conclusion (§6) names these as future work and says the omissions "may produce optimistic estimates of the mission makespan, residual energy, and flight risk".
- All UAVs in the experiments have the same speed (15 m/s).

### A.3 Entity representation [PAPER Tables 1–2]

| UAV Uᵢ | Symbol | | Target Tⱼ | Symbol |
|---|---|---|---|---|
| Position at time t | pᵢᵗ | | Location at time t | pⱼᵗ |
| Flight speed | vᵢ | | Observational value at time t | Vⱼᵗ |
| Remaining energy (%) at t | Eᵢᵗ | | Required observation time | t_obse,j |
| Max. endurance time | t_endu,i | | Completion status at t | cⱼᵗ ∈ {0,1} |
| Vulnerability factor | ξᵢ | | | |
| Sensing capability at t | ηᵢ(t) | | | |
| Sensor availability vector at t | Iᵢᵗ | | | |

A target has no deadline, no priority class beyond its value, and no sensor requirement: any UAV can observe any target.

### A.4 Objectives [PAPER Eqs. 3–8]

Decision variable: Aᵗ, an **ordered task sequence for every UAV**, Aᵢᵗ = (T_{i,1}, …, T_{i,|Aᵢᵗ|}). Both objectives are minimized jointly: F(Aᵗ,t) = (f₁, f₂).

**f₁, remaining target value (Eq. 4):**

f₁(Aᵗ,t) = Σ_{Tⱼ ∈ T_unfin} Vⱼᵗ − Σᵢ Σ_{k ∈ Aᵢᵗ} Vₖᵗ · ηᵢ(t) · p_surv,i^(k) · exp(−λ_obse · Δt_{i,k})

- λ_obse = 0.001 s⁻¹ [PAPER §5.1.1].
- Δt_{i,k} is described as "the time elapsed since UAV Uᵢ reached target Tₖ". **[GAP]** Read literally, this is zero at arrival. The only sensible reading in a planning objective is *the time from the current planning instant t until the planned arrival at Tₖ*. The paper never states the reference time.

**Sensing capability, probabilistic multi-sensor fusion (Eq. 5):**

ηᵢ(t) = 1 − Π_{m ∈ Mᵢ} (1 − I_{m,i}ᵗ · ω_mᵗ), with I ∈ {0,1} and ω ∈ (0,1].

**Survival probability (Eqs. 6–7):** for each segment of length d_{i,k} in Uᵢ's sequence,

θ_{i,k} = 1 − exp(−λ_risk · ξᵢ · d_{i,k}), p_surv,i^(k) = p_surv,i^(k−1) · (1 − θ_{i,k}), p^(0) = 1,

with λ_risk = 0.0002 m⁻¹ [PAPER §5.1.1]. This is an *expected-value discount*. It is not a stochastic crash model: crashes in the experiments are scripted events. **[GAP]** The paper does not say whether p^(0) = 1 is reset at every re-planning instant. The equation suggests it is, which forgets risk already incurred.

**f₂, makespan (Eq. 8):** f₂ = maxᵢ t_comp,i, with t_comp,i = t + Σ_{k∈Aᵢᵗ} (d_{i,k}/vᵢ + t_obse,k), where d_{i,k} is measured from the previous position in the sequence.

### A.5 Constraints [PAPER Eqs. 9–11]

- **Completeness (9):** ⋃ᵢ Aᵢᵗ = T_unfin. Every unfinished target must be assigned.
- **Collaboration (10):** Aᵢᵗ ∩ Aⱼᵗ = ∅. Each target is assigned to exactly one UAV.
- **Energy (11):** gᵢ = Eᵢᵗ − γᵢ (t_comp,i − t) > 0, where γᵢ = 1/t_endu,i (fraction per second). Energy is therefore **purely time-based**: hovering and flying cost the same. No energy is reserved for return-to-base.
- Constraint (11) is handled softly during search (violation-tolerant ranking, tolerance ε₀ = |U|, Eq. 12). The *executed* solution is taken from the feasible archive A_feas. **[GAP]** The paper does not say what happens if A_feas is empty.

### A.6 Random-key encoding [PAPER §4.1, Fig. 1]

x ∈ [0, S)^Z, where Z is the number of targets that have appeared so far (Z only grows). ⌊xⱼ⌋ is the assigned UAV and the fractional part {xⱼ} is the execution priority (smaller runs first). Completed targets remain in the vector as skipped placeholders. An ongoing target is locked to its UAV with fractional part 0 so it runs first. New targets are appended.

By construction, decoding satisfies (9) and (10) with no repair step. Only (11) can be violated.

**Relevance to RL:** this encoding contains two ideas worth reusing: *stable index slots that never shrink* and *locking of ongoing tasks*. I reuse both in the observation design (§D).

### A.7 The five MDCSATA strategies [PAPER §4.2–4.6]

| # | Strategy | Purpose | Mechanism (paper) |
|---|---|---|---|
| 1 | Violation-tolerant multi-swarm co-evolution | Feasibility + diversity | M = 2 sub-swarms (one per objective, ⌊N/M⌋ = 50 each). Three-tier sort: feasible / violation ≤ ε / strictly infeasible. ε(τ) = ε₀(1−τ/τ_t)² (Eq. 12). Dual archives A_feas and A_inf with crowding-distance truncation. Memory update by ε-dominance with a stagnation counter mᵢ. |
| 2 | Objective-oriented heuristic initialization | Faster convergence | 50% Latin hypercube sampling. 25% spatiotemporal greedy: nearest UAV, within-UAV order by descending V/max(d,1). 25% round-robin load balancing. |
| 3 | Adaptive position update | Exploration/exploitation balance | Leader drawn by roulette on objective (3/8), binary tournament on crowding distance (3/8), random in-swarm (1/4). APᵢ = AP(0.5 + ln(1+mᵢ)) (Eq. 13). Lévy flight length (Mantegna, β = 1.5) under a cosine envelope (Eq. 14). CSA update (Eq. 15). |
| 4 | Stagnation & elite guided perturbation | Local exploitation | Stagnated crows: swap the integer parts of two free variables, roll back if worse. Elite: Gaussian σ = 0.1·range on one free variable for ⌈N/(5M)⌉ archive members. |
| 5 | Event-aware change response | Fast adaptation | Adapt dimensions (append / lock / release). Severity δ = δ_base + mean relative objective change over K = min(10,|A_feas|) re-evaluated elites (Eq. 16), δ_base = 0.15 if crash or new target. Regenerate with R_keep = max(0.2, 0.6−min(3δ,0.4)), R_heur = min(0.5, 0.2+min(3δ,0.3)), R_lhs = rest. |

Parameters [PAPER Table 6]: N = 100, AP = 0.05, fl = 1.5, fl_min = 1.0, archive size 100. Budget: 2τ_t = 40 iterations initially, then τ_t = 20 per event. 30 independent runs.

**Ablation [PAPER Tables 10–11]:** heuristic initialization (strategy 2) provides by far the largest gain (MHV +14–42%). The event-aware response (strategy 5) gives only small improvements; the authors themselves say it "mainly stabilizes adaptation". **This matters for us:** a large part of MDCSATA's quality comes from greedy and load-balanced seeds, so a well-designed greedy baseline is a strong, fair comparator.

### A.8 Dynamic events [PAPER §3.1, Table 5]

There are four categories: UAV (sensor malfunction, crash), target (new target, value decrease, movement), environment (weather reduces ω), and preference (decision-maker weight shift). All 15 events in the experiments are **scheduled at fixed, deterministic times**:

| No. | t (s) | Event | S1 | S2 | S3 |
|---|---|---|---|---|---|
| 1 | 100 | U5 crashes | | | ✓ |
| 2 | 100 | NT1 appears (2184, 6889, 83) | ✓ | ✓ | ✓ |
| 3 | 170 | Preference [0.5,0.5] → [0.75,0.25] | | | ✓ |
| 4 | 170 | T12 value 71 → 35 | | ✓ | ✓ |
| 5 | 250 | U2 VL sensor damaged | | | ✓ |
| 6 | 250 | U3 crashes | ✓ | ✓ | ✓ |
| 7 | 330 | Weather: ω = (VL .94, IR .58, RD .54) | | | ✓ |
| 8 | 330 | U1 IR sensor damaged | | ✓ | ✓ |
| 9 | 440 | T5 moves (1948,1934,82) → (1537,1365,125) | | | ✓ |
| 10 | 440 | Preference [0.5,0.5] → [0.75,0.25] | ✓ | ✓ | ✓ |
| 11 | 540 | T13 value 89 → 44 | | | ✓ |
| 12 | 540 | NT1 moves → (1206, 6405, 129) | | ✓ | |
| 13 | 540 | T15 moves → (1206, 6405, 129) | | | ✓ |
| 14 | 660 | NT2 appears (6792, 7246, 80) | | | ✓ |
| 15 | 660 | Weather: ω = (VL .75, IR .76, RD .91) | ✓ | ✓ | ✓ |

### A.9 Benchmark scenarios [PAPER §5.1.1, Tables 3–4]

- Airspace 8000 m × 8000 m × 200 m. S1/S2/S3 have 3/4/5 UAVs and 10/14/18 initial targets (plus NT1, and NT2 in S3). S1 and S2 are nested subsets of S3.
- UAV type A: 15 m/s, 1800 s endurance, VL+IR, ξ = 0.9. Type B: 15 m/s, 2400 s, VL+IR+RD, ξ = 1.0. All five UAVs start in the south-west corner (x ≤ 1466, y ≤ 1400, z ≈ 72–80 m).
- Target values range from 52 to 99 and observation times from 10 to 20 s. Targets are spread over the whole area, with z between 77 and 140 m.
- All coordinates and values are fully specified, **so S1–S3 can be rebuilt exactly.**

Scale check (my arithmetic): the airspace diagonal is ≈ 11.3 km, about 754 s at 15 m/s. Type A endurance (1800 s) therefore covers only about 2.4 diagonal crossings, so the energy constraint does bind.

### A.10 Baselines and metrics [PAPER §5.1.2–5.1.4, §5.2]

- **Baselines:** seven DMOEAs (DMOAWPSO, DCTAEA, DCMPSO, DNSGA-II, SGEA, MOEA/D-SVR, DBCSA-II), all with N = 100 and the same iteration budget.
- **Metrics:** MHV (Eq. 17) and MIGD (Eq. 18), averaged over the environment time steps, against a reference Pareto front built by a large-budget DNSGA-II. Also reported: Wilcoxon rank-sum with Bonferroni correction, a preference-weighted objective value of the *executed* solution (Fig. 4, plotted only), Gantt charts, and runtime.
- **Runtime [PAPER Table 16]:** MDCSATA takes 0.41 / 0.79 / 0.94 s per run on S1 / S2 / S3.
- **Headline:** best mean MHV and MIGD in all scenarios.
  - Table 9 contains 7 × 2 × 3 = 42 tests. 40 are significant at α = 0.05. 38 survive Bonferroni correction: S2-MIGD vs DCTAEA and S3-MHV vs DNSGA-II pass only at 0.05, and MHV vs DCTAEA on S2 and S3 is not significant.
  - The conclusion instead states "53 out of 56 pairwise comparisons" (see I4).

### A.11 Strengths, limitations, assumptions

**Strengths:**
- The state-inheritance idea is clean.
- The event taxonomy is broad.
- Ongoing tasks are locked by construction.
- The ablation and sensitivity studies are honest.
- Runtimes are short.
- The scenario data is fully tabulated.

**Limitations and gaps, which I separate from the paper's claims:**
1. **Quality is measured per planning step, not as realized outcomes.** MHV and MIGD measure how well the optimizer approximates each step's Pareto front. The paper never reports realized f₁/f₂ after executing the whole mission (Fig. 4 comes closest, but only as a plot).
2. **Only three hand-built, deterministic scenarios with fixed event times.** There is no randomized scenario distribution and no test of generalization.
3. Planning uses an expected-value survival model, while crashes are separate scripted events. The two risk notions are not linked.
4. The between-event execution simulator is not described: for example, how the executed plan is advanced, and what happens to a UAV that is in transit when an event hits.
5. Code is not released ("data will be made available on request").
6. Several ambiguities, listed next.

**[GAP] Missing information (needed for exact reproduction):**
- G1: the reference time for Δt_{i,k} in Eq. 4.
- G2: whether p_surv resets at each re-plan.
- G3: initial weather factors ω_m⁰. Presumably 1.0, but this is never stated.
- G4: the objective normalization used when choosing the executed solution with preference weights.
- G5: the exact ε-dominance definition and the constraint-violation magnitude.
- G6: how the reference front PF*ₜ is built (budget and seeds).
- G7: the between-event execution rules: in-transit reassignment, and what happens to an observation interrupted by an event.
- G8: whether idle UAVs consume energy.
- G9: the mission termination rule.
- G10: what is executed when A_feas is empty.

**[INCONSISTENCY]:**
- I1: Eq. 5's text says ω is "non-increasing during deterioration events". In S3, however, event 7 sets ω_IR = 0.58 and event 15 later sets ω_IR = 0.76, an *increase*. The same is true for RD (0.54 → 0.91). Either the factors are absolute values that can recover, or they are meant to combine multiplicatively; the paper does not say.
- I2: Events 12 (NT1, S2) and 13 (T15, S3) move two different targets to the identical point (1206, 6405, 129). This may be a copy error.
- I3: Event 10 (t = 440 s) shifts the preference "from [0.5,0.5]" in S3, but S3 already shifted to [0.75,0.25] at t = 170 s (event 3).

- I4: §6 claims significance in "53 out of 56 pairwise comparisons", but Table 9 has only 42 comparisons (40 significant at 0.05). Our report should cite Table 9, not the summary sentence.

None of these blocks the project, but they must be documented, and our simulator must pick and state one interpretation for each.

### A.12–A.14 What to preserve, what to simplify, what would break comparability

| Aspect | Decision | Effect on comparability |
|---|---|---|
| 3D Euclidean point-mass travel, constant speed | **Preserve** | Needed |
| Eqs. 4–7 value model (decay, η fusion, survival) | **Preserve** as the definition of collected value | Needed for f₁ to keep its meaning |
| Eq. 8 makespan | **Preserve** as an evaluation metric | Needed |
| Eq. 11 time-based energy, γ = 1/t_endu | **Preserve**, enforced as a hard constraint by action masking | Needed |
| Eqs. 9–10 completeness/exclusivity | Exclusivity: **preserve** (hard). Completeness: becomes an *outcome* (fraction completed), because a dispatch policy cannot assign targets that do not exist yet | Changes the problem class; must be stated |
| State inheritance, task locking, stable target indices | **Preserve** | Core to the paper |
| All six event types plus scenarios S1–S3 | **Preserve**, and add a stochastic generator | S1–S3 are a direct, apples-to-apples test set |
| Full-sequence Pareto optimization | **Replace** with a sequential dispatch policy (one assignment per decision) | We do not solve the same optimization problem. MHV/MIGD are not comparable (§I.4) |
| Preference shift event | Simplify: log only (v1), or condition the policy on w (extension, decision J4) | A fixed-w policy ignores this event entirely |
| MDCSATA itself | **Not reproduced.** Optionally implement a simpler random-key re-planning baseline, clearly labeled | We cannot claim "PPO vs MDCSATA" |

**Simplifications that would invalidate the comparison:** different units or λ constants; dropping the survival or decay factors; letting a UAV be reassigned mid-observation; allowing energy-infeasible tasks; comparing PPO's realized outcomes against the paper's MHV numbers; or tuning PPO on S1–S3 and then "testing" on them.

---

## B. Research question and proposed RL formulation

### B.1 Research question [PROPOSAL]

> **RQ.** In a simulator that implements the task-value, survival, makespan and energy model of Li et al. (2026), does a masked-PPO dispatch policy trained on randomly generated scenarios achieve realized mission outcomes (expected collected value fraction and makespan) at least as good as a value-rate greedy dispatcher, and at least within a stated margin of a rolling-horizon re-planning optimizer, while using at least 10× less decision time? How does its relative performance change across classes of dynamic events and across event rates, including rates and scenario sizes not seen in training?

**Independent variables:**
- Allocation method: PPO, random, greedy, and optionally RH-RK.
- Scenario size (S, Z₀).
- Event class mix and event rate.
- Energy scale (endurance multiplier).
- Target-value distribution.
- Reward weights (w₁, w₂).
- PPO training seed and evaluation scenario seed.

**Dependent variables:** the metrics defined in §I.3, chiefly the collected-value fraction, makespan, completion rate, the weighted score J, and decision latency.

**Hypotheses** (each is falsifiable; a rejected hypothesis is still a valid result):
- **H1 (sanity):** PPO > random valid allocation on J (paired test, α = 0.05).
- **H2 (core):** PPO ≥ greedy on J on held-out in-distribution scenarios (paired Wilcoxon). We will report the effect size and CI whether or not the hypothesis holds.
- **H3:** On S1–S3 (never used for training), PPO's J lies within δ = 5% of RH-RK's J. The margin is to be agreed; this hypothesis only applies if RH-RK is implemented.
- **H4 (latency):** PPO's median per-decision latency is at least 10× below RH-RK's per-replan latency.
- **H5 (event classes):** PPO's advantage over greedy differs by event class. My expectation is that it is largest for crashes and new targets, which make greedy myopia costly, and smallest for value changes. This is exploratory.
- **H6 (generalization):** PPO's advantage over greedy shrinks under out-of-distribution event rates and team sizes.

### B.2 Contribution and limits of claims

**Contribution** (measurable, not "PPO is novel"): an empirical, multi-seed characterization of *when* a learned dispatch policy is competitive with heuristic and re-planning baselines under each class of disruption in the paper's model. It is evaluated on realized mission outcomes, which the paper does not report, and on decision latency. A secondary contribution is an open, tested, reproducible simulator of the paper's model that includes an explicit resolution of gaps G1–G10.

**What we cannot claim:**
- That we reproduce MDCSATA, or beat it.
- That the results transfer to real UAVs.
- That the reward penalties guarantee safety.
- Generalization beyond the tested distribution.

**Reproduction vs. new solution:** the *environment* is a reproduction of the paper's problem, with documented interpretations. The *solver* (PPO dispatch) is a different method for a related but sequential decision problem.

### B.3 Scope assessment

This is realistic for one semester **if** RH-RK and the preference-conditioned extension are treated as optional. The rough training cost estimate comes from the toy benchmark in §H:
- Assume the environment runs at about 3–5k steps/s in pure Python.
- Training: 2M steps × 5 seeds ≈ 1–3 h on CPU per configuration.
- The core experiment matrix (2 reward configs × 5 seeds) is then a few hours per week of compute.

### B.4 MDP formulation [PROPOSAL]

The problem is formulated as an **event-driven semi-MDP with single-UAV sequential decisions.**

- **Decision epoch k:** occurs at continuous simulation time t_k whenever at least one *operational, unassigned ("ready")* UAV exists and a decision is required. If several UAVs are ready at the same t, they decide one at a time within the same instant, with no time advancing between them. The decision UAV is the ready UAV with the earliest ready time; ties go to the smallest index. This is deterministic and visible in the observation.
- **Full state s_k:** all UAV and target attributes (Tables 1–2), the per-UAV cumulative survival probability, the per-target appearance time and lock owner, weather ω, preference w, simulation time t, the decision-UAV index, and **the realized future event schedule and the RNG state.**
- **Observation o_k:** s_k *without* the future event schedule or RNG state (§D).
- **Action a_k ∈ {0,…,Z_max−1} ∪ {WAIT}:** the target slot assigned to the decision UAV (§C).
- **Transition:** deterministic physics plus exogenous events (§F, step order in §G.3).
- **Reward r_k:** accumulated over the simulated interval [t_k, t_{k+1}) (§E).
- **Discount:** applied per decision (approximation discussed in §G.4).

**Markov property.** The full state is Markov: given s_k and a_k, the next state depends only on physics plus the event process, which is fixed by the hidden schedule. The observation lacks the future schedule, so the agent faces a POMDP with respect to *future events only*. This is intended: the paper's control center also reacts only after detection. For memoryless (Poisson) stochastic events, the hidden schedule carries no information that past observations could reveal, so the agent loses nothing beyond what is inherently unpredictable. For **scheduled** events at fixed times (S1–S3), a policy could in principle learn the times from t. This is one more reason why training must use randomized event times and why S1–S3 are test-only.

---

## C. Action-space designs

The three candidate designs, with N_u = S_max UAVs and N_t = Z_max targets:

| | **A. Flat joint (UAV, target) pair** | **B. Factorized / hierarchical** | **C. UAV-centric discrete + invalid-action masking** |
|---|---|---|---|
| Space | Discrete(N_u·(N_t+1)) | MultiDiscrete([N_t+1]·N_u): every UAV picks at once; or two-stage (pick UAV, then target) | Environment selects the decision UAV; action = Discrete(N_t+1) (target slot or WAIT), with a mask |
| Size (S_max=6, Z_max=24) | 150 | 6 heads × 25 | 25 |
| Invalid actions | Many (busy UAV, done target, infeasible energy) | Duplicates across heads **cannot** be prevented by per-head masks, because SB3-contrib masks each dimension independently | All excluded by the mask; exclusivity is guaranteed because locked targets are masked |
| Credit assignment | Must also learn *which UAV* to move, which is redundant | Simultaneous joint action; conflicts need a resolution rule that the agent does not see | One clear decision per step |
| SB3 compatibility | PPO; masking via MaskablePPO | MaskablePPO supports MultiDiscrete masks, but cross-dimension conflicts need post-hoc repair, which silently changes actions | MaskablePPO (sb3-contrib), Discrete + `action_masks()` [VERIFIED §H] |
| Complexity per decision | O(N_u·N_t) mask | O(N_u·N_t) + conflict repair | O(N_t) mask |
| Main risk | Sample inefficiency; unmasked PPO wastes steps on penalties | Hidden repair violates "do not silently discard invalid actions" | Sequential decisions within one instant are ordered, so UAV order can bias the result; mitigated by showing the decision-UAV identity in the observation |

A fourth option, D, would be a continuous random-key action, Box([0,S)^Z), that outputs the paper's full encoding each step. It is noted but not recommended: PPO's Gaussian policy over a sort-key vector has a highly discontinuous decode, and it would really be a learned optimizer rather than a dispatch policy.

**Recommendation: C**, implemented with `sb3_contrib.MaskablePPO`. Option A with masks is kept as a possible ablation, and unmasked PPO with an invalid-action penalty as a second ablation, to show what masking buys.

**Validity rules (mask = 1 iff all hold).** For decision UAV i and target slot j:
1. Slot j is occupied by an appeared target.
2. The target is not completed (c_j = 0).
3. The target is not locked to another UAV (no duplicate assignment, Eq. 10).
4. UAV i is operational.
5. **Energy feasible:** Eᵢ − γᵢ·(d(pᵢ, pⱼ)/vᵢ + t_obse,j) > E_min, with E_min = 0 by default to match Eq. 11's strict "> 0". This is a configurable reserve.

**WAIT:** see decision J3. The recommended v1 is *work-conserving*: WAIT is allowed only when no target is valid. If no target is valid and the UAV has nothing to do, WAIT is the only valid action, so the mask is never all-zero. MaskablePPO would fail on an all-zero mask, and a unit test will guard this.

**After an action is selected:**
1. The environment re-checks the action against the mask. A violation raises `InvalidActionError` in `strict` mode (the default, which catches bugs). In `penalize` mode, used only for the unmasked ablation, it becomes a WAIT plus a penalty, the counter `n_invalid_actions` is incremented, and the event is logged. It is never silently discarded.
2. For a target: target j is locked to UAV i, the UAV's state becomes TRANSIT with destination pⱼ, and both the departure time and the planned arrival time are recorded.
3. For WAIT: the UAV becomes IDLE-WAITING until the next decision epoch (the next state change of any kind).
4. Low-level flight control is out of scope. Movement is linear interpolation at speed vᵢ.

**Commitment (continuity of ongoing tasks):** voluntary re-tasking is **not** allowed. A UAV in TRANSIT or OBSERVING cannot be redirected by the policy, which matches the paper's locking of ongoing tasks. An event can still *force* a change (§F). Whether transit (not just observation) counts as "ongoing" is gap G7; the alternative is listed as decision J2.

---

## D. Observation-space schema [PROPOSAL]

The observation is a fixed-shape flat `Box(low=-1, high=1, shape=(D,), float32)`. Entity slots are padded to S_max and Z_max (configurable, defaults 6 and 24, which covers S3's 5 UAVs and 20 targets with headroom). A separate *presence* bit marks real versus padded slots, and padded slots are all zeros.

**Normalization constants** are fixed per configuration, not per episode:
- L = (Lx, Ly, Lz)
- D_max = ‖L‖
- T_max (the mission horizon, 1800 s by default)
- V_max (the maximum possible target value, 100)
- v_ref = 15 m/s

**Per-UAV block (S_max × 20):**

| Feature | Dim | Range | Notes |
|---|---|---|---|
| present | 1 | {0,1} | padding mask |
| operational | 1 | {0,1} | 0 after crash |
| status one-hot (idle, transit, observing, failed) | 4 | {0,1} | |
| position / L | 3 | [0,1] | |
| energy Eᵢ | 1 | [0,1] | fraction |
| endurance-normalized consumption γᵢ·T_max | 1 | ≥0, clipped | type information |
| speed / v_ref | 1 | | |
| vulnerability ξᵢ | 1 | | |
| sensor availability I (VL, IR, RD) | 3 | {0,1} | 0 if not carried or damaged |
| sensing capability ηᵢ(t) | 1 | [0,1] | Eq. 5 |
| cumulative survival p_surv,i | 1 | [0,1] | Eqs. 6–7 |
| time until free / T_max | 1 | [0,1] | 0 if idle |
| is decision UAV | 1 | {0,1} | |

**Per-target block (Z_max × 16):**

| Feature | Dim | Range | Notes |
|---|---|---|---|
| present | 1 | {0,1} | slot used |
| completed | 1 | {0,1} | |
| locked (assigned/ongoing) | 1 | {0,1} | |
| locked by decision UAV | 1 | {0,1} | |
| position / L | 3 | [0,1] | current position (after any move) |
| current value Vⱼᵗ / V_max | 1 | [0,1] | after value-change events |
| decay factor exp(−λ_obse·(t − t_ref,j)) | 1 | [0,1] | t_ref per J1 |
| t_obse,j / 60 s | 1 | | |
| age (t − t_appear,j)/T_max | 1 | [0,1] | |
| *decision-UAV-relative*: distance / D_max | 1 | [0,1] | |
| *decision-UAV-relative*: energy margin after task | 1 | [−1,1] | Eᵢ − γᵢ(d/v + t_obs) |
| *decision-UAV-relative*: expected value if taken now / V_max | 1 | [0,1] | Eq. 4 term with arrival-time decay and segment survival |
| *decision-UAV-relative*: travel time / T_max | 1 | [0,1] | |
| valid for decision UAV | 1 | {0,1} | duplicates the mask; helps the value function |

**Global block (16):**
- t / T_max
- weather ω (3)
- preference w (2)
- operational UAV fraction
- unfinished target fraction
- one-hot of the most recent event type (6), zero if none
- time since last event / T_max
- whether any event is unacknowledged by this UAV

**Dimension:** D = 6·20 + 24·16 + 16 = **520**.

**New targets** take the next free slot, in order of appearance, matching the paper's append-only encoding. Slots are never reused within an episode, so indices stay stable. If a scenario would exceed Z_max, the environment **raises a configuration error at reset or on arrival**; a target is never silently dropped. The scenario generator caps total arrivals so that Z₀ + arrivals ≤ Z_max.

**No leakage:** the observation contains no future event times or types, no RNG state and no scheduled values. The `info` dictionary does carry privileged metrics for evaluation, but SB3 does not feed `info` to the policy. A unit test will assert that observations at time t are identical for two episodes that share the same past but differ in future events.

**Trade-off of a padded MLP versus entity encoders:** an MLP over padded slots is not permutation-invariant and generalizes poorly to slot counts it never saw filled during training. The decision-UAV-relative features reduce this problem, because the policy mostly needs to rank targets by locally computed quantities. If H6 shows poor size generalization, there would be a testable reason to try a shared per-target scoring network (a pointer-style head: one small MLP applied per target, then a softmax over targets). This would be a Phase 11 option. A Transformer or GNN is not planned.

---

## E. Reward function [PROPOSAL]

For decision step k, which covers the simulated interval [t_k, t_{k+1}):

r_k = w₁ · R_val,k − w₂ · R_time,k − w₃ · R_energy,k − c_inv · 1[invalid_k]

| Term | Definition | Sign / units | Purpose |
|---|---|---|---|
| R_val,k | (1/V_ref) · Σ over observations **completed** in the interval of V_j(t_c) · ηᵢ(t_c) · p_surv,i(t_arr) · exp(−λ_obse(t_arr − t_ref,j)) | + / dimensionless | Expected collected value, i.e. the realized form of Eq. 4. Its episode sum is (Σ initially present value − f₁ at termination)/V_ref. |
| R_time,k | (t_{k+1} − t_k)/T_max **while the mission is active** (some unfinished target is feasible for some operational UAV) | − / dimensionless | Its episode sum equals realized makespan / T_max, i.e. Eq. 8 measured on the executed mission |
| R_energy,k | (1/S_max) · Σᵢ ΔEᵢ over the interval | − / fraction | **Off by default** (w₃ = 0). The paper treats energy as a constraint, not an objective. Used only in the trade-off ablation |
| invalid | 1 if an invalid action was given (penalize mode only) | − | Only for the unmasked ablation; c_inv = 0.1 |

- V_ref = the sum of the values of the targets present at reset. This is known at reset, so nothing leaks.
- T_max = the mission horizon.
- Default weights: w = (w₁, w₂) = (0.5, 0.5), which mirrors the paper's initial preference.

**Why this form:**
- Both of the paper's objectives stay separately identifiable: the episode return is w₁·(collected fraction) − w₂·(makespan fraction), each in [0,1]. These two quantities are also logged separately.
- There are no unexplained magic constants: each term is normalized by a quantity defined by the scenario.
- Collected value is realized at observation *completion*, so interrupted observations (crash, target move) earn nothing.

**Interpretation choices (these resolve gaps; see J1):**
- t_ref,j = 0 for initial targets and t_appear,j for new targets. Decay is absolute, not relative to re-planning instants. This avoids the inconsistency where re-planning resets decay.
- p_surv,i is cumulative from mission start, not reset at re-plan (gap G2).
- ηᵢ and Vⱼ are evaluated at completion t_c, and decay at arrival t_arr. This is the paper's arrival-time decay, with sensing and value as they are when the data is actually obtained.

**Hard constraints are enforced by the simulator and the mask, not by penalties:** energy feasibility, exclusivity, and no re-assignment of completed targets. No reward penalty is claimed to guarantee safety. Runtime assertions check that no energy goes negative, that no target is completed twice, and that no two UAVs hold the same lock. A violated assertion is a bug and raises an exception.

**Reward hacking risks identified:**
1. *Early termination by energy drain:* if idle UAVs consume energy (decision J5) and WAIT is voluntary, a policy could deliberately drain energy until the remaining targets become infeasible, which ends the mission and stops the time cost. The work-conserving WAIT (J3) removes this, and the evaluation reports completion % alongside the return.
2. *Cherry-picking:* with w₂ large, the agent may skip far targets to shorten makespan. This is a legitimate trade-off, made visible by sweeping w.
3. *Return ≠ quality:* the paper-comparable metrics (§I.3) are reported separately from the return.

**Is a weighted scalar appropriate?** For a single policy, yes, and it matches how the paper picks its *executed* solution (preference weights). It cannot represent non-convex parts of the front. The trade-off study therefore trains separate policies for w₁ ∈ {0.25, 0.5, 0.75}, or a preference-conditioned policy (J4), and plots realized (value, makespan) pairs.

**Reward ablations (Phase 11):**
- (i) value only
- (ii) value + time (default)
- (iii) + energy term
- (iv) no survival factor in the reward, while evaluation still uses it
- (v) unmasked PPO with an invalid penalty versus masked PPO

---

## F. Dynamic event model [PROPOSAL, grounded in PAPER Table 5]

**Generation.** At `reset(seed)`, a schedule is drawn from a dedicated RNG stream (via `numpy.random.SeedSequence(seed).spawn(...)`, kept separate from the scenario-layout stream). Two modes are supported:
1. **Scheduled:** an explicit list from YAML. S1–S3 are reproduced exactly this way.
2. **Stochastic:** a Poisson process per event class with rate ρ_c (events per 1000 s), giving continuous times in (0, T_event_max]. Optionally, a per-class cap on the number of events. Fixed-step events exist only in unit-test fixtures.

| Event | Trigger (stochastic) | Severity / parameters | State change | Effect on ongoing tasks | Indicator |
|---|---|---|---|---|---|
| New target | Poisson ρ_new; capped so Z ≤ Z_max | V ~ U[50,100], t_obs ~ U[10,20] s (brackets the paper's values), position uniform over the area, z ~ U[75,140] | Fill the next slot; t_appear = t | None | One-hot "new target" |
| UAV crash | Poisson ρ_crash on a uniformly chosen operational UAV; optional `min_operational` guard | — | operational = 0, status FAILED, energy frozen | Its lock is released. An in-progress observation is lost (no value) and the target becomes free again | One-hot "crash" |
| Sensor damage | Poisson ρ_sens; random UAV and one carried, working sensor | — | I_{m,i} = 0, recompute ηᵢ | Ongoing task continues with lower η (evaluated at completion) | One-hot "sensor" |
| Weather | Poisson ρ_wx | Per-sensor multiplicative factor ~ U[0.5,1.0] (non-increasing within an episode, per the paper's text) **or** absolute set (S1–S3 replay, per I1) | ω updated, η recomputed for all UAVs | Same as sensor damage | One-hot "weather" + ω in the observation |
| Target move | Poisson ρ_move; random unfinished target | Displacement ~ U[300, 1500] m in a random direction, clipped to the airspace (the paper's moves are ≈ 0.7–1.1 km) | pⱼ updated | In transit: redirected from the current position, feasibility re-checked, aborted if infeasible. **Observing: interrupted, redirected, observation timer restarts** (G7; alternative in J2) | One-hot "move" |
| Value change | Poisson ρ_val; random unfinished target | Vⱼ ← f·Vⱼ, f ~ U[0.4,0.7] (the paper's two cases are both ≈ 0.49) | Vⱼ updated | None (value is read at completion) | One-hot "value" |
| Preference (optional) | Scheduled only in v1 | w ← new weights | w updated; reward weights change if J4 is adopted | None | w in the observation |

- Sampling ranges are design choices, not paper values. They are configurable and documented as such.
- **Abort semantics:** an aborted UAV becomes ready immediately at its current position with its energy as consumed so far, the target is unlocked, and the abort is logged.
- **Same-time ordering:** events are applied in a fixed type order: crash → sensor → weather → move → value → new target → preference. Within a type, they are applied in schedule order. Crash comes first so that no newly freed task goes to a UAV that crashes in the same instant.
- **Logging:** every event is recorded in `info["events"]` with its time, type, parameters and affected entities. The episode log keeps the full timeline used for the response-latency metric and the timeline plot.
- **Response at the next decision:** any event that frees a task or changes feasibility triggers a decision epoch for every ready UAV at the event time, so the policy can respond immediately.

---

## G. Software architecture [PROPOSAL]

### G.1 Repository layout

```
Project/
├── README.md, pyproject.toml, CLAUDE.md
├── docs/                     phase1_design.md, assumptions.md (G/I resolutions), report/
├── configs/
│   ├── env/                  base.yaml, paper_S1.yaml, paper_S2.yaml, paper_S3.yaml, gen_small.yaml …
│   ├── events/               none.yaml, paper_S{1,2,3}.yaml, poisson_low/med/high.yaml, per_class/*.yaml
│   ├── train/                ppo_default.yaml, ppo_ablation_*.yaml
│   └── eval/                 heldout.yaml, ood_size.yaml, ood_rate.yaml
├── src/uavta/
│   ├── entities/             uav.py, target.py           (dataclasses, no logic beyond invariants)
│   ├── dynamics/             kinematics.py, energy.py, sensing.py, survival.py   (pure functions, Eqs. 4–11)
│   ├── events/               types.py, schedule.py, generator.py, apply.py
│   ├── scenarios/            loader.py (YAML→Scenario), generator.py (random layouts), paper.py (S1–S3 tables)
│   ├── environment/          simulator.py (event-driven core, no Gym), env.py (MultiUAVDynamicEnv),
│   │                         observation.py, masking.py, reward.py
│   ├── baselines/            random_policy.py, greedy.py, rh_rk.py (optional)
│   ├── agents/               ppo_factory.py, callbacks.py
│   ├── training/             train.py
│   ├── evaluation/           runner.py, metrics.py, stats.py
│   ├── visualization/        map.py, gantt.py, timeline.py, curves.py, comparisons.py
│   └── utils/                config.py, seeding.py, device.py, logging.py, provenance.py
├── tests/                    unit/, integration/, fixtures/
├── scripts/                  train.py, evaluate.py, make_figures.py, check_env.py
└── results/                  runs/<run_id>/{config.yaml, versions.json, seeds.json, model.zip, eval.parquet}
```

The **`simulator.py` vs `env.py` split** keeps all physics and event logic free of Gymnasium and PyTorch. Baselines call the simulator through the same interface as PPO, so the comparison is fair. Unit tests can drive the simulator directly.

### G.2 Gymnasium API

- `__init__(config)` builds spaces from S_max, Z_max and the feature specification.
- `reset(seed=None, options=None)` calls `super().reset(seed=seed)` and derives independent streams from `self.np_random` (layout, events). `options` can pin a specific scenario, for example `{"scenario": "paper_S1"}`. It returns `(obs, info)`.
- `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- `action_masks()` is the MaskablePPO hook.
- `render()` returns an RGB array via the visualization module, optional and off the hot path.

**terminated vs truncated:**
- `terminated = True` when the true MDP ends: all appeared targets are completed **and** no future arrivals are possible (t > T_event_max), **or** no operational UAV remains, **or** no unfinished target is feasible for any operational UAV and no future event could change that.
- `truncated = True` when t reaches T_max, a time limit that is *not* part of the task. PPO bootstraps the value there.

**`info` contents:** collected value, makespan so far, energy per UAV, completed count, invalid count, the event log, and the decision UAV id. These are for evaluation only.

**Checks:** `gymnasium.utils.env_checker.check_env` in CI, and `stable_baselines3.common.env_checker.check_env`.

### G.3 Exact `step()` order

1. **Validate:** a ∈ action_space, and mask[a] = 1. Otherwise, strict mode raises and penalize mode logs a WAIT.
2. **Apply** the decision to the current decision UAV (lock + TRANSIT, or WAIT).
3. **Same-instant check:** if another UAV is still ready at t_k with something to decide, return its observation (reward = rewards accrued so far at this instant, Δt = 0).
4. **Advance** to t_next = min(next arrival, next observation completion, next exogenous event, T_max). Move UAVs linearly, deduct energy γᵢ·Δt (for idle UAVs, per J5), multiply survival by exp(−λ_risk ξᵢ Δd) for flown distance, and accrue R_time.
5. **At t_next, process in fixed order:** (a) arrivals → start observation and record t_arr; (b) observation completions → R_val, mark completed, unlock, UAV ready; (c) exogenous events in type order (§F); (d) re-validate every committed task (redirect or abort); (e) recompute readiness and masks.
   - Justification: physical completions at t happen *before* an event detected at the same t, so data already collected is not destroyed retroactively. The order is the same in every situation.
6. **Repeat** steps 4–5 until some UAV needs a decision, or the episode terminates or truncates.
7. **Build** the observation for the new decision UAV, then return.

**Survival is applied per segment flown:** the product of exp(−λ_risk ξ Δd) equals Eqs. 6–7 for complete segments, and handles partial segments (a redirect) consistently.

### G.4 RL algorithm [PROPOSAL]

**PPO vs alternatives:**
- **MaskablePPO:** an on-policy, clipped-ratio policy-gradient method with native action masking via sb3-contrib. It is stable and well documented, and it works on CPU.
- **DQN (SB3):** more sample-efficient per step, but SB3's DQN has no action masking. Masking would require changing the target max and the ε-greedy step, which means custom code and more risk.
- **QR-DQN (sb3-contrib):** has no masking either.
- **MAPPO and other multi-agent methods:** no clear testable need yet, because the problem is centrally controlled, exactly as in the paper.
- **Decision:** MaskablePPO.

**Default network and hyperparameters:**
- Network: MLP 520 → [256, 256] (tanh), with separate policy and value heads (`net_arch=dict(pi=[256,256], vf=[256,256])`).
- Action distribution: a categorical over 25 logits. Masked logits are set to −∞ by the `MaskableCategorical` distribution.
- Value function: V(o) on the same observation.
- Objective: clipped surrogate + c_v·value loss − c_e·entropy.
- Starting hyperparameters (to be tuned on a validation split, not on the test split): γ = 0.995 per decision, GAE λ = 0.95, clip 0.2, ent_coef = 0.01, n_steps = 2048 × 8 envs, batch 256, 10 epochs, lr 3e-4 with linear decay.

**Semi-MDP caveat:** decisions are irregularly spaced in time, and SB3 discounts per step, not per second. Because R_time already prices elapsed time explicitly and γ is close to 1, per-step discounting is an acceptable approximation. It is stated as a limitation. The alternative, γ^{Δt}, would require a custom rollout buffer.

**Diagnostics** logged to TensorBoard or CSV:
- Training curves: episodic return, approximate KL, clip fraction, explained variance, and entropy.
- Separate eval callbacks on a fixed validation scenario set every N steps, tracking the true metrics.
- Early warnings: entropy collapse, or explained variance below 0.

---

## H. Compatibility assessment: Apple Silicon M3

**[VERIFIED] on this machine (Apple M3, arm64), Python 3.12.x from Homebrew, in a throwaway venv:**

| Package | Version installed | Notes |
|---|---|---|
| torch | 2.14.1 | `torch.backends.mps.is_available() == True` |
| gymnasium | 1.4.0 | |
| stable-baselines3 | 2.9.0 | requires torch ≥ 2.8, gymnasium ≥ 0.29.1, < 2.0 |
| sb3-contrib | 2.9.0 | provides MaskablePPO |
| numpy | 2.5.4 | |

**Smoke benchmark** (toy env: 400-dim observation, Discrete(126) with random masks, MaskablePPO MlpPolicy, n_steps 2048, batch 256, 8192 steps, `check_env` passed):

| device | throughput |
|---|---|
| **cpu** | **≈ 5,300 steps/s (1.5 s)** |
| mps | ≈ 340 steps/s (24.0 s) |

**MPS was about 15× slower than CPU** for this workload size. This is consistent with kernel-launch and transfer overhead dominating tiny MLPs and per-step single-observation inference during rollouts.

**SB3 device handling [VERIFIED by reading `stable_baselines3/common/utils.py::get_device`]:** `device="auto"` maps to `"cuda"` and falls back to CPU. **It never selects MPS.** MPS is used only if `device="mps"` is passed explicitly, which worked in the test above. A naive "auto" therefore silently runs on CPU on a Mac, so our logs must report `model.device`, not the requested string.

**Device-selection strategy** (`utils/device.py`):
1. Read the config value `device: cpu | mps | auto`. The default is **cpu**.
2. `auto` → MPS only if `torch.backends.mps.is_available()` **and** the policy is marked as accelerator-worthy (a large network, for example a future entity encoder). Otherwise CPU.
3. If MPS is requested but unavailable, warn and fall back to CPU.
4. Log the *resolved* `model.device`, the torch version and the MPS availability into the run's `versions.json`. Never claim GPU acceleration unless `model.device.type == "mps"`.
5. The simulation always runs in NumPy on CPU. Environments are parallelized with `SubprocVecEnv` (using the macOS spawn start method) or `DummyVecEnv`, chosen by measurement.

**MPS limitations relevant here:**
- No float64 support. SB3 already casts float64 actions to float32, and our observation is float32.
- Some ops fall back to CPU or are unsupported, depending on the torch version (`PYTORCH_ENABLE_MPS_FALLBACK=1` can hide this, so we will not set it silently).
- Results are not bit-identical between MPS and CPU, and MPS is not fully deterministic. Reproducibility runs use CPU.

**Python version:** I propose **3.12**. It is installed, and every package above ships arm64 wheels for it. 3.13 is also installed, but 3.12 has the longer track record across the scientific stack. This is decision J8.

**Installation** (to be finalized in Phase 5; the pins match the verified run above):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # pins: torch==2.14.1 gymnasium==1.4.0 stable-baselines3==2.9.0 sb3-contrib==2.9.0
```

Other planned dependencies: numpy, pyyaml, pydantic (config validation), pandas + pyarrow (results), matplotlib, scipy (statistics), pytest, pytest-cov, ruff, mypy, and tensorboard (optional).

---

## I. Experimental plan

### I.1 Baselines

1. **Random valid:** uniform over the mask, seeded.
2. **Greedy value-rate (deterministic):** the decision UAV picks argmax_j [expected collected value_j / (travel_j + t_obs,j)] over valid j; ties go to the lowest slot index; WAIT if nothing is valid. This is inspired by the paper's spatiotemporal greedy heuristic, which uses V/max(d,1).
   - *Limitations:* myopic, ignores the other UAVs' future positions, ignores load balance, and cannot anticipate.
3. **Greedy nearest:** the closest valid target. This is a simpler second heuristic that isolates the value of the value-awareness in baseline 2.
4. **(Optional) RH-RK, a rolling-horizon random-key re-planner:** at t = 0 and after every event, run a single-objective, weighted-sum optimizer over the paper's random-key encoding. The optimizer would be a plain CSA (Eq. 2) or a GA with N = 100 and 40/20 iterations, seeded with the greedy and round-robin heuristics. The resulting sequences are then executed by dispatch.
   - This is **our own baseline, inspired by, not a reproduction of, MDCSATA.** A faithful MDCSATA would need G4–G6 and G10 resolved and a multi-objective archive. It is out of scope unless you want it (J7).

All methods use the same simulator, the same scenarios and seeds, the same `step()` semantics and the same metrics code.

### I.2 Data splits

| Split | Content | Use |
|---|---|---|
| Train | Generated scenarios, scenario seeds from range A, S ∈ {3,4,5}, Z₀ ∈ {8..18}, medium Poisson rates for all classes | Training only |
| Validation | 100 generated scenarios from range B | Hyperparameter choice and checkpoint selection |
| Test-ID | 500 generated scenarios from range C | H1, H2, H5 |
| Test-Paper | S1, S2, S3 exactly as in Tables 3–5 (with the I1–I3 interpretations documented) | H3; direct link to the paper's setting |
| Test-OOD | Larger teams (S = 6, Z₀ up to 22), high event rates, single-class "stress" schedules, tighter endurance (×0.7), skewed value distributions | H6 |

Seed ranges A, B and C are disjoint and recorded.

### I.3 Metrics

Each metric is computed per episode, then aggregated.

| Metric | Definition | Better |
|---|---|---|
| Mean episodic return | Σ r_k | ↑ (training diagnostic only) |
| Collected value fraction | Σ expected collected value / Σ value of all targets that appeared (current values) | ↑ |
| Realized residual value f̂₁ | Paper Eq. 4 evaluated on the executed mission at termination | ↓ |
| Completed targets | count and % of appeared targets | ↑ |
| Realized makespan f̂₂ | Time of the last observation completion | ↓ (report with completion %) |
| Weighted score J | w₁·value fraction − w₂·makespan/T_max (the same weights as training) | ↑ |
| Energy | Total and per-UAV ΔE; minimum residual energy | ↓ / report |
| Failed missions | Episodes with completion < 100% while feasible targets remained at T_max | ↓ |
| Invalid actions | Count (0 by construction when masked; non-zero only in the ablation) | ↓ |
| Risk exposure | 1 − mean cumulative survival over UAVs at the end | ↓ |
| Post-event response delay | Simulated seconds from an event until the first new assignment touching an affected entity | ↓ |
| Value lost per event | Difference in collected value versus the same scenario with that event removed (counterfactual replay; the simulator supports this) | ↓ |
| Decision latency | Wall-clock ms per `predict()` (PPO) or per decision/re-plan (baselines), median and p95, CPU, single thread | ↓ |
| Training time | Wall-clock time and environment steps per run | report |
| Seed variability | std across training seeds | report |

**MHV / MIGD are not reported.** They measure how well an approximate Pareto *front* tracks a reference front at each environment step. A single scalarized dispatch policy outputs one executed trajectory per episode, not a front, and there is no per-step reference front for sequential dispatch. Reporting them would be a category error.

If the preference-conditioned extension (J4) is adopted, we can report the **hypervolume of realized (f̂₁, f̂₂) outcomes across a sweep of w**. This metric would be clearly named "realized-outcome HV" and not equated with the paper's MHV.

### I.4 Statistics

- **PPO:** 5 independent training seeds (10 if the budget allows). Each trained policy is evaluated deterministically (argmax) on the entire test split.
- **Reporting:** mean ± std across training seeds of the per-seed mean, plus 95% bootstrap CIs, plus IQM (interquartile mean) for robustness.
- **Paired comparisons:** on the same scenarios, a Wilcoxon signed-rank test with Holm correction over the hypothesis family, plus effect sizes (median paired difference, rank-biserial r).
- No single-run comparisons.

### I.5 Generalization and sensitivity tests

The grid covers:
- S ∈ {3,4,5,6}
- Z₀ ∈ {8, 14, 20}
- event rate × {0, 0.5, 1, 2, 4}
- single-class schedules for each of the 6 types
- value distributions (uniform / bimodal / heavy-tailed)
- endurance × {0.7, 1, 1.3}

Report the PPO − greedy difference in J with CIs per cell.

### I.6 Visualizations (Phase 10)

| Figure | What it shows | Question it answers |
|---|---|---|
| 2D/3D map with routes, locks and completion state at event times | The paper's Figs. 6–8 equivalent | Do allocations make spatial sense? |
| Gantt per UAV (transit / observe / idle / failed) with event markers | The paper's Fig. 5 equivalent | Load balance and reaction to events |
| Event timeline with value collected over time per method | | Where value is lost |
| Training curves with seed bands (return, value fraction, makespan on validation) | | Convergence and stability |
| Box/strip plots of J, value fraction and makespan per method; heatmap of PPO−greedy ΔJ over (event rate × size) | | H2, H5, H6 |
| Realized value-vs-makespan scatter across w | | Trade-off |

---

## J. Unresolved decisions requiring your approval

| ID | Decision | Options | My recommendation |
|---|---|---|---|
| **J1** | Interpretation of G1/G2 (decay reference, survival reset) | (a) absolute decay from appearance + cumulative survival; (b) the paper-literal per-replan reset | **(a)**: consistent for realized outcomes; documented as a deviation |
| **J2** | Commitment rule | (a) no voluntary re-tasking; transit and observation both locked; events force redirect or abort; (b) allow re-tasking of in-transit UAVs at event epochs | **(a)** for v1; (b) as an ablation |
| **J3** | WAIT semantics | (a) work-conserving (WAIT only if nothing is valid); (b) voluntary WAIT allowed | **(a)**: removes a reward-hacking path and matches greedy |
| **J4** | Preference events | (a) fixed w, preference events logged only; (b) w in the observation and reward, policy trained over random w (preference-conditioned) | **(a)** for the core; (b) as the main extension if time allows. It is the most interesting, paper-aligned contribution |
| **J5** | Idle energy use (G8) | (a) hovering idle UAVs consume γ like flight; (b) no consumption when idle | **(a)**: consistent with Eq. 11's time-based model and the multirotor hover assumption |
| **J6** | Interpretation of I1 for S1–S3 replay | (a) absolute ω values as tabulated; (b) multiplicative | **(a)**: matches the table literally; flagged in the report |
| **J7** | Optimization baseline | (a) none; (b) RH-RK (our own, labeled); (c) attempt MDCSATA reproduction | **(b)** if Phases 6–10 finish on time; otherwise (a) |
| **J8** | Python version | 3.12 / 3.13 | **3.12** |
| **J9** | Entity caps | S_max = 6, Z_max = 24 | Accept (covers S3 plus the OOD test) |
| **J10** | Mission horizon | T_max = 1800 s (the longest endurance among type-A UAVs; type-B UAVs have 2400 s) or 2400 s | **2400 s**: no UAV is cut off by the horizon before its energy |
| **J11** | H3 non-inferiority margin δ | 2% / 5% / 10% | **5%**, fixed *before* running experiments |
| **J12** | Training budget per run | 1M / 2M / 5M steps | **2M**, revisited after the learning curves in Phase 9 |

---

## K. Prioritized implementation roadmap

Each phase ends with a review gate: I list the design decisions, open issues, files changed and acceptance criteria, and you approve before the next phase starts.

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| **2** Math model | `docs/assumptions.md`: every symbol, unit and range; the G1–G10 and I1–I3 resolutions per J | You approve the document; each equation maps to one named function |
| **3** Spaces | Final observation/action specification table and mask rules | Feature list frozen; mask invariants written as test cases |
| **4** Reward + events | Final reward formulas, event YAML schema, S1–S3 YAML transcribed from Tables 3–5 | S1–S3 YAML double-checked cell by cell against the paper |
| **5** Repo + dependencies | `pyproject.toml`, skeleton packages, config loader, seeding, device utility, CI commands | `pip install -e .[dev]` and `pytest` (empty) pass on the M3; `versions.json` written |
| **6** Simulator + env | entities, dynamics, events, simulator, env, observation, masking, reward | Hand-computed 1-UAV/2-target scenario matches the closed-form value and makespan to 1e-9 |
| **7** Tests | Unit + integration tests (list below); `check_env` passes | All pass; coverage ≥ 85% on `environment/`, `dynamics/`, `events/` |
| **8** Baselines | random, greedy-value-rate, greedy-nearest + evaluation runner | Deterministic greedy results on S1–S3 are reproducible bit-for-bit across two runs |
| **9** PPO | factory, training script, callbacks, checkpointing | A 50k-step smoke run beats random on validation; a 2M-step run shows a stable learning curve |
| **10** Evaluation | Multi-seed experiments, statistics, figures | All Test-ID/Paper tables generated from saved results by one script |
| **11** Ablations + OOD | Reward ablations, masking ablation, generalization grid, optional J4/J7 | Each ablation runs with ≥ 5 seeds |
| **12** Report | Academic report + slides, limitations | Every number in the report is traceable to a results file |

**Planned tests (Phase 7):**
- **Core:** reset determinism with a fixed seed, observation-space containment, action-space and mask shape, mask never all-zero, energy accounting (closed form), target completion, duplicate-lock prevention.
- **Events:** new-target slotting, crash during transit, crash during observation, sensor degradation effect on η, target move redirect and abort.
- **Mission lifecycle:** task continuity across an unrelated event, termination (all done / no UAVs / nothing feasible), truncation at T_max, reward sum = w₁·value fraction − w₂·makespan/T_max.
- **Leakage and compliance:** no-leakage test (same past, different future), Gymnasium `check_env`.
- **Edge cases:** no targets, all UAVs failed, new target while all are busy, an event making a committed task infeasible.
- **Integration:** a 2k-step MaskablePPO train plus evaluation episode on every baseline.

---

## Sources

- Paper: Li et al., *Drones* 2026, 10(6), 467. <https://www.mdpi.com/2504-446X/10/6/467>. The PDF was retrieved from MDPI's file server because the article page returned HTTP 403 to automated requests.
- Related paper by the same authors on the CSA core: Li et al., "Multi-Swarm Dynamic Crow Search Algorithm for Dynamic Multi-Objective Optimization", *IEEJ Trans. Electr. Electron. Eng.* 21(3):421–430, DOI 10.1002/tee.70165. Not read; identified via Crossref.
- Package metadata: PyPI JSON API, queried 2026-10-18. Local install and benchmark on the user's M3 as described in §H.
