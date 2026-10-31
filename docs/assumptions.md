# Model Assumptions and Interpretations (Phases 2–4)

**Status:** Living document. Every simulator rule that is not stated verbatim in the paper is listed here.
**Companion:** `docs/phase1_design.md` (full analysis and rationale).
**Paper:** Li et al., *Drones* 2026, 10(6), 467 — referred to as "the paper".

Labels: **[PAPER]** stated in the paper · **[DECISION Jx]** approved design decision · **[NEW Nx]** decision found during implementation, needs review.

---

## 1. Approved decisions

The user instructed implementation to start on 2026-10-18. This is recorded as approval of the recommended option for every Phase 1 decision:

| ID | Topic | Adopted option |
|---|---|---|
| J1 | Decay reference / survival reset (G1, G2) | (a) Absolute decay from t_ref,j (0 for initial targets, appearance time for new ones); survival cumulative from mission start, never reset |
| J2 | Commitment | (a) No voluntary re-tasking; transit and observation are both locked; only events force redirect/abort |
| J3 | WAIT | (a) Work-conserving: WAIT is valid only when no target is valid for the decision UAV |
| J4 | Preference events | (a) Fixed reward weights; preference events are logged and shown in the observation only |
| J5 | Idle energy (G8) | (a) Idle (hovering) UAVs consume γᵢ per second like flying UAVs |
| J6 | Weather in S1–S3 replay (I1) | (a) Tabulated ω values are absolute |
| J7 | Optimization baseline | (b) RH-RK, our own labelled baseline, if Phases 6–10 finish on time |
| J8 | Python | 3.12 |
| J9 | Entity caps | S_max = 6, Z_max = 24 |
| J10 | Mission horizon | T_max = 2400 s |
| J11 | H3 margin | δ = 5 %, fixed before experiments |
| J12 | Training budget | 2M steps per run, revisited after Phase 9 curves |

---

## 2. Units and symbols

| Symbol | Code name | Unit | Range | Source |
|---|---|---|---|---|
| pᵢ, pⱼ | `position` | m, shape (3,) | inside airspace [0,Lx]×[0,Ly]×[0,Lz] | [PAPER Tables 1–2] |
| L | `airspace` | m | (8000, 8000, 200) for S1–S3 | [PAPER §5.1.1] |
| vᵢ | `speed` | m/s | > 0; 15 in S1–S3 | [PAPER Table 3] |
| t_endu,i | `endurance` | s | > 0; 1800 (A) / 2400 (B) | [PAPER Table 3] |
| γᵢ = 1/t_endu,i | `consumption_rate` | 1/s (energy fraction per s) | > 0 | [PAPER Eq. 11] |
| Eᵢ | `energy` | fraction of full battery | [0, 1]; 1 at t = 0 | [PAPER Table 1, "percentage"] |
| ξᵢ | `vulnerability` | — | > 0; 0.9 (A) / 1.0 (B) | [PAPER Table 3] |
| Iᵢ | `sensor_ok` | bool (VL, IR, RD) | carried ∧ not damaged | [PAPER Eq. 5] |
| ω | `weather` | — | (0, 1] per sensor; ω⁰ = 1 | [PAPER Eq. 5]; ω⁰ is G3 |
| ηᵢ | `sensing_capability` | — | [0, 1) | [PAPER Eq. 5] |
| λ_obse | `lambda_obse` | 1/s | 0.001 | [PAPER §5.1.1] |
| λ_risk | `lambda_risk` | 1/m | 0.0002 | [PAPER §5.1.1] |
| p_surv,i | `survival` | probability | (0, 1] | [PAPER Eqs. 6–7] |
| Vⱼ | `value` | value units | > 0; 52–99 in S1–S3 | [PAPER Table 4] |
| t_obse,j | `observation_time` | s | > 0; 10–20 in S1–S3 | [PAPER Table 4] |
| t_ref,j | `reference_time` | s | 0 or appearance time | [DECISION J1] |
| w = (w₁, w₂) | `preference` | — | w₁ + w₂ = 1; (0.5, 0.5) at t = 0 | [PAPER §5.1.3] |
| T_max | `horizon` | s | 2400 | [DECISION J10] |

All times are continuous simulation seconds from mission start. All arrays are float64 inside the simulator; only the RL observation is float32.

---

## 3. Equation → function map

| Paper | Function |
|---|---|
| Euclidean segment length (§3.1) | `uavta.dynamics.kinematics.distance` |
| d/v travel time (Eq. 8) | `uavta.dynamics.kinematics.travel_time` |
| Linear motion between events | `uavta.dynamics.kinematics.advance_toward` |
| γ = 1/t_endu (Eq. 11) | `uavta.dynamics.energy.consumption_rate` |
| gᵢ = E − γ·duration (Eq. 11) | `uavta.dynamics.energy.residual_energy`, `is_energy_feasible` |
| η fusion (Eq. 5) | `uavta.dynamics.sensing.sensing_capability` |
| θ and segment survival (Eq. 6) | `uavta.dynamics.survival.segment_survival` |
| Cumulative survival (Eq. 7) | `uavta.dynamics.survival.cumulative_survival` |
| Timeliness decay (Eq. 4) | `uavta.dynamics.value.decay_factor` |
| Collected value term (Eq. 4) | `uavta.dynamics.value.expected_collected_value` |
| f₁, f₂, gᵢ of a full plan (Eqs. 4, 8, 11) | `uavta.dynamics.objectives.evaluate_plan` |

`evaluate_plan` exists for the optional RH-RK baseline and for cross-checking the simulator; the RL environment itself realizes value event by event (§5).

---

## 4. Resolution of gaps and inconsistencies

| ID | Resolution |
|---|---|
| G1 | Δt in Eq. 4 = t_arrival − t_ref,j (J1). In `evaluate_plan` the reference can be switched to the planning time t (`decay_reference="plan"`) to obtain the paper-literal planning objective. |
| G2 | p_surv cumulative from mission start in the simulator (J1). `evaluate_plan` accepts an initial survival per UAV; passing 1.0 gives the paper-literal Eq. 7. |
| G3 | ω⁰ = (1, 1, 1). |
| G4 | Not applicable: we do not select from a Pareto set. Reward terms are normalized (phase1 §E). |
| G5, G6 | Not applicable: MDCSATA is not reproduced. |
| G7 | J2: no voluntary re-tasking. Crash → lock released, observation in progress yields nothing. Target move → in-transit UAV is redirected from its current position (aborted if no longer energy-feasible); an observing UAV is interrupted, redirected, and its observation timer restarts. |
| G8 | J5: idle UAVs consume energy. |
| G9 | Termination rules in phase1 §G.2; truncation at T_max. |
| G10 | Not applicable. |
| I1 | J6: scheduled weather values are absolute (may increase). Stochastic weather is multiplicative and non-increasing, following the paper's text. |
| I2 | Events 12 (S2) and 13 (S3) both move a target to (1206, 6405, 129). Transcribed as printed; flagged in the scenario file. |
| I3 | Event 10 says "from [0.5, 0.5]" but in S3 the preference is already [0.75, 0.25] after event 3. Transcribed as printed; the simulator applies the `to` value and logs the mismatch instead of raising. |
| I4 | The report cites Table 9 (42 comparisons), not the "53 of 56" sentence. |

---

## 5. Decisions found during implementation (need review)

| ID | Issue | Adopted for now | Alternative |
|---|---|---|---|
| N1 | Under J5 an idle UAV with nothing valid to do keeps hovering; its energy would eventually reach 0, which the paper never models. | The simulator schedules an internal *depletion* at t + Eᵢ/γᵢ for idle UAVs. A depleted UAV becomes non-operational (status `DEPLETED`, distinct from a crash, not counted as a dynamic event). | Idle UAVs with no valid target "land" and stop consuming until the next event. |
| N2 | Energy feasibility (Eq. 11) is defined over a UAV's whole remaining sequence. A dispatch policy commits one task at a time. | Feasibility is checked for the next task only: Eᵢ − γᵢ(d/v + t_obse) > E_min. Since the policy can never commit a second task before finishing the first, this guarantees Eᵢ > 0 at every completion, which is the physical content of Eq. 11. | — |
| N3 | Table 5 gives value changes and moves as "from X to Y". | Scheduled events store both values. On application the simulator compares the current state with `from`; a mismatch is logged as a transcription/consistency warning, and `to` is applied. | Raise an error on mismatch (would break S3 event 10, see I3). |
| N4 | A scheduled move/value event may hit a target that is already completed (e.g. S3 event 9: T5 is often observed before t = 440 s). | The event is recorded with `effect: no_effect` and changes nothing. | — |
| N5 | Stochastic crashes could remove the last UAV and end every episode early. | Selector-based crashes are skipped (logged `skipped_min_operational`) when they would leave fewer than `min_operational_uavs = 1` UAVs. Named crashes (S1–S3) always apply. | No guard. |
| N6 | Time cost while nothing can be done (all targets finished or infeasible, waiting for possible future events). | R_time accrues only while some operational UAV holds a task. Waiting that no policy can avoid is not penalized. | Accrue until termination. |
| N7 | Termination "no future event could change that" needs knowledge of the schedule. | The simulator ends the episode when no UAV is busy, no decision is possible and no scheduled event remains. This privileged check only affects *when the episode ends*; it never enters the observation. | Run idle until T_max (wastes compute, depletes UAVs). |
| N8 | The Phase 1 target feature "locked by decision UAV" is always 0 (the decision UAV is by definition unassigned). | Replaced by `competitor_arrival`: the earliest time any other operational UAV could reach the target (finishing its own task first). This is current-state information available to the control center. Observation dimension = 6·21 + 24·16 + 17 = 527 (UAV status one-hot now has 5 entries incl. `DEPLETED`). | Keep the useless feature. |
| N9 | Gymnasium's `check_env` samples unmasked actions, which strict mode rejects. | Compliance tests run with `invalid_action_mode: penalize` (invalid action → penalty, state unchanged, logged); a `max_decisions` truncation bounds episode length in that mode. Training of masked policies uses strict mode. | — |
| N10 | The flat-MLP pilot learned slowly (validation J −0.283 → −0.187 in 400k steps), while a shared per-target scorer reached −0.122 in 50k steps. | A pointer-style shared per-target scoring policy (`agents/entity_policy.py`, foreseen in phase1 §D) is evaluated as an architecture ablation. The pre-registered flat MLP remains the primary PPO configuration; both are reported. | MLP only. |
| N11 | RH-RK seeded only with the paper's heuristics started worse than greedy dispatch. | The CSA population is additionally seeded with the plan the greedy dispatcher would produce, so RH-RK starts no worse than greedy on its planning objective. | Paper seeds only. |
| N12 | J is negative for all methods, so a "5 % margin" (J11) needs a sign convention. | H3 non-inferiority: J_PPO ≥ J_RH-RK − 0.05·|J_RH-RK|, fixed before evaluation. | — |

---

## 6. Scenario transcription (S1–S3)

`configs/scenarios/paper.yaml` is a direct transcription of Tables 3, 4 and 5. Each UAV, target and event carries the `scenarios` column from the paper, and the loader filters by the requested variant. Target NT1 and NT2 are listed in Table 4 but only appear through their `new_target` events. The transcription is checked by `tests/unit/test_paper_scenarios.py` against counts stated in §5.1.1 (3/4/5 UAVs, 10/14/18 initial targets) and against spot values.

---

## 7. Random scenario distribution (design choices, not paper values)

| Quantity | Training distribution | Basis |
|---|---|---|
| UAVs | 3–5, type A (p = 0.6) or B (p = 0.4), start in [0,1500]×[0,1500]×[70,80] m | S1–S3 sizes, Table 3 types, start corner |
| Initial targets | 8–18, uniform over the area, z ∈ [75,140] m | §5.1.1, Table 4 |
| Value / observation time | U[50,100] / U[10,20] s | brackets Table 4 (52–99, 10–20 s) |
| Event window | (0, 800] s (S1–S3 events end at 660 s) | Table 5 |
| Rates per 1000 s | new target 3, crash 1.5, sensor 3, weather 3, move 3, value 3 | about the density of S3's 14 events in 660 s |
| Move | 300–1500 m horizontal, ±30 m vertical, clipped | the paper's moves are ≈ 0.7–1.1 km |
| Value change | × U[0.4, 0.7] | the paper's two cases are both ≈ × 0.49 |
| Weather | multiplicative × U[0.5, 1.0] per sensor | non-increasing, per §3.2 text |
| Target cap | Z₀ + new targets ≤ 24 | J9 |

Seed ranges are disjoint: training < 1,000,000 ≤ validation < 2,000,000 ≤ test < 3,000,000 ≤ OOD.

## 8. Simulation, observation and action summary

* **Simulator:** event-driven, continuous time, order at each instant: arrivals → completions → idle depletion → events (type order) → termination/decision check (`environment/simulator.py`).
* **Decision epoch:** whenever an operational unassigned UAV has ≥ 1 valid target; lowest index decides first; several decisions can occur at the same instant.
* **Action:** `Discrete(25)`: target slot 0–23 or WAIT (24). Mask = appeared ∧ unfinished ∧ unlocked ∧ UAV operational ∧ energy-feasible; WAIT only when nothing else is valid. The mask is never all-zero.
* **Observation:** `Box(-1, 1, (527,), float32)`; feature names in `environment/observation.py`; padded slots are zero; target slots are append-only and never reused.
* **Reward:** `environment/reward.py`; the return equals w_value·collected/V_ref − w_time·active time/T_max (unit-tested).

## 9. Ablations not performed

The following Phase 1 options were **not** run, to stay within the compute budget: reward ablation (iv) "no survival factor in the reward"; voluntary WAIT (J3b); in-transit re-tasking (J2b); preference-conditioned policy (J4b). The code supports J3b (`allow_voluntary_wait`), but no results exist for it.

## 10. Extended environment v2 (added 2026-10-27, not part of the paper)

**Why it exists.** In the paper model (v1) the realized value is dominated by travel time: timeliness
decay and flight risk both grow with distance, so "fly to the nearest target" is already close to the
best a dispatcher can do. A decision-sensitivity probe on 30 validation scenarios found that five
hand-written rules (nearest, value ÷ time, cμ-style urgency, sensor scarcity, best-placed UAV) lie
within 1–2 % of each other in that regime, also after enlarging teams, spreading start positions and
adding urgent fast-decaying targets. No dispatcher, learned or not, can then be clearly better than the
nearest-target rule. v2 adds the two things that make allocation decisions matter in a priority-driven
surveillance mission: **priorities with deadlines** and **sensor capabilities**. v1 is unchanged and
remains the paper-comparable environment; all v1 results stay valid (every new option defaults to off
and draws no extra random numbers, which a unit test checks).

| Element | v2 rule | Code |
|---|---|---|
| Critical targets | Each target is critical with probability 0.25: value U[150, 250] (routine U[10, 40]); the observation must **start** within U[200, 700] s of the target's appearance, otherwise it is worth `late_factor` = 0 | `TargetSpec.deadline`, `TargetState.timeliness` |
| Masking | A critical target that the UAV would reach after its deadline is invalid (worthless); a committed task is aborted if an event (target move) makes the deadline unreachable | `Simulator.valid_targets`, `_event_move` |
| Sensor requirements | 25 % of targets can only be observed with infrared, 25 % only with radar (type B carries radar); a requirement no UAV of the scenario carries is never drawn. Capability for such a target is I_m·ω_m of that sensor only (Eq. 5 with one sensor) | `target_sensing_capability`, `Simulator.eta_for`, `can_sense` |
| Re-tasking | `retask_on_event`: at the instant a new target appears or a UAV crashes, every UAV **in transit** gets a re-decision (continue = its own target is a valid action, or switch to another valid target, evaluated from its current position); observing UAVs are never interrupted. The pending re-decisions are cleared as soon as time advances. All methods get the same re-decision points and apply their own rule (RH-RK re-plans with the deciding UAV free at its current position). v1 keeps J2a (no re-tasking) | `Simulator.is_retask_decision`, `assign` |
| Sensor damage | If the damaged sensor is the one a committed target requires, the task is aborted and the target released | `_event_sensor` |
| Scale | 4–7 UAVs, 16–28 initial targets, up to 40 targets; new targets 8 per 1000 s over a 1200 s window; endurance × 0.8 | `configs/train/ppo_entity_hard.yaml` |
| Uniform discounts | λ_obse = 0.0005 1/s, λ_risk = 0.00005 1/m (paper: 0.001, 0.0002) | same |
| Reward / score | collected value / V_ref; the score is the collected value fraction (w_time = 0) | same |
| Observation | 8 additional features per target slot: decay rate, required sensor (3), fraction of operational UAVs able to observe it, has-deadline, time to deadline, slack of the decision UAV (deadline − arrival); 8 × 21 + 40 × 24 + 17 = 1145 numbers | `EXTRA_TARGET_FEATURE_NAMES` |
| Per-target decay | `TargetSpec.decay_rate` (generator option `urgent_fraction`) is implemented and tested but **not used** in v2: urgent fast-decaying targets did not make decisions more consequential in the probe | — |
| Metric | `critical_on_time_rate`: critical targets whose observation started before the deadline ÷ critical targets that appeared | `episode_summary` |

All baselines see the same rules: the greedy rules use the simulator's own value and mask, and RH-RK's
plan model includes deadlines, per-target capability and per-target decay.

**Re-tasking was added after the first v2 training run** (seeds 0–2, 400k steps) had
plateaued at the value ÷ time rule's level on validation (35.3–35.6 % vs. 35.5 %); those runs were
discarded, not reported. With re-tasking the validation probe gave nearest 34.0 %, value ÷ time
37.8 %, rollout 39.9 % (40 scenarios).

**Design was chosen on validation scenarios only** (seeds 1,000,000+), with the classic rules and a
rollout policy (each candidate target played out with the value ÷ time rule on an event-free copy) as
probes, before any v2 test-set evaluation. Headroom found on 40 validation scenarios: nearest 33.2 %,
value ÷ time 36.3 %, rollout 38.1 % collected value. In v2 the difference to the priority-blind
nearest rule is therefore expected to be large by construction, and the difference to value-aware
rules small; both are reported.
