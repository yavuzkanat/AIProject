# Deep Reinforcement Learning for Dynamic Multi-UAV Task Allocation — Results Report

**Date:** 2026-10-18 · **Status:** all planned v1 experiments complete; improvement experiments in §11
**Basis:** problem model of Li et al., *Drones* 2026, 10(6), 467 (MDCSATA paper).
Design: `docs/phase1_design.md`. Every interpretation of the paper: `docs/assumptions.md`.
All numbers below are produced by `scripts/analyze.py` / `scripts/make_figures.py` from the CSV
files in `results/eval/` and are reproducible with the commands in §10.

---

## 1. Summary

We re-implemented the task-value, survival, makespan and energy model of Li et al. (Eqs. 4–11)
and their benchmark scenarios S1–S3 (Tables 3–5) as an event-driven Gymnasium environment with
six classes of dynamic events, and trained MaskablePPO dispatch policies that assign the next
observation target to whichever UAV becomes free.

* The **pre-registered flat-MLP PPO policy** (5 seeds × 2M steps) is significantly **worse**
  than every heuristic baseline on 500 held-out scenarios (ΔJ = −0.015 to −0.027 per episode),
  although far better than random (+0.128).
* A **shared per-target scoring policy** ("entity" policy; same PPO, same data, different
  network) is significantly **better** than every baseline: ΔJ = +0.010 vs. the rolling-horizon
  re-planner RH-RK, +0.016 vs. greedy-nearest, +0.022 vs. greedy value-rate (all Holm-adjusted
  p < 1e-10; 5/5 seeds agree). It collects more value (22.96 % vs. 21.74 % for RH-RK) and
  finishes earlier (1125 s vs. 1145 s), and is never significantly worse than any baseline in
  17 out-of-distribution conditions.
* The absolute improvements are **small** (about 1–2 % of the value scale): in this model the
  next-target choice is a problem where simple greedy rules are already strong.
* On the three single paper scenarios the entity policy beats RH-RK on S1 but not on S2/S3;
  the pre-registered non-inferiority test H3 is therefore only met for S1.
* A learned policy decides in 0.19 ms, ~71× faster than RH-RK (13.5 ms) — but greedy rules
  are faster still (0.01–0.04 ms).

* **Follow-up (§11):** decision-time lookahead without future knowledge improves the entity
  policy by a further +0.0045 J (5/5 seeds; RH-RK lead grows to +0.015) at 4.6 ms per decision;
  an attention-based network trained for 1M steps did not improve on the entity scorer.

The entity architecture was added after a pilot run (decision N10) and is therefore an
**exploratory** result, evaluated with the same held-out protocol as the pre-registered model.

---

## 2. Research question and hypotheses (fixed before evaluation)

> In a simulator implementing the model of Li et al. (2026), does a masked-PPO dispatch policy
> trained on randomly generated scenarios achieve realized mission outcomes at least as good as
> greedy dispatch, within a stated margin of a re-planning optimizer, at ≥ 10× lower decision
> time? How does this change across event classes, event rates and scenario sizes?

Primary metric: weighted score **J = 0.5 · collected value fraction − 0.5 · makespan / 2400 s**
(higher is better; it is negative for every method because the makespan term dominates).

| ID | Hypothesis | Test |
|---|---|---|
| H1 | PPO > random | paired Wilcoxon on 500 test scenarios, Holm |
| H2 | PPO ≥ greedy (value-rate and nearest) | same |
| H3 | PPO within 5 % of RH-RK on S1–S3: J_PPO ≥ J_RHRK − 0.05·\|J_RHRK\| (N12) | per scenario, mean over seeds |
| H4 | PPO decision latency ≥ 10× lower than RH-RK | controlled benchmark (idle CPU, 1 thread) |
| H5 | advantage over greedy differs by event class (exploratory) | single-class conditions |
| H6 | advantage shrinks out of distribution (exploratory) | rate / size / energy / value grid |

---

## 3. Environment (v1, "paper model")

* **Entities:** 3–6 heterogeneous UAVs (types A/B of Table 3: speed, endurance, sensors,
  vulnerability) and up to 24 target slots; targets have a value, an observation time and a
  position.
* **Collected value of an observation (Eq. 4):** V · η · p_surv · exp(−λ_obse · Δt), with
  multi-sensor fusion η (Eq. 5), cumulative path-dependent survival (Eqs. 6–7; λ_risk =
  0.0002 m⁻¹) and timeliness decay (λ_obse = 0.001 s⁻¹).
* **Energy (Eq. 11):** time-based, enforced as a hard constraint through the action mask.
* **Events:** UAV crash, sensor damage, weather, target move, value change, new target
  (+ preference shift, logged only). S1–S3 replay the paper's 15 scheduled events exactly;
  training and testing use seeded Poisson processes with pre-drawn selectors so that every
  method faces identical disruptions.
* **Decisions:** event-driven semi-MDP. Whenever an operational, unassigned UAV has a valid
  target, it receives a target slot (Discrete(25) with masking; WAIT only if nothing is valid).
  Committed tasks are not re-tasked voluntarily (J2a); events can force redirects or aborts.
* **Observation:** fixed 527-dim vector (6 UAV slots × 21, 24 target slots × 16, 17 global
  features) — current state only, no future-event information (tested).
* **Reward:** 0.5 · collected value / V_ref − 0.5 · active mission time / 2400 s per step.

Key deviations from the paper (full list in `docs/assumptions.md`): absolute decay and
cumulative survival (J1), idle UAVs consume energy and can deplete (J5, N1), dispatch instead of
whole-plan optimization (the paper's MHV/MIGD are not applicable, phase1 §I.3).

Verification: 243 unit/integration tests, including a hand-computed scenario matching the
closed-form value and makespan to 1e-9, Gymnasium/SB3 environment checkers, determinism, and a
no-leakage test. Retraining a seed reproduces its weights bit-identically on CPU.

---

## 4. Methods compared

| Method | Description |
|---|---|
| random | uniform over valid actions |
| greedy_nearest | closest valid target |
| greedy_value_rate | argmax expected value / (travel + observation time) — inspired by the paper's §4.3 heuristic |
| rh_rk | **our** rolling-horizon re-planner: at every decision, crow search (paper Eq. 2, AP = 0.05, fl = 1.5, 40 × 40 evaluations) over the paper's random-key encoding, seeded with the greedy-dispatch plan and the paper's two heuristics; executes the first planned task. *Not* MDCSATA. |
| ppo | MaskablePPO, MLP [256, 256] tanh actor and critic (pre-registered) |
| ppo_entity | MaskablePPO with one small network applied to every target slot plus a context vector; slot scores are the action logits (permutation-equivariant) |

PPO settings for both networks: lr 3e-4 (linear decay), n_steps 1024 × 8 envs, batch 256,
10 epochs, γ = 0.995, GAE λ = 0.95, clip 0.2, entropy 0.01; 2M environment steps; checkpoint
selected on the validation split (100 scenarios). Training on CPU: MLP 9–15 min per seed,
entity 31–48 min per seed (two queues shared the CPU part of the time).

---

## 5. Experimental protocol

* **Disjoint seed ranges:** training < 1e6 ≤ validation < 2e6 ≤ test < 3e6 ≤ OOD.
* **Test-ID:** 500 unseen scenarios from the training distribution (3–5 UAVs, 8–18 targets,
  all six event classes at the training rates).
* **Test-Paper:** S1, S2, S3 exactly as tabulated (never used for training or selection);
  stochastic baselines repeated 30×.
* **OOD grid:** 17 conditions × 100 scenarios (event rate ×0…×4, single event class at ×3,
  team/target size, endurance ×0.7/×1.3, bimodal/skewed values).
* **Statistics:** PPO evaluated deterministically for each of 5 training seeds; spread =
  std over seeds of the per-seed mean. Paired two-sided Wilcoxon signed-rank tests on
  per-scenario means, Holm correction per family, rank-biserial r, 95 % bootstrap CIs.

---

## 6. Results

### 6.1 Learning curves (validation, 100 scenarios; `results/figures/training_curves_J.png`)

| Steps | MLP J (5 seeds) | Entity J (5 seeds) |
|---|---|---|
| 100k | −0.236 … −0.247 | −0.137 … −0.142 |
| 1M | −0.179 … −0.184 | ≈ −0.135 (plateau) |
| best | −0.164 … −0.172 | −0.129 … −0.133 |

The entity policy reaches its plateau within ~100k steps; the MLP improves slowly and has not
clearly converged at 2M steps (three of five seeds peak in the last 300k steps).

### 6.2 Held-out test set (500 scenarios)

| Method | Seeds | J ↑ | Value fraction ↑ | Makespan (s) ↓ | Completion ↑ | Energy used (Σ) ↓ | Risk exposure ↓ |
|---|---|---|---|---|---|---|---|
| random | – | −0.2846 | 0.1029 | 1613 | 0.909 | 2.372 | 0.853 |
| greedy_value_rate | – | −0.1412 | 0.2200 | 1206 | 0.984 | 1.842 | 0.804 |
| greedy_nearest | – | −0.1360 | 0.2208 | 1183 | 0.994 | 1.828 | 0.802 |
| rh_rk | – | −0.1299 | 0.2174 | 1145 | 0.992 | 1.769 | 0.798 |
| ppo (MLP) | 5 | −0.1565 ± 0.0023 | 0.2066 | 1247 | 0.981 | 1.892 | 0.812 |
| **ppo_entity** | 5 | **−0.1197 ± 0.0013** | **0.2296** | **1125** | 0.994 | **1.752** | **0.792** |

± = std over training seeds. 95 % bootstrap CIs of the means are in `results/tables/summary.md`.

### 6.3 Hypotheses

| ID | Model | Result | Evidence |
|---|---|---|---|
| H1 | MLP | **supported** | +0.128 [+0.122, +0.135] vs. random |
| H1 | entity | **supported** | +0.165 [+0.157, +0.173] |
| H2 | MLP | **rejected** | −0.015 [−0.018, −0.012] vs. value-rate; −0.020 [−0.024, −0.017] vs. nearest (Holm p < 1e-20) |
| H2 | entity | **supported** (superior) | +0.022 [+0.018, +0.024] vs. value-rate; +0.016 [+0.014, +0.019] vs. nearest (Holm p < 1e-25) |
| – | entity vs. RH-RK | entity better | +0.010 [+0.007, +0.013], p = 4.5e-11, r = 0.34 |
| H3 | entity | S1 met, **S2/S3 not met** | see §6.4 |
| H3 | MLP | not met on S1–S3 | see §6.4 |
| H4 | both | **supported** | RH-RK/entity = 71×, RH-RK/MLP = 125× (§6.7) |
| H5 | entity | partly | largest gain vs. RH-RK for new-target (+0.015) and crash (+0.012) events, smallest for weather (+0.0035) and value changes (+0.0055) |
| H6 | entity | mostly not supported | advantage persists in 16/17 OOD conditions vs. RH-RK; shrinks vs. greedy-nearest at high event rates (+0.0085 at ×2, +0.0053 at ×4) and small teams |

### 6.4 Paper scenarios S1–S3 (single deterministic scenarios)

| Scenario | greedy_nearest | greedy_value_rate | rh_rk (30 runs) | ppo (5 seeds) | ppo_entity (5 seeds) | H3 threshold |
|---|---|---|---|---|---|---|
| S1 | −0.0428 | −0.0428 | −0.0527 | −0.0761 | **−0.0470** (4/5 seeds meet) | −0.0553 |
| S2 | −0.0679 | −0.0020 | **−0.0006** | −0.0542 | −0.0252 (1/5) | −0.0006 |
| S3 | **−0.0520** | −0.0548 | −0.0626 | −0.1295 | −0.0740 (2/5) | −0.0657 |

A single scenario is a noisy test: entity seeds differ by up to 0.057 on S2. The paper
scenarios start all UAVs in one corner and contain two crashes; the learned policy is not
better than the best heuristic on S2/S3. This must be reported alongside the test-set result.

### 6.5 Generalization (17 × 100 unseen scenarios; mean J)

| Condition | greedy_nearest | rh_rk | ppo | ppo_entity |
|---|---|---|---|---|
| rate ×0 (no events) | −0.0420 | −0.0260 | −0.0533 | **−0.0237** |
| rate ×1 | −0.1329 | −0.1253 | −0.1518 | **−0.1161** |
| rate ×4 | −0.3442 | −0.3475 | −0.3653 | **−0.3389** |
| only new targets | −0.1238 | −0.1207 | −0.1561 | **−0.1055** |
| only crashes | −0.1474 | −0.1472 | −0.1809 | **−0.1348** |
| small (2 UAVs, 5–7 targets) | −0.1918 | **−0.1831** | −0.2039 | −0.1852 |
| large (6 UAVs, 19–22 targets) | −0.0927 | −0.0813 | −0.1437 | **−0.0740** |
| endurance ×0.7 | −0.1146 | −0.1036 | −0.1278 | **−0.1016** |

Full grid: `results/tables/summary.md`, `results/figures/generalization_delta_*.png`.
After Holm correction over 102 comparisons the entity policy is significantly better than
greedy value-rate in 17/17, greedy-nearest in 13/17 and RH-RK in 6/17 conditions, and
significantly worse in none. The flat MLP degrades most on the larger team size (−0.144 vs.
−0.093 greedy), consistent with its slot-specific weights; the entity network transfers.

### 6.6 Ablations (500 test scenarios, J with default weights; 3 seeds × 1M steps)

| Variant | J | Value fraction | Makespan | vs. reference |
|---|---|---|---|---|
| entity, reward (0.5, 0.5) — reference (2M) | −0.1197 | 0.2296 | 1125 | – |
| value-only reward (1, 0) | −0.1220 | 0.2318 | 1142 | −0.0023 [−0.0040, −0.0006], p = 0.007 |
| reward (0.25, 0.75) | −0.1205 | 0.2276 | 1125 | −0.0009, p = 0.19 |
| reward (0.75, 0.25) | −0.1193 | 0.2312 | 1128 | +0.0004, p = 0.86 |
| + energy term (w_E = 0.5) | −0.1191 | 0.2303 | 1124 | +0.0006, p = 0.32 |
| MLP without action mask (penalty 0.1) | −0.1849 | 0.1899 | 1343 | −0.0284 vs. masked MLP (2M); validation at 1M: −0.197…−0.204 vs. −0.179…−0.184 masked |

* **Masking matters:** at an equal budget of 1M steps the unmasked policy is ~0.020 worse on
  validation and remains worse after evaluation-time masking.
* **Reward weights barely change realized behaviour.** Even a value-only reward yields short
  missions, because value decays with time and survival decays with distance: the value term
  already rewards speed. The (value, makespan) trade-off is therefore nearly flat in this model
  (`results/figures/tradeoff_value_vs_makespan.png`) — a scalarized policy cannot trace a
  meaningful Pareto front here.
* Note that ablations ran for 1M instead of 2M steps; the entity policy plateaus by ~100k steps,
  so this does not affect the entity ablations, but it does affect the unmasked-MLP comparison
  (hence the equal-budget validation comparison above).

### 6.7 Decision latency (controlled benchmark, 50 scenarios, idle Apple M3, 1 CPU thread)

| Method | median ms / decision |
|---|---|
| random | 0.005 |
| greedy_nearest | 0.011 |
| greedy_value_rate | 0.043 |
| ppo (MLP) | 0.108 |
| ppo_entity | 0.190 |
| ppo_attention (§11) | 0.308 |
| ppo_entity + lookahead (§11) | 4.58 (p95 ≈ 10) |
| rh_rk | 13.5 |

Training time is reported separately (§4). Latencies recorded during the main evaluation ran
alongside training jobs and are not used for H4. The benchmark was re-run on 2026-10-19 after the
improvement experiments; the H4 ratios above use that run (the first run gave 69× / 121×).

### 6.8 Qualitative behaviour (`results/figures/`)

`map_S*_*.png`, `gantt_S*_*.png` and `value_timeline_S*.png` show routes, per-UAV schedules and
cumulative value on S1–S3 for every method. The live dashboard (`python scripts/dashboard.py`)
replays any checkpoint on S1–S3.

---

## 7. Discussion

1. **Architecture, not algorithm, decided the outcome.** The same PPO with the same data
   failed with a flat MLP and succeeded with a shared per-target scorer. Dispatch is a ranking
   problem over a variable set of entities; weight sharing across target slots gives 24× more
   training signal per network parameter and invariance to slot order.
2. **Why the gains are small.** With work-conserving dispatch and no voluntary re-tasking the
   only decision is "which target next", for which greedy rules are near-optimal locally. The
   cumulative survival model also compresses all methods into a narrow value range (risk
   exposure ≈ 0.8 for everyone). RL's room to improve is limited by the decision structure,
   not only by the learner. An oracle upper bound (not computed) would quantify the headroom.
3. **Where the learned policy helps most** is after structural events — new targets and
   crashes (H5) — and on larger teams, i.e. where myopic choices interact.
4. **RH-RK is a strong comparator but plans without future events** and re-plans only at
   decision instants; it is best on S2, and the entity policy still beats it on average.

---

## 8. Limitations and threats to validity

* **Post-hoc architecture choice.** The entity policy was added after observing that the MLP
  pilot learned slowly (N10). Held-out test scenarios and OOD conditions were never used for
  design decisions, but the result should be presented as exploratory and replicated before
  being claimed as confirmatory.
* **Not MDCSATA.** Nothing here measures MDCSATA; no claim that PPO outperforms the paper.
  MHV/MIGD are not comparable to our realized-outcome metrics.
* **Simulation only.** Point masses, straight-line flight, time-based energy, no
  communication, collisions, wind or return-to-base (as in the paper). Results say nothing
  about real UAV performance.
* **Model interpretations matter.** Cumulative survival and absolute decay (J1) change the
  scale of the objectives relative to the paper's per-replan formulation.
* **Single paper scenarios are noisy**; conclusions rest on the 500-scenario test set.
* **Unequal budgets:** MLP not fully converged at 2M steps; ablations used 1M steps.
* **Semi-MDP discounting** per decision rather than per second.
* **Not run:** reward ablation without the survival factor, voluntary WAIT (J3b), re-tasking
  (J2b), preference-conditioned policies (J4b) — `docs/assumptions.md` §9.
* **Latency** measured on one machine; absolute values are hardware-dependent.

---

## 9. Relation to the project theme and next steps

The project theme — a swarm surveillance system that reorganizes itself by task priority — is
covered by v1 as *event-triggered reallocation with value-as-priority*. Proposed v2 extensions
(design not yet approved): explicit priority classes with priority-increase events and
deadlines, voluntary re-tasking (preemption) of committed UAVs, priority-specific metrics
(response time to critical targets), an oracle upper bound, and later persistent surveillance
(revisits, age of information). v1 results should be kept as the paper-comparable baseline.

---

## 10. Reproducing these results

```bash
pip install -e ".[rl,analysis,dev]"
pytest
./scripts/run_all_training.sh                       # 25 runs, ~4–6 h on an M3 CPU
python scripts/evaluate.py --spec configs/eval/main.yaml --skip-ppo --output results/eval/main_baselines.csv
python scripts/evaluate.py --spec configs/eval/main.yaml --skip-baselines --ppo ppo --output results/eval/main_ppo.csv
python scripts/evaluate.py --spec configs/eval/main.yaml --skip-baselines --ppo ppo_entity --output results/eval/main_entity.csv
python scripts/evaluate.py --spec configs/eval/generalization.yaml --skip-ppo --output results/eval/generalization_baselines.csv
python scripts/evaluate.py --spec configs/eval/generalization.yaml --skip-baselines --ppo ppo --output results/eval/generalization_ppo.csv
python scripts/evaluate.py --spec configs/eval/generalization.yaml --skip-baselines --ppo ppo_entity --output results/eval/generalization_entity.csv
python scripts/evaluate.py --spec configs/eval/ablations.yaml --ppo abl_value_only abl_w25 abl_w75 abl_energy abl_unmasked
python scripts/benchmark_latency.py --count 50      # on an idle machine
python scripts/analyze.py --main results/eval/main_{baselines,ppo,entity}.csv \
    --gen results/eval/generalization_{baselines,ppo,entity}.csv
# Improvement experiments (§11)
python scripts/train.py --config configs/train/ppo_attention.yaml --seed 0   # … seeds 0-4
python scripts/evaluate.py --spec configs/eval/improvements.yaml --ppo ppo_entity_lookahead --output results/eval/improvements_lookahead.csv
python scripts/evaluate.py --spec configs/eval/improvements.yaml --ppo ppo_attention --output results/eval/improvements_attention.csv
python scripts/analyze_improvements.py
python scripts/make_figures.py --main results/eval/main_{baselines,ppo,entity}.csv \
    --gen results/eval/generalization_{baselines,ppo,entity}.csv
```

Each run directory stores its configuration, library versions, seeds, logs and checkpoints.

---

## 11. Improvement experiments (after the main study)

**Question:** has the learned policy reached the limit of what is achievable? Three indicators
(all from existing results): (i) the entity policy's validation score is flat over its last 1M
steps (slope −0.003 … +0.006 across seeds) and validation ≈ test, so more training of the same
model will not help and it does not overfit; (ii) it beats greedy-nearest in 69 % and RH-RK in
61 % of test scenarios, i.e. it still loses 30–39 %; (iii) choosing, per scenario, the better of
the entity policy and the best classic method would give J = −0.1076 versus −0.1197 — a
**lower bound of 0.012 on the remaining headroom**, larger than the policy's lead over RH-RK.
Two improvements were then tested with the main protocol (5 seeds, the same 500 test scenarios
and S1–S3; separate Holm family; `scripts/analyze_improvements.py`, `results/tables/improvements.md`).

**11.1 Decision-time lookahead (no retraining).** At each decision the policy's top-k actions
are each played out to mission end in a copy of the simulator from which **all future events are
removed** (`Simulator.clone_without_future`; a unit test checks that two episodes with the same
past but different futures produce identical decisions). The action with the best rollout return
is executed. Variants were selected on 50 *validation* scenarios only (seed-0 policy):

| Variant | Validation J | Latency |
|---|---|---|
| no lookahead | −0.1160 | 0.2 ms |
| top-2, event-free rollouts (**selected**) | −0.1149 | 8 ms |
| top-3 / top-5, event-free | −0.1177 / −0.1194 | 11 / 20 ms |
| top-2 / top-3 with N = 4 sampled futures | −0.1228 / −0.1294 | 43 / 61 ms |
| top-2 / top-3 with N = 8 sampled futures | −0.1217 / −0.1181 | 79 / 138 ms |

Sampled futures were drawn from the *known event statistics* of the training distribution (never
the realized future). They made decisions worse; a plausible reason is that noisy rollout
estimates override a policy whose first choice is usually already best ("winner's curse").

**11.2 Attention over UAV and target tokens (retraining).** The entity scorer rates every target
independently given a pooled context; one Transformer layer over UAV and target tokens
(`AttentionExtractor`) lets targets see each other and every UAV. Two layers trained at only
200–320 steps/s on the M3 (CPU or MPS), so one layer, 4 torch threads and **1M steps** per seed
were used (fixed before seeing results; a smaller budget than the entity policy's 2M, which can
only disadvantage this variant).

**11.3 Results on the held-out test set (500 scenarios, 5 seeds, Holm within this family)**

| Method | J | Value fraction | Makespan (s) | vs. entity | vs. RH-RK |
|---|---|---|---|---|---|
| ppo_entity (reference) | −0.1197 | 0.2296 | 1125 | – | +0.0102 |
| **ppo_entity + lookahead** | **−0.1152** | 0.2267 | **1097** | **+0.0045** [+0.0027, +0.0063], p_Holm = 1e-6 | +0.0147 [+0.0119, +0.0176] |
| ppo_attention (1M steps) | −0.1256 | 0.2249 | 1143 | −0.0060 [−0.0074, −0.0045], p_Holm = 1e-14 | +0.0042 [+0.0016, +0.0069], p_Holm = 0.008 |

* **Lookahead helps**, on all 5 seeds (test J −0.114 … −0.117), closing ≈ 37 % of the measured
  headroom. It gains by finishing missions earlier, not by collecting more value. The price is
  latency: 4.6 ms median (≈ 24× the plain policy), so the RH-RK/AI latency ratio drops from 71× to
  ≈ 3× and H4 would **not** hold for this variant. On the single paper scenarios it does not help
  (S1 −0.0624, S2 −0.0290, S3 −0.0765 vs. entity −0.0470 / −0.0252 / −0.0740).
  The 50-scenario validation sample showed only +0.001 — too small to detect the effect; this is
  reported, not used to change the selection.
* **Attention did not help under this budget**: significantly worse than the entity scorer,
  although still significantly better than every classic method. Its validation curves were
  still rising at 1M steps for some seeds, so a longer run might change this; that was not tested.

**Remaining options** (not run): a better critic (explained variance ≈ 0.5) and per-second
discounting; imitation of RH-RK followed by PPO fine-tuning; richer decisions (re-tasking,
priority events — v2), which give a learned policy more room than "which target next".
