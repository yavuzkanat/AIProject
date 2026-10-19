# Deep RL for Dynamic Multi-UAV Task Allocation

A university AI project that asks whether a masked-PPO dispatch policy can allocate
observation tasks to a heterogeneous UAV team under dynamic events as well as heuristic and
re-planning baselines, using the problem model of:

> G. Li, Y. Liu, Q. Zheng, K. Liu, "A Multi-Swarm Dynamic Crow Search Algorithm for
> Multi-UAV Dynamic Task Allocation", *Drones* 2026, 10(6), 467.
> <https://www.mdpi.com/2504-446X/10/6/467>

**This project does not reproduce MDCSATA.** It re-implements the paper's task-value,
survival, makespan and energy model (Eqs. 4–11) and its three benchmark scenarios (Tables 3–5)
as an event-driven simulator, and studies a different solver: a learned sequential dispatch
policy. MHV/MIGD are not reported because a single dispatch policy does not produce a Pareto
front (see `docs/phase1_design.md` §I.3).

| Document | Content |
|---|---|
| `docs/phase1_design.md` | paper analysis, MDP formulation, design alternatives |
| `docs/assumptions.md` | every interpretation of the paper and every implementation decision |
| `docs/report.md` | experimental results, hypotheses, limitations |
| `results/tables/summary.md` | all result tables (generated) |

## Installation (macOS, Apple Silicon; also works on Linux CPU)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[rl,analysis,dev]"
```

`requirements-lock.txt` records the exact versions of the verified environment (Python 3.12,
torch 2.14.1, gymnasium 1.4.0, stable-baselines3 / sb3-contrib 2.9.0). Neural-network training
defaults to the **CPU**: for these small networks MPS was measured to be ~15× slower, and
SB3's `device="auto"` never selects MPS anyway. `device: mps` can be set in a training config;
the resolved device is written to each run's `versions.json`. The simulator always runs on CPU.

## Quick start

```bash
pytest                                   # 230+ unit and integration tests
python scripts/check_env.py              # Gymnasium + SB3 environment checks, one paper episode

# Train (one seed ≈ 10–15 min on an M3 CPU for 2M steps)
python scripts/train.py --config configs/train/ppo_default.yaml --seed 0
python scripts/train.py --config configs/train/ppo_entity.yaml  --seed 0
./scripts/run_all_training.sh           # every run of the study; safe to re-run after an
                                         # interruption: finished runs are skipped, partial
                                         # runs restart cleanly (--force on train.py retrains)

# Dashboard: http://127.0.0.1:8765/ is the end-user page ("Drone Yarışı": AI vs. the
# nearest-target rule on 20 scenarios, plain-language narration and results);
# http://127.0.0.1:8765/panel is the technical panel (all methods, training curves,
# PPO diagnostics). Link to a moment: /?scenario=paper_S3&t=420
python scripts/dashboard.py

# Evaluate all runs and baselines on identical scenarios, then analyse
python scripts/evaluate.py --spec configs/eval/main.yaml
python scripts/evaluate.py --spec configs/eval/generalization.yaml
python scripts/evaluate.py --spec configs/eval/ablations.yaml
python scripts/analyze.py
python scripts/make_figures.py
```

Any training option can be overridden without editing code:
`--set '{"env": {"reward": {"w_value": 0.75, "w_time": 0.25}}, "ppo": {"ent_coef": 0.0}}'`.

## Reproducibility

* Every random quantity comes from a seeded `numpy.random.Generator`; scenario layout and event
  schedule use independent child streams of the scenario seed (`utils/seeding.py`).
* Stochastic events carry pre-drawn selectors, so all methods evaluated on a seed face the
  identical sequence of disruptions.
* Seed ranges are disjoint: training < 1e6 ≤ validation < 2e6 ≤ test < 3e6 ≤ OOD.
  Checkpoints are selected on validation only (`best_model.zip`).
* Each run directory stores `config.yaml`, `versions.json`, `progress.csv`, `validation.csv`,
  `training_summary.json`, `best_model.zip`, `final_model.zip`.
* The same seed and action sequence always give bit-identical simulator output (tested).

## Layout

```text
configs/scenarios/paper.yaml   Tables 3–5 of the paper (S1/S2/S3 variants)
configs/train/                 PPO configurations (default MLP, entity scorer)
configs/eval/                  evaluation conditions (test, paper, generalization, ablations)
src/uavta/entities/            UAV and target data models
src/uavta/dynamics/            pure functions for Eqs. 4–11 (+ plan-level objectives)
src/uavta/events/              event types
src/uavta/scenarios/           YAML schema/loader, random scenario generator
src/uavta/environment/         simulator (no Gym), observation, masking, reward, Gym env
src/uavta/baselines/           random, greedy value-rate, greedy nearest, RH-RK re-planner
src/uavta/agents/              MaskablePPO factory, shared per-target scoring policy, callbacks
src/uavta/training/            training entry point
src/uavta/evaluation/          episode runner, metrics, statistics, experiment driver, tables
src/uavta/visualization/       Gantt charts, route maps, curves, comparison plots
tests/                         unit and integration tests
scripts/                       command-line entry points
results/                       runs, evaluation CSVs, tables, figures
```

## Environment in one paragraph

`MultiUAVDynamicEnv` (Gymnasium id `MultiUAVDynamic-v0`) is an event-driven semi-MDP. Each step
assigns one target slot (or WAIT) to the *decision UAV*; continuous time then advances to the
next decision epoch (a UAV becomes free, or an event frees/changes a task). Observations are a
fixed 527-dimensional vector with padded UAV (6) and target (24) slots; new targets take the
next free slot. Invalid actions (completed, locked or energy-infeasible targets) are removed by
an action mask used by `sb3_contrib.MaskablePPO`. The reward is
`0.5·collected value/V_ref − 0.5·active mission time/T_max`; the two paper objectives are also
logged separately. Six event classes are supported (new target, crash, sensor damage, weather,
target move, value change) plus preference shifts (logged only), both scheduled (S1–S3) and as
seeded Poisson processes.

## Known limitations

See `docs/report.md` §Limitations. In short: point-mass straight-line motion and time-based
energy (as in the paper), a simplified expected-value risk model, simulation-only evidence, a
fixed scalarization of the two objectives, and per-decision (not per-second) discounting.
