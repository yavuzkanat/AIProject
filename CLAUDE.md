ROLE AND PROJECT MISSION

Act as a senior Reinforcement Learning researcher, autonomous systems engineer, Python software architect, and academic research supervisor.

I am developing a university Artificial Intelligence project inspired by the research paper:

“A Multi-Swarm Dynamic Crow Search Algorithm for Multi-UAV Dynamic Task Allocation”

Published in Drones (2026), Volume 10, Issue 6, Article 467.

Official paper:
https://www.mdpi.com/2504-446X/10/6/467

My goal is to investigate whether Deep Reinforcement Learning can learn effective policies for dynamic multi-UAV task allocation and how its performance compares with conventional heuristic approaches.

This is an academic research project, not merely a programming exercise.

I want a scientifically defensible, reproducible, modular, well-tested, and extensible Python implementation.

Do not assume that Reinforcement Learning will outperform the paper’s algorithm. Treat this as an empirical research question and design experiments that could support or reject that hypothesis.

Do not immediately generate the entire project. First analyze the paper, formulate the problem, identify design decisions, and propose an implementation architecture. Wait for my approval before beginning the implementation.

1. RESEARCH PAPER ANALYSIS

Read the complete paper, including its mathematical formulation, tables, figures, algorithm descriptions, experimental setup, evaluation metrics, and conclusions.

Use the official paper as the primary source. Do not rely only on its abstract.

Produce a structured technical analysis covering:

1. The real-world problem being solved.
2. The assumptions made by the authors.
3. The UAV and target representations.
4. The original objective functions and constraints.
5. The mathematical formulation of mission makespan and residual target value.
6. Energy feasibility and task allocation constraints.
7. The random-key task allocation representation.
8. The purpose of each of the five MDCSATA strategies.
9. The dynamic events modeled in the paper.
10. The benchmark scenarios, including their UAV counts, target counts, and relevant environmental settings.
11. The baseline algorithms and evaluation metrics.
12. The original method’s strengths, limitations, and assumptions.
13. Which aspects should be preserved in my RL environment.
14. Which aspects can reasonably be simplified in a university project.
15. Which simplifications could invalidate a comparison with the original paper.

Clearly distinguish information explicitly stated in the paper from your own design recommendations.

Never invent missing parameters, equations, experimental results, or implementation details.

If the paper does not provide sufficient information to reproduce a component exactly, identify the missing information and propose a transparent alternative.

2. RESEARCH QUESTION AND CONTRIBUTION

Help me formulate a precise research question.

The central research question should be similar to:

“Can a trained Deep Reinforcement Learning policy provide effective and computationally efficient decisions for dynamic multi-UAV task allocation under changing mission conditions, compared with simple heuristic baselines?”

Refine this question to make it measurable.

Identify:

* The independent variables.
* The dependent variables.
* The experimental hypotheses.
* The expected scientific contribution.
* The limits of what can be claimed.
* The difference between reproducing the original problem and proposing a new solution.

Do not describe the project as novel merely because it replaces an optimization algorithm with PPO.

Instead, identify a specific, measurable contribution, such as evaluating policy adaptation under different classes of dynamic events or studying the trade-off between task value, makespan, and energy consumption.

Assess whether the proposed scope is realistic for a university AI course project.

3. MATHEMATICAL FORMULATION

Before writing code, formulate the environment as a Markov Decision Process.

Define every variable, its dimensions, units, valid ranges, and interpretation.

3.1 State and observation

Design a state representation that includes the information necessary for task allocation.

Consider:

* UAV positions in three-dimensional space.
* UAV remaining energy or battery levels.
* UAV capabilities and sensor effectiveness.
* UAV operational status.
* Current destinations and ongoing tasks.
* Target positions.
* Target observation values and time-dependent value decay.
* Target completion status.
* Target deadlines, if applicable.
* Weather or environmental conditions.
* Threat or risk information.
* Current simulation time.
* Relevant information about dynamic events.

Avoid including information that would be unavailable to the controller in a realistic scenario.

Distinguish the complete simulator state from the observation available to the RL agent.

Explain the Markov property and whether the proposed observation satisfies it.

If the observation is only partially observable, explain the implications.

Because Stable-Baselines3 generally expects fixed-shape observations, propose a representation that supports changing numbers of targets and UAVs.

Consider configurable maximum entity counts, padding, feature masks, and normalized features.

Specify the observation-space shape and the meaning of every feature.

Explain how newly appearing targets enter the observation without changing the neural network input dimensions.

3.2 Action space

Compare at least three action-space designs:

A. Discrete UAV-target assignment.

B. Factorized or hierarchical assignment decisions.

C. A discrete action space with invalid-action masking, where supported by the selected algorithm and implementation.

Discuss the advantages, limitations, computational complexity, and compatibility of each option with PPO and Stable-Baselines3.

Recommend one approach for the initial implementation and justify your choice.

The selected action representation must handle:

* Idle UAVs.
* Already completed targets.
* Targets currently being observed.
* Failed or unavailable UAVs.
* Insufficient remaining energy.
* Duplicate assignments.
* Newly appearing targets.
* No available valid action.

Do not silently discard invalid actions or allow multiple UAVs to perform the same exclusive task.

Explain exactly what happens after an action is selected.

Distinguish task allocation from low-level flight control.

The initial project should focus on allocation and simplified movement simulation, not realistic aircraft stabilization or motor control.

3.3 Transition dynamics

Define how the environment changes after each action.

Specify:

* UAV travel between locations.
* Flight duration.
* Energy consumption.
* Target observation duration.
* Target value decay.
* Task completion.
* UAV failure.
* Sensor degradation.
* Weather changes.
* Target movement.
* New target arrivals.
* Mission termination.

Use a clearly defined discrete-time or event-driven simulation model.

Avoid mixing incompatible time conventions.

For example, if a UAV travels for several simulation steps before reaching its destination, define whether it can receive a new task while in transit.

Model the continuity of ongoing tasks explicitly.

When a dynamic event occurs, preserve the actual execution state rather than resetting the entire scenario.

3.4 Reward function

Propose a mathematically explicit reward function.

Consider the following terms:

* Reward for successful target observation.
* Target value or urgency.
* Time and delay costs.
* Energy consumption.
* Risk exposure.
* Collision penalties, if collision dynamics are modeled.
* Invalid-action penalties.
* Deadline violations.
* Mission failure.

Explain the purpose, sign, scale, and units of each term.

Avoid arbitrary reward constants without justification.

Explain reward scaling, normalization, and the possibility of reward hacking.

Investigate whether a weighted scalar reward is appropriate for PPO and how the weights affect the trade-off between target value, makespan, energy, and risk.

Keep the original paper’s two main objectives conceptually identifiable even if the RL agent receives a scalar reward.

Define separate evaluation metrics so that training reward is not mistaken for overall mission performance.

Consider reward ablation experiments to determine which reward components actually help.

Do not claim that a reward penalty alone guarantees hard safety constraints.

If safety constraints are included, explain whether they are enforced by the simulator, action filtering, or a constrained optimization method.

4. DYNAMIC EVENT MODEL

The environment must support configurable dynamic events.

At minimum, consider:

1. New target arrival.
2. UAV failure.
3. Sensor degradation.
4. Weather deterioration.
5. Target movement.
6. Target value changes.

Decision-maker preference changes may be included as an optional extension.

For each event, define:

* Trigger conditions.
* Probability or scheduled occurrence.
* Event severity.
* Changes to the simulator state.
* Consequences for ongoing tasks.
* Whether the agent receives an explicit event indicator.
* Whether the policy can respond at the next decision step.
* How the event is recorded for evaluation.

Support both scheduled and stochastic event generation.

Avoid making every event occur at a fixed step such as step 50.

Such events may be useful for debugging, but the training and evaluation setup must include varied event times and scenarios.

Implement configurable event schedules and random seeds.

Explain how the environment avoids accidentally leaking future event information to the agent.

5. SIMULATION DESIGN

Use a lightweight Python simulation rather than a high-fidelity physical flight simulator for the first version.

The simulation should model task allocation, simplified travel, target observation, energy consumption, and dynamic disruptions.

Represent UAVs and targets with clear data structures.

Use explicit units for distance, speed, time, energy, and target value.

The model should be internally consistent and deterministic when given the same seed and action sequence.

Use seeded random-number generators rather than uncontrolled global randomness.

Keep simulation logic separate from neural network and training logic.

Design the environment so that the number of UAVs, targets, event frequencies, and scenario difficulty can be configured without modifying the core source code.

Provide small, deterministic test scenarios before scaling up to the scenarios inspired by the paper.

6. PYTHON ARCHITECTURE

Propose a clean, modular repository structure before implementing the project.

Prefer a structure conceptually similar to:

project/
README.md
pyproject.toml
configs/
src/
environment/
entities/
dynamics/
events/
agents/
baselines/
training/
evaluation/
visualization/
utils/
tests/
scripts/
results/

You may improve this structure if you have a strong technical reason.

Do not put the entire project into one Python file.

Separate:

* UAV and target data models.
* Environment and transition logic.
* Dynamic event generation.
* Observation construction.
* Action validation and masking.
* Reward computation.
* RL policy and training configuration.
* Baseline algorithms.
* Evaluation.
* Visualization.
* Configuration and reproducibility utilities.

Avoid unnecessary abstraction, excessive inheritance, and premature optimization.

Use type hints, meaningful names, docstrings, and focused functions.

Follow current Python conventions.

Choose Python dependencies based on verified compatibility rather than automatically selecting the newest version.

7. CUSTOM GYMNASIUM ENVIRONMENT

Design a custom Gymnasium environment named MultiUAVDynamicEnv.

It should follow the Gymnasium API correctly.

The design must explicitly address:

* init
* reset
* step
* observation construction
* action validation
* reward computation
* dynamic event processing
* termination conditions
* truncation conditions
* info dictionaries
* seeded randomness
* environment rendering or visualization, if appropriate

Use appropriate Gymnasium spaces.

Explain how terminated and truncated differ.

Ensure reset(seed=…) behaves correctly.

Include informative metrics in the info dictionary without exposing privileged information to the policy.

Design the environment to pass Gymnasium’s environment checker and appropriate unit tests.

Before implementing it, describe the state-transition sequence in step() in precise order.

Avoid ambiguous ordering such as completing tasks before applying an event in one situation and after applying it in another without justification.

8. REINFORCEMENT LEARNING ALGORITHM

Evaluate whether PPO is appropriate for this problem.

Compare PPO briefly with at least one alternative relevant to discrete or structured assignment decisions.

Use PPO for the first implementation unless your analysis identifies a strong technical reason not to.

Explain:

* The policy network architecture.
* The observation encoder.
* The action distribution.
* How padded entities are represented.
* How action masks are handled, if applicable.
* The value function.
* The optimization objective.
* The role of the discount factor and generalized advantage estimation.
* Entropy regularization.
* Training stability and convergence diagnostics.

Do not claim that PPO automatically supports variable-length entity sets or arbitrary action masks.

Verify the compatibility of the selected algorithm, libraries, and action representation.

If the selected action-mask implementation requires a compatible third-party library, identify that dependency and its limitations.

If fixed-size padded observations are used, explain the trade-offs compared with entity-based encoders or attention-based architectures.

Start with the simplest architecture that can adequately represent the problem.

Do not introduce a Transformer, graph neural network, or multi-agent architecture unless there is a clear, testable reason to do so.

9. APPLE SILICON MAC M3 COMPATIBILITY

My development machine uses an Apple Silicon Mac M3.

The project must support PyTorch’s Metal Performance Shaders backend where available.

However, do not assume that MPS is always faster than CPU for small workloads or that every operation is supported.

Explain the practical limitations of MPS for this project.

Provide a device-selection strategy that:

* Detects whether MPS is available.
* Falls back gracefully to CPU.
* Logs the selected device.
* Avoids claiming GPU acceleration when the actual workload runs on CPU.

Check the compatibility of the proposed PyTorch and reinforcement-learning stack with Apple Silicon.

Distinguish the device used for neural network training from the device used for environment simulation.

Do not force every simulation operation onto the GPU.

Stable-Baselines3 uses PyTorch, but its support for different devices depends on the model and operations involved. Verify the relevant implementation details rather than assuming that every configuration works on MPS.

Provide tested or clearly qualified installation instructions for macOS.

Do not hard-code a CUDA device into the project.

10. BASELINES AND SCIENTIFIC EVALUATION

Implement at least two baseline methods:

1. Random valid allocation.
2. A deterministic greedy heuristic.

The greedy baseline should have a clearly defined decision rule, such as selecting a feasible target based on travel cost and target value.

Explain its limitations.

If feasible within the project scope, propose an additional optimization baseline.

Clearly distinguish a newly implemented baseline from a faithful reproduction of an algorithm in the paper.

Use the same simulator, event scenarios, and evaluation conventions for fair comparisons whenever possible.

Evaluate the trained policy on scenarios and random seeds that were not used during training.

Include tests across different:

* UAV counts.
* Target counts.
* Event frequencies.
* Event types.
* Target-value distributions.
* Energy constraints.
* Random seeds.

Do not claim generalization merely because the agent performs well on training scenarios.

11. EVALUATION METRICS

Track and report at least:

* Mean episodic return.
* Total target value successfully observed.
* Number and percentage of completed targets.
* Mission makespan.
* Total and per-UAV energy consumption.
* Number of failed missions.
* Number of invalid actions.
* Risk exposure, if modeled.
* Response latency after dynamic events.
* Wall-clock decision latency.
* Training time.
* Performance variation across random seeds.

Define each metric precisely.

Explain which metrics should be minimized and which should be maximized.

Report means and standard deviations across repeated runs.

Where appropriate, use confidence intervals and statistical comparisons.

Avoid comparing a single PPO run against a single baseline run.

Measure decision latency separately from training time.

If the original paper’s MHV and MIGD metrics are not directly applicable to a single-policy scalar-reward RL setup, explain why rather than reporting them incorrectly.

12. VISUALIZATION AND INTERPRETABILITY

Plan useful visualizations, including:

* UAV and target positions.
* Task assignments.
* UAV routes in the simplified simulation.
* Target completion status.
* Dynamic event timelines.
* Training reward curves.
* Mission makespan comparisons.
* Energy consumption comparisons.
* Performance under different event frequencies.

Visualizations should help explain the learned behavior, not merely make the project look attractive.

Keep visualization code separate from the environment’s core logic.

13. TESTING AND SOFTWARE QUALITY

Create a testing plan before writing the complete implementation.

Include unit tests for:

* Environment reset.
* Observation-space validity.
* Action-space validity.
* Energy consumption.
* Target completion.
* Duplicate assignment prevention.
* Dynamic target arrival.
* UAV failure.
* Sensor degradation.
* Task continuity.
* Mission termination.
* Reproducibility under fixed seeds.
* Reward calculation.
* Gymnasium environment compliance.

Include integration tests for short training runs and evaluation episodes.

Test edge cases such as:

* No available targets.
* No operational UAVs.
* All targets completed.
* A new target appearing when all UAVs are busy.
* A UAV failing while a task is in progress.
* An event making a previously valid assignment infeasible.
* An episode reaching its maximum simulation length.

Do not hide failed assertions or silently ignore invalid states.

Use explicit exceptions for programmer errors and well-defined environment behavior for legitimate invalid actions.

14. IMPLEMENTATION WORKFLOW

Work in carefully separated phases.

PHASE 1 — Paper analysis and research question.

PHASE 2 — Mathematical model and assumptions.

PHASE 3 — Action-space and observation-space design.

PHASE 4 — Reward function and dynamic event specification.

PHASE 5 — Repository architecture and dependency compatibility.

PHASE 6 — Minimal environment implementation.

PHASE 7 — Unit tests and Gymnasium compliance.

PHASE 8 — Random and greedy baselines.

PHASE 9 — PPO training implementation.

PHASE 10 — Evaluation, repeated experiments, and visualization.

PHASE 11 — Ablation studies and generalization tests.

PHASE 12 — Academic report and limitations.

At the end of each phase:

1. Explain the design decisions.
2. Identify unresolved issues.
3. List the files to be created or changed.
4. Define objective acceptance criteria.
5. Ask me to review and approve the phase before moving forward.

Do not silently skip phases.

Do not generate every file in a single enormous response.

When implementation begins, produce complete files in manageable groups, explain where each file belongs, and provide the commands required to run the relevant tests.

Do not rewrite unrelated files unnecessarily.

15. DOCUMENTATION AND REPRODUCIBILITY

Provide:

* A clear README.
* Installation instructions.
* A dependency file.
* Configuration examples.
* Training commands.
* Evaluation commands.
* Test commands.
* A reproducibility guide.
* A description of the environment assumptions.
* A list of known limitations.

Record configuration values, random seeds, library versions, model checkpoints, and evaluation results.

Ensure that training and evaluation configurations can be changed without editing core source code.

16. ACADEMIC INTEGRITY AND RESEARCH LIMITATIONS

Do not invent experimental results.

Do not claim the project reproduces MDCSATA unless its relevant algorithmic details and experimental setup have actually been reproduced.

Do not claim that PPO outperforms the original paper unless a valid, comparable experiment supports that claim.

Distinguish simulation results from evidence about real-world UAV performance.

Discuss the limitations of simplified movement, Euclidean distance, battery models, sensor models, and event simulation.

Include a clear discussion of reward design, distribution shift, generalization, and safety.

Help me explain the work accurately in a university project report and presentation.

17. FIRST RESPONSE: DO NOT WRITE CODE YET

For your first response, deliver only the following:

A. A detailed technical analysis of the paper.

B. A proposed RL problem formulation.

C. A comparison of at least three candidate action-space designs.

D. A proposed observation-space schema.

E. A mathematically explicit candidate reward function, with all terms explained.

F. A dynamic event model.

G. A proposed software architecture.

H. A compatibility assessment for the proposed Python, Gymnasium, PyTorch, and PPO stack on Apple Silicon M3.

I. An experimental plan with baselines, metrics, and generalization tests.

J. A list of unresolved decisions that require my approval.

K. A prioritized implementation roadmap.

For every major design decision, explain the alternatives, trade-offs, and your recommendation.

Do not start generating implementation files until I approve the design.

Your priority is correctness, scientific rigor, reproducibility, and a clear learning path—not maximum code volume.

Remember: the central objective is to build an experimentally defensible RL solution to dynamic multi-UAV task allocation, not merely to generate a large amount of Python code.