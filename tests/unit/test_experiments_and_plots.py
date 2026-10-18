"""Experiment configuration plumbing and figure generation smoke tests."""

from uavta.baselines.heuristics import GreedyValueRatePolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.evaluation.experiments import Condition, condition_env_config, condition_refs
from uavta.evaluation.report import compare_family, method_summary
from uavta.evaluation.runner import ScenarioRef, evaluate_many, run_episode, seed_refs
from uavta.scenarios.loader import load_paper_scenario


def test_condition_env_config_scaling_and_single_class():
    cond = Condition(
        name="c",
        scenarios={"seed_base": 5, "count": 2},
        event_rate_scale=2.0,
        event_class_only="uav_crash",
        generator={"n_uavs": [6, 6]},
    )
    cfg = condition_env_config({}, cond)
    rates = cfg.scenario.generator.event_rates
    assert rates.uav_crash == 3.0 and rates.new_target == 0.0
    assert cfg.scenario.generator.n_uavs == (6, 6)
    assert [r.seed for r in condition_refs(cond)] == [5, 6]


def test_paper_condition_refs():
    refs = condition_refs(Condition(name="p", scenarios={"paper": "S2"}))
    assert refs[0].scenario.name == "paper_S2"


def test_summary_and_comparisons():
    from uavta.baselines.heuristics import GreedyNearestPolicy

    env = MultiUAVDynamicEnv()
    df = evaluate_many(
        env,
        [
            ("a", GreedyValueRatePolicy(), {"train_seed": -1}),
            ("b", GreedyNearestPolicy(), {"train_seed": -1}),
        ],
        seed_refs(10, 6),
    )
    df["condition"] = "c"
    s = method_summary(df, "c", ["weighted_score"])
    assert set(s["method"]) == {"a", "b"}
    comp = compare_family(df, [("c", "a", "b", "weighted_score")])
    assert comp.loc[0, "n"] == 6 and 0 <= comp.loc[0, "p_holm"] <= 1


def test_plots_write_files(tmp_path):
    from uavta.visualization import plots

    sc = load_paper_scenario("S1")
    rec = run_episode(
        MultiUAVDynamicEnv(),
        GreedyValueRatePolicy(),
        ScenarioRef("S1", scenario=sc),
        keep_logs=True,
    )
    assert plots.gantt(rec, [u.id for u in sc.uavs], "g", tmp_path / "g.png").exists()
    assert plots.route_map(rec, sc.airspace, "m", tmp_path / "m.png").exists()
    assert plots.value_timeline({"greedy": rec}, "v", tmp_path / "v.png").exists()
