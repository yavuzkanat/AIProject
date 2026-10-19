"""Dashboard data layer reads run directories safely (including half-written CSVs)."""

import pytest

from uavta.visualization.dashboard import DashboardData, read_csv_rows, split_name


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(DashboardData, "_compute_baselines", lambda self: None)
    run = tmp_path / "runs" / "ppo_entity_seed3"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("total_timesteps: 1000\nppo: {policy: entity}\n")
    (run / "progress.csv").write_text(
        "time/total_timesteps,time/fps,train/approx_kl\n500,100,0.01\n600,"  # truncated last line
    )
    (run / "validation.csv").write_text(
        "timesteps,weighted_score,value_fraction,makespan,completion_rate\n"
        "0,-0.3,0.1,1600,0.9\n400,-0.15,0.2,1200,0.95\n"
    )
    return DashboardData(tmp_path / "runs", cache_dir=tmp_path / "cache")


def test_state(data):
    (r,) = data.state()["runs"]
    assert r["group"] == "ppo_entity" and r["seed"] == 3 and r["policy"] == "entity"
    assert r["timesteps"] == 500 and r["status"] == "running"
    assert r["best_validation_score"] == -0.15 and r["best_validation_step"] == 400
    assert r["eta_s"] == pytest.approx(5.0)


def test_series(data):
    s = data.run_series("ppo_entity_seed3")
    assert s["progress"]["train/approx_kl"] == [0.01]
    assert s["validation"]["weighted_score"] == [-0.3, -0.15]


def test_rejects_paths_outside_runs(data):
    with pytest.raises(FileNotFoundError):
        data.run_series("../cache")


def test_greedy_replay(data):
    rep = data.replay("", "S1", "greedy")
    assert rep["metrics"]["n_targets_appeared"] == 11 and rep["activities"]


def test_helpers(tmp_path):
    assert split_name("abl_w25_seed2") == ("abl_w25", 2)
    assert read_csv_rows(tmp_path / "missing.csv") == []


def test_generated_presets_are_deterministic_and_differ_by_seed(data):
    ids = [p["id"] for p in data.presets()]
    assert {"paper_S3", "calm", "crisis", "large"} <= set(ids)
    a = data.replay("", "crisis", "greedy_nearest", seed=1)
    b = data.replay("", "crisis", "greedy_nearest", seed=1)
    c = data.replay("", "crisis", "greedy_nearest", seed=2)
    assert a is b  # cached
    assert a["targets"] != c["targets"]
    assert all({"endurance", "speed", "sensors"} <= set(u) for u in a["uavs"])


def test_calm_preset_has_no_events(data):
    assert data.replay("", "calm", "rh_rk")["events"] == []


def test_unknown_preset_and_model(data):
    with pytest.raises(ValueError):
        data.replay("", "nope", "greedy")
    with pytest.raises(ValueError):
        data.replay("", "calm", "nope")


def test_results_missing_is_reported(data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert data.results() == {"status": "missing"}


def test_win_counts_by_metric(tmp_path, monkeypatch):
    from uavta.visualization.dashboard import DashboardData

    (tmp_path / "results/eval").mkdir(parents=True)
    rows = ["condition,method,scenario_ref,train_seed,weighted_score,value_fraction"]
    # s1: AI better on both; s2: AI worse on J but tied on information; s3: AI worse on both
    for scen, ai_j, ai_v, rule_j, rule_v in (
        ("s1", -0.1, 0.30, -0.2, 0.20),
        ("s2", -0.3, 0.25, -0.2, 0.2505),
        ("s3", -0.4, 0.10, -0.1, 0.30),
    ):
        rows.append(f"test_id,ppo_entity,{scen},0,{ai_j},{ai_v}")
        rows.append(f"test_id,greedy_nearest,{scen},-1,{rule_j},{rule_v}")
    (tmp_path / "results/eval/main_x.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.chdir(tmp_path)
    j = DashboardData._win_counts()
    assert (j["wins"], j["ties"], j["losses"]) == (1, 0, 2)
    v = DashboardData._win_counts(metric="value_fraction", tie_tol=0.001)
    assert (v["wins"], v["ties"], v["losses"]) == (1, 1, 1)
