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
