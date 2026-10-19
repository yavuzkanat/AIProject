"""Training run directories: completed runs are kept, partial runs restart cleanly."""

import os

import pytest

from uavta.training.train import LOCK_FILE, RunInProgressError, prepare_run_dir


def test_new_run(tmp_path):
    d = tmp_path / "r_seed0"
    assert prepare_run_dir(d, force=False)
    assert (d / LOCK_FILE).read_text() == str(os.getpid())


def test_completed_run_is_kept(tmp_path):
    d = tmp_path / "r_seed0"
    d.mkdir()
    (d / "training_summary.json").write_text("{}")
    (d / "best_model.zip").write_text("model")
    assert not prepare_run_dir(d, force=False)
    assert (d / "best_model.zip").exists()


def test_force_and_partial_runs_are_cleaned(tmp_path):
    d = tmp_path / "r_seed0"
    d.mkdir()
    (d / "validation.csv").write_text("old rows")
    assert prepare_run_dir(d, force=False)
    assert not (d / "validation.csv").exists()
    (d / "training_summary.json").write_text("{}")
    assert prepare_run_dir(d, force=True)
    assert not (d / "training_summary.json").exists()


def test_live_lock_refused_and_stale_lock_ignored(tmp_path):
    d = tmp_path / "r_seed0"
    d.mkdir()
    (d / LOCK_FILE).write_text(str(os.getppid()))  # a live process that is not us
    with pytest.raises(RunInProgressError):
        prepare_run_dir(d, force=False)
    (d / LOCK_FILE).write_text("999999")  # no such process
    assert prepare_run_dir(d, force=False)
