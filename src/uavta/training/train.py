"""Train a (Maskable)PPO dispatch policy from a YAML configuration."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from uavta.agents.ppo_factory import PPOConfig, make_model
from uavta.environment.env import EnvConfig, MultiUAVDynamicEnv
from uavta.evaluation.runner import seed_refs
from uavta.scenarios.generator import VALIDATION_SEED_BASE
from uavta.utils.config import deep_merge, load_yaml, save_yaml
from uavta.utils.device import DeviceRequest, resolve_device
from uavta.utils.provenance import write_versions
from uavta.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    """Everything needed to reproduce one training run."""

    model_config = ConfigDict(extra="forbid")

    name: str = "ppo_default"
    seed: int = 0
    total_timesteps: int = Field(2_000_000, gt=0)
    n_envs: int = Field(8, gt=0)
    vec_env: str = "subproc"  # "subproc" | "dummy"
    device: DeviceRequest = "cpu"
    torch_threads: int = Field(1, gt=0)
    eval_freq: int = Field(100_000, gt=0)
    n_validation_scenarios: int = Field(100, gt=0)
    env: EnvConfig = Field(default_factory=EnvConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    data = load_yaml(path)
    if overrides:
        data = deep_merge(data, overrides)
    return TrainConfig.model_validate(data)


def _make_env_fn(env_config: EnvConfig, rank: int, seed: int) -> Any:
    def _init() -> Any:
        from stable_baselines3.common.monitor import Monitor

        env = MultiUAVDynamicEnv(env_config)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


# Files a training run writes; removed before a partial run is restarted from scratch.
RUN_FILES = (
    "progress.csv",
    "validation.csv",
    "best_model.zip",
    "final_model.zip",
    "training_summary.json",
    "config.yaml",
    "versions.json",
)
LOCK_FILE = ".training.lock"


class RunInProgressError(RuntimeError):
    """Another live process is already training this run directory."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prepare_run_dir(run_dir: Path, force: bool) -> bool:
    """Make ``run_dir`` ready for a fresh run. Returns False if the run is complete and kept.

    * Completed run (``training_summary.json`` present) and not ``force``: left untouched.
    * Another live process holds the lock: :class:`RunInProgressError`.
    * Otherwise (new, partial, or forced): stale files are removed so logs never mix runs.
    """
    lock = run_dir / LOCK_FILE
    if lock.exists():
        try:
            pid = int(lock.read_text().strip() or "0")
        except ValueError:
            pid = 0
        if pid and pid != os.getpid() and _pid_alive(pid):
            raise RunInProgressError(f"{run_dir.name} is being trained by process {pid}")
    if (run_dir / "training_summary.json").exists() and not force:
        return False
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (run_dir / name).unlink(missing_ok=True)
    lock.write_text(str(os.getpid()))
    return True


def train(config: TrainConfig, out_root: str | Path = "results/runs", force: bool = False) -> Path:
    """Run training and write model, configuration, provenance and logs to a run directory.

    Completed runs are skipped unless ``force`` is set; interrupted runs restart from scratch
    (SB3 cannot resume a PPO run bit-identically, and a fresh run with the same seed reproduces
    the original exactly on the CPU).
    """
    run_dir = Path(out_root) / f"{config.name}_seed{config.seed}"
    if not prepare_run_dir(run_dir, force):
        logger.info("%s already completed; skipping (use --force to retrain)", run_dir.name)
        return run_dir
    try:
        _train(config, run_dir)
    finally:
        (run_dir / LOCK_FILE).unlink(missing_ok=True)
    return run_dir


def _train(config: TrainConfig, run_dir: Path) -> None:
    import torch
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from uavta.agents.callbacks import ValidationCallback

    seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = resolve_device(config.device)

    fns = [_make_env_fn(config.env, k, config.seed * 1000) for k in range(config.n_envs)]
    vec = (
        SubprocVecEnv(fns, start_method="spawn")
        if config.vec_env == "subproc"
        else DummyVecEnv(fns)
    )
    model = make_model(vec, config.ppo, seed=config.seed, device=device)
    model.set_logger(configure(str(run_dir), ["csv"]))
    save_yaml(config.model_dump(mode="json"), run_dir / "config.yaml")
    write_versions(
        run_dir, {"requested_device": config.device, "resolved_device": str(model.device)}
    )
    logger.info("training %s on %s", run_dir.name, model.device)

    callback = ValidationCallback(
        config.env,
        seed_refs(VALIDATION_SEED_BASE, config.n_validation_scenarios, "val"),
        run_dir,
        eval_freq=config.eval_freq,
        use_masking=config.ppo.use_masking,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    train_time = time.perf_counter() - t0
    model.save(run_dir / "final_model.zip")
    vec.close()
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "train_time_s": train_time,
                "timesteps": int(model.num_timesteps),
                "steps_per_s": model.num_timesteps / train_time,
                "best_validation_score": callback.best_score,
                "best_validation_step": callback.best_step,
                "device": str(model.device),
                "seed": config.seed,
            },
            indent=2,
        )
    )
