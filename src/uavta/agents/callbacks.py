"""Training callbacks: periodic evaluation on the validation split with the true metrics."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from uavta.agents.ppo_factory import PPOPolicy
from uavta.environment.env import EnvConfig, MultiUAVDynamicEnv
from uavta.evaluation.runner import ScenarioRef, evaluate_policy


class ValidationCallback(BaseCallback):
    """Every ``eval_freq`` environment steps, evaluate the deterministic policy on fixed
    validation scenarios, log mission metrics, and keep the checkpoint with the best mean
    weighted score J (checkpoint selection uses validation data only)."""

    def __init__(
        self,
        env_config: EnvConfig,
        refs: list[ScenarioRef],
        out_dir: Path,
        eval_freq: int,
        use_masking: bool = True,
    ) -> None:
        super().__init__(verbose=0)
        self.env = MultiUAVDynamicEnv(env_config)
        self.refs = refs
        self.out_dir = Path(out_dir)
        self.eval_freq = eval_freq
        self.use_masking = use_masking
        self.best_score = -float("inf")
        self.best_step = 0
        self._last_eval = 0
        self._csv_path = self.out_dir / "validation.csv"
        self._t0 = time.perf_counter()

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path.unlink(missing_ok=True)  # never append to a previous run's log
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last_eval:
            self._evaluate()

    def _evaluate(self) -> None:
        self._last_eval = self.num_timesteps
        policy = PPOPolicy(self.model, use_masking=self.use_masking)
        df = evaluate_policy(self.env, policy, self.refs, method="ppo")
        row: dict[str, Any] = {
            "timesteps": self.num_timesteps,
            "wall_time_s": time.perf_counter() - self._t0,
            "weighted_score": df["weighted_score"].mean(),
            "value_fraction": df["value_fraction"].mean(),
            "makespan": df["makespan"].mean(),
            "completion_rate": df["completion_rate"].mean(),
            "episode_return": df["episode_return"].mean(),
            "mask_overrides": policy.mask_overrides,
        }
        for key in ("weighted_score", "value_fraction", "makespan", "completion_rate"):
            self.logger.record(f"validation/{key}", row[key])
        new = not self._csv_path.exists()
        with self._csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        if row["weighted_score"] > self.best_score:
            self.best_score = row["weighted_score"]
            self.best_step = self.num_timesteps
            self.model.save(self.out_dir / "best_model.zip")
