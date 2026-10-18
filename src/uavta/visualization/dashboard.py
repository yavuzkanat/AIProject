"""Local live-training dashboard (standard-library HTTP server, read-only).

Serves ``dashboard.html`` and a small JSON API built from the files that training already
writes (``results/runs/<run>/{config.yaml, progress.csv, validation.csv, best_model.zip}``).
It never writes into run directories and never influences training.

Endpoints:
    /api/state                      all runs with progress, speed, ETA, status
    /api/run?name=RUN               time series from progress.csv and validation.csv
    /api/baselines                  baseline metrics on the same validation scenarios
    /api/replay?name=RUN&scenario=S3&model=best|final|greedy
                                    one recorded episode for the animated map
"""

from __future__ import annotations

import csv
import json
import logging
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

logger = logging.getLogger(__name__)

STATIC = Path(__file__).with_name("dashboard.html")
ACTIVE_WINDOW_S = 120.0  # a run is "running" if its logs changed this recently
PROGRESS_KEYS = (
    "time/total_timesteps",
    "time/fps",
    "time/time_elapsed",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/learning_rate",
)
VALIDATION_KEYS = ("timesteps", "weighted_score", "value_fraction", "makespan", "completion_rate")


def _float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV that may be mid-write: malformed or truncated rows are skipped."""
    if not path.exists():
        return []
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return []
    if rows and None in rows[-1].values():
        rows = rows[:-1]  # last line still being written
    return rows


def split_name(run: str) -> tuple[str, int | None]:
    head, sep, tail = run.rpartition("_seed")
    if sep and tail.isdigit():
        return head, int(tail)
    return run, None


class DashboardData:
    """Reads run directories and caches expensive results (baselines, replays)."""

    def __init__(self, runs_dir: Path, cache_dir: Path, n_validation: int = 100) -> None:
        self.runs_dir = runs_dir
        self.cache_dir = cache_dir
        self.n_validation = n_validation
        self._baselines: dict[str, Any] = {"status": "pending"}
        self._replays: dict[tuple[str, str, str, float], dict[str, Any]] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._compute_baselines, daemon=True).start()

    # ---------------------------------------------------------------- runs

    def run_dirs(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted(p for p in self.runs_dir.iterdir() if (p / "config.yaml").exists())

    def state(self) -> dict[str, Any]:
        now = time.time()
        runs = []
        for d in self.run_dirs():
            group, seed = split_name(d.name)
            try:
                cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
            except (OSError, yaml.YAMLError):
                cfg = {}
            total = int(cfg.get("total_timesteps", 0) or 0)
            prog = read_csv_rows(d / "progress.csv")
            val = read_csv_rows(d / "validation.csv")
            last = prog[-1] if prog else {}
            steps = _float(last.get("time/total_timesteps")) or 0.0
            fps = _float(last.get("time/fps"))
            elapsed = _float(last.get("time/time_elapsed"))
            mtimes = [
                p.stat().st_mtime for p in (d / "progress.csv", d / "validation.csv") if p.exists()
            ]
            age = now - max(mtimes) if mtimes else math.inf
            done = (d / "training_summary.json").exists()
            status = "done" if done else ("running" if age < ACTIVE_WINDOW_S else "stopped")
            raw_scores = [_float(r.get("weighted_score")) for r in val]
            scores = [s for s in raw_scores if s is not None]
            best = max(scores) if scores else None
            best_step = None
            if best is not None:
                best_step = next(
                    _float(r["timesteps"]) for r in val if _float(r.get("weighted_score")) == best
                )
            summary: dict[str, Any] = {}
            if done:
                try:
                    summary = json.loads((d / "training_summary.json").read_text())
                except (OSError, json.JSONDecodeError):
                    summary = {}
            remaining = max(total - steps, 0.0)
            runs.append(
                {
                    "name": d.name,
                    "group": group,
                    "seed": seed,
                    "policy": (cfg.get("ppo") or {}).get("policy", "mlp"),
                    "total_timesteps": total,
                    "timesteps": steps,
                    "fps": fps,
                    "elapsed_s": summary.get("train_time_s", elapsed),
                    "eta_s": remaining / fps if (fps and status == "running") else None,
                    "status": status,
                    "seconds_since_update": None if math.isinf(age) else age,
                    "best_validation_score": best,
                    "best_validation_step": best_step,
                    "last_validation": {k: _float(val[-1].get(k)) for k in VALIDATION_KEYS}
                    if val
                    else None,
                    "reward": (cfg.get("env") or {}).get("reward"),
                }
            )
        return {"now": now, "runs": runs}

    def run_series(self, name: str) -> dict[str, Any]:
        d = self._run_dir(name)
        prog = read_csv_rows(d / "progress.csv")
        val = read_csv_rows(d / "validation.csv")
        progress: dict[str, list[float | None]] = {k: [] for k in PROGRESS_KEYS}
        for r in prog:
            if _float(r.get("time/total_timesteps")) is None:
                continue
            for k in PROGRESS_KEYS:
                progress[k].append(_float(r.get(k)))
        validation = {k: [_float(r.get(k)) for r in val] for k in VALIDATION_KEYS}
        return {"name": name, "progress": progress, "validation": validation}

    def _run_dir(self, name: str) -> Path:
        d = (self.runs_dir / name).resolve()
        if d.parent != self.runs_dir.resolve() or not (d / "config.yaml").exists():
            raise FileNotFoundError(name)
        return d

    # ---------------------------------------------------------------- baselines

    def baselines(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._baselines)

    def _compute_baselines(self) -> None:
        """Baselines on the exact validation scenarios used by the training callback."""
        cache = self.cache_dir / f"validation_baselines_{self.n_validation}.json"
        if cache.exists():
            with self._lock:
                self._baselines = json.loads(cache.read_text())
            return
        try:
            from uavta.environment.env import MultiUAVDynamicEnv
            from uavta.evaluation.experiments import BASELINES
            from uavta.evaluation.runner import evaluate_policy, seed_refs
            from uavta.scenarios.generator import VALIDATION_SEED_BASE

            env = MultiUAVDynamicEnv()
            refs = seed_refs(VALIDATION_SEED_BASE, self.n_validation, "val")
            result: dict[str, Any] = {"status": "computing", "methods": {}}
            for name in ("greedy_nearest", "greedy_value_rate", "random", "rh_rk"):
                df = evaluate_policy(env, BASELINES[name](), refs, method=name)
                result["methods"][name] = {k: float(df[k].mean()) for k in VALIDATION_KEYS[1:]}
                with self._lock:
                    self._baselines = json.loads(json.dumps(result))
            result["status"] = "ready"
            result["n_scenarios"] = self.n_validation
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2))
            with self._lock:
                self._baselines = result
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never crashes the server
            logger.exception("baseline computation failed")
            with self._lock:
                self._baselines = {"status": "error", "error": str(exc)}

    # ---------------------------------------------------------------- replay

    def replay(self, name: str, scenario: str, model: str) -> dict[str, Any]:
        from uavta.baselines.heuristics import GreedyValueRatePolicy
        from uavta.environment.env import MultiUAVDynamicEnv
        from uavta.evaluation.runner import ScenarioRef, run_episode
        from uavta.scenarios.loader import load_paper_scenario

        if scenario not in ("S1", "S2", "S3"):
            raise ValueError("scenario must be S1, S2 or S3")
        if model == "greedy":
            policy: Any = GreedyValueRatePolicy()
            mtime = 0.0
            label = "greedy_value_rate"
        else:
            if model not in ("best", "final"):
                raise ValueError("model must be best, final or greedy")
            path = self._run_dir(name) / f"{model}_model.zip"
            if not path.exists():
                raise FileNotFoundError(f"{path.name} not written yet")
            mtime = path.stat().st_mtime
            label = f"{name}/{path.name}"
        key = (name if model != "greedy" else "-", scenario, model, mtime)
        with self._lock:
            if key in self._replays:
                return self._replays[key]
        if model != "greedy":
            import torch

            from uavta.agents.ppo_factory import PPOPolicy

            torch.set_num_threads(1)
            policy = PPOPolicy.load(path, name=label)
        sc = load_paper_scenario(scenario)
        rec = run_episode(
            MultiUAVDynamicEnv(), policy, ScenarioRef(scenario, scenario=sc), keep_logs=True
        )
        data = {
            "label": label,
            "scenario": scenario,
            "model_saved_at": mtime or None,
            "airspace": list(sc.airspace),
            "uavs": [{"id": u.id, "type": u.uav_type} for u in sc.uavs],
            "metrics": {
                k: rec.metrics[k]
                for k in (
                    "weighted_score",
                    "value_fraction",
                    "makespan",
                    "completion_rate",
                    "n_completed",
                    "n_targets_appeared",
                    "risk_exposure",
                    "termination_reason",
                    "collected_value",
                )
            },
            "activities": [
                {
                    "uav": a.uav,
                    "kind": a.kind,
                    "t0": a.t_start,
                    "t1": a.t_end,
                    "p0": a.p_start.tolist(),
                    "p1": None if a.p_end is None else a.p_end.tolist(),
                    "slot": a.slot,
                }
                for a in rec.activities
            ],
            "targets": [{"id": tid, **info} for tid, info in rec.target_positions.items()],
            "events": rec.event_log,
            "assignments": rec.assignment_log,
        }
        data = json.loads(json.dumps(data, default=float))
        with self._lock:
            self._replays[key] = data
        return data


def make_handler(data: DashboardData) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            logger.debug(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(data.state())
                elif url.path == "/api/run":
                    self._json(data.run_series(q["name"]))
                elif url.path == "/api/baselines":
                    self._json(data.baselines())
                elif url.path == "/api/replay":
                    self._json(
                        data.replay(
                            q.get("name", ""), q.get("scenario", "S3"), q.get("model", "best")
                        )
                    )
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except FileNotFoundError as exc:
                self._json({"error": f"not found: {exc}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001
                logger.exception("request failed")
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(
    runs_dir: Path, port: int = 8765, host: str = "127.0.0.1", open_browser: bool = True
) -> None:
    data = DashboardData(runs_dir, cache_dir=Path("results/eval"))
    server = ThreadingHTTPServer((host, port), make_handler(data))
    url = f"http://{host}:{port}/"
    print(f"Canlı eğitim paneli: {url}  (durdurmak için Ctrl+C)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
