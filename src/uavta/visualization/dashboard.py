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

STATIC = Path(__file__).with_name("dashboard.html")  # full panel (simulation + training details)
SIMPLE_PAGE = Path(__file__).with_name("simple.html")  # end-user page served at "/"
STATIC_FILES = {"/static/sim_view.js": (Path(__file__).with_name("sim_view.js"), "text/javascript")}
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
DEMO_SEED_BASE = 5_000_000  # demo scenarios never overlap training/validation/test/OOD seeds

_ALL_ZERO = {
    "new_target": 0.0,
    "uav_crash": 0.0,
    "sensor_damage": 0.0,
    "weather": 0.0,
    "target_move": 0.0,
    "value_change": 0.0,
}
# Scenario library of the replay panel. Generated presets are ordinary GeneratorConfig dicts.
_RATES = {
    "new_target": 3.0,
    "uav_crash": 1.5,
    "sensor_damage": 3.0,
    "weather": 3.0,
    "target_move": 3.0,
    "value_change": 3.0,
}


def _preset(pid: str, group: str, icon: str, title: str, desc: str, **gen: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "kind": "generated",
        "group": group,
        "icon": icon,
        "title": title,
        "description": desc,
        "generator": gen,
    }


_PAPER = "Makaledeki görevler"
_EVENTS = "Beklenmedik olaylar"
_SIZE = "Sürü ve görev büyüklüğü"
_ENERGY = "Batarya ve öncelik"

# Scenario library of the replay panel (20 scenarios). Generated presets are GeneratorConfig
# dicts; their seeds come from DEMO_SEED_BASE and never overlap the evaluation seeds.
PRESETS: list[dict[str, Any]] = [
    {
        "id": "paper_S1",
        "kind": "paper",
        "variant": "S1",
        "group": _PAPER,
        "icon": "🗺",
        "title": "Küçük görev",
        "description": "Makaledeki S1: 3 drone, 10 hedef; 1 drone düşer, 1 yeni hedef çıkar.",
    },
    {
        "id": "paper_S2",
        "kind": "paper",
        "variant": "S2",
        "group": _PAPER,
        "icon": "🗺",
        "title": "Orta görev",
        "description": "Makaledeki S2: 4 drone, 14 hedef; hedef yer değiştirir, sensör bozulur.",
    },
    {
        "id": "paper_S3",
        "kind": "paper",
        "variant": "S3",
        "group": _PAPER,
        "icon": "🗺",
        "title": "Zor görev",
        "description": "Makaledeki S3: 5 drone, 18 hedef; 2 drone düşer, fırtına çıkar.",
    },
    _preset(
        "calm",
        _EVENTS,
        "🌤",
        "Sakin gün",
        "Hiç sürpriz yok; sadece hedeflerin akıllı sırayla gezilmesi önemli.",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        event_rates=_ALL_ZERO,
    ),
    _preset(
        "crisis",
        _EVENTS,
        "🚨",
        "Kriz günü",
        "Her şey ters gidiyor: kayıplar, yeni hedefler, kötü hava, hepsi sık sık.",
        n_uavs=[5, 5],
        n_targets=[14, 14],
        event_rates={k: 3 * v for k, v in _RATES.items()},
    ),
    _preset(
        "losses",
        _EVENTS,
        "💥",
        "Drone kayıpları",
        "Dronelar art arda düşer; kalanlar boşalan işleri devralmalı.",
        n_uavs=[5, 5],
        n_targets=[14, 14],
        event_rates={**_ALL_ZERO, "uav_crash": 5.0},
    ),
    _preset(
        "storm",
        _EVENTS,
        "⛈",
        "Fırtına",
        "Hava bozulur, sensörler arızalanır; gözlem kalitesi düşer.",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        event_rates={**_ALL_ZERO, "weather": 8.0, "sensor_damage": 8.0},
    ),
    _preset(
        "fog",
        _EVENTS,
        "🌫",
        "Sisli sabah",
        "Görevin hemen başında sis çöker; kameralar baştan zayıf görür.",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        event_window=150.0,
        event_rates={**_ALL_ZERO, "weather": 20.0},
    ),
    _preset(
        "moving",
        _EVENTS,
        "🏃",
        "Hareketli hedefler",
        "Hedefler yer değiştirir, yenileri çıkar; rotalar sürekli değişir.",
        n_uavs=[4, 4],
        n_targets=[10, 10],
        event_rates={**_ALL_ZERO, "target_move": 10.0, "new_target": 6.0},
    ),
    _preset(
        "surge",
        _EVENTS,
        "📍",
        "Hedef yağmuru",
        "Az hedefle başlar, görev boyunca birçok yeni hedef ortaya çıkar.",
        n_uavs=[4, 4],
        n_targets=[6, 6],
        event_rates={**_ALL_ZERO, "new_target": 15.0},
    ),
    _preset(
        "value_drop",
        _EVENTS,
        "📉",
        "Önemini yitiren hedefler",
        "Bazı hedeflerin önemi görev sırasında azalır; öncelik sırası değişir.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        event_rates={**_ALL_ZERO, "value_change": 15.0},
    ),
    _preset(
        "small_team",
        _SIZE,
        "👥",
        "Küçük ekip",
        "Sadece 2 drone ile 8 hedef: her karar daha önemli.",
        n_uavs=[2, 2],
        n_targets=[8, 8],
    ),
    _preset(
        "large",
        _SIZE,
        "🐝",
        "Büyük sürü",
        "6 drone ve 20 hedef: eğitimde görülenden daha büyük bir görev.",
        n_uavs=[6, 6],
        n_targets=[20, 20],
    ),
    _preset(
        "many_targets",
        _SIZE,
        "🎯",
        "Kalabalık hedef",
        "4 drone, 22 hedef: işi dengeli paylaştırmak gerekir.",
        n_uavs=[4, 4],
        n_targets=[22, 22],
    ),
    _preset(
        "single_type",
        _SIZE,
        "✈",
        "Tek tip drone",
        "Bütün dronelar aynı (daha kısa pilli, 2 sensörlü) tipte.",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        uav_type_probs={"A": 1.0},
    ),
    _preset(
        "low_battery",
        _ENERGY,
        "🪫",
        "Zayıf batarya",
        "Piller normalin %60'ı: her hedefe yetişmek mümkün olmayabilir.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        endurance_scale=0.6,
    ),
    _preset(
        "long_range",
        _ENERGY,
        "🔋",
        "Güçlü batarya",
        "Piller normalin 1,5 katı: enerji sorun değil, sıralama önemli.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        endurance_scale=1.5,
    ),
    _preset(
        "vip",
        _ENERGY,
        "💎",
        "Birkaç çok önemli hedef",
        "Hedeflerin çoğu sıradan, birkaçı çok değerli: onları kaçırmamak gerek.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        value_distribution="skewed",
    ),
    _preset(
        "extremes",
        _ENERGY,
        "⚖",
        "Ya çok önemli ya önemsiz",
        "Hedefler iki uçta: ya çok değerli ya da çok az değerli.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        value_distribution="bimodal",
    ),
    _preset(
        "hard",
        _ENERGY,
        "🔥",
        "En zor mod",
        "Olaylar iki kat sık, piller %70: her şey bir arada.",
        n_uavs=[5, 5],
        n_targets=[16, 16],
        endurance_scale=0.7,
        event_rates={k: 2 * v for k, v in _RATES.items()},
    ),
]


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
        self._replays: dict[tuple[str, str, str, float, int], dict[str, Any]] = {}
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

    # ---------------------------------------------------------------- scenarios & replay

    def presets(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in p.items() if k != "generator"} for p in PRESETS]

    def _scenario(self, preset_id: str, seed: int) -> Any:
        from uavta.scenarios.generator import GeneratorConfig, ScenarioGenerator
        from uavta.scenarios.loader import load_paper_scenario

        if preset_id in ("S1", "S2", "S3"):  # short names used by older clients
            preset_id = f"paper_{preset_id}"
        preset = next((p for p in PRESETS if p["id"] == preset_id), None)
        if preset is None:
            raise ValueError(f"unknown scenario {preset_id!r}")
        if preset["kind"] == "paper":
            return load_paper_scenario(preset["variant"])
        cfg = GeneratorConfig.model_validate(preset["generator"])
        return ScenarioGenerator(cfg, name=preset_id).generate(DEMO_SEED_BASE + seed)

    def _policy(self, name: str, model: str) -> tuple[Any, str, float]:
        """Return (policy, label, model mtime) for a run checkpoint or a baseline."""
        from uavta.evaluation.experiments import BASELINES

        alias = {"greedy": "greedy_value_rate"}
        model = alias.get(model, model)
        if model in BASELINES:
            return BASELINES[model](), model, 0.0
        if model not in ("best", "final", "lookahead"):
            raise ValueError(f"unknown model {model!r}")
        file = "best" if model == "lookahead" else model
        path = self._run_dir(name) / f"{file}_model.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not written yet")
        import torch

        from uavta.agents.ppo_factory import PPOPolicy

        torch.set_num_threads(1)
        label = f"{name}/{path.name}"
        policy: Any = PPOPolicy.load(path, name=label)
        if model == "lookahead":
            from uavta.agents.lookahead import LookaheadPolicy

            policy = LookaheadPolicy(policy, top_k=2, samples=0)
            label += " + ileriye bakma (k=2)"
        return policy, label, path.stat().st_mtime

    def replay(self, name: str, scenario: str, model: str, seed: int = 0) -> dict[str, Any]:
        """Run one recorded episode for the animated map (cached per inputs and checkpoint)."""
        from uavta.environment.env import MultiUAVDynamicEnv
        from uavta.evaluation.runner import ScenarioRef, run_episode

        if model in ("best", "final", "lookahead"):
            path = self._run_dir(name) / f"{'best' if model == 'lookahead' else model}_model.zip"
            mtime = path.stat().st_mtime if path.exists() else -1.0
        else:
            mtime = 0.0
        key = (
            name if model in ("best", "final", "lookahead") else "-",
            scenario,
            model,
            mtime,
            seed,
        )
        with self._lock:
            if key in self._replays:
                return self._replays[key]
        policy, label, _ = self._policy(name, model)
        sc = self._scenario(scenario, seed)
        rec = run_episode(
            MultiUAVDynamicEnv(),
            policy,
            ScenarioRef(scenario, scenario=sc),
            keep_logs=True,
            policy_seed=seed,
        )
        data = {
            "label": label,
            "model": model,
            "scenario": scenario,
            "seed": seed,
            "model_saved_at": mtime or None,
            "airspace": list(sc.airspace),
            "initial_weather": list(sc.initial_weather),
            "uavs": [
                {
                    "id": u.id,
                    "type": u.uav_type,
                    "endurance": u.endurance,
                    "speed": u.speed,
                    "sensors": [
                        s for s, has in zip(("VL", "IR", "RD"), u.sensors, strict=True) if has
                    ],
                }
                for u in sc.uavs
            ],
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
                    "appeared_value",
                    "energy_used_total",
                    "n_operational_end",
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

    # ---------------------------------------------------------------- final results

    @staticmethod
    def _win_counts(
        ai: str = "ppo_entity",
        rival: str = "greedy_nearest",
        condition: str = "test_id",
        metric: str = "weighted_score",
        tie_tol: float = 1e-9,
    ) -> dict[str, Any] | None:
        """Per-scenario wins of ``ai`` (mean over seeds) against ``rival`` on ``metric``."""
        evals = Path("results/eval")
        sums: dict[str, dict[str, list[float]]] = {ai: {}, rival: {}}
        for f in sorted(evals.glob("main_*.csv")):
            for r in read_csv_rows(f):
                if r.get("condition") != condition or r.get("method") not in sums:
                    continue
                v = _float(r.get(metric))
                if v is not None:
                    sums[r["method"]].setdefault(r["scenario_ref"], []).append(v)
        common = sorted(set(sums[ai]) & set(sums[rival]))
        if not common:
            return None
        win = tie = 0
        for k in common:
            a = sum(sums[ai][k]) / len(sums[ai][k])
            b = sum(sums[rival][k]) / len(sums[rival][k])
            if abs(a - b) <= tie_tol:
                tie += 1
            elif a > b:
                win += 1
        return {
            "ai": ai,
            "rival": rival,
            "metric": metric,
            "n": len(common),
            "wins": win,
            "ties": tie,
            "losses": len(common) - win - tie,
        }

    def results(self) -> dict[str, Any]:
        """Final test-set summary written by scripts/analyze.py, if available."""
        tables = Path("results/tables")
        summary = tables / "summary_test_id.csv"
        if not summary.exists():
            return {"status": "missing"}
        methods = []
        for r in read_csv_rows(summary):
            row: dict[str, Any] = {"method": r["method"], "n_seeds": int(float(r["n_seeds"] or 0))}
            for k in (
                "weighted_score",
                "value_fraction",
                "makespan",
                "completion_rate",
                "energy_used_total",
                "risk_exposure",
            ):
                row[k] = _float(r.get(k))
                row[k + "_spread"] = _float(r.get(k + "_spread"))
            row["n_scenarios"] = int(float(r["n_scenarios"]))
            methods.append(row)
        # Post-study improvements (separate analysis) are appended for the expert view.
        extra = tables / "summary_improvements.csv"
        if extra.exists():
            known = {m["method"] for m in methods}
            for r in read_csv_rows(extra):
                if r["method"] in known:
                    continue
                row = {
                    "method": r["method"],
                    "n_seeds": int(float(r["n_seeds"] or 0)),
                    "n_scenarios": int(float(r["n_scenarios"])),
                }
                for k in (
                    "weighted_score",
                    "value_fraction",
                    "makespan",
                    "completion_rate",
                    "risk_exposure",
                ):
                    row[k] = _float(r.get(k))
                    row[k + "_spread"] = _float(r.get(k + "_spread"))
                methods.append(row)
        out: dict[str, Any] = {"status": "ready", "methods": methods}
        lat = tables / "latency.json"
        if lat.exists():
            out["latency_ms"] = {
                k: v["median_of_episode_median_ms"] for k, v in json.loads(lat.read_text()).items()
            }
        wins = self._win_counts()
        if wins:
            out["wins"] = wins  # on J (training objective), used by the technical panel
        # The end-user page judges a mission by collected information only (one visible rule);
        # differences below 0.1 percentage point count as a draw, as on that page.
        wins_info = self._win_counts(metric="value_fraction", tie_tol=0.001)
        if wins_info:
            out["wins_info"] = wins_info
        hyp = tables / "hypotheses.json"
        if hyp.exists():
            out["hypotheses"] = json.loads(hyp.read_text())
        return out


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
                    self._send(200, SIMPLE_PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path in ("/panel", "/panel/"):
                    self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
                elif url.path in STATIC_FILES:
                    path, ctype = STATIC_FILES[url.path]
                    self._send(200, path.read_bytes(), ctype + "; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(data.state())
                elif url.path == "/api/run":
                    self._json(data.run_series(q["name"]))
                elif url.path == "/api/baselines":
                    self._json(data.baselines())
                elif url.path == "/api/replay":
                    self._json(
                        data.replay(
                            q.get("name", ""),
                            q.get("scenario", "paper_S3"),
                            q.get("model", "best"),
                            int(q.get("seed", "0")),
                        )
                    )
                elif url.path == "/api/presets":
                    self._json(data.presets())
                elif url.path == "/api/results":
                    self._json(data.results())
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
