"""Local live-training dashboard (standard-library HTTP server, read-only).

Serves ``dashboard.html`` and a small JSON API built from the files that training already
writes (``results/runs/<run>/{config.yaml, progress.csv, validation.csv, best_model.zip}``).
It never writes into run directories and never influences training.

Endpoints:
    /                               the panel (dashboard.html); /panel is an alias
    /api/envs                       the two environments (v1 paper model, v2 extended)
    /api/state                      all runs with progress, speed, ETA, status
    /api/run?name=RUN               time series from progress.csv and validation.csv
    /api/baselines?env=v1|v2        baseline metrics on that environment's validation scenarios
    /api/presets                    scenario library (each scenario belongs to one environment)
    /api/replay?name=RUN&scenario=ID&model=best|final|lookahead|<baseline>&seed=N
                                    one recorded episode for the animated map
    /api/results?env=v1|v2          test-set summary, hypotheses and per-scenario win counts
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

STATIC = Path(__file__).with_name("dashboard.html")  # the only page: the detailed panel
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
_RATES = {
    "new_target": 3.0,
    "uav_crash": 1.5,
    "sensor_damage": 3.0,
    "weather": 3.0,
    "target_move": 3.0,
    "value_change": 3.0,
}

# The two environments the dashboard can show. A trained model only runs in the environment it
# was trained for (the observation layouts differ), so every scenario, model group, baseline
# cache and result table belongs to exactly one environment.
ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "v1": {
        "id": "v1",
        "title": "Makale ortamı (v1)",
        "description": (
            "Makaledeki model: 3–5 drone, 8–18 hedef, hedeflerin değeri zamanla ve uzun "
            "uçuşlarda azalır. Skor J = 0,5 × toplanan değer oranı − 0,5 × görev süresi / 2400 s."
        ),
        "train_config": "configs/train/ppo_entity.yaml",
        "ai_group": "ppo_entity",
        "groups": ["ppo_entity", "ppo_default", "ppo_attention"],
        "results_dir": "results/tables",
        "eval_glob": "main_*.csv",
        "baseline_cache": "validation_baselines_100.json",
    },
    "v2": {
        "id": "v2",
        "title": "Genişletilmiş ortam (v2)",
        "description": (
            "Makale modeline ek olarak: son tarihli kritik hedefler (süresi geçince değersiz), "
            "yalnızca kızılötesi veya radarla gözlenebilen hedefler, yeni hedef çıktığında "
            "yoldaki dronelar için yeniden görevlendirme, 4–7 drone, 16–28 hedef, daha kısa pil. "
            "Skor = toplanan değer oranı."
        ),
        "train_config": "configs/train/ppo_entity_hard.yaml",
        "ai_group": "ppo_entity_hard",
        "groups": ["ppo_entity_hard"],
        "results_dir": "results/tables/v2",
        "eval_glob": "v2_main_*.csv",
        "baseline_cache": "validation_baselines_v2_100.json",
    },
}


def env_config(env_id: str) -> Any:
    """EnvConfig of an environment, read from its training configuration."""
    from uavta.environment.env import EnvConfig

    if env_id not in ENVIRONMENTS:
        raise ValueError(f"unknown environment {env_id!r}")
    cfg = yaml.safe_load(Path(ENVIRONMENTS[env_id]["train_config"]).read_text()) or {}
    return EnvConfig.model_validate(cfg.get("env") or {})


def _preset(pid: str, env: str, group: str, title: str, desc: str, **gen: Any) -> dict[str, Any]:
    """A generated scenario: ``gen`` overrides the environment's scenario generator."""
    return {
        "id": pid,
        "env": env,
        "kind": "generated",
        "group": group,
        "title": title,
        "description": desc,
        "generator": gen,
    }


def _paper(variant: str, title: str, desc: str) -> dict[str, Any]:
    return {
        "id": f"paper_{variant}",
        "env": "v1",
        "kind": "paper",
        "variant": variant,
        "group": "Makaledeki görevler",
        "title": title,
        "description": desc,
    }


_EVENTS = "Beklenmedik olaylar"
_SIZE = "Sürü ve görev büyüklüğü"
_ENERGY = "Batarya ve değer dağılımı"
_V2 = "Kritik hedefler ve sensörler"
_V2_SIZE = "Sürü büyüklüğü ve olaylar"

# Scenario library of the replay panel. Seeds come from DEMO_SEED_BASE and never overlap the
# training, validation, test or OOD seeds.
PRESETS: list[dict[str, Any]] = [
    _paper("S1", "S1 - küçük", "Makaledeki S1: 3 drone, 10 hedef; 1 drone düşer, 1 yeni hedef çıkar."),
    _paper(
        "S2", "S2 - orta", "Makaledeki S2: 4 drone, 14 hedef; hedef yer değiştirir, sensör bozulur."
    ),
    _paper("S3", "S3 - büyük", "Makaledeki S3: 5 drone, 18 hedef; 2 drone düşer, hava bozulur."),
    _preset(
        "calm",
        "v1",
        _EVENTS,
        "Olaysız",
        "Hiç olay yok; yalnızca hedeflerin sırası önemli.",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        event_rates=_ALL_ZERO,
    ),
    _preset(
        "crisis",
        "v1",
        _EVENTS,
        "Yoğun olay",
        "Bütün olay türleri normalin 3 katı sıklıkta.",
        n_uavs=[5, 5],
        n_targets=[14, 14],
        event_rates={k: 3 * v for k, v in _RATES.items()},
    ),
    _preset(
        "losses",
        "v1",
        _EVENTS,
        "Drone kayıpları",
        "Dronelar art arda düşer; kalanlar boşalan işleri devralır.",
        n_uavs=[5, 5],
        n_targets=[14, 14],
        event_rates={**_ALL_ZERO, "uav_crash": 5.0},
    ),
    _preset(
        "storm",
        "v1",
        _EVENTS,
        "Kötü hava ve arıza",
        "Hava bozulur, sensörler arızalanır; gözlem kalitesi düşer.",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        event_rates={**_ALL_ZERO, "weather": 8.0, "sensor_damage": 8.0},
    ),
    _preset(
        "moving",
        "v1",
        _EVENTS,
        "Hareketli hedefler",
        "Hedefler yer değiştirir, yenileri çıkar.",
        n_uavs=[4, 4],
        n_targets=[10, 10],
        event_rates={**_ALL_ZERO, "target_move": 10.0, "new_target": 6.0},
    ),
    _preset(
        "surge",
        "v1",
        _EVENTS,
        "Sonradan çıkan hedefler",
        "6 hedefle başlar; görev boyunca çok sayıda yeni hedef çıkar.",
        n_uavs=[4, 4],
        n_targets=[6, 6],
        event_rates={**_ALL_ZERO, "new_target": 15.0},
    ),
    _preset(
        "value_drop",
        "v1",
        _EVENTS,
        "Değeri düşen hedefler",
        "Bazı hedeflerin değeri görev sırasında azalır.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        event_rates={**_ALL_ZERO, "value_change": 15.0},
    ),
    _preset(
        "small_team",
        "v1",
        _SIZE,
        "2 drone, 8 hedef",
        "Eğitimdekinden küçük ekip.",
        n_uavs=[2, 2],
        n_targets=[8, 8],
    ),
    _preset(
        "large",
        "v1",
        _SIZE,
        "6 drone, 20 hedef",
        "Eğitimdekinden büyük görev.",
        n_uavs=[6, 6],
        n_targets=[20, 20],
    ),
    _preset(
        "many_targets",
        "v1",
        _SIZE,
        "4 drone, 22 hedef",
        "Hedef sayısı fazla; işi dengeli paylaştırmak gerekir.",
        n_uavs=[4, 4],
        n_targets=[22, 22],
    ),
    _preset(
        "single_type",
        "v1",
        _SIZE,
        "Tek tip drone",
        "Bütün dronelar A tipi (1800 s pil, kamera + kızılötesi).",
        n_uavs=[4, 4],
        n_targets=[12, 12],
        uav_type_probs={"A": 1.0},
    ),
    _preset(
        "low_battery",
        "v1",
        _ENERGY,
        "Pil %60",
        "Pil süresi normalin %60'ı; her hedefe yetişilemeyebilir.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        endurance_scale=0.6,
    ),
    _preset(
        "long_range",
        "v1",
        _ENERGY,
        "Pil %150",
        "Pil sorun değil; yalnızca sıralama önemli.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        endurance_scale=1.5,
    ),
    _preset(
        "vip",
        "v1",
        _ENERGY,
        "Birkaç çok değerli hedef",
        "Hedeflerin çoğu düşük değerli, birkaçı çok değerli.",
        n_uavs=[4, 4],
        n_targets=[14, 14],
        value_distribution="skewed",
    ),
    _preset(
        "hard",
        "v1",
        _ENERGY,
        "Zor karışım",
        "Olaylar iki kat sık, pil %70.",
        n_uavs=[5, 5],
        n_targets=[16, 16],
        endurance_scale=0.7,
        event_rates={k: 2 * v for k, v in _RATES.items()},
    ),
    _preset(
        "v2_standard",
        "v2",
        _V2,
        "Standart",
        "Eğitim dağılımından bir görev: 5 drone, 22 hedef.",
        n_uavs=[5, 5],
        n_targets=[22, 22],
    ),
    _preset(
        "v2_many_critical",
        "v2",
        _V2,
        "Kritik hedef çok",
        "Hedeflerin yarısı son tarihli kritik hedef.",
        n_uavs=[5, 5],
        n_targets=[22, 22],
        critical_fraction=0.5,
    ),
    _preset(
        "v2_tight",
        "v2",
        _V2,
        "Sıkı son tarih",
        "Kritik hedeflerin süresi 150–400 s; geç kalınan hedef değersizleşir.",
        n_uavs=[5, 5],
        n_targets=[22, 22],
        critical_deadline_range=[150.0, 400.0],
    ),
    _preset(
        "v2_radar",
        "v2",
        _V2,
        "Radar gerektiren hedef çok",
        "Hedeflerin yarısını yalnızca radarlı (B tipi) dronelar gözleyebilir.",
        n_uavs=[5, 5],
        n_targets=[22, 22],
        sensor_requirements={"none": 0.3, "IR": 0.2, "RD": 0.5},
    ),
    _preset(
        "v2_small",
        "v2",
        _V2_SIZE,
        "4 drone, 16 hedef",
        "Eğitim dağılımının alt ucu.",
        n_uavs=[4, 4],
        n_targets=[16, 16],
    ),
    _preset(
        "v2_large",
        "v2",
        _V2_SIZE,
        "7 drone, 28 hedef",
        "Eğitim dağılımının üst ucu.",
        n_uavs=[7, 7],
        n_targets=[28, 28],
    ),
    _preset(
        "v2_calm",
        "v2",
        _V2_SIZE,
        "Olaysız",
        "Hiç olay yok; yalnızca son tarihler ve sensör gereksinimleri.",
        n_uavs=[5, 5],
        n_targets=[22, 22],
        event_rates=_ALL_ZERO,
    ),
    _preset(
        "v2_crisis",
        "v2",
        _V2_SIZE,
        "Yoğun olay",
        "Bütün olay türleri normalin 2 katı sıklıkta.",
        n_uavs=[5, 5],
        n_targets=[22, 22],
        event_rate_scale=2.0,
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
        self._baselines: dict[str, dict[str, Any]] = {}
        self._replays: dict[tuple[str, str, str, float, int], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._compute_baselines("v1")

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

    def baselines(self, env_id: str = "v1") -> dict[str, Any]:
        """Classic methods on the validation scenarios of ``env_id`` (computed once, cached)."""
        if env_id not in ENVIRONMENTS:
            raise ValueError(f"unknown environment {env_id!r}")
        with self._lock:
            current = self._baselines.get(env_id)
        if current is None:
            self._compute_baselines(env_id)
            with self._lock:
                current = self._baselines.get(env_id, {"status": "pending"})
        return dict(current)

    def _compute_baselines(self, env_id: str) -> None:
        """Start the baseline computation for ``env_id`` in a background thread (or load it)."""
        cache = self.cache_dir / ENVIRONMENTS[env_id]["baseline_cache"]
        with self._lock:
            if env_id in self._baselines:
                return
            if cache.exists():
                self._baselines[env_id] = json.loads(cache.read_text())
                return
            self._baselines[env_id] = {"status": "pending"}
        threading.Thread(target=self._baseline_worker, args=(env_id, cache), daemon=True).start()

    def _baseline_worker(self, env_id: str, cache: Path) -> None:
        """Baselines on the exact validation scenarios used by the training callback."""
        try:
            from uavta.environment.env import MultiUAVDynamicEnv
            from uavta.evaluation.experiments import BASELINES
            from uavta.evaluation.runner import evaluate_policy, seed_refs
            from uavta.scenarios.generator import VALIDATION_SEED_BASE

            env = MultiUAVDynamicEnv(env_config(env_id))
            refs = seed_refs(VALIDATION_SEED_BASE, self.n_validation, "val")
            result: dict[str, Any] = {"status": "computing", "methods": {}}
            for name in ("greedy_nearest", "greedy_value_rate", "random", "rh_rk"):
                df = evaluate_policy(env, BASELINES[name](), refs, method=name)
                result["methods"][name] = {k: float(df[k].mean()) for k in VALIDATION_KEYS[1:]}
                with self._lock:
                    self._baselines[env_id] = json.loads(json.dumps(result))
            result["status"] = "ready"
            result["n_scenarios"] = self.n_validation
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2))
            with self._lock:
                self._baselines[env_id] = result
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never crashes the server
            logger.exception("baseline computation failed")
            with self._lock:
                self._baselines[env_id] = {"status": "error", "error": str(exc)}

    # ---------------------------------------------------------------- scenarios & replay

    @staticmethod
    def environments() -> list[dict[str, Any]]:
        keys = ("id", "title", "description", "ai_group", "groups")
        return [{k: e[k] for k in keys} for e in ENVIRONMENTS.values()]

    def presets(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in p.items() if k != "generator"} for p in PRESETS]

    @staticmethod
    def _preset(preset_id: str) -> dict[str, Any]:
        if preset_id in ("S1", "S2", "S3"):  # short names used by older clients
            preset_id = f"paper_{preset_id}"
        preset = next((p for p in PRESETS if p["id"] == preset_id), None)
        if preset is None:
            raise ValueError(f"unknown scenario {preset_id!r}")
        return preset

    def _scenario(self, preset_id: str, seed: int) -> Any:
        from uavta.scenarios.generator import EventRates, GeneratorConfig, ScenarioGenerator
        from uavta.scenarios.loader import load_paper_scenario
        from uavta.utils.config import deep_merge

        preset = self._preset(preset_id)
        if preset["kind"] == "paper":
            return load_paper_scenario(preset["variant"])
        base = env_config(preset["env"]).scenario.generator.model_dump(mode="json")
        overrides = dict(preset["generator"])
        scale = overrides.pop("event_rate_scale", None)
        gen = deep_merge(base, overrides)
        if "sensor_requirements" in overrides:  # replace, do not merge, the distribution
            gen["sensor_requirements"] = overrides["sensor_requirements"]
        if scale is not None:
            gen["event_rates"] = EventRates(**gen["event_rates"]).scaled(scale).model_dump()
        cfg = GeneratorConfig.model_validate(gen)
        return ScenarioGenerator(cfg, name=preset["id"]).generate(DEMO_SEED_BASE + seed)

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
        env_id = self._preset(scenario)["env"]
        env = MultiUAVDynamicEnv(env_config(env_id))
        model_obj = getattr(policy, "model", None) or getattr(
            getattr(policy, "base", None), "model", None
        )
        if model_obj is not None and model_obj.observation_space.shape != env.observation_space.shape:
            raise ValueError(
                f"{name} bu ortam için eğitilmedi: {ENVIRONMENTS[env_id]['title']} görevleri "
                "yalnızca o ortamda eğitilmiş modellerle oynatılabilir"
            )
        sc = self._scenario(scenario, seed)
        rec = run_episode(
            env,
            policy,
            ScenarioRef(scenario, scenario=sc),
            keep_logs=True,
            policy_seed=seed,
        )
        data = {
            "label": label,
            "model": model,
            "env": env_id,
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
                    "n_critical",
                    "critical_on_time",
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
        pattern: str = "main_*.csv",
    ) -> dict[str, Any] | None:
        """Per-scenario wins of ``ai`` (mean over seeds) against ``rival`` on ``metric``."""
        evals = Path("results/eval")
        sums: dict[str, dict[str, list[float]]] = {ai: {}, rival: {}}
        for f in sorted(evals.glob(pattern)):
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

    def results(self, env_id: str = "v1") -> dict[str, Any]:
        """Final test-set summary of ``env_id`` written by scripts/analyze.py, if available."""
        if env_id not in ENVIRONMENTS:
            raise ValueError(f"unknown environment {env_id!r}")
        env = ENVIRONMENTS[env_id]
        tables = Path(env["results_dir"])
        summary = tables / "summary_test_id.csv"
        if not summary.exists():
            return {"status": "missing", "env": env_id}
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
                "critical_on_time_rate",
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
        out: dict[str, Any] = {
            "status": "ready",
            "env": env_id,
            "ai_method": env["ai_group"],
            "methods": methods,
        }
        lat = tables / "latency.json"
        if lat.exists():
            out["latency_ms"] = {
                k: v["median_of_episode_median_ms"] for k, v in json.loads(lat.read_text()).items()
            }
        # Per-scenario wins on collected value, against every classic method; differences below
        # 0.1 percentage point count as a draw (the same rule the replay verdict uses).
        wins = {}
        for rival in ("greedy_nearest", "greedy_value_rate", "rh_rk"):
            w = self._win_counts(
                ai=env["ai_group"],
                rival=rival,
                metric="value_fraction",
                tie_tol=0.001,
                pattern=env["eval_glob"],
            )
            if w:
                wins[rival] = w
        if wins:
            out["wins"] = wins
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
                if url.path in ("/", "/index.html", "/panel", "/panel/"):
                    self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/envs":
                    self._json(data.environments())
                elif url.path == "/api/state":
                    self._json(data.state())
                elif url.path == "/api/run":
                    self._json(data.run_series(q["name"]))
                elif url.path == "/api/baselines":
                    self._json(data.baselines(q.get("env", "v1")))
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
                    self._json(data.results(q.get("env", "v1")))
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
