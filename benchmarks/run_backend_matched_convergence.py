"""Find converged CPU schedules for the five-model, four-backend benchmark.

Existing three-repeat normal-schedule results are reused when all diagnostics
passed.  Failed model/backend combinations are rerun at progressively larger
budgets.  A schedule is selected only when all three independent repetitions
pass the common convergence criteria.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "performance_runner.py"
PERFORMANCE = ROOT / "benchmarks" / "performance_benchmark"
NORMAL_SUMMARY = PERFORMANCE / "wsl_summary" / "python_cpu_backend_summary.csv"
OUT = PERFORMANCE / "backend_matched_convergence"

MODELS = ("snail", "lake", "geese", "wolves", "cladocera")
BACKENDS = ("pymc", "nutpie", "numpyro", "blackjax")
REPEATS = (1, 2, 3)
SCHEDULES = (
    ("long", dict(chains=4, tune=10000, draws=2000, target_accept=0.99)),
    ("very_long", dict(chains=4, tune=20000, draws=5000, target_accept=0.995)),
)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def passes(metrics: dict | None) -> bool:
    if not metrics:
        return False
    tree_hits = metrics.get("treedepth_hits")
    return bool(
        metrics.get("max_rhat") is not None
        and metrics["max_rhat"] <= 1.05
        and metrics.get("min_ess_bulk") is not None
        and metrics["min_ess_bulk"] >= 100
        and metrics.get("min_ess_tail") is not None
        and metrics["min_ess_tail"] >= 100
        and metrics.get("divergences", 0) == 0
        and (tree_hits is None or tree_hits == 0)
        and metrics.get("min_bfmi") is not None
        and metrics["min_bfmi"] > 0.30
    )


def existing_normal_status() -> tuple[pd.DataFrame, dict[tuple[str, str], bool]]:
    frame = pd.read_csv(NORMAL_SUMMARY)
    frame["backend"] = frame["method"].str.replace("_cpu", "", regex=False)
    frame["passed"] = (
        (frame["worst_rhat"] <= 1.05)
        & (frame["lowest_bulk_ess"] >= 100)
        & (frame["lowest_tail_ess"] >= 100)
        & (frame["total_divergences"] == 0)
    )
    status = {
        (row.model, row.backend): bool(row.passed)
        for row in frame.itertuples(index=False)
    }
    return frame, status


def run_one(model: str, backend: str, schedule: str, settings: dict, repeat: int) -> None:
    target = OUT / schedule / model / f"{backend}_cpu" / f"repeat_{repeat:02d}"
    if (target / "DONE").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        model,
        "--backend", backend,
        "--device", "cpu",
        "--chains", str(settings["chains"]),
        "--tune", str(settings["tune"]),
        "--draws", str(settings["draws"]),
        "--target-accept", str(settings["target_accept"]),
        "--seed", str(20260914 + repeat * 1009),
        "--repeat", str(repeat),
        "--output", str(target),
    ]
    env = os.environ.copy()
    if backend in {"numpyro", "blackjax"}:
        env["JAX_PLATFORMS"] = "cpu"
    with (target / "stdout.log").open("w", encoding="utf-8") as stdout, \
         (target / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    print(
        f"{schedule}/{model}/{backend}/repeat_{repeat:02d}: "
        f"{'DONE' if result.returncode == 0 else 'FAILED'}",
        flush=True,
    )


def schedule_metrics(model: str, backend: str, schedule: str) -> list[dict | None]:
    rows = []
    for repeat in REPEATS:
        path = OUT / schedule / model / f"{backend}_cpu" / f"repeat_{repeat:02d}" / "cold" / "metrics.json"
        rows.append(read_json(path))
    return rows


def write_summary(normal: pd.DataFrame, selected: dict[tuple[str, str], str], pending: set[tuple[str, str]]) -> None:
    rows = []
    normal_lookup = normal.set_index(["model", "backend"])
    for model in MODELS:
        for backend in BACKENDS:
            key = (model, backend)
            schedule = selected.get(key)
            if schedule == "normal":
                row = normal_lookup.loc[key]
                rows.append({
                    "model": model,
                    "backend": backend,
                    "selected_schedule": "normal",
                    "status": "CONVERGED",
                    "chains": 3,
                    "tune": 5000,
                    "draws": 1000,
                    "target_accept": 0.95,
                    "repeats": int(row["n"]),
                    "median_execution_seconds": row["median_execution_seconds"],
                    "q25_execution_seconds": row["q25_execution_seconds"],
                    "q75_execution_seconds": row["q75_execution_seconds"],
                    "median_min_bulk_ess_per_second": row["median_min_bulk_ess_per_second"],
                    "q25_min_bulk_ess_per_second": row["q25_min_bulk_ess_per_second"],
                    "q75_min_bulk_ess_per_second": row["q75_min_bulk_ess_per_second"],
                    "worst_rhat": row["worst_rhat"],
                    "lowest_bulk_ess": row["lowest_bulk_ess"],
                    "lowest_tail_ess": row["lowest_tail_ess"],
                    "total_divergences": row["total_divergences"],
                    "lowest_bfmi": None,
                })
            elif schedule:
                settings = dict(SCHEDULES)[schedule]
                metrics = schedule_metrics(model, backend, schedule)
                valid = [m for m in metrics if m]
                times = pd.Series([m["execution_seconds"] for m in valid], dtype=float)
                rates = pd.Series([m["min_bulk_ess_per_second"] for m in valid], dtype=float)
                rows.append({
                    "model": model,
                    "backend": backend,
                    "selected_schedule": schedule,
                    "status": "CONVERGED" if len(valid) == 3 and all(passes(m) for m in valid) else "NOT_CONVERGED",
                    **settings,
                    "repeats": len(valid),
                    "median_execution_seconds": times.median() if len(times) else None,
                    "q25_execution_seconds": times.quantile(0.25) if len(times) else None,
                    "q75_execution_seconds": times.quantile(0.75) if len(times) else None,
                    "median_min_bulk_ess_per_second": rates.median() if len(rates) else None,
                    "q25_min_bulk_ess_per_second": rates.quantile(0.25) if len(rates) else None,
                    "q75_min_bulk_ess_per_second": rates.quantile(0.75) if len(rates) else None,
                    "worst_rhat": max(m["max_rhat"] for m in valid) if valid else None,
                    "lowest_bulk_ess": min(m["min_ess_bulk"] for m in valid) if valid else None,
                    "lowest_tail_ess": min(m["min_ess_tail"] for m in valid) if valid else None,
                    "total_divergences": sum(m["divergences"] for m in valid) if valid else None,
                    "lowest_bfmi": min(m["min_bfmi"] for m in valid) if valid else None,
                })
            else:
                rows.append({
                    "model": model,
                    "backend": backend,
                    "selected_schedule": None,
                    "status": "PENDING" if key in pending else "NOT_CONVERGED_AT_MAXIMUM",
                })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "backend_matched_convergence_summary.csv", index=False)


def main() -> None:
    normal, normal_pass = existing_normal_status()
    selected = {key: "normal" for key, passed in normal_pass.items() if passed}
    pending = {key for key, passed in normal_pass.items() if not passed}
    write_summary(normal, selected, pending)
    print(f"normal converged: {len(selected)}/20; pending: {len(pending)}", flush=True)

    for schedule, settings in SCHEDULES:
        if not pending:
            break
        stage_targets = sorted(pending)
        for model, backend in stage_targets:
            for repeat in REPEATS:
                run_one(model, backend, schedule, settings, repeat)
            metrics = schedule_metrics(model, backend, schedule)
            if len(metrics) == 3 and all(passes(m) for m in metrics):
                selected[(model, backend)] = schedule
                pending.remove((model, backend))
            write_summary(normal, selected, pending)
        print(f"after {schedule}: converged={len(selected)}/20; pending={len(pending)}", flush=True)

    write_summary(normal, selected, pending)
    print(f"finished: converged={len(selected)}/20; unresolved={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
