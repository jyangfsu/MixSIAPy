"""Aggregate Section 2.4.2 run-level outputs into manuscript-ready tables."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
root = args.input.resolve()
out = (args.output or root / "summary").resolve()
out.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(root / "performance_runs.csv")
runs["total_wall_seconds"] = (
    runs["load_seconds"] + runs["model_build_seconds"] + runs["execution_seconds"]
)


def q25(series):
    return series.quantile(.25)


def q75(series):
    return series.quantile(.75)


aggregations = {
    "n": ("repeat", "count"),
    "median_total_seconds": ("total_wall_seconds", "median"),
    "q25_total_seconds": ("total_wall_seconds", q25),
    "q75_total_seconds": ("total_wall_seconds", q75),
    "median_execution_seconds": ("execution_seconds", "median"),
    "q25_execution_seconds": ("execution_seconds", q25),
    "q75_execution_seconds": ("execution_seconds", q75),
    "median_min_bulk_ess_per_second": ("min_bulk_ess_per_second", "median"),
    "q25_min_bulk_ess_per_second": ("min_bulk_ess_per_second", q25),
    "q75_min_bulk_ess_per_second": ("min_bulk_ess_per_second", q75),
    "median_peak_rss_mb": ("peak_rss_mb", "median"),
    "worst_rhat": ("max_rhat", "max"),
    "lowest_bulk_ess": ("min_ess_bulk", "min"),
    "lowest_tail_ess": ("min_ess_tail", "min"),
    "total_divergences": ("divergences", "sum"),
}

cpu = runs[(runs.experiment == "python_cpu") & (runs.phase == "cold")]
cpu_summary = cpu.groupby(["model", "method"], as_index=False).agg(**aggregations)
cpu_summary.to_csv(out / "python_cpu_backend_summary.csv", index=False)

jax = runs[runs.experiment == "jax_device"].copy()
jax[["backend", "device"]] = jax["method"].str.rsplit("_", n=1, expand=True)
jax_summary = jax.groupby(["model", "backend", "device", "phase"], as_index=False).agg(
    **aggregations
)
jax_summary.to_csv(out / "jax_cpu_gpu_summary.csv", index=False)

paired = jax.pivot_table(index=["model", "backend", "repeat", "phase"],
                         columns="device", values="execution_seconds").reset_index()
paired = paired.dropna(subset=["cpu", "gpu"])
paired["gpu_speedup"] = paired["cpu"] / paired["gpu"]
paired.to_csv(out / "jax_cpu_gpu_paired_runs.csv", index=False)
speedup = paired.groupby(["model", "backend", "phase"], as_index=False).agg(
    n=("repeat", "count"), median_speedup=("gpu_speedup", "median"),
    q25_speedup=("gpu_speedup", q25), q75_speedup=("gpu_speedup", q75),
)
speedup.to_csv(out / "jax_gpu_speedup_summary.csv", index=False)

cold_warm = jax.pivot_table(index=["model", "backend", "device", "repeat"],
                            columns="phase", values="execution_seconds").reset_index()
cold_warm = cold_warm.dropna(subset=["cold", "warm"])
cold_warm["cold_to_warm_ratio"] = cold_warm["cold"] / cold_warm["warm"]
cold_warm.to_csv(out / "jax_cold_warm_paired_runs.csv", index=False)

# Freeze one complete environment record for every method/device combination.
environment_rows = []
for path in root.glob("*/*/*/repeat_*/metadata.json"):
    meta = json.loads(path.read_text(encoding="utf-8"))
    environment_rows.append({
        "experiment": path.parents[3].name,
        "model": path.parents[2].name,
        "method": path.parents[1].name,
        "repeat": path.parent.name,
        "hardware_json": json.dumps(meta.get("hardware"), sort_keys=True),
        "versions_json": json.dumps(meta.get("versions"), sort_keys=True),
        "input_sha256_json": json.dumps(meta.get("input_sha256"), sort_keys=True),
        "chains": meta.get("chains"), "tune": meta.get("tune"),
        "draws": meta.get("draws"), "target_accept": meta.get("target_accept"),
        "thread_limit": meta.get("thread_limit"),
    })
pd.DataFrame(environment_rows).to_csv(out / "environment_manifest.csv", index=False)

expected = {"python_cpu": 5 * 4 * 3, "jax_device": 5 * 2 * 2 * 3 * 2}
completion = []
for experiment, count in expected.items():
    observed = int((runs.experiment == experiment).sum())
    completion.append({"experiment": experiment, "expected_phase_rows": count,
                       "observed_phase_rows": observed, "complete": observed == count})
pd.DataFrame(completion).to_csv(out / "completion_status.csv", index=False)
print(pd.DataFrame(completion).to_string(index=False))
