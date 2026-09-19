"""Run the five-model NumPyro CPU/GPU parallel-chain scaling experiment.

The experiment keeps the per-chain workload fixed while the number of
vectorized chains increases. Completed runs are reused, so the original geese
results remain part of the expanded experiment.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "benchmarks" / "performance_runner.py"
DEFAULT_OUT = ROOT / "benchmarks" / "performance_benchmark" / "gpu_chain_scaling"
MODELS = ("snail", "lake", "geese", "wolves", "cladocera")


def summarize(root: Path) -> None:
    rows = []
    for path in root.glob("*/numpyro_*/chains_*/repeat_*/cold/metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        repeat_dir = path.parents[1]
        rows.append({
            "model": path.parents[4].name,
            "backend": "numpyro",
            "device": path.parents[3].name.rsplit("_", 1)[-1],
            "chains": int(path.parents[2].name.split("_")[-1]),
            "repeat": int(repeat_dir.name.split("_")[-1]),
            **metrics,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame.sort_values(["model", "chains", "device", "repeat"], inplace=True)
    frame.to_csv(root / "gpu_chain_scaling_runs.csv", index=False)
    paired = frame.pivot_table(
        index=["model", "backend", "chains", "repeat"],
        columns="device",
        values=["execution_seconds", "min_bulk_ess_per_second"],
    )
    paired.columns = [f"{metric}_{device}" for metric, device in paired.columns]
    paired.reset_index(inplace=True)
    required = ["execution_seconds_cpu", "execution_seconds_gpu"]
    # During a resumable run, one device may finish before its paired result
    # exists. Keep the run-level table and defer paired summaries until both
    # CPU and GPU columns are available.
    if not all(column in paired.columns for column in required):
        return
    paired.dropna(subset=required, inplace=True)
    if paired.empty:
        return
    paired["gpu_time_speedup"] = (
        paired["execution_seconds_cpu"] / paired["execution_seconds_gpu"]
    )
    paired["gpu_ess_rate_speedup"] = (
        paired["min_bulk_ess_per_second_gpu"]
        / paired["min_bulk_ess_per_second_cpu"]
    )
    paired.to_csv(root / "gpu_chain_scaling_paired.csv", index=False)
    summary = paired.groupby(["model", "chains"], as_index=False).agg(
        repeats=("repeat", "count"),
        median_cpu_seconds=("execution_seconds_cpu", "median"),
        median_gpu_seconds=("execution_seconds_gpu", "median"),
        median_gpu_time_speedup=("gpu_time_speedup", "median"),
        median_gpu_ess_rate_speedup=("gpu_ess_rate_speedup", "median"),
    )
    summary.to_csv(root / "gpu_chain_scaling_summary.csv", index=False)


parser = ArgumentParser()
parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
parser.add_argument("--chains", nargs="+", type=int, default=[4, 8, 16, 32, 64])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--tune", type=int, default=500)
parser.add_argument("--draws", type=int, default=500)
parser.add_argument("--target-accept", type=float, default=0.95)
parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

out = args.output.resolve()
for model in args.models:
    for chains in args.chains:
        for device in ("cpu", "gpu"):
            for repeat in range(1, args.repeats + 1):
                target = (
                    out / model / f"numpyro_{device}" / f"chains_{chains}"
                    / f"repeat_{repeat:02d}"
                )
                if (target / "DONE").exists() and not args.force:
                    summarize(out)
                    continue
                target.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, str(RUNNER), model,
                    "--backend", "numpyro", "--device", device,
                    "--chains", str(chains), "--tune", str(args.tune),
                    "--draws", str(args.draws),
                    "--target-accept", str(args.target_accept),
                    "--repeat", str(repeat), "--output", str(target),
                ]
                env = os.environ.copy()
                if device == "cpu":
                    env["JAX_PLATFORMS"] = "cpu"
                else:
                    env.pop("JAX_PLATFORMS", None)
                with (target / "stdout.log").open("w", encoding="utf-8") as stdout, \
                        (target / "stderr.log").open("w", encoding="utf-8") as stderr:
                    result = subprocess.run(command, cwd=ROOT, env=env,
                                            stdout=stdout, stderr=stderr)
                summarize(out)
                print(f"model={model}, chains={chains}, device={device}, "
                      f"repeat={repeat}: "
                      f"{'DONE' if result.returncode == 0 else 'FAILED'}", flush=True)

summarize(out)
