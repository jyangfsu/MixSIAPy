"""Resumable orchestrator for Section 2.4.2 performance experiments."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import os
import subprocess
import sys


ROOT = Path(__file__).parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "performance_benchmark"
MODELS = ("snail", "lake", "geese", "wolves", "cladocera")
PYTHON_BACKENDS = ("pymc", "nutpie", "numpyro", "blackjax")
JAX_BACKENDS = ("numpyro", "blackjax")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_summary(out):
    rows = []
    for path in out.glob("*/*/*/repeat_*/*/metrics.json"):
        metrics = read_json(path)
        meta = read_json(path.parents[1] / "metadata.json")
        if not metrics or not meta:
            continue
        rows.append({
            "experiment": path.parents[4].name,
            "model": path.parents[3].name,
            "method": path.parents[2].name,
            "repeat": int(path.parents[1].name.split("_")[-1]),
            **metrics,
            "load_seconds": meta.get("load_seconds"),
            "model_build_seconds": meta.get("model_build_seconds"),
            "resolved_backend": meta.get("resolved_backend"),
            "resolved_device": meta.get("resolved_device"),
        })
    rows.sort(key=lambda x: (x["experiment"], MODELS.index(x["model"]),
                             x["method"], x["repeat"], x["phase"]))
    columns = list(rows[0]) if rows else ["experiment", "model", "method", "repeat"]
    with (out / "performance_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


parser = ArgumentParser()
parser.add_argument("--experiment", choices=("python_cpu", "jax_device"), required=True)
parser.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
parser.add_argument("--backends", nargs="*", choices=PYTHON_BACKENDS)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--chains", type=int, default=3)
parser.add_argument("--tune", type=int, default=5000)
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--target-accept", type=float, default=.95)
parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
parser.add_argument("--save-posterior", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

if args.smoke:
    args.chains, args.tune, args.draws, args.repeats = 2, 25, 25, 1

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
backends = tuple(args.backends or (
    PYTHON_BACKENDS if args.experiment == "python_cpu" else JAX_BACKENDS
))
devices = ("cpu",) if args.experiment == "python_cpu" else ("cpu", "gpu")

for model in args.models:
    for backend in backends:
        if args.experiment == "jax_device" and backend not in JAX_BACKENDS:
            continue
        for device in devices:
            method = f"{backend}_{device}"
            for repeat in range(1, args.repeats + 1):
                target = out / args.experiment / model / method / f"repeat_{repeat:02d}"
                target.mkdir(parents=True, exist_ok=True)
                if (target / "DONE").exists() and not args.force:
                    update_summary(out)
                    continue
                command = [
                    sys.executable, str(ROOT / "benchmarks" / "performance_runner.py"),
                    model, "--backend", backend, "--device", device,
                    "--chains", str(args.chains), "--tune", str(args.tune),
                    "--draws", str(args.draws), "--target-accept", str(args.target_accept),
                    "--repeat", str(repeat), "--output", str(target),
                ]
                if args.experiment == "jax_device":
                    command.append("--paired-warm")
                if args.save_posterior:
                    command.append("--save-posterior")
                env = os.environ.copy()
                if device == "cpu" and backend in JAX_BACKENDS:
                    env["JAX_PLATFORMS"] = "cpu"
                elif device == "gpu":
                    env.pop("JAX_PLATFORMS", None)
                with (target / "stdout.log").open("w", encoding="utf-8") as stdout, \
                     (target / "stderr.log").open("w", encoding="utf-8") as stderr:
                    result = subprocess.run(command, cwd=ROOT, env=env,
                                            stdout=stdout, stderr=stderr)
                update_summary(out)
                print(f"{args.experiment}/{model}/{method}/repeat_{repeat:02d}: "
                      f"{'DONE' if result.returncode == 0 else 'FAILED'}", flush=True)

update_summary(out)
