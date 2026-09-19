"""Run one isolated MixSIAPy backend-performance benchmark job.

One process handles one model/backend/device/repeat combination. JAX jobs may
execute a cold and an immediately repeated warm fit on the same built model.
All requested models, initial points, chain counts, tuning iterations and
retained draws are otherwise held constant.
"""
from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback


parser = ArgumentParser()
parser.add_argument("model", choices=("snail", "lake", "geese", "wolves", "cladocera"))
parser.add_argument("--backend", required=True,
                    choices=("pymc", "nutpie", "numpyro", "blackjax"))
parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
parser.add_argument("--chains", type=int, default=3)
parser.add_argument("--tune", type=int, default=5000)
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--target-accept", type=float, default=.95)
parser.add_argument("--seed", type=int, default=20260816)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--paired-warm", action="store_true")
parser.add_argument("--save-posterior", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

# Fix CPU thread policy before importing NumPy/PyMC/JAX.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if args.device == "cpu" and args.backend in {"numpyro", "blackjax"}:
    os.environ["JAX_PLATFORMS"] = "cpu"

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from mixsiapy import build_model, load_discr_data, load_mix_data, load_source_data
from mixsiapy.backends import resolve_backend, resolve_device
from numerical_configs import CONFIGS


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardware_metadata():
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        metadata["cpu"] = subprocess.check_output(
            ["bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1"],
            text=True, timeout=10).strip()
    except Exception:
        metadata["cpu"] = platform.processor()
    try:
        metadata["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except Exception:
        metadata["gpu"] = None
    return metadata


def selected_variables(idata):
    prefixes = (
        "p_global", "p_fac", "p_both", "ilr_", "fac", "resid_prop",
        "Sigma", "src_mu", "src_var", "src_rho",
    )
    return [name for name in idata.posterior.data_vars if name.startswith(prefixes)]


def diagnose(idata, elapsed):
    variables = selected_variables(idata)
    diagnostics = az.summary(idata, var_names=variables, kind="diagnostics").reset_index()
    diagnostics.rename(columns={"index": "parameter"}, inplace=True)
    full = az.summary(idata, var_names=variables, kind="all").reset_index()
    full.rename(columns={"index": "parameter"}, inplace=True)
    usable = full[(full["sd"] > 0) & np.isfinite(full["sd"])].copy()
    usable["mcse_over_sd"] = usable["mcse_mean"] / usable["sd"]
    stats = idata.sample_stats
    divergences = int(stats["diverging"].sum()) if "diverging" in stats else 0
    reached = int(stats["reached_max_treedepth"].sum()) \
        if "reached_max_treedepth" in stats else None
    bfmi = az.bfmi(idata)
    bulk = diagnostics["ess_bulk"].dropna()
    tail = diagnostics["ess_tail"].dropna()
    rhats = diagnostics["r_hat"].dropna()
    metrics = {
        "monitored_parameter_count": int(len(diagnostics)),
        "max_rhat": float(rhats.max()) if len(rhats) else None,
        "min_ess_bulk": float(bulk.min()) if len(bulk) else None,
        "median_ess_bulk": float(bulk.median()) if len(bulk) else None,
        "min_ess_tail": float(tail.min()) if len(tail) else None,
        "max_mcse_over_sd": float(usable["mcse_over_sd"].max()) if len(usable) else None,
        "divergences": divergences,
        "treedepth_hits": reached,
        "min_bfmi": float(np.min(bfmi)) if len(bfmi) else None,
    }
    metrics["min_bulk_ess_per_second"] = (
        metrics["min_ess_bulk"] / elapsed if elapsed > 0 and metrics["min_ess_bulk"] else None
    )
    return diagnostics, full, metrics


def main():
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    failed = output / "FAILED"
    failed.unlink(missing_ok=True)

    cfg = dict(CONFIGS[args.model])
    data = ROOT / "mixsiapy" / "data"
    paths = {kind: data / cfg[kind] for kind in ("mix", "source", "discr")}
    frozen = {
        "model": args.model, "model_configuration": cfg,
        "backend": args.backend, "device": args.device,
        "chains": args.chains, "tune": args.tune, "draws": args.draws,
        "target_accept": args.target_accept, "seed": args.seed,
        "repeat": args.repeat, "paired_warm": args.paired_warm,
        "thread_limit": 1,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    (output / "config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    try:
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        mix = load_mix_data(paths["mix"], cfg["iso"], cfg.get("factors"),
                            cfg.get("random"), cfg.get("nested"), cfg.get("continuous"))
        source = load_source_data(paths["source"], cfg.get("source_factor"),
                                  cfg.get("conc_dep", False), cfg["source_type"], mix)
        discr = load_discr_data(paths["discr"], mix)
        load_seconds = time.perf_counter() - load_start

        build_start = time.perf_counter()
        model = build_model(mix, source, discr, cfg["alpha_prior"],
                            cfg["process_err"], cfg["resid_err"])
        build_seconds = time.perf_counter() - build_start
        backend = resolve_backend(args.backend, device=args.device)
        actual_device, _ = resolve_device(backend, args.device)
        base_init = model.initial_point()
        initvals = [deepcopy(base_init) for _ in range(args.chains)]

        phases = ["cold", "warm"] if args.paired_warm else ["cold"]
        phase_rows = []
        for phase_index, phase in enumerate(phases):
            phase_dir = output / phase
            phase_dir.mkdir(exist_ok=True)
            sample_kwargs = dict(
                draws=args.draws, tune=args.tune, chains=args.chains, cores=1,
                target_accept=args.target_accept,
                random_seed=args.seed + phase_index,
                initvals=deepcopy(initvals), progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": False},
                nuts_sampler=backend,
            )
            if backend == "pymc":
                sample_kwargs["nuts"] = {"max_treedepth": 10}
            elif backend in {"numpyro", "blackjax"}:
                # Both JAX samplers support vectorized chains on one physical
                # device; BlackJAX does not expose a sequential-chain mode via
                # PyMC. Use the same one-device policy for both JAX backends.
                sample_kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}

            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            started = time.perf_counter()
            # Context-manager instances returned by jax.default_device are
            # single use, so resolve a fresh context for each paired phase.
            _, context = resolve_device(backend, args.device)
            with context:
                with model:
                    fit = pm.sample(**sample_kwargs)
            elapsed = time.perf_counter() - started
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            diagnostics, summary, metrics = diagnose(fit, elapsed)
            diagnostics.to_csv(phase_dir / "diagnostics.csv", index=False)
            summary.to_csv(phase_dir / "summary.csv", index=False)
            if args.save_posterior:
                fit.to_netcdf(phase_dir / "posterior.nc")
            row = {
                "phase": phase, "execution_seconds": elapsed,
                "peak_rss_mb": peak_rss_mb,
                "rss_increment_mb": max(0.0, peak_rss_mb - rss_before),
                **metrics,
            }
            phase_rows.append(row)
            (phase_dir / "metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8")

        metadata = {
            **frozen, "resolved_backend": backend, "resolved_device": actual_device,
            "load_seconds": load_seconds, "model_build_seconds": build_seconds,
            "total_process_seconds": time.perf_counter() - total_start,
            "hardware": hardware_metadata(),
            "versions": {name: package_version(name) for name in (
                "mixsiapy", "numpy", "pandas", "scipy", "pymc", "pytensor",
                "arviz", "nutpie", "jax", "jaxlib", "numpyro", "blackjax")},
            "phases": phase_rows,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pd.DataFrame(phase_rows).to_csv(output / "phase_summary.csv", index=False)
        (output / "DONE").write_text("ok\n", encoding="utf-8")
        print(json.dumps({"status": "DONE", "output": str(output),
                          "phases": phase_rows}, indent=2), flush=True)
    except Exception:
        text = traceback.format_exc()
        failed.write_text(text, encoding="utf-8")
        print(text, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
