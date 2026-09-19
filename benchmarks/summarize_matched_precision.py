"""Summarize the R/JAGS versus native PyMC matched-precision experiment."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "benchmarks" / "numerical_agreement_v2"
OUT = ROOT / "benchmarks" / "performance_benchmark"
MODELS = ("snail", "lake", "geese", "wolves", "cladocera")
ROUNDS = ("normal", "long", "very_long")


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def passes(meta, engine):
    required = (
        meta.get("max_rhat") is not None and meta["max_rhat"] <= 1.05
        and meta.get("min_ess_bulk") is not None and meta["min_ess_bulk"] >= 100
        and meta.get("min_ess_tail") is not None and meta["min_ess_tail"] >= 100
    )
    if engine == "python":
        required = (
            required
            and meta.get("divergences", 0) == 0
            and meta.get("treedepth_hits", 0) == 0
            and meta.get("min_bfmi") is not None
            and meta["min_bfmi"] > 0.30
        )
    return bool(required)


rows = []
for model in MODELS:
    for engine in ("r", "python"):
        candidates = []
        for round_name in ROUNDS:
            path = SOURCE / model / round_name / engine / "metadata.json"
            meta = read(path)
            if meta:
                candidates.append((round_name, meta))
        selected = next(((r, m) for r, m in candidates if passes(m, engine)), None)
        if selected is None and candidates:
            selected = candidates[-1]
        if selected is None:
            rows.append({"model": model, "engine": engine, "status": "NOT_RUN"})
            continue
        round_name, meta = selected
        rows.append({
            "model": model,
            "engine": "R/MixSIAR–JAGS" if engine == "r" else "MixSIAPy–PyMC",
            "status": "CONVERGED" if passes(meta, engine) else "UNCONVERGED_AT_MAXIMUM",
            "selected_round": round_name,
            "wall_seconds": meta.get("elapsed_seconds"),
            "chains": meta.get("chains"),
            "retained_draws_per_chain": meta.get("retained_draws_per_chain"),
            "max_rhat": meta.get("max_rhat"),
            "min_ess_bulk": meta.get("min_ess_bulk"),
            "min_ess_tail": meta.get("min_ess_tail"),
            "max_mcse_over_sd": meta.get("max_mcse_over_sd"),
            "divergences": meta.get("divergences") if engine == "python" else None,
            "treedepth_hits": meta.get("treedepth_hits") if engine == "python" else None,
            "min_bfmi": meta.get("min_bfmi") if engine == "python" else None,
        })

OUT.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame(rows)
frame.to_csv(OUT / "r_vs_python_matched_precision.csv", index=False)
print(frame.to_string(index=False))
