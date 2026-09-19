"""Summarize all converged R/Python pairs and five illustrative speedups."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "benchmarks" / "numerical_agreement_v2"
OUT = ROOT / "benchmarks" / "performance_benchmark"
RECOVERY = ROOT / "benchmarks" / "numerical_recovery_20260904"
ROUNDS = ("normal", "long", "very_long")
SELECTED = (
    "alligator_01", "alligator_04", "alligator_05",
    "alligator_06", "mantis",
)
PYTHON_RECOVERY = {
    "alligator_length_ind": RECOVERY / "alligator_length_ind" / "ncp_long" / "python",
    "wolves": RECOVERY / "wolves" / "ncp_long" / "python",
}


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def passes(meta, engine):
    ok = (
        meta.get("max_rhat") is not None and meta["max_rhat"] <= 1.05
        and meta.get("min_ess_bulk") is not None and meta["min_ess_bulk"] >= 100
        and meta.get("min_ess_tail") is not None and meta["min_ess_tail"] >= 100
    )
    if engine == "python":
        ok = (
            ok and meta.get("divergences", 0) == 0
            and meta.get("treedepth_hits", 0) == 0
            and meta.get("min_bfmi") is not None and meta["min_bfmi"] > 0.30
        )
    return bool(ok)


rows = []
for unit_dir in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
    selected = {}
    for engine in ("r", "python"):
        for round_name in ROUNDS:
            meta = read(unit_dir / round_name / engine / "metadata.json")
            if meta and passes(meta, engine):
                selected[engine] = (round_name, meta)
                break
        if engine == "python" and engine not in selected and unit_dir.name in PYTHON_RECOVERY:
            recovery = read(PYTHON_RECOVERY[unit_dir.name] / "metadata.json")
            if recovery:
                recovery_ok = passes(recovery, engine)
                # The wolves recovery fit had one divergence among 20,000
                # retained draws. It was accepted for the numerical benchmark
                # after removal of that draw changed key posterior means by at
                # most 0.00073 posterior SD.
                if unit_dir.name == "wolves":
                    recovery_ok = (
                        recovery.get("max_rhat", 99) <= 1.05
                        and recovery.get("min_ess_bulk", 0) >= 100
                        and recovery.get("min_ess_tail", 0) >= 100
                        and recovery.get("divergences") == 1
                        and recovery.get("treedepth_hits", 0) == 0
                        and recovery.get("min_bfmi", 0) > 0.30
                    )
                if recovery_ok:
                    selected[engine] = ("recovery", recovery)
    if len(selected) != 2:
        continue
    r_round, r_meta = selected["r"]
    py_round, py_meta = selected["python"]
    rows.append({
        "benchmark": unit_dir.name,
        "r_round": r_round,
        "r_seconds": r_meta["elapsed_seconds"],
        "python_round": py_round,
        "python_seconds": py_meta["elapsed_seconds"],
        "r_over_python_speedup": (
            r_meta["elapsed_seconds"] / py_meta["elapsed_seconds"]
        ),
        "selected_illustrative_case": unit_dir.name in SELECTED,
        "python_acceptance_basis": (
            "single-divergence sensitivity check"
            if unit_dir.name == "wolves" else "standard diagnostics"
        ),
    })

OUT.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame(rows).sort_values("r_over_python_speedup", ascending=False)
frame.to_csv(OUT / "all_accepted_r_python_speedups.csv", index=False)
frame[frame.selected_illustrative_case].to_csv(
    OUT / "five_additional_speedup_cases.csv", index=False,
)
print(frame.to_string(index=False))
