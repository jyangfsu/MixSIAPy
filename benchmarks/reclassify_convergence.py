"""Reclassify completed numerical-agreement fits without resampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "numerical_agreement_v2"
MARKERS = ("CONVERGED", "NEEDS_LONGER_RUN", "NOT_CONVERGED")


def classify(meta: dict, engine: str) -> str:
    rhat = float(meta.get("max_rhat", float("inf")))
    bulk = float(meta.get("min_ess_bulk", 0))
    tail = float(meta.get("min_ess_tail", 0))
    sampler_ok = True
    if engine == "python":
        sampler_ok = (
            int(meta.get("divergences", 0)) == 0
            and int(meta.get("treedepth_hits", 0)) == 0
            and float(meta.get("min_bfmi", 0)) > 0.30
        )
    if rhat < 1.05 and bulk >= 100 and tail >= 100 and sampler_ok:
        return "CONVERGED"
    if rhat < 1.10 and bulk >= 50 and tail >= 50 and sampler_ok:
        return "NEEDS_LONGER_RUN"
    return "NOT_CONVERGED"


rows = []
for metadata_path in ROOT.glob("*/*/*/metadata.json"):
    engine = metadata_path.parent.name
    if engine not in {"r", "python"}:
        continue
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    status = classify(meta, engine)
    meta["convergence_status"] = status
    meta["convergence_rule"] = (
        "max_rhat < 1.05; min bulk/tail ESS >= 100"
        + ("; zero divergences/treedepth hits; BFMI > 0.30" if engine == "python" else "")
    )
    text = json.dumps(meta, indent=2)
    metadata_path.write_text(text, encoding="utf-8")
    done = metadata_path.parent / "DONE"
    if done.exists():
        done.write_text(text, encoding="utf-8")
    for marker in MARKERS:
        (metadata_path.parent / marker).unlink(missing_ok=True)
    (metadata_path.parent / status).write_text(text, encoding="utf-8")
    rows.append({
        "unit": meta.get("unit"),
        "round": metadata_path.parents[1].name,
        "engine": engine,
        "convergence_status": status,
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "chains": meta.get("chains"),
        "retained_draws_per_chain": meta.get("retained_draws_per_chain"),
        "max_rhat": meta.get("max_rhat"),
        "min_ess_bulk": meta.get("min_ess_bulk"),
        "min_ess_tail": meta.get("min_ess_tail"),
        "min_bulk_ess_per_second": meta.get("min_bulk_ess_per_second"),
        "divergences": meta.get("divergences"),
        "treedepth_hits": meta.get("treedepth_hits"),
        "min_bfmi": meta.get("min_bfmi"),
    })

rows.sort(key=lambda row: (str(row["unit"]), str(row["round"]), str(row["engine"])))
if rows:
    with (ROOT / "performance_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"fits_reclassified": len(rows)}, indent=2))
