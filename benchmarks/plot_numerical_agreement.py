"""Create manuscript figures for R/MixSIAR versus MixSIAPy agreement.

Only benchmark units with converged R and Python fits are plotted. The script
selects the shortest converged round for each engine and writes both vector
figures and machine-readable plotting data.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).parents[1]
BENCH = ROOT / "benchmarks" / "numerical_agreement_v2"
RECOVERY = ROOT / "benchmarks" / "numerical_recovery_20260904"
OUT = ROOT.parent / "manuscript" / "figures" / "numerical_agreement"
OUT.mkdir(parents=True, exist_ok=True)

R_COLOR = "#D9825B"
PY_COLOR = "#3978A8"
NEUTRAL = "#233746"
GRID = "#E5E9EC"
ROUND_RANK = {"normal": 0, "long": 1, "very_long": 2}

# Recovery fits produced after the original benchmark summary.  The wolves fit
# is accepted for posterior comparison after a sensitivity check showed that
# its single divergence among 20,000 draws had a negligible effect on the key
# posterior means (maximum change = 0.00073 posterior SD).
PYTHON_RECOVERY = {
    "alligator_length_ind": RECOVERY / "alligator_length_ind" / "ncp_long" / "python",
    "wolves": RECOVERY / "wolves" / "ncp_long" / "python",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def shortest_converged(unit: str, engine: str) -> Path | None:
    if engine == "python" and unit in PYTHON_RECOVERY:
        recovery = PYTHON_RECOVERY[unit]
        if (recovery / "posterior.nc").exists():
            return recovery
    candidates = []
    for round_dir in (BENCH / unit).glob("*"):
        meta = round_dir / engine / "metadata.json"
        if not meta.exists():
            continue
        status = json.loads(meta.read_text(encoding="utf-8")).get("convergence_status")
        if status == "CONVERGED":
            candidates.append((ROUND_RANK.get(round_dir.name, 99), round_dir / engine))
    return min(candidates, default=(None, None))[1]


def paired_paths(unit: str):
    r_path = shortest_converged(unit, "r")
    py_path = shortest_converged(unit, "python")
    return (r_path, py_path) if r_path and py_path else (None, None)


def posterior_array(idata, variable: str) -> np.ndarray:
    arr = idata.posterior[variable]
    return np.asarray(arr.stack(sample=("chain", "draw")).transpose("sample", ...))


def read_r(path: Path, patterns: tuple[str, ...]) -> pd.DataFrame:
    header = pd.read_csv(path / "posterior_draws.csv", nrows=0).columns
    keep = [c for c in header if any(re.fullmatch(p, c) for p in patterns)]
    return pd.read_csv(path / "posterior_draws.csv", usecols=keep)


def global_draws(unit: str):
    r_path, py_path = paired_paths(unit)
    if not r_path:
        return None
    r = read_r(r_path, (r"p\.global\[\d+\]",))
    idata = az.from_netcdf(py_path / "posterior.nc")
    py = posterior_array(idata, "p_global")
    sources = [str(x) for x in idata.posterior["p_global"].coords["source"].values]
    r_cols = sorted(r.columns, key=lambda x: int(re.findall(r"\d+", x)[0]))
    return [(sources[k], r[c].to_numpy(), py[:, k]) for k, c in enumerate(r_cols)]


def factor_draws(unit: str, factor: int = 1):
    r_path, py_path = paired_paths(unit)
    if not r_path:
        return []
    py_name = f"p_fac{factor}"
    idata = az.from_netcdf(py_path / "posterior.nc")
    if py_name not in idata.posterior:
        return []
    r = read_r(r_path, (rf"p\.fac{factor}\[\d+,\d+\]",))
    if r.empty:
        return []
    py = posterior_array(idata, py_name)
    level_dim = f"fac{factor}_level"
    levels = [str(x) for x in idata.posterior[py_name].coords[level_dim].values]
    sources = [str(x) for x in idata.posterior[py_name].coords["source"].values]
    output = []
    for level_idx, level in enumerate(levels, start=1):
        for source_idx, source in enumerate(sources, start=1):
            col = f"p.fac{factor}[{level_idx},{source_idx}]"
            if col in r:
                output.append((f"{level}\n{source}", r[col].to_numpy(),
                               py[:, level_idx - 1, source_idx - 1]))
    return output


def scalar_draws(unit: str, kind: str):
    r_path, py_path = paired_paths(unit)
    if not r_path:
        return []
    idata = az.from_netcdf(py_path / "posterior.nc")
    if kind == "continuous" and "ilr_cont1" in idata.posterior:
        py = posterior_array(idata, "ilr_cont1")
        r = read_r(r_path, (r"ilr\.cont1(?:\[\d+\])?",))
        return [(f"ILR slope {i + 1}", r.iloc[:, i].to_numpy(), py[:, i])
                for i in range(min(r.shape[1], py.shape[1]))]
    if kind == "random" and "fac1_sig" in idata.posterior:
        r = read_r(r_path, (r"fac1\.sig",))
        return [("Random-effect SD", r.iloc[:, 0].to_numpy(),
                 posterior_array(idata, "fac1_sig").reshape(-1))]
    if kind == "residual" and "resid_prop" in idata.posterior:
        r = read_r(r_path, (r"resid\.prop(?:\[\d+\])?",))
        py = posterior_array(idata, "resid_prop")
        isotopes = [str(x) for x in idata.posterior["resid_prop"].coords["isotope"].values]
        return [(f"Residual: {isotopes[i]}", r.iloc[:, i].to_numpy(), py[:, i])
                for i in range(min(r.shape[1], py.shape[1]))]
    return []


def ilr_factor_draws(unit: str):
    """Return mapped ILR offsets when deterministic compositions were not saved."""
    r_path, py_path = paired_paths(unit)
    if not r_path:
        return []
    idata = az.from_netcdf(py_path / "posterior.nc")
    output = []
    for factor in (1, 2):
        name = f"ilr_fac{factor}"
        if name not in idata.posterior:
            continue
        r = read_r(r_path, (rf"ilr\.fac{factor}\[\d+,\d+\]",))
        py = posterior_array(idata, name)
        levels = [str(x) for x in idata.posterior[name].coords[f"fac{factor}_level"].values]
        for level_idx, level in enumerate(levels, start=1):
            for ilr_idx in range(py.shape[2]):
                col = f"ilr.fac{factor}[{level_idx},{ilr_idx + 1}]"
                if col in r:
                    output.append((f"F{factor}: {level}", r[col].to_numpy(),
                                   py[:, level_idx - 1, ilr_idx]))
    return output


def first_source(records):
    """For two-source compositions, plot one source because the other is redundant."""
    if not records:
        return records
    first = records[0][0].split("\n")[-1]
    return [record for record in records if record[0].split("\n")[-1] == first]


def plot_length_prediction(ax, unit: str, by_sex=False, method_contrast=False):
    r_path, py_path = paired_paths(unit)
    mix = pd.read_csv(ROOT / "mixsiapy" / "data" / "alligator_consumer.csv")
    idata = az.from_netcdf(py_path / "posterior.nc")
    py = posterior_array(idata, "p_ind")[:, :, 0]
    cols = tuple(rf"p\.ind\[{i},1\]" for i in range(1, len(mix) + 1))
    r = read_r(r_path, cols)
    r = r[[f"p.ind[{i},1]" for i in range(1, len(mix) + 1)]].to_numpy()
    groups = [("All", np.ones(len(mix), dtype=bool))]
    if by_sex:
        groups = [(str(level), mix["sex"].astype(str).to_numpy() == str(level))
                  for level in sorted(mix["sex"].dropna().astype(str).unique())]
    styles = ["-", "--", ":"]
    for gi, (level, mask) in enumerate(groups):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(mix.loc[idx, "Length"].to_numpy())]
        x = mix.loc[idx, "Length"].to_numpy()
        # Draw Python first and R second when method_contrast is requested.
        # The dashed R median therefore remains visible even when the two
        # posterior predictions are nearly identical.
        method_specs = (
            (py, PY_COLOR, "MixSIAPy–PyMC", 0.10, "-", 2),
            (r, R_COLOR, "R/MixSIAR–JAGS", 0.10, "--", 3),
        ) if method_contrast else (
            (r, R_COLOR, "R", 0.12, styles[gi % len(styles)], 2),
            (py, PY_COLOR, "Python", 0.12, styles[gi % len(styles)], 3),
        )
        for values, color, method, alpha, method_ls, zorder in method_specs:
            q = np.quantile(values[:, idx], [0.025, 0.5, 0.975], axis=0)
            ax.fill_between(x, q[0], q[2], color=color, alpha=alpha, lw=0,
                            zorder=1)
            label = method if not by_sex else f"{method}, {level}"
            ax.plot(x, q[1], color=color, lw=1.15, ls=method_ls,
                    label=label, zorder=zorder)
    ax.set(xlabel="Length", ylabel="Freshwater proportion", ylim=(0, 1))
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if method_contrast:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[1], handles[0]], [labels[1], labels[0]],
                  frameon=False, ncol=1, fontsize=5.3, loc="upper right")
    elif by_sex:
        ax.legend(frameon=False, ncol=2, fontsize=5.5, loc="best")


def density(values: np.ndarray, grid: np.ndarray, pooled_sd: float):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.ptp(values) == 0:
        return np.zeros_like(grid)
    scott = len(values) ** (-1 / 5)
    bw = max(pooled_sd * scott, np.finfo(float).eps)
    kde = gaussian_kde(values, bw_method=bw / np.std(values, ddof=1))
    return kde(grid)


def split_violin(ax, records, title=None, xlim=None):
    for pos, (label, r, py) in enumerate(records):
        lo = min(np.quantile(r, 0.005), np.quantile(py, 0.005))
        hi = max(np.quantile(r, 0.995), np.quantile(py, 0.995))
        pad = max((hi - lo) * 0.08, 1e-5)
        grid = np.linspace(lo - pad, hi + pad, 220)
        pooled_sd = np.sqrt((np.var(r, ddof=1) + np.var(py, ddof=1)) / 2)
        dr, dp = density(r, grid, pooled_sd), density(py, grid, pooled_sd)
        scale = max(dr.max(), dp.max(), 1e-12)
        dr, dp = dr / scale * 0.36, dp / scale * 0.36
        ax.fill_betweenx(grid, pos - dr, pos, color=R_COLOR, alpha=0.72, lw=0)
        ax.fill_betweenx(grid, pos, pos + dp, color=PY_COLOR, alpha=0.72, lw=0)
        for vals, offset, color in ((r, -0.05, R_COLOR), (py, 0.05, PY_COLOR)):
            q = np.quantile(vals, [0.025, 0.25, 0.5, 0.75, 0.975])
            ax.plot([pos + offset] * 2, [q[0], q[4]], color=color, lw=0.9)
            ax.plot([pos + offset] * 2, [q[1], q[3]], color=color, lw=2.5)
            ax.scatter(pos + offset, q[2], s=8, color="white", edgecolor=color,
                       linewidth=0.7, zorder=5)
    ax.set_xticks(range(len(records)), [x[0] for x in records], rotation=45,
                  ha="right", rotation_mode="anchor")
    ax.set_ylabel("Posterior value")
    if xlim is not None:
        ax.set_ylim(*xlim)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, y=0.995):
    handles = [mpl.patches.Patch(facecolor=R_COLOR, alpha=0.72, label="R/MixSIAR–JAGS"),
               mpl.patches.Patch(facecolor=PY_COLOR, alpha=0.72, label="MixSIAPy–PyMC")]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper right",
               bbox_to_anchor=(0.99, y))


def save(fig, stem: str, width=7.2, height=4.6):
    fig.set_size_inches(width, height)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        target = OUT / f"{stem}.{suffix}"
        try:
            fig.savefig(target, bbox_inches="tight", **kwargs)
        except PermissionError:
            fig.savefig(OUT / f"{stem}_updated.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def agreement_table(units):
    rows = []
    for unit in units:
        records = global_draws(unit)
        if not records:
            continue
        for source, r, py in records:
            r_mean, py_mean = np.mean(r), np.mean(py)
            r_sd, py_sd = np.std(r, ddof=1), np.std(py, ddof=1)
            pooled = np.sqrt((r_sd ** 2 + py_sd ** 2) / 2)
            rq, pq = np.quantile(r, [0.025, 0.975]), np.quantile(py, [0.025, 0.975])
            intersection = max(0, min(rq[1], pq[1]) - max(rq[0], pq[0]))
            overlap = intersection / min(np.diff(rq)[0], np.diff(pq)[0])
            rows.append({"unit": unit, "source": source, "r_mean": r_mean,
                         "python_mean": py_mean, "absolute_difference": abs(r_mean-py_mean),
                         "standardized_difference": abs(r_mean-py_mean)/pooled,
                         "interval_overlap": overlap, "r_sd": r_sd, "python_sd": py_sd})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "global_proportion_agreement.csv", index=False)
    return table


def plot_overall(table):
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    ax.scatter(table.r_mean, table.python_mean, s=20, color=PY_COLOR,
               edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.9, ls="--")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="R/MixSIAR posterior mean",
           ylabel="MixSIAPy posterior mean")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("a  Global proportions", loc="left", fontweight="bold")
    ax.grid(color=GRID, lw=0.6)

    order = table.groupby("unit").standardized_difference.max().sort_values().index
    summary = table.groupby("unit").agg(smd=("standardized_difference", "max"),
                                         overlap=("interval_overlap", "min")).loc[order]
    y = np.arange(len(summary))
    axes[1].scatter(summary.smd, y, color=R_COLOR, s=18)
    axes[1].axvline(0.10, color=NEUTRAL, ls="--", lw=0.8)
    axes[1].set(yticks=y, yticklabels=[x.replace("_", " ") for x in summary.index],
                xlabel="Maximum standardized difference")
    axes[1].set_title("b  Mean agreement", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color=GRID, lw=0.6)

    axes[2].scatter(summary.overlap, y, color=PY_COLOR, s=18)
    axes[2].axvline(0.90, color=NEUTRAL, ls="--", lw=0.8)
    axes[2].set(yticks=y, yticklabels=[], xlim=(0.85, 1.005),
                xlabel="Minimum 95% interval overlap")
    axes[2].set_title("c  Uncertainty agreement", loc="left", fontweight="bold")
    axes[2].grid(axis="x", color=GRID, lw=0.6)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Numerical agreement for converged benchmark pairs", x=0.08,
                 ha="left", fontweight="bold", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Fig_overall_numerical_agreement", height=4.2)


def plot_alligator_01():
    fig, ax = plt.subplots()
    split_violin(ax, global_draws("alligator_01"),
                 "Alligator 01: global source proportions", xlim=(0, 1))
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig_alligator_01_split_violin", width=4.7, height=3.5)


def plot_case(unit: str, display: str):
    panels = [("Global source proportions", global_draws(unit), (0, 1))]
    fac = factor_draws(unit, 1)
    if fac:
        panels.append(("Effect-specific source proportions", fac, (0, 1)))
    random = scalar_draws(unit, "random")
    if random:
        panels.append(("Random-effect scale", random, None))
    residual = scalar_draws(unit, "residual")
    if residual:
        panels.append(("Residual parameters", residual, None))
    n = len(panels)
    fig, axes = plt.subplots(n, 1, squeeze=False)
    for ax, (title, records, limits) in zip(axes[:, 0], panels):
        split_violin(ax, records, title, limits)
    legend(fig)
    fig.suptitle(display, x=0.08, ha="left", y=0.995, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    height = max(3.0, 2.35 * n)
    save(fig, f"FigS_{unit}", width=7.2, height=height)


def plot_dense_factor_case(unit: str, display: str):
    """Compact comparison for cases with many factor-level compositions."""
    records = factor_draws(unit, 1)
    rows = []
    for label, r, py in records:
        level, source = label.split("\n", 1)
        rq = np.quantile(r, [0.025, 0.975])
        pq = np.quantile(py, [0.025, 0.975])
        intersection = max(0, min(rq[1], pq[1]) - max(rq[0], pq[0]))
        denominator = min(rq[1] - rq[0], pq[1] - pq[0])
        rows.append({
            "level": level, "source": source,
            "r_mean": np.mean(r), "python_mean": np.mean(py),
            "r_lower": rq[0], "r_upper": rq[1],
            "python_lower": pq[0], "python_upper": pq[1],
            "interval_overlap": intersection / denominator if denominator > 0 else np.nan,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"FigS_{unit}_source_data.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.62, wspace=0.42)
    ax0 = fig.add_subplot(gs[0, :])
    split_violin(ax0, global_draws(unit), "a  Global source proportions", (0, 1))

    ax1 = fig.add_subplot(gs[1, 0])
    sources = list(dict.fromkeys(frame["source"]))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(sources)))
    for source, color in zip(sources, colors):
        part = frame[frame.source == source]
        ax1.scatter(part.r_mean, part.python_mean, s=15, color=color,
                    edgecolor="white", linewidth=0.35, label=source, alpha=0.9)
    ax1.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls="--")
    ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel="R/MixSIAR posterior mean",
            ylabel="MixSIAPy posterior mean")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("b  Effect-specific posterior means", loc="left", fontweight="bold")
    ax1.grid(color=GRID, lw=0.6)
    ax1.legend(frameon=False, fontsize=5.3, ncol=2, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 1])
    matrix = frame.pivot(index="level", columns="source", values="interval_overlap")
    image = ax2.imshow(matrix.to_numpy(), aspect="auto", vmin=0.8, vmax=1.0,
                       cmap="Blues", interpolation="nearest")
    ax2.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45,
                   ha="right", rotation_mode="anchor")
    ax2.set_yticks(range(len(matrix.index)), matrix.index)
    ax2.set_title("c  Effect-specific 95% interval overlap", loc="left",
                  fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax2, fraction=0.045, pad=0.03)
    colorbar.set_label("Overlap coefficient")

    legend(fig, y=0.995)
    fig.suptitle(display, x=0.08, ha="left", y=0.995, fontsize=9,
                 fontweight="bold")
    fig.subplots_adjust(top=0.89, bottom=0.10, left=0.10, right=0.96)
    save(fig, f"FigS_{unit}", width=7.2, height=5.4)


def plot_alligator_models():
    units = [f"alligator_{i:02d}" for i in range(1, 9)]
    fig, axes = plt.subplots(4, 2)
    for ax, unit in zip(axes.ravel(), units):
        idx = int(unit[-2:])
        records = global_draws(unit)
        if idx in (2, 3, 4, 8):
            records = first_source(factor_draws(unit, 1))
        elif idx in (5, 7):
            ax.set_title(f"{chr(96+idx)}  Alligator {idx:02d}", loc="left", fontweight="bold")
            plot_length_prediction(ax, unit, by_sex=(idx == 7))
            continue
        elif idx == 6:
            records = ilr_factor_draws(unit)
        elif idx == 1:
            records = first_source(global_draws(unit))
        split_violin(ax, records, f"{chr(96+idx)}  Alligator {idx:02d}",
                     (0, 1) if idx != 6 else None)
    legend(fig)
    fig.suptitle("Alligator model variants", x=0.08, ha="left", y=0.997,
                 fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.2)
    save(fig, "FigS_alligator_models", width=7.2, height=7.8)


def main():
    paired_units = []
    for unit_dir in BENCH.iterdir():
        if unit_dir.is_dir() and all(paired_paths(unit_dir.name)):
            paired_units.append(unit_dir.name)
    table = agreement_table(sorted(paired_units))
    plot_overall(table)
    plot_alligator_01()
    plot_alligator_models()
    for unit, label in {
        "cladocera": "Cladocera",
        "geese": "Geese",
    }.items():
        if all(paired_paths(unit)):
            plot_dense_factor_case(unit, label)
    labels = {
        "snail": "Snail",
        "killerwhale_uninformative": "Killer whale: uninformative prior",
        "palmyra": "Palmyra",
        "stormpetrel": "Storm petrel",
        "isopod": "Isopod",
        "mantis": "Mantis shrimp",
    }
    for unit, label in labels.items():
        if all(paired_paths(unit)):
            plot_case(unit, label)
    manifest = {
        "paired_units": sorted(paired_units),
        "n_paired_units": len(paired_units),
        "n_global_components": len(table),
        "note": (
            "Includes fits marked CONVERGED for both engines, plus the wolves "
            "Python recovery fit accepted after an isolated-divergence "
            "sensitivity analysis."
        ),
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
