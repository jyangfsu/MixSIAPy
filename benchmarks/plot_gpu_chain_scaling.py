"""Create the manuscript figure for the NumPyro CPU/GPU chain-scaling test.

The figure separates absolute execution cost from relative GPU speed-up.  It
uses the three paired repetitions produced by ``run_gpu_chain_scaling.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep text editable in the SVG and PDF exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "performance_benchmark" / "gpu_chain_scaling"
RUNS_FILE = DATA_DIR / "gpu_chain_scaling_paired.csv"
OUT_DIR = ROOT.parent / "manuscript" / "figures" / "performance"
OUT_BASE = OUT_DIR / "Fig_gpu_chain_scaling"

MODEL_ORDER = ["snail", "lake", "geese", "wolves", "cladocera"]
MODEL_LABEL = {
    "cladocera": "Cladocera",
    "geese": "Geese",
    "wolves": "Wolves",
    "snail": "Snail",
    "lake": "Lake",
}
MODEL_COLOR = {
    "cladocera": "#254B87",
    "geese": "#2A8C82",
    "wolves": "#8064A2",
    "snail": "#C47A35",
    "lake": "#6F7782",
}
CHAINS = [4, 8, 16, 32, 64]


def q25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def q75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["model", "chains"], as_index=False).agg(
        repeats=("repeat", "count"),
        cpu_median=("execution_seconds_cpu", "median"),
        cpu_q25=("execution_seconds_cpu", q25),
        cpu_q75=("execution_seconds_cpu", q75),
        gpu_median=("execution_seconds_gpu", "median"),
        gpu_q25=("execution_seconds_gpu", q25),
        gpu_q75=("execution_seconds_gpu", q75),
        speedup_median=("gpu_time_speedup", "median"),
        speedup_q25=("gpu_time_speedup", q25),
        speedup_q75=("gpu_time_speedup", q75),
    )
    return grouped


def draw_time_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows: list[list[float]] = []
    labels: list[str] = []
    for model in MODEL_ORDER:
        block = summary.loc[summary["model"] == model].set_index("chains")
        rows.append([block.loc[c, "cpu_median"] for c in CHAINS])
        rows.append([block.loc[c, "gpu_median"] for c in CHAINS])
        labels.extend([f"{MODEL_LABEL[model]}  CPU", "                 GPU"])

    values = np.asarray(rows, dtype=float)
    norm = mcolors.LogNorm(vmin=10, vmax=1600)
    image = ax.imshow(values, cmap="YlGnBu", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(CHAINS)), CHAINS)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Number of chains")
    ax.set_title("Median execution time", loc="left", fontweight="bold", pad=7)
    ax.tick_params(length=0)

    for boundary in [1.5, 3.5, 5.5, 7.5]:
        ax.axhline(boundary, color="white", lw=2.0)

    cmap = plt.get_cmap("YlGnBu")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.52 else "#222222"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=6.2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.075,
        pad=0.16,
        aspect=28,
        ticks=[10, 30, 100, 300, 1000],
    )
    cbar.ax.set_xticklabels(["10", "30", "100", "300", "1,000"])
    cbar.set_label("Execution time (s; log scale)", labelpad=2)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2.5, width=0.5, labelsize=6.4)
    panel_label(ax, "a")


def draw_speedup(ax: plt.Axes, summary: pd.DataFrame) -> None:
    # A subtle region communicates the direction of benefit without using a
    # saturated red/green pass-fail scheme.
    ax.axhspan(1, 4.2, color="#E6F2EF", zorder=0)
    ax.axhline(1, color="#3E4650", lw=0.9, ls=(0, (4, 3)), zorder=1)

    for model in MODEL_ORDER:
        block = (
            summary.loc[summary["model"] == model]
            .sort_values("chains")
            .set_index("chains")
            .loc[CHAINS]
        )
        x = np.asarray(CHAINS, dtype=float)
        med = block["speedup_median"].to_numpy(dtype=float)
        lo = block["speedup_q25"].to_numpy(dtype=float)
        hi = block["speedup_q75"].to_numpy(dtype=float)
        color = MODEL_COLOR[model]
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(x, med, color=color, lw=1.45, marker="o", ms=3.8, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(3.7, 92)
    ax.set_ylim(0.04, 4.2)
    ax.set_xticks(CHAINS, [str(c) for c in CHAINS])
    ax.set_yticks([0.0625, 0.125, 0.25, 0.5, 1, 2, 4], ["0.06", "0.13", "0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Number of chains")
    ax.set_ylabel("GPU speed-up (CPU time / GPU time)")
    ax.set_title("GPU acceleration emerges with parallel workload", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="y", which="major", color="#D9DDE2", lw=0.45, zorder=0)
    ax.text(4.1, 1.08, "GPU faster", color="#396F68", fontsize=6.6, va="bottom")
    ax.text(4.1, 0.91, "CPU faster", color="#666666", fontsize=6.6, va="top")

    label_y = {
        "cladocera": 3.25,
        "wolves": 1.76,
        "geese": 1.30,
        "snail": 0.70,
        "lake": 0.31,
    }
    final_values = {}
    for model in MODEL_ORDER:
        final_value = float(
            summary.loc[(summary["model"] == model) & (summary["chains"] == 64), "speedup_median"].iloc[0]
        )
        final_values[model] = final_value
        ax.plot([64, 69], [final_value, label_y[model]], color=MODEL_COLOR[model], lw=0.6)
        ax.text(
            71,
            label_y[model],
            f"{MODEL_LABEL[model]}  {final_value:.2f}×",
            color=MODEL_COLOR[model],
            fontsize=6.6,
            fontweight="bold" if model == "cladocera" else "normal",
            ha="left",
            va="center",
        )

    panel_label(ax, "b")


def main() -> None:
    runs = pd.read_csv(RUNS_FILE)
    expected = len(MODEL_ORDER) * len(CHAINS) * 3
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} paired runs, found {len(runs)}")
    if runs[["execution_seconds_cpu", "execution_seconds_gpu"]].isna().any().any():
        raise ValueError("Execution-time data contain missing values")

    summary = summarize_runs(runs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "Fig_gpu_chain_scaling_source_data.csv", index=False)

    fig = plt.figure(figsize=(7.20, 3.45), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.91, 1.22],
        left=0.105,
        right=0.965,
        bottom=0.19,
        top=0.91,
        wspace=0.31,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    draw_time_heatmap(ax_a, summary)
    draw_speedup(ax_b, summary)

    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
