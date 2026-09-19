"""Create standalone panel (d): GPU chain-scaling bubble plot.

Each bubble summarizes three paired CPU/GPU NumPyro runs. Bubble area encodes
the median GPU speed-up (CPU wall time / GPU wall time), while color indicates
which device was faster. Numeric labels retain the exact median values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 8.1,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "validation"
    / "performance_benchmark"
    / "gpu_chain_scaling"
    / "gpu_chain_scaling_paired.csv"
)
OUT = ROOT.parent / "manuscript" / "figures" / "performance"
OUT.mkdir(parents=True, exist_ok=True)
BASE = OUT / "Fig_GPU_chain_scaling_bubble_d"

MODEL_ORDER = ["snail", "lake", "geese", "wolves", "cladocera"]
MODEL_LABELS = {
    "snail": "Snail",
    "lake": "Lake",
    "geese": "Geese",
    "wolves": "Wolves",
    "cladocera": "Cladocera",
}
CHAIN_ORDER = [4, 8, 16, 32, 64]

GPU_COLOR = "#357FA3"       # restrained blue: GPU faster
CPU_COLOR = "#D27A52"       # restrained orange: CPU faster
EDGE_COLOR = "#FFFFFF"
GRID_COLOR = "#E4E8EA"
TEXT_COLOR = "#27323A"
SUBTLE_TEXT = "#63717A"


def bubble_area(speedup: np.ndarray | float) -> np.ndarray | float:
    """Map speed-up to visible marker area while preserving monotonicity."""
    values = np.asarray(speedup, dtype=float)
    # A small baseline keeps severe CPU-faster cases visible; area then grows
    # linearly with the reported GPU speed-up.
    area = 42.0 + 92.0 * values
    return float(area) if np.ndim(speedup) == 0 else area


runs = pd.read_csv(DATA)
required = {"model", "chains", "gpu_time_speedup"}
missing = required.difference(runs.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

summary = (
    runs.groupby(["model", "chains"], as_index=False)["gpu_time_speedup"]
    .agg(median_speedup="median", q25_speedup=lambda x: x.quantile(0.25),
         q75_speedup=lambda x: x.quantile(0.75), n="size")
)
summary = summary[
    summary["model"].isin(MODEL_ORDER) & summary["chains"].isin(CHAIN_ORDER)
].copy()
summary["device_faster"] = np.where(
    summary["median_speedup"] > 1.0, "GPU faster", "CPU faster"
)
summary["model_order"] = pd.Categorical(
    summary["model"], categories=MODEL_ORDER, ordered=True
)
summary = summary.sort_values(["model_order", "chains"])

if len(summary) != len(MODEL_ORDER) * len(CHAIN_ORDER):
    raise ValueError("The bubble grid is incomplete; expected 25 model-chain cells.")

fig, ax = plt.subplots(figsize=(5.55, 3.35))
fig.subplots_adjust(left=0.16, right=0.985, top=0.83, bottom=0.25)

x_lookup = {chain: idx for idx, chain in enumerate(CHAIN_ORDER)}
# Put Snail at the top and Cladocera at the bottom, matching earlier figures.
y_lookup = {model: len(MODEL_ORDER) - 1 - idx for idx, model in enumerate(MODEL_ORDER)}

for row in summary.itertuples(index=False):
    x = x_lookup[int(row.chains)]
    y = y_lookup[str(row.model)]
    color = GPU_COLOR if row.median_speedup > 1.0 else CPU_COLOR
    size = bubble_area(row.median_speedup)
    ax.scatter(
        x,
        y,
        s=size,
        color=color,
        edgecolor=EDGE_COLOR,
        linewidth=0.9,
        alpha=0.96,
        zorder=3,
    )
    # Exact values make the nonlinear visual encoding auditable.
    label_color = "white" if size >= 92 else TEXT_COLOR
    if size >= 92:
        ax.text(
            x, y, f"{row.median_speedup:.2f}",
            ha="center", va="center", fontsize=6.4,
            fontweight="bold", color=label_color, zorder=4,
        )
    else:
        ax.annotate(
            f"{row.median_speedup:.2f}", (x, y), xytext=(0, 9),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=6.1, color=TEXT_COLOR, zorder=4,
        )

ax.set_xticks(range(len(CHAIN_ORDER)), [str(v) for v in CHAIN_ORDER])
ax.set_yticks(
    [y_lookup[m] for m in MODEL_ORDER],
    [MODEL_LABELS[m] for m in MODEL_ORDER],
)
ax.set_xlim(-0.55, len(CHAIN_ORDER) - 0.28)
ax.set_ylim(-0.55, len(MODEL_ORDER) - 0.45)
ax.set_xlabel("Number of parallel chains")
ax.set_ylabel("Benchmark model")

ax.set_axisbelow(True)
ax.grid(axis="both", color=GRID_COLOR, linewidth=0.65)
ax.tick_params(axis="both", length=0, pad=5)
ax.spines["left"].set_color("#606970")
ax.spines["bottom"].set_color("#606970")

ax.text(
    -0.12, 1.105, "(d)", transform=ax.transAxes,
    fontsize=10.0, fontweight="bold", ha="left", va="bottom",
)
ax.set_title(
    "GPU acceleration across increasing chain counts",
    loc="left", fontweight="bold", pad=18,
)
ax.text(
    0.0, 1.025,
    "NumPyro; bubble labels show median CPU time / GPU time from three paired runs",
    transform=ax.transAxes, fontsize=7.0, color=SUBTLE_TEXT,
    ha="left", va="bottom",
)

direction_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CPU_COLOR,
           markeredgecolor="white", markersize=7.2, label="CPU faster"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=GPU_COLOR,
           markeredgecolor="white", markersize=7.2, label="GPU faster"),
]
direction_legend = ax.legend(
    handles=direction_handles,
    loc="upper center",
    bbox_to_anchor=(0.28, -0.22),
    ncol=2,
    handletextpad=0.45,
    columnspacing=1.25,
)
ax.add_artist(direction_legend)

size_values = [0.25, 1.0, 2.0, 3.0]
size_handles = [
    plt.scatter([], [], s=bubble_area(value), color="#AAB3B8",
                edgecolor="white", linewidth=0.8, label=f"{value:g}×")
    for value in size_values
]
ax.legend(
    handles=size_handles,
    title="GPU speed-up",
    loc="upper center",
    bbox_to_anchor=(0.77, -0.188),
    ncol=4,
    handletextpad=0.15,
    columnspacing=0.55,
    borderaxespad=0,
    title_fontsize=7.0,
)

summary.drop(columns="model_order").to_csv(
    OUT / "Fig_GPU_chain_scaling_bubble_d_source_data.csv", index=False
)
fig.savefig(BASE.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
fig.savefig(BASE.with_suffix(".png"), dpi=350, bbox_inches="tight")
plt.close(fig)

print(BASE)
