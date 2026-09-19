"""Create compact four-panel computational-performance figure (version 4).

Panels:
  (a) all 20 normal-schedule R/Python fits;
  (b) wall time to the first acceptable posterior, retaining all 20 table rows;
  (c) normal-schedule CPU-backend efficiency for five representative models;
  (d) NumPyro CPU/GPU chain-scaling bubble plot.
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
        "font.size": 7.1,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.7,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "benchmarks"
AGREEMENT = VALIDATION / "numerical_agreement_v2"
PERFORMANCE = VALIDATION / "performance_benchmark"
WSL_SUMMARY = PERFORMANCE / "wsl_summary"
GPU_SCALING = PERFORMANCE / "gpu_chain_scaling"
OUT = ROOT.parent / "manuscript" / "figures" / "performance"
OUT.mkdir(parents=True, exist_ok=True)
BASE = OUT / "Fig_computational_performance_v4"

# Order and spelling exactly match Table 1.
BENCHMARKS = [
    "alligator_01", "alligator_02", "alligator_03", "alligator_04",
    "alligator_05", "alligator_06", "alligator_07", "alligator_08",
    "alligator_length_ind", "cladocera", "geese", "isopod",
    "killerwhale_informative", "killerwhale_uninformative", "lake",
    "mantis", "palmyra", "snail", "stormpetrel", "wolves",
]
TABLE_LABEL = {
    "alligator_01": "Alligator01",
    "alligator_02": "Alligator02",
    "alligator_03": "Alligator03",
    "alligator_04": "Alligator04",
    "alligator_05": "Alligator05",
    "alligator_06": "Alligator06",
    "alligator_07": "Alligator07",
    "alligator_08": "Alligator08",
    "alligator_length_ind": "Alligator_length_ind",
    "cladocera": "Cladocera",
    "geese": "Geese",
    "isopod": "Isopod",
    "killerwhale_informative": "Killerwhale_informative",
    "killerwhale_uninformative": "Killerwhale_uninformative",
    "lake": "Lake",
    "mantis": "Mantis",
    "palmyra": "Palmyra",
    "snail": "Snail",
    "stormpetrel": "Stormpetrel",
    "wolves": "Wolves",
}
REPRESENTATIVE = ["snail", "lake", "geese", "wolves", "cladocera"]
REP_LABEL = {m: TABLE_LABEL[m] for m in REPRESENTATIVE}
CHAINS = [4, 8, 16, 32, 64]

# Match the method colors used in the numerical-agreement figure.
R_COLOR = "#7A628F"
PY_COLOR = "#267C78"
GPU_COLOR = "#337FA8"
CPU_COLOR = "#D47A51"
BACKEND_COLOR = {
    "pymc": "#337FA8",
    "nutpie": "#6B9B57",
    "numpyro": "#8366A8",
    "blackjax": "#D47A51",
}
GRID = "#E1E6E9"
SUBTLE = "#65727A"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.19) -> None:
    ax.text(
        x, 1.075, f"({label})", transform=ax.transAxes,
        fontsize=9.0, fontweight="bold", ha="left", va="bottom",
    )


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.58)
    ax.tick_params(axis="y", length=2.8, width=0.65, color="#5B666D", pad=3.5)
    ax.spines["left"].set_color("#5B666D")
    ax.spines["bottom"].set_color("#5B666D")


def paired_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    status_col: str | None,
    show_inconclusive: bool = False,
) -> None:
    y = np.arange(len(BENCHMARKS), dtype=float)
    wide = frame.pivot(index="unit", columns="engine", values="elapsed_seconds")
    for yi, model in enumerate(BENCHMARKS):
        if model in wide.index and {"r", "python"}.issubset(wide.columns):
            pair = wide.loc[model, ["r", "python"]]
            if pair.notna().all():
                ax.plot(pair.to_numpy(float), [yi, yi], color="#C9D0D4", lw=0.72, zorder=1)

    for engine, color, offset in [("r", R_COLOR, -0.095), ("python", PY_COLOR, 0.095)]:
        part = frame[frame["engine"] == engine].set_index("unit")
        for yi, model in enumerate(BENCHMARKS):
            if model not in part.index:
                continue
            row = part.loc[model]
            value = row["elapsed_seconds"]
            if pd.isna(value):
                continue
            passed = True if status_col is None else row[status_col] == "CONVERGED"
            ax.scatter(
                float(value), yi + offset, s=17,
                facecolor=color if passed else "white", edgecolor=color,
                linewidth=0.9, zorder=3,
            )

    if show_inconclusive:
        yi = BENCHMARKS.index("killerwhale_informative")
        xmin, xmax = ax.get_xlim()
        ax.text(
            xmin * 1.12, yi, "× inconclusive", color="#747E85",
            fontsize=5.8, ha="left", va="center", zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([TABLE_LABEL[m] for m in BENCHMARKS])
    ax.set_ylim(len(BENCHMARKS) - 0.35, -0.65)


# ------------------------------- data ------------------------------------
performance = pd.read_csv(AGREEMENT / "performance_summary.csv")
normal = performance[
    (performance["round"] == "normal") & performance["unit"].isin(BENCHMARKS)
].copy()

accepted = pd.read_csv(PERFORMANCE / "all_accepted_r_python_speedups.csv")
accepted_rows: list[dict] = []
for row in accepted.itertuples(index=False):
    accepted_rows.extend(
        [
            {"unit": row.benchmark, "engine": "r", "elapsed_seconds": row.r_seconds},
            {"unit": row.benchmark, "engine": "python", "elapsed_seconds": row.python_seconds},
        ]
    )
accepted_frame = pd.DataFrame(accepted_rows)

cpu = pd.read_csv(WSL_SUMMARY / "python_cpu_backend_summary.csv")
cpu["backend"] = cpu["method"].str.replace("_cpu", "", regex=False)
cpu["diagnostics_met"] = (
    (cpu["worst_rhat"] <= 1.05)
    & (cpu["lowest_bulk_ess"] >= 100)
    & (cpu["lowest_tail_ess"] >= 100)
    & (cpu["total_divergences"] == 0)
)

scaling_runs = pd.read_csv(GPU_SCALING / "gpu_chain_scaling_paired.csv")
scaling = (
    scaling_runs.groupby(["model", "chains"], as_index=False)
    .agg(
        median_speedup=("gpu_time_speedup", "median"),
        q25_speedup=("gpu_time_speedup", lambda x: x.quantile(0.25)),
        q75_speedup=("gpu_time_speedup", lambda x: x.quantile(0.75)),
        repeats=("repeat", "count"),
    )
)

if len(normal) != 40:
    raise ValueError(f"Expected 40 normal-fit rows, found {len(normal)}")
if accepted["benchmark"].nunique() != 19:
    raise ValueError("Expected 19 converged paired analyses")
if len(scaling) != 25:
    raise ValueError("Expected a complete five-model by five-chain scaling grid")


# ------------------------------ layout -----------------------------------
# A compact double-column layout; the upper row retains all 20 benchmark rows.
fig = plt.figure(figsize=(7.20, 6.45))
gs = fig.add_gridspec(
    2, 2, height_ratios=[1.36, 1.0],
    left=0.198, right=0.985, bottom=0.085, top=0.955,
    wspace=0.54, hspace=0.55,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])


# (a) Equal retained output under each implementation's normal schedule.
ax_a.set_xscale("log")
paired_points(ax_a, normal, "convergence_status")
ax_a.set_xlabel("Normal-schedule wall time (s)")
ax_a.set_title("(a)  Fixed-output comparison", loc="left", fontweight="bold", y=1.07, pad=0)
ax_a.text(
    0, 1.012, "20 analyses; 3,000 retained draws per implementation",
    transform=ax_a.transAxes, fontsize=6.25, color=SUBTLE, va="bottom",
)
style_axis(ax_a)


# (b) Earliest schedule satisfying the convergence requirements.
ax_b.set_xscale("log")
# Establish a stable range before placing the inconclusive annotation.
ax_b.set_xlim(40, 60000)
paired_points(ax_b, accepted_frame, None, show_inconclusive=True)
ax_b.set_xlabel("Wall time to acceptable posterior (s)")
ax_b.set_title("(b)  Matched-convergence comparison", loc="left", fontweight="bold", y=1.07, pad=0)
ax_b.text(
    0, 1.012, "19 acceptable pairs; one inconclusive analysis retained on axis",
    transform=ax_b.transAxes, fontsize=6.25, color=SUBTLE, va="bottom",
)
style_axis(ax_b)


# Shared legend for panels (a) and (b), reducing their vertical footprint.
top_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=R_COLOR,
           markeredgecolor=R_COLOR, markersize=4.6, label="MixSIAR–JAGS"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PY_COLOR,
           markeredgecolor=PY_COLOR, markersize=4.6, label="MixSIAPy–PyMC"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor="#68737B", markersize=4.6, label="Diagnostics not met"),
]
top_legend_ax = fig.add_axes([0.20, 0.445, 0.785, 0.038])
top_legend_ax.set_axis_off()
top_legend_ax.legend(
    handles=top_handles, ncol=3, loc="center",
    handletextpad=0.35, columnspacing=0.8,
)


# (c) Fixed normal schedule for the four CPU backends.
y5 = np.arange(len(REPRESENTATIVE))
backend_order = ["pymc", "nutpie", "numpyro", "blackjax"]
offset = {"pymc": -0.21, "nutpie": -0.07, "numpyro": 0.07, "blackjax": 0.21}
for backend in backend_order:
    part = cpu[cpu["backend"] == backend].set_index("model").loc[REPRESENTATIVE]
    med = part["median_min_bulk_ess_per_second"].to_numpy(float)
    lo = part["q25_min_bulk_ess_per_second"].to_numpy(float)
    hi = part["q75_min_bulk_ess_per_second"].to_numpy(float)
    for i in range(len(REPRESENTATIVE)):
        passed = bool(part.iloc[i]["diagnostics_met"])
        ax_c.errorbar(
            med[i], y5[i] + offset[backend],
            xerr=np.array([[med[i] - lo[i]], [hi[i] - med[i]]]),
            fmt="o", ms=3.9, capsize=1.8, elinewidth=0.75,
            color=BACKEND_COLOR[backend],
            markerfacecolor=BACKEND_COLOR[backend] if passed else "white",
            markeredgewidth=0.95, zorder=3,
        )
ax_c.set_xscale("log")
ax_c.set_yticks(y5, [REP_LABEL[m] for m in REPRESENTATIVE])
ax_c.set_ylim(len(REPRESENTATIVE) - 0.55, -0.55)
ax_c.set_xlabel("Minimum bulk ESS s$^{-1}$")
ax_c.set_title("(c)  CPU backend efficiency", loc="left", fontweight="bold", y=1.07, pad=0)
ax_c.text(
    0, 1.012, "Normal schedule; median and interquartile range (n = 3)",
    transform=ax_c.transAxes, fontsize=6.25, color=SUBTLE, va="bottom",
)
style_axis(ax_c)
ax_c.legend(
    handles=[
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=BACKEND_COLOR[b], markeredgecolor=BACKEND_COLOR[b],
               markersize=4.5,
               label={"pymc": "PyMC", "nutpie": "Nutpie", "numpyro": "NumPyro", "blackjax": "BlackJAX"}[b])
        for b in backend_order
    ] + [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#68737B", markersize=4.5, label="Diagnostics not met")
    ],
    ncol=3, loc="upper center", bbox_to_anchor=(0.50, -0.28),
    handletextpad=0.28, columnspacing=0.55,
)


# (d) Bubble grid: size is median GPU speed-up; color identifies faster device.
def bubble_area(value: float | np.ndarray) -> float | np.ndarray:
    values = np.asarray(value, dtype=float)
    # Restore the stronger visual separation used in the first bubble version.
    areas = 22.0 + 52.0 * values
    return float(areas) if np.ndim(value) == 0 else areas


x_lookup = {chain: i for i, chain in enumerate(CHAINS)}
y_lookup = {model: len(REPRESENTATIVE) - 1 - i for i, model in enumerate(REPRESENTATIVE)}
for row in scaling.itertuples(index=False):
    if row.model not in y_lookup or row.chains not in x_lookup:
        continue
    x = x_lookup[row.chains]
    y = y_lookup[row.model]
    color = GPU_COLOR if row.median_speedup > 1 else CPU_COLOR
    area = bubble_area(row.median_speedup)
    ax_d.scatter(x, y, s=area, color=color, edgecolor="white", linewidth=0.75, zorder=3)
    ax_d.annotate(
        f"{row.median_speedup:.2f}", (x, y),
        xytext=(0, 5.8 + 0.12 * np.sqrt(area)), textcoords="offset points",
        ha="center", va="bottom", fontsize=5.1, color="#2F3940", zorder=4,
    )

ax_d.set_xticks(range(len(CHAINS)), [str(c) for c in CHAINS])
ax_d.set_yticks(
    [y_lookup[m] for m in REPRESENTATIVE],
    [REP_LABEL[m] for m in REPRESENTATIVE],
)
ax_d.set_xlim(-0.52, len(CHAINS) - 0.28)
ax_d.set_ylim(-0.52, len(REPRESENTATIVE) - 0.52)
ax_d.set_xlabel("Number of parallel chains")
ax_d.set_title("(d)  GPU scaling", loc="left", fontweight="bold", y=1.07, pad=0)
ax_d.text(
    0, 1.012, "NumPyro; labels show median CPU time / GPU time (n = 3)",
    transform=ax_d.transAxes, fontsize=6.25, color=SUBTLE, va="bottom",
)
style_axis(ax_d, grid_axis="both")
ax_d.tick_params(axis="both", length=0)

direction_legend = ax_d.legend(
    handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CPU_COLOR,
               markeredgecolor="white", markersize=5.5, label="CPU faster"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=GPU_COLOR,
               markeredgecolor="white", markersize=5.5, label="GPU faster"),
    ],
    ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
    handletextpad=0.35, columnspacing=0.9,
)
ax_d.add_artist(direction_legend)

size_values = [0.25, 1.0, 2.0, 3.0]
size_handles = [
    ax_d.scatter(
        [], [], s=bubble_area(value), color="#A8B1B6",
        edgecolor="white", linewidth=0.65, label=f"{value:g}×",
    )
    for value in size_values
]
ax_d.legend(
    handles=size_handles,
    title="GPU speed-up",
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.31),
    handletextpad=0.18,
    columnspacing=0.62,
    borderaxespad=0,
    fontsize=5.8,
    title_fontsize=6.2,
)


# ------------------------------ export -----------------------------------
fig.savefig(BASE.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
fig.savefig(BASE.with_suffix(".png"), dpi=350, bbox_inches="tight")
plt.close(fig)

normal.to_csv(OUT / "Fig_computational_performance_v4_normal_fits.csv", index=False)
accepted.to_csv(OUT / "Fig_computational_performance_v4_matched_convergence.csv", index=False)
cpu.to_csv(OUT / "Fig_computational_performance_v4_cpu_backends.csv", index=False)
scaling.to_csv(OUT / "Fig_computational_performance_v4_gpu_scaling.csv", index=False)

print(BASE)
