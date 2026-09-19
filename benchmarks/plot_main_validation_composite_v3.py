"""Create the revised main-text numerical-agreement composite.

This version keeps all 20 benchmark analyses visible in the two model-level
summary panels.  The informative-prior killer-whale analysis is labelled as
inconclusive and is not assigned an agreement statistic.
"""
from __future__ import annotations

from pathlib import Path
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_numerical_agreement as pna


# Muted purple/teal pairing: the two implementations remain distinguishable
# in colour and in grayscale, without the strong orange/blue contrast used in
# the first version.
R_COLOR = "#7A628F"
PY_COLOR = "#267C78"
PAIR_COLOR = "#314A5A"
NEUTRAL = "#4D5961"
TEXT_GREY = "#53646D"
GRID = "#E1E7E9"
INCONCLUSIVE = "#9AA3A8"

# The helper functions use module-level colours; update them before drawing.
pna.R_COLOR = R_COLOR
pna.PY_COLOR = PY_COLOR
pna.NEUTRAL = NEUTRAL
pna.GRID = GRID

OUT = pna.OUT

ALL_UNITS = [
    "alligator_01", "alligator_02", "alligator_03", "alligator_04",
    "alligator_05", "alligator_06", "alligator_07", "alligator_08",
    "alligator_length_ind", "cladocera", "geese", "isopod",
    "killerwhale_informative", "killerwhale_uninformative", "lake",
    "mantis", "palmyra", "snail", "stormpetrel", "wolves",
]

DISPLAY_LABELS = {
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

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.6,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def first_level(records):
    if not records:
        return []
    level = records[0][0].split("\n", 1)[0]
    return [
        (label.split("\n", 1)[-1], r, py)
        for label, r, py in records
        if label.split("\n", 1)[0] == level
    ]


def short_labels(records):
    replacements = {
        "Marine Mammals": "Marine\nmammals",
        "Random-effect SD": "Site SD",
    }
    return [(replacements.get(label, label), r, py) for label, r, py in records]


def panel_title(ax, letter, heading, subtitle=None, pad=7,
                subtitle_fontsize=7.1, subtitle_y=0.90):
    ax.set_title(f"({letter})  {heading}", loc="left", fontweight="bold", pad=pad)
    if subtitle:
        ax.text(
            0, subtitle_y, subtitle, transform=ax.transAxes,
            ha="left", va="bottom", color=TEXT_GREY,
            fontsize=subtitle_fontsize,
        )


def violin_panel(ax, records, letter, heading, subtitle, limits=None,
                 subtitle_fontsize=7.1, subtitle_y=1.012):
    pna.split_violin(ax, short_labels(records), None, limits)
    panel_title(ax, letter, heading, subtitle, pad=8,
                subtitle_fontsize=subtitle_fontsize,
                subtitle_y=subtitle_y)
    ax.set_ylabel("Posterior value")
    ax.tick_params(axis="x", pad=1)


def add_source_rows(source_rows, section, subtype, example, records):
    for parameter, r, py in records:
        for engine, values in (("R/MixSIAR–JAGS", r), ("MixSIAPy–PyMC", py)):
            q = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
            source_rows.append({
                "section": section,
                "subtype": subtype,
                "example": example,
                "parameter": parameter,
                "engine": engine,
                "mean": np.mean(values),
                "sd": np.std(values, ddof=1),
                "q2.5": q[0],
                "q25": q[1],
                "median": q[2],
                "q75": q[3],
                "q97.5": q[4],
                "draws": len(values),
            })


def model_summary(overall):
    measured = overall.groupby("unit").agg(
        smd=("standardized_difference", "max"),
        overlap=("interval_overlap", "min"),
    )
    summary = measured.reindex(ALL_UNITS)
    summary["status"] = np.where(summary["smd"].notna(), "compared", "inconclusive")
    summary["label"] = [DISPLAY_LABELS[x] for x in summary.index]
    return summary


def model_metric_panel(ax, summary, metric, letter, title, xlabel, threshold,
                       xlim, color):
    y = np.arange(len(summary))[::-1]
    valid = summary[metric].notna().to_numpy()
    values = summary[metric].to_numpy(float)
    ax.scatter(values[valid], y[valid], s=19, color=color,
               edgecolor="white", linewidth=0.45, zorder=3)
    ax.axvline(threshold, color=NEUTRAL, ls="--", lw=0.9, zorder=1)

    # Retain the twentieth benchmark row without inventing a numerical value.
    missing_y = y[~valid]
    if len(missing_y):
        note_x = xlim[0] + 0.055 * (xlim[1] - xlim[0])
        ax.scatter([note_x] * len(missing_y), missing_y, marker="x", s=24,
                   color=INCONCLUSIVE, linewidth=1.2, zorder=4)
        for yi in missing_y:
            ax.text(note_x + 0.035 * (xlim[1] - xlim[0]), yi, "inconclusive",
                    va="center", ha="left", color=TEXT_GREY, fontsize=6.2)

    ax.set(
        yticks=y,
        yticklabels=summary["label"].tolist(),
        xlabel=xlabel,
        xlim=xlim,
    )
    ax.tick_params(axis="y", labelsize=6.3, colors=TEXT_GREY, pad=2)
    ax.grid(axis="x", color=GRID, lw=0.55, zorder=0)
    panel_title(ax, letter, title)


def main():
    comparable = [
        unit for unit in ALL_UNITS
        if unit != "killerwhale_informative" and all(pna.paired_paths(unit))
    ]
    overall = pna.agreement_table(comparable)
    summary = model_summary(overall)
    rows = []

    # Quantitative grid: one overview row followed by three evidence rows.
    fig = plt.figure(figsize=(8.4, 9.75))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1.52, 1.0, 1.0, 1.0],
        hspace=0.54,
        wspace=0.62,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(3)]

    # (a) Pairwise agreement in posterior means.
    ax = axes[0]
    ax.scatter(
        overall.r_mean, overall.python_mean,
        s=20, color=PAIR_COLOR, edgecolor="white", linewidth=0.5, alpha=0.88,
    )
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.9, ls="--")
    r = float(np.corrcoef(overall.r_mean, overall.python_mean)[0, 1])
    r2 = r * r
    ax.text(
        0.10, 0.90, rf"$R^2$ = {r2:.4f}", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8.2, color='k',
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=GRID, lw=0.7),
    )
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="MixSIAR posterior mean",
        ylabel="MixSIAPy posterior mean",
    )
    ax.set_aspect("equal", adjustable="box")
    panel_title(ax, "a", "Overall agreement")
    ax.grid(color=GRID, lw=0.55)

    # (b, c) All twenty benchmark analyses remain visible.  The one model that
    # did not meet convergence criteria is explicitly labelled inconclusive.
    model_metric_panel(
        axes[1], summary, "smd", "b", "Posterior means",
        "Maximum standardized mean difference", 0.10, (0, 0.145), R_COLOR,
    )
    model_metric_panel(
        axes[2], summary, "overlap", "c", "Posterior uncertainty",
        "Minimum 95% interval overlap", 0.90, (0.84, 1.006), PY_COLOR,
    )

    # Three source-data representations.
    source_panels = [
        ("Summary data", "   Alligator01 with source means, SD and $n_k$", "Alligator01",
         pna.global_draws("alligator_01")),
        ("Raw observations", "   Palmyra with source covariance retained", "Palmyra",
         pna.global_draws("palmyra")),
        ("Stratified sources", "   Mantis with habitat-specific signatures",
         "Mantis shrimp", first_level(pna.factor_draws("mantis", 1))),
    ]
    for idx, (heading, subtitle, example, records) in enumerate(source_panels, 3):
        letter = string.ascii_lowercase[idx]
        violin_panel(axes[idx], records, letter, heading, subtitle, (0, 1),
                     subtitle_fontsize=7.8, subtitle_y=0.95)
        add_source_rows(rows, "source data", heading, example, records)

    # Representative effect structures.
    fixed = first_level(pna.factor_draws("stormpetrel", 1))
    violin_panel(axes[6], fixed, "g", "Fixed effect",
                 "   Stormpetrel with one regional level", (0, 1),
                 subtitle_fontsize=7.8, subtitle_y=0.95)
    add_source_rows(rows, "effects", "fixed", "Storm petrel", fixed)

    pna.plot_length_prediction(axes[7], "alligator_05", method_contrast=True)
    panel_title(axes[7], "h", "Continuous effect", "   Alligator05 with body length",
                pad=8, subtitle_fontsize=7.8, subtitle_y=0.95)
    h_handles, h_labels = axes[7].get_legend_handles_labels()
    axes[7].legend(
        [h_handles[1], h_handles[0]], [h_labels[1], h_labels[0]],
        frameon=False, fontsize=5.8, loc="lower left",
    )

    random = pna.scalar_draws("isopod", "random")
    violin_panel(axes[8], random, "i", "Random effect", "   Isopod with among-site SD",
                 subtitle_fontsize=7.8, subtitle_y=0.95)
    add_source_rows(rows, "effects", "random", "Isopod", random)

    # Make categorical labels closer to horizontal so the rows can be packed
    # more tightly. Panel h already uses horizontal numeric tick labels.
    for idx in (3, 4, 5, 6, 8):
        plt.setp(axes[idx].get_xticklabels(), rotation=24,
                 ha="right", rotation_mode="anchor")

    # Three error structures.
    process_resid = pna.scalar_draws("mantis", "residual")
    violin_panel(axes[9], process_resid, "j", "Process × residual",
                 "   Mantis with residual multipliers",(0, 1),
                 subtitle_fontsize=7.8, subtitle_y=0.95)
    add_source_rows(rows, "error", "process × residual", "Mantis shrimp",
                    process_resid)

    residual_only = first_level(pna.factor_draws("geese", 1))
    violin_panel(axes[10], residual_only, "k", "Residual only",
                 "   Geese with group composition", (0, 1),
                 subtitle_fontsize=7.8, subtitle_y=0.95)
    add_source_rows(rows, "error", "residual only", "Geese", residual_only)

    process_only = first_level(pna.factor_draws("cladocera", 1))
    violin_panel(axes[11], process_only, "l", "Process only",
                 "   Cladocera with individual composition", (0, 1),
                 subtitle_fontsize=7.8, subtitle_y=0.95)
    add_source_rows(rows, "error", "process only", "Cladocera", process_only)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpl.patches.Patch(facecolor=R_COLOR, alpha=0.72, label="R/MixSIAR–JAGS"),
        mpl.patches.Patch(facecolor=PY_COLOR, alpha=0.72, label="MixSIAPy–PyMC"),
    ]
    fig.legend(handles=handles, ncol=2, loc="lower center",
               bbox_to_anchor=(0.50, 0.006), frameon=False)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.975, bottom=0.09)
    # Place each section heading relative to the actual subplot geometry so
    # later changes in figure height cannot make it collide with panel titles.
    for row_start, label in (
        (3, "SOURCE-DATA REPRESENTATION"),
        (6, "EFFECT STRUCTURE"),
        (9, "ERROR STRUCTURE"),
    ):
        y = axes[row_start].get_position().y1 + 0.030
        fig.text(0.055, y, label, fontsize=7.4, fontweight="bold",
                 color=TEXT_GREY, ha="left", va="bottom")

    stem = OUT / "Fig_main_numerical_validation_composite_v3"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        OUT / "Fig_main_numerical_validation_composite_v3_source_data.csv",
        index=False,
    )
    summary.reset_index(names="benchmark").to_csv(
        OUT / "Fig_main_numerical_validation_composite_v3_overview_data.csv",
        index=False,
    )
    print(stem)


if __name__ == "__main__":
    main()
