"""The main figure of the measurement report, from the numbers of the run files. No project code."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

numbers = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])

SURFACE, PRIMARY, SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NO_EEG, EEG = "#eb6834", "#2a78d6"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": PRIMARY,
        "axes.labelcolor": SECONDARY,
        "xtick.color": SECONDARY,
        "ytick.color": PRIMARY,
        "axes.edgecolor": GRID,
        "svg.fonttype": "none",
    }
)

drug = numbers["drug_alone_known"]
rows = [
    ("Maintenance drug listed (138 patients)", drug[0], drug[1], drug[2], NO_EEG, True),
    ("Sessions per patient", 0.771, 0.699, 0.841, NO_EEG, True),
    ("Segments per patient", 0.761, 0.680, 0.833, NO_EEG, False),
    ("Spectral", 0.693, 0.608, 0.776, EEG, True),
    ("HYDRA", 0.670, 0.582, 0.752, EEG, True),
    ("Spectral, matched age and sex", 0.680, 0.564, 0.788, EEG, False),
    ("HYDRA, matched age and sex", 0.659, 0.547, 0.764, EEG, False),
    ("S4-R1 (R=1 code)", 0.619, 0.530, 0.707, EEG, False),
    ("S4-untrained", 0.527, 0.435, 0.619, EEG, False),
    ("S4-proj (peak rate)", 0.483, 0.392, 0.577, EEG, True),
]
fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.62})
fig.patch.set_facecolor(SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=GRID)

y = np.arange(len(rows))[::-1]
left.axvline(0.5, color=SECONDARY, lw=1, ls=(0, (3, 3)), zorder=1)
left.text(0.5, len(rows) - 0.42, "chance", ha="center", va="bottom", color=SECONDARY, fontsize=9)
for i, (name, value, low, high, color, label) in zip(y, rows):
    left.plot([low, high], [i, i], color=color, lw=2, solid_capstyle="round", zorder=2)
    left.scatter([value], [i], s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    if label:
        left.text(high + 0.012, i, f"{value:.2f}", va="center", color=PRIMARY, fontsize=9.5)
left.set_yticks(y, [r[0] for r in rows])
left.set_xlim(0.36, 1.0)
left.set_xlabel("AUROC, with 95% patient bootstrap interval")
left.set_title(
    "Information with no EEG reads above every EEG arm", loc="left", fontsize=11.5, fontweight="bold", pad=16
)
left.scatter([], [], s=64, color=NO_EEG, label="No EEG content")
left.scatter([], [], s=64, color=EEG, label="EEG arm")
left.legend(loc="lower right", frameon=False, fontsize=9.5, handletextpad=0.2, labelcolor=SECONDARY)

cells = numbers["Spectral_cells"]
names = list(cells)
rng = np.random.default_rng(3)
positions = np.arange(len(names))[::-1]
right.axvline(0.0, color=GRID, lw=1, zorder=1)
for position, name in zip(positions, names):
    cell = cells[name]
    values = np.array(cell["values"])
    jitter = (
        rng.uniform(-0.15, 0.15, len(values))
        if len(values) > 4
        else np.array([-0.09, 0.09, -0.03, 0.03])[: len(values)]
    )
    right.scatter(values, position + 0.2 + jitter, s=20, color="#b9b8b2", edgecolor=SURFACE, linewidth=0.5, zorder=2)
    right.plot([cell["low"], cell["high"]], [position - 0.18] * 2, color=EEG, lw=2, solid_capstyle="round", zorder=3)
    right.scatter([cell["mean"]], [position - 0.18], s=64, color=EEG, edgecolor=SURFACE, linewidth=2, zorder=4)
right.set_yticks(positions, [f"{n}\n({cells[n]['n']} patients)" for n in names])
right.set_ylim(-0.6, len(names) - 0.4)
right.set_xlabel("Spectral score, standardized within fold")
right.set_title(
    "In the EEG score, only treated epilepsy stands out", loc="left", fontsize=11.5, fontweight="bold", pad=16
)
right.text(
    0.99,
    0.02,
    "Gray: patients. Blue: mean, 95% interval.",
    transform=right.transAxes,
    ha="right",
    color=SECONDARY,
    fontsize=9,
)
for suffix in (".png", ".svg"):
    fig.savefig(out.with_suffix(suffix), dpi=200, bbox_inches="tight", facecolor=SURFACE)
