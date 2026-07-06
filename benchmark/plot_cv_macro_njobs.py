import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

df = pd.read_csv("benchmark/results/macrobenchmark.csv")
df = df[df["version"] == "rust"]

UNIT = "ms"
n_jobs_vals = sorted(df["n_jobs"].unique())
word_vals = sorted(df["n_unique_words"].unique())
length_vals = [10_000, 100_000, 1_000_000]

GRID_COLS = 1
GRID_ROWS = len(length_vals)

fig, axes = plt.subplots(
    GRID_ROWS,
    GRID_COLS,
    figsize=(8, 3.4 * GRID_ROWS),
    squeeze=False,
)

njobs_colors = {
    1: "#C4C4C4",
    2: "#525252",
    4: "#f47e7e",
    8: "#620000",
    24: "#c1272d",
}

n_bars = len(n_jobs_vals)
total_width = 0.7
bar_width = total_width / n_bars
offsets = np.linspace(
    -(total_width - bar_width) / 2, (total_width - bar_width) / 2, n_bars
)

for i, length in enumerate(length_vals):
    ax = axes[i][0]
    sub = df[df["dataset_length"] == length]
    x = np.arange(len(word_vals))

    for offset, n_jobs in zip(offsets, n_jobs_vals):
        times = []
        for w in word_vals:
            row = sub[(sub.n_unique_words == w) & (sub.n_jobs == n_jobs)]
            times.append(row.time_ms.iloc[0] if not row.empty else float("nan"))
        ax.bar(x + offset, times, bar_width, color=njobs_colors[n_jobs])

    ax.set_xticks(x)
    ax.set_xticklabels(word_vals)
    ax.set_title(f"n_documents = {length:,}")
    ax.set_xlabel("n_unique_words")
    ax.set_ylabel(f"time ({UNIT})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

legend_handles = [Patch(color=njobs_colors[j], label=str(j)) for j in n_jobs_vals]
fig.legend(
    handles=legend_handles,
    title="n_jobs",
    ncol=len(n_jobs_vals),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
)
fig.suptitle("Rust CountVectorizer fit_transform — by n_jobs", fontsize=14, y=1.06)
fig.tight_layout()
fig.savefig(
    f"benchmark/results/macrobenchmark_njobs_{sys.argv[1]}.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
