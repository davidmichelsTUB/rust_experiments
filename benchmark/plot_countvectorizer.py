from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "results" / "countvectorizer.csv")

df["label"] = df.apply(
    lambda r: "sklearn" if r.backend == "sklearn" else f"rust ({int(r.n_chunks)} threads)",
    axis=1,
)

summary = (
    df.groupby("label")[["fit_ms", "transform_ms"]]
    .mean()
    .reindex(["sklearn", "rust (1 threads)", f"rust ({df.n_chunks.max()} threads)"])
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

for ax, col, title in zip(axes, ["fit_ms", "transform_ms"], ["Fit", "Transform"]):
    bars = ax.bar(summary.index, summary[col], color=["steelblue", "tomato", "seagreen"])
    ax.set_title(title)
    ax.set_ylabel("ms (mean of 5 runs)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{bar.get_height():.0f}",
            ha="center", va="bottom", fontsize=9,
        )

fig.suptitle("CountVectorizer benchmark", fontsize=13)
fig.tight_layout()

out = HERE / "results" / "countvectorizer.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
