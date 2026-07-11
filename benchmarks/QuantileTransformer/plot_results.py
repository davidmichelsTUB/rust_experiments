import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd


def plot_benchmark_data(title, xlabel, ylabel, x, y_rust_uniform, y_rust_normal, y_sklearn_uniform, y_sklearn_normal):
    width = 0.4
    x_pos = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_rust_uniform = ax.bar(x_pos - width/2, y_rust_uniform, width, label="Rust - Uniform", color='blue')
    bars_rust_normal = ax.bar(x_pos - width/2, y_rust_normal, width, bottom=y_rust_uniform, label="Rust - Normal", color='lightblue')
    bars_sklearn_uniform = ax.bar(x_pos + width/2, y_sklearn_uniform, width, label="Scikit-Learn - Uniform", color='orange')
    bars_sklearn_normal = ax.bar(x_pos + width/2, y_sklearn_normal, width, bottom=y_sklearn_uniform, label="Scikit-Learn - Normal", color='red')

    ax.bar_label(bars_rust_uniform, fmt="%.2f")
    ax.bar_label(bars_rust_normal, fmt="%.2f")
    ax.bar_label(bars_sklearn_uniform, fmt="%.2f")
    ax.bar_label(bars_sklearn_normal, fmt="%.2f")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for QuantileTransformer.")
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file containing benchmark results.")
    parser.add_argument("--func", type=str, choices=["fit", "transform", "fit_transform"], help="Function to plot results for.")

    args = parser.parse_args()

    results_df = pd.read_csv(args.csv_path)


    results_df_uniform = results_df[
    (results_df["kwargs"].str.contains("uniform")) &
    (results_df["kwargs"].str.contains("1000,")) &
    (results_df["cols"] == 500)
]

    results_df_normal = results_df[
        (results_df["kwargs"].str.contains("normal")) &
        (results_df["kwargs"].str.contains("1000,")) &
        (results_df["cols"] == 500)
    ]

    rows = results_df_uniform["rows"]
    
    print(results_df_uniform)

    fig = plot_benchmark_data(
        title=f"QuantileTransformer Benchmark with 500 columns - {args.func}",
        xlabel="Number of Rows",
        ylabel="Time (seconds)",
        x=rows,
        y_rust_uniform=results_df_uniform["parallel_mean"],
        y_rust_normal=results_df_normal["parallel_mean"],
        y_sklearn_uniform=results_df_uniform["sklearn_mean"],
        y_sklearn_normal=results_df_normal["sklearn_mean"]
    )

    plt.savefig(f"quantile_transformer_benchmark_{args.func}.png")






if __name__ == "__main__":
    main()

