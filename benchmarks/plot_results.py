import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


_ROWS = [
    100,
    500,
    1000,
    5000,
    10000,
    25_000,
    # 50_000,
    # 100_000,
    # 500_000,
    # 1_000_000,
    # 5_000_000
]

_COLUMNS = [
    10,
    50,
    100,
    500,
    1000,
    # 5000,
    # 10_000,
    # 100_000,
    # 500_000,
]


def plot_benchmark_data(title, xlabel, ylabel, x, y_rust, y_sklearn):
    width = 0.4
    x_pos = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_rust = ax.bar(x_pos - width/2, y_rust, width, label="Rust")
    bars_sklearn = ax.bar(x_pos + width/2, y_sklearn, width, label="Scikit-Learn")

    ax.bar_label(bars_rust, fmt="%.2f")
    ax.bar_label(bars_sklearn, fmt="%.2f")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    return fig

def plot_for_fixed_cols(columns: int, df: pd.DataFrame):
    df = df.loc[df["cols"] == columns, :]
    df.drop(columns=["cols"], inplace=True)

    rows = df["rows"].to_numpy()
    fit_time = np.array(df["fit_time"].map(lambda x: [float(val) for val in x[1:-1].split(",")]).tolist(), dtype="float64") * 1000
    predict_time = np.array(df["predict_time"].map(lambda x: [float(val) for val in x[1:-1].split(",")]).tolist(), dtype="float64") * 1000
    accuracies = np.array(df["acc"].map(lambda x : [float(val[len("np.float64("):-1]) for val in x[1:-1].split(", ")]).tolist(), dtype="float64")

    fig_fit_time = plot_benchmark_data(
        title=f"Fit Time for {columns} Columns",
        xlabel="Rows",
        ylabel="Time (milliseconds)",
        x=rows,
        y_rust=fit_time[:, 0],
        y_sklearn=fit_time[:, 1]
    )

    fig_predict_time = plot_benchmark_data(
        title=f"Predict Time for {columns} Columns",
        xlabel="Rows",
        ylabel="Time (milliseconds)",
        x=rows,
        y_rust=predict_time[:, 0],
        y_sklearn=predict_time[:, 1]
    )

    fig_accuracy = plot_benchmark_data(
        title=f"Accuracy comparison for {columns} Columns",
        xlabel="Rows",
        ylabel="Accuracy in [0,1]",
        x=rows,
        y_rust=accuracies[:, 0],
        y_sklearn=accuracies[:, 1]
    )

    return fig_fit_time, fig_predict_time, fig_accuracy


def plot_for_fixed_rows(rows: int, df: pd.DataFrame):
    df = df.loc[df["rows"] == rows, :]
    df.drop(columns=["rows"], inplace=True)

    cols = df["cols"].to_numpy()
    fit_time = np.array(df["fit_time"].map(lambda x: [float(val) for val in x[1:-1].split(",")]).tolist(), dtype="float64") * 1000
    predict_time = np.array(df["predict_time"].map(lambda x: [float(val) for val in x[1:-1].split(",")]).tolist(), dtype="float64") * 1000
    accuracies = np.array(df["acc"].map(lambda x : [float(val[len("np.float64("):-1]) for val in x[1:-1].split(", ")]).tolist(), dtype="float64")

    fig_fit_time = plot_benchmark_data(
        title=f"Fit Time for {rows} Rows",
        xlabel="Columns",
        ylabel="Time (milliseconds)",
        x=cols,
        y_rust=fit_time[:, 0],
        y_sklearn=fit_time[:, 1]
    )

    fig_predict_time = plot_benchmark_data(
        title=f"Predict Time for {rows} Rows",
        xlabel="Columns",
        ylabel="Time (milliseconds)",
        x=cols,
        y_rust=predict_time[:, 0],
        y_sklearn=predict_time[:, 1]
    )

    fig_accuracy = plot_benchmark_data(
        title=f"Accuracy comparison for {rows} Rows",
        xlabel="Columns",
        ylabel="Accuracy in [0,1]",
        x=cols,
        y_rust=accuracies[:, 0],
        y_sklearn=accuracies[:, 1]
    )

    return fig_fit_time, fig_predict_time, fig_accuracy


def main():
    argparser = argparse.ArgumentParser(
        description="Benchmark Rust vs scikit-learn Logistic Regression"
    )
    argparser.add_argument(
        "--csv_path",
        type=str,
        default="/home/bl4ck_r4bbit/Schreibtisch/TUBerlin/MLMMI-Project/rust_experiments/benchmarks/results/col_row_iter.csv",
        help="Path to save the benchmark results CSV file",
    )

    args = argparser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    df.drop(columns=["max_iters", "kernel", "repeats"], inplace=True)

    parent = Path(args.csv_path).parent
    
    for cols in _COLUMNS:
        fit, predict, acc = plot_for_fixed_cols(cols, df)
        fit.savefig(parent/f"fit_time_cols_{cols}.png")
        predict.savefig(parent/f"predict_time_cols_{cols}.png")
        acc.savefig(parent/f"accuracy_cols_{cols}.png")
    
    for rows in _ROWS:
        fit, predict, acc = plot_for_fixed_rows(rows, df)
        fit.savefig(parent/f"fit_time_rows_{rows}.png")
        predict.savefig(parent/f"predict_time_rows_{rows}.png")
        acc.savefig(parent/f"accuracy_rows_{rows}.png")

if __name__ == "__main__":
    main()