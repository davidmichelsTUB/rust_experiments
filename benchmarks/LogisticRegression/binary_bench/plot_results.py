import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


_ROWS = [
    100,
    1000,
    10_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000
]

_COLUMNS = [
    # 10,
    100,
    500,
    1_000,
    # 10_000,
    # 100_000,
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
    fit_time_rust = np.array(df["rust_fit_time"].to_numpy(), dtype="float64") * 1000
    fit_time_sklearn = np.array(df["sklearn_fit_time"].to_numpy(), dtype="float64") * 1000
    predict_time_rust = np.array(df["rust_predict_time"].to_numpy(), dtype="float64") * 1000
    predict_time_sklearn = np.array(df["sklearn_predict_time"].to_numpy(), dtype="float64") * 1000
    predict_proba_time_rust = np.array(df["rust_predict_proba_time"].to_numpy(), dtype="float64") * 1000
    predict_proba_time_sklearn = np.array(df["sklearn_predict_proba_time"].to_numpy(), dtype="float64") * 1000
    rust_acc = np.array(df["rust_acc"].to_numpy(), dtype="float64")
    sklearn_acc = np.array(df["sklearn_acc"].to_numpy(), dtype="float64")

    fig_fit_time = plot_benchmark_data(
        title=f"Fit Time for {columns} Columns",
        xlabel="Rows",
        ylabel="Time (milliseconds)",
        x=rows,
        y_rust=fit_time_rust,
        y_sklearn=fit_time_sklearn 
    )

    fig_predict_time = plot_benchmark_data(
        title=f"Predict Time for {columns} Columns",
        xlabel="Rows",
        ylabel="Time (milliseconds)",
        x=rows,
        y_rust=predict_time_rust,
        y_sklearn=predict_time_sklearn
    )

    fig_accuracy = plot_benchmark_data(
        title=f"Accuracy comparison for {columns} Columns",
        xlabel="Rows",
        ylabel="Accuracy in [0,1]",
        x=rows,
        y_rust=rust_acc,
        y_sklearn=sklearn_acc
    )

    fig_predict_proba_time = plot_benchmark_data(
        title=f"Predict Proba Time for {columns} Columns",
        xlabel="Rows",
        ylabel="Time (milliseconds)",
        x=rows,
        y_rust=predict_proba_time_rust,
        y_sklearn=predict_proba_time_sklearn
    )

    return fig_fit_time, fig_predict_time, fig_predict_proba_time, fig_accuracy


def plot_for_fixed_rows(rows: int, df: pd.DataFrame):
    df = df.loc[df["rows"] == rows, :]
    df.drop(columns=["rows"], inplace=True)

    cols = df["cols"].to_numpy()


    fit_time_rust = np.array(df["rust_fit_time"].to_numpy(), dtype="float64") * 1000
    fit_time_sklearn = np.array(df["sklearn_fit_time"].to_numpy(), dtype="float64") * 1000
    predict_time_rust = np.array(df["rust_predict_time"].to_numpy(), dtype="float64") * 1000
    predict_time_sklearn = np.array(df["sklearn_predict_time"].to_numpy(), dtype="float64") * 1000
    predict_proba_time_rust = np.array(df["rust_predict_proba_time"].to_numpy(), dtype="float64") * 1000
    predict_proba_time_sklearn = np.array(df["sklearn_predict_proba_time"].to_numpy(), dtype="float64") * 1000
    rust_acc = np.array(df["rust_acc"].to_numpy(), dtype="float64")
    sklearn_acc = np.array(df["sklearn_acc"].to_numpy(), dtype="float64")

    fig_fit_time = plot_benchmark_data(
        title=f"Fit Time for {rows} Rows",
        xlabel="Columns",
        ylabel="Time (milliseconds)",
        x=cols,
        y_rust=fit_time_rust,
        y_sklearn=fit_time_sklearn
    )

    fig_predict_time = plot_benchmark_data(
        title=f"Predict Time for {rows} Rows",
        xlabel="Columns",
        ylabel="Time (milliseconds)",
        x=cols,
        y_rust=predict_time_rust,
        y_sklearn=predict_time_sklearn
    )

    fig_accuracy = plot_benchmark_data(
        title=f"Accuracy comparison for {rows} Rows",
        xlabel="Columns",
        ylabel="Accuracy in [0,1]",
        x=cols,
        y_rust=rust_acc,
        y_sklearn=sklearn_acc
    )

    fig_predict_proba_time = plot_benchmark_data(
        title=f"Predict Proba Time for {rows} Rows",
        xlabel="Columns",
        ylabel="Time (milliseconds)",
        x=cols,
        y_rust=predict_proba_time_rust,
        y_sklearn=predict_proba_time_sklearn
    )

    return fig_fit_time, fig_predict_time, fig_accuracy, fig_predict_proba_time


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
    df.drop(columns=["sklearn_kwargs"], inplace=True)

    parent = Path(args.csv_path).parent
    
    for cols in _COLUMNS:
        fit, predict, predict_proba, acc = plot_for_fixed_cols(cols, df)
        fit.savefig(parent/f"fit_time_cols_{cols}.png")
        predict.savefig(parent/f"predict_time_cols_{cols}.png")
        predict_proba.savefig(parent/f"predict_proba_time_cols_{cols}.png")
        acc.savefig(parent/f"accuracy_cols_{cols}.png")
    
    for rows in _ROWS:
        fit, predict, predict_proba, acc = plot_for_fixed_rows(rows, df)
        fit.savefig(parent/f"fit_time_rows_{rows}.png")
        predict.savefig(parent/f"predict_time_rows_{rows}.png")
        predict_proba.savefig(parent/f"predict_proba_time_rows_{rows}.png")
        acc.savefig(parent/f"accuracy_rows_{rows}.png")

if __name__ == "__main__":
    main()