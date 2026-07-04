import sys, os

sys.path.insert(0, os.path.abspath(".."))  # project root

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from adapters.LogisticRegression import LogisticRegression as RustLogisticRegression
import timeit
import pandas as pd

_ROWS = [
    # 100,
    # 500,
    1000,
    5000,
    10000,
    50_000,
    100_000,
    # 500_000,
    # 1_000_000,
    # 5_000_000
]

_COLUMNS = [
    # 10,
    # 50,
    100,
    500,
    1000,
    # 5000,
    # 10_000,
    # 100_000,
    # 500_000,
]

_MAX_ITERS = [1000]


import time
import statistics
import argparse
from pathlib import Path


def bench(
    rows: int,
    cols: int,
    n_classes: int,
    repeats: int = 4,
    sklearn_kwargs: dict = {},
):
    X, y = make_classification(
        n_samples=rows,
        n_features=cols,
        n_classes=n_classes,
        n_informative=cols // 2,
        n_redundant=cols // 4,
        random_state=42
    )

    X = X.astype(dtype=np.float32, order="C")
    y = y.astype(dtype=np.uint32, order="C")

    results = {
        "rows": rows,
        "cols": cols,
        "n_classes": n_classes,
        "repeats": repeats,
        "sklearn_kwargs": sklearn_kwargs,
    }

    def time_it(fn):
        """Best-of-N wall-clock timing. Returns (seconds, last_result).

        One untimed warm-up absorbs thread-pool spin-up / allocation / page
        faults; min() over the repeats strips OS scheduling jitter. The result
        is captured so callers never recompute just to score it.
        """
        result = fn()  # warm-up, not timed
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            ts.append(time.perf_counter() - t0)
        mean = sum(ts) / len(ts)
        std = statistics.stdev(ts) if len(ts) > 1 else 0.0
        return mean, std, result

    ################################## Rust implementation
    rust_model = RustLogisticRegression(**sklearn_kwargs)

    # Fitting
    rust_fit_time, rust_fit_std, _ = time_it(lambda: rust_model.fit(X, y))

    # Predicting
    rust_predict_time, rust_predict_std, rust_predict = time_it(
        lambda: rust_model.predict(X)
    )
    
    rust_acc = (rust_predict == y).mean()

    # Predicting probabilities
    rust_predict_proba_time, rust_predict_proba_std, rust_predict_proba = time_it(
        lambda: rust_model.predict_proba(X)
    )

    print(f"Rust fit time: {rust_fit_time:.6f}s ± {rust_fit_std:.6f}s")
    print(f"Rust predict time: {rust_predict_time:.6f}s ± {rust_predict_std:.6f}s")
    print(f"Rust predict proba time: {rust_predict_proba_time:.6f}s ± {rust_predict_proba_std:.6f}s")
    ############################################### scikit-learn implementation
    sklearn_model = SklearnLogisticRegression(**sklearn_kwargs)

    #  Fitting
    sklearn_fit_time, sklearn_fit_std, _ = time_it(lambda: sklearn_model.fit(X, y))

    # Predicting
    sklearn_predict_time, sklearn_predict_std, sklearn_predict = time_it(lambda: sklearn_model.predict(X))
    sklearn_acc = (sklearn_predict == y).mean()

    # Predicting probabilities
    sklearn_predict_proba_time, sklearn_predict_proba_std, sklearn_predict_proba = time_it(lambda: sklearn_model.predict_proba(X))

    print(f"Scikit-learn fit time: {sklearn_fit_time:.6f}s ± {sklearn_fit_std:.6f}s")
    print(f"Scikit-learn predict time: {sklearn_predict_time:.6f}s ± {sklearn_predict_std:.6f}s")
    print(f"Scikit-learn predict proba time: {sklearn_predict_proba_time:.6f}s ± {sklearn_predict_proba_std:.6f}s")

    proba_error_mean = np.abs(rust_predict_proba - sklearn_predict_proba).mean()
    proba_error_max = np.abs(rust_predict_proba - sklearn_predict_proba).max()

    results["fit_time"] = [rust_fit_time, sklearn_fit_time]
    results["fit_std"] = [rust_fit_std, sklearn_fit_std]
    results["predict_time"] = [rust_predict_time, sklearn_predict_time]
    results["predict_std"] = [rust_predict_std, sklearn_predict_std]
    results["acc"] = [rust_acc, sklearn_acc]
    results["fit_rust_speedup"] = sklearn_fit_time / rust_fit_time
    results["predict_rust_speedup"] = sklearn_predict_time / rust_predict_time
    results["predict_proba_time"] = [rust_predict_proba_time, sklearn_predict_proba_time]
    results["predict_proba_std"] = [rust_predict_proba_std, sklearn_predict_proba_std]
    results["predict_proba_rust_speedup"] = sklearn_predict_proba_time / rust_predict_proba_time
    results["predict_proba_error_mean"] = proba_error_mean
    results["predict_proba_error_max"] = proba_error_max

    return results

def main():

    sklearn_kwargs = {
        "C" : 1.0,
        "l1_ratio" : 0.0,
        "dual" : False,
        "tol" : 0.0001,
        "fit_intercept" : True,
        "intercept_scaling" : 1,
        "class_weight" : None,
        "random_state" : 42,
        "solver" : "lbfgs",
        "max_iter" : 1000,
        "verbose" : 0,
        "warm_start" : False,
    }

    argparser = argparse.ArgumentParser(
        description="Benchmark Rust vs scikit-learn Logistic Regression"
    )
    argparser.add_argument(
        "--save",
        type=str,
        default="benchmark_results.csv",
        help="Path to save the benchmark results CSV file",
    )
    args = argparser.parse_args()
    df = []
    for rows in _ROWS:
        for cols in _COLUMNS:
            for max_iters in _MAX_ITERS:
                print(f"Benchmarking: rows={rows}, cols={cols}, max_iters={max_iters}")
                result = bench(
                    rows, cols, 5, repeats=4, sklearn_kwargs=sklearn_kwargs
                )
                print(result["acc"])
                df.append(result)

    df = pd.DataFrame(df)

    p = Path(args.save)
    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.save, index=False)


if __name__ == "__main__":
    main()