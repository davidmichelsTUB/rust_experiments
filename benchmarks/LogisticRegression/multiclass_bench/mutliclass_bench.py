import sys, os

sys.path.insert(0, os.path.abspath("../../../"))  # project root

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from adapters.LogisticRegression import LogisticRegression as RustLogisticRegression
import pandas as pd
import statistics
import time
from pathlib import Path


_ROWS = [
    500,
    1000,
    10_000,
    100_000,
]

_COLUMNS = [
    # 10,
    100,
    500,
    1_000,
]

_CLASSES = [
    5,
    10,
    20
]

_SKLEARN_KWARGS = {
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


def run_save_one_bench(
    rows: int,
    cols: int,
    classes: int,
    sklearn_init_dict: dict = _SKLEARN_KWARGS,
    repeats: int = 4
):
    X,y = make_classification(
        n_samples=rows,
        n_features=cols,
        n_classes=classes,
        n_informative=cols // 2,
        n_redundant=cols // 4,
        random_state=42
    )

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
    
    X = X.astype(dtype=np.float32, order="C")
    y = y.astype(dtype=np.uint32, order="C")

    results = {
        "rows": rows,
        "cols": cols,
        "n_classes": classes,
        "repeats": repeats,
    }

    rust_model = RustLogisticRegression(**sklearn_init_dict)
    sklearn_model = SklearnLogisticRegression(**sklearn_init_dict)

    rust_fit_time, rust_fit_std, _ = time_it(lambda: rust_model.fit(X, y))
    sklearn_fit_time, sklearn_fit_std, _ = time_it(lambda: sklearn_model.fit(X, y))
    rust_predict_time, rust_predict_std, rust_predict = time_it(
        lambda: rust_model.predict(X)
    )
    sklearn_predict_time, sklearn_predict_std, sklearn_predict = time_it(
        lambda: sklearn_model.predict(X)
    )

    rust_acc = (rust_predict == y).mean()
    sklearn_acc = (sklearn_predict == y).mean()

    # Predicting probabilities
    rust_predict_proba_time, rust_predict_proba_std, rust_predict_proba = time_it(
        lambda: rust_model.predict_proba(X)
    )
    sklearn_predict_proba_time, sklearn_predict_proba_std, sklearn_predict_proba = time_it(
        lambda: sklearn_model.predict_proba(X)
    )

    results.update({
        "rust_fit_time": rust_fit_time,
        "rust_fit_std": rust_fit_std,
        "sklearn_fit_time": sklearn_fit_time,
        "sklearn_fit_std": sklearn_fit_std,
        "rust_predict_time": rust_predict_time,
        "rust_predict_std": rust_predict_std,
        "sklearn_predict_time": sklearn_predict_time,
        "sklearn_predict_std": sklearn_predict_std,
        "rust_predict_proba_time": rust_predict_proba_time,
        "rust_predict_proba_std": rust_predict_proba_std,
        "sklearn_predict_proba_time": sklearn_predict_proba_time,
        "sklearn_predict_proba_std": sklearn_predict_proba_std,
        "rust_acc": rust_acc,
        "sklearn_acc": sklearn_acc,
        "fit_rust_speedup": sklearn_fit_time / rust_fit_time,
        "predict_rust_speedup": sklearn_predict_time / rust_predict_time,
        "predict_proba_rust_speedup": sklearn_predict_proba_time / rust_predict_proba_time,
        "proba_error_mean": np.abs(rust_predict_proba - sklearn_predict_proba).mean(),
        "proba_error_max": np.abs(rust_predict_proba - sklearn_predict_proba).max(),
    })

    results["sklearn_kwargs"] = str(sklearn_init_dict)
    if not Path(__file__).parent.joinpath("results.csv").exists():
        results_csv = pd.DataFrame([results])
        results_csv.to_csv(Path(__file__).parent / "results.csv", index=False)
    else:
        results_csv = pd.read_csv(Path(__file__).parent / "results.csv")
        results_csv = pd.concat([results_csv, pd.DataFrame([results])], ignore_index=True)
        results_csv.to_csv(Path(__file__).parent / "results.csv", index=False)


def main():

    for classes in _CLASSES:
        for cols in _COLUMNS:
            for rows in _ROWS:
                run_save_one_bench(rows, cols, classes, sklearn_init_dict=_SKLEARN_KWARGS, repeats=4)

if __name__ == "__main__":
    main()