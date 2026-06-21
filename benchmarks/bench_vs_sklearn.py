"""End-to-end benchmark: our Rust-backed LogisticRegression vs sklearn.

Measures fit time, predict time, and accuracy/agreement parity across a few
dataset shapes. Both estimators use fit_intercept=False (our adapter requires
it) so the comparison is apples-to-apples.
"""
import sys
import os
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
from LogisticRegression import LogisticRegression as RustLR

REPEATS = 5
C = 1.0
MAX_ITER = 100
TOL = 1e-4

SHAPES = [
    (10_000, 20),
    (50_000, 50),
    (200_000, 100),
    (500_000, 200),
]


def timeit(fn, repeats=REPEATS):
    """Return (best_seconds, result_of_last_call)."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def bench_shape(n_samples, n_features):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        random_state=0,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
    Xte = np.ascontiguousarray(Xte, dtype=np.float32)
    ytr = ytr.astype(np.float32)

    # --- sklearn ---
    sk = SklearnLR(C=C, fit_intercept=False, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
    sk_fit_t, _ = timeit(lambda: sk.fit(Xtr, ytr))
    sk_pred_t, sk_pred = timeit(lambda: sk.predict(Xte))
    sk_acc = (sk_pred == yte).mean()

    # --- ours ---
    rs = RustLR(C=C, fit_intercept=False, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
    rs_fit_t, _ = timeit(lambda: rs.fit(Xtr, ytr))
    rs_pred_t, rs_pred = timeit(lambda: rs.predict(Xte))
    rs_acc = (rs_pred == yte).mean()

    # parity
    agree = (rs_pred == sk_pred).mean()
    coef_diff = np.linalg.norm(
        np.ravel(rs.coef_) - np.ravel(sk.coef_)
    ) / (np.linalg.norm(np.ravel(sk.coef_)) + 1e-12)

    print(f"\n=== n={n_samples:,}  d={n_features} ===")
    print(f"{'':12}{'sklearn':>12}{'ours':>12}{'speedup':>10}")
    print(f"{'fit (s)':12}{sk_fit_t:12.4f}{rs_fit_t:12.4f}{sk_fit_t/rs_fit_t:9.2f}x")
    print(f"{'predict (s)':12}{sk_pred_t:12.4f}{rs_pred_t:12.4f}{sk_pred_t/rs_pred_t:9.2f}x")
    print(f"{'accuracy':12}{sk_acc:12.4f}{rs_acc:12.4f}")
    print(f"agreement (ours vs sklearn preds): {agree:.4f}")
    print(f"rel. coef L2 diff:                 {coef_diff:.4e}")


if __name__ == "__main__":
    print(f"repeats={REPEATS} (best-of), C={C}, max_iter={MAX_ITER}, tol={TOL}")
    for n, d in SHAPES:
        bench_shape(n, d)
