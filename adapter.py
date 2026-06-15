from __future__ import annotations
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler as _SKMinMaxScaler
from sklearn.utils.validation import check_is_fitted
import logging
import time

logger = logging.getLogger(__name__)
MIN_BLOCK_LEN = 10_000

try:
    import rust_backend as _native
    HAVE_RUST = True
except ImportError as e:
    _native = None
    HAVE_RUST = False
    logger.debug(f"Rust backend not available: {e}")

minmax_scale_fit      = getattr(_native, "minmax_scale_fit",      None)
minmax_scale_transform = getattr(_native, "minmax_scale_transform", None)


class RustyMinMaxScaler(_SKMinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True, clip=False, n_jobs=None):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)
        self._supported_params = not clip  # Rust doesn't implement clip

        cores = os.cpu_count()
        if n_jobs is None:
            self.n_jobs = cores
        elif n_jobs > cores:
            logger.warning(f"n_jobs {n_jobs} > core count {cores}, clamping")
            self.n_jobs = cores
        else:
            self.n_jobs = n_jobs

    def _n_chunks(self, X):
        print(f"N_rows = {X.shape[0]}")
        
        blocks = max(1, X.shape[0] // MIN_BLOCK_LEN)
        print(f"Number of blocks: {blocks}")
        return min(blocks, self.n_jobs)

    def fit(self, X, y=None, sample_weight=None):
        if not HAVE_RUST or minmax_scale_fit is None or not self._supported_params or sample_weight is not None:
            return super().fit(X, y)
        t0 = time.perf_counter()

        X_arr = np.asarray(X, dtype=np.float32)
        t1 = time.perf_counter()

        data_min, data_max = minmax_scale_fit(X_arr, self._n_chunks(X_arr))

        self.data_min_ = data_min.astype(np.float64)
        self.data_max_ = data_max.astype(np.float64)
        t2 = time.perf_counter()
        print(f"convert={1000*(t1-t0):.3f}ms  rust={1000*(t2-t1):.3f}ms")

        return self

    def transform(self, X):
        if not HAVE_RUST or minmax_scale_transform is None or not self._supported_params:
            return super().transform(X)
        t0 = time.perf_counter()

        check_is_fitted(self)
        X_arr = np.asarray(X, dtype=np.float32)
        data_min = self.data_min_.astype(np.float32)
        data_max = self.data_max_.astype(np.float32)
        t1 = time.perf_counter()

        try:
            out = minmax_scale_transform(X_arr, data_min, data_max, self._n_chunks(X_arr))
        except Exception as e:
            logger.warning(f"Rust minmax_scale_transform failed, falling back: {e}")
            return super().transform(X)
        t2 = time.perf_counter()
        print(f"convert={1000*(t1-t0):.3f}ms  rust={1000*(t2-t1):.3f}ms")
        return out

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
