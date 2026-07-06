from __future__ import annotations
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler as _SKMinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer as _SKCountVectorizer
from scipy.sparse import csr_matrix
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

minmax_scale_fit = getattr(_native, "minmax_scale_fit", None)
minmax_scale_transform = getattr(_native, "minmax_scale_transform", None)
count_vectorize_fit = getattr(_native, "count_vectorize_fit", None)
count_vectorize_transform = getattr(_native, "count_vectorize_transform", None)
count_vectorize_fit_transform = getattr(_native, "count_vectorize_fit_transform", None)


class RustyMinMaxScaler(_SKMinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)
        self._supported_params = not clip  # Rust doesn't implement clip

        self.cores = os.cpu_count()

    def fit(self, X, n_jobs, y=None, sample_weight=None):

        if (
            not HAVE_RUST
            or minmax_scale_fit is None
            or not self._supported_params
            or sample_weight is not None
        ):
            return super().fit(X, y)
        t0 = time.perf_counter()

        X_arr = np.asarray(X, dtype=np.float32)
        t1 = time.perf_counter()

        data_min, data_max = minmax_scale_fit(X_arr, n_jobs)

        self.data_min_ = data_min.astype(np.float64)
        self.data_max_ = data_max.astype(np.float64)
        t2 = time.perf_counter()
        print(f"convert={1000 * (t1 - t0):.3f}ms  rust={1000 * (t2 - t1):.3f}ms")

        return self

    def transform(self, X, n_jobs):
        if (
            not HAVE_RUST
            or minmax_scale_transform is None
            or not self._supported_params
        ):
            return super().transform(X)
        t0 = time.perf_counter()

        check_is_fitted(self)
        X_arr = np.asarray(X, dtype=np.float32)
        data_min = self.data_min_.astype(np.float32)
        data_max = self.data_max_.astype(np.float32)
        t1 = time.perf_counter()

        try:
            out = minmax_scale_transform(X_arr, data_min, data_max, n_jobs)
        except Exception as e:
            logger.warning(f"Rust minmax_scale_transform failed, falling back: {e}")
            return super().transform(X)
        t2 = time.perf_counter()
        logger.debug("using rust")
        print(f"convert={1000 * (t1 - t0):.3f}ms  rust={1000 * (t2 - t1):.3f}ms")
        return out

    def fit_transform(
        self,
        X,
        n_jobs,
        y=None,
        **fit_params,
    ):
        return self.fit(X, n_jobs, y, **fit_params).transform(X, n_jobs)


class RustyCountVectorizer(_SKCountVectorizer):
    def __init__(self, n_jobs=None, **kwargs):
        super().__init__(**kwargs)
        self.cores = os.cpu_count()

    def _stopwords_set(self):
        sw = self.stop_words
        if sw is None or sw == "english":
            return set()
        return set(sw)

    def fit(self, raw_documents, n_jobs):
        if not HAVE_RUST or count_vectorize_fit is None:
            logger.debug(f"Rust count_vectorizer failed, falling back")
            return super().fit(raw_documents)
        logger.debug("Using RustyCountVectorizer fit")

        corpus = list(raw_documents)
        vocab = count_vectorize_fit(
            corpus, self._stopwords_set(), self.token_pattern, n_jobs
        )
        self.vocabulary_ = vocab
        return self

    def transform(self, raw_documents, n_jobs):
        if not HAVE_RUST or count_vectorize_transform is None:
            logger.debug(f"Rust count_vectorizer failed, falling back")
            return super().transform(raw_documents)
        logger.debug("Using RustyCountVectorizer transform")

        corpus = list(raw_documents)
        data, indices, indptr = count_vectorize_transform(
            corpus, self.vocabulary_, self._stopwords_set(), self.token_pattern, n_jobs
        )
        csr = csr_matrix(
            (data, indices, indptr), shape=(len(raw_documents), len(self.vocabulary_))
        )
        return csr

    def fit_transform(self, raw_documents, n_jobs, y=None):
        if not HAVE_RUST or count_vectorize_fit_transform is None:
            print("USING SKLEARN")
            logger.debug(f"Rust count_vectorizer failed, falling back")
            return super().fit_transform(raw_documents)
        logger.debug("Using RustyCountVectorizer fit_transform")
        corpus = list(raw_documents)

        vocab, data, indices, indptr = count_vectorize_fit_transform(
            corpus, self._stopwords_set(), self.token_pattern, n_jobs
        )
        self.vocabulary_ = vocab
        csr = csr_matrix(
            (data, indices, indptr), shape=(len(raw_documents), len(self.vocabulary_))
        )
        return csr
