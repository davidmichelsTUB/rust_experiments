from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from sklearn.utils import resample
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
)
from scipy import sparse
from numbers import Integral

import numpy as np
from scipy.special import ndtri, ndtr
from yaml import warnings

from rust_logistic import rust_fit_dense

BOUNDS_THRESHOLD = 1e-7


class QuantileTransformer(SklearnQuantileTransformer):
    _parameter_constraints: dict = {
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],
        "output_distribution": [StrOptions({"uniform", "normal"})],
        "ignore_implicit_zeros": ["boolean"],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=10000,
        random_state=None,
        copy=True,
        n_jobs=1,
    ):
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.n_jobs = n_jobs

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        if sparse.issparse(X):
            # Sparse is simple fallback
            for feature_idx in range(X.shape[1]):
                column_slice = slice(
                    X.indptr[feature_idx],
                    X.indptr[feature_idx + 1],
                )
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice],
                    self.quantiles_[:, feature_idx],
                    inverse,
                )
            return X

        def transform_feature(feature_idx):
            return (
                feature_idx,
                self._transform_col(
                    X[:, feature_idx],
                    self.quantiles_[:, feature_idx],
                    inverse,
                ) if self.output_distribution == "uniform" else self._transform_col_normal(
                    X[:, feature_idx],
                    self.quantiles_[:, feature_idx],
                    inverse,
                ),
            )

        if self.n_jobs == 1:
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(
                    X[:, feature_idx],
                    self.quantiles_[:, feature_idx],
                    inverse,
                )
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                for feature_idx, transformed in executor.map(
                    transform_feature,
                    range(X.shape[1]),
                ):
                    X[:, feature_idx] = transformed

        return X

    

    def _transform_col_normal(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                X_col = ndtr(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
            upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = 0.5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1])
            )
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                X_col = ndtri(X_col)
                # find the value to clip the data to avoid mapping to
                # infinity. Clip such that the inverse transform will be
                # consistent
                clip_min = ndtri(BOUNDS_THRESHOLD - np.spacing(1))
                clip_max = ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                X_col = np.clip(X_col, clip_min, clip_max)
            # else output distribution is uniform and the ppf is the
            # identity function so we let X_col unchanged

        return X_col
    
    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn(
                "'ignore_implicit_zeros' takes effect only with"
                " sparse matrix. This parameter has no effect."
            )

        n_samples, n_features = X.shape
        references = self.references_

        if self.subsample is not None and self.subsample < n_samples:
            # Take a subsample of `X`
            X = resample(
                X, replace=False, n_samples=self.subsample, random_state=random_state
            )

        self.quantiles_ = rust_fit_dense(X, references)