from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import rust_logistic
import numpy as np


_FIT = {
    "lbfgs": rust_logistic.binary_lbfgs_fit,
    "multiclass_lbfgs": rust_logistic.multiclass_lbfgs_fit,
}


class LogisticRegression(SklearnLogisticRegression):
    def __init__(
        self,
        penalty="deprecated",
        *,
        C=1.0,
        l1_ratio=0.0,
        dual=False,
        tol=0.0001,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    ):
        super().__init__(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            dual=dual,
            tol=tol,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )

        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.classes_weight = class_weight

    def _check_before_predict(self, X):
        if (
            not hasattr(self, "coef_")
            or not hasattr(self, "n_classes")
            or not hasattr(self, "n_features")
        ):
            raise ValueError(
                "This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if self.coef_.flags["C_CONTIGUOUS"] is False:
            raise ValueError(
                "The coefficient array is not C-contiguous. Please ensure that the coefficients are stored in a C-contiguous manner... Fit again or convert the array to C-contiguous format before calling predict or predict_proba."
            )

        if self.coef_.dtype != np.float32:
            raise ValueError(
                "The coefficient array must be of type float32. Please ensure that the coefficients are of the correct type before calling predict or predict_proba... Fit again or convert the array to float32 before calling predict or predict_proba."
            )

        if X.flags["C_CONTIGUOUS"] is False:
            raise ValueError(
                "The input array X is not C-contiguous. Please ensure that the input data is stored in a C-contiguous manner... Convert the array to C-contiguous format before calling predict or predict_proba."
            )

        if X.dtype != np.float32:
            raise ValueError(
                "The input array X must be of type float32. Please ensure that the input data is of the correct type before calling predict or predict_proba."
            )

        if self.n_features != X.shape[1]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match the number of features the model was trained on ({self.n_features})."
            )

        if self.fit_intercept and self.coef_.shape[0] != (self.n_classes if self.n_classes > 2 else 1) * (self.n_features + 1):
            raise ValueError(
                f"Number of classes ({self.n_classes}) and number of features ({self.n_features}) do not match the shape of the coefficient array ({self.coef_.shape[0]} = {self.n_classes * (self.n_features + 1)})."
            )
        
        if not self.fit_intercept and self.coef_.shape[0] != (self.n_classes if self.n_classes > 2 else 1) * self.n_features:
            raise ValueError(
                f"Number of classes ({self.n_classes}) and number of features ({self.n_features}) do not match the shape of the coefficient array ({self.coef_.shape[0]} = {self.n_classes * self.n_features})."
            )

    def _check_before_fit(self, X, y, sample_weight):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must be the same.")

        if X.dtype != np.float32:
            raise ValueError(
                "The input array X must be of type float32. Please ensure that the input data is of the correct type before calling fit."
            )

        if y.dtype != np.uint32:
            raise ValueError(
                "The input array y must be of type uint32. Please ensure that the input data is of the correct type before calling fit."
            )

        if X.flags["C_CONTIGUOUS"] is False:
            raise ValueError(
                "The input array X is not C-contiguous. Please ensure that the input data is stored in a C-contiguous manner... Convert the array to C-contiguous format before calling fit."
            )

        if y.flags["C_CONTIGUOUS"] is False:
            raise ValueError(
                "The input array y is not C-contiguous. Please ensure that the input data is stored in a C-contiguous manner... Convert the array to C-contiguous format before calling fit."
            )

        if sample_weight is not None:
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a 1D array.")
            if len(sample_weight) != len(y):
                raise ValueError(
                    "Number of samples in sample_weight and y must be the same."
                )
            if sample_weight.dtype != np.float32:
                raise ValueError(
                    "The input array sample_weight must be of type float32. Please ensure that the input data is of the correct type before calling fit."
                )
            if sample_weight.flags["C_CONTIGUOUS"] is False:
                raise ValueError(
                    "The input array sample_weight is not C-contiguous. Please ensure that the input data is stored in a C-contiguous manner... Convert the array to C-contiguous format before calling fit."
                )

        self.n_classes = max(y.max() + 1, 2)
        self.n_features = X.shape[1]

    def predict_proba(self, X):

        # self._check_before_predict(X)

        if self.n_classes == 2:
            return rust_logistic.binary_predict_proba(
                X,
                self.coef_,
                self.intercept_,
                intercept=self.fit_intercept
            )

        else:
            return rust_logistic.multiclass_predict_proba(
                X,
                self.coef_,
                intercept=self.fit_intercept,
                bias=self.intercept_ if self.fit_intercept else None,
            )

    def predict(self, X):

        # self._check_before_predict(X)

        if self.n_classes == 2:
            return rust_logistic.binary_predict(
                X,
                self.coef_,
                self.intercept_,
                intercept=self.fit_intercept
            )
            
        else:
            return rust_logistic.multiclass_predict(
                X,
                self.coef_,
                intercept=self.fit_intercept,
                bias=self.intercept_ if self.fit_intercept else None
            )

    def fit(self, X, y, sample_weight=None):

        self._check_before_fit(X, y, sample_weight)

        if self.n_classes == 2:
            all_weights = _FIT["lbfgs"](
                x=X,
                y=y,
                l1_reg=1 / self.C * self.l1_ratio,
                l2_reg= 1 / self.C * (1 - self.l1_ratio),
                intercept=self.fit_intercept,
                max_iters=self.max_iter,
                m=10,
                tolerance=self.tol,
                sample_weights=sample_weight,
            )
            
            if self.fit_intercept:
                self.coef_ = all_weights[:-1]
                self.intercept_ = all_weights[-1]
            
            else:
                self.coef_ = all_weights
                self.intercept_ = 0.0 
        else:
            all_weights = _FIT["multiclass_lbfgs"](
                x=X,
                y=y,
                l1_reg=1 / self.C * self.l1_ratio,
                l2_reg= 1 / self.C * (1 - self.l1_ratio),
                n_classes=self.n_classes,
                n_features=self.n_features,
                intercept=self.fit_intercept,
                max_iters=self.max_iter,
                m=10,
                tolerance=self.tol,
                sample_weights=sample_weight,
            )

            

            if self.fit_intercept:
                self.intercept_ = all_weights[:self.n_classes]
                self.coef_ = all_weights[self.n_classes:].reshape((self.n_features, self.n_classes))
            
            else:
                self.intercept_ = 0.0
                self.coef_ = all_weights.reshape((self.n_features, self.n_classes))