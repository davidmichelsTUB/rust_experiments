from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import rust_logistic
import numpy as np

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

    def predict_proba(self, X, kernel: str = "fused_parallel"):

        if kernel not in ["fused_parallel", "parallel"]:
            raise ValueError("Invalid kernel specified. Supported kernels are 'fused_parallel' and 'parallel'.")
        
        if not hasattr(self, "coef_"):
            raise ValueError("This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if len(self.coef_) == 1:
            return rust_logistic.binary_predict_proba(X, self.coef_, kernel)
    

    def predict(self, X, kernel: str = "fused_parallel"):
        pred_proba = self.predict_proba(X, kernel)
        return (pred_proba >= 0.5).astype(int) if pred_proba.ndim == 1 else pred_proba.argmax(axis=1)
    

    def fit(self, X, y, sample_weight=None):
        
        classes = set(y)

        if len(classes) == 2:


            if classes != {0, 1}:
                raise ValueError("Only binary classification with labels 0 and 1 is supported in this implementation.")
            else:
                if X.ndim != 2:
                    raise ValueError("X must be a 2D array.")
                if y.ndim != 1:
                    raise ValueError("y must be a 1D array.")
                if X.shape[0] != len(y):
                    raise ValueError("Number of samples in X and y must be the same.")
                
                

                X = X.astype(dtype=np.float32, order='C', copy=False)
                y = y.astype(dtype=np.float32, order='C', copy=False)

                if sample_weight is not None:
                    if sample_weight.ndim != 1:
                        raise ValueError("sample_weight must be a 1D array.")
                    if len(sample_weight) != len(y):
                        raise ValueError("Number of samples in sample_weight must be the same as in y.")
                    sample_weight = sample_weight.astype(dtype=np.float32, order='C', copy=False)
                
                self.coef_ = rust_logistic.binary_fit(
                    X,
                    y,
                    l1_reg=1.0 / self.C * self.l1_ratio,
                    l2_reg=1.0 / self.C * (1 - self.l1_ratio),
                    max_iters=self.max_iter,
                    m=10,
                    sample_weights=sample_weight
                )
                
    