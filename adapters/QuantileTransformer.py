from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from sklearn.utils._param_validation import (
    Interval,
    Options,
    StrOptions,
    validate_params,
)
from numbers import Integral

class QuantileTransformer(SklearnQuantileTransformer):

    _parameter_constraints: dict = {
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],
        "output_distribution": [StrOptions({"uniform", "normal"})],
        "ignore_implicit_zeros": ["boolean"],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "copy": ["boolean"],
    }

    @validate_params(_parameter_constraints)
    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=1e5,
        random_state=None,
        copy=True
    ):
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy
        )

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    
        

    def _transform(self, X, inverse=False):
        """Perform the quantile transformation on X."""
        return "Can be sped up in rust"
    
    def _sparse_fit(self, X, y=None):
        """Fit the transformer on sparse data."""
        return "Can be sped up in rust"
    

    

